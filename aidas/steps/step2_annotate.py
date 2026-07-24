"""Step 2 — OCT boundary annotation with AI-assisted and manual drawing.

This module implements the Step 2 GUI panel for annotating retinal boundary
lines over OCT images. It supports manual ImageJ-style polyline drawing,
AI_ForAIDAS batch segmentation import, and saving boundary coordinates.

Core Functionality:
  • Manual Annotation: Click-to-place polyline drawing for 6 preset retinal
    boundaries (RPE, ELM, ONL-OPL, INL-IPL, GCL-RNFL, RNFL-Vitreous) with
    vertex undo and visual feedback.
  • AI Segmentation: Run AI_ForAIDAS batch predictions and import boundary
    traces for review.
    Predictions are automatically imported as boundary traces.
  • Boundary Workflow: Tracks annotation progress with separate incomplete/
    completed boundary lists; auto-advances to next boundary after finish.
  • Foveal Center Line: Always-visible draggable vertical marker for the
    foveal center X-coordinate with nudge buttons and keyboard entry.
  • Export: CSV export of all boundary row coordinates (one row per boundary).
  • MARKED Images: Generate Light_MARKED Analyze volumes
    (8-bit) with boundary pixels marked at specific intensity values per
    ImageJ macro conventions. Auto-scales boundaries if output size differs
    from input.

"""

import csv
import datetime
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from aidas.ai.client import AIWorkerClient
from aidas.canvas.image_canvas import ImageCanvas, RESAMPLE_NEAREST
from aidas.utils.filesystem import skipped_directories_warning, walk_accessible_directories
from aidas.utils.io_utils import read_analyze, read_tiff, write_analyze, scale_image
from aidas.utils.log_paths import app_log_dir
from aidas.utils.ui_utils import HoverToolTip, NativeNumericSpinbox, SidebarStepFrame, apply_app_icon_to, load_ui_icon


BOUNDARY_PRESETS = [
    ("RNFL-Vitreous", "#8338ec"),
    ("GCL-RNFL", "#fb8500"),
    ("INL-IPL", "#8ac926"),
    ("ONL-OPL", "#ef476f"),
    ("ELM", "#00b4d8"),
    ("RPE", "#ffb703"),
]
BOUNDARY_NAMES = [name for name, _ in BOUNDARY_PRESETS]
BOUNDARY_COLORS = {name: color for name, color in BOUNDARY_PRESETS}
TRACE_EXPORT_SUFFIX = "_step2_boundaries.csv"
FOVEA_BOUNDARY_NAME = "Fovea-Center"
MARKED_BACKGROUND_MAX = 230
COMMON_MARK_VALUES = {
    "RPE": 255,
    FOVEA_BOUNDARY_NAME: 243,
}
ADDITIONAL_MARK_VALUES = {
    "ELM": 254,
    "ONL-OPL": 253,
    "INL-IPL": 252,
    "GCL-RNFL": 250,
    "RNFL-Vitreous": 249,
}
# Line widths per boundary to match ImageJ macro: RPE and ELM use width 5, all others use width 1
MARKED_BOUNDARY_WIDTHS = {
    "RPE": 5,
    "ELM": 5,
    "ONL-OPL": 1,
    "INL-IPL": 1,
    "GCL-RNFL": 1,
    "RNFL-Vitreous": 1,
    "Fovea-Center": 1,
}
LIGHT_MARKED_BASENAME = "Light_MARKED"
LIGHT_SOURCE_BASENAME = "Light"
IMG_DEFAULT_DIR = os.path.expanduser("~/Desktop")
SUPPORTED_IMAGE_FILETYPES = [
    ("Analyze image", "*.img"),
    ("All files", "*.*"),
]

# Standard output dimensions (test1 format: 2 slices, original height, 2133 width)
STANDARD_OUTPUT_SLICES = 2
STANDARD_OUTPUT_HEIGHT = 177
STANDARD_OUTPUT_WIDTH = 2133


def _bresenham_line(start, end):
    """Return integer pixel coordinates for a line using Bresenham's algo.

    start, end -- (x, y) pairs (may be floats); returns list of (x, y) ints
    covering the discrete line between the points.
    """
    x0, y0 = (int(start[0]), int(start[1]))
    x1, y1 = (int(end[0]), int(end[1]))

    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        twice_err = 2 * err
        if twice_err > -dy:
            err -= dy
            x0 += sx
        if twice_err < dx:
            err += dx
            y0 += sy

    return points


def _polyline_pixels(points):
    """Convert a polyline (sequence of vertices) into pixel coordinates.

    Uses Bresenham per segment and avoids duplicating shared endpoints.
    Returns a list of (x, y) integer pixel coordinates.
    """
    points = [tuple(map(int, point)) for point in points]
    if not points:
        return []
    if len(points) == 1:
        return [points[0]]

    pixel_points = []
    for start, end in zip(points, points[1:]):
        segment = _bresenham_line(start, end)
        if pixel_points and segment and pixel_points[-1] == segment[0]:
            segment = segment[1:]
        pixel_points.extend(segment)

    return pixel_points or [points[0]]


def _resize_to_standard_format(volume_3d):
    """Resize a 3-D volume (slices, height, width) to standard output format.

    Standard format: (2 slices, original height, 2133 width)
    Uses pixel replication scaling (no interpolation) to match ImageJ behavior.

    Args:
        volume_3d: numpy array of shape (n_slices, height, width).

    Returns:
        Resized array of shape (STANDARD_OUTPUT_SLICES, original height, STANDARD_OUTPUT_WIDTH).
    """
    if volume_3d.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume_3d.shape}")
    
    current_slices, current_height, current_width = volume_3d.shape
    target_slices = STANDARD_OUTPUT_SLICES
    target_height = current_height
    target_width = STANDARD_OUTPUT_WIDTH
    
    # If already at target size, return as-is
    if (current_slices == target_slices and current_height == target_height 
        and current_width == target_width):
        return volume_3d.copy()
    
    # Resize each slice independently using nearest-neighbor indexing so the
    # original dtype is preserved for 16-bit LIGHT volumes.
    y_idx = np.arange(current_height, dtype=np.int64)
    x_idx = np.floor(np.arange(target_width) * current_width / target_width).astype(np.int64)
    y_idx = np.clip(y_idx, 0, current_height - 1)
    x_idx = np.clip(x_idx, 0, current_width - 1)

    resized_slices = []
    for s in range(min(current_slices, target_slices)):
        slice_data = volume_3d[s]
        resized_arr = np.ascontiguousarray(slice_data[np.ix_(y_idx, x_idx)])
        resized_slices.append(resized_arr)

    # If we need more slices (shouldn't happen), pad with zeros
    while len(resized_slices) < target_slices:
        resized_slices.append(np.zeros((target_height, target_width), dtype=volume_3d.dtype))

    return np.stack(resized_slices[:target_slices], axis=0)


def _resize_volume_to_shape(volume_3d, target_shape):
    """Resize a 3-D volume to an exact (slices, height, width) shape."""
    if volume_3d.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume_3d.shape}")

    target_slices, target_height, target_width = (int(v) for v in target_shape)
    current_slices, current_height, current_width = volume_3d.shape
    if (
        current_slices == target_slices
        and current_height == target_height
        and current_width == target_width
    ):
        return volume_3d.copy()

    y_idx = np.floor(np.arange(target_height) * current_height / target_height).astype(np.int64)
    x_idx = np.floor(np.arange(target_width) * current_width / target_width).astype(np.int64)
    y_idx = np.clip(y_idx, 0, current_height - 1)
    x_idx = np.clip(x_idx, 0, current_width - 1)

    resized_slices = []
    for s in range(min(current_slices, target_slices)):
        slice_data = volume_3d[s]
        resized_slices.append(np.ascontiguousarray(slice_data[np.ix_(y_idx, x_idx)]))

    while len(resized_slices) < target_slices:
        resized_slices.append(np.zeros((target_height, target_width), dtype=volume_3d.dtype))

    return np.stack(resized_slices[:target_slices], axis=0)


class Step2BatchSegmentationTable(ttk.Frame):
    """Fast folder table for Step 2 batch segmentation selection."""

    COLUMNS = ("folder", "status", "images")

    def __init__(self, parent):
        super().__init__(parent)
        self.rows = []
        self._row_by_iid = {}
        self._checkbox_images = self._make_checkbox_images()
        self._tree_font = tkfont.nametofont("TkDefaultFont")
        self._heading_font = self._tree_font.copy()
        self._heading_font.configure(weight="bold")

        self._tree_style = "Step2Batch.Treeview"
        self._style = ttk.Style(self)
        try:
            self._style.configure(self._tree_style, indent=0)
        except tk.TclError:
            pass

        self.tree = ttk.Treeview(
            self,
            columns=self.COLUMNS,
            show=("tree", "headings"),
            selectmode="none",
            style=self._tree_style,
        )
        self.yscroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.yscroll.set, xscrollcommand=self.xscroll.set)

        self.tree.heading(
            "#0",
            text="",
            image=self._checkbox_images["unchecked"],
            anchor="center",
            command=self._toggle_all_ready,
        )
        self.tree.heading("folder", text="Folder")
        self.tree.heading("status", text="Status")
        self.tree.heading("images", text="Images")

        self.tree.column("#0", width=40, minwidth=40, stretch=False, anchor="center")
        self.tree.column("folder", width=520, minwidth=220, stretch=False, anchor="w")
        self.tree.column("status", width=360, minwidth=120, stretch=False, anchor="w")
        self.tree.column("images", width=72, minwidth=60, stretch=False, anchor="center")

        self.tree.tag_configure("locked", foreground="#6b7280")
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.yscroll.grid(row=0, column=1, sticky="ns")
        self.xscroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.bind("<Button-1>", self._on_click, add="+")
        self.tree.bind("<Configure>", self._on_tree_configure, add="+")

    def _make_checkbox_images(self):
        images = {
            "checked": tk.PhotoImage(width=16, height=16),
            "unchecked": tk.PhotoImage(width=16, height=16),
            "locked": tk.PhotoImage(width=16, height=16),
        }
        for image in images.values():
            image.put("#ffffff", to=(0, 0, 16, 16))
            image.put("#6b7280", to=(2, 2, 14, 3))
            image.put("#6b7280", to=(2, 13, 14, 14))
            image.put("#6b7280", to=(2, 2, 3, 14))
            image.put("#6b7280", to=(13, 2, 14, 14))

        checked = images["checked"]
        for x, y in ((4, 8), (5, 9), (6, 10), (7, 9), (8, 8), (9, 7), (10, 6), (11, 5)):
            checked.put("#111827", to=(x, y, x + 1, y + 1))
            checked.put("#111827", to=(x, y + 1, x + 1, y + 2))

        locked = images["locked"]
        locked.put("#e5e7eb", to=(3, 3, 13, 13))
        locked.put("#9ca3af", to=(5, 7, 11, 9))
        return images

    def set_rows(self, rows):
        self.rows = list(rows or [])
        self._row_by_iid = {}
        self.tree.delete(*self.tree.get_children(""))

        if not self.rows:
            self.tree.insert(
                "",
                "end",
                text="",
                values=("No folders with Light.img were found.", "", ""),
                tags=("locked",),
            )
            self._refresh_header_checkbox()
            return

        for index, row in enumerate(self.rows):
            iid = str(index)
            self._row_by_iid[iid] = row
            self.tree.insert(
                "",
                "end",
                iid=iid,
                text="",
                image=self._image_for_row(row),
                values=self._values_for_row(row),
                tags=("locked",) if row.get("locked") else (),
            )
        self._fit_columns_to_content()
        self._refresh_header_checkbox()

    def _image_for_row(self, row):
        if row.get("locked"):
            return self._checkbox_images["locked"]
        if row.get("include"):
            return self._checkbox_images["checked"]
        return self._checkbox_images["unchecked"]

    def _values_for_row(self, row):
        values = row.get("values") or {}
        return (
            values.get("folder", ""),
            values.get("status", ""),
            values.get("images", ""),
        )

    def _measure_text(self, text, *, heading=False, padding=18):
        font = self._heading_font if heading else self._tree_font
        return int(font.measure(str(text or ""))) + int(padding)

    def _fit_columns_to_content(self):
        widths = {
            "folder": self._measure_text("Folder", heading=True),
            "status": self._measure_text("Status", heading=True),
            "images": self._measure_text("Images", heading=True),
        }
        for row in self.rows:
            folder, status, images = self._values_for_row(row)
            widths["folder"] = max(widths["folder"], self._measure_text(folder))
            widths["status"] = max(widths["status"], self._measure_text(status))
            widths["images"] = max(widths["images"], self._measure_text(images))

        self.tree.column("folder", width=max(220, widths["folder"]))
        self.tree.column("status", width=max(120, widths["status"]))
        self.tree.column("images", width=max(60, widths["images"]))
        self._expand_folder_to_view()

    def _on_tree_configure(self, _event=None):
        self._expand_folder_to_view()

    def _expand_folder_to_view(self):
        if not self.rows:
            return
        try:
            view_width = max(1, int(self.tree.winfo_width()))
            checkbox_width = int(self.tree.column("#0", "width"))
            folder_width = int(self.tree.column("folder", "width"))
            status_width = int(self.tree.column("status", "width"))
            images_width = int(self.tree.column("images", "width"))
        except tk.TclError:
            return

        non_folder_width = checkbox_width + status_width + images_width
        desired_folder_width = max(220, view_width - non_folder_width - 2)
        if desired_folder_width > folder_width:
            try:
                self.tree.column("folder", width=desired_folder_width)
            except tk.TclError:
                pass

    def _refresh_row(self, iid, row):
        try:
            self.tree.item(iid, image=self._image_for_row(row), values=self._values_for_row(row))
        except tk.TclError:
            pass

    def _refresh_header_checkbox(self):
        ready_rows = [row for row in self.rows if not row.get("locked")]
        image = self._checkbox_images["unchecked"]
        if ready_rows and all(bool(row.get("include")) for row in ready_rows):
            image = self._checkbox_images["checked"]
        try:
            self.tree.heading("#0", image=image)
        except tk.TclError:
            pass

    def _on_click(self, event):
        if self.tree.identify_region(event.x, event.y) not in {"cell", "tree"}:
            return None
        if self.tree.identify_column(event.x) != "#0":
            return None
        iid = self.tree.identify_row(event.y)
        row = self._row_by_iid.get(iid)
        if not row or row.get("locked"):
            return "break"
        row["include"] = not bool(row.get("include"))
        self._refresh_row(iid, row)
        self._refresh_header_checkbox()
        return "break"

    def _toggle_all_ready(self):
        ready_rows = [row for row in self.rows if not row.get("locked")]
        if not ready_rows:
            return
        include = not all(bool(row.get("include")) for row in ready_rows)
        for iid, row in self._row_by_iid.items():
            if row.get("locked"):
                continue
            row["include"] = include
            self._refresh_row(iid, row)
        self._refresh_header_checkbox()

    def selected_rows(self):
        return [row for row in self.rows if row.get("include") and not row.get("locked")]


class Step2BatchSegmentationSelectionPanel(ttk.Frame):
    """Embedded panel for selecting folders to run through Step 2 AI segmentation."""

    def __init__(self, step_frame, parent, root_dir):
        super().__init__(parent)
        self.step_frame = step_frame
        self.root_dir = os.path.abspath(root_dir)
        self.rows = []
        self.table = None

        self._build_ui()
        self._start_scan()

    def _build_ui(self):
        wrapper = ttk.Frame(self, padding=12)
        wrapper.pack(fill="both", expand=True)

        ttk.Label(wrapper, text="Batch Segmentation", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            wrapper,
            text=(
                "AIDaS will search the selected folder and subfolders for Light.img. "
                "Folders with existing MARKED segmentation are shown as already segmented and skipped."
            ),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        top = ttk.Frame(wrapper)
        top.pack(fill="x", pady=(0, 8))
        self.summary_var = tk.StringVar(value=f"Scanning: {self.root_dir}")
        ttk.Label(top, textvariable=self.summary_var, wraplength=760, justify="left").pack(
            side="left",
            fill="x",
            expand=True,
        )
        self.more_label = ttk.Label(top, text="", foreground="#0066cc", cursor="hand2")
        self.more_label.pack(side="right", padx=(8, 0))
        self.more_tooltip = HoverToolTip(self.more_label, "")

        self.table_host = ttk.Frame(wrapper)
        self.table_host.pack(fill="both", expand=True)
        self.scan_label = ttk.Label(
            self.table_host,
            text="Scanning folders...",
            anchor="center",
            justify="center",
        )
        self.scan_label.pack(fill="both", expand=True)

        run_box = ttk.Frame(wrapper)
        run_box.pack(fill="x", pady=(10, 0))
        self.next_button = ttk.Button(run_box, text="Next >", command=self._run_selected)
        self.next_button.pack(side="right")
        self.next_button.state(["disabled"])
        ttk.Button(run_box, text="Cancel", command=self._cancel).pack(side="left")

    def _start_scan(self):
        self.step_frame.status_var.set(f"Scanning subfolders under {self.root_dir}...")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            rows, scanned, skipped, access_errors = self.step_frame._scan_step2_batch_segmentation_folders(self.root_dir)
        except Exception as exc:
            self.after(0, lambda exc=exc: self._scan_failed(exc))
            return
        self.after(0, lambda: self._scan_done(rows, scanned, skipped, access_errors))

    def _scan_failed(self, exc):
        if not self.winfo_exists():
            return
        self.summary_var.set("Scan failed. Move the mouse over More for details.")
        self.more_label.configure(text="More")
        self.more_tooltip.text = f"Could not scan folders.\n{exc}"
        self.step_frame.status_var.set("Batch segmentation scan failed.")
        try:
            self.next_button.state(["disabled"])
        except tk.TclError:
            pass

    def _show_results_table(self, rows):
        for child in self.table_host.winfo_children():
            try:
                child.destroy()
            except tk.TclError:
                pass

        table = Step2BatchSegmentationTable(self.table_host)
        table.set_rows(rows)
        table.pack(fill="both", expand=True)
        self.table = table

    def _scan_done(self, rows, scanned, skipped, access_errors):
        if not self.winfo_exists():
            return
        self.rows = rows
        for row in rows:
            try:
                folder_text = os.path.relpath(row["folder"], self.root_dir)
                if folder_text == ".":
                    folder_text = self.root_dir
            except ValueError:
                folder_text = row["folder"]
            row["values"] = {
                "folder": folder_text,
                "status": row.get("status", ""),
                "images": str(len(row.get("image_paths") or [])),
            }
        self._show_results_table(rows)

        ready = sum(1 for row in rows if not row["locked"])
        already_segmented = sum(1 for row in rows if row["locked"])
        ready_images = sum(len(row.get("image_paths") or []) for row in rows if not row["locked"])
        summary = (
            f"Scanned {scanned} folder(s). Found {ready} ready folder(s) with {ready_images} image(s) to segment, "
            f"{already_segmented} folder(s) already segmented. {skipped} folder(s) did not contain Light.img. "
            f"{len(access_errors)} inaccessible folder(s) skipped."
        )
        self.summary_var.set(summary)
        self.more_label.configure(text="More" if access_errors else "")
        self.more_tooltip.text = skipped_directories_warning(access_errors) if access_errors else ""
        self.step_frame.status_var.set("Batch segmentation scan complete. Confirm folders to process.")
        try:
            if ready:
                self.next_button.state(["!disabled"])
            else:
                self.next_button.state(["disabled"])
        except tk.TclError:
            pass
    def _run_selected(self):
        if self.table is None:
            return
        rows = self.table.selected_rows()
        if not rows:
            messagebox.showwarning("Batch Step 2", "Select at least one ready folder.", parent=self)
            return
        self.step_frame._start_step2_batch_segmentation_from_rows(rows, self.root_dir)

    def _cancel(self):
        self.step_frame._close_step2_batch_segmentation_panel(restore_previous=True)


class Step2Frame(SidebarStepFrame):
    """GUI panel for tracing boundary lines and exporting pixel coordinates.

    This frame manages the complete Step 2 workflow:
      1. Load or receive OCT image data
      2. Manual trace or AI-assisted boundary annotation for 6 retinal layers
      3. Place foveal center vertical line (X coordinate)
      4. Export boundary rows as CSV or generate MARKED Analyze volumes

    The UI is split into:
      - Left panel: scrollable controls (load, segmentation, fovea, export)
      - Right panel: image canvas with line-drawing and coordinate display

    Key data structures:
      - boundary_traces: {name -> {points, pixels, color}} for saved boundaries
      - boundary_completion_vars: {name -> BooleanVar} for progress tracking
      - image_data: current numpy array being annotated
    """

    def __init__(self, parent, preferences=None, source_step=None, on_output_folder_changed=None):
        """Initialize the Step 2 annotation panel.

        Args:
            parent: Tkinter parent widget.
            preferences: User preferences dict (optional).
            source_step: Reference to Step 1 panel for linked image loading (optional).
        """
        super().__init__(parent)
        self.preferences = preferences
        self.source_step = source_step
        self.on_output_folder_changed = on_output_folder_changed

        # ─ Image data state ─
        self.current_file = None  # Path to currently loaded image
        self.image_data = None  # Current numpy array displayed on canvas
        self._source_was_8bit = False  # True when an opened 8-bit source was promoted for saving
        
        # ─ Boundary tracing state ─
        self.active_boundary = None  # Name of boundary currently being traced
        self.boundary_traces = {}  # Dict mapping boundary name -> {points, pixels, color}
        self.boundary_order = []  # List of boundary names in order they were completed
        
        # ─ Foveal center line state ─
        self.fovea_x = None  # Current X-coordinate of foveal center line (or None if not set)
        
        # ─ UI state flags ─
        self._segmenter_running = False  # True while AI segmentation is executing
        self._segmenter_progress_target = 0.0
        self._drawing_locked = False  # True if foveal center is locked (prevents other drawing)
        self._updating_fovea_entry = False  # Flag to avoid feedback loop when updating entry widget
        self._syncing_boundary_selection = False  # Flag to avoid feedback loop when selecting boundaries
        
        # ─ Fovea nudge repeat (keyboard acceleration) ─
        self._fovea_repeat_job = None  # After() job ID for fovea nudge repeat
        self._fovea_repeat_dir = 0  # Direction for nudge repeat (+1, -1, or 0 for stopped)
        self._fovea_repeat_ticks = 0  # Counter for acceleration of repeated nudges
        
        # ─ Progress animation during segmentation ─
        self._progress_animation_job = None  # After() job ID for progress bar animation
        
        # ─ Analyze template ─
        self._input_analyze_template = None  # Cached input Analyze header for MARKED output
        
        # Initialize completion tracking for all 6 preset boundaries
        self.boundary_completion_vars = {name: tk.BooleanVar(value=False) for name in BOUNDARY_NAMES}

        app_root = self._app_root()
        self.ai_for_aidas_root = os.path.join(app_root, "OCT Segmenter", "AI_ForAIDAS")
        self.ai_for_aidas_default_model = os.path.join(self.ai_for_aidas_root, "model_img.onnx")

        self.build_standard_layout()
        right = self.content

        self.image_info_var = tk.StringVar(value="No image loaded")
        self.image_info_frame = self.add_content_header(self.image_info_var, parent=right)

        self.canvas_area = ttk.Frame(right)
        self.canvas_area.pack(fill="both", expand=True)
        self.batch_results_notebook = None
        self.batch_segmentation_panel = None
        self._batch_result_canvases = []
        self._batch_result_tab_canvases = {}
        self._batch_result_states = {}
        self._active_batch_result_tab = None
        self._single_editor_state = None

        self.image_canvas = ImageCanvas(
            self.canvas_area,
            on_mouse_move=self._on_mouse_moved,
            on_line_change=self._on_active_line_changed,
            on_vertical_line_change=self._on_vertical_line_changed,
        )
        self.single_image_canvas = self.image_canvas
        self.image_canvas.enable_line(False)
        self.image_canvas.enable_roi(False)
        self.image_canvas.enable_vertical_line(False)
        self.image_canvas.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(
            value="Ready - process an image in Step 1, then trace boundaries or run batch segmentation."
        )
        self.add_status_bar(self.status_var, parent=right)

        self._build_controls()

    # ═══════════════════════════════════════════════════════════════════════
    #  UI construction
    # ═══════════════════════════════════════════════════════════════════════
    def _build_controls(self):
        """Construct and lay out all left-side control widgets.

        The left panel contains segmentation review controls, foveal center
        placement, and save/export actions.
        """
        self.aidas_model_path = self.ai_for_aidas_default_model

        self.segmentation_section = self.add_sidebar_section("Segmentation", pady=2)
        self.segmentation_frame = self.segmentation_section.body
        segmentation = self.segmentation_frame

        ai_folder_buttons = ttk.Frame(segmentation)
        ai_folder_buttons.pack(fill="x", pady=(6, 0))
        self.batch_ai_button = ttk.Button(
            ai_folder_buttons,
            text="Run Batch Segmentation",
            command=self._open_batch_segmentation_scanner,
        )
        self.batch_ai_button.pack(fill="x")

        workflow = ttk.LabelFrame(segmentation, text="Boundary Progress", padding=3)
        workflow.pack(fill="x", pady=(4, 0))

        workflow_lists = ttk.Frame(workflow)
        workflow_lists.pack(fill="x")

        incomplete_box = ttk.Frame(workflow_lists)
        incomplete_box.pack(side="left", fill="both", expand=True, padx=(0, 4))
        completed_box = ttk.Frame(workflow_lists)
        completed_box.pack(side="right", fill="both", expand=True, padx=(4, 0))

        ttk.Label(incomplete_box, text="Incomplete").pack(anchor="w")
        ttk.Label(completed_box, text="Completed").pack(anchor="w")

        self.boundary_incomplete_listbox = tk.Listbox(
            incomplete_box,
            height=6,
            selectmode="browse",
            exportselection=False,
        )
        incomplete_scroll = ttk.Scrollbar(incomplete_box, orient="vertical", command=self.boundary_incomplete_listbox.yview)
        self.boundary_incomplete_listbox.configure(yscrollcommand=incomplete_scroll.set)
        self.boundary_incomplete_listbox.pack(side="left", fill="both", expand=True)
        incomplete_scroll.pack(side="right", fill="y")
        self.boundary_incomplete_listbox.bind("<<ListboxSelect>>", self._on_boundary_incomplete_selected)

        self.boundary_completed_listbox = tk.Listbox(
            completed_box,
            height=6,
            selectmode="browse",
            exportselection=False,
        )
        completed_scroll = ttk.Scrollbar(completed_box, orient="vertical", command=self.boundary_completed_listbox.yview)
        self.boundary_completed_listbox.configure(yscrollcommand=completed_scroll.set)
        self.boundary_completed_listbox.pack(side="left", fill="both", expand=True)
        completed_scroll.pack(side="right", fill="y")
        self.boundary_completed_listbox.bind("<<ListboxSelect>>", self._on_boundary_completed_selected)

        workflow_buttons = ttk.Frame(workflow)
        workflow_buttons.pack(fill="x", pady=(6, 0))
        self.button_finish_icon = load_ui_icon(self, "el--ok.png")
        self.finish_boundary_btn = ttk.Button(workflow_buttons, text="Done", command=self._finish_boundary, image=self.button_finish_icon, compound="left")
        self.finish_boundary_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.clear_all_traces_btn_icon = load_ui_icon(self, "solar--eraser-bold-duotone.png")
        self.clear_all_traces_btn = ttk.Button(workflow_buttons, text="Clear", command=self._clear_all_traces, 
                                               image=self.clear_all_traces_btn_icon, compound="left")
        self.button_revert_icon = load_ui_icon(self, "grommet-icons--revert.png")
        self.clear_all_traces_btn.pack(side="left", fill="x", padx=(2, 2))
        self.revert_boundary_btn = ttk.Button(workflow_buttons, text="Revert", command=self._revert_boundary, image=self.button_revert_icon, compound="left")
        self.revert_boundary_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))

        self.segmenter_progress_var = tk.StringVar(value="Idle")
        self.segmenter_progress = ttk.Progressbar(workflow, mode="determinate", maximum=6, value=0)
        self.segmenter_progress.pack(fill="x", pady=(6, 4))

        self.active_trace_var = tk.StringVar(value="No active boundary")
        ttk.Label(workflow, textvariable=self.active_trace_var, wraplength=240, justify="left").pack(fill="x", pady=(0, 4))

        self.boundary_workflow_status_var = tk.StringVar(value="Select a boundary to make it active.")
        ttk.Label(workflow, textvariable=self.boundary_workflow_status_var, wraplength=240, justify="left").pack(
            fill="x",
            pady=(4, 0),
        )

        fovea = ttk.LabelFrame(segmentation, text="Foveal Center Line", padding=3)
        fovea.pack(fill="x", pady=(4, 2))
        self.fovea_frame = fovea  # Store reference for later state management

        # Keep this state var even when the label is hidden; callbacks rely on it.
        self.fovea_line_var = tk.StringVar(value="Fovea line: not set")

        coord_row = ttk.Frame(fovea)
        coord_row.pack(fill="x", pady=(0, 2))
        ttk.Label(coord_row, text="Center X:").pack(side="left")
        self.fovea_x_entry_var = tk.StringVar(value="")
        self.fovea_x_entry_var.trace_add("write", self._on_fovea_x_entry_changed)
        self.fovea_stepper = NativeNumericSpinbox(
            coord_row,
            self.fovea_x_entry_var,
            width=6,
            step=1,
            minimum=0,
        )
        self.fovea_stepper.pack(side="left", padx=(8, 4))
        self.fovea_x_entry = self.fovea_stepper.entry

        # Reset icon matches Step 1 numeric reset affordance.
        self.fovea_reset_btn = ttk.Button(coord_row, text="↺", width=2, command=self._center_vertical_line)
        self.fovea_reset_btn.pack(side="left")

        fovea_action_row = ttk.Frame(fovea)
        fovea_action_row.pack(fill="x", pady=(2, 0))

        self._set_fovea_controls_enabled(False)

        saved_buttons = ttk.Frame(segmentation)
        saved_buttons.pack( pady=(6, 0))
        self.saved_buttons_frame = saved_buttons  # Store reference for later state management
        # ttk.Button(saved_buttons, text="Export CSV", command=self._export_csv).pack(
        #     side="left",
        #     expand=True,
        #     fill="x",
        #     padx=(0, 2),
        # )
        self.button_save_icon = load_ui_icon(self, "ic--baseline-save.png")
        self.button_save_all_icon = load_ui_icon(self, "ic--sharp-save-all.png")
        self.saved_button = ttk.Button(saved_buttons, text="Save", command=self._save_current_marked_image_button, 
                   image=self.button_save_icon, compound="left")
        self.saved_button.pack(
            side="left",
            anchor="center",
            padx=4,
        )
        self.save_all_button = ttk.Button(
            saved_buttons,
            text="Save All",
            command=self._save_all_batch_result_tabs_button,
            image=self.button_save_all_icon,
            compound="left",
        )
        self.save_all_button.pack(side="left", anchor="center", padx=4)

        # help_section = self.add_sidebar_section("How to Trace", padding=3, pady=(2, 6))
        # help_box = help_section.body
        # ttk.Label(
        #     help_box,
        #     text=(
        #         "1. Open a directory and choose an .img image.\n"
        #         "2. Pick a boundary name.\n"
        #         "3. Left-click points along the boundary.\n"
        #         "4. Press Finish Boundary to save all pixels on the line.\n"
        #         "5. Drag the vertical foveal center line to adjust it."
        #     ),
        #     justify="left",
        # ).pack(anchor="w")

        self._refresh_boundary_lists(auto_select=False)
        self._set_segmentation_frame_enabled(False)
        self._update_batch_ai_button_state()

    # ═══════════════════════════════════════════════════════════════════════
    #  Image loading
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _image_pair_key(path):
        ext = os.path.splitext(path)[1].lower()
        key_path = os.path.splitext(path)[0] if ext in {".hdr", ".img"} else path
        return os.path.normcase(os.path.abspath(key_path))

    @staticmethod
    def _preferred_analyze_pair_path(path):
        base, ext = os.path.splitext(path)
        if ext.lower() in {".hdr", ".img"}:
            img_path = base + ".img"
            hdr_path = base + ".hdr"
            if os.path.isfile(img_path):
                return img_path
            if os.path.isfile(hdr_path):
                return hdr_path
        return path

    def _default_batch_initial_dir(self):
        if self.current_file and self.current_file != "Step 1 output":
            folder = os.path.dirname(self.current_file)
            if os.path.isdir(folder):
                return folder
        if os.path.isdir(IMG_DEFAULT_DIR):
            return IMG_DEFAULT_DIR
        return self._app_root()

    def _open_batch_segmentation_scanner(self):
        """Open an embedded folder scanner before running Step 2 batch segmentation."""
        if self._segmenter_running:
            messagebox.showinfo("Please wait", "Segmentation is already running.")
            return
        model_path = self.aidas_model_path
        if not os.path.isfile(model_path):
            messagebox.showerror("Missing model", f"AI_ForAIDAS boundary model not found:\n{model_path}")
            return

        folder = filedialog.askdirectory(
            title="Select root folder for Step 2 batch segmentation",
            initialdir=self._default_batch_initial_dir(),
        )
        if not folder:
            return

        self._open_step2_batch_segmentation_panel(folder)

    def _open_step2_batch_segmentation_panel(self, root_dir):
        self._close_step2_batch_segmentation_panel(restore_previous=False)

        if getattr(self, "_active_batch_result_tab", None):
            self._sync_active_batch_result_state()
        try:
            self.single_image_canvas.pack_forget()
        except tk.TclError:
            pass
        notebook = getattr(self, "batch_results_notebook", None)
        if notebook is not None:
            notebook.pack_forget()
        info_frame = getattr(self, "image_info_frame", None)
        if info_frame is not None:
            try:
                info_frame.pack_forget()
            except tk.TclError:
                pass

        self.batch_segmentation_panel = Step2BatchSegmentationSelectionPanel(self, self.canvas_area, root_dir)
        self.batch_segmentation_panel.pack(fill="both", expand=True)
        self.status_var.set(f"Scanning batch segmentation root: {os.path.abspath(root_dir)}")
        self._update_batch_ai_button_state()

    def _close_step2_batch_segmentation_panel(self, restore_previous=True):
        panel = getattr(self, "batch_segmentation_panel", None)
        self.batch_segmentation_panel = None
        if panel is not None:
            try:
                panel.destroy()
            except tk.TclError:
                pass

        if restore_previous:
            info_frame = getattr(self, "image_info_frame", None)
            if info_frame is not None and info_frame.winfo_manager() != "pack":
                try:
                    info_frame.pack(fill="x", pady=(0, 8), before=self.canvas_area)
                except tk.TclError:
                    pass
            notebook = getattr(self, "batch_results_notebook", None)
            if getattr(self, "_active_batch_result_tab", None) and notebook is not None:
                self._show_batch_results_canvas()
            else:
                self._show_single_image_canvas()
            self.status_var.set("Batch segmentation selection closed.")
        self._update_batch_ai_button_state()

    def _scan_step2_batch_segmentation_folders(self, root_dir):
        """Return folder rows showing ready and already-segmented Light inputs."""
        root_dir = os.path.abspath(root_dir)
        rows = []
        scanned = 0
        skipped = 0
        targets = (("Light", "light.img", "light_marked.img"),)

        folders, access_errors = walk_accessible_directories(root_dir)
        for folder_path in folders:
            folder = str(folder_path)
            try:
                with os.scandir(folder) as entries:
                    filenames = [entry.name for entry in entries if entry.is_file()]
            except OSError as exc:
                access_errors.append((folder_path, str(exc)))
                continue
            scanned += 1
            lower_files = {name.lower(): name for name in filenames}
            pending = []
            completed = []
            image_paths = []

            for label, source_name, marked_name in targets:
                source_file = lower_files.get(source_name)
                if source_file is None:
                    continue
                if marked_name in lower_files:
                    completed.append(label)
                    continue
                pending.append(label)
                image_paths.append(self._preferred_analyze_pair_path(os.path.join(folder, source_file)))

            if not pending and not completed:
                skipped += 1
                continue

            locked = not pending
            if locked:
                status = f"Already segmented: {', '.join(completed)}"
            elif completed:
                status = f"Ready: {', '.join(pending)} pending; {', '.join(completed)} done"
            else:
                status = f"Ready: {', '.join(pending)}"

            rows.append(
                {
                    "folder": folder,
                    "include": not locked,
                    "locked": locked,
                    "status": status,
                    "image_paths": image_paths,
                    "pending": pending,
                    "completed": completed,
                }
            )

        rows.sort(key=lambda row: os.path.relpath(row["folder"], root_dir).lower())
        return rows, scanned, skipped, access_errors

    def _start_step2_batch_segmentation_from_rows(self, rows, root_dir):
        image_paths = []
        seen = set()
        for row in rows:
            for path in row.get("image_paths") or []:
                key = self._image_pair_key(path)
                if key in seen:
                    continue
                seen.add(key)
                image_paths.append(path)

        if not image_paths:
            messagebox.showwarning("Batch Step 2", "No unsegmented Light.img files were selected.")
            return

        self._close_step2_batch_segmentation_panel(restore_previous=True)
        self.status_var.set(f"Found {len(image_paths)} target image(s) for AI batch segmentation.")

        self._show_single_image_canvas()
        manual_fovea_by_path = self._collect_folder_fovea_lines(image_paths)
        if manual_fovea_by_path is None:
            self.status_var.set("AI batch segmentation cancelled before running.")
            return

        image_paths = [path for path in image_paths if path in manual_fovea_by_path]
        if not image_paths:
            self.status_var.set("AI batch segmentation cancelled: all images were skipped.")
            return

        self._run_aidas_batch_segmentation(
            image_paths=image_paths,
            manual_fovea_by_path=manual_fovea_by_path,
        )

    def _collect_folder_fovea_lines(self, image_paths):
        """Prompt for a fovea center line for each image before folder segmentation in the main canvas."""
        fovea_by_path = {}
        total = len(image_paths)

        next_var = tk.StringVar(value="")
        
        # Disable batch controls to prevent duplicate triggers.
        if hasattr(self, "batch_ai_button"):
            self.batch_ai_button.state(["disabled"])

        # Create a temporary top toolbar to manage interactive fovea selection
        temp_frame = ttk.Frame(self.canvas_area, relief="solid", borderwidth=1)
        temp_frame.pack(side="top", fill="x", pady=(0, 4), before=self.image_canvas)
        
        prompt_label_var = tk.StringVar(value="")
        ttk.Label(temp_frame, textvariable=prompt_label_var, font=("", 10, "bold"), padding=4).pack(side="left")
        
        def on_skip():
            next_var.set("skip")

        def on_set():
            next_var.set("set")
            
        def on_cancel():
            next_var.set("cancel")
            
        btn_cancel = ttk.Button(temp_frame, text="Exit", command=on_cancel)
        btn_cancel.pack(side="right", padx=4, pady=4)
        
        btn_skip = ttk.Button(temp_frame, text="Skip >", command=on_skip)
        btn_skip.pack(side="right", padx=4, pady=4)

        btn_set = ttk.Button(temp_frame, text="Confirm", command=on_set)
        btn_set.pack(side="right", padx=4, pady=4)
        
        # Save current editor state to restore later if canceled (optional, but good practice)
        saved_state = self._capture_current_editor_state()

        try:
            for index, path in enumerate(image_paths, start=1):
                name = os.path.basename(path)
                msg = f"Select fovea {index}/{total}: {path}"
                prompt_label_var.set(msg)
                self.status_var.set(msg)
                self.update_idletasks()
                
                try:
                    image_data = self._load_image_from_path(path)
                except (OSError, ValueError, RuntimeError) as exc:
                    messagebox.showwarning(
                        "Fovea picker skipped",
                        f"Could not open this image for fovea selection:\n{path}\n\n{exc}",
                    )
                    self._clear_image_display("Image skipped. No image is loaded.")
                    continue

                self._show_image(image_data, path)
                self.update_idletasks()

                self.wait_variable(next_var)
                
                action = next_var.get()
                if action == "cancel":
                    self._clear_image_display("All Images skipped. No image is loaded.")
                    return None
                elif action == "skip":
                    # Do not add it to fovea_by_path, which explicitly drops it from the batch list
                    self._clear_image_display("Image skipped. No image is loaded.")
                    continue
                else:  # "set"
                    # Read whatever they left in the entry
                    try:
                        x_val = int(float(self.fovea_x_entry_var.get()))
                    except ValueError:
                        x_val = None
                    fovea_by_path[path] = x_val
                
        finally:
            temp_frame.destroy()
            self._update_boundary_action_buttons()
            self._update_batch_ai_button_state()


        return fovea_by_path

    def load_external_image(self, image, source_path=None):
        """Load an externally supplied image into Step 2.

        Used by Step 1 auto-sync after crop so the latest .img-like result is
        immediately available in this panel.
        """
        if image is None:
            return
        display_path = source_path or "Step 1 output"
        self._input_analyze_template = None
        img = self._image_for_annotation(image)
        self._show_image(img, display_path)

    def _load_image_from_path(self, path):
        """Read an image file from disk and return a 2-D numpy array.

        Supports Analyze (.hdr/.img), TIFF stacks, and standard images
        (PNG/JPEG). For multi-frame inputs a single slice is returned
        (the first frame). 16-bit sources are preserved for annotation.
        """
        image, template, source_was_8bit = self._read_image_for_annotation(path)
        self._input_analyze_template = template
        self._source_was_8bit = source_was_8bit
        return image

    @classmethod
    def _read_image_for_annotation(cls, path):
        """Read an image file without mutating Step 2 UI state."""
        ext = os.path.splitext(path)[1].lower()
        template = None
        if ext in {".hdr", ".img"}:
            data = read_analyze(path)
            template = cls._analyze_template_from_data(data)
        elif ext in {".tif", ".tiff"}:
            data = read_tiff(path)
        else:
            data = np.array(Image.open(path).convert("L"))

        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2:
            raise ValueError("Step 2 expects a 2-D grayscale image or a 2-D slice from a stack.")
        image, source_was_8bit = cls._coerce_image_for_annotation(data)
        return image, template, source_was_8bit

    @staticmethod
    def _analyze_template_from_data(data):
        template_data = np.array(data, copy=False)
        if template_data.ndim == 2:
            slices = 1
            height, width = template_data.shape
        elif template_data.ndim == 3:
            slices, height, width = template_data.shape
        else:
            raise ValueError("Analyze image must be 2-D or 3-D.")
        return {
            "slices": int(slices),
            "height": int(height),
            "width": int(width),
            "dtype": np.dtype(template_data.dtype),
        }

    def _show_single_image_canvas(self):
        notebook = getattr(self, "batch_results_notebook", None)
        if notebook is not None:
            notebook.pack_forget()
        self.image_canvas = self.single_image_canvas
        self._active_batch_result_tab = None
        if self.single_image_canvas.winfo_manager() != "pack":
            self.single_image_canvas.pack(fill="both", expand=True)

    def _show_batch_results_canvas(self):
        self.single_image_canvas.pack_forget()
        notebook = getattr(self, "batch_results_notebook", None)
        if notebook is not None and notebook.winfo_manager() != "pack":
            notebook.pack(fill="both", expand=True)

    def _show_image(self, image, path):
        """Load a new image and reset the annotation UI.

        Displays the image on the canvas and clears all tracing state:
          - Clears all saved and active boundary traces
          - Resets boundary completion status
          - Resets foveal center line
          - Updates info display and UI controls
          - Re-enables all controls for fresh annotation

        Args:
            image: numpy array to display and annotate. Source bit depth is preserved;
                the canvas normalizes a preview for display only.
            path: Path or description string for the image source.
        """
        self._show_single_image_canvas()
        self.current_file = path
        self.image_data = image
        self.active_boundary = None
        self.boundary_traces = {}
        self.boundary_order = []
        self.fovea_x = None

        self.image_canvas.set_image(image)
        self.image_canvas.enable_roi(False)

        filename = os.path.basename(path) if path and path != "Step 1 output" else "Step 1 output"
        self.image_info_var.set(
            f"{filename} | Size: {image.shape[1]} × {image.shape[0]} px | Type: {image.dtype}"
        )
        self.active_trace_var.set("No active boundary")
        self._reset_boundary_completion()
        self._set_drawing_locked(False)
        self._refresh_trace_list()
        self._sync_boundary_canvas_state()
        self._set_segmentation_frame_enabled(True)
        self._update_ai_button_states()
        self.status_var.set(
            "Image loaded. Drag the foveal center line or select an incomplete boundary to trace."
        )
        self._notify_output_folder_changed()

    def _clear_image_display(self, status_message=None):
        """Clear the editor canvas and reset image-specific annotation state."""
        self._show_single_image_canvas()
        self.current_file = None
        self.image_data = None
        self.active_boundary = None
        self.boundary_traces = {}
        self.boundary_order = []
        self.fovea_x = None
        self._input_analyze_template = None
        self._source_was_8bit = False

        self.image_canvas.set_image(None)
        self.image_canvas.enable_roi(False)
        self.image_canvas.enable_line(False)
        self.image_canvas.enable_vertical_line(False)

        self.image_info_var.set("No image loaded")
        self.active_trace_var.set("No active boundary")
        self.fovea_line_var.set("Fovea line: not set")
        self._updating_fovea_entry = True
        try:
            self.fovea_x_entry_var.set("")
        finally:
            self._updating_fovea_entry = False

        self._reset_boundary_completion()
        self._refresh_trace_list()
        self._set_fovea_controls_enabled(False)
        self._set_segmentation_frame_enabled(False)
        self._update_boundary_action_buttons()
        self._sync_boundary_canvas_state()
        self._update_ai_button_states()
        if status_message:
            self.status_var.set(status_message)

    def _notify_output_folder_changed(self):
        if not self.current_file or self.current_file == "Step 1 output":
            return
        folder = os.path.dirname(self.current_file)
        if folder:
            if self.on_output_folder_changed is not None:
                self.on_output_folder_changed(folder)

    @staticmethod
    def _copy_trace_dict(traces):
        copied = {}
        for name, trace in (traces or {}).items():
            copied[name] = {
                "points": list(trace.get("points", [])),
                "pixels": list(trace.get("pixels", [])),
                "color": trace.get("color", ""),
            }
        return copied

    @staticmethod
    def _vertical_line_trace(x, height):
        x = int(x)
        max_y = max(0, int(height) - 1)
        points = [(x, 0), (x, max_y)]
        return {
            "points": points,
            "pixels": _polyline_pixels(points),
            "color": BOUNDARY_COLORS.get(FOVEA_BOUNDARY_NAME, "#ffd500"),
        }

    def _ensure_fovea_line(self, x=None):
        if self.image_data is None:
            self.image_canvas.clear_vertical_line()
            self._set_fovea_controls_enabled(False)
            return None
        width = int(self.image_data.shape[1])
        height = int(self.image_data.shape[0])
        if x is None:
            x = self.fovea_x if self.fovea_x is not None else width // 2
        fovea_x = int(np.clip(int(x), 0, max(0, width - 1)))

        self.fovea_x = fovea_x
        self.boundary_traces[FOVEA_BOUNDARY_NAME] = self._vertical_line_trace(fovea_x, height)
        if FOVEA_BOUNDARY_NAME not in self.boundary_order:
            self.boundary_order.append(FOVEA_BOUNDARY_NAME)

        self._updating_fovea_entry = True
        try:
            self.fovea_x_entry_var.set(str(fovea_x))
        finally:
            self._updating_fovea_entry = False
        self.fovea_line_var.set(f"Fovea line: x={fovea_x}")
        self.image_canvas.set_vertical_line_x(fovea_x)
        self._set_fovea_controls_enabled(True)
        return fovea_x

    def _set_fovea_from_prediction(self, x):
        if self._ensure_fovea_line(x) is None:
            return
        self._rebuild_saved_overlays()
        self._refresh_trace_list()
        self._select_trace_by_name(FOVEA_BOUNDARY_NAME)
        self._sync_boundary_canvas_state()

    def _has_saved_fovea_trace(self):
        return FOVEA_BOUNDARY_NAME in self.boundary_traces

    def _capture_current_editor_state(self):
        return {
            "input": self.current_file,
            "image": self.image_data,
            "traces": self._copy_trace_dict(self.boundary_traces),
            "order": list(self.boundary_order),
            "fovea_x": self.fovea_x,
            "template": self._input_analyze_template,
            "source_was_8bit": getattr(self, "_source_was_8bit", False),
        }

    def _set_completion_from_traces(self):
        for name in BOUNDARY_NAMES:
            if name in self.boundary_completion_vars:
                self.boundary_completion_vars[name].set(name in self.boundary_traces)

    def _load_editor_state(self, state, canvas, status_message=None):
        if not state or canvas is None:
            return

        self.image_canvas = canvas
        self.current_file = state.get("input")
        self.image_data = state.get("image")
        self.boundary_traces = state.setdefault("traces", {})
        self.boundary_order = state.setdefault("order", [])
        self.fovea_x = state.get("fovea_x")
        if self.fovea_x is None and FOVEA_BOUNDARY_NAME in self.boundary_traces:
            points = self.boundary_traces.get(FOVEA_BOUNDARY_NAME, {}).get("points", [])
            if points:
                self.fovea_x = int(points[0][0])
                state["fovea_x"] = self.fovea_x
        self._input_analyze_template = state.get("template")
        self._source_was_8bit = bool(state.get("source_was_8bit", False))
        self.active_boundary = None

        self.image_canvas.enable_roi(False)
        self.image_canvas.clear_active_line()
        self._rebuild_saved_overlays()
        self._set_completion_from_traces()

        self._ensure_fovea_line(self.fovea_x)
        state["fovea_x"] = self.fovea_x

        filename = (
            os.path.basename(self.current_file)
            if self.current_file and self.current_file != "Step 1 output"
            else "Step 1 output"
        )
        if self.image_data is not None:
            self.image_info_var.set(
                f"{filename} | Size: {self.image_data.shape[1]} x {self.image_data.shape[0]} px | "
                f"Type: {self.image_data.dtype}"
            )

        self.active_trace_var.set("No active boundary")
        self._refresh_trace_list()
        selected_trace = None
        if self.fovea_x is not None and FOVEA_BOUNDARY_NAME in self.boundary_traces:
            selected_trace = FOVEA_BOUNDARY_NAME
        elif self.boundary_order:
            selected_trace = self.boundary_order[0]
        if selected_trace is not None:
            self._select_trace_by_name(selected_trace)
        self._refresh_boundary_lists(auto_select=False)
        self._set_drawing_locked(False)
        self._set_segmentation_frame_enabled(True)
        self._update_ai_button_states()
        self._sync_boundary_canvas_state()
        self.image_canvas.fit_to_window()
        if status_message:
            self.status_var.set(status_message)

    # ═══════════════════════════════════════════════════════════════════════
    #  Boundary tracing actions
    # ═══════════════════════════════════════════════════════════════════════
    def _get_listbox_selection(self, listbox, name_list):
        """Get the currently selected name from a listbox.

        Safe helper that handles None listbox, empty selection, and bounds checking.

        Args:
            listbox: tk.Listbox widget (or None).
            name_list: List of strings to index into.

        Returns:
            Selected string name, or None if no valid selection.
        """
        if listbox is None:
            return None
        selection = listbox.curselection()
        if selection:
            index = int(selection[0])
            if 0 <= index < len(name_list):
                return name_list[index]
        return None

    def _selected_boundary_name(self):
        """Get currently selected boundary, with fallback to active or next incomplete."""
        incomplete_names = self._incomplete_boundary_names()
        selected = self._get_listbox_selection(getattr(self, "boundary_incomplete_listbox", None), incomplete_names)
        if selected:
            return selected
        if self.active_boundary in BOUNDARY_NAMES:
            return self.active_boundary
        return self._next_incomplete_boundary() or BOUNDARY_NAMES[0]

    def _completed_boundary_names(self):
        """Return list of completed boundary names.

        A boundary is complete when its BooleanVar in boundary_completion_vars is True.
        """
        return [name for name in BOUNDARY_NAMES if self.boundary_completion_vars.get(name) and self.boundary_completion_vars[name].get()]

    def _incomplete_boundary_names(self):
        """Return list of incomplete (not yet finished) boundary names.

        Inverse of completed boundaries; used to populate the incomplete listbox.
        """
        completed = set(self._completed_boundary_names())
        return [name for name in BOUNDARY_NAMES if name not in completed]

    def _start_boundary_for_name(self, name, auto_advance=False):
        """Begin drawing a new boundary trace.

        Switches to the given boundary name and prepares for point-by-point drawing.
        If a trace already exists for this boundary, prompts user to overwrite.
        Keeps the foveal center marker visible and clears any active unfinished trace.

        Args:
            name: Boundary name from BOUNDARY_NAMES.
            auto_advance: If True, skip overwrite prompts (used after completing a boundary).

        Updates:
          - active_boundary to the given name
          - canvas to line-drawing mode
          - status bar with instructions
        """
        if self.image_data is None:
            messagebox.showwarning("No image", "Load an image before tracing boundaries.")
            return

        if name not in BOUNDARY_NAMES:
            return

        if self.image_canvas.get_active_line() and not auto_advance:
            if not messagebox.askyesno(
                "Discard current trace?",
                "Replace the current unfinished trace and start a new boundary?",
            ):
                return

        if name in self.boundary_traces and not auto_advance:
            if not messagebox.askyesno(
                "Overwrite boundary?",
                f"Boundary '{name}' already exists. Overwrite it with a new trace?",
            ):
                return
            self.boundary_traces.pop(name, None)
            if name in self.boundary_order:
                self.boundary_order.remove(name)
            self._rebuild_saved_overlays()

        self.active_boundary = name
        self.image_canvas.clear_active_line()
        self.boundary_workflow_status_var.set(f"Tracing {name}. Click a boundary name to switch targets.")
        self.status_var.set(
            f"Tracing {name}. Left-click to add points, undo points if needed, then finish the boundary."
        )
        self._update_active_trace_summary()
        self._update_boundary_action_buttons()
        self._sync_boundary_canvas_state()

    def _set_active_boundary_target(self, name):
        if name not in BOUNDARY_NAMES or self.image_data is None:
            return

        self.active_boundary = name
        self.image_canvas.clear_active_line()

        if name in self._completed_boundary_names():
            self._refresh_boundary_lists(select_completed_name=name)
        else:
            self._refresh_boundary_lists(select_incomplete_name=name)

        self.boundary_workflow_status_var.set(f"Active boundary: {name}")
        self.status_var.set(
            f"Tracing {name}. Left-click to add points, undo points if needed, then finish the boundary."
        )
        self._update_active_trace_summary()
        self._update_boundary_action_buttons()
        self._sync_boundary_canvas_state()

    def _selected_active_boundary_name(self):
        """Get the active boundary or selected from lists (incomplete preferred)."""
        if self.active_boundary in BOUNDARY_NAMES:
            return self.active_boundary
        incomplete = self._incomplete_boundary_names()
        selected = self._get_listbox_selection(getattr(self, "boundary_incomplete_listbox", None), incomplete)
        if selected:
            return selected
        completed = self._completed_boundary_names()
        return self._get_listbox_selection(getattr(self, "boundary_completed_listbox", None), completed)

    def _has_clearable_boundary_markers(self):
        has_saved_boundary = any(name != FOVEA_BOUNDARY_NAME for name in self.boundary_traces)
        has_active_boundary = bool(self.image_canvas.get_active_line())
        return has_saved_boundary or has_active_boundary

    def _update_boundary_action_buttons(self):
        active_name = self.active_boundary if self.active_boundary in BOUNDARY_NAMES else None
        is_completed = active_name in self._completed_boundary_names()
        is_incomplete = active_name in self._incomplete_boundary_names()
        has_clearable_markers = self._has_clearable_boundary_markers()

        if getattr(self, "finish_boundary_btn", None) is not None:
            self.finish_boundary_btn.configure(state="normal" if is_incomplete else "disabled")
        if getattr(self, "revert_boundary_btn", None) is not None:
            self.revert_boundary_btn.configure(state="normal" if is_completed else "disabled")
        if getattr(self, "clear_all_traces_btn", None) is not None:
            self.clear_all_traces_btn.configure(state="normal" if has_clearable_markers else "disabled")

        if getattr(self, "saved_button", None) is not None:
            if self._all_required_boundaries_complete():
                self.saved_button.state(["!disabled"]) # Enable if all 6 boundaries exist
            else:
                self.saved_button.state(["disabled"])  # Disable otherwise

    def _boundary_color(self, name):
        if name in BOUNDARY_COLORS:
            return BOUNDARY_COLORS[name]
        palette = ["#ffb703", "#00b4d8", "#ef476f", "#8ac926", "#fb8500", "#8338ec", "#2ec4b6", "#f15bb5"]
        index = sum(ord(ch) for ch in name) % len(palette)
        return palette[index]

    def _sync_boundary_canvas_state(self):
        """Enable or disable line drawing on canvas based on current state.

        Line drawing is DISABLED if:
          - AI segmentation is actively running (prevents accidental edits)
          - No boundary is selected or selected boundary is complete

        Completed boundaries must be reverted to incomplete to edit.

        The foveal center line remains visible whenever an image is loaded.

        This method is called after each state change (AI finish, boundary selection,
        vertical line toggle) to maintain consistent canvas interactivity.
        """
        active_name = self.active_boundary if self.active_boundary in BOUNDARY_NAMES else None
        # Only allow drawing on incomplete boundaries
        is_incomplete = active_name in self._incomplete_boundary_names()
        drawing_enabled = (
            self.image_data is not None
            and active_name is not None
            and is_incomplete
            and not self._drawing_locked
            and not self._segmenter_running
        )
        if self.image_data is not None and self.image_canvas.get_vertical_line_x() is None:
            self._ensure_fovea_line()
        self.image_canvas.enable_line(drawing_enabled)
        self.image_canvas.enable_vertical_line(self.image_data is not None)

    def _start_boundary(self):
        """Begin drawing the currently selected boundary from the incomplete list.

        Convenience method that resolves the selected incomplete boundary name and
        delegates to _start_boundary_for_name(). Used by the "Start" button.
        """
        if self.image_data is None:
            return

        name = self._selected_boundary_name()
        self._start_boundary_for_name(name)

    def _undo_point(self):
        """Remove the last vertex from the active boundary trace.

        Delegates to ImageCanvas.undo_active_line_vertex(). User can call this
        multiple times to step back through all points in the current trace.
        """
        self.image_canvas.undo_active_line_vertex()

    def _clear_active_trace(self):
        """Discard the current unfinished boundary trace and start fresh.

        Clears the active line on the canvas and updates status. Does not affect
        any completed boundaries.
        """
        self.image_canvas.clear_active_line()
        self._update_active_trace_summary()
        self._update_boundary_action_buttons()
        self.status_var.set("Current unfinished trace cleared.")

    def _finish_boundary(self):
        """Complete tracing the current active boundary and save it.

        Steps:
          1. Validate an incomplete boundary is selected
          2. Extract points from canvas active line
          3. Check minimum 2-point requirement
          4. Convert points to pixel coordinates using Bresenham's algorithm
          5. Save boundary to boundary_traces dict and mark as complete
          6. Refresh UI lists and auto-advance to next boundary

        Auto-saves MARKED images if all 6 boundaries are complete (unless disabled).
        Clears active line and prepares UI for next boundary.
        """
        if self.image_data is None:
            messagebox.showwarning("No image", "Load an image first.")
            return

        name = self._selected_active_boundary_name()
        if name is None or name not in self._incomplete_boundary_names():
            messagebox.showinfo("Select a boundary", "Choose an incomplete boundary first.")
            return
        points = self.image_canvas.get_active_line()
        if len(points) < 2:
            messagebox.showwarning("Incomplete trace", "Trace at least two points before finishing the boundary.")
            return

        pixels = _polyline_pixels(points)
        color = self._boundary_color(name)
        self.boundary_traces[name] = {
            "points": list(points),
            "pixels": pixels,
            "color": color,
        }
        if name not in self.boundary_order:
            self.boundary_order.append(name)

        self.image_canvas.commit_active_line(color=color, label=name)
        self.active_boundary = None
        next_name = self._mark_boundary_complete(name)
        self._refresh_trace_list()
        self._select_trace_by_name(name)
        self.active_trace_var.set("No active boundary")
        self._update_boundary_action_buttons()
        if next_name is not None:
            self._start_boundary_for_name(next_name, auto_advance=True)
            self.status_var.set(f"Saved '{name}'. Next boundary: {next_name}.")
        else:
            self.boundary_workflow_status_var.set("All preset boundaries are complete.")
            if getattr(self, "_active_batch_result_tab", None):
                self.status_var.set(
                    f"Saved '{name}'. All preset boundaries are complete. Use Save current MARKED Image when ready."
                )
                return
            try:
                self._save_marked_images(require_complete=True)
            except OSError as exc:
                messagebox.showerror("Save error", f"Could not save Light_MARKED Analyze image:\n{exc}")
                self.status_var.set(f"Saved '{name}'. All preset boundaries are complete.")

    def _revert_boundary(self):
        """Mark a completed boundary as incomplete and re-open it for editing.

        Removes the boundary from saved traces and completion status, clearing
        its pixels from all MARKED images. User can re-trace or edit.
        Prompts for confirmation to prevent accidental reverts.
        """
        name = self._selected_active_boundary_name()
        if name is None or name not in self._completed_boundary_names():
            messagebox.showinfo("Select a boundary", "Choose a completed boundary first.")
            return

        self.boundary_traces.pop(name, None)
        if name in self.boundary_order:
            self.boundary_order.remove(name)
        if name in self.boundary_completion_vars:
            self.boundary_completion_vars[name].set(False)

        self.active_boundary = name
        self.image_canvas.clear_active_line()
        self._rebuild_saved_overlays()
        self._refresh_trace_list()
        self._refresh_boundary_lists(select_incomplete_name=name)
        self._update_boundary_action_buttons()
        self._sync_boundary_canvas_state()
        self.active_trace_var.set(f"Active: {name} | no vertices placed yet")
        self.status_var.set(f"Reverted '{name}' back to incomplete.")

    def _delete_selected_boundary(self):
        """Delete a saved boundary from the trace list.

        Allows removal of individual saved boundaries without reverting all.
        Boundary must be selected in the trace listbox.
        """
        trace_listbox = getattr(self, "trace_listbox", None)
        if trace_listbox is None:
            messagebox.showinfo("Unavailable", "Boundary list is hidden in this layout.")
            return
        selection = trace_listbox.curselection()
        if not selection:
            messagebox.showinfo("Nothing selected", "Select a saved boundary first.")
            return
        name = self.boundary_order[selection[0]]
        if name == FOVEA_BOUNDARY_NAME:
            self._center_vertical_line()
            self.status_var.set("Foveal center line reset to image center.")
            return
        self.boundary_traces.pop(name, None)
        self.boundary_order.remove(name)
        if name in self.boundary_completion_vars:
            self.boundary_completion_vars[name].set(False)
        self._refresh_boundary_lists(select_incomplete_name=name)
        self._rebuild_saved_overlays()
        self._refresh_trace_list()
        self.status_var.set(f"Deleted boundary '{name}'.")

    def _clear_all_traces(self):
        """Remove all saved boundaries and the active trace at once.

        Asks for confirmation before clearing to prevent accidental data loss.
        Resets boundary completion status and clears canvas overlays.
        """
        has_boundary_traces = any(name != FOVEA_BOUNDARY_NAME for name in self.boundary_traces)
        if not has_boundary_traces and not self.image_canvas.get_active_line():
            return
        if not messagebox.askyesno(
            "Clear all traces?",
            "Remove every saved and active boundary trace? The foveal center line will be kept.",
        ):
            return
        fovea_x = self.fovea_x
        self.boundary_traces.clear()
        self.boundary_order.clear()
        self._reset_boundary_completion()
        self._ensure_fovea_line(fovea_x)
        self.active_boundary = None
        self.image_canvas.clear_line_overlays()
        self.image_canvas.clear_active_line()
        self._refresh_boundary_lists()
        self._refresh_trace_list()
        self.active_trace_var.set("No active boundary")
        self._update_boundary_action_buttons()
        self.status_var.set("All boundary traces cleared. Foveal center line kept.")

    def _rebuild_saved_overlays(self):
        """Redraw all saved boundary overlays on the canvas from boundary_traces.

        Called after any modification to boundary_traces or boundary_order.
        Maintains order and colors when re-rendering the canvas display.
        The fovea line is drawn only by the interactive vertical marker.
        """
        overlays = []
        for name in self.boundary_order:
            if name == FOVEA_BOUNDARY_NAME:
                continue
            trace = self.boundary_traces.get(name)
            if trace:
                overlays.append({
                    "points": trace["points"],
                    "color": trace.get("color") or self._boundary_color(name),
                    "label": name,
                })
        self.image_canvas.set_line_overlays(overlays)

    def _reset_boundary_completion(self):
        """Reset all boundaries to incomplete status and update UI.

        Used when clearing all traces or loading a new image.
        Resets all BooleanVar completion flags and refreshes listboxes.
        """
        for var in self.boundary_completion_vars.values():
            var.set(False)
        self._refresh_boundary_lists(auto_select=False)
        self._update_boundary_progress_bar()

    def _mark_boundary_complete(self, name):
        """Mark a boundary as complete and auto-advance to the next incomplete.

        Updates boundary_completion_vars and refreshes UI lists.
        Returns the next incomplete boundary name for auto-advancing, or None if complete.

        Args:
            name: Boundary name to mark as complete.

        Returns:
            Next incomplete boundary name, or None if all boundaries are complete.
        """
        if name in self.boundary_completion_vars:
            self.boundary_completion_vars[name].set(True)

        next_name = self._next_incomplete_boundary(name)
        if next_name is not None:
            self._refresh_boundary_lists(select_incomplete_name=next_name)
        else:
            self._refresh_boundary_lists(select_completed_name=name)
        self._update_boundary_progress_bar()
        return next_name

    def _update_boundary_progress_bar(self):
        """Update progress bar to show completed boundaries out of 6."""
        completed_count = len(self._completed_boundary_names())
        if hasattr(self, "segmenter_progress"):
            self.segmenter_progress.configure(maximum=len(BOUNDARY_NAMES), mode="determinate")
            self.segmenter_progress["value"] = completed_count

    def _next_incomplete_boundary(self, current_name=None):
        """Get the next incomplete boundary in priority order.

        Starts from the boundary after current_name (or from the first boundary).
        Cycles through BOUNDARY_NAMES to find the next one with BooleanVar.get() = False.

        Args:
            current_name: Current boundary name to start search after (optional).

        Returns:
            Next incomplete boundary name, or None if all boundaries are complete.
        """
        if current_name in BOUNDARY_NAMES:
            start_index = BOUNDARY_NAMES.index(current_name) + 1
        else:
            start_index = 0

        for name in BOUNDARY_NAMES[start_index:]:
            var = self.boundary_completion_vars.get(name)
            if var is not None and not var.get():
                return name
        for name in BOUNDARY_NAMES[:start_index]:
            var = self.boundary_completion_vars.get(name)
            if var is not None and not var.get():
                return name
        return None

    def _populate_boundary_listbox(self, listbox, names, selected_name):
        """Populate a boundary listbox with names, marking the selected one."""
        listbox.delete(0, "end")
        for name in names:
            active = "▶" if name == selected_name else " "
            listbox.insert("end", f"{active} {name}")

    def _refresh_boundary_lists(self, select_incomplete_name=None, select_completed_name=None, auto_select=True):
        """Update the incomplete/completed boundary listboxes and sync selection.

        Rebuilds both listboxes from current boundary completion state and optionally
        selects a specific boundary. Auto-selects the first incomplete boundary if
        auto_select=True and no explicit selection provided.

        This method is called after boundary state changes (marking complete, reverting,
        clearing) to keep the UI synchronized with the internal boundary_completion_vars dict.

        Args:
            select_incomplete_name: Boundary name to select in incomplete list (if it exists).
            select_completed_name: Boundary name to select in completed list (if it exists).
            auto_select: If True, auto-select first incomplete if no explicit selection.
        """
        if getattr(self, "boundary_incomplete_listbox", None) is None:
            return

        incomplete_names = self._incomplete_boundary_names()
        completed_names = self._completed_boundary_names()
        selected_name = select_incomplete_name if select_incomplete_name is not None else select_completed_name

        self._syncing_boundary_selection = True
        self.boundary_incomplete_listbox.selection_clear(0, "end")
        self.boundary_completed_listbox.selection_clear(0, "end")
        
        self._populate_boundary_listbox(self.boundary_incomplete_listbox, incomplete_names, selected_name)
        self._populate_boundary_listbox(self.boundary_completed_listbox, completed_names, selected_name)

        current = selected_name
        if current is None:
            current = self.active_boundary if self.active_boundary in incomplete_names else None
        if current is None and auto_select:
            current = self._next_incomplete_boundary()

        if current in incomplete_names:
            index = incomplete_names.index(current)
            self.boundary_incomplete_listbox.selection_clear(0, "end")
            self.boundary_incomplete_listbox.selection_set(index)
            self.boundary_incomplete_listbox.see(index)

        if select_completed_name in completed_names:
            index = completed_names.index(select_completed_name)
            self.boundary_completed_listbox.selection_clear(0, "end")
            self.boundary_completed_listbox.selection_set(index)
            self.boundary_completed_listbox.see(index)
        self._syncing_boundary_selection = False

        fovea_status = " | Fovea center: done" if self._has_saved_fovea_trace() else ""
        self.boundary_workflow_status_var.set(
            f"Completed: {len(completed_names)}/{len(BOUNDARY_NAMES)}"
            f"{fovea_status}"
        )
        self._update_boundary_action_buttons()
        self._sync_boundary_canvas_state()

    def _on_boundary_selected(self, _event, list_type):
        """Handle selection from incomplete or completed boundary listbox."""
        if self._syncing_boundary_selection:
            return
        if list_type == "incomplete":
            names = self._incomplete_boundary_names()
            listbox = getattr(self, "boundary_incomplete_listbox", None)
            other_listbox = getattr(self, "boundary_completed_listbox", None)
        else:
            names = self._completed_boundary_names()
            listbox = getattr(self, "boundary_completed_listbox", None)
            other_listbox = getattr(self, "boundary_incomplete_listbox", None)
        
        name = self._get_listbox_selection(listbox, names)
        if name:
            if other_listbox:
                other_listbox.selection_clear(0, "end")
            self._set_active_boundary_target(name)

    def _on_boundary_incomplete_selected(self, _event):
        self._on_boundary_selected(_event, "incomplete")

    def _on_boundary_completed_selected(self, _event):
        self._on_boundary_selected(_event, "completed")

    def _center_vertical_line(self):
        if self.image_data is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        width = int(self.image_data.shape[1])
        center_x = width // 2
        self._set_fovea_from_prediction(center_x)

    def _set_drawing_locked(self, locked):
        self._drawing_locked = bool(locked)
        self._stop_fovea_repeat()

        self._set_fovea_controls_enabled(self.image_data is not None and not self._drawing_locked)
        if self._drawing_locked:
            self.image_canvas.enable_line(False)
        self._sync_boundary_canvas_state()

    def _set_fovea_controls_enabled(self, enabled):
        """Enable/disable fovea-specific controls based on mode and lock state."""
        state = "normal" if (enabled and not self._drawing_locked) else "disabled"
        for widget in (
            self.fovea_stepper,
            self.fovea_reset_btn,
        ):
            widget.configure(state=state)

    def _set_segmentation_frame_enabled(self, enabled):
        """Enable/disable segmentation controls, keeping the batch AI button available."""
        state = "normal" if enabled else "disabled"
        # Recursively disable/enable all children in the segmentation frame
        def set_state(widget, s):
            if widget in (
                self.finish_boundary_btn,
                self.revert_boundary_btn,
                getattr(self, "batch_ai_button", None),
                getattr(self, "save_btn", None),
            ):
                return
            try:
                widget.configure(state=s)
            except Exception:
                pass
            for child in widget.winfo_children():
                set_state(child, s)
        
        if hasattr(self, "segmentation_frame"):
            set_state(self.segmentation_frame, state)
        self._update_batch_ai_button_state()


    def _on_fovea_x_entry_changed(self, *_):
        """Apply Center X edits immediately when they are valid integers in range."""
        if self._updating_fovea_entry or self.image_data is None:
            return
        raw_value = self.fovea_x_entry_var.get().strip()
        if not raw_value:
            return
        try:
            x = int(raw_value)
        except ValueError:
            return
        max_x = int(self.image_data.shape[1]) - 1
        x = max(0, min(x, max_x))
        if str(x) != raw_value:
            self._updating_fovea_entry = True
            self.fovea_x_entry_var.set(str(x))
            self._updating_fovea_entry = False
        self._ensure_fovea_line(x)
        self._refresh_trace_list()

    def _nudge_fovea_x(self, delta):
        if self.image_data is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        width = int(self.image_data.shape[1])
        max_x = width - 1
        current = self.fovea_x
        if current is None:
            try:
                current = int(self.fovea_x_entry_var.get().strip())
            except ValueError:
                current = width // 2
        next_x = max(0, min(int(current) + int(delta), max_x))
        self._updating_fovea_entry = True
        self.fovea_x_entry_var.set(str(next_x))
        self._updating_fovea_entry = False
        self._ensure_fovea_line(next_x)
        self._refresh_trace_list()

    def _bind_fovea_nudge_button(self, button, delta):
        button.bind("<ButtonPress-1>", lambda _event, d=delta: self._start_fovea_repeat(d))
        button.bind("<ButtonRelease-1>", lambda _event: self._stop_fovea_repeat())
        button.bind("<Leave>", lambda _event: self._stop_fovea_repeat())

    def _start_fovea_repeat(self, delta):
        if self.image_data is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        self._stop_fovea_repeat()
        self._fovea_repeat_dir = 1 if delta > 0 else -1
        self._fovea_repeat_ticks = 0

        # Apply one immediate step on initial click.
        self._nudge_fovea_x(self._fovea_repeat_dir)

        # Brief initial delay, then repeat with acceleration.
        self._fovea_repeat_job = self.after(180, self._run_fovea_repeat)

    def _run_fovea_repeat(self):
        if self._fovea_repeat_dir == 0 or self.image_data is None:
            self._stop_fovea_repeat()
            return

        self._fovea_repeat_ticks += 1

        step = 1
        interval = 90
        if self._fovea_repeat_ticks >= 5:
            step = 3
            interval = 55
        if self._fovea_repeat_ticks >= 14:
            step = 6
            interval = 35

        self._nudge_fovea_x(self._fovea_repeat_dir * step)
        self._fovea_repeat_job = self.after(interval, self._run_fovea_repeat)

    def _stop_fovea_repeat(self):
        if self._fovea_repeat_job is not None:
            self.after_cancel(self._fovea_repeat_job)
            self._fovea_repeat_job = None
        self._fovea_repeat_dir = 0
        self._fovea_repeat_ticks = 0

    def _on_vertical_line_changed(self, x):
        self.fovea_x = x
        fovea_line_var = getattr(self, "fovea_line_var", None)
        fovea_x_entry_var = getattr(self, "fovea_x_entry_var", None)

        if x is None:
            if fovea_line_var is not None:
                fovea_line_var.set("Fovea line: not set")
            self._updating_fovea_entry = True
            if fovea_x_entry_var is not None:
                fovea_x_entry_var.set("")
            self._updating_fovea_entry = False
            return
        if self.image_data is not None and FOVEA_BOUNDARY_NAME in self.boundary_traces:
            self.boundary_traces[FOVEA_BOUNDARY_NAME] = self._vertical_line_trace(
                int(x),
                int(self.image_data.shape[0]),
            )
        if fovea_line_var is not None:
            fovea_line_var.set(f"Fovea line: x={x}")
        self._updating_fovea_entry = True
        if fovea_x_entry_var is not None:
            fovea_x_entry_var.set(str(x))
        self._updating_fovea_entry = False

    # ═══════════════════════════════════════════════════════════════════════
    #  Live updates and selection handling
    # ═══════════════════════════════════════════════════════════════════════
    def _on_active_line_changed(self, points):
        if self.image_data is None:
            return
        if not points:
            self._update_active_trace_summary()
            self._update_boundary_action_buttons()
            return

        pixels = _polyline_pixels(points)
        name = self.active_boundary or self._selected_boundary_name()
        self.active_trace_var.set(
            f"Active: {name} | {len(points)} vertices | {len(pixels)} pixel(s) on line"
        )
        self._update_boundary_action_buttons()

    def _on_mouse_moved(self, ix, iy, val):
        if self.image_data is None:
            return
        ih, iw = self.image_data.shape[:2]
        z = self.image_canvas.get_zoom()
        fovea_text = f"  |  Fovea x: {self.fovea_x}" if self.fovea_x is not None else ""
        self.status_var.set(
            f"({ix}, {iy})  val={val}  |  Image: {iw}×{ih} {self.image_data.dtype}  |  Zoom: {z * 100:.0f}%"
            f"{fovea_text}"
        )

    def _refresh_trace_list(self):
        trace_listbox = getattr(self, "trace_listbox", None)
        if trace_listbox is None:
            return
        trace_listbox.delete(0, "end")
        for name in self.boundary_order:
            trace = self.boundary_traces.get(name)
            if not trace:
                continue
            if name == FOVEA_BOUNDARY_NAME:
                points = trace.get("points", [])
                x_value = int(points[0][0]) if points else self.fovea_x
                trace_listbox.insert("end", f"{name} - done, x={x_value}")
                continue
            trace_listbox.insert(
                "end",
                f"{name} — {len(trace['points'])} vertices, {len(trace['pixels'])} pixels",
            )

    def _select_trace_by_name(self, name):
        trace_listbox = getattr(self, "trace_listbox", None)
        if trace_listbox is None:
            return
        if name not in self.boundary_order:
            return
        index = self.boundary_order.index(name)
        trace_listbox.selection_clear(0, "end")
        trace_listbox.selection_set(index)
        trace_listbox.see(index)

    def _on_saved_boundary_selected(self, _event):
        trace_listbox = getattr(self, "trace_listbox", None)
        if trace_listbox is None:
            return
        selection = trace_listbox.curselection()
        if not selection:
            return
        name = self.boundary_order[selection[0]]

    def _update_active_trace_summary(self):
        if self.active_boundary is None:
            self.active_trace_var.set("No active boundary")
            return
        points = self.image_canvas.get_active_line()
        if not points:
            self.active_trace_var.set(f"Active: {self.active_boundary} | no vertices placed yet")
            return
        pixels = _polyline_pixels(points)
        self.active_trace_var.set(
            f"Active: {self.active_boundary} | {len(points)} vertices | {len(pixels)} pixel(s) on line"
        )


    def _all_required_boundaries_complete(self):
        """Check if all 6 preset boundaries have been traced and completed.

        Returns:
            Boolean: True if all BOUNDARY_NAMES are present in boundary_traces.
        """
        return all(name in self.boundary_traces for name in BOUNDARY_NAMES)

    def _marked_output_basepath(self, basename):
        if self.current_file and self.current_file != "Step 1 output":
            out_dir = os.path.dirname(self.current_file)
        else:
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(out_dir, basename)

    def _source_output_basepath(self, basename):
        return self._marked_output_basepath(basename)

    def _source_output_basepaths(self):
        return [self._source_output_basepath(LIGHT_SOURCE_BASENAME)]

    def _reference_marked_volume_spec(self):
        """Get the target dimensions and dtype for MARKED output volumes.

        Returns standard output format (test1 format):
          - Slices: 2
          - Height: current image height
          - Width: 2133
          - Dtype: uint8

        This ensures all MARKED images are saved in a consistent, normalized format
        while preserving the input image height.

        Returns:
            Tuple (slices, height, width, dtype): Standard dimensions for output Analyze volumes.
        """
        if self.image_data is None:
            target_height = STANDARD_OUTPUT_HEIGHT
        else:
            target_height = int(self.image_data.shape[0])
        return (STANDARD_OUTPUT_SLICES, target_height, STANDARD_OUTPUT_WIDTH, np.dtype(np.uint8))

    def _build_marked_image(self):
        """Create the base marked image with background set to MARKED_BACKGROUND_MAX.

        Starts with a copy of the current image (scaled to 8-bit if needed) and clamps
        all values to MARKED_BACKGROUND_MAX (230) so that boundary mark values (243, 254, etc.)
        stand out clearly against the background.

        Returns:
            8-bit uint8 numpy array with background clamped to 230.
        """
        marked = self._image_uint8(self.image_data).copy()
        marked = np.minimum(marked, np.uint8(MARKED_BACKGROUND_MAX))
        return marked

    @staticmethod
    def _render_trace_mask(trace, width, height, boundary_name=None):
        """Rasterize a boundary trace into a binary mask for marking.

        Converts polyline vertices to pixel coordinates, applies anti-aliasing via
        Bresenham's algorithm, and thickens the line by the specified boundary width.
        The mask marks pixels at 1.0 (white) where the boundary should be marked.

        Args:
            trace: Boundary dict with 'points' or 'pixels' key (list of (x, y) tuples).
            width: Target mask width (pixels).
            height: Target mask height (pixels).
            boundary_name: Boundary name to look up line width (optional).

        Returns:
            Boolean numpy array (height, width), or None if no pixels to mark.
        """
        points = trace.get("points") or trace.get("pixels") or []
        if not points:
            return None

        # Determine line width based on boundary type
        line_width = MARKED_BOUNDARY_WIDTHS.get(boundary_name, 1)

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        line_points = [(int(x), int(y)) for x, y in points]

        if len(line_points) == 1:
            x, y = line_points[0]
            radius = max(1, line_width // 2)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        else:
            draw.line(line_points, fill=1, width=line_width)

        return np.asarray(mask, dtype=np.uint8) > 0

    @staticmethod
    def _apply_mark_values_to_image(target_image, traces, mark_values):
        height, width = target_image.shape[:2]
        for name, mark_value in mark_values.items():
            trace = traces.get(name)
            if not trace:
                continue
            trace_mask = Step2Frame._render_trace_mask(trace, width, height, boundary_name=name)
            if trace_mask is None:
                continue
            target_image[trace_mask] = np.uint8(mark_value)

    def _apply_mark_values(self, target_image, mark_values):
        self._apply_mark_values_to_image(target_image, self.boundary_traces, mark_values)

    def _build_marked_volume(self):
        """Generate a Light_MARKED 8-bit Analyze volume from traced boundaries.

        The volume contains all six traced boundaries and the fovea marker on
        the capped source-image background.

        All output volumes are automatically resized to standard format (current annotation height × 2133 width, 2 slices).

        Returns:
            A 3-D uint8 numpy array in standard format.
        """
        target_slices, target_h, target_w, _target_dtype = self._reference_marked_volume_spec()
        base_marked = self._build_marked_image()
        
        # Get original dimensions before any resize
        orig_h, orig_w = base_marked.shape[:2]
        
        # Resize image to target standard width if needed. Height is preserved.
        if (orig_h, orig_w) != (target_h, target_w):
            base_marked = np.array(
                Image.fromarray(base_marked).resize((target_w, target_h), RESAMPLE_NEAREST),
                dtype=np.uint8,
            )
            # Calculate scaling factors for boundary coordinates
            scale_x = target_w / orig_w
            scale_y = 1.0
            
            # Create scaled versions of boundary traces
            scaled_traces = {}
            for name, trace in self.boundary_traces.items():
                scaled_points = []
                for x, y in trace.get("points", []):
                    scaled_x = int(np.round(x * scale_x))
                    scaled_y = int(np.round(y * scale_y))
                    # Clamp to valid range
                    scaled_x = max(0, min(scaled_x, target_w - 1))
                    scaled_y = max(0, min(scaled_y, target_h - 1))
                    scaled_points.append((scaled_x, scaled_y))

                scaled_pixels = []
                for x, y in trace.get("pixels", []):
                    scaled_x = int(np.round(x * scale_x))
                    scaled_y = int(np.round(y * scale_y))
                    # Clamp to valid range
                    scaled_x = max(0, min(scaled_x, target_w - 1))
                    scaled_y = max(0, min(scaled_y, target_h - 1))
                    scaled_pixels.append((scaled_x, scaled_y))
                scaled_traces[name] = {
                    "points": scaled_points,
                    "pixels": scaled_pixels,
                    "color": trace.get("color", ""),
                }
            
            # Temporarily use scaled traces for marking
            original_traces = self.boundary_traces
            self.boundary_traces = scaled_traces
        else:
            original_traces = None

        # Standard MARKED format: 8-bit with base intensities capped at 230.
        light_slice = np.array(base_marked, copy=True)
        self._apply_mark_values(light_slice, COMMON_MARK_VALUES)
        self._apply_mark_values(light_slice, ADDITIONAL_MARK_VALUES)

        # Restore original traces if we modified them
        if original_traces is not None:
            self.boundary_traces = original_traces

        # Create volume with standard number of slices (always 2 in standard format)
        nslices = max(1, int(target_slices))
        light_volume = np.stack([light_slice] * nslices, axis=0).astype(np.uint8, copy=False)

        # Ensure output volumes are in standard format
        light_volume = _resize_to_standard_format(light_volume)

        return light_volume

    def _source_image_for_original_save(self):
        """Return the 16-bit source image that corresponds to current annotations."""
        return self.image_data

    def _save_light_image(self, reference_shape=None):
        """Export an unmarked LIGHT Analyze volume from the current image.

        This output uses the 16-bit image that corresponds to the current
        annotation coordinates, resized to the saved MARKED volume shape.

        Returns:
            List of saved file basepaths (without .img/.hdr extension).
        """
        source_image = self._source_image_for_original_save()
        if source_image is None:
            return []

        image = self._image_int16_for_original_save(source_image)
        if image.ndim != 2:
            raise ValueError("Current Step 2 image must be 2-D.")

        # Create a 16-bit 3D stack with 2 identical slices
        stack = np.stack([image, image], axis=0)

        # Match the annotation/MARKED output shape exactly.
        if reference_shape is not None:
            stack = _resize_volume_to_shape(stack, reference_shape)
        else:
            stack = _resize_to_standard_format(stack)

        saved_paths = []
        for base_path in self._source_output_basepaths():
            write_analyze(base_path, stack)
            saved_paths.append(base_path)
        return saved_paths

    def _save_marked_images(self, require_complete=False, prompt_on_incomplete=False):
        """Generate and save a Light_MARKED Analyze volume.

        The MARKED volumes are 8-bit Analyze files with boundary pixels marked at
        specific intensity values per ImageJ macro conventions. Boundaries are
        rasterized at their specified line widths and mark values.

        Args:
            require_complete: If True, only save if all 6 preset boundaries complete.
            prompt_on_incomplete: If True and incomplete, ask user for confirmation.

        Returns:
            Boolean: True if save succeeded, False if cancelled or no traces.
        """
        if self.image_data is None:
            return False
        if not self.boundary_traces:
            return False

        complete = self._all_required_boundaries_complete()
        if require_complete and not complete:
            return False
        if prompt_on_incomplete and not complete:
            proceed = messagebox.askyesno(
                "Boundaries incomplete",
                "Not all six preset boundaries are complete. Save MARKED images with current traces anyway?",
            )
            if not proceed:
                return False

        light_marked = self._build_marked_volume()
        saved_paths = []

        light_base_path = self._marked_output_basepath(LIGHT_MARKED_BASENAME)
        write_analyze(light_base_path, light_marked)
        saved_paths.append(light_base_path)

        saved_paths.extend(self._save_light_image(reference_shape=light_marked.shape))

        self.status_var.set(
            "Saved marked images -> "
            + ", ".join(f"{path}.img" for path in saved_paths)
        )
        return True

    def _current_marked_output_basepath(self):
        if self.current_file and self.current_file != "Step 1 output":
            return os.path.splitext(self.current_file)[0] + "_MARKED"
        return self._marked_output_basepath("Current_MARKED")

    def _build_current_marked_volume(self):
        if self.image_data is None:
            raise ValueError("No image is loaded.")

        target_slices, target_h, target_w, _target_dtype = self._reference_marked_volume_spec()
        base_marked = self._build_marked_image()
        orig_h, orig_w = base_marked.shape[:2]
        traces = self._copy_trace_dict(self.boundary_traces)

        if (orig_h, orig_w) != (target_h, target_w):
            base_marked = np.array(
                Image.fromarray(base_marked).resize((target_w, target_h), RESAMPLE_NEAREST),
                dtype=np.uint8,
            )
            traces = self._scale_traces_to_shape(traces, (orig_h, orig_w), (target_h, target_w))

        marked_slice = np.array(base_marked, copy=True)
        self._apply_mark_values_to_image(marked_slice, traces, COMMON_MARK_VALUES)
        self._apply_mark_values_to_image(marked_slice, traces, ADDITIONAL_MARK_VALUES)

        nslices = max(1, int(target_slices))
        volume = np.stack([marked_slice] * nslices, axis=0).astype(np.uint8, copy=False)
        return _resize_to_standard_format(volume)

    def _save_current_marked_image(self, prompt_on_incomplete=False, sync_active=True):
        if self.image_data is None or not self.boundary_traces:
            return None

        if sync_active:
            self._sync_active_batch_result_state()
        complete = self._all_required_boundaries_complete()
        if prompt_on_incomplete and not complete:
            proceed = messagebox.askyesno(
                "Boundaries incomplete",
                "Not all six preset boundaries are complete. Save the current MARKED image with current traces anyway?",
            )
            if not proceed:
                return None

        out_base = self._current_marked_output_basepath()
        write_analyze(out_base, self._build_current_marked_volume())
        self.status_var.set(f"Saved current MARKED image -> {out_base}.img")
        return out_base

    def _batch_state_complete(self, state):
        traces = state.get("traces") if state else None
        return bool(traces) and all(name in traces for name in BOUNDARY_NAMES)

    def _save_batch_result_state(self, tab_key, save_orientation_pair=False):
        state = self._batch_result_states.get(tab_key)
        if not state:
            return None

        if tab_key == getattr(self, "_active_batch_result_tab", None):
            self._sync_active_batch_result_state()
            if save_orientation_pair:
                return self._save_current_marked_orientation_pair()
            return self._save_current_marked_image(sync_active=False)

        saved_context = {
            "current_file": self.current_file,
            "image_data": self.image_data,
            "boundary_traces": self.boundary_traces,
            "boundary_order": self.boundary_order,
            "fovea_x": self.fovea_x,
            "template": self._input_analyze_template,
            "source_was_8bit": getattr(self, "_source_was_8bit", False),
            "completion": {
                name: var.get()
                for name, var in self.boundary_completion_vars.items()
            },
        }
        try:
            self.current_file = state.get("input")
            self.image_data = state.get("image")
            self.boundary_traces = state.get("traces") or {}
            self.boundary_order = state.get("order") or []
            self.fovea_x = state.get("fovea_x")
            self._input_analyze_template = state.get("template")
            self._source_was_8bit = bool(state.get("source_was_8bit", False))
            self._set_completion_from_traces()
            if save_orientation_pair:
                return self._save_current_marked_orientation_pair()
            return self._save_current_marked_image(sync_active=False)
        finally:
            self.current_file = saved_context["current_file"]
            self.image_data = saved_context["image_data"]
            self.boundary_traces = saved_context["boundary_traces"]
            self.boundary_order = saved_context["boundary_order"]
            self.fovea_x = saved_context["fovea_x"]
            self._input_analyze_template = saved_context["template"]
            self._source_was_8bit = saved_context["source_was_8bit"]
            for name, value in saved_context["completion"].items():
                if name in self.boundary_completion_vars:
                    self.boundary_completion_vars[name].set(value)

    def _save_current_marked_orientation_pair(self):
        """Save original and MARKED Light volumes for nasal and temporal orientations."""
        if self.image_data is None or not self.boundary_traces:
            return None
        if not self.current_file or self.current_file == "Step 1 output":
            raise ValueError("The batch image does not have a source file path.")

        source_base = os.path.splitext(os.path.basename(self.current_file))[0] + "_MARKED"
        parent = os.path.dirname(os.path.abspath(self.current_file))
        nasal_base = os.path.join(parent, "nasal", source_base)
        temporal_base = os.path.join(parent, "temporal", source_base)
        nasal_light_base = os.path.join(parent, "nasal", LIGHT_SOURCE_BASENAME)
        temporal_light_base = os.path.join(parent, "temporal", LIGHT_SOURCE_BASENAME)
        os.makedirs(os.path.dirname(nasal_base), exist_ok=True)
        os.makedirs(os.path.dirname(temporal_base), exist_ok=True)

        volume = self._build_current_marked_volume()
        source_image = self._image_int16_for_original_save(self.image_data)
        source_volume = np.stack([source_image, source_image], axis=0)
        source_volume = _resize_volume_to_shape(source_volume, volume.shape)
        write_analyze(nasal_base, volume)
        write_analyze(nasal_light_base, source_volume)
        # Analyze volumes are (slice, row, column); reversing columns mirrors left/right.
        write_analyze(temporal_base, np.ascontiguousarray(np.flip(volume, axis=2)))
        write_analyze(temporal_light_base, np.ascontiguousarray(np.flip(source_volume, axis=2)))
        return nasal_base, temporal_base

    def _save_all_batch_result_tabs(self):
        self._sync_active_batch_result_state()
        tab_keys = [
            tab
            for tab in (self.batch_results_notebook.tabs() if self.batch_results_notebook is not None else [])
            if tab in self._batch_result_states
        ]
        if not tab_keys:
            return []

        incomplete = [
            os.path.basename(self._batch_result_states[key].get("input") or "")
            for key in tab_keys
            if not self._batch_state_complete(self._batch_result_states[key])
        ]
        if incomplete:
            preview = "\n".join(incomplete[:6])
            if len(incomplete) > 6:
                preview += f"\n...and {len(incomplete) - 6} more"
            proceed = messagebox.askyesno(
                "Boundaries incomplete",
                "Some open tabs do not have all six preset boundaries complete.\n\n"
                f"{preview}\n\nSave all open tabs anyway?",
            )
            if not proceed:
                return []

        saved = []
        saved_tab_keys = []
        failures = []
        for tab_key in tab_keys:
            state = self._batch_result_states.get(tab_key)
            try:
                output_pair = self._save_batch_result_state(tab_key, save_orientation_pair=True)
                if output_pair:
                    saved.append(output_pair)
                    saved_tab_keys.append(tab_key)
            except (OSError, ValueError, RuntimeError) as exc:
                name = os.path.basename((state or {}).get("input") or tab_key)
                failures.append(f"{name}: {exc}")

        self._close_saved_batch_result_tabs(saved_tab_keys)
        if failures:
            messagebox.showerror("Save error", "Could not save some open tabs:\n" + "\n".join(failures[:8]))
        if saved:
            self.status_var.set(
                f"Saved original and MARKED Light volumes and closed {len(saved)} batch tab(s)."
            )
        return saved

    def _close_saved_batch_result_tabs(self, tab_keys):
        """Close already-saved batch tabs without showing per-tab save prompts."""
        notebook = getattr(self, "batch_results_notebook", None)
        if notebook is None:
            return

        for tab_key in tab_keys:
            canvas = self._batch_result_tab_canvases.pop(tab_key, None)
            self._batch_result_states.pop(tab_key, None)
            if canvas in self._batch_result_canvases:
                self._batch_result_canvases.remove(canvas)
            try:
                tab = notebook.nametowidget(tab_key)
                notebook.forget(tab)
                tab.destroy()
            except tk.TclError:
                pass

        if notebook.tabs():
            selected = notebook.select()
            if selected:
                self._activate_batch_result_tab(notebook.nametowidget(selected))
            self.image_info_var.set(f"AI batch results | {len(notebook.tabs())} image(s)")
            return

        notebook.destroy()
        self.batch_results_notebook = None
        self._batch_result_canvases = []
        self._batch_result_tab_canvases = {}
        self._batch_result_states = {}
        self._active_batch_result_tab = None
        self._single_editor_state = None
        self._show_single_image_canvas()
        self._clear_image_display("All batch result images were saved and closed.")

    def _save_all_batch_result_tabs_button(self):
        if getattr(self, "batch_results_notebook", None) is None or not self._batch_result_states:
            messagebox.showinfo("No open tabs", "There are no open batch result tabs to save.")
            return

        saved = self._save_all_batch_result_tabs()
        if saved:
            messagebox.showinfo(
                "Saved All",
                f"Saved and closed {len(saved)} image(s).\n\n"
                "Each nasal and temporal folder contains Light and Light_MARKED.\n"
                "Temporal images were mirrored left to right.",
            )

    def _save_current_marked_image_button(self):
        if self.image_data is None:
            messagebox.showwarning("No image", "Load or select an image before saving a MARKED output.")
            return
        if not self.boundary_traces:
            messagebox.showinfo("Nothing to save", "Trace or load boundaries before saving a MARKED output.")
            return

        active_tab = getattr(self, "_active_batch_result_tab", None)
        try:
            if active_tab and active_tab in self._batch_result_states:
                if not self._all_required_boundaries_complete():
                    proceed = messagebox.askyesno(
                        "Boundaries incomplete",
                        "Not all six preset boundaries are complete. Save this image anyway?",
                    )
                    if not proceed:
                        return
                output_pair = self._save_batch_result_state(
                    active_tab,
                    save_orientation_pair=True,
                )
                if output_pair:
                    self.status_var.set(
                        "Saved Light and Light_MARKED in the nasal and temporal folders."
                    )
                    messagebox.showinfo(
                        "Saved",
                        "Saved original and MARKED Light volumes in:\n"
                        f"{os.path.dirname(output_pair[0])}\n"
                        f"{os.path.dirname(output_pair[1])}",
                    )
                return
            out_base = self._save_current_marked_image(prompt_on_incomplete=True)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Save error", f"Could not save current MARKED image:\n{exc}")
            return

        if out_base:
            messagebox.showinfo("Saved", f"Saved current MARKED image:\n{out_base}.img")

    def _save_marked_images_button(self):
        """Button callback to manually save MARKED Analyze volumes.

        Provides user-facing save functionality with confirmation if boundaries
        are incomplete. Handles errors gracefully and shows save location.
        """
        if getattr(self, "batch_results_notebook", None) is not None and self._batch_result_states:
            saved = self._save_all_batch_result_tabs()
            if saved:
                message_lines = ["Saved open batch tabs:"]
                message_lines.extend(pair[0] + ".img" for pair in saved[:12])
                if len(saved) > 12:
                    message_lines.append(f"...and {len(saved) - 12} more.")
                messagebox.showinfo("Saved", "\n".join(message_lines))
            return

        if self.image_data is None:
            messagebox.showwarning("No image", "Load an image before saving MARKED outputs.")
            return
        if not self.boundary_traces:
            messagebox.showinfo("Nothing to save", "Trace boundaries first, then save MARKED outputs.")
            return

        try:
            saved = self._save_marked_images(prompt_on_incomplete=True)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Save error", f"Could not save MARKED Analyze images:\n{exc}")
            return

        if not saved:
            return

        light_path = self._marked_output_basepath(LIGHT_MARKED_BASENAME) + ".img"
        message_lines = ["Saved MARKED image:", light_path]
        message_lines.extend([path + ".img" for path in self._source_output_basepaths()])
        messagebox.showinfo("Saved", "\n".join(message_lines))

    # ═══════════════════════════════════════════════════════════════════════
    #  Neural segmentation
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _scale_traces_to_shape(traces, source_shape, target_shape):
        source_h, source_w = int(source_shape[0]), int(source_shape[1])
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        if source_h <= 0 or source_w <= 0:
            return {}
        if (source_h, source_w) == (target_h, target_w):
            return {
                name: {
                    "points": list(trace.get("points", [])),
                    "pixels": list(trace.get("pixels", [])),
                    "color": trace.get("color", ""),
                }
                for name, trace in traces.items()
            }

        scale_x = target_w / source_w
        scale_y = target_h / source_h
        scaled = {}
        for name, trace in traces.items():
            scaled_points = []
            for x, y in trace.get("points", []):
                sx = int(np.round(float(x) * scale_x))
                sy = int(np.round(float(y) * scale_y))
                sx = max(0, min(sx, target_w - 1))
                sy = max(0, min(sy, target_h - 1))
                scaled_points.append((sx, sy))
            scaled[name] = {
                "points": scaled_points,
                "pixels": _polyline_pixels(scaled_points),
                "color": trace.get("color", ""),
            }
        return scaled

    def _segmenter_subprocess_env(self):
        """Return an environment suitable for launching Windows CLI tools."""
        env = os.environ.copy()
        if os.name != "nt":
            return env

        system_root = env.get("SystemRoot") or env.get("WINDIR") or r"C:\Windows"

        def set_default_env(name, value):
            if not value:
                return
            for key in env:
                if key.lower() == name.lower():
                    return
            env[name] = value

        set_default_env("SystemRoot", system_root)
        set_default_env("WINDIR", system_root)

        system32 = os.path.join(system_root, "System32")
        cmd_exe = os.path.join(system32, "cmd.exe")
        if os.path.exists(cmd_exe):
            set_default_env("ComSpec", cmd_exe)

        path_key = next((key for key in env if key.lower() == "path"), "PATH")
        existing_path = env.get(path_key, "")
        existing_parts = [part for part in existing_path.split(os.pathsep) if part]
        existing_norms = {
            os.path.normcase(os.path.normpath(part))
            for part in existing_parts
        }
        windows_dirs = [
            system32,
            system_root,
            os.path.join(system32, "Wbem"),
            os.path.join(system32, "WindowsPowerShell", "v1.0"),
        ]
        appdata = env.get("APPDATA")
        if not appdata:
            userprofile = env.get("USERPROFILE")
            if userprofile:
                appdata = os.path.join(userprofile, "AppData", "Roaming")
        user_python_root = os.path.join(appdata, "Python") if appdata else None
        if user_python_root and os.path.isdir(user_python_root):
            for name in sorted(os.listdir(user_python_root), reverse=True):
                scripts_dir = os.path.join(user_python_root, name, "Scripts")
                if os.path.isdir(scripts_dir):
                    windows_dirs.append(scripts_dir)

        prepend_parts = []
        for path in windows_dirs:
            if not os.path.isdir(path):
                continue
            norm_path = os.path.normcase(os.path.normpath(path))
            if norm_path not in existing_norms:
                prepend_parts.append(path)
                existing_norms.add(norm_path)

        if prepend_parts:
            env[path_key] = os.pathsep.join(prepend_parts + existing_parts)

        return env

    def _aidas_worker_command(self):
        """Return the isolated AI_ForAIDAS worker command."""
        if getattr(sys, "frozen", False):
            return [sys.executable, "--aidas-ai-worker"]
        return [sys.executable, "-m", "aidas.ai.worker"]

    def _aidas_worker_env(self):
        """Return an environment for AI_ForAIDAS worker subprocesses."""
        env = self._segmenter_subprocess_env()
        if not getattr(sys, "frozen", False):
            root = self._app_root()
            path_key = next((key for key in env if key.lower() == "pythonpath"), "PYTHONPATH")
            existing = env.get(path_key, "")
            parts = [part for part in existing.split(os.pathsep) if part]
            root_norm = os.path.normcase(os.path.abspath(root))
            if root_norm not in {os.path.normcase(os.path.abspath(part)) for part in parts}:
                env[path_key] = os.pathsep.join([root] + parts)
        return env

    @staticmethod
    def _hidden_subprocess_kwargs():
        """Hide worker console windows on Windows."""
        kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        if os.name == "nt":
            try:
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = subprocess.SW_HIDE
                kwargs["startupinfo"] = si
            except Exception:
                pass
        return kwargs

    def _app_root(self):
        bundled_root = getattr(sys, "_MEIPASS", None)
        if bundled_root:
            return os.path.abspath(bundled_root)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _image_uint8(self, image):
        """Convert image to 8-bit (0-255) range with auto-scaling.

        If already 8-bit uint8, returns a copy. Otherwise, rescales to [0, 255]
        using min-max normalization:
          - Subtract minimum value
          - Divide by range (max - min)
          - Scale by 255
          - Clip to [0, 255] and convert to uint8

        Args:
            image: numpy array of any numeric dtype.

        Returns:
            8-bit uint8 numpy array.
        """
        return self._image_uint8_for_save(
            image,
            source_was_8bit=getattr(self, "_source_was_8bit", False),
        )

    @staticmethod
    def _image_uint8_for_save(image, source_was_8bit=False):
        if image.dtype == np.uint8:
            return np.array(image, copy=False)
        if source_was_8bit and image.dtype == np.uint16:
            return np.clip(
                np.rint(image.astype(np.float64) * (255.0 / 65535.0)),
                0,
                255,
            ).astype(np.uint8)
        arr = image.astype(np.float64)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _image_int16_for_original_save(self, image):
        """Convert unmarked LIGHT save data to signed 16-bit for Analyze.

        Step 1/Analyze data may already be signed int16. Preserve those values
        exactly; clipping negatives to zero can turn valid OCT signal black.
        8-bit sources promoted by Step 2 are scaled into the positive signed
        range before writing.
        """
        arr = np.asarray(image)
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)

        if arr.dtype == np.int16:
            return np.ascontiguousarray(arr)

        if getattr(self, "_source_was_8bit", False) and arr.dtype == np.uint16:
            arr_int = np.rint(arr.astype(np.float64) * (32767.0 / 65535.0)).astype(np.int64)
            return np.ascontiguousarray(np.clip(arr_int, 0, 32767).astype(np.int16))

        if arr.dtype == np.uint16:
            return np.ascontiguousarray(arr.astype(np.int16, copy=False))

        if np.issubdtype(arr.dtype, np.integer):
            arr_int = arr.astype(np.int64)
            if int(np.min(arr_int)) >= 0 and int(np.max(arr_int)) <= 255:
                arr_int = np.rint(arr_int * (32767.0 / 255.0)).astype(np.int64)
            return np.ascontiguousarray(np.clip(arr_int, -32768, 32767).astype(np.int16))

        arr_float = arr.astype(np.float64)
        lo = float(np.nanmin(arr_float))
        hi = float(np.nanmax(arr_float))
        if hi > lo:
            arr_float = (arr_float - lo) / (hi - lo) * 32767.0
        else:
            arr_float = np.zeros_like(arr_float)
        return np.ascontiguousarray(np.clip(arr_float, 0, 32767).astype(np.int16))

    def _image_for_annotation(self, image):
        """Return a 2-D image array for Step 2 annotation and saving.

        SDB data arrives from Step 1 already opened as 16-bit; keep that data
        as-is. If a user opens an 8-bit image directly, promote it to a real
        16-bit working copy so LIGHT saves never become 8-bit originals.
        """
        arr, source_was_8bit = self._coerce_image_for_annotation(image)
        self._source_was_8bit = source_was_8bit
        return arr

    @staticmethod
    def _coerce_image_for_annotation(image):
        arr = np.asarray(image)
        if arr.ndim != 2:
            raise ValueError("Step 2 expects a 2-D grayscale image.")
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
        source_was_8bit = arr.dtype == np.uint8
        if source_was_8bit:
            arr = np.rint(arr.astype(np.float64) * (65535.0 / 255.0)).astype(np.uint16)
        return np.ascontiguousarray(arr), source_was_8bit

    def _write_segmenter_log_file(self, content):
        log_dir = app_log_dir()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"step2_segmentation_{ts}.log"
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(content)
        return str(log_path)

    def _append_segmenter_log(self, text):
        """Append a short entry to the in-memory and (if present) UI segmenter log."""
        if not hasattr(self, "_segmenter_log_lines"):
            self._segmenter_log_lines = []
        ts = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        entry = f"{ts} - {text}"
        self._segmenter_log_lines.append(entry)
        try:
            with (app_log_dir() / "step2_activity.log").open("a", encoding="utf-8") as fh:
                fh.write(entry + "\n")
        except OSError:
            pass
        # If there's a text widget for logs, append there; otherwise print to stdout.
        try:
            widget = getattr(self, "segmenter_log_text", None)
            if widget is not None:
                try:
                    widget.insert("end", entry + "\n")
                    widget.see("end")
                except Exception:
                    print(entry)
            else:
                print(entry)
        except Exception:
            print(entry)

    @staticmethod
    def _boundary_traces_from_rows(data, image_shape, source_label="predicted boundaries"):
        """Convert model boundary rows into Step 2 trace dictionaries."""
        data = np.asarray(data, dtype=float)
        data = np.atleast_2d(data)
        if data.shape[0] < len(BOUNDARY_PRESETS):
            raise RuntimeError(
                f"{source_label} has {data.shape[0]} rows; expected at least {len(BOUNDARY_PRESETS)} rows."
            )

        height = int(image_shape[0])
        width = int(image_shape[1])
        max_x = min(width, data.shape[1])

        traces = {}
        order = []
        for row_idx, (name, _) in enumerate(BOUNDARY_PRESETS):
            y_row = np.rint(data[row_idx, :max_x]).astype(int)
            y_row = np.clip(y_row, 0, height - 1)
            points = [(x, int(y_row[x])) for x in range(max_x)]
            if len(points) < 2:
                continue
            pixels = _polyline_pixels(points)
            traces[name] = {
                "points": points,
                "pixels": pixels,
                "color": BOUNDARY_COLORS.get(name, "#ffb703"),
            }
            order.append(name)
        return traces, order

    def _batch_ai_button_enabled(self):
        if getattr(self, "batch_segmentation_panel", None) is not None:
            return False
        return not self._segmenter_running

    def _update_batch_ai_button_state(self):
        if not hasattr(self, "batch_ai_button"):
            return
        if self._batch_ai_button_enabled():
            self.batch_ai_button.state(["!disabled"])
        else:
            self.batch_ai_button.state(["disabled"])

    def _update_ai_button_states(self):
        """Update AI-related button states."""
        self._update_batch_ai_button_state()

    def _set_segmentation_running(
        self,
        running,
        status_message=None,
        *,
        progress_max=None,
        animate_progress=True,
        restore_boundary_progress=True,
    ):
        """Lock/unlock UI controls while AI segmentation is running.

        During segmentation (running=True):
          - Disables all AI buttons
          - Disables boundary listboxes to prevent selection changes
          - Disables fovea controls while keeping the marker visible
          - Starts animated progress bar showing activity
          - Disables line drawing

        After segmentation (running=False):
          - Re-enables controls
          - Stops progress animation
          - Updates AI button states
          - Re-enables line drawing if appropriate

        Args:
            running: Boolean indicating if segmentation is in progress.
            status_message: Optional status text to display to user.
        """
        self._segmenter_running = bool(running)
        if running:
            if hasattr(self, "batch_ai_button"):
                self.batch_ai_button.state(["disabled"])
        else:
            self._update_batch_ai_button_state()
        
        # Disable/enable boundary listboxes during segmentation
        listbox_state = "disabled" if running else "normal"
        if hasattr(self, "boundary_incomplete_listbox"):
            self.boundary_incomplete_listbox.configure(state=listbox_state)
        if hasattr(self, "boundary_completed_listbox"):
            self.boundary_completed_listbox.configure(state=listbox_state)
        
        # Disable/enable entire fovea frame during segmentation
        if running:
            if hasattr(self, "fovea_frame"):
                # Recursively disable all controls in the fovea frame
                def disable_widget(widget):
                    try:
                        widget.configure(state="disabled")
                    except Exception:
                        pass
                    for child in widget.winfo_children():
                        disable_widget(child)
                disable_widget(self.fovea_frame)
            # Also disable save/export buttons frame
            if hasattr(self, "saved_buttons_frame"):
                def disable_widget(widget):
                    try:
                        widget.configure(state="disabled")
                    except Exception:
                        pass
                    for child in widget.winfo_children():
                        disable_widget(child)
                disable_widget(self.saved_buttons_frame)
        else:
            if hasattr(self, "fovea_frame"):
                # Recursively enable all controls in the fovea frame
                def enable_widget(widget):
                    # Skip the LabelFrame itself, just enable children
                    for child in widget.winfo_children():
                        try:
                            child.configure(state="normal")
                        except Exception:
                            pass
                        enable_widget(child)
                enable_widget(self.fovea_frame)
                # Then properly update fovea controls state based on image availability.
                self._set_fovea_controls_enabled(self.image_data is not None)
            # Also re-enable save/export buttons
            if hasattr(self, "saved_buttons_frame"):
                def enable_widget(widget):
                    for child in widget.winfo_children():
                        try:
                            child.configure(state="normal")
                        except Exception:
                            pass
                        enable_widget(child)
                enable_widget(self.saved_buttons_frame)
        
        if running:
            if hasattr(self, "segmenter_progress"):
                self._cancel_progress_animation()
                maximum = progress_max if progress_max is not None else len(BOUNDARY_NAMES)
                self.segmenter_progress.configure(maximum=max(1, int(maximum)), mode="determinate")
                self.segmenter_progress["value"] = 0
                self._segmenter_progress_target = 0.0
                if animate_progress:
                    self._animate_progress_bar()
        else:
            # Stop animation and show actual boundary count
            self._cancel_progress_animation()
            if hasattr(self, "segmenter_progress") and restore_boundary_progress:
                self._update_boundary_progress_bar()
            # Update button states after segmentation finishes
            self._update_ai_button_states()
            if self.image_data is None:
                self._set_segmentation_frame_enabled(False)

        # Update canvas state to disable/enable drawing based on segmentation status
        self._sync_boundary_canvas_state()

        if status_message:
            self.status_var.set(status_message)

    def _cancel_progress_animation(self):
        job = getattr(self, "_progress_animation_job", None)
        if job is None:
            return
        try:
            self.after_cancel(job)
        except tk.TclError:
            pass
        self._progress_animation_job = None

    def _set_segmenter_progress_value(self, value, maximum=None, smooth=True):
        if not hasattr(self, "segmenter_progress"):
            return
        if maximum is not None:
            self.segmenter_progress.configure(maximum=max(1, int(maximum)), mode="determinate")
        max_value = float(self.segmenter_progress.cget("maximum") or 1)
        target = max(0.0, min(float(value), max_value))
        self._segmenter_progress_target = target
        if smooth and self._segmenter_running:
            if getattr(self, "_progress_animation_job", None) is None:
                self._animate_progress_bar()
            return
        self.segmenter_progress["value"] = target

    def _animate_progress_bar(self):
        """Gradually fill progress bar during segmentation to show activity."""
        if not self._segmenter_running or not hasattr(self, "segmenter_progress"):
            self._progress_animation_job = None
            return
        
        current = float(self.segmenter_progress["value"])
        maximum = float(self.segmenter_progress.cget("maximum") or len(BOUNDARY_NAMES))
        target = max(0.0, min(float(getattr(self, "_segmenter_progress_target", current)), maximum))

        if current < target:
            step = max(0.01, (target - current) * 0.18)
            self.segmenter_progress["value"] = min(current + step, target)

        self._progress_animation_job = self.after(80, self._animate_progress_bar)

    def _run_aidas_batch_segmentation(self, image_paths=None, manual_fovea_by_path=None):
        """Run AI_ForAIDAS predictions for multiple images and preview them in tabs."""
        if self._segmenter_running:
            messagebox.showinfo("Please wait", "Segmentation is already running.")
            return

        model_path = self.aidas_model_path
        if not os.path.isfile(model_path):
            messagebox.showerror("Missing model", f"AI_ForAIDAS boundary model not found:\n{model_path}")
            return

        paths = image_paths
        if paths is None:
            paths = filedialog.askopenfilenames(
                title="Select OCT images for AI_ForAIDAS batch segmentation",
                filetypes=SUPPORTED_IMAGE_FILETYPES,
            )
        if not paths:
            return

        image_paths = []
        seen = set()
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            dedupe_path = os.path.splitext(path)[0] if ext in {".hdr", ".img"} else path
            if os.path.basename(dedupe_path).lower() != LIGHT_SOURCE_BASENAME.lower():
                continue
            norm = os.path.normcase(os.path.abspath(dedupe_path))
            if norm in seen:
                continue
            seen.add(norm)
            image_paths.append(path)

        if not image_paths:
            messagebox.showwarning("Batch Step 2", "Select one or more Light.img files.")
            return

        manual_fovea_by_key = None
        if manual_fovea_by_path is not None:
            manual_fovea_by_key = {
                self._image_pair_key(path): (None if x is None else int(x))
                for path, x in manual_fovea_by_path.items()
            }

        provider_name = "auto"
        device_id = 0
        self._append_segmenter_log(
            f"Starting AI_ForAIDAS batch for {len(image_paths)} image(s); "
            f"model={model_path}; provider={provider_name}; DirectML adapter={device_id}"
        )
        self._set_segmentation_running(
            True,
            status_message=f"Running AI_ForAIDAS batch on {len(image_paths)} image(s)...",
            progress_max=len(image_paths),
            animate_progress=True,
        )

        worker = threading.Thread(
            target=self._run_aidas_batch_segmenter_worker,
            args=(image_paths, model_path, provider_name, device_id, manual_fovea_by_key),
            daemon=True,
        )
        worker.start()

    def _run_aidas_batch_segmenter_worker(
        self,
        image_paths,
        model_path,
        provider_name,
        device_id,
        manual_fovea_by_key=None,
    ):
        results = []
        failures = []
        log_lines = [
            "AIDaS Step 2 AI_ForAIDAS Batch Segmentation",
            f"Boundary model: {model_path}",
            f"Manual fovea lines: {'yes' if manual_fovea_by_key is not None else 'no'}",
            f"Requested provider: {provider_name}",
            f"DirectML adapter: {device_id}",
            f"Images: {len(image_paths)}",
            "",
        ]
        device = None
        batch_exception = None

        def report_status(index, path):
            total = len(image_paths)
            name = os.path.basename(path)
            self.after(0, lambda i=index, t=total: self._set_segmenter_progress_value(i - 1, t))
            self.after(0, lambda: self.status_var.set(f"AI_ForAIDAS batch {index}/{total}: {name}"))
            self.after(0, lambda: self._append_segmenter_log(f"Batch {index}/{total}: {path}"))

        def report_model_progress(index, total, path, fraction, stage):
            progress_value = (index - 1) + max(0.0, min(float(fraction), 1.0))
            label = str(stage or "running").replace("_", " ")
            name = os.path.basename(path)
            self.after(0, lambda v=progress_value, t=total: self._set_segmenter_progress_value(v, t))
            self.after(0, lambda: self.status_var.set(f"AI_ForAIDAS batch {index}/{total}: {name} - {label}"))

        try:
            worker_client = AIWorkerClient(
                self._aidas_worker_command(),
                model_path=model_path,
                provider_name=provider_name,
                device_id=device_id,
                env=self._aidas_worker_env(),
                popen_kwargs=self._hidden_subprocess_kwargs(),
                startup_progress_callback=lambda fraction, stage: report_model_progress(
                    1,
                    len(image_paths),
                    image_paths[0],
                    fraction,
                    stage,
                ),
            )
            with worker_client:
                startup = worker_client.startup_result or {}
                device = startup.get("device") or device
                log_lines.append(f"Worker: {worker_client.command_line}")
                log_lines.append(
                    f"Execution provider: {startup.get('execution_provider') or 'unknown'}"
                )
                if startup.get("fallback_reason"):
                    log_lines.append(f"CPU fallback: {startup['fallback_reason']}")
                log_lines.append("")

                for index, path in enumerate(image_paths, start=1):
                    report_status(index, path)
                    try:
                        image, template, _source_was_8bit = self._read_image_for_annotation(path)
                        model_input_uses_stored_y = template is not None
                        image_for_ai = np.array(image, copy=True)
                        if model_input_uses_stored_y:
                            image_for_ai = np.ascontiguousarray(np.flipud(image_for_ai))

                        prediction = worker_client.predict(
                            image_for_ai,
                            progress_callback=lambda fraction, stage, i=index, t=len(image_paths), p=path: (
                                report_model_progress(i, t, p, fraction, stage)
                            ),
                        )
                        if prediction.get("device"):
                            device = prediction["device"]
                        if prediction.get("fallback_reason"):
                            log_lines.append(
                                f"  CPU fallback: {prediction['fallback_reason']}"
                            )
                        fovea_x = None
                        manual_fovea = manual_fovea_by_key is not None
                        if manual_fovea_by_key is not None:
                            fovea_x = manual_fovea_by_key.get(self._image_pair_key(path))
                        if fovea_x is not None:
                            fovea_x = int(np.clip(int(fovea_x), 0, max(0, image.shape[1] - 1)))
                        boundaries = self._aidas_boundaries_from_model_to_display(
                            prediction["boundaries"],
                            image_for_ai.shape[0],
                            model_input_uses_stored_y,
                        )
                        traces, _order = self._boundary_traces_from_rows(
                            boundaries,
                            image.shape,
                            source_label=f"AI_ForAIDAS prediction for {os.path.basename(path)}",
                        )
                        results.append({
                            "input": path,
                            "boundaries": np.asarray(boundaries, dtype=np.float32),
                            "traces": traces,
                            "fovea_x": fovea_x,
                        })
                        log_lines.append(f"OK: {path}")
                        log_lines.append("  Preview generated in Step 2; no image or CSV was saved.")
                        if fovea_x is not None:
                            log_lines.append(f"  Fovea x (manual): {int(fovea_x)}")
                        elif manual_fovea:
                            log_lines.append("  Fovea x: skipped by user")
                    except Exception as exc:
                        failures.append({"input": path, "error": str(exc)})
                        log_lines.append(f"FAILED: {path}")
                        log_lines.append(f"  Error: {exc}")
                    finally:
                        self.after(0, lambda i=index, t=len(image_paths): self._set_segmenter_progress_value(i, t))
        except Exception as exc:
            batch_exception = str(exc)
            log_lines.append(f"FATAL: {batch_exception}")

        log_lines.append("")
        log_lines.append(f"Completed: {len(results)}")
        log_lines.append(f"Failed: {len(failures)}")
        if device:
            log_lines.append(f"Device: {device}")

        try:
            log_path = self._write_segmenter_log_file("\n".join(log_lines) + "\n")
        except Exception:
            log_path = None

        result = {
            "results": results,
            "failures": failures,
            "exception": batch_exception,
            "device": device,
            "log_path": log_path,
            "total": len(image_paths),
        }
        self.after(0, lambda: self._on_aidas_batch_segmenter_worker_done(result))

    def _on_aidas_batch_segmenter_worker_done(self, result):
        running_cleared = False
        try:
            log_path = result.get("log_path")
            if log_path:
                self._append_segmenter_log(f"Saved batch run log: {log_path}")

            completed = len(result.get("results") or [])
            failed = len(result.get("failures") or [])
            total = int(result.get("total") or (completed + failed) or 1)
            self._set_segmenter_progress_value(completed + failed, total, smooth=False)
            self._set_segmentation_running(False, restore_boundary_progress=False)
            running_cleared = True

            if result.get("exception") and completed == 0:
                message = result["exception"]
                if log_path:
                    message += f"\n\nLog saved to:\n{log_path}"
                messagebox.showerror("Batch segmentation failed", message)
                self.status_var.set("AI_ForAIDAS batch segmentation failed.")
                return

            device = result.get("device") or "unknown"
            self.status_var.set(
                f"AI_ForAIDAS batch complete on {device}: {completed} processed, {failed} failed."
            )
            self._append_segmenter_log(
                f"AI_ForAIDAS batch complete: {completed} processed, {failed} failed."
            )

            if completed:
                self._open_aidas_batch_results_viewer(result.get("results") or [])

            message_lines = [f"Processed {completed} image(s)."]
            shown_results = (result.get("results") or [])[:6]
            if shown_results:
                message_lines.append("")
                message_lines.append("Processed:")
                message_lines.extend(os.path.basename(item["input"]) for item in shown_results)
                if completed > len(shown_results):
                    message_lines.append(f"...and {completed - len(shown_results)} more.")
            if failed:
                message_lines.append("")
                message_lines.append(f"Failed: {failed}")
                for failure in (result.get("failures") or [])[:3]:
                    message_lines.append(f"{os.path.basename(failure['input'])}: {failure['error']}")
            if log_path:
                message_lines.append("")
                message_lines.append(f"Log saved to:\n{log_path}")

            if failed:
                messagebox.showwarning("Batch segmentation complete with errors", "\n".join(message_lines))
            # else:
            #     messagebox.showinfo("Batch segmentation complete", "\n".join(message_lines))
        finally:
            if not running_cleared:
                self._set_segmentation_running(False)

    def _open_aidas_batch_results_viewer(self, results):
        viewable_results = [
            item for item in results
            if item.get("input") and (item.get("traces") or item.get("boundaries") is not None)
        ]
        if not viewable_results:
            return

        if self._active_batch_result_tab is None:
            self._single_editor_state = self._capture_current_editor_state()

        previous_notebook = getattr(self, "batch_results_notebook", None)
        if previous_notebook is not None:
            previous_notebook.destroy()

        notebook = ttk.Notebook(self.canvas_area)
        self.batch_results_notebook = notebook
        self._batch_result_canvases = []
        self._batch_result_tab_canvases = {}
        self._batch_result_states = {}
        self._active_batch_result_tab = None
        notebook.bind("<Button-1>", self._on_batch_result_notebook_click, add="+")
        notebook.bind("<Motion>", self._on_batch_result_notebook_motion, add="+")
        notebook.bind("<Leave>", lambda event: event.widget.configure(cursor=""), add="+")
        notebook.bind("<<NotebookTabChanged>>", self._on_batch_result_tab_changed, add="+")

        for index, item in enumerate(viewable_results, start=1):
            frame = ttk.Frame(notebook)
            input_path = item["input"]
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            tab_text = self._batch_result_tab_text(input_name or f"Result {index}")
            notebook.add(frame, text=tab_text)

            info_var = tk.StringVar(value=input_path)
            ttk.Label(frame, textvariable=info_var, anchor="w", padding=4).pack(fill="x")

            canvas = ImageCanvas(
                frame,
                on_mouse_move=self._on_mouse_moved,
                on_line_change=self._on_active_line_changed,
                on_vertical_line_change=self._on_vertical_line_changed,
            )
            canvas.enable_roi(False)
            canvas.enable_line(False)
            canvas.enable_vertical_line(False)
            canvas.pack(fill="both", expand=True)
            self._batch_result_canvases.append(canvas)
            self._batch_result_tab_canvases[str(frame)] = canvas

            try:
                data, _template, _source_was_8bit = self._read_image_for_annotation(input_path)
                canvas.set_image(data)
                traces = item.get("traces")
                if traces is None:
                    traces, _order = self._boundary_traces_from_rows(
                        item["boundaries"],
                        data.shape,
                        source_label=f"AI_ForAIDAS prediction for {os.path.basename(input_path)}",
                    )
                traces = self._copy_trace_dict(traces)
                order = [name for name in BOUNDARY_NAMES if name in traces]
                overlays = []
                for name in BOUNDARY_NAMES:
                    trace = traces.get(name)
                    if trace:
                        overlays.append({
                            "points": trace["points"],
                            "color": trace.get("color"),
                            "label": name,
                        })
                fovea_x = item.get("fovea_x")
                if fovea_x is not None:
                    fovea_trace = self._vertical_line_trace(int(fovea_x), data.shape[0])
                    traces[FOVEA_BOUNDARY_NAME] = fovea_trace
                    if FOVEA_BOUNDARY_NAME not in order:
                        order.append(FOVEA_BOUNDARY_NAME)
                    overlays.append({
                        "points": fovea_trace["points"],
                        "color": fovea_trace.get("color"),
                        "label": FOVEA_BOUNDARY_NAME,
                    })
                canvas.set_line_overlays(overlays)
                self._batch_result_states[str(frame)] = {
                    "input": input_path,
                    "image": data,
                    "traces": traces,
                    "order": order,
                    "fovea_x": None if fovea_x is None else int(fovea_x),
                    "template": _template,
                    "source_was_8bit": _source_was_8bit,
                    "canvas": canvas,
                }
                canvas.fit_to_window()
            except (OSError, ValueError, RuntimeError) as exc:
                info_var.set(f"Could not load {input_path}: {exc}")

        self._show_batch_results_canvas()
        if notebook.tabs():
            first_tab = notebook.nametowidget(notebook.tabs()[0])
            notebook.select(first_tab)
            self._activate_batch_result_tab(first_tab)
        self.status_var.set(f"AI batch results opened: {len(viewable_results)} image(s). Select a tab to edit its boundaries.")
        self.after(100, self._fit_batch_result_canvases)

    @staticmethod
    def _batch_result_tab_text(title):
        return f"{str(title)[:24]}\tx"

    def _sync_active_batch_result_state(self):
        tab_key = getattr(self, "_active_batch_result_tab", None)
        if not tab_key:
            return
        state = self._batch_result_states.get(tab_key)
        if not state:
            return
        state["input"] = self.current_file
        state["image"] = self.image_data
        if (
            self.fovea_x is not None
            and self.image_data is not None
            and FOVEA_BOUNDARY_NAME in self.boundary_traces
        ):
            self.boundary_traces[FOVEA_BOUNDARY_NAME] = self._vertical_line_trace(
                int(self.fovea_x),
                self.image_data.shape[0],
            )
            if FOVEA_BOUNDARY_NAME not in self.boundary_order:
                self.boundary_order.append(FOVEA_BOUNDARY_NAME)
        state["traces"] = self.boundary_traces
        state["order"] = self.boundary_order
        state["fovea_x"] = self.fovea_x if FOVEA_BOUNDARY_NAME in self.boundary_traces else None

    def _activate_batch_result_tab(self, tab):
        if tab is None:
            return
        self._sync_active_batch_result_state()
        tab_key = str(tab)
        state = self._batch_result_states.get(tab_key)
        if not state:
            return
        canvas = state.get("canvas") or self._batch_result_tab_canvases.get(tab_key)
        self._active_batch_result_tab = tab_key
        self._load_editor_state(
            state,
            canvas,
            status_message=f"Editing batch result: {os.path.basename(state.get('input') or '')}",
        )

    def _on_batch_result_tab_changed(self, event):
        notebook = event.widget
        selected = notebook.select()
        if not selected:
            return
        try:
            self._activate_batch_result_tab(notebook.nametowidget(selected))
        except tk.TclError:
            return

    def _batch_result_close_tab_at(self, notebook, x, y):
        try:
            index = notebook.index(f"@{x},{y}")
        except tk.TclError:
            return None

        tab_bounds = self._batch_result_tab_bounds(notebook, index, y)
        if tab_bounds is None:
            return None
        left, right = tab_bounds

        close_width = min(15, max(1, right - left + 1))
        if right - close_width <= x <= right:
            return notebook.nametowidget(notebook.tabs()[index])
        return None

    @staticmethod
    def _batch_result_tab_bounds(notebook, index, y):
        try:
            x0, _y0, width, _height = notebook.bbox(index)
            if width > 0:
                return x0, x0 + width
        except tk.TclError:
            pass

        first_x = None
        last_x = None
        probe_y = max(1, int(y))
        for probe_x in range(max(1, notebook.winfo_width())):
            try:
                probe_index = notebook.index(f"@{probe_x},{probe_y}")
            except tk.TclError:
                if first_x is not None:
                    break
                continue
            if probe_index != index:
                if first_x is not None:
                    break
                continue
            if first_x is None:
                first_x = probe_x
            last_x = probe_x

        if first_x is None or last_x is None:
            return None
        return first_x, last_x

    def _on_batch_result_notebook_click(self, event):
        notebook = event.widget
        tab = self._batch_result_close_tab_at(notebook, event.x, event.y)
        if tab is None:
            return None
        canvas = self._batch_result_tab_canvases.get(str(tab))
        if canvas is None:
            return None
        self._close_batch_result_tab(notebook, tab, canvas)
        return "break"

    def _on_batch_result_notebook_motion(self, event):
        notebook = event.widget
        cursor = "hand2" if self._batch_result_close_tab_at(notebook, event.x, event.y) is not None else ""
        try:
            notebook.configure(cursor=cursor)
        except tk.TclError:
            pass

    def _fit_batch_result_canvases(self):
        for canvas in list(getattr(self, "_batch_result_canvases", [])):
            try:
                if canvas.winfo_exists():
                    canvas.fit_to_window()
            except tk.TclError:
                pass

    def _close_batch_result_tab(self, notebook, tab, canvas):
        if tab is None:
            return
        tab_key = str(tab)
        closing_active = tab_key == getattr(self, "_active_batch_result_tab", None)
        if closing_active:
            self._sync_active_batch_result_state()

        if not self._confirm_close_batch_result_tab(tab_key):
            return

        self._batch_result_tab_canvases.pop(tab_key, None)
        self._batch_result_states.pop(tab_key, None)
        try:
            if canvas in self._batch_result_canvases:
                self._batch_result_canvases.remove(canvas)
        except ValueError:
            pass

        try:
            notebook.forget(tab)
        except tk.TclError:
            pass
        tab.destroy()

        remaining_tabs = len(notebook.tabs())
        if remaining_tabs:
            self.image_info_var.set(f"AI batch results | {remaining_tabs} image(s)")
            if closing_active:
                try:
                    selected = notebook.select()
                    if selected:
                        self._activate_batch_result_tab(notebook.nametowidget(selected))
                except tk.TclError:
                    pass
            return

        if not remaining_tabs:
            notebook.destroy()
            if getattr(self, "batch_results_notebook", None) is notebook:
                self.batch_results_notebook = None
            self._batch_result_canvases = []
            self._batch_result_tab_canvases = {}
            self._batch_result_states = {}
            self._active_batch_result_tab = None
            self._single_editor_state = None
            self._show_single_image_canvas()
            self._clear_image_display("All batch result images saved or removed. No image is loaded.")

    def _confirm_close_batch_result_tab(self, tab_key):
        state = self._batch_result_states.get(tab_key)
        if not state:
            return True

        name = os.path.basename(state.get("input") or "this tab")
        answer = messagebox.askyesnocancel(
            "Close result tab",
            f"Save MARKED image for {name} before closing this tab?",
        )
        if answer is None:
            return False
        if not answer:
            return True

        try:
            out_base = self._save_batch_result_state(tab_key, save_orientation_pair=True)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Save error", f"Could not save {name}:\n{exc}")
            return False

        if out_base:
            self.status_var.set("Saved Light and Light_MARKED in nasal and temporal folders.")
        return True

    def _aidas_should_use_stored_analyze_y(self):
        """Return True when AI_ForAIDAS should mimic app.py's Analyze input orientation."""
        if self.image_data is None:
            return False
        if self._input_analyze_template is not None:
            return True
        current_file = str(self.current_file or "")
        return os.path.splitext(current_file)[1].lower() in {".hdr", ".img"}

    @staticmethod
    def _aidas_boundaries_from_model_to_display(boundaries, image_height, model_input_uses_stored_y):
        boundaries = np.asarray(boundaries, dtype=np.float32)
        if not model_input_uses_stored_y:
            return boundaries

        height = int(image_height)
        if height <= 0:
            return boundaries

        # AIDaS read_analyze() flips Analyze slices for display, while the
        # standalone AI_ForAIDAS app runs on the stored .img rows directly.
        # Flip rows back into Step 2 display coordinates and reverse boundary
        # order so the imported traces remain top-to-bottom on the displayed image.
        converted = (height - 1) - boundaries[::-1]
        return np.clip(converted, 0, height - 1).astype(np.float32, copy=False)

    # ═══════════════════════════════════════════════════════════════════════
    #  Export boundary rows as CSV
    # ═══════════════════════════════════════════════════════════════════════
    def _boundary_row_for_export(self, trace, image_width):
        """Convert a boundary polyline into a single row of Y-coordinates.

        Exports one row per image column (X=0 to X=image_width-1), with each value
        being the Y-coordinate where the boundary crosses that column. Uses linear
        interpolation to fill in columns not explicitly sampled.

        Args:
            trace: Boundary dict with 'pixels' key (list of (x, y) tuples).
            image_width: Width of image (number of columns).

        Returns:
            List of Y-values (ints, or empty strings if no pixels), indexed by X.
        """
        pixels = trace.get("pixels") or []
        if not pixels or image_width <= 0:
            return [""] * image_width

        ordered_samples = {}
        for x, y in pixels:
            x_int = int(x)
            if x_int not in ordered_samples:
                ordered_samples[x_int] = int(y)

        xs = np.array(sorted(ordered_samples.keys()), dtype=np.float64)
        ys = np.array([ordered_samples[int(x)] for x in xs], dtype=np.float64)

        if xs.size == 1:
            return [int(ys[0])] * image_width

        x_grid = np.arange(image_width, dtype=np.float64)
        row = np.interp(x_grid, xs, ys, left=ys[0], right=ys[-1])
        return np.rint(row).astype(int).tolist()

    def _export_csv(self):
        """Export all boundary traces as a single CSV file.

        CSV format:
          - Header row: boundary names
          - Data rows: one per image column (X), with Y-coordinate values
          - Y values are interpolated from the polyline trace

        The export includes all boundaries in boundary_order, with empty columns
        for boundaries that have no pixels at that X coordinate.

        File is saved next to the current image with suffix _step2_boundaries.csv.
        """
        if not self.boundary_traces:
            messagebox.showinfo("Nothing to export", "Trace at least one boundary first.")
            return

        if self.image_data is None:
            messagebox.showwarning("No image", "Load an image before exporting boundary rows.")
            return

        default_name = self._default_export_name()
        path = filedialog.asksaveasfilename(
            title="Export boundary coordinates",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        try:
            image_width = int(self.image_data.shape[1])
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                for name in BOUNDARY_NAMES:
                    trace = self.boundary_traces.get(name)
                    if not trace:
                        writer.writerow([name] + [""] * image_width)
                        continue
                    writer.writerow([name] + self._boundary_row_for_export(trace, image_width))
        except OSError as exc:
            messagebox.showerror("Export error", str(exc))
            return

        self.status_var.set(f"Exported boundary rows → {path}")
        messagebox.showinfo(
            "Exported",
            "Saved six boundary rows to:\n"
            f"{path}\n\nRows follow the preset order: {', '.join(BOUNDARY_NAMES)}",
        )

    def _default_export_name(self):
        if self.current_file:
            stem = os.path.splitext(os.path.basename(self.current_file))[0]
        else:
            stem = "step2"
        return f"{stem}{TRACE_EXPORT_SUFFIX}"
