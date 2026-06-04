"""Step 2 — OCT boundary annotation with AI-assisted and manual drawing.

This module implements the Step 2 GUI panel for annotating retinal boundary
lines over OCT images. It supports manual ImageJ-style polyline drawing,
AI segmentation import for automated predictions, and saving boundary
coordinates.

Core Functionality:
  • Manual Annotation: Click-to-place polyline drawing for 6 preset retinal
    boundaries (RPE, ELM, ONL-OPL, INL-IPL, GCL-RNFL, RNFL-Vitreous) with
    vertex undo and visual feedback.
  • AI Segmentation: Run neural network predictions via oct-segmenter tool,
    with optional auto-preprocessing (center-crop to model input size).
    Predictions are automatically imported as boundary traces.
  • Boundary Workflow: Tracks annotation progress with separate incomplete/
    completed boundary lists; auto-advances to next boundary after finish.
  • Foveal Center Line: Dedicated vertical line mode for placing/adjusting
    the foveal center X-coordinate with nudge buttons and keyboard entry.
  • Export: CSV export of all boundary row coordinates (one row per boundary).
  • MARKED Images: Generate Light_MARKED and Dark_MARKED Analyze volumes
    (8-bit) with boundary pixels marked at specific intensity values per
    ImageJ macro conventions. Auto-scales boundaries if output size differs
    from input.

"""

import csv
import datetime
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from aidas.image_canvas import ImageCanvas, RESAMPLE_NEAREST
from aidas.utils.io_utils import read_analyze, read_tiff, write_analyze, scale_image
from aidas.utils.ui_utils import SidebarStepFrame, apply_app_icon_to


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
DARK_FIRST_SLICE_EXTRA_MARK_VALUES = {
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
DARK_MARKED_BASENAME = "Dark_MARKED"
LIGHT_PREPROCESSED_BASENAME = "Light"
DARK_PREPROCESSED_BASENAME = "Dark"
AI_BACKEND_OCT_SEGMENTER = "oct_segmenter"
AI_BACKEND_AIDAS = "ai_for_aidas"
AI_BACKEND_LABELS = {
    AI_BACKEND_OCT_SEGMENTER: "OCT Segmenter (old, Keras-based)",
    AI_BACKEND_AIDAS: "AI_ForAIDAS (New, PyTorch-based)",
}
AI_BACKEND_BY_LABEL = {label: key for key, label in AI_BACKEND_LABELS.items()}
AI_DEVICE_OPTIONS = ("auto", "cpu", "cuda")
IMG_DEFAULT_DIR = os.path.expanduser("~/Desktop")
SUPPORTED_IMAGE_EXTENSIONS = (".img",)
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
    # original dtype is preserved for 16-bit LIGHT/DARK volumes.
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


class _FoveaLinePicker(tk.Toplevel):
    """Modal image viewer for choosing one foveal center x-coordinate."""

    DISPLAY_W = 1100
    DISPLAY_H = 320

    def __init__(self, parent, image, *, title=None, initial_x=None):
        super().__init__(parent)
        self.title(title or "Mark Foveal Center")
        self.resizable(False, False)
        self.transient(parent)
        apply_app_icon_to(self)

        arr = np.asarray(image)
        if arr.ndim != 2:
            raise ValueError("Fovea picker expects a 2-D grayscale image.")
        height, width = arr.shape
        self._orig_w = int(width)
        self._orig_h = int(height)
        self.result = None
        self.cancelled = False

        scale = min(self.DISPLAY_W / max(1, width), self.DISPLAY_H / max(1, height))
        self._scale = max(scale, 1e-6)
        self._display_w = max(1, int(round(width * self._scale)))
        self._display_h = max(1, int(round(height * self._scale)))

        display = arr.astype(np.float32, copy=False)
        lo = float(np.nanmin(display))
        hi = float(np.nanmax(display))
        if hi <= lo:
            u8 = np.zeros(display.shape, dtype=np.uint8)
        else:
            u8 = ((display - lo) / (hi - lo + 1e-8) * 255).clip(0, 255).astype(np.uint8)
        try:
            resample_bilinear = Image.Resampling.BILINEAR
        except AttributeError:
            resample_bilinear = Image.BILINEAR
        pil = Image.fromarray(u8, mode="L").resize((self._display_w, self._display_h), resample_bilinear)
        self._tk_img = ImageTk.PhotoImage(pil)

        self._canvas = tk.Canvas(
            self,
            width=self._display_w,
            height=self._display_h,
            cursor="sb_h_double_arrow",
            highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)
        self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        start_x = self._orig_w // 2 if initial_x is None else int(initial_x)
        self._line_x = int(np.clip(round(start_x * self._scale), 0, max(0, self._display_w - 1)))
        self._canvas.create_line(
            self._line_x,
            0,
            self._line_x,
            self._display_h,
            fill="#ffd500",
            width=2,
            tags="vline",
        )

        self._label_var = tk.StringVar()
        ttk.Label(self, textvariable=self._label_var, font=("Consolas", 10)).pack(pady=4)

        buttons = ttk.Frame(self)
        buttons.pack(pady=(0, 8))
        ttk.Button(buttons, text="Confirm", command=self._confirm).pack(side="left", padx=6)
        ttk.Button(buttons, text="Skip", command=self._skip).pack(side="left", padx=6)
        ttk.Button(buttons, text="Cancel Folder", command=self._cancel).pack(side="left", padx=6)

        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_drag)
        self._canvas.bind("<Left>", lambda _event: self._move_line(self._line_x - 1))
        self._canvas.bind("<Right>", lambda _event: self._move_line(self._line_x + 1))
        self._canvas.bind("<Shift-Left>", lambda _event: self._move_line(self._line_x - 10))
        self._canvas.bind("<Shift-Right>", lambda _event: self._move_line(self._line_x + 10))
        self.bind("<Return>", lambda _event: self._confirm())
        self.bind("<Escape>", lambda _event: self._skip())
        self.protocol("WM_DELETE_WINDOW", self._skip)

        self._move_line(self._line_x)
        self.grab_set()
        self._canvas.focus_set()

    def _move_line(self, x):
        x = int(np.clip(int(x), 0, max(0, self._display_w - 1)))
        self._line_x = x
        self._canvas.coords("vline", x, 0, x, self._display_h)
        orig_col = int(np.clip(round(x / self._scale), 0, max(0, self._orig_w - 1)))
        self._label_var.set(f"Fovea center column: {orig_col}")

    def _on_press(self, event):
        self._move_line(event.x)

    def _on_drag(self, event):
        self._move_line(event.x)

    def _confirm(self):
        self.result = int(np.clip(round(self._line_x / self._scale), 0, max(0, self._orig_w - 1)))
        self.destroy()

    def _skip(self):
        self.result = None
        self.destroy()

    def _cancel(self):
        self.cancelled = True
        self.destroy()


class Step2BatchSegmentationSelectionPanel(ttk.Frame):
    """Embedded panel for selecting folders to run through Step 2 AI segmentation."""

    TABLE_COLUMNS = (
        ("select", "", 42, "center"),
        ("folder", "Folder", 560, "w"),
        ("status", "Status", 380, "w"),
        ("images", "Images", 92, "center"),
    )
    COLUMN_MIN_WIDTHS = {
        "select": 42,
        "folder": 320,
        "status": 96,
        "images": 64,
    }
    COLUMN_MAX_WIDTHS = {
        "images": 92,
    }
    HEADER_HEIGHT = 34
    ROW_HEIGHT = 34
    GRID_COLOR = "#d1d5db"
    HEADER_BG = "#f3f4f6"
    ROW_BG = "#ffffff"
    ALT_ROW_BG = "#f9fafb"
    TEXT_COLOR = "#111827"
    MUTED_COLOR = "#6b7280"

    def __init__(self, step_frame, parent, root_dir):
        super().__init__(parent)
        self.step_frame = step_frame
        self.root_dir = os.path.abspath(root_dir)
        self.rows = []
        self.header_cells = {}
        self.empty_message_cell = None
        self.table_filler_cells = []
        self._table_resize_after_id = None
        self._table_font = tkfont.nametofont("TkDefaultFont")
        self._header_font = self._table_font.copy()
        self._header_font.configure(weight="bold")
        self.select_all_ready_var = tk.BooleanVar(value=False)
        self.column_widths = {key: width for key, _title, width, _anchor in self.TABLE_COLUMNS}
        self.table_width = sum(self.column_widths.values()) + len(self.TABLE_COLUMNS)

        self._build_ui()
        self._start_scan()

    def _build_ui(self):
        wrapper = ttk.Frame(self, padding=12)
        wrapper.pack(fill="both", expand=True)

        ttk.Label(wrapper, text="Batch Segmentation", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            wrapper,
            text=(
                "AIDaS will search the selected folder and subfolders for Light.img and Dark.img."
                "Folders with existing MARKED segmentation are shown as already segmented and skipped."
            ),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        self.table_outer = ttk.Frame(wrapper)
        self.table_outer.pack(fill="both", expand=True)
        self.table_canvas = tk.Canvas(
            self.table_outer,
            bg=self.ROW_BG,
            highlightthickness=1,
            highlightbackground=self.GRID_COLOR,
        )
        self.table_yscroll = ttk.Scrollbar(self.table_outer, orient="vertical", command=self.table_canvas.yview)
        self.table_xscroll = ttk.Scrollbar(self.table_outer, orient="horizontal", command=self._table_xview)

        self.table_inner = tk.Frame(self.table_canvas, bg=self.GRID_COLOR)
        self.table_window = self.table_canvas.create_window((0, 0), window=self.table_inner, anchor="nw")

        self.table_canvas.configure(yscrollcommand=self.table_yscroll.set, xscrollcommand=self.table_xscroll.set)

        self.table_canvas.grid(row=0, column=0, sticky="nsew")
        self.table_yscroll.grid(row=0, column=1, sticky="ns")
        self.table_outer.rowconfigure(0, weight=1)
        self.table_outer.columnconfigure(0, weight=1)

        self._build_table_header()
        self.table_inner.bind("<Configure>", lambda _event: self._refresh_scrollregion())
        self.table_canvas.bind("<Configure>", self._on_table_canvas_configure, add="+")
        self.table_canvas.bind("<MouseWheel>", self._on_table_mousewheel, add="+")

        top = ttk.Frame(wrapper)
        top.pack(fill="x", pady=(0, 8))
        self.summary_var = tk.StringVar(value=f"Scanning: {self.root_dir}")
        ttk.Label(top, textvariable=self.summary_var, wraplength=760, justify="left").pack(
            side="left",
            fill="x",
            expand=True,
        )

        run_box = ttk.Frame(wrapper)
        run_box.pack(fill="x", pady=(10, 0))
        ttk.Button(run_box, text="Start Segmentation", command=self._run_selected).pack(side="left")
        ttk.Button(run_box, text="Exit", command=self._cancel).pack(side="right")
        self.after_idle(self._fit_table_to_window)

    def _build_table_header(self):
        for col, (key, title, _width, anchor) in enumerate(self.TABLE_COLUMNS):
            cell = self._make_table_cell(
                self.table_inner,
                row=0,
                col=col,
                width=self._column_width(key),
                height=self.HEADER_HEIGHT,
                bg=self.HEADER_BG,
            )
            self.header_cells[key] = cell
            if key == "select":
                self.select_all_ready_check = ttk.Checkbutton(
                    cell,
                    variable=self.select_all_ready_var,
                    command=self._toggle_all_ready,
                )
                self.select_all_ready_check.pack(anchor="center", expand=True)
                self.select_all_ready_check.state(["disabled"])
                continue

            label = tk.Label(
                cell,
                text=title,
                anchor=anchor,
                bg=self.HEADER_BG,
                fg=self.TEXT_COLOR,
                font=self._header_font,
                padx=8,
            )
            label.pack(fill="both", expand=True)

    def _make_table_cell(self, parent, row, col, width, height, bg):
        cell = tk.Frame(parent, bg=bg, width=width, height=height)
        cell.grid(row=row, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))
        cell.grid_propagate(False)
        return cell

    def _column_width(self, key):
        return int(self.column_widths.get(key, self.COLUMN_MIN_WIDTHS.get(key, 80)))

    def _text_width(self, text, *, header=False, padding=18):
        font = self._header_font if header else self._table_font
        return int(font.measure(str(text or ""))) + int(padding)

    def _fit_content_column_width(self, key, title, values):
        measured = [self._text_width(title, header=True)]
        measured.extend(self._text_width(value) for value in values)
        width = max([self.COLUMN_MIN_WIDTHS.get(key, 80), *measured])
        max_width = self.COLUMN_MAX_WIDTHS.get(key)
        if max_width is not None:
            width = min(width, max_width)
        return int(width)

    def _on_table_canvas_configure(self, _event=None):
        if self._table_resize_after_id is not None:
            try:
                self.after_cancel(self._table_resize_after_id)
            except tk.TclError:
                pass
        self._table_resize_after_id = self.after_idle(self._fit_table_to_window)

    def _fit_table_to_window(self):
        self._table_resize_after_id = None
        try:
            available = max(1, int(self.table_canvas.winfo_width()) - 2)
            visible_height = max(1, int(self.table_canvas.winfo_height()) - 2)
        except tk.TclError:
            return

        gap_width = len(self.TABLE_COLUMNS)
        select_width = self.COLUMN_MIN_WIDTHS["select"]
        images_width = self._fit_content_column_width(
            "images",
            "Images",
            [len(row.get("image_paths") or []) for row in self.rows],
        )
        status_width = self._fit_content_column_width(
            "status",
            "Status",
            [row.get("status", "") for row in self.rows],
        )

        fixed_width = select_width + status_width + images_width + gap_width
        min_table_width = fixed_width + self.COLUMN_MIN_WIDTHS["folder"]
        target_width = max(min_table_width, available)
        folder_width = max(self.COLUMN_MIN_WIDTHS["folder"], target_width - fixed_width)

        self.column_widths = {
            "select": select_width,
            "folder": folder_width,
            "status": status_width,
            "images": images_width,
        }
        self.table_width = sum(self.column_widths.values()) + gap_width
        self._set_horizontal_scrollbar_visible(self.table_width > available + 1)
        self._apply_table_size(visible_height)

    def _set_vertical_scrollbar_visible(self, visible):
        try:
            managed = bool(self.table_yscroll.winfo_manager())
            changed = False
            if visible and not managed:
                self.table_yscroll.grid(row=0, column=1, sticky="ns")
                changed = True
            elif not visible and managed:
                self.table_yscroll.grid_remove()
                self.table_canvas.yview_moveto(0)
                changed = True
            if changed and self._table_resize_after_id is None:
                self._table_resize_after_id = self.after_idle(self._fit_table_to_window)
        except tk.TclError:
            pass

    def _set_horizontal_scrollbar_visible(self, visible):
        try:
            managed = bool(self.table_xscroll.winfo_manager())
            if visible and not managed:
                self.table_xscroll.grid(row=1, column=0, sticky="ew")
            elif not visible and managed:
                self.table_xscroll.grid_remove()
        except tk.TclError:
            pass

    def _apply_table_size(self, visible_height=None):
        try:
            if visible_height is None:
                visible_height = max(1, int(self.table_canvas.winfo_height()) - 2)
        except tk.TclError:
            return

        for key, cell in self.header_cells.items():
            try:
                cell.configure(width=self._column_width(key))
            except tk.TclError:
                pass
        for row in self.rows:
            for key, cell in (row.get("cells") or {}).items():
                try:
                    cell.configure(width=self._column_width(key))
                except tk.TclError:
                    pass
        if self.empty_message_cell is not None:
            try:
                self.empty_message_cell.configure(width=self.table_width)
            except tk.TclError:
                pass
        content_overflows = self._update_table_filler(visible_height)
        self._set_vertical_scrollbar_visible(content_overflows)
        try:
            table_height = max(visible_height, self._table_content_height())
            self.table_canvas.itemconfigure(
                self.table_window,
                width=self.table_width,
                height=table_height,
            )
        except tk.TclError:
            return
        self._refresh_scrollregion()

    def _table_content_height(self):
        try:
            self.table_inner.update_idletasks()
            return int(self.table_inner.grid_bbox()[3])
        except tk.TclError:
            return 0

    def _visible_body_row_count(self):
        if self.rows:
            return len(self.rows)
        return 1 if self.empty_message_cell is not None else 0

    def _update_table_filler(self, viewport_height):
        for cell in self.table_filler_cells:
            try:
                cell.destroy()
            except tk.TclError:
                pass
        self.table_filler_cells = []

        content_rows = self._visible_body_row_count()
        used_height = self._table_content_height()
        filler_height = max(0, int(viewport_height) - used_height - 1)
        content_overflows = used_height > int(viewport_height) + 1
        if filler_height <= 0:
            return content_overflows

        filler_row = content_rows + 1
        for col, (key, _title, _width, _anchor) in enumerate(self.TABLE_COLUMNS):
            cell = self._make_table_cell(
                self.table_inner,
                row=filler_row,
                col=col,
                width=self._column_width(key),
                height=filler_height,
                bg=self.ROW_BG,
            )
            self.table_filler_cells.append(cell)
        return content_overflows

    def _refresh_scrollregion(self):
        try:
            self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))
        except tk.TclError:
            pass

    def _table_xview(self, *args):
        self.table_canvas.xview(*args)

    def _on_table_mousewheel(self, event):
        if event.state & 0x0001:
            self._table_xview("scroll", -1 * int(event.delta / 120), "units")
        else:
            self.table_canvas.yview_scroll(-1 * int(event.delta / 120), "units")
        return "break"

    def _start_scan(self):
        self.step_frame.status_var.set(f"Scanning subfolders under {self.root_dir}...")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            rows, scanned, skipped = self.step_frame._scan_step2_batch_segmentation_folders(self.root_dir)
        except Exception as exc:
            self.after(0, lambda exc=exc: self._scan_failed(exc))
            return
        self.after(0, lambda: self._scan_done(rows, scanned, skipped))

    def _scan_failed(self, exc):
        self.summary_var.set(f"Scan failed: {exc}")
        self.step_frame.status_var.set("Batch segmentation scan failed.")
        messagebox.showerror("Batch Step 2", f"Could not scan folders.\n{exc}", parent=self)

    def _scan_done(self, rows, scanned, skipped):
        self._clear_table_body()
        self.rows = rows
        self.empty_message_cell = None
        for idx, row in enumerate(rows):
            self._add_row(idx, row)

        ready = sum(1 for row in rows if not row["locked"])
        already_segmented = sum(1 for row in rows if row["locked"])
        ready_images = sum(len(row.get("image_paths") or []) for row in rows if not row["locked"])
        self.summary_var.set(
            f"Scanned {scanned} folder(s). Found {ready} ready folder(s) with {ready_images} image(s) to segment, "
            f"{already_segmented} folder(s) already segmented. {skipped} folder(s) did not contain Light.img or Dark.img."
        )
        if not rows:
            self._add_empty_message()
        self._refresh_select_all_ready_checkbox()
        self._fit_table_to_window()
        self.step_frame.status_var.set("Batch segmentation scan complete. Confirm folders to process.")

    def _clear_table_body(self):
        for child in self.table_inner.winfo_children():
            try:
                row = int(child.grid_info().get("row", 0))
            except (TypeError, ValueError):
                row = 0
            if row > 0:
                child.destroy()
        self.table_filler_cells = []

    def _add_empty_message(self):
        cell = self._make_table_cell(
            self.table_inner,
            row=1,
            col=0,
            width=self.table_width,
            height=self.ROW_HEIGHT,
            bg=self.ROW_BG,
        )
        cell.grid(columnspan=len(self.TABLE_COLUMNS))
        self.empty_message_cell = cell
        tk.Label(
            cell,
            text="No folders with Light.img or Dark.img were found.",
            anchor="w",
            bg=self.ROW_BG,
            fg=self.MUTED_COLOR,
            padx=10,
        ).pack(fill="both", expand=True)

    def _add_row(self, idx, row):
        row["var"] = tk.BooleanVar(value=bool(row.get("include")))
        row["widgets"] = {}
        row["cells"] = {}
        bg = self.ROW_BG if idx % 2 == 0 else self.ALT_ROW_BG

        try:
            folder_text = os.path.relpath(row["folder"], self.root_dir)
            if folder_text == ".":
                folder_text = self.root_dir
        except ValueError:
            folder_text = row["folder"]

        values = {
            "folder": folder_text,
            "status": row.get("status", ""),
            "images": str(len(row.get("image_paths") or [])),
        }

        for col, (key, _title, width, anchor) in enumerate(self.TABLE_COLUMNS):
            cell = self._make_table_cell(
                self.table_inner,
                row=idx + 1,
                col=col,
                width=self._column_width(key),
                height=self.ROW_HEIGHT,
                bg=bg,
            )
            row["cells"][key] = cell
            if key == "select":
                checkbutton = ttk.Checkbutton(
                    cell,
                    variable=row["var"],
                    command=lambda item=row: self._on_row_checkbutton_toggled(item),
                )
                checkbutton.pack(anchor="center", expand=True)
                if row.get("locked"):
                    checkbutton.state(["disabled"])
                row["widgets"]["checkbutton"] = checkbutton
                continue

            label = tk.Label(
                cell,
                text=values.get(key, ""),
                anchor=anchor,
                bg=bg,
                fg=self.MUTED_COLOR if row.get("locked") else self.TEXT_COLOR,
                padx=8,
            )
            label.pack(fill="both", expand=True)
            row["widgets"][key] = label

    def _refresh_row(self, row):
        if "var" in row:
            row["var"].set(bool(row.get("include")))
        widgets = row.get("widgets") or {}
        checkbutton = widgets.get("checkbutton")
        if checkbutton is not None:
            checkbutton.state(["disabled"] if row.get("locked") else ["!disabled"])

    def _on_row_checkbutton_toggled(self, row):
        if row.get("locked"):
            row["include"] = False
            if "var" in row:
                row["var"].set(False)
            return
        row["include"] = bool(row.get("var").get()) if row.get("var") is not None else False
        self._refresh_select_all_ready_checkbox()

    def _toggle_all_ready(self):
        self._set_all_ready_selection(bool(self.select_all_ready_var.get()))

    def _set_all_ready_selection(self, include):
        for row in self.rows:
            if not row["locked"]:
                row["include"] = bool(include)
                self._refresh_row(row)
        self._refresh_select_all_ready_checkbox()

    def _refresh_select_all_ready_checkbox(self):
        ready_rows = [row for row in self.rows if not row["locked"]]
        if not ready_rows:
            self.select_all_ready_var.set(False)
            self.select_all_ready_check.state(["disabled"])
            return
        self.select_all_ready_check.state(["!disabled"])
        self.select_all_ready_var.set(all(bool(row.get("include")) for row in ready_rows))

    def _run_selected(self):
        rows = [row for row in self.rows if row["include"] and not row["locked"]]
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
      - _preprocessing_info: metadata for AI preprocessing crop params
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
        self._step1_after_id = None
        self._step1_watcher_active = False

        self.preferences = preferences
        self.source_step = source_step
        self.on_output_folder_changed = on_output_folder_changed

        # ─ Image data state ─
        self.current_file = None  # Path to currently loaded image
        self.image_data = None  # Current numpy array displayed on canvas
        self._source_was_8bit = False  # True when an opened 8-bit source was promoted for saving
        
        # ─ Preprocessing pipeline state ─
        self._last_auto_preprocessed_image = None  # Preprocessed image cached for AI segmentation
        self._original_image_for_ai = None  # Backup of original image before preprocessing
        self._original_file_for_ai = None  # Backup file path before preprocessing
        self._preprocessing_done = False  # Flag: True if preprocessing applied to current image
        self._preprocessing_info = None  # Dict: crop metadata to inverse-transform AI predictions
        
        # ─ Boundary tracing state ─
        self.active_boundary = None  # Name of boundary currently being traced
        self.boundary_traces = {}  # Dict mapping boundary name -> {points, pixels, color}
        self.boundary_order = []  # List of boundary names in order they were completed
        
        # ─ Foveal center line state ─
        self.fovea_x = None  # Current X-coordinate of foveal center line (or None if not set)
        
        # ─ UI state flags ─
        self._segmenter_running = False  # True while AI segmentation is executing
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
        self.source_label_var = tk.StringVar(value="No source selected")

        app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        segmenter_root = os.path.join(app_root, "OCT Segmenter")
        self.segmenter_default_config = os.path.join(segmenter_root, "config.json")
        self.segmenter_default_model = os.path.join(segmenter_root, "Model", "human_OCT.h5")
        self.ai_for_aidas_root = os.path.join(segmenter_root, "AI_ForAIDAS")
        self.ai_for_aidas_default_model = os.path.join(self.ai_for_aidas_root, "model_img.pth")
        self.ai_for_aidas_default_vline_model = os.path.join(self.ai_for_aidas_root, "vline_model.pth")

        self.build_standard_layout()
        right = self.content

        info_frame = ttk.Frame(right, relief="solid", borderwidth=1)
        info_frame.pack(fill="x", padx=2, pady=2)
        self.image_info_frame = info_frame
        self.image_info_var = tk.StringVar(value="No image loaded")
        info_label = ttk.Label(
            info_frame,
            textvariable=self.image_info_var,
            font=("", 10, "bold"),
            padding=8,
            anchor="w",
        )
        info_label.pack(fill="x")

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
        self.image_canvas.enable_line(True)
        self.image_canvas.enable_roi(False)
        self.image_canvas.enable_vertical_line(False)
        self.image_canvas.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(
            value="Ready — load a Step 1 result or any OCT image, then trace boundaries with left-clicks."
        )
        self.add_status_bar(self.status_var, parent=right)

        self._build_controls()

    def destroy(self):
        """Clean up when the panel is destroyed."""
        self._step1_watcher_active = False
        if getattr(self, "_step1_after_id", None) is not None:
            self.after_cancel(self._step1_after_id)
            self._step1_after_id = None
        super().destroy()

    # ═══════════════════════════════════════════════════════════════════════
    #  AI Settings Dialog
    # ═══════════════════════════════════════════════════════════════════════
    def _open_ai_settings_dialog(self):
        """Open a separate dialog for AI segmentation configuration.

        This dialog allows users to:
          - Choose the AI backend used by the single-image AI segmentation button
          - Set conda/config/model settings for oct-segmenter
          - Select PyTorch model files for AI_ForAIDAS
          - Specify output directory for segmentation results
        """
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("AI Segmentation Settings")
        dialog.geometry("620x620")
        dialog.resizable(True, True)

        # Apply shared helper to propagate the app icon to this dialog
        apply_app_icon_to(dialog)

        main = ttk.Frame(dialog, padding=10)
        main.pack(fill="both", expand=True)

        legacy = ttk.LabelFrame(main, text="OCT Segmenter (old)", padding=6)
        legacy.pack(fill="x", pady=(0, 8))

        ttk.Label(legacy, text="Conda Environment:").pack(anchor="w", pady=(0, 2))
        env_frame = ttk.Frame(legacy)
        env_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(env_frame, textvariable=self.segmenter_env_var).pack(fill="x")

        ttk.Label(legacy, text="Config (.json):").pack(anchor="w", pady=(2, 2))
        cfg_frame = ttk.Frame(legacy)
        cfg_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(cfg_frame, textvariable=self.segmenter_config_var).pack(side="left", fill="x", expand=True)
        ttk.Button(cfg_frame, text="Browse", width=10, command=self._browse_segmenter_config).pack(side="right", padx=(4, 0))

        ttk.Label(legacy, text="Model (.h5):").pack(anchor="w", pady=(2, 2))
        model_frame = ttk.Frame(legacy)
        model_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(model_frame, textvariable=self.segmenter_model_var).pack(side="left", fill="x", expand=True)
        ttk.Button(model_frame, text="Browse", width=10, command=self._browse_segmenter_model).pack(side="right", padx=(4, 0))

        aidas = ttk.LabelFrame(main, text="AI_ForAIDAS (new)", padding=6)
        aidas.pack(fill="x", pady=(0, 8))
        ttk.Label(aidas, text="Conda Environment:").pack(anchor="w", pady=(6, 2))
        ttk.Entry(aidas, textvariable=self.aidas_env_var).pack(fill="x", pady=(0, 6))

        ttk.Label(aidas, text="Boundary Model (.pth):").pack(anchor="w", pady=(0, 2))
        aidas_model_frame = ttk.Frame(aidas)
        aidas_model_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(aidas_model_frame, textvariable=self.aidas_model_var).pack(side="left", fill="x", expand=True)
        ttk.Button(aidas_model_frame, text="Browse", width=10, command=self._browse_aidas_model).pack(side="right", padx=(4, 0))

        ttk.Label(aidas, text="Fovea/VLine Model (.pth):").pack(anchor="w", pady=(2, 2))
        aidas_vline_frame = ttk.Frame(aidas)
        aidas_vline_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(aidas_vline_frame, textvariable=self.aidas_vline_model_var).pack(side="left", fill="x", expand=True)
        ttk.Button(aidas_vline_frame, text="Browse", width=10, command=self._browse_aidas_vline_model).pack(side="right", padx=(4, 0))

        aidas_options = ttk.Frame(aidas)
        aidas_options.pack(fill="x", pady=(0, 2))
        ttk.Checkbutton(
            aidas_options,
            text="Predict foveal center when vline model exists",
            variable=self.aidas_predict_fovea_var,
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(aidas_options, text="Device:").pack(side="left", padx=(8, 4))
        ttk.Combobox(
            aidas_options,
            textvariable=self.aidas_device_var,
            values=AI_DEVICE_OPTIONS,
            state="readonly",
            width=7,
        ).pack(side="right")


        ttk.Label(main, text="Output Folder:", font=(" ", 9, "bold")).pack(anchor="w", pady=(0, 2))
        out_frame = ttk.Frame(main)
        out_frame.pack(fill="x", pady=(0, 12))
        ttk.Entry(out_frame, textvariable=self.segmenter_output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(out_frame, text="Browse", width=10, command=self._browse_segmenter_output_dir).pack(side="right", padx=(4, 0))

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", expand=True, side="bottom")
        ttk.Button(buttons, text="OK", command=dialog.destroy).pack(side="right", padx=(2, 0))
        dialog.focus_set()
        dialog.grab_set()

    # ═══════════════════════════════════════════════════════════════════════
    #  UI construction
    # ═══════════════════════════════════════════════════════════════════════
    def _build_controls(self):
        """Construct and lay out all left-side control widgets.

        The left panel is organized into titled LabelFrames arranged vertically:
          1. Image Source: Load image or fetch from Step 1
          2. Segmentation: Workflow (incomplete/completed lists), AI buttons,
             clear traces button, and progress indicator
          3. Foveal Center Line: Vertical line placement and adjustment
          4. Export buttons: CSV export and MARKED image generation
          5. Help text: Usage instructions

        All controls are placed in a scrollable canvas to keep the UI compact
        and accessible even on smaller screens.
        """
        load_section = self.add_sidebar_section("Image Source", padding=3, pady=5)
        load = load_section.body

        self.step1_button = ttk.Button(load, text="Load from Step 1", command=self._load_from_step1)
        self.step1_button.pack(fill="x")
        if self.source_step is None:
            self.step1_button.state(["disabled"])
        else:
            # Start a watcher to enable the button only when Step 1 has a processed (cropped) image
            self._step1_watcher_active = True
            self.after(100, self._update_step1_button_state)

        # browser = ttk.LabelFrame(load, text="IMG Files", padding=3)
        # browser.pack(fill="x", pady=(6, 0))

        ttk.Label(load, text="Input dir:").pack(anchor="w")
        dir_frame = ttk.Frame(load)
        dir_frame.pack(fill="x")
        self.image_browser_dir_var = tk.StringVar(value=IMG_DEFAULT_DIR)
        self.image_browser_dir_entry = ttk.Entry(dir_frame, textvariable=self.image_browser_dir_var)
        self.image_browser_dir_entry.pack(side="left", fill="x", expand=True)
        self.image_browser_dir_entry.bind("<Return>", lambda _e: self._refresh_directory_image_list())
        self.image_browser_reset_btn = ttk.Button(
            dir_frame,
            text="⌂",
            width=2,
            command=self._reset_image_browser_dir_to_default,
        )
        self.image_browser_reset_btn.pack(side="right", padx=(2, 0))
        self.image_browser_refresh_btn = ttk.Button(
            dir_frame,
            text="↻",
            width=3,
            command=self._refresh_directory_image_list,
        )
        self.image_browser_refresh_btn.pack(side="right", padx=(2, 0))
        self.image_browser_folder_btn = ttk.Button(
            dir_frame,
            text="…",
            width=3,
            command=self._choose_image_browser_directory,
        )
        self.image_browser_folder_btn.pack(side="right")


        browser_list_frame = ttk.Frame(load)
        browser_list_frame.pack(fill="both", expand=True, pady=(2, 0))
        self.directory_image_listbox = tk.Listbox(
            browser_list_frame,
            height=8,
            selectmode="extended",
            exportselection=False,
        )
        image_scroll = ttk.Scrollbar(browser_list_frame, orient="vertical", command=self.directory_image_listbox.yview)
        self.directory_image_listbox.configure(yscrollcommand=image_scroll.set)
        self.directory_image_listbox.pack(side="left", fill="both", expand=True)
        image_scroll.pack(side="right", fill="y")
        self.directory_image_listbox.bind("<<ListboxSelect>>", self._on_directory_image_selected)
        self.directory_image_listbox.bind("<Double-Button-1>", self._open_selected_directory_image)

        nav_frame = ttk.Frame(load)
        nav_frame.pack(fill="x", pady=(4, 0))
        ttk.Button(nav_frame, text="◀ Prev",
                   command=self._prev_directory_image).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(nav_frame, text="Next ▶",
                   command=self._next_directory_image).pack(side="right", expand=True, fill="x", padx=(2, 0))

        self.image_browser_status_var = tk.StringVar(value="Open a directory to show images")
        ttk.Label(load, textvariable=self.image_browser_status_var, wraplength=240, justify="left").pack(
            fill="x",
            pady=(4, 0),
        )

        self.segmenter_config_var = tk.StringVar(value=self.segmenter_default_config)
        self.segmenter_model_var = tk.StringVar(value=self.segmenter_default_model)
        self.segmenter_output_var = tk.StringVar(value=self._default_segmenter_output_dir())
        self.segmenter_env_var = tk.StringVar(value="oct-segmenter-legacy-env")
        self.ai_backend_var = tk.StringVar(value=AI_BACKEND_LABELS[AI_BACKEND_AIDAS])
        self.aidas_model_var = tk.StringVar(value=self.ai_for_aidas_default_model)
        self.aidas_vline_model_var = tk.StringVar(value=self.ai_for_aidas_default_vline_model)
        self.aidas_predict_fovea_var = tk.BooleanVar(value=True)
        self.aidas_device_var = tk.StringVar(value=AI_DEVICE_OPTIONS[0])
        self.aidas_env_var = tk.StringVar(value="aidas-env")
        self.aidas_python_var = tk.StringVar(value="")

        self.segmentation_section = self.add_sidebar_section("Segmentation", padding=3, pady=2)
        self.segmentation_frame = self.segmentation_section.body
        segmentation = self.segmentation_frame

        self.active_trace_var = tk.StringVar(value="No active boundary")
        ttk.Label(segmentation, textvariable=self.active_trace_var, wraplength=240, justify="left").pack(fill="x", pady=(0, 4))

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
        self.finish_boundary_btn = ttk.Button(workflow_buttons, text="Finish", command=self._finish_boundary)
        self.finish_boundary_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.revert_boundary_btn = ttk.Button(workflow_buttons, text="Revert", command=self._revert_boundary)
        self.revert_boundary_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))

        ai_select = ttk.Frame(workflow)
        ai_select.pack(fill="x", pady=(6, 0))
        ttk.Label(ai_select, text="AI Version:").pack(side="left")
        self.ai_backend_combo = ttk.Combobox(
            ai_select,
            textvariable=self.ai_backend_var,
            values=[AI_BACKEND_LABELS[AI_BACKEND_OCT_SEGMENTER], AI_BACKEND_LABELS[AI_BACKEND_AIDAS]],
            state="readonly",
            width=20,
        )
        self.ai_backend_combo.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.ai_backend_combo.bind("<<ComboboxSelected>>", self._on_ai_backend_changed)

        ai_buttons = ttk.Frame(workflow)
        ai_buttons.pack(fill="x", pady=(6, 0))
        self.preprocess_button = ttk.Button(
            ai_buttons,
            text="Preprocess Image",
            command=self._preprocess_image_for_ai,
        )
        self.preprocess_button.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.ai_settings_btn = ttk.Button(ai_buttons, text="AI Settings", command=self._open_ai_settings_dialog)
        self.ai_settings_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))

        ai_run_buttons = ttk.Frame(workflow)
        ai_run_buttons.pack(fill="x", pady=(4, 0))
        self.segment_button = ttk.Button(
            ai_run_buttons,
            text="AI Single Image Seg",
            command=self._run_neural_segmentation,
        )
        self.segment_button.pack(fill="x")

        ai_folder_buttons = ttk.Frame(workflow)
        ai_folder_buttons.pack(fill="x", pady=(4, 0))
        self.batch_ai_button = ttk.Button(
            ai_folder_buttons,
            text="Run Batch Segmentation",
            command=self._open_batch_segmentation_scanner,
        )
        self.batch_ai_button.pack(fill="x")

        self.aidas_model_var.trace_add("write", lambda *_: self._update_batch_ai_button_state())

        self.clear_all_traces_btn = ttk.Button(workflow, text="Clear All Traces", command=self._clear_all_traces)
        self.clear_all_traces_btn.pack(fill="x", pady=(6, 4))

        self.segmenter_progress_var = tk.StringVar(value="Idle")
        self.segmenter_progress = ttk.Progressbar(workflow, mode="determinate", maximum=6, value=0)
        self.segmenter_progress.pack(fill="x", pady=(6, 4))

        self.boundary_workflow_status_var = tk.StringVar(value="Select a boundary to make it active.")
        ttk.Label(workflow, textvariable=self.boundary_workflow_status_var, wraplength=240, justify="left").pack(
            fill="x",
            pady=(4, 0),
        )

        fovea = ttk.LabelFrame(segmentation, text="Foveal Center Line", padding=3)
        fovea.pack(fill="x", pady=(4, 2))
        self.fovea_frame = fovea  # Store reference for later state management

        self.vertical_mode_var = tk.BooleanVar(value=False)
        self.vertical_mode_check = ttk.Checkbutton(
            fovea,
            text="Vertical line mode (click/drag on image)",
            variable=self.vertical_mode_var,
            command=self._on_vertical_mode_toggled,
        )
        self.vertical_mode_check.pack(anchor="w")

        # Keep this state var even when the label is hidden; callbacks rely on it.
        self.fovea_line_var = tk.StringVar(value="Fovea line: not set")

        coord_row = ttk.Frame(fovea)
        coord_row.pack(fill="x", pady=(0, 2))
        ttk.Label(coord_row, text="Center X:").pack(side="left")
        self.fovea_x_entry_var = tk.StringVar(value="")
        self.fovea_x_entry_var.trace_add("write", self._on_fovea_x_entry_changed)
        self.fovea_minus_btn = ttk.Button(coord_row, text="-", width=2)
        self.fovea_minus_btn.pack(side="left", padx=(0, 2))
        self.fovea_x_entry = ttk.Entry(coord_row, textvariable=self.fovea_x_entry_var, width=6)
        self.fovea_x_entry.pack(side="left", padx=(6, 4))
        self.fovea_plus_btn = ttk.Button(coord_row, text="+", width=2)
        self.fovea_plus_btn.pack(side="left", padx=(0, 4))
        self._bind_fovea_nudge_button(self.fovea_minus_btn, -1)
        self._bind_fovea_nudge_button(self.fovea_plus_btn, 1)
        self.fovea_x_entry.bind("<Return>", lambda _e: self._apply_vertical_line_x())

        # Reset icon matches Step 1 numeric reset affordance.
        self.fovea_reset_btn = ttk.Button(coord_row, text="↺", width=2, command=self._center_vertical_line)
        self.fovea_reset_btn.pack(side="left")

        fovea_action_row = ttk.Frame(fovea)
        fovea_action_row.pack(fill="x", pady=(2, 0))
        self.fovea_set_btn = ttk.Button(fovea_action_row, text="Set", command=self._apply_vertical_line_x)
        self.fovea_set_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.fovea_clear_btn = ttk.Button(fovea_action_row, text="Clear", command=self._clear_fovea_lock)
        self.fovea_clear_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))

        self._set_fovea_controls_enabled(False)

        self.trace_detail_var = tk.StringVar(value="No saved boundary")
        ttk.Label(segmentation, textvariable=self.trace_detail_var, wraplength=240, justify="left").pack(
            fill="x",
            pady=(4, 0),
        )

        saved_buttons = ttk.Frame(segmentation)
        saved_buttons.pack(fill="x", pady=(6, 0))
        self.saved_buttons_frame = saved_buttons  # Store reference for later state management
        # ttk.Button(saved_buttons, text="Export CSV", command=self._export_csv).pack(
        #     side="left",
        #     expand=True,
        #     fill="x",
        #     padx=(0, 2),
        # )
        ttk.Button(saved_buttons, text="Save this image", command=self._save_current_marked_image_button).pack(
            side="left",
            expand=True,
            fill="x",
            padx=2,
        )
        ttk.Button(saved_buttons, text="Save all", command=self._save_marked_images_button).pack(
            side="right",
            expand=True,
            fill="x",
            padx=(2, 0),
        )

        help_section = self.add_sidebar_section("How to Trace", padding=3, pady=(2, 6))
        help_box = help_section.body
        ttk.Label(
            help_box,
            text=(
                "1. Open a directory and choose an .img image.\n"
                "2. Pick a boundary name.\n"
                "3. Left-click points along the boundary.\n"
                "4. Press Finish Boundary to save all pixels on the line.\n"
                "5. Enable Vertical line mode to place/drag the foveal center line."
            ),
            justify="left",
        ).pack(anchor="w")

        self._refresh_boundary_lists(auto_select=False)
        self._set_segmentation_frame_enabled(False)
        self._directory_image_paths_cache = []
        self._image_browser_dir = ""
        self._update_batch_ai_button_state()
        self._set_image_browser_directory(IMG_DEFAULT_DIR, preserve_selection=False)

    # ═══════════════════════════════════════════════════════════════════════
    #  Image loading
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _is_supported_image_path(path):
        return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS

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

    def _directory_image_paths(self, folder):
        """Return .img images in a folder."""
        if not folder or not os.path.isdir(folder):
            return []

        selected_by_key = {}
        for name in sorted(os.listdir(folder), key=str.lower):
            path = os.path.join(folder, name)
            if not os.path.isfile(path) or not self._is_supported_image_path(path):
                continue
            path = self._preferred_analyze_pair_path(path)
            key = self._image_pair_key(path)
            current = selected_by_key.get(key)
            if current is None or os.path.splitext(path)[1].lower() == ".img":
                selected_by_key[key] = path

        return sorted(selected_by_key.values(), key=lambda item: os.path.basename(item).lower())

    def _recursive_directory_image_paths(self, folder):
        """Return supported images in a folder tree."""
        if not folder or not os.path.isdir(folder):
            return []

        selected_by_key = {}
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort(key=str.lower)
            for name in sorted(filenames, key=str.lower):
                path = os.path.join(dirpath, name)
                if not os.path.isfile(path) or not self._is_supported_image_path(path):
                    continue
                path = self._preferred_analyze_pair_path(path)
                key = self._image_pair_key(path)
                current = selected_by_key.get(key)
                if current is None or os.path.splitext(path)[1].lower() == ".img":
                    selected_by_key[key] = path

        root = os.path.abspath(folder)
        return sorted(
            selected_by_key.values(),
            key=lambda item: os.path.relpath(os.path.abspath(item), root).lower(),
        )

    def _set_image_browser_directory(self, folder, select_path=None, preserve_selection=True):
        folder = os.path.abspath(folder) if folder else ""
        self._image_browser_dir = folder
        if hasattr(self, "image_browser_dir_var"):
            self.image_browser_dir_var.set(folder)
        self._refresh_directory_image_list(select_path=select_path, preserve_selection=preserve_selection)

    def _refresh_directory_image_list(self, select_path=None, preserve_selection=True):
        listbox = getattr(self, "directory_image_listbox", None)
        if listbox is None:
            return

        folder = self.image_browser_dir_var.get().strip() if hasattr(self, "image_browser_dir_var") else ""
        if not folder:
            self._image_browser_dir = ""
            self._directory_image_paths_cache = []
            listbox.delete(0, "end")
            self.image_browser_status_var.set("Open a directory to list .img images")
            if self.image_data is None:
                self._set_segmentation_frame_enabled(False)
            self._update_batch_ai_button_state()
            return
        folder = os.path.abspath(folder)
        self._image_browser_dir = folder
        if self.image_browser_dir_var.get() != folder:
            self.image_browser_dir_var.set(folder)
        if not os.path.isdir(folder):
            self._directory_image_paths_cache = []
            listbox.delete(0, "end")
            self.image_browser_status_var.set("Folder not found")
            if self.image_data is None:
                self._set_segmentation_frame_enabled(False)
            self._update_batch_ai_button_state()
            return

        selected_keys = set()
        if preserve_selection:
            for index in listbox.curselection():
                try:
                    selected_keys.add(self._image_pair_key(self._directory_image_paths_cache[int(index)]))
                except (AttributeError, IndexError, ValueError):
                    pass
        if select_path:
            selected_keys.add(self._image_pair_key(select_path))

        paths = self._directory_image_paths(folder)
        filter_text = self.image_browser_filter_var.get().strip().lower() if hasattr(self, "image_browser_filter_var") else ""
        if filter_text:
            paths = [
                path for path in paths
                if filter_text in os.path.basename(path).lower()
            ]
        self._directory_image_paths_cache = paths

        self._updating_image_browser = True
        try:
            listbox.delete(0, "end")
            for path in paths:
                listbox.insert("end", os.path.basename(path))
            for index, path in enumerate(paths):
                if self._image_pair_key(path) in selected_keys:
                    listbox.selection_set(index)
                    listbox.see(index)
        finally:
            self._updating_image_browser = False

        count = len(paths)
        suffix = " matching filter" if filter_text else ""
        self.image_browser_status_var.set(f"{count} image(s){suffix}")
        if paths:
            self._set_segmentation_frame_enabled(True)
            self._update_ai_button_states()
        elif self.image_data is None:
            self._set_segmentation_frame_enabled(False)
        self._update_batch_ai_button_state()

    def _choose_image_browser_directory(self):
        typed_dir = self.image_browser_dir_var.get().strip() if hasattr(self, "image_browser_dir_var") else ""
        initialdir = typed_dir if os.path.isdir(typed_dir) else (getattr(self, "_image_browser_dir", "") or self._app_root())
        file_path = filedialog.askopenfilename(
            title="Select IMG file (directory will be used)",
            initialdir=initialdir,
            filetypes=SUPPORTED_IMAGE_FILETYPES,
        )
        if file_path:
            self._set_image_browser_directory(os.path.dirname(file_path), select_path=file_path, preserve_selection=False)

    def _reset_image_browser_dir_to_default(self):
        self._set_image_browser_directory(IMG_DEFAULT_DIR, preserve_selection=False)
        self.status_var.set(f"IMG directory reset to default: {IMG_DEFAULT_DIR}")

    def _selected_directory_image_paths(self):
        listbox = getattr(self, "directory_image_listbox", None)
        paths = getattr(self, "_directory_image_paths_cache", [])
        if listbox is None:
            return []
        selected = []
        for index in listbox.curselection():
            try:
                selected.append(paths[int(index)])
            except (IndexError, ValueError):
                pass
        return selected

    def _on_directory_image_selected(self, _event=None):
        if getattr(self, "_updating_image_browser", False) or self._segmenter_running:
            return
        paths = self._selected_directory_image_paths()
        self._update_batch_ai_button_state()
        if len(paths) != 1:
            if paths:
                self.status_var.set(f"{len(paths)} IMG files selected.")
            elif getattr(self, "_directory_image_paths_cache", []):
                self.status_var.set("Select an IMG file to load, or use Run Batch Segmentation for folder processing.")
            return

        path = paths[0]
        self.status_var.set(f"Selected IMG: {os.path.basename(path)}")
        pending = getattr(self, "_image_browser_open_after", None)
        if pending:
            self.after_cancel(pending)
        self._image_browser_open_after = self.after(
            120,
            lambda p=path: self._open_directory_image_path_if_still_single(p),
        )

    def _open_directory_image_path_if_still_single(self, path):
        self._image_browser_open_after = None
        paths = self._selected_directory_image_paths()
        if len(paths) == 1 and self._image_pair_key(paths[0]) == self._image_pair_key(path):
            self._open_directory_image_path(path)

    def _open_selected_directory_image(self, _event=None):
        paths = self._selected_directory_image_paths()
        if not paths:
            messagebox.showinfo("No image selected", "Select an image from the current directory list first.")
            return
        self._open_directory_image_path(paths[0])

    def _prev_directory_image(self):
        paths = getattr(self, "_directory_image_paths_cache", [])
        if not paths:
            return
        selection = self.directory_image_listbox.curselection()
        index = max(0, selection[0] - 1) if selection else 0
        self.directory_image_listbox.selection_clear(0, "end")
        self.directory_image_listbox.selection_set(index)
        self.directory_image_listbox.see(index)
        self._open_directory_image_path(paths[index])

    def _next_directory_image(self):
        paths = getattr(self, "_directory_image_paths_cache", [])
        if not paths:
            return
        selection = self.directory_image_listbox.curselection()
        index = min(len(paths) - 1, selection[0] + 1) if selection else 0
        self.directory_image_listbox.selection_clear(0, "end")
        self.directory_image_listbox.selection_set(index)
        self.directory_image_listbox.see(index)
        self._open_directory_image_path(paths[index])

    def _open_directory_image_path(self, path):
        try:
            image = self._load_image_from_path(path)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Open error", str(exc))
            return

        self._loading_from_image_browser = True
        try:
            self._show_image(image, path)
        finally:
            self._loading_from_image_browser = False
        self.image_browser_status_var.set(f"Loaded {os.path.basename(path)}")

    def _segment_selected_directory_images(self):
        paths = self._selected_directory_image_paths()
        if not self._is_aidas_ai_backend():
            messagebox.showinfo(
                "AI_ForAIDAS required",
                "Select AI_ForAIDAS (New, PyTorch-based) before running batch segmentation.",
            )
            return
        model_path = self.aidas_model_var.get().strip()
        if not os.path.isfile(model_path):
            messagebox.showerror("Missing model", f"AI_ForAIDAS boundary model not found:\n{model_path}")
            return
        if len(paths) < 2:
            messagebox.showinfo("Select multiple images", "Select at least two .img files for batch segmentation.")
            return
        self._run_aidas_batch_segmentation(image_paths=paths)

    def _open_batch_segmentation_scanner(self):
        """Open an embedded folder scanner before running Step 2 batch segmentation."""
        if self._segmenter_running:
            messagebox.showinfo("Please wait", "Segmentation is already running.")
            return
        if not self._is_aidas_ai_backend():
            messagebox.showinfo(
                "AI_ForAIDAS required",
                "Select AI_ForAIDAS (New, PyTorch-based) before running batch segmentation.",
            )
            return
        model_path = self.aidas_model_var.get().strip()
        if not os.path.isfile(model_path):
            messagebox.showerror("Missing model", f"AI_ForAIDAS boundary model not found:\n{model_path}")
            return

        current_dir = getattr(self, "_image_browser_dir", "") or self.image_browser_dir_var.get().strip()
        if not os.path.isdir(current_dir):
            current_dir = IMG_DEFAULT_DIR if os.path.isdir(IMG_DEFAULT_DIR) else self._app_root()
        folder = filedialog.askdirectory(
            title="Select root folder for Step 2 batch segmentation",
            initialdir=current_dir,
        )
        if not folder:
            return

        self._open_step2_batch_segmentation_panel(folder)

    def _open_step2_batch_segmentation_panel(self, root_dir):
        self._close_step2_batch_segmentation_panel(restore_previous=False)

        pending_open = getattr(self, "_image_browser_open_after", None)
        if pending_open:
            try:
                self.after_cancel(pending_open)
            except tk.TclError:
                pass
            self._image_browser_open_after = None

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
                    info_frame.pack(fill="x", padx=2, pady=2, before=self.canvas_area)
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
        """Return folder rows showing ready and already-segmented Light/Dark inputs."""
        root_dir = os.path.abspath(root_dir)
        rows = []
        scanned = 0
        skipped = 0
        targets = (
            ("Light", "light.img", "light_marked.img"),
            ("Dark", "dark.img", "dark_marked.img"),
        )

        for folder, dirnames, filenames in os.walk(root_dir):
            dirnames.sort(key=str.lower)
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
        return rows, scanned, skipped

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
            messagebox.showwarning("Batch Step 2", "No unsegmented Light.img or Dark.img files were selected.")
            return

        self._close_step2_batch_segmentation_panel(restore_previous=True)
        self._set_image_browser_directory(root_dir, preserve_selection=False)
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

    def _run_folder_segmentation(self):
        """Select a folder, recursively find images, and run AI_ForAIDAS batch segmentation."""
        if self._segmenter_running:
            messagebox.showinfo("Please wait", "Segmentation is already running.")
            return
        if not self._is_aidas_ai_backend():
            messagebox.showinfo(
                "AI_ForAIDAS required",
                "Select AI_ForAIDAS (New, PyTorch-based) before running folder segmentation.",
            )
            return
        model_path = self.aidas_model_var.get().strip()
        if not os.path.isfile(model_path):
            messagebox.showerror("Missing model", f"AI_ForAIDAS boundary model not found:\n{model_path}")
            return

        current_dir = getattr(self, "_image_browser_dir", "") or self.image_browser_dir_var.get().strip()
        if not os.path.isdir(current_dir):
            current_dir = IMG_DEFAULT_DIR if os.path.isdir(IMG_DEFAULT_DIR) else self._app_root()
        folder = filedialog.askdirectory(
            title="Select folder for AI folder segmentation",
            initialdir=current_dir,
        )
        if not folder:
            return

        image_paths = []
        found_any = False
        all_completed = True

        for root_dir, _, files in os.walk(folder):
            lower_files = {f.lower(): f for f in files}
            
            has_light = "light.img" in lower_files
            has_dark = "dark.img" in lower_files
            has_light_marked = "light_marked.img" in lower_files
            has_dark_marked = "dark_marked.img" in lower_files

            if not has_light and not has_dark:
                continue

            found_any = True

            complete_light = has_light_marked if has_light else True
            complete_dark = has_dark_marked if has_dark else True

            if complete_light and complete_dark:
                continue

            all_completed = False
            if has_light and not has_light_marked:
                image_paths.append(os.path.join(root_dir, lower_files["light.img"]))
            if has_dark and not has_dark_marked:
                image_paths.append(os.path.join(root_dir, lower_files["dark.img"]))

        if not found_any:
            messagebox.showwarning(
                "No images found",
                f"No Light.img or Dark.img files were found in:\n{folder}",
            )
            return

        if all_completed:
            messagebox.showinfo("Completed", "They all completed.")
            return

        self._set_image_browser_directory(folder, preserve_selection=False)
        self.status_var.set(f"Found {len(image_paths)} target image(s) in folder tree for AI segmentation.")
        
        manual_fovea_by_path = self._collect_folder_fovea_lines(image_paths)
        if manual_fovea_by_path is None:
            self.status_var.set("AI folder segmentation cancelled before running.")
            return

        # Keep only the images that were not skipped
        image_paths = [p for p in image_paths if p in manual_fovea_by_path]
        if not image_paths:
            self.status_var.set("AI folder segmentation cancelled: all images were skipped.")
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
        
        # Disable batch buttons to prevent duplicate triggers
        if hasattr(self, "folder_segment_button"):
            self.folder_segment_button.state(["disabled"])
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
            
        btn_cancel = ttk.Button(temp_frame, text="Cancel AI Batch", command=on_cancel)
        btn_cancel.pack(side="right", padx=4, pady=4)
        
        btn_skip = ttk.Button(temp_frame, text="Skip Image", command=on_skip)
        btn_skip.pack(side="right", padx=4, pady=4)

        btn_set = ttk.Button(temp_frame, text="Set Fovea & Process", command=on_set)
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
                    fovea_by_path[path] = None
                    continue

                self._show_image(image_data, path)
                self.vertical_mode_var.set(True)
                self._on_vertical_mode_toggled()
                
                # Predict fovea centrally if enabled
                if self.aidas_predict_fovea_var.get():
                    vline_path = self.aidas_vline_model_var.get().strip()
                    if os.path.isfile(vline_path):
                        try:
                            from aidas.ai_for_aidas_inference import AIForAIDASPredictor
                            device_name = self.aidas_device_var.get().strip() or "cpu"
                            import torch
                            device = torch.device(device_name if torch.cuda.is_available() else "cpu")
                            pred = AIForAIDASPredictor(None, vline_path, device)
                            fovea_x, _ = pred.predict_fovea([image_data])
                            if fovea_x is not None:
                                self.fovea_x_entry_var.set(str(int(fovea_x[0])))
                                self.image_canvas.vertical_line_x = int(fovea_x[0])
                                self.image_canvas._draw_vertical_line()
                        except Exception as e:
                            print("Auto-predict fovea failed:", e)

                self.wait_variable(next_var)
                
                action = next_var.get()
                if action == "cancel":
                    self._load_editor_state(saved_state, self.image_canvas)
                    return None
                elif action == "skip":
                    # Do not add it to fovea_by_path, which explicitly drops it from the batch list
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
            if hasattr(self, "folder_segment_button"):
                self.folder_segment_button.state(["!disabled"])
            self._update_batch_ai_button_state()
            self.vertical_mode_var.set(False)
            self._on_vertical_mode_toggled()

        return fovea_by_path

    def _open_image(self):
        """Show open-file dialog and load the selected image.

        Supports multiple formats:
          - TIFF stacks (.tif, .tiff) → extracts first slice
          - Analyze 7.5 (.hdr/.img pairs) → reads header/data
          - Standard images (.png, .jpg, .jpeg) → converts to grayscale
        
        Auto-detects format by file extension and preserves 16-bit grayscale
        data when available. Clears all existing traces and resets UI.
        """
        path = filedialog.askopenfilename(
            title="Select an OCT image",
            filetypes=SUPPORTED_IMAGE_FILETYPES,
        )
        if not path:
            return
        try:
            image = self._load_image_from_path(path)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Open error", str(exc))
            return
        self._show_image(image, path)

    def _load_from_step1(self):
        """Load the processed image exported by the connected Step 1 panel.

        Step 1 provides a link to automatically import the cropped/scaled image
        after preprocessing. If Step 1 has a processed_image (cropped), it is
        preferred; otherwise the raw image is used.

        Preserves the source image bit depth and resets all traces.
        """
        if self.source_step is None:
            messagebox.showinfo("Unavailable", "No Step 1 panel is connected to this view.")
            return

        image = getattr(self.source_step, "processed_image", None)
        source_path = getattr(self.source_step, "current_file", None)
        if image is None:
            image = getattr(self.source_step, "raw_image", None)
        if image is None:
            messagebox.showinfo("No image", "Step 1 has no loaded image yet.")
            return

        display_path = source_path or "Step 1 output"
        self._input_analyze_template = None
        img = self._image_for_annotation(image)
        self._show_image(img, display_path)

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

    def _update_step1_button_state(self):
        """Enable the Step 1 load button only when `source_step.processed_image` exists.

        This method re-schedules itself via `after` while the frame exists.
        """
        if not self.winfo_exists():
            return
            
        try:
            has = self.source_step is not None and getattr(self.source_step, "processed_image", None) is not None
        except Exception:
            has = False

        if has:
            self.step1_button.state(["!disabled"])
        else:
            self.step1_button.state(["disabled"])

        # continue polling as long as this frame is mapped
        if getattr(self, "_step1_watcher_active", False) and self.winfo_exists():
            self._step1_after_id = self.after(500, self._update_step1_button_state)

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
          - Clears preprocessing state and backup
          - Resets foveal center line
          - Updates info display and UI controls
          - Re-enables all controls for fresh annotation

        Args:
            image: numpy array to display and annotate. Source bit depth is preserved;
                the canvas normalizes a preview for display only.
            path: Path or description string for the image source.
        """
        self._last_auto_preprocessed_image = None
        self._original_image_for_ai = None
        self._original_file_for_ai = None
        self._preprocessing_done = False
        self._preprocessing_info = None
        self._show_single_image_canvas()
        self.current_file = path
        self.image_data = image
        self.active_boundary = None
        self.boundary_traces = {}
        self.boundary_order = []
        self.fovea_x = None

        self.image_canvas.set_image(image)
        self.image_canvas.enable_roi(False)
        self.image_canvas.clear_active_line()
        self.image_canvas.clear_line_overlays()
        self.image_canvas.clear_vertical_line()
        self.image_canvas.fit_to_window()

        filename = os.path.basename(path) if path and path != "Step 1 output" else "Step 1 output"
        self.source_label_var.set(path)
        self.image_info_var.set(
            f"{filename} | Size: {image.shape[1]} × {image.shape[0]} px | Type: {image.dtype}"
        )
        self.trace_detail_var.set("No saved boundary")
        self.active_trace_var.set("No active boundary")
        self.fovea_line_var.set("Fovea line: not set")
        self.fovea_x_entry_var.set("")
        self._reset_boundary_completion()
        self._set_drawing_locked(False)
        self._refresh_trace_list()
        self._sync_boundary_canvas_state()
        self._set_segmentation_frame_enabled(True)
        self._update_ai_button_states()
        self.status_var.set(
            "Image loaded. Left-click to place points only after selecting an incomplete boundary."
        )
        self._notify_output_folder_changed()

    def _notify_output_folder_changed(self):
        if not self.current_file or self.current_file == "Step 1 output":
            return
        folder = os.path.dirname(self.current_file)
        if folder:
            if not getattr(self, "_loading_from_image_browser", False):
                self._set_image_browser_directory(folder, select_path=self.current_file)
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

    def _set_fovea_from_prediction(self, x):
        if self.image_data is None:
            return
        width = int(self.image_data.shape[1])
        height = int(self.image_data.shape[0])
        fovea_x = int(np.clip(int(x), 0, max(0, width - 1)))

        self.fovea_x = fovea_x
        self.boundary_traces[FOVEA_BOUNDARY_NAME] = self._vertical_line_trace(fovea_x, height)
        if FOVEA_BOUNDARY_NAME not in self.boundary_order:
            self.boundary_order.append(FOVEA_BOUNDARY_NAME)

        self.vertical_mode_var.set(True)
        self._updating_fovea_entry = True
        try:
            self.fovea_x_entry_var.set(str(fovea_x))
        finally:
            self._updating_fovea_entry = False
        self.fovea_line_var.set(f"Fovea line: x={fovea_x}")
        self.image_canvas.enable_vertical_line(False)
        self._set_fovea_controls_enabled(True)
        self._rebuild_saved_overlays()
        self._refresh_trace_list()
        self._select_trace_by_name(FOVEA_BOUNDARY_NAME)
        self._update_saved_trace_summary(FOVEA_BOUNDARY_NAME)
        self._sync_boundary_canvas_state()

    def _has_saved_fovea_trace(self):
        return FOVEA_BOUNDARY_NAME in self.boundary_traces

    def _fovea_live_edit_mode(self):
        return bool(self.vertical_mode_var.get() and not self._has_saved_fovea_trace())

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
        self._last_auto_preprocessed_image = None
        self._original_image_for_ai = None
        self._original_file_for_ai = None
        self._preprocessing_done = False
        self._preprocessing_info = None
        self.active_boundary = None

        self.image_canvas.enable_roi(False)
        self.image_canvas.clear_active_line()
        self._rebuild_saved_overlays()
        self._set_completion_from_traces()

        if self.fovea_x is not None:
            self._set_fovea_from_prediction(self.fovea_x)
        else:
            self.vertical_mode_var.set(False)
            self.image_canvas.clear_vertical_line()
            self._set_fovea_controls_enabled(False)
            self._updating_fovea_entry = True
            try:
                self.fovea_x_entry_var.set("")
            finally:
                self._updating_fovea_entry = False
            self.fovea_line_var.set("Fovea line: not set")

        filename = (
            os.path.basename(self.current_file)
            if self.current_file and self.current_file != "Step 1 output"
            else "Step 1 output"
        )
        if self.image_data is not None:
            self.source_label_var.set(self.current_file or "")
            self.image_info_var.set(
                f"{filename} | Size: {self.image_data.shape[1]} x {self.image_data.shape[0]} px | "
                f"Type: {self.image_data.dtype}"
            )

        self.trace_detail_var.set("No saved boundary")
        self.active_trace_var.set("No active boundary")
        self._refresh_trace_list()
        selected_trace = None
        if self.fovea_x is not None and FOVEA_BOUNDARY_NAME in self.boundary_traces:
            selected_trace = FOVEA_BOUNDARY_NAME
        elif self.boundary_order:
            selected_trace = self.boundary_order[0]
        if selected_trace is not None:
            self._select_trace_by_name(selected_trace)
            self._update_saved_trace_summary(selected_trace)
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
        Automatically disables vertical line mode and clears any active unfinished trace.

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

        if self._fovea_live_edit_mode():
            self.vertical_mode_var.set(False)
            self._on_vertical_mode_toggled()

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

        if self._fovea_live_edit_mode():
            self.vertical_mode_var.set(False)
            self._on_vertical_mode_toggled()

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

    def _update_boundary_action_buttons(self):
        active_name = self.active_boundary if self.active_boundary in BOUNDARY_NAMES else None
        is_completed = active_name in self._completed_boundary_names()
        is_incomplete = active_name in self._incomplete_boundary_names()

        if getattr(self, "finish_boundary_btn", None) is not None:
            self.finish_boundary_btn.configure(state="normal" if is_incomplete else "disabled")
        if getattr(self, "revert_boundary_btn", None) is not None:
            self.revert_boundary_btn.configure(state="normal" if is_completed else "disabled")

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
          - Vertical line mode is active

        Completed boundaries must be reverted to incomplete to edit.

        Vertical line mode is DISABLED during AI segmentation to prevent conflicts.

        This method is called after each state change (AI finish, boundary selection,
        vertical line toggle) to maintain consistent canvas interactivity.
        """
        active_name = self.active_boundary if self.active_boundary in BOUNDARY_NAMES else None
        # Only allow drawing on incomplete boundaries
        is_incomplete = active_name in self._incomplete_boundary_names()
        drawing_enabled = (
            active_name is not None
            and is_incomplete
            and not self._fovea_live_edit_mode()
            and not self._segmenter_running
        )
        self.image_canvas.enable_line(drawing_enabled)
        self.image_canvas.enable_vertical_line(self._fovea_live_edit_mode() and not self._segmenter_running)

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
        self._update_saved_trace_summary(name)
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
        self.trace_detail_var.set("No saved boundary")
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
        self.boundary_traces.pop(name, None)
        self.boundary_order.remove(name)
        if name in self.boundary_completion_vars:
            self.boundary_completion_vars[name].set(False)
        self._refresh_boundary_lists(select_incomplete_name=name)
        self._rebuild_saved_overlays()
        self._refresh_trace_list()
        self.trace_detail_var.set("No saved boundary")
        self.status_var.set(f"Deleted boundary '{name}'.")

    def _clear_all_traces(self):
        """Remove all saved boundaries and the active trace at once.

        Asks for confirmation before clearing to prevent accidental data loss.
        Resets boundary completion status and clears canvas overlays.
        """
        if not self.boundary_traces and not self.image_canvas.get_active_line():
            return
        if not messagebox.askyesno("Clear all traces?", "Remove every saved and active boundary trace?"):
            return
        self.boundary_traces.clear()
        self.boundary_order.clear()
        self._reset_boundary_completion()
        self.active_boundary = None
        self.image_canvas.clear_line_overlays()
        self.image_canvas.clear_active_line()
        self._refresh_boundary_lists()
        self._refresh_trace_list()
        self.trace_detail_var.set("No saved boundary")
        self.active_trace_var.set("No active boundary")
        self.status_var.set("All boundary traces cleared.")

    def _rebuild_saved_overlays(self):
        """Redraw all saved boundary overlays on the canvas from boundary_traces.

        Called after any modification to boundary_traces or boundary_order.
        Maintains order and colors when re-rendering the canvas display.
        """
        self.image_canvas.clear_line_overlays()
        for name in self.boundary_order:
            trace = self.boundary_traces.get(name)
            if trace:
                self.image_canvas.add_line_overlay(trace["points"], color=trace["color"], label=name)

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

    def _on_vertical_mode_toggled(self):
        """Toggle foveal center line mode on/off.

        When enabled:
          - Shows the saved fovea trace as complete when it already exists
          - Otherwise switches to vertical-line drawing mode on the canvas
          - Enables foveal center nudge controls (left/right buttons, X entry)

        When disabled:
          - Disables foveal center editing
          - Keeps an existing saved fovea trace visible
          - Re-enables polyline drawing for boundaries
          - Disables foveal center controls

        Prevents mode toggle during AI segmentation (_drawing_locked).
        """
        if self._drawing_locked:
            self.vertical_mode_var.set(False)
            return
        enabled = self.vertical_mode_var.get()
        has_saved_fovea = self._has_saved_fovea_trace()
        live_fovea = bool(enabled and not has_saved_fovea)
        self._set_fovea_controls_enabled(enabled)
        self.image_canvas.enable_vertical_line(live_fovea)
        self.image_canvas.enable_line(not live_fovea)
        if enabled:
            if self.image_data is not None:
                if has_saved_fovea:
                    if self.fovea_x is None:
                        points = self.boundary_traces.get(FOVEA_BOUNDARY_NAME, {}).get("points", [])
                        if points:
                            self.fovea_x = int(points[0][0])
                    self._rebuild_saved_overlays()
                    self.status_var.set("Foveal center is complete in the segmentation map.")
                elif self.fovea_x is not None:
                    self.image_canvas.set_vertical_line_x(int(self.fovea_x))
                    self.status_var.set(f"Vertical line mode enabled. Foveal center line set to x={int(self.fovea_x)}.")
                else:
                    self._center_vertical_line()
                    self.status_var.set("Vertical line mode enabled. Foveal center line set to image center.")
            else:
                self.status_var.set("Vertical line mode enabled. Load an image to place the foveal center line.")
        else:
            if not has_saved_fovea:
                self.image_canvas.clear_vertical_line()
            self._rebuild_saved_overlays()
            self._refresh_trace_list()
            if has_saved_fovea:
                self._select_trace_by_name(FOVEA_BOUNDARY_NAME)
                self._update_saved_trace_summary(FOVEA_BOUNDARY_NAME)
                self.status_var.set("Boundary tracing mode enabled. Saved foveal center line is preserved.")
            else:
                self.trace_detail_var.set("No saved boundary")
                self.status_var.set("Boundary tracing mode enabled. Left-click to place boundary points.")

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
            f"Incomplete: {len(incomplete_names)} | Completed: {len(completed_names)}/{len(BOUNDARY_NAMES)}"
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
        if self._has_saved_fovea_trace():
            self._set_fovea_from_prediction(center_x)
        else:
            self.image_canvas.set_vertical_line_x(center_x)

    def _clear_vertical_line(self):
        self.image_canvas.clear_vertical_line()
        self.status_var.set("Foveal center line cleared.")

    def _apply_vertical_line_x(self):
        if self.image_data is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        raw_value = self.fovea_x_entry_var.get().strip()
        if not raw_value:
            messagebox.showinfo("Missing coordinate", "Enter Center X first.")
            return
        try:
            x = int(raw_value)
        except ValueError:
            messagebox.showerror("Invalid coordinate", "Center X must be an integer.")
            return
        width = int(self.image_data.shape[1])
        height = int(self.image_data.shape[0])
        if x < 0 or x >= width:
            messagebox.showerror("Out of range", f"Center X must be between 0 and {width - 1}.")
            return

        # Save the foveal center line into the segmentation map so there is one
        # fovea marker source for the editor and MARKED output.
        if height >= 2:
            self._set_fovea_from_prediction(x)

        next_name = self._next_incomplete_boundary()
        if next_name is not None:
            self._set_active_boundary_target(next_name)

        self.status_var.set(f"Foveal center line set to x={x} and saved in the segmentation map.")

    def _set_drawing_locked(self, locked):
        self._drawing_locked = bool(locked)
        self._stop_fovea_repeat()

        self.vertical_mode_check.configure(state="disabled" if self._drawing_locked else "normal")
        self._set_fovea_controls_enabled(self.vertical_mode_var.get())

        if self._drawing_locked:
            self.vertical_mode_var.set(False)
            self.image_canvas.enable_vertical_line(False)
            # Keep boundary tracing active while fovea controls are locked.
            self.image_canvas.enable_line(True)
        else:
            live_fovea = self._fovea_live_edit_mode()
            self.image_canvas.enable_vertical_line(live_fovea)
            self.image_canvas.enable_line(not live_fovea)

    def _set_fovea_controls_enabled(self, enabled):
        """Enable/disable fovea-specific controls based on mode and lock state."""
        state = "normal" if (enabled and not self._drawing_locked) else "disabled"
        for widget in (
            self.fovea_minus_btn,
            self.fovea_x_entry,
            self.fovea_plus_btn,
            self.fovea_set_btn,
            self.fovea_reset_btn,
            self.fovea_clear_btn,
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
                getattr(self, "ai_backend_combo", None),
                getattr(self, "ai_settings_btn", None),
                getattr(self, "folder_segment_button", None),
                getattr(self, "batch_ai_button", None),
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
        if hasattr(self, "ai_backend_combo"):
            self.ai_backend_combo.configure(state="disabled" if self._segmenter_running else "readonly")
        if hasattr(self, "ai_settings_btn"):
            self.ai_settings_btn.state(["disabled"] if self._segmenter_running else ["!disabled"])
        self._update_batch_ai_button_state()

    def _clear_fovea_lock(self):
        """Remove saved fovea lock line, unlock controls, then show default center line."""
        self.boundary_traces.pop(FOVEA_BOUNDARY_NAME, None)
        if FOVEA_BOUNDARY_NAME in self.boundary_order:
            self.boundary_order.remove(FOVEA_BOUNDARY_NAME)

        self._set_drawing_locked(False)
        self.image_canvas.clear_vertical_line()
        self._rebuild_saved_overlays()
        self._refresh_trace_list()
        self.trace_detail_var.set("No saved boundary")

        if self.image_data is not None:
            self._center_vertical_line()
            self.status_var.set("Fovea lock cleared. Default center line restored.")
        else:
            self.status_var.set("Fovea lock cleared.")

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
        if self._has_saved_fovea_trace():
            self._set_fovea_from_prediction(x)
            return
        self.image_canvas.set_vertical_line_x(x)

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
        if self._has_saved_fovea_trace():
            self._set_fovea_from_prediction(next_x)
            return
        self.image_canvas.set_vertical_line_x(next_x)

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
            return

        pixels = _polyline_pixels(points)
        name = self.active_boundary or self._selected_boundary_name()
        self.active_trace_var.set(
            f"Active: {name} | {len(points)} vertices | {len(pixels)} pixel(s) on line"
        )

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
        self._update_saved_trace_summary(name)

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

    def _update_saved_trace_summary(self, name):
        trace = self.boundary_traces.get(name)
        if not trace:
            self.trace_detail_var.set("No saved boundary")
            return
        if name == FOVEA_BOUNDARY_NAME:
            x_value = self.fovea_x
            if x_value is None:
                points = trace.get("points", [])
                if points:
                    x_value = int(points[0][0])
            suffix = f" at x={int(x_value)}" if x_value is not None else ""
            self.trace_detail_var.set(
                f"{name}: done{suffix}; populated from the segmentation map."
            )
            return
        self.trace_detail_var.set(
            f"{name}: {len(trace['points'])} vertices, {len(trace['pixels'])} pixels saved."
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

    def _preprocessed_output_basepath(self, basename):
        return self._marked_output_basepath(basename)

    def _preprocessed_output_basepaths(self):
        return [
            self._preprocessed_output_basepath(LIGHT_PREPROCESSED_BASENAME),
            self._preprocessed_output_basepath(DARK_PREPROCESSED_BASENAME),
        ]

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

    def _build_marked_volumes(self):
        """Generate Light_MARKED and Dark_MARKED 8-bit Analyze volumes from traced boundaries.

        Creates two volumes in standard output format (test1 dimensions):
          - Light_MARKED: RPE and Fovea marked on a light background (230)
          - Dark_MARKED: All 6 boundaries + Fovea marked on a dark background

        All output volumes are automatically resized to standard format (current annotation height × 2133 width, 2 slices).
        If preprocessing was applied, the annotation crop stays as the coordinate reference.

        Returns:
            Tuple (light_volume, dark_volume): Two 3-D uint8 numpy arrays in standard format.
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

        # Standard MARKED format: 8-bit with base intensities capped at 230,
        # All traced boundaries on both LIGHT and DARK slices.
        light_slice = np.array(base_marked, copy=True)
        self._apply_mark_values(light_slice, COMMON_MARK_VALUES)
        self._apply_mark_values(light_slice, DARK_FIRST_SLICE_EXTRA_MARK_VALUES)

        dark_marked_slice = np.array(base_marked, copy=True)
        self._apply_mark_values(dark_marked_slice, COMMON_MARK_VALUES)
        # Apply all traced boundaries (including extra layers) to dark volume on all slices
        self._apply_mark_values(dark_marked_slice, DARK_FIRST_SLICE_EXTRA_MARK_VALUES)

        # Restore original traces if we modified them
        if original_traces is not None:
            self.boundary_traces = original_traces

        # Create volume with standard number of slices (always 2 in standard format)
        nslices = max(1, int(target_slices))
        light_volume = np.stack([light_slice] * nslices, axis=0).astype(np.uint8, copy=False)
        dark_volume = np.stack([dark_marked_slice] * nslices, axis=0).astype(np.uint8, copy=False)
        
        # Ensure output volumes are in standard format
        light_volume = _resize_to_standard_format(light_volume)
        dark_volume = _resize_to_standard_format(dark_volume)
        
        return light_volume, dark_volume

    def _source_image_for_original_save(self):
        """Return the 16-bit source image that corresponds to current annotations."""
        if (
            self._preprocessing_done
            and self._original_image_for_ai is not None
            and self._preprocessing_info
        ):
            source = np.asarray(self._original_image_for_ai)
            if source.ndim != 2:
                raise ValueError("Original Step 2 image must be 2-D.")

            x0 = int(self._preprocessing_info.get("crop_offset_x", 0))
            y0 = int(self._preprocessing_info.get("crop_offset_y", 0))
            crop_w = int(self._preprocessing_info.get("crop_w", self.image_data.shape[1]))
            crop_h = int(self._preprocessing_info.get("crop_h", self.image_data.shape[0]))
            y1 = min(y0 + crop_h, source.shape[0])
            x1 = min(x0 + crop_w, source.shape[1])
            return np.array(source[y0:y1, x0:x1], copy=True)

        return self.image_data

    def _save_light_dark_images(self, reference_shape=None):
        """Export unmarked LIGHT and DARK Analyze volumes from the current image.

        These outputs use the 16-bit image that corresponds to the current
        annotation coordinates. If AI preprocessing displays an 8-bit crop,
        save the same crop from the original image, then resize it to the
        saved MARKED/annotation volume shape.

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
        for base_path in self._preprocessed_output_basepaths():
            write_analyze(base_path, stack)
            saved_paths.append(base_path)
        return saved_paths

    def _save_marked_images(self, require_complete=False, prompt_on_incomplete=False):
        """Generate and save Light_MARKED and Dark_MARKED Analyze volumes.

        The MARKED volumes are 8-bit Analyze files with boundary pixels marked at
        specific intensity values per ImageJ macro conventions. Boundaries are
        rasterized at their specified line widths and mark values.

        Preprocessing inverse-scaling is applied if needed: if the image was
        preprocessed for AI, the boundaries are re-scaled to match the original
        image size before marking.

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

        light_marked, dark_marked = self._build_marked_volumes()
        saved_paths = []

        light_base_path = self._marked_output_basepath(LIGHT_MARKED_BASENAME)
        write_analyze(light_base_path, light_marked)
        saved_paths.append(light_base_path)

        dark_base_path = self._marked_output_basepath(DARK_MARKED_BASENAME)
        write_analyze(dark_base_path, dark_marked)
        saved_paths.append(dark_base_path)

        saved_paths.extend(self._save_light_dark_images(reference_shape=light_marked.shape))

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
        self._apply_mark_values_to_image(marked_slice, traces, DARK_FIRST_SLICE_EXTRA_MARK_VALUES)

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

    def _save_batch_result_state(self, tab_key):
        state = self._batch_result_states.get(tab_key)
        if not state:
            return None

        if tab_key == getattr(self, "_active_batch_result_tab", None):
            self._sync_active_batch_result_state()
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
        failures = []
        for tab_key in tab_keys:
            state = self._batch_result_states.get(tab_key)
            try:
                out_base = self._save_batch_result_state(tab_key)
                if out_base:
                    saved.append(out_base)
            except (OSError, ValueError, RuntimeError) as exc:
                name = os.path.basename((state or {}).get("input") or tab_key)
                failures.append(f"{name}: {exc}")

        if failures:
            messagebox.showerror("Save error", "Could not save some open tabs:\n" + "\n".join(failures[:8]))
        if saved:
            self.status_var.set(f"Saved {len(saved)} open batch tab(s).")
        return saved

    def _save_current_marked_image_button(self):
        if self.image_data is None:
            messagebox.showwarning("No image", "Load or select an image before saving a MARKED output.")
            return
        if not self.boundary_traces:
            messagebox.showinfo("Nothing to save", "Trace or load boundaries before saving a MARKED output.")
            return

        try:
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
                message_lines.extend(path + ".img" for path in saved[:12])
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
        dark_path = self._marked_output_basepath(DARK_MARKED_BASENAME) + ".img"
        message_lines = ["Saved MARKED images:", light_path, dark_path]
        message_lines.extend([path + ".img" for path in self._preprocessed_output_basepaths()])
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

    def _default_segmenter_output_dir(self):
        if self.current_file and self.current_file != "Step 1 output":
            root = os.path.dirname(self.current_file)
        else:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(root, "segmenter_output")

    def _selected_ai_backend(self):
        label = self.ai_backend_var.get().strip()
        return AI_BACKEND_BY_LABEL.get(label, AI_BACKEND_OCT_SEGMENTER)

    def _is_aidas_ai_backend(self):
        return self._selected_ai_backend() == AI_BACKEND_AIDAS

    def _on_ai_backend_changed(self, _event=None):
        if hasattr(self, "ai_backend_combo"):
            self.ai_backend_combo.set(AI_BACKEND_LABELS[self._selected_ai_backend()])
        self._update_ai_button_states()
        if self.image_data is not None:
            label = AI_BACKEND_LABELS[self._selected_ai_backend()]
            self.status_var.set(f"AI version set to {label}.")

    def _browse_segmenter_config(self):
        path = filedialog.askopenfilename(
            title="Select segmenter config",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.segmenter_config_var.set(path)

    def _browse_segmenter_model(self):
        path = filedialog.askopenfilename(
            title="Select segmenter model",
            filetypes=[("Keras model", "*.h5"), ("All files", "*.*")],
        )
        if path:
            self.segmenter_model_var.set(path)

    def _browse_aidas_model(self):
        kwargs = {}
        if os.path.isdir(self.ai_for_aidas_root):
            kwargs["initialdir"] = self.ai_for_aidas_root
        path = filedialog.askopenfilename(
            title="Select AI_ForAIDAS boundary model",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")],
            **kwargs,
        )
        if path:
            self.aidas_model_var.set(path)

    def _browse_aidas_vline_model(self):
        kwargs = {}
        if os.path.isdir(self.ai_for_aidas_root):
            kwargs["initialdir"] = self.ai_for_aidas_root
        path = filedialog.askopenfilename(
            title="Select AI_ForAIDAS fovea/vline model",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")],
            **kwargs,
        )
        if path:
            self.aidas_vline_model_var.set(path)


    def _browse_segmenter_output_dir(self):
        path = filedialog.askdirectory(title="Select segmentation output folder")
        if path:
            self.segmenter_output_var.set(path)

    def _segmenter_command(self):
        import sys
        
        env_name = self.segmenter_env_var.get().strip()
        conda_bin = shutil.which("conda") or os.environ.get("CONDA_EXE")

        # If user specified a conda env and conda is available, prefer `conda run -n <env> oct-segmenter`
        if env_name and conda_bin:
            return [conda_bin, "run", "-n", env_name, "--no-capture-output", "oct-segmenter"]

        # Prefer an `oct-segmenter` executable on PATH if present.
        oct_exec = shutil.which("oct-segmenter")
        if oct_exec:
            return [oct_exec]

        # When running as a frozen/bundled executable (pyinstaller, etc.),
        # `sys.executable` points to the app exe. Calling
        # `sys.executable -m oct_segmenter` will re-launch the app executable
        # (causing a new window) rather than a separate Python environment
        # that has the `oct_segmenter` package installed. Avoid that.
        if getattr(sys, "frozen", False):
            return None

        # Fallback: run as module when not frozen (developer machine with Python).
        return [sys.executable, "-m", "oct_segmenter"]

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

    def _app_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _aidas_external_python_command(self):
        """Return a Python command for running AI_ForAIDAS outside this process."""
        python_cmd = self.aidas_python_var.get().strip()
        if python_cmd:
            if os.path.isfile(python_cmd):
                return [python_cmd]
            return shlex.split(python_cmd, posix=(os.name != "nt"))

        path_python = self._find_torch_python_on_path()
        if path_python:
            return [path_python]

        env_name = self.aidas_env_var.get().strip()
        conda_bin = shutil.which("conda") or os.environ.get("CONDA_EXE")
        if env_name and conda_bin:
            return [conda_bin, "run", "-n", env_name, "--no-capture-output", "python"]

        return None

    def _candidate_path_pythons(self):
        candidates = []
        seen = set()

        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["where.exe", "python"],
                    capture_output=True,
                    text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    startupinfo=None,
                )
                for line in (result.stdout or "").splitlines():
                    path = line.strip()
                    if path and os.path.isfile(path):
                        norm = os.path.normcase(os.path.abspath(path))
                        if norm not in seen:
                            candidates.append(path)
                            seen.add(norm)
            except Exception:
                pass

        which_python = shutil.which("python")
        if which_python and os.path.isfile(which_python):
            norm = os.path.normcase(os.path.abspath(which_python))
            if norm not in seen:
                candidates.append(which_python)
                seen.add(norm)

        return candidates

    def _find_torch_python_on_path(self):
        """Return the first PATH python.exe that can import torch."""
        for python_path in self._candidate_path_pythons():
            if "windowsapps" in os.path.normcase(python_path):
                continue
            if self._python_command_has_torch([python_path]):
                return python_path
        return None

    def _python_command_has_torch(self, python_cmd):
        try:
            result = subprocess.run(
                python_cmd + ["-c", "import torch"],
                capture_output=True,
                text=True,
                cwd=self._app_root(),
                env=self._segmenter_subprocess_env(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return result.returncode == 0
        except Exception:
            return False

    def _missing_torch_message(self):
        env_name = self.aidas_env_var.get().strip() or "aidas-env"
        return (
            "AI_ForAIDAS requires PyTorch in the Python environment used to run the .pth model.\n\n"
            "Install it in the configured AI_ForAIDAS Conda environment:\n"
            f"  conda activate {env_name}\n"
            "  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n\n"
            "Or set AI Settings > AI_ForAIDAS > Python Executable to a python.exe that already has torch installed."
        )

    @staticmethod
    def _is_missing_torch_error(exc):
        text = str(exc).lower()
        return "requires pytorch" in text or "no module named 'torch'" in text or "no module named torch" in text

    def _run_aidas_external_prediction(
        self,
        image,
        model_path,
        vline_path,
        predict_fovea,
        device_name,
        output_dir,
        fallback_reason=None,
    ):
        base_cmd = self._aidas_external_python_command()
        if not base_cmd:
            raise RuntimeError(self._missing_torch_message())

        temp_dir = tempfile.mkdtemp(prefix="aidas_ai_for_aidas_")
        try:
            image_path = os.path.join(temp_dir, "step2_image.npy")
            result_path = os.path.join(temp_dir, "prediction.npz")
            np.save(image_path, image)

            cmd = base_cmd + [
                "-m",
                "aidas.ai_for_aidas_cli",
                "--image-npy",
                image_path,
                "--model",
                model_path,
                "--output-npz",
                result_path,
                "--device",
                device_name,
            ]
            if predict_fovea and vline_path:
                cmd.extend(["--vline-model", vline_path])
            else:
                cmd.append("--no-vline")

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = None
            if os.name == "nt":
                try:
                    si = subprocess.STARTUPINFO()
                    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    si.wShowWindow = subprocess.SW_HIDE
                    startupinfo = si
                except Exception:
                    startupinfo = None

            run_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self._app_root(),
                env=self._segmenter_subprocess_env(),
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
            stdout = (run_result.stdout or "").strip()
            stderr = (run_result.stderr or "").strip()
            log_content = (
                "AIDaS Step 2 AI_ForAIDAS External Segmentation\n"
                f"Fallback reason: {fallback_reason or '(configured external run)'}\n"
                f"Command: {subprocess.list2cmdline(cmd)}\n"
                f"Return code: {run_result.returncode}\n\n"
                f"STDOUT:\n{stdout or '(empty)'}\n\n"
                f"STDERR:\n{stderr or '(empty)'}\n"
            )
            log_path = self._write_segmenter_log_file(output_dir, log_content)

            if run_result.returncode != 0:
                details = stderr or stdout or "External Python returned a non-zero exit code."
                if "no module named torch" in details.lower():
                    details += "\n\n" + self._missing_torch_message()
                raise RuntimeError(f"{details}\n\nLog saved to:\n{log_path}")

            if not os.path.isfile(result_path):
                raise RuntimeError(f"External AI_ForAIDAS finished but did not write predictions.\n\nLog saved to:\n{log_path}")

            with np.load(result_path, allow_pickle=False) as npz:
                boundaries = np.array(npz["boundaries"], copy=True)
                fovea_values = np.array(npz["fovea_x"], copy=False)
                fovea_x = int(fovea_values[0]) if fovea_values.size and int(fovea_values[0]) >= 0 else None
                device_values = np.array(npz["device"], copy=False)
                device = str(device_values[0]) if device_values.size else "external"

            return {
                "boundaries": boundaries,
                "fovea_x": fovea_x,
                "device": device,
                "log_path": log_path,
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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
        """Convert unmarked LIGHT/DARK save data to signed 16-bit for Analyze.

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
        16-bit working copy so LIGHT/DARK saves never become 8-bit originals.
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

    def _segmenter_model_size(self):
        """Extract target input size (height, width) from model_config.json.

        Reads model_config.json located next to the selected .h5 model file
        to determine the expected input dimensions for the neural network.

        Returns:
            Tuple (target_h, target_w, config_path) if found and valid.
            None if model path not set, config file missing, or JSON invalid.
        """
        model_path = self.segmenter_model_var.get().strip()
        if not model_path:
            return None
        cfg_path = os.path.join(os.path.dirname(model_path), "model_config.json")
        if not os.path.isfile(cfg_path):
            return None

        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            target_h = int(cfg.get("image_height"))
            target_w = int(cfg.get("image_width"))
            if target_h <= 0 or target_w <= 0:
                return None
            return target_h, target_w, cfg_path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _center_crop_to_size(image, target_w, target_h):
        """Crop image to the exact target size without resizing.

        Width is center-cropped. Extra height is removed from the top only so
        the returned crop keeps the bottom target_h rows.
        """
        src_h, src_w = image.shape[:2]
        if src_w < target_w or src_h < target_h:
            raise ValueError(
                f"Image is {src_w}x{src_h}, smaller than AI input {target_w}x{target_h}; "
                "cannot center-crop without upsampling."
            )
        x0 = (src_w - target_w) // 2
        y0 = src_h - target_h
        cropped = np.array(image[y0:y0 + target_h, x0:x0 + target_w], copy=True)
        return cropped, x0, y0

    def _prepare_segmenter_input_image(self, auto_preprocess=False):
        image_u8 = self._image_uint8(self.image_data)
        source_h, source_w = image_u8.shape[:2]
        process_note = f"Using source size: {source_w}x{source_h}"
        self._preprocessing_info = None

        if not auto_preprocess:
            return image_u8, process_note

        model_size = self._segmenter_model_size()
        if model_size is None:
            raise RuntimeError(
                "Auto preprocess could not find model size. Expected model_config.json next to the selected .h5 model."
            )

        target_h, target_w, cfg_path = model_size
        cropped, crop_offset_x, crop_offset_y = self._center_crop_to_size(image_u8, target_w, target_h)
        crop_h, crop_w = cropped.shape[:2]

        # Store preprocessing info for later inverse transformation of AI predictions
        self._preprocessing_info = {
            "source_h": source_h,
            "source_w": source_w,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "crop_offset_x": crop_offset_x,
            "crop_offset_y": crop_offset_y,
            "target_h": target_h,
            "target_w": target_w,
        }
        
        process_note = (
            f"Auto preprocess: center-cropped {source_w}x{source_h} -> {crop_w}x{crop_h} "
            f"without resizing using {cfg_path}"
        )
        return cropped, process_note

    def _segmenter_input_tiff(self, auto_preprocess=False):
        image_for_segmenter, process_note = self._prepare_segmenter_input_image(auto_preprocess=auto_preprocess)
        temp_dir = tempfile.mkdtemp(prefix="aidas_segmenter_")
        temp_path = os.path.join(temp_dir, "step2_input.tiff")
        Image.fromarray(image_for_segmenter).save(temp_path)
        return temp_path, temp_dir, process_note, image_for_segmenter


    def _write_segmenter_log_file(self, output_dir, content):
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"segmentation_{ts}.log")
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return log_path

    def _append_segmenter_log(self, text):
        """Append a short entry to the in-memory and (if present) UI segmenter log."""
        if not hasattr(self, "_segmenter_log_lines"):
            self._segmenter_log_lines = []
        ts = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        entry = f"{ts} - {text}"
        self._segmenter_log_lines.append(entry)
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

    def _import_boundary_rows(self, data, source_label="predicted boundaries"):
        """Import six boundary rows into the Step 2 trace state."""
        traces, order = self._boundary_traces_from_rows(data, self.image_data.shape, source_label=source_label)

        self.boundary_traces.clear()
        self.boundary_order.clear()
        self.image_canvas.clear_active_line()
        self.image_canvas.clear_line_overlays()

        self.boundary_traces.update(traces)
        self.boundary_order.extend(order)
        for name in self.boundary_order:
            trace = self.boundary_traces[name]
            self.image_canvas.add_line_overlay(trace["points"], color=trace["color"], label=name)

        self._refresh_trace_list()
        if self.boundary_order:
            self._select_trace_by_name(self.boundary_order[0])
            self._update_saved_trace_summary(self.boundary_order[0])

        # Mark imported boundaries as complete to indicate AI has processed them
        for name in BOUNDARY_NAMES:
            if name in self.boundary_completion_vars:
                self.boundary_completion_vars[name].set(name in self.boundary_traces)
        self._refresh_boundary_lists()
        self._update_boundary_progress_bar()

    def _import_segmenter_boundaries(self, csv_path):
        """Import boundary traces from oct-segmenter CSV output."""
        try:
            data = np.loadtxt(csv_path, delimiter=",", dtype=float)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read predicted boundaries CSV: {exc}") from exc

        self._import_boundary_rows(data, source_label="Predicted CSV")

    def _image_matches_model_input(self):
        """Return True when current image already matches model input size."""
        if self.image_data is None:
            return False
        model_size = self._segmenter_model_size()
        if model_size is None:
            return False
        target_h, target_w, _cfg_path = model_size
        image_h, image_w = self.image_data.shape[:2]
        return int(image_h) == int(target_h) and int(image_w) == int(target_w)

    def _aidas_batch_model_ready(self):
        if self._segmenter_running:
            return False
        if not self._is_aidas_ai_backend():
            return False
        model_path = self.aidas_model_var.get().strip()
        return bool(model_path) and os.path.isfile(model_path)

    def _folder_ai_button_enabled(self):
        return not self._segmenter_running and self._is_aidas_ai_backend()

    def _batch_ai_button_enabled(self):
        if getattr(self, "batch_segmentation_panel", None) is not None:
            return False
        return not self._segmenter_running and self._is_aidas_ai_backend()

    def _update_batch_ai_button_state(self):
        if hasattr(self, "folder_segment_button"):
            if self._folder_ai_button_enabled():
                self.folder_segment_button.state(["!disabled"])
            else:
                self.folder_segment_button.state(["disabled"])
        if not hasattr(self, "batch_ai_button"):
            return
        if self._batch_ai_button_enabled():
            self.batch_ai_button.state(["!disabled"])
        else:
            self.batch_ai_button.state(["disabled"])

    def _update_ai_button_states(self):
        """Update AI button states based on preprocessing and input readiness.

        Three possible states:
          1. Preprocessing already done: Show "Remove Preprocess" (enabled) + enable AI
          2. Image matches model size: Show "Preprocess Not Needed" (disabled) + enable AI
          3. Need preprocessing: Show "Preprocess Image" (enabled) + disable AI

        This implements the smart preprocessing workflow where unnecessary preprocessing
        is skipped if the image already matches the model's expected input dimensions.
        """
        if not hasattr(self, "preprocess_button") or not hasattr(self, "segment_button"):
            return

        if self.image_data is None:
            self.preprocess_button.configure(text="Preprocess Image", command=self._preprocess_image_for_ai)
            self.preprocess_button.state(["disabled"])
            self.segment_button.state(["disabled"])
            self._update_batch_ai_button_state()
            return

        if self._is_aidas_ai_backend():
            self.preprocess_button.configure(text="Preprocess Not Needed", command=self._preprocess_image_for_ai)
            self.preprocess_button.state(["disabled"])
            self.segment_button.state(["!disabled"])
            self._update_batch_ai_button_state()
            return

        if self._preprocessing_done:
            self.preprocess_button.configure(text="Remove Preprocess", command=self._remove_preprocess)
            self.preprocess_button.state(["!disabled"])
            self.segment_button.state(["!disabled"])
            self._update_batch_ai_button_state()
            return

        if self._image_matches_model_input():
            self.preprocess_button.configure(text="Preprocess Not Needed", command=self._preprocess_image_for_ai)
            self.preprocess_button.state(["disabled"])
            self.segment_button.state(["!disabled"])
            self._update_batch_ai_button_state()
            return

        self.preprocess_button.configure(text="Preprocess Image", command=self._preprocess_image_for_ai)
        self.preprocess_button.state(["!disabled"])
        self.segment_button.state(["disabled"])
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
          - Disables vertical line mode and fovea controls
          - Starts animated progress bar showing activity
          - Disables line drawing

        After segmentation (running=False):
          - Re-enables controls
          - Stops progress animation
          - Updates AI button states based on preprocessing
          - Re-enables line drawing if appropriate

        Args:
            running: Boolean indicating if segmentation is in progress.
            status_message: Optional status text to display to user.
        """
        self._segmenter_running = bool(running)
        button_state = ["disabled"] if running else ["!disabled"]
        self.preprocess_button.state(button_state)
        self.segment_button.state(button_state)
        if hasattr(self, "ai_backend_combo"):
            self.ai_backend_combo.configure(state="disabled" if running else "readonly")
        if hasattr(self, "ai_settings_btn"):
            self.ai_settings_btn.state(["disabled"] if running else ["!disabled"])
        if running:
            if hasattr(self, "folder_segment_button"):
                self.folder_segment_button.state(["disabled"])
            if hasattr(self, "batch_ai_button"):
                self.batch_ai_button.state(["disabled"])
        else:
            self._update_batch_ai_button_state()
        for widget_name in (
            "image_browser_reset_btn",
            "image_browser_folder_btn",
            "image_browser_refresh_btn",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.state(["disabled"] if running else ["!disabled"])
        
        # Disable/enable boundary listboxes during segmentation
        listbox_state = "disabled" if running else "normal"
        if hasattr(self, "boundary_incomplete_listbox"):
            self.boundary_incomplete_listbox.configure(state=listbox_state)
        if hasattr(self, "boundary_completed_listbox"):
            self.boundary_completed_listbox.configure(state=listbox_state)
        if hasattr(self, "directory_image_listbox"):
            self.directory_image_listbox.configure(state=listbox_state)
        
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
            # If vertical mode is currently on, turn it off during segmentation
            if self._fovea_live_edit_mode():
                self.vertical_mode_var.set(False)
                self._on_vertical_mode_toggled()
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
                # Then properly update fovea controls state based on vertical mode
                self._set_fovea_controls_enabled(self.vertical_mode_var.get())
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
                if animate_progress:
                    self._animate_progress_bar()
        else:
            # Stop animation and show actual boundary count
            self._cancel_progress_animation()
            if hasattr(self, "segmenter_progress") and restore_boundary_progress:
                self._update_boundary_progress_bar()
            # Update button states after segmentation finishes
            self._update_ai_button_states()
            if self.image_data is None and not getattr(self, "_directory_image_paths_cache", []):
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

    def _set_segmenter_progress_value(self, value, maximum=None):
        if not hasattr(self, "segmenter_progress"):
            return
        if maximum is not None:
            self.segmenter_progress.configure(maximum=max(1, int(maximum)), mode="determinate")
        max_value = float(self.segmenter_progress.cget("maximum") or 1)
        self.segmenter_progress["value"] = max(0, min(float(value), max_value))

    def _animate_progress_bar(self):
        """Gradually fill progress bar during segmentation to show activity."""
        if not self._segmenter_running or not hasattr(self, "segmenter_progress"):
            self._progress_animation_job = None
            return
        
        current = float(self.segmenter_progress["value"])
        maximum = float(self.segmenter_progress.cget("maximum") or len(BOUNDARY_NAMES))
        # Gradually increase but slow down as it approaches the max.
        if current < maximum:
            increment = max(0.02, (maximum - current) * 0.08)
            self.segmenter_progress["value"] = min(current + increment, maximum)
            self._progress_animation_job = self.after(500, self._animate_progress_bar)
        else:
            # Keep it near max until segmentation finishes.
            self._progress_animation_job = self.after(500, self._animate_progress_bar)

    def _run_segmenter_worker(self, cmd, output_dir):
        try:
            # Prevent spawning a visible console window on Windows when running
            # console-based CLIs like `conda` or `oct-segmenter`.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = None
            if os.name == "nt":
                try:
                    si = subprocess.STARTUPINFO()
                    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    si.wShowWindow = subprocess.SW_HIDE
                    startupinfo = si
                except Exception:
                    startupinfo = None

            run_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._segmenter_subprocess_env(),
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
            stdout = (run_result.stdout or "").strip()
            stderr = (run_result.stderr or "").strip()
            run_log = (
                "AIDaS Step 2 Neural Segmentation\n"
                f"Command: {subprocess.list2cmdline(cmd)}\n"
                f"Return code: {run_result.returncode}\n\n"
                f"STDOUT:\n{stdout or '(empty)'}\n\n"
                f"STDERR:\n{stderr or '(empty)'}\n"
            )
            log_path = self._write_segmenter_log_file(output_dir, run_log)
            result = {
                "returncode": run_result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "log_path": log_path,
            }
        except Exception as exc:  # pragma: no cover - defensive runtime error handling
            result = {
                "exception": str(exc),
                "returncode": -1,
                "stdout": "",
                "stderr": "",
                "log_path": None,
            }

        self.after(0, lambda: self._on_segmenter_worker_done(result))

    def _on_segmenter_worker_done(self, result):
        input_tiff = getattr(self, "_segmenter_input_tiff_path", None)
        temp_dir = getattr(self, "_segmenter_temp_dir", None)
        output_dir = getattr(self, "_segmenter_output_dir", None)

        try:
            log_path = result.get("log_path")
            if result.get("exception"):
                self._append_segmenter_log(f"Segmentation execution failed: {result['exception']}")
                messagebox.showerror("Segmentation failed", result["exception"])
                self.status_var.set("Neural segmentation failed.")
                return

            if log_path:
                self._append_segmenter_log(f"Saved run log: {log_path}")

            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            if stdout:
                self._append_segmenter_log("STDOUT:\n" + stdout)
            if stderr:
                self._append_segmenter_log("STDERR:\n" + stderr)

            if result.get("returncode", 1) != 0:
                details = stderr or stdout or "Unknown error"
                message = details
                if log_path:
                    message += f"\n\nLog saved to:\n{log_path}"
                messagebox.showerror("Segmentation failed", message)
                self.status_var.set("Neural segmentation failed.")
                return

            labeled_dir = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_tiff))[0]}_labeled")
            boundaries_csv = os.path.join(labeled_dir, "gs_boundaries.csv")
            if not os.path.isfile(boundaries_csv):
                message = (
                    "Segmentation finished but gs_boundaries.csv was not found.\n"
                    "Check that graph_search is enabled in the config and try again."
                )
                if log_path:
                    message += f"\n\nLog saved to:\n{log_path}"
                messagebox.showerror("Segmentation output missing", message)
                self.status_var.set("Segmentation finished, but no boundary CSV was found.")
                return

            self._import_segmenter_boundaries(boundaries_csv)
            self.status_var.set(f"Segmentation completed and boundaries loaded from {boundaries_csv}")
            self._append_segmenter_log(f"Loaded predicted boundaries from: {boundaries_csv}")
            msg = f"Loaded predicted boundaries from:\n{boundaries_csv}"
            if log_path:
                msg += f"\n\nLog saved to:\n{log_path}"
            messagebox.showinfo("Segmentation complete", msg)
        finally:
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            self._set_segmentation_running(False)
            # Ensure listboxes are updated after controls are re-enabled
            try:
                incomplete_names = self._incomplete_boundary_names()
                completed_names = self._completed_boundary_names()
                self._populate_boundary_listbox(self.boundary_incomplete_listbox, incomplete_names, None)
                self._populate_boundary_listbox(self.boundary_completed_listbox, completed_names, None)
                # Select first completed boundary if any
                if completed_names:
                    self.boundary_completed_listbox.selection_clear(0, "end")
                    self.boundary_completed_listbox.selection_set(0)
                    self.boundary_completed_listbox.see(0)
                    # Explicitly set active boundary to first completed
                    self._set_active_boundary_target(completed_names[0])
                else:
                    # Otherwise select first incomplete
                    if incomplete_names:
                        self.boundary_incomplete_listbox.selection_clear(0, "end")
                        self.boundary_incomplete_listbox.selection_set(0)
                        self.boundary_incomplete_listbox.see(0)
            except Exception:
                pass

    def _preprocess_image_for_ai(self):
        """Preprocess image for AI segmentation with exact center crop.

        Preprocessing pipeline:
          1. Backup original image and file path for later restore
          2. Convert current image to 8-bit if needed
          3. Center-crop to exact model input dimensions (no downsampling or resizing)
          4. Display preprocessed image on canvas
          5. Store preprocessing metadata
          6. Update button states to show "Remove Preprocess" and enable AI

        Original image is preserved and can be restored with _remove_preprocess().
        Preprocessing info is saved for later use (e.g., rescaling AI predictions).

        The model size comes from model_config.json next to the .h5 model file.
        """
        if self.image_data is None:
            messagebox.showwarning("No image", "Load an image before preprocessing.")
            return

        model_size = self._segmenter_model_size()
        if model_size is None:
            messagebox.showerror(
                "Missing model config",
                "Could not find model size. Expected model_config.json next to the selected .h5 model."
            )
            return

        target_h, target_w, cfg_path = model_size
        try:
            original_image = np.array(self.image_data, copy=True)
            original_path = self.current_file
            image_u8 = self._image_uint8(self.image_data)
            cropped, crop_offset_x, crop_offset_y = self._center_crop_to_size(image_u8, target_w, target_h)
            
            # Calculate crop offsets for inverse transformation
            source_h, source_w = image_u8.shape[:2]
            crop_h, crop_w = cropped.shape[:2]
            
            # Display the preprocessed image (note: _show_image clears preprocessing state)
            display_path = self.current_file or "Step 2 input (preprocessed)"
            self._show_image(cropped, display_path)
            
            # Restore preprocessing info and state after _show_image clears them
            self._preprocessing_info = {
                "source_h": source_h,
                "source_w": source_w,
                "crop_h": crop_h,
                "crop_w": crop_w,
                "crop_offset_x": crop_offset_x,
                "crop_offset_y": crop_offset_y,
                "target_h": target_h,
                "target_w": target_w,
            }
            self._last_auto_preprocessed_image = np.array(cropped, copy=True)
            self._original_image_for_ai = original_image
            self._original_file_for_ai = original_path
            self._preprocessing_done = True
            
            self._update_ai_button_states()
            self.status_var.set(
                f"Image preprocessed: cropped {source_w}x{source_h} → {crop_w}x{crop_h}, "
                "no resizing/downsampling. Ready for AI segmentation."
            )
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Preprocess failed", str(exc))

    def _remove_preprocess(self):
        """Remove preprocessing and restore original image shape and data.

        Restores the image to its original state before preprocessing was applied.
        This is a lossless operation—all original pixels are preserved exactly.

        Updates:
          - Displays the original image on canvas
          - Clears preprocessing metadata
          - Updates AI button states to show "Preprocess Image" (enabled)
          - Resets preprocessing_done flag
        """
        if self._original_image_for_ai is None:
            return

        restored_image = np.array(self._original_image_for_ai, copy=True)
        restored_path = self._original_file_for_ai or self.current_file or "Original image"

        self._show_image(restored_image, restored_path)
        
        self._update_ai_button_states()
        self.status_var.set("Preprocessing removed. Original image restored.")

    def _run_neural_segmentation(self, auto_preprocess=False):
        """Launch the selected AI backend."""
        if self._is_aidas_ai_backend():
            self._run_aidas_neural_segmentation()
            return
        self._run_oct_segmenter_segmentation(auto_preprocess=auto_preprocess)

    def _run_aidas_batch_segmentation(self, image_paths=None, manual_fovea_by_path=None):
        """Run AI_ForAIDAS predictions for multiple images and preview them in tabs."""
        if self._segmenter_running:
            messagebox.showinfo("Please wait", "Segmentation is already running.")
            return

        model_path = self.aidas_model_var.get().strip()
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
            norm = os.path.normcase(os.path.abspath(dedupe_path))
            if norm in seen:
                continue
            seen.add(norm)
            image_paths.append(path)

        manual_fovea_by_key = None
        if manual_fovea_by_path is not None:
            manual_fovea_by_key = {
                self._image_pair_key(path): (None if x is None else int(x))
                for path, x in manual_fovea_by_path.items()
            }
            predict_fovea = False
            vline_path = None
        else:
            predict_fovea = bool(self.aidas_predict_fovea_var.get())
            vline_path = self.aidas_vline_model_var.get().strip()
            if predict_fovea and vline_path and not os.path.isfile(vline_path):
                self._append_segmenter_log(f"AI_ForAIDAS vline model not found; batch fovea prediction skipped: {vline_path}")
                vline_path = None
            if not predict_fovea:
                vline_path = None

        output_dir = self.segmenter_output_var.get().strip() or self._default_segmenter_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        self.segmenter_output_var.set(output_dir)

        device_name = self.aidas_device_var.get().strip() or AI_DEVICE_OPTIONS[0]
        self._append_segmenter_log(
            f"Starting AI_ForAIDAS batch for {len(image_paths)} image(s); model={model_path}; device={device_name}"
        )
        self._set_segmentation_running(
            True,
            status_message=f"Running AI_ForAIDAS batch on {len(image_paths)} image(s)...",
            progress_max=len(image_paths),
            animate_progress=False,
        )

        worker = threading.Thread(
            target=self._run_aidas_batch_segmenter_worker,
            args=(image_paths, model_path, vline_path, predict_fovea, device_name, output_dir, manual_fovea_by_key),
            daemon=True,
        )
        worker.start()

    def _run_aidas_batch_segmenter_worker(
        self,
        image_paths,
        model_path,
        vline_path,
        predict_fovea,
        device_name,
        output_dir,
        manual_fovea_by_key=None,
    ):
        results = []
        failures = []
        log_lines = [
            "AIDaS Step 2 AI_ForAIDAS Batch Segmentation",
            f"Boundary model: {model_path}",
            f"VLine model: {vline_path or '(not used)'}",
            f"Manual fovea lines: {'yes' if manual_fovea_by_key is not None else 'no'}",
            f"Requested device: {device_name}",
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

        try:
            try:
                from aidas.ai_for_aidas_inference import AIForAIDASPredictor

                predictor = AIForAIDASPredictor(
                    boundary_model_path=model_path,
                    vline_model_path=vline_path,
                    predict_fovea=predict_fovea and bool(vline_path),
                    device_name=device_name,
                )
                device = str(predictor.device)

                def run_prediction_with_result(image_for_ai):
                    prediction = predictor.predict(image_for_ai)
                    return {
                        "boundaries": prediction.boundaries,
                        "fovea_x": prediction.fovea_x,
                        "device": prediction.device,
                    }

            except Exception as inner_exc:
                if not self._is_missing_torch_error(inner_exc):
                    raise
                fallback_reason_text = str(inner_exc)
                log_lines.append(f"Internal PyTorch unavailable; using external Python fallback: {fallback_reason_text}")

                def run_prediction_with_result(image_for_ai):
                    return self._run_aidas_external_prediction(
                        image_for_ai,
                        model_path,
                        vline_path,
                        predict_fovea,
                        device_name,
                        output_dir,
                        fallback_reason=fallback_reason_text,
                    )

            for index, path in enumerate(image_paths, start=1):
                report_status(index, path)
                try:
                    image, template, _source_was_8bit = self._read_image_for_annotation(path)
                    model_input_uses_stored_y = template is not None
                    image_for_ai = np.array(image, copy=True)
                    if model_input_uses_stored_y:
                        image_for_ai = np.ascontiguousarray(np.flipud(image_for_ai))

                    prediction = run_prediction_with_result(image_for_ai)
                    if prediction.get("device"):
                        device = prediction["device"]
                    fovea_x = prediction.get("fovea_x")
                    manual_fovea = False
                    if manual_fovea_by_key is not None:
                        manual_fovea = True
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
                        source = "manual" if manual_fovea else "predicted"
                        log_lines.append(f"  Fovea x ({source}): {int(fovea_x)}")
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
            log_path = self._write_segmenter_log_file(output_dir, "\n".join(log_lines) + "\n")
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
            self._set_segmenter_progress_value(completed + failed, total)
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
            else:
                messagebox.showinfo("Batch segmentation complete", "\n".join(message_lines))
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
                for name in BOUNDARY_NAMES:
                    trace = traces.get(name)
                    if trace:
                        canvas.add_line_overlay(trace["points"], color=trace.get("color"), label=name)
                fovea_x = item.get("fovea_x")
                if fovea_x is not None:
                    fovea_trace = self._vertical_line_trace(int(fovea_x), data.shape[0])
                    traces[FOVEA_BOUNDARY_NAME] = fovea_trace
                    if FOVEA_BOUNDARY_NAME not in order:
                        order.append(FOVEA_BOUNDARY_NAME)
                    canvas.add_line_overlay(fovea_trace["points"], color=fovea_trace.get("color"), label=FOVEA_BOUNDARY_NAME)
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
            self._show_single_image_canvas()
            if self._single_editor_state is not None:
                self._load_editor_state(self._single_editor_state, self.single_image_canvas)
            if self.image_data is None:
                self.image_info_var.set("No image loaded")
            else:
                filename = (
                    os.path.basename(self.current_file)
                    if self.current_file and self.current_file != "Step 1 output"
                    else "Step 1 output"
                )
                self.image_info_var.set(
                    f"{filename} | Size: {self.image_data.shape[1]} x {self.image_data.shape[0]} px | "
                    f"Type: {self.image_data.dtype}"
                )
            self.status_var.set("Batch result tabs closed.")

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
            out_base = self._save_batch_result_state(tab_key)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Save error", f"Could not save {name}:\n{exc}")
            return False

        if out_base:
            self.status_var.set(f"Saved {out_base}.img")
        return True

    def _run_oct_segmenter_segmentation(self, auto_preprocess=False):
        """Launch neural network segmentation in background thread.

        Validates inputs, prepares a TIFF file for the segmenter tool, and
        runs the oct-segmenter command via subprocess. The segmentation happens
        in a daemon thread to keep the UI responsive. Results are automatically
        imported when complete.

        Steps:
          1. Check segmentation is not already running and image is loaded
          2. Validate config (.json) and model (.h5) file paths exist
          3. Determine which image to segment (preprocessed, already-sized, or error)
          4. Create temp TIFF in temp directory for segmenter input
          5. Build command-line arguments for oct-segmenter predict
          6. Lock UI (disable buttons, show progress animation)
          7. Start daemon thread with _run_segmenter_worker
          8. Worker reports back via _on_segmenter_worker_done callback

        Args:
            auto_preprocess: Not currently used; left for future auto-preprocess feature.
        """
        if self._segmenter_running:
            messagebox.showinfo("Please wait", "Segmentation is already running.")
            return
        if self.image_data is None:
            messagebox.showwarning("No image", "Load an image before running neural segmentation.")
            return

        config_path = self.segmenter_config_var.get().strip()
        model_path = self.segmenter_model_var.get().strip()
        output_dir = self.segmenter_output_var.get().strip() or self._default_segmenter_output_dir()

        if not os.path.isfile(config_path):
            messagebox.showerror("Missing config", f"Config file not found:\n{config_path}")
            return
        if not os.path.isfile(model_path):
            messagebox.showerror("Missing model", f"Model file not found:\n{model_path}")
            return

        os.makedirs(output_dir, exist_ok=True)
        self.segmenter_output_var.set(output_dir)

        try:
            # Use preprocessed image when available.
            if self._preprocessing_done and self._last_auto_preprocessed_image is not None:
                processed_image = self._last_auto_preprocessed_image
                process_note = "Using preprocessed image."
            elif self._image_matches_model_input():
                processed_image = self._image_uint8(self.image_data)
                process_note = "Input already matches model size; preprocessing skipped."
            else:
                messagebox.showinfo(
                    "Preprocess required",
                    "Preprocess Image first, or load an image that already matches the model input size.",
                )
                return
            
            # Create temp TIFF for segmenter input
            temp_dir = tempfile.mkdtemp(prefix="aidas_segmenter_")
            temp_path = os.path.join(temp_dir, "step2_input.tiff")
            Image.fromarray(processed_image).save(temp_path)
            input_tiff = temp_path
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Preprocess failed", str(exc))
            return

        base_cmd = self._segmenter_command()
        if not base_cmd:
            # Could not determine an external oct-segmenter command to run.
            shutil.rmtree(temp_dir, ignore_errors=True)
            messagebox.showerror(
                "Segmenter unavailable",
                (
                    "Could not find an 'oct-segmenter' executable or a usable conda installation on this machine.\n\n"
                    "When running AIDaS as a bundled executable, the embedded Python cannot run\n"
                    "the 'oct_segmenter' module via '-m' without a separate Python interpreter.\n\n"
                    "Please install 'oct-segmenter' on the target machine (pip/conda), ensure 'oct-segmenter'\n"
                    "is on PATH, or set a valid conda environment in AI Settings."
                ),
            )
            self._set_segmentation_running(False)
            return

        cmd = base_cmd + [
            "predict",
            "-c",
            config_path,
            "-m",
            model_path,
            "-i",
            input_tiff,
            "-o",
            output_dir,
        ]
        cmd_display = subprocess.list2cmdline(cmd)
        self._append_segmenter_log(process_note)
        self._append_segmenter_log(f"Running command: {cmd_display}")

        self._segmenter_input_tiff_path = input_tiff
        self._segmenter_temp_dir = temp_dir
        self._segmenter_output_dir = output_dir

        self._set_segmentation_running(True, status_message="Running neural segmentation...")
        worker = threading.Thread(target=self._run_segmenter_worker, args=(cmd, output_dir), daemon=True)
        worker.start()

    def _run_aidas_neural_segmentation(self):
        """Run the AI_ForAIDAS PyTorch backend in a background thread."""
        if self._segmenter_running:
            messagebox.showinfo("Please wait", "Segmentation is already running.")
            return
        if self.image_data is None:
            messagebox.showwarning("No image", "Load an image before running neural segmentation.")
            return

        model_path = self.aidas_model_var.get().strip()
        if not os.path.isfile(model_path):
            messagebox.showerror("Missing model", f"AI_ForAIDAS boundary model not found:\n{model_path}")
            return

        predict_fovea = bool(self.aidas_predict_fovea_var.get())
        vline_path = self.aidas_vline_model_var.get().strip()
        if predict_fovea and vline_path and not os.path.isfile(vline_path):
            self._append_segmenter_log(f"AI_ForAIDAS vline model not found; fovea prediction skipped: {vline_path}")
            vline_path = None
        if not predict_fovea:
            vline_path = None

        output_dir = self.segmenter_output_var.get().strip() or self._default_segmenter_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        self.segmenter_output_var.set(output_dir)

        model_input_uses_stored_y = self._aidas_should_use_stored_analyze_y()
        image_for_ai = np.array(self.image_data, copy=True)
        if model_input_uses_stored_y:
            image_for_ai = np.ascontiguousarray(np.flipud(image_for_ai))
        device_name = self.aidas_device_var.get().strip() or AI_DEVICE_OPTIONS[0]
        height, width = self.image_data.shape[:2]
        self._append_segmenter_log(
            f"Running AI_ForAIDAS on {width}x{height}; model={model_path}; device={device_name}"
        )
        if model_input_uses_stored_y:
            self._append_segmenter_log(
                "AI_ForAIDAS Analyze compatibility: using stored .img vertical orientation to match app.py."
            )

        self._set_segmentation_running(True, status_message="Running AI_ForAIDAS segmentation...")
        worker = threading.Thread(
            target=self._run_aidas_segmenter_worker,
            args=(
                image_for_ai,
                model_path,
                vline_path,
                predict_fovea,
                device_name,
                output_dir,
                model_input_uses_stored_y,
            ),
            daemon=True,
        )
        worker.start()

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

    def _run_aidas_segmenter_worker(
        self,
        image,
        model_path,
        vline_path,
        predict_fovea,
        device_name,
        output_dir,
        model_input_uses_stored_y=False,
    ):
        try:
            try:
                from aidas.ai_for_aidas_inference import predict_boundaries_and_fovea

                prediction = predict_boundaries_and_fovea(
                    image,
                    boundary_model_path=model_path,
                    vline_model_path=vline_path,
                    predict_fovea=predict_fovea and bool(vline_path),
                    device_name=device_name,
                )
                log_content = (
                    "AIDaS Step 2 AI_ForAIDAS Segmentation\n"
                    f"Boundary model: {prediction.boundary_model_path}\n"
                    f"VLine model: {prediction.vline_model_path or '(not used)'}\n"
                    f"Device: {prediction.device}\n"
                    f"Image shape: {image.shape[1]}x{image.shape[0]}\n"
                    f"Boundary rows: {prediction.boundaries.shape}\n"
                    f"Fovea x: {prediction.fovea_x if prediction.fovea_x is not None else '(not predicted)'}\n"
                )
                log_path = self._write_segmenter_log_file(output_dir, log_content)
                result = {
                    "boundaries": self._aidas_boundaries_from_model_to_display(
                        prediction.boundaries,
                        image.shape[0],
                        model_input_uses_stored_y,
                    ),
                    "fovea_x": prediction.fovea_x,
                    "device": prediction.device,
                    "log_path": log_path,
                }
            except Exception as inner_exc:
                if not self._is_missing_torch_error(inner_exc):
                    raise
                result = self._run_aidas_external_prediction(
                    image,
                    model_path,
                    vline_path,
                    predict_fovea,
                    device_name,
                    output_dir,
                    fallback_reason=str(inner_exc),
                )
                result["boundaries"] = self._aidas_boundaries_from_model_to_display(
                    result["boundaries"],
                    image.shape[0],
                    model_input_uses_stored_y,
                )
        except Exception as exc:  # pragma: no cover - defensive runtime error handling
            result = {
                "exception": str(exc),
                "boundaries": None,
                "fovea_x": None,
                "device": None,
                "log_path": None,
            }

        self.after(0, lambda: self._on_aidas_segmenter_worker_done(result))

    def _on_aidas_segmenter_worker_done(self, result):
        predicted_fovea_x = None
        try:
            if result.get("exception"):
                self._append_segmenter_log(f"AI_ForAIDAS segmentation failed: {result['exception']}")
                messagebox.showerror("Segmentation failed", result["exception"])
                self.status_var.set("AI_ForAIDAS segmentation failed.")
                return

            log_path = result.get("log_path")
            if log_path:
                self._append_segmenter_log(f"Saved run log: {log_path}")

            self._import_boundary_rows(result["boundaries"], source_label="AI_ForAIDAS prediction")

            predicted_fovea_x = result.get("fovea_x")
            if predicted_fovea_x is not None:
                self._set_fovea_from_prediction(predicted_fovea_x)

            device = result.get("device") or "unknown"
            self.status_var.set(f"AI_ForAIDAS segmentation completed on {device}.")
            self._append_segmenter_log("Loaded AI_ForAIDAS predicted boundaries into Step 2.")
            msg = "Loaded AI_ForAIDAS predicted boundaries."
            if predicted_fovea_x is not None:
                msg += f"\nPredicted foveal center: x={int(predicted_fovea_x)}"
            if log_path:
                msg += f"\n\nLog saved to:\n{log_path}"
            messagebox.showinfo("Segmentation complete", msg)
        finally:
            self._set_segmentation_running(False)
            try:
                incomplete_names = self._incomplete_boundary_names()
                completed_names = self._completed_boundary_names()
                self._populate_boundary_listbox(self.boundary_incomplete_listbox, incomplete_names, None)
                self._populate_boundary_listbox(self.boundary_completed_listbox, completed_names, None)
                if predicted_fovea_x is not None and FOVEA_BOUNDARY_NAME in self.boundary_traces:
                    self._select_trace_by_name(FOVEA_BOUNDARY_NAME)
                    self._update_saved_trace_summary(FOVEA_BOUNDARY_NAME)
                elif completed_names:
                    self.boundary_completed_listbox.selection_clear(0, "end")
                    self.boundary_completed_listbox.selection_set(0)
                    self.boundary_completed_listbox.see(0)
                    self._set_active_boundary_target(completed_names[0])
                elif incomplete_names:
                    self.boundary_incomplete_listbox.selection_clear(0, "end")
                    self.boundary_incomplete_listbox.selection_set(0)
                    self.boundary_incomplete_listbox.see(0)
            except Exception:
                pass
            if not result.get("exception"):
                device = result.get("device") or "unknown"
                self.status_var.set(f"AI_ForAIDAS segmentation completed on {device}.")

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
