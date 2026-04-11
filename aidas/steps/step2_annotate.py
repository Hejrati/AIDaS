"""Step 2 — Trace boundary lines over OCT images.

This panel provides an ImageJ-style polyline tracing workflow for marking
retinal boundaries and exporting the exact pixel coordinates along each line.
"""

import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image

from aidas.image_canvas import ImageCanvas
from aidas.utils.io_utils import read_analyze, read_tiff


BOUNDARY_PRESETS = [
    ("RPE", "#ffb703"),
    ("ELM", "#00b4d8"),
    ("ONL-OPL", "#ef476f"),
    ("INL-IPL", "#8ac926"),
    ("GCL-RNFL", "#fb8500"),
    ("RNFL-Vitreous", "#8338ec"),
]
BOUNDARY_NAMES = [name for name, _ in BOUNDARY_PRESETS]
BOUNDARY_COLORS = {name: color for name, color in BOUNDARY_PRESETS}
TRACE_EXPORT_SUFFIX = "_step2_boundaries.csv"
SEGMENTER_BOUNDARY_ORDER = ["RNFL-Vitreous", "GCL-RNFL", "INL-IPL", "ONL-OPL", "ELM", "RPE"]
FOVEA_BOUNDARY_NAME = "Fovea-Center"


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


class Step2Frame(ttk.Frame):
    """GUI panel for tracing boundary lines and exporting pixel coordinates."""

    def __init__(self, parent, preferences=None, source_step=None):
        super().__init__(parent)

        self.preferences = preferences
        self.source_step = source_step

        self.current_file = None
        self.image_data = None
        self.active_boundary = None
        self.boundary_traces = {}
        self.boundary_order = []
        self.boundary_completion_vars = {}
        self.fovea_x = None
        self._segmenter_running = False
        self._drawing_locked = False
        self._updating_fovea_entry = False
        self._syncing_boundary_selection = False
        self._fovea_repeat_job = None
        self._fovea_repeat_dir = 0
        self._fovea_repeat_ticks = 0
        self.boundary_completion_vars = {name: tk.BooleanVar(value=False) for name in BOUNDARY_NAMES}

        app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        segmenter_root = os.path.join(app_root, "OCT Segmenter")
        self.segmenter_default_config = os.path.join(segmenter_root, "config.json")
        self.segmenter_default_model = os.path.join(segmenter_root, "Model", "human_OCT.h5")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        left_outer = ttk.Frame(main)
        left_outer.pack(side="left", fill="y")

        ctrl_canvas = tk.Canvas(left_outer, highlightthickness=0, bd=0)
        ctrl_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=ctrl_canvas.yview)
        self.ctrl = ttk.Frame(ctrl_canvas)
        self.ctrl.bind(
            "<Configure>",
            lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")),
        )
        ctrl_window = ctrl_canvas.create_window((0, 0), window=self.ctrl, anchor="nw")
        ctrl_canvas.bind(
            "<Configure>",
            lambda e: ctrl_canvas.itemconfigure(ctrl_window, width=e.width),
        )
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_canvas.pack(side="left", fill="both", expand=True)
        ctrl_scroll.pack(side="right", fill="y")

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        info_frame = ttk.Frame(right, relief="solid", borderwidth=1)
        info_frame.pack(fill="x", padx=2, pady=2)
        self.image_info_var = tk.StringVar(value="No image loaded")
        info_label = ttk.Label(
            info_frame,
            textvariable=self.image_info_var,
            font=("", 10, "bold"),
            padding=8,
            anchor="w",
        )
        info_label.pack(fill="x")

        self.image_canvas = ImageCanvas(
            right,
            on_mouse_move=self._on_mouse_moved,
            on_line_change=self._on_active_line_changed,
            on_vertical_line_change=self._on_vertical_line_changed,
        )
        self.image_canvas.enable_line(True)
        self.image_canvas.enable_roi(False)
        self.image_canvas.enable_vertical_line(False)
        self.image_canvas.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(
            value="Ready — load a Step 1 result or any OCT image, then trace boundaries with left-clicks."
        )
        ttk.Label(right, textvariable=self.status_var, relief="sunken", anchor="w", padding=3).pack(
            side="bottom",
            fill="x",
        )

        self._build_controls()

    # ═══════════════════════════════════════════════════════════════════════
    #  UI construction
    # ═══════════════════════════════════════════════════════════════════════
    def _build_controls(self):
        """Construct and lay out all left-side control widgets.

        Controls are placed in titled LabelFrames and arranged vertically
        inside a scrollable canvas to keep the UI compact.
        """
        pad = dict(fill="x", padx=(14, 8))

        load = ttk.LabelFrame(self.ctrl, text="Image Source", padding=3)
        load.pack(**pad, pady=5)

        ttk.Button(load, text="Open Image...", command=self._open_image).pack(fill="x", pady=(0, 2))
        self.step1_button = ttk.Button(load, text="Load from Step 1", command=self._load_from_step1)
        self.step1_button.pack(fill="x")
        if self.source_step is None:
            self.step1_button.state(["disabled"])
        else:
            # Start a watcher to enable the button only when Step 1 has a processed (cropped) image
            self._step1_watcher_active = True
            self.after(100, self._update_step1_button_state)

        self.source_label_var = tk.StringVar(value="No source selected")
        ttk.Label(load, textvariable=self.source_label_var, wraplength=240, justify="left").pack(
            fill="x",
            pady=(4, 0),
        )

        ttk.Label(load, text="Step 2 input is always loaded as 8-bit.", wraplength=240, justify="left").pack(
            fill="x",
            pady=(4, 0),
        )

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        segmenter = ttk.LabelFrame(self.ctrl, text="AI Segmentation", padding=3)
        segmenter.pack(**pad, pady=2)

        self.segmenter_config_var = tk.StringVar(value=self.segmenter_default_config)
        self.segmenter_model_var = tk.StringVar(value=self.segmenter_default_model)
        self.segmenter_output_var = tk.StringVar(value=self._default_segmenter_output_dir())
        self.segmenter_env_var = tk.StringVar(value="oct-segmenter-env")

        env_row = ttk.Frame(segmenter)
        env_row.pack(fill="x", pady=(0, 4))
        ttk.Label(env_row, text="Conda env:").pack(side="left")
        ttk.Entry(env_row, textvariable=self.segmenter_env_var, width=22).pack(side="right", fill="x", expand=True)

        ttk.Label(segmenter, text="Config (.json):").pack(anchor="w")
        seg_cfg_row = ttk.Frame(segmenter)
        seg_cfg_row.pack(fill="x", pady=(0, 2))
        ttk.Entry(seg_cfg_row, textvariable=self.segmenter_config_var).pack(side="left", fill="x", expand=True)
        ttk.Button(seg_cfg_row, text="...", width=3, command=self._browse_segmenter_config).pack(side="right")

        ttk.Label(segmenter, text="Model (.h5):").pack(anchor="w")
        seg_model_row = ttk.Frame(segmenter)
        seg_model_row.pack(fill="x", pady=(0, 2))
        ttk.Entry(seg_model_row, textvariable=self.segmenter_model_var).pack(side="left", fill="x", expand=True)
        ttk.Button(seg_model_row, text="...", width=3, command=self._browse_segmenter_model).pack(side="right")

        ttk.Label(segmenter, text="Output folder:").pack(anchor="w")
        seg_out_row = ttk.Frame(segmenter)
        seg_out_row.pack(fill="x", pady=(0, 4))
        ttk.Entry(seg_out_row, textvariable=self.segmenter_output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(seg_out_row, text="...", width=3, command=self._browse_segmenter_output_dir).pack(side="right")

        self.segment_button = ttk.Button(
            segmenter,
            text="AI Segment (no pre-process)",
            command=self._run_neural_segmentation,
        )
        self.segment_button.pack(fill="x")
        self.segment_auto_button = ttk.Button(
            segmenter,
            text="AI Segment with Auto Pre-process",
            command=lambda: self._run_neural_segmentation(auto_preprocess=True),
            
        )
        self.segment_auto_button.pack(fill="x", pady=(3, 0))

        self.segmenter_progress_var = tk.StringVar(value="Idle")
        ttk.Label(segmenter, textvariable=self.segmenter_progress_var).pack(anchor="w", pady=(6, 0))
        self.segmenter_progress = ttk.Progressbar(segmenter, mode="indeterminate")
        self.segmenter_progress.pack(fill="x", pady=(2, 0))

        log_box = ttk.LabelFrame(segmenter, text="Segmentation Logs", padding=2)
        log_box.pack(fill="both", expand=True, pady=(6, 0))
        self.segmenter_log_text = tk.Text(log_box, height=8, wrap="word")
        seg_log_scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.segmenter_log_text.yview)
        self.segmenter_log_text.configure(yscrollcommand=seg_log_scroll.set)
        self.segmenter_log_text.pack(side="left", fill="both", expand=True)
        seg_log_scroll.pack(side="right", fill="y")

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        boundary = ttk.LabelFrame(self.ctrl, text="Boundary Tracing", padding=3)
        boundary.pack(**pad, pady=2)

        self.active_trace_var = tk.StringVar(value="No active boundary")
        ttk.Label(boundary, textvariable=self.active_trace_var, wraplength=240, justify="left").pack(fill="x", pady=(0, 4))

        workflow = ttk.LabelFrame(boundary, text="Boundary Progress", padding=3)
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

        self.clear_all_traces_btn = ttk.Button(workflow, text="Clear All Traces", command=self._clear_all_traces)
        self.clear_all_traces_btn.pack(fill="x", pady=(4, 0))

        self.boundary_workflow_status_var = tk.StringVar(value="Select a boundary to make it active.")
        ttk.Label(workflow, textvariable=self.boundary_workflow_status_var, wraplength=240, justify="left").pack(
            fill="x",
            pady=(4, 0),
        )

        fovea = ttk.LabelFrame(boundary, text="Foveal Center Line", padding=3)
        fovea.pack(fill="x", pady=(4, 2))

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

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        summary = ttk.LabelFrame(self.ctrl, text="Saved Boundaries", padding=3)
        summary.pack(**pad, pady=2)

        self.trace_listbox = tk.Listbox(summary, height=8, selectmode="browse")
        trace_scroll = ttk.Scrollbar(summary, orient="vertical", command=self.trace_listbox.yview)
        self.trace_listbox.configure(yscrollcommand=trace_scroll.set)
        self.trace_listbox.pack(side="left", fill="both", expand=True)
        trace_scroll.pack(side="right", fill="y")
        self.trace_listbox.bind("<<ListboxSelect>>", self._on_saved_boundary_selected)

        self.trace_detail_var = tk.StringVar(value="No saved boundary")
        ttk.Label(summary, textvariable=self.trace_detail_var, wraplength=240, justify="left").pack(
            fill="x",
            pady=(4, 0),
        )

        saved_buttons = ttk.Frame(self.ctrl)
        saved_buttons.pack(**pad, pady=(2, 0))
        ttk.Button(saved_buttons, text="Delete Selected", command=self._delete_selected_boundary).pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 2),
        )
        ttk.Button(saved_buttons, text="Export CSV", command=self._export_csv, state="disabled").pack(
            side="right",
            expand=True,
            fill="x",
            padx=(2, 0),
        )

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        help_box = ttk.LabelFrame(self.ctrl, text="How to Trace", padding=3)
        help_box.pack(**pad, pady=(2, 6))
        ttk.Label(
            help_box,
            text=(
                "1. Load an image.\n"
                "2. Pick a boundary name.\n"
                "3. Left-click points along the boundary.\n"
                "4. Press Finish Boundary to save all pixels on the line.\n"
                "5. Enable Vertical line mode to place/drag the foveal center line."
            ),
            justify="left",
        ).pack(anchor="w")

        self._refresh_boundary_lists()

    # ═══════════════════════════════════════════════════════════════════════
    #  Image loading
    # ═══════════════════════════════════════════════════════════════════════
    def _open_image(self):
        """Show open-file dialog and load the selected image.

        The function auto-detects Analyze/TIFF/standard image formats
        by extension and converts the image according to the current
        import bit-depth selection.
        """
        path = filedialog.askopenfilename(
            title="Select an OCT image",
            filetypes=[
                ("Supported images", "*.tif *.tiff *.hdr *.img *.png *.jpg *.jpeg"),
                ("TIFF", "*.tif *.tiff"),
                ("Analyze 7.5", "*.hdr *.img"),
                ("All files", "*.*"),
            ],
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

        If Step 1 has a `processed_image` (cropped/scaled), prefer that;
        otherwise fall back to the raw image. Applies current bit-depth.
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
        img = self._coerce_to_8bit(np.array(image, copy=True))
        self._show_image(img, display_path)

    def load_external_image(self, image, source_path=None):
        """Load an externally supplied image into Step 2.

        Used by Step 1 auto-sync after crop so the latest .img-like result is
        immediately available in this panel.
        """
        if image is None:
            return
        display_path = source_path or "Step 1 output"
        img = self._coerce_to_8bit(np.array(image, copy=True))
        self._show_image(img, display_path)

    def _update_step1_button_state(self):
        """Enable the Step 1 load button only when `source_step.processed_image` exists.

        This method re-schedules itself via `after` while the frame exists.
        """
        try:
            has = self.source_step is not None and getattr(self.source_step, "processed_image", None) is not None
        except Exception:
            has = False

        if has:
            self.step1_button.state(["!disabled"])
        else:
            self.step1_button.state(["disabled"])

        # continue polling as long as this frame is mapped
        if getattr(self, "_step1_watcher_active", False):
            self.after(500, self._update_step1_button_state)

    def _load_image_from_path(self, path):
        """Read an image file from disk and return a 2-D numpy array.

        Supports Analyze (.hdr/.img), TIFF stacks, and standard images
        (PNG/JPEG). For multi-frame TIFFs a single slice is returned
        (the first frame).
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in {".hdr", ".img"}:
            data = read_analyze(path)
        elif ext in {".tif", ".tiff"}:
            data = read_tiff(path)
        else:
            data = np.array(Image.open(path).convert("L"))

        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2:
            raise ValueError("Step 2 expects a 2-D grayscale image or a 2-D slice from a stack.")
        return self._coerce_to_8bit(data)

    def _coerce_to_8bit(self, data):
        if data.dtype == np.uint8:
            return np.array(data, copy=False)
        scaled = data.astype(np.float64)
        lo = float(np.min(scaled))
        hi = float(np.max(scaled))
        if hi > lo:
            scaled = (scaled - lo) / (hi - lo) * 255.0
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def _show_image(self, image, path):
        self.current_file = path
        self.image_data = image
        self.active_boundary = None
        self.boundary_traces.clear()
        self.boundary_order.clear()
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
        self.status_var.set(
            "Image loaded. Left-click to place points only after selecting an incomplete boundary."
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Boundary tracing actions
    # ═══════════════════════════════════════════════════════════════════════
    def _selected_boundary_name(self):
        if getattr(self, "boundary_incomplete_listbox", None) is not None:
            selection = self.boundary_incomplete_listbox.curselection()
            if selection:
                index = int(selection[0])
                incomplete_names = self._incomplete_boundary_names()
                if 0 <= index < len(incomplete_names):
                    return incomplete_names[index]
        if self.active_boundary in BOUNDARY_NAMES:
            return self.active_boundary
        next_name = self._next_incomplete_boundary()
        return next_name or BOUNDARY_NAMES[0]

    def _completed_boundary_names(self):
        return [name for name in BOUNDARY_NAMES if self.boundary_completion_vars.get(name) and self.boundary_completion_vars[name].get()]

    def _incomplete_boundary_names(self):
        completed = set(self._completed_boundary_names())
        return [name for name in BOUNDARY_NAMES if name not in completed]

    def _start_boundary_for_name(self, name, auto_advance=False):
        if self.image_data is None:
            messagebox.showwarning("No image", "Load an image before tracing boundaries.")
            return

        if name not in BOUNDARY_NAMES:
            return

        if self.vertical_mode_var.get():
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

    def _set_active_boundary_target(self, name):
        if name not in BOUNDARY_NAMES:
            return
        if self.image_data is None:
            return

        if self.vertical_mode_var.get():
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
        self._sync_boundary_canvas_state()

    def _selected_active_boundary_name(self):
        if self.active_boundary in BOUNDARY_NAMES:
            return self.active_boundary
        incomplete = self._incomplete_boundary_names()
        if getattr(self, "boundary_incomplete_listbox", None) is not None:
            selection = self.boundary_incomplete_listbox.curselection()
            if selection:
                index = int(selection[0])
                if 0 <= index < len(incomplete):
                    return incomplete[index]
        completed = self._completed_boundary_names()
        if getattr(self, "boundary_completed_listbox", None) is not None:
            selection = self.boundary_completed_listbox.curselection()
            if selection:
                index = int(selection[0])
                if 0 <= index < len(completed):
                    return completed[index]
        return None

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
        active_name = self.active_boundary if self.active_boundary in BOUNDARY_NAMES else None
        drawing_enabled = (
            active_name is not None
            and active_name in self._incomplete_boundary_names()
            and not self.vertical_mode_var.get()
        )
        self.image_canvas.enable_line(drawing_enabled)
        self.image_canvas.enable_vertical_line(self.vertical_mode_var.get())

    def _start_boundary(self):
        if self.image_data is None:
            return

        name = self._selected_boundary_name()
        self._start_boundary_for_name(name)

    def _undo_point(self):
        self.image_canvas.undo_active_line_vertex()

    def _clear_active_trace(self):
        self.image_canvas.clear_active_line()
        self._update_active_trace_summary()
        self.status_var.set("Current unfinished trace cleared.")

    def _finish_boundary(self):
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
            self.status_var.set(f"Saved '{name}'. All preset boundaries are complete.")

    def _revert_boundary(self):
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
        selection = self.trace_listbox.curselection()
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
        self.image_canvas.clear_line_overlays()
        for name in self.boundary_order:
            trace = self.boundary_traces.get(name)
            if trace:
                self.image_canvas.add_line_overlay(trace["points"], color=trace["color"], label=name)

    def _reset_boundary_completion(self):
        for var in self.boundary_completion_vars.values():
            var.set(False)
        self._refresh_boundary_lists()

    def _mark_boundary_complete(self, name):
        if name in self.boundary_completion_vars:
            self.boundary_completion_vars[name].set(True)

        next_name = self._next_incomplete_boundary(name)
        if next_name is not None:
            self._refresh_boundary_lists(select_incomplete_name=next_name)
        else:
            self._refresh_boundary_lists(select_completed_name=name)
        return next_name

    def _next_incomplete_boundary(self, current_name=None):
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
        if self._drawing_locked:
            self.vertical_mode_var.set(False)
            return
        enabled = self.vertical_mode_var.get()
        self._set_fovea_controls_enabled(enabled)
        self.image_canvas.enable_vertical_line(enabled)
        self.image_canvas.enable_line(not enabled)
        if enabled:
            if self.image_data is not None:
                self._center_vertical_line()
                self.status_var.set("Vertical line mode enabled. Foveal center line set to image center.")
            else:
                self.status_var.set("Vertical line mode enabled. Load an image to place the foveal center line.")
        else:
            # When fovea mode is off, clear fovea line/data and saved fovea boundary.
            self.boundary_traces.pop(FOVEA_BOUNDARY_NAME, None)
            if FOVEA_BOUNDARY_NAME in self.boundary_order:
                self.boundary_order.remove(FOVEA_BOUNDARY_NAME)
            self.image_canvas.clear_vertical_line()
            self._rebuild_saved_overlays()
            self._refresh_trace_list()
            self.trace_detail_var.set("No saved boundary")
            self.status_var.set("Boundary tracing mode enabled. Left-click to place boundary points.")

    def _refresh_boundary_lists(self, select_incomplete_name=None, select_completed_name=None):
        if getattr(self, "boundary_incomplete_listbox", None) is None:
            return

        incomplete_names = self._incomplete_boundary_names()
        completed_names = self._completed_boundary_names()
        selected_name = select_incomplete_name if select_incomplete_name is not None else select_completed_name

        self._syncing_boundary_selection = True
        self.boundary_incomplete_listbox.selection_clear(0, "end")
        self.boundary_completed_listbox.selection_clear(0, "end")
        self.boundary_incomplete_listbox.delete(0, "end")
        for name in incomplete_names:
            active = "▶" if name == selected_name else " "
            self.boundary_incomplete_listbox.insert("end", f"{active} {name}")

        self.boundary_completed_listbox.delete(0, "end")
        for name in completed_names:
            active = "▶" if name == selected_name else " "
            self.boundary_completed_listbox.insert("end", f"{active} {name}")

        current = selected_name
        if current is None:
            current = self.active_boundary if self.active_boundary in incomplete_names else None
        if current is None:
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

        self.boundary_workflow_status_var.set(
            f"Incomplete: {len(incomplete_names)} | Completed: {len(completed_names)}/{len(BOUNDARY_NAMES)}"
        )
        self._update_boundary_action_buttons()
        self._sync_boundary_canvas_state()

    def _on_boundary_incomplete_selected(self, _event):
        if self._syncing_boundary_selection:
            return
        if getattr(self, "boundary_incomplete_listbox", None) is None:
            return
        selection = self.boundary_incomplete_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        incomplete_names = self._incomplete_boundary_names()
        if 0 <= index < len(incomplete_names):
            name = incomplete_names[index]
            self.boundary_completed_listbox.selection_clear(0, "end")
            self._set_active_boundary_target(name)

    def _on_boundary_completed_selected(self, _event):
        if self._syncing_boundary_selection:
            return
        if getattr(self, "boundary_completed_listbox", None) is None:
            return
        selection = self.boundary_completed_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        completed_names = self._completed_boundary_names()
        if 0 <= index < len(completed_names):
            name = completed_names[index]
            self.boundary_incomplete_listbox.selection_clear(0, "end")
            self._set_active_boundary_target(name)

    def _center_vertical_line(self):
        if self.image_data is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        width = int(self.image_data.shape[1])
        center_x = width // 2
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
        self.image_canvas.set_vertical_line_x(x)

        # Save the foveal center line as a boundary entry so it appears in the list.
        if height >= 2:
            points = [(x, 0), (x, height - 1)]
            pixels = _polyline_pixels(points)
            color = self._boundary_color(FOVEA_BOUNDARY_NAME)
            self.boundary_traces[FOVEA_BOUNDARY_NAME] = {
                "points": points,
                "pixels": pixels,
                "color": color,
            }
            if FOVEA_BOUNDARY_NAME not in self.boundary_order:
                self.boundary_order.append(FOVEA_BOUNDARY_NAME)
            self._rebuild_saved_overlays()
            self._refresh_trace_list()
            self._select_trace_by_name(FOVEA_BOUNDARY_NAME)
            self._update_saved_trace_summary(FOVEA_BOUNDARY_NAME)

        self._set_drawing_locked(True)

        next_name = self._next_incomplete_boundary()
        if next_name is not None:
            self._set_active_boundary_target(next_name)

        self.status_var.set(f"Foveal center line set to x={x} and saved to boundary list.")

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
            self.image_canvas.enable_vertical_line(self.vertical_mode_var.get())
            self.image_canvas.enable_line(not self.vertical_mode_var.get())

    def _set_fovea_controls_enabled(self, enabled):
        """Enable/disable fovea-specific controls based on mode and lock state."""
        state = "normal" if (enabled and not self._drawing_locked) else "disabled"
        for widget in (
            self.fovea_minus_btn,
            self.fovea_x_entry,
            self.fovea_plus_btn,
            self.fovea_set_btn,
            self.fovea_reset_btn,
        ):
            widget.configure(state=state)

        # Clear should remain available while locked so the user can unlock.
        clear_state = "normal" if (self._drawing_locked or (enabled and not self._drawing_locked)) else "disabled"
        self.fovea_clear_btn.configure(state=clear_state)

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
        self.trace_listbox.delete(0, "end")
        for name in self.boundary_order:
            trace = self.boundary_traces.get(name)
            if not trace:
                continue
            self.trace_listbox.insert(
                "end",
                f"{name} — {len(trace['points'])} vertices, {len(trace['pixels'])} pixels",
            )

    def _select_trace_by_name(self, name):
        if name not in self.boundary_order:
            return
        index = self.boundary_order.index(name)
        self.trace_listbox.selection_clear(0, "end")
        self.trace_listbox.selection_set(index)
        self.trace_listbox.see(index)

    def _on_saved_boundary_selected(self, _event):
        selection = self.trace_listbox.curselection()
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
        self.trace_detail_var.set(
            f"{name}: {len(trace['points'])} vertices, {len(trace['pixels'])} pixels saved."
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Neural segmentation
    # ═══════════════════════════════════════════════════════════════════════
    def _default_segmenter_output_dir(self):
        if self.current_file and self.current_file != "Step 1 output":
            root = os.path.dirname(self.current_file)
        else:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(root, "segmenter_output")

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

    def _browse_segmenter_output_dir(self):
        path = filedialog.askdirectory(title="Select segmentation output folder")
        if path:
            self.segmenter_output_var.set(path)

    def _segmenter_command(self):
        env_name = self.segmenter_env_var.get().strip()
        conda_bin = shutil.which("conda") or os.environ.get("CONDA_EXE")
        if env_name and conda_bin:
            return [conda_bin, "run", "-n", env_name, "--no-capture-output", "oct-segmenter"]
        if shutil.which("oct-segmenter"):
            return ["oct-segmenter"]
        return [sys.executable, "-m", "oct_segmenter"]

    def _image_uint8(self, image):
        if image.dtype == np.uint8:
            return np.array(image, copy=False)
        arr = image.astype(np.float64)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _segmenter_model_size(self):
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
    def _crop_to_aspect(image, target_w, target_h):
        """Center-crop image to match the target aspect ratio before resizing."""
        src_h, src_w = image.shape[:2]
        target_ratio = target_w / target_h
        src_ratio = src_w / src_h

        if abs(src_ratio - target_ratio) < 1e-6:
            return np.array(image, copy=True)

        if src_ratio > target_ratio:
            crop_w = int(round(src_h * target_ratio))
            crop_w = max(1, min(crop_w, src_w))
            x0 = (src_w - crop_w) // 2
            return np.array(image[:, x0:x0 + crop_w], copy=True)

        crop_h = int(round(src_w / target_ratio))
        crop_h = max(1, min(crop_h, src_h))
        y0 = (src_h - crop_h) // 2
        return np.array(image[y0:y0 + crop_h, :], copy=True)

    def _prepare_segmenter_input_image(self, auto_preprocess=False):
        image_u8 = self._image_uint8(self.image_data)
        source_h, source_w = image_u8.shape[:2]
        process_note = f"Using source size: {source_w}x{source_h}"

        if not auto_preprocess:
            return image_u8, process_note

        model_size = self._segmenter_model_size()
        if model_size is None:
            raise RuntimeError(
                "Auto preprocess could not find model size. Expected model_config.json next to the selected .h5 model."
            )

        target_h, target_w, cfg_path = model_size
        cropped = self._crop_to_aspect(image_u8, target_w, target_h)
        crop_h, crop_w = cropped.shape[:2]
        resized = np.array(
            Image.fromarray(cropped).resize((target_w, target_h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        process_note = (
            f"Auto preprocess: cropped {source_w}x{source_h} -> {crop_w}x{crop_h}, "
            f"then resized to {target_w}x{target_h} using {cfg_path}"
        )
        return resized, process_note

    def _segmenter_input_tiff(self, auto_preprocess=False):
        image_for_segmenter, process_note = self._prepare_segmenter_input_image(auto_preprocess=auto_preprocess)
        temp_dir = tempfile.mkdtemp(prefix="aidas_segmenter_")
        temp_path = os.path.join(temp_dir, "step2_input.tiff")
        Image.fromarray(image_for_segmenter).save(temp_path)
        return temp_path, temp_dir, process_note, image_for_segmenter

    def _append_segmenter_log(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.segmenter_log_text.insert("end", f"[{timestamp}] {message}\n")
        self.segmenter_log_text.see("end")

    def _write_segmenter_log_file(self, output_dir, content):
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"segmentation_{ts}.log")
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return log_path

    def _import_segmenter_boundaries(self, csv_path):
        try:
            data = np.loadtxt(csv_path, delimiter=",", dtype=float)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read predicted boundaries CSV: {exc}") from exc

        data = np.atleast_2d(data)
        if data.shape[0] < len(SEGMENTER_BOUNDARY_ORDER):
            raise RuntimeError(
                f"Predicted CSV has {data.shape[0]} rows; expected at least {len(SEGMENTER_BOUNDARY_ORDER)} rows."
            )

        self.boundary_traces.clear()
        self.boundary_order.clear()
        self.image_canvas.clear_active_line()
        self.image_canvas.clear_line_overlays()

        width = int(self.image_data.shape[1])
        height = int(self.image_data.shape[0])
        max_x = min(width, data.shape[1])

        for row_idx, name in enumerate(SEGMENTER_BOUNDARY_ORDER):
            y_row = np.rint(data[row_idx, :max_x]).astype(int)
            y_row = np.clip(y_row, 0, height - 1)
            points = [(x, int(y_row[x])) for x in range(max_x)]
            if len(points) < 2:
                continue
            pixels = _polyline_pixels(points)
            color = self._boundary_color(name)
            self.boundary_traces[name] = {"points": points, "pixels": pixels, "color": color}
            self.boundary_order.append(name)
            self.image_canvas.add_line_overlay(points, color=color, label=name)

        self._refresh_trace_list()
        if self.boundary_order:
            self._select_trace_by_name(self.boundary_order[0])
            self._update_saved_trace_summary(self.boundary_order[0])

    def _set_segmentation_running(self, running, status_message=None):
        self._segmenter_running = bool(running)
        if running:
            self.segment_button.state(["disabled"])
            self.segment_auto_button.state(["disabled"])
            self.segmenter_progress_var.set("Running segmentation...")
            self.segmenter_progress.start(12)
        else:
            self.segment_button.state(["!disabled"])
            self.segment_auto_button.state(["!disabled"])
            self.segmenter_progress.stop()
            self.segmenter_progress_var.set("Idle")

        if status_message:
            self.status_var.set(status_message)

    def _run_segmenter_worker(self, cmd, output_dir):
        try:
            run_result = subprocess.run(cmd, capture_output=True, text=True)
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

    def _run_neural_segmentation(self, auto_preprocess=False):
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
            input_tiff, temp_dir, process_note, processed_image = self._segmenter_input_tiff(
                auto_preprocess=True
            )
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Preprocess failed", str(exc))
            return

        display_path = self.current_file or "Step 2 input"
        self._show_image(processed_image, display_path)

        cmd = self._segmenter_command() + [
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

    # ═══════════════════════════════════════════════════════════════════════
    #  Export
    # ═══════════════════════════════════════════════════════════════════════
    def _boundary_row_for_export(self, trace, image_width):
        pixels = trace.get("pixels") or []
        if not pixels or image_width <= 0:
            return [""] * image_width

        ordered_samples = {}
        for x, y in pixels:
            x = int(x)
            y = int(y)
            if x not in ordered_samples:
                ordered_samples[x] = y

        x_values = sorted(ordered_samples.keys())
        xs = np.array(x_values, dtype=np.float64)
        ys = np.array([ordered_samples[x] for x in x_values], dtype=np.float64)

        if xs.size == 1:
            return [int(round(ys[0])) for _ in range(image_width)]

        x_grid = np.arange(image_width, dtype=np.float64)
        row = np.interp(x_grid, xs, ys, left=ys[0], right=ys[-1])
        return np.rint(row).astype(int).tolist()

    def _export_csv(self):
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
