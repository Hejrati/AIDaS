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
IMPORT_BIT_DEPTH_CHOICES = ["8-bit", "16-bit"]
SEGMENTER_BOUNDARY_ORDER = ["RNFL-Vitreous", "GCL-RNFL", "INL-IPL", "ONL-OPL", "ELM", "RPE"]


def _bresenham_line(start, end):
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
        self.fovea_x = None
        self._segmenter_running = False

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

        # Supported file types are shown in the dialog; files open automatically
        bitdepth_row = ttk.Frame(load)
        bitdepth_row.pack(side="bottom", fill="x", pady=(0, 4))
        ttk.Label(bitdepth_row, text="Image Bit Depth:").pack(side="left")
        self.import_bitdepth_var = tk.StringVar(value="8-bit")
        # Radio-buttons styled as toggle buttons (indicatoroff) to act like flip-flop
        rb_frame = ttk.Frame(bitdepth_row)
        rb_frame.pack(side="left", padx=(6, 0))
        ttk.Radiobutton(
            rb_frame,
            text="8-bit",
            variable=self.import_bitdepth_var,
            value="8-bit",
            command=lambda: self._apply_import_bitdepth("8-bit"),
        ).pack(side="left", padx=(0, 4))
        ttk.Radiobutton(
            rb_frame,
            text="16-bit",
            variable=self.import_bitdepth_var,
            value="16-bit",
            command=lambda: self._apply_import_bitdepth("16-bit"),
        ).pack(side="left")

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

        # Segmentation actions disabled in prototype — buttons left visible but inactive
        self.segment_button = ttk.Button(
            segmenter,
            text="Segment Image (Neural Net)",
            command=self._run_neural_segmentation,
            state="disabled",
        )
        self.segment_button.pack(fill="x")
        self.segment_auto_button = ttk.Button(
            segmenter,
            text="Auto Process + Segment (AI)",
            command=lambda: self._run_neural_segmentation(auto_preprocess=True),
            state="disabled",
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

        ttk.Label(boundary, text="Active boundary:").pack(anchor="w")
        self.boundary_var = tk.StringVar(value=BOUNDARY_NAMES[0])
        self.boundary_combo = ttk.Combobox(
            boundary,
            textvariable=self.boundary_var,
            values=BOUNDARY_NAMES,
            state="normal",
        )
        self.boundary_combo.pack(fill="x", pady=(0, 4))

        self.active_trace_var = tk.StringVar(value="No active boundary")
        ttk.Label(boundary, textvariable=self.active_trace_var, wraplength=240, justify="left").pack(
            fill="x",
            pady=(0, 4),
        )

        trace_buttons = ttk.Frame(boundary)
        trace_buttons.pack(fill="x", pady=(2, 0))
        ttk.Button(trace_buttons, text="Start / Reset", command=self._start_boundary).pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 2),
        )
        ttk.Button(trace_buttons, text="Undo Point", command=self._undo_point).pack(
            side="right",
            expand=True,
            fill="x",
            padx=(2, 0),
        )

        trace_buttons_2 = ttk.Frame(boundary)
        trace_buttons_2.pack(fill="x", pady=(2, 0))
        ttk.Button(trace_buttons_2, text="Finish Boundary", command=self._finish_boundary).pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 2),
        )
        ttk.Button(trace_buttons_2, text="Clear Active", command=self._clear_active_trace).pack(
            side="right",
            expand=True,
            fill="x",
            padx=(2, 0),
        )

        ttk.Button(boundary, text="Clear All Traces", command=self._clear_all_traces).pack(fill="x", pady=(4, 0))

        fovea = ttk.LabelFrame(self.ctrl, text="Foveal Center Line", padding=3)
        fovea.pack(**pad, pady=(4, 2))

        self.vertical_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            fovea,
            text="Vertical line mode (click/drag on image)",
            variable=self.vertical_mode_var,
            command=self._on_vertical_mode_toggled,
        ).pack(anchor="w")

        self.fovea_line_var = tk.StringVar(value="Fovea line: not set")
        ttk.Label(fovea, textvariable=self.fovea_line_var, wraplength=240, justify="left").pack(
            fill="x",
            pady=(2, 4),
        )

        coord_row = ttk.Frame(fovea)
        coord_row.pack(fill="x", pady=(0, 4))
        ttk.Label(coord_row, text="Center X:").pack(side="left")
        self.fovea_x_entry_var = tk.StringVar(value="")
        self.fovea_x_entry = ttk.Entry(coord_row, textvariable=self.fovea_x_entry_var, width=10)
        self.fovea_x_entry.pack(side="left", padx=(6, 4), fill="x", expand=True)
        self.fovea_x_entry.bind("<Return>", lambda _e: self._apply_vertical_line_x())
        ttk.Button(coord_row, text="Set", command=self._apply_vertical_line_x).pack(side="right")

        fovea_buttons = ttk.Frame(fovea)
        fovea_buttons.pack(fill="x")
        ttk.Button(fovea_buttons, text="Center", command=self._center_vertical_line).pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 2),
        )
        ttk.Button(fovea_buttons, text="Clear", command=self._clear_vertical_line).pack(
            side="right",
            expand=True,
            fill="x",
            padx=(2, 0),
        )

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
        ttk.Button(saved_buttons, text="Export CSV", command=self._export_csv).pack(
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

    # ═══════════════════════════════════════════════════════════════════════
    #  Image loading
    # ═══════════════════════════════════════════════════════════════════════
    def _open_image(self):
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
            image = self._load_image_from_path(path, bit_depth=self.import_bitdepth_var.get())
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Open error", str(exc))
            return
        self._show_image(image, path)

    def _load_from_step1(self):
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
        img = np.array(image, copy=True)
        img = self._coerce_import_depth(img, self.import_bitdepth_var.get())
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

    def _load_image_from_path(self, path, bit_depth="Auto"):
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
        # Apply current import bit depth selection when loading
        return self._coerce_import_depth(data, bit_depth)

    def _apply_import_bitdepth(self, mode=None):
        """Convert the currently-displayed image to 8-bit or 16-bit.

        Mode should be either "8-bit" or "16-bit". If no image is loaded,
        simply update the selection for future loads.
        """
        mode = mode or (self.import_bitdepth_var.get() or "8-bit")
        self.import_bitdepth_var.set(mode)
        if self.image_data is None:
            self.status_var.set(f"Import bit depth set to {mode}.")
            return

        try:
            converted = self._coerce_import_depth(self.image_data, mode)
        except Exception as exc:
            messagebox.showerror("Conversion failed", str(exc))
            return

        self.image_data = converted
        self.image_canvas.set_image(converted)
        self.image_canvas.fit_to_window()
        filename = os.path.basename(self.current_file) if self.current_file else ""
        self.image_info_var.set(
            f"{filename} | Size: {converted.shape[1]} × {converted.shape[0]} px | Type: {converted.dtype}"
        )
        self.status_var.set(f"Image converted to {mode}.")

    def _coerce_import_depth(self, data, bit_depth):
        mode = (bit_depth or "Auto").strip().lower()
        if mode == "auto":
            return np.array(data, copy=False)

        if mode == "8-bit":
            if data.dtype == np.uint8:
                return np.array(data, copy=False)
            scaled = data.astype(np.float64)
            lo = float(np.min(scaled))
            hi = float(np.max(scaled))
            if hi > lo:
                scaled = (scaled - lo) / (hi - lo) * 255.0
            return np.clip(scaled, 0, 255).astype(np.uint8)

        if mode == "16-bit":
            if data.dtype == np.uint16:
                return np.array(data, copy=False)
            scaled = data.astype(np.float64)
            lo = float(np.min(scaled))
            hi = float(np.max(scaled))
            if hi > lo:
                scaled = (scaled - lo) / (hi - lo) * 65535.0
            return np.clip(scaled, 0, 65535).astype(np.uint16)

        raise ValueError(f"Unsupported import bit depth: {bit_depth}")

    def _show_image(self, image, path):
        self.current_file = path
        self.image_data = image
        self.active_boundary = None
        self.boundary_traces.clear()
        self.boundary_order.clear()
        self.fovea_x = None

        self.image_canvas.set_image(image)
        self.image_canvas.enable_line(not self.vertical_mode_var.get())
        self.image_canvas.enable_roi(False)
        self.image_canvas.enable_vertical_line(self.vertical_mode_var.get())
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
        self._refresh_trace_list()
        self.status_var.set(
            "Image loaded. Left-click to place points, then press Finish Boundary to save the traced pixels."
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Boundary tracing actions
    # ═══════════════════════════════════════════════════════════════════════
    def _selected_boundary_name(self):
        name = self.boundary_var.get().strip()
        return name or "Boundary"

    def _boundary_color(self, name):
        if name in BOUNDARY_COLORS:
            return BOUNDARY_COLORS[name]
        palette = ["#ffb703", "#00b4d8", "#ef476f", "#8ac926", "#fb8500", "#8338ec", "#2ec4b6", "#f15bb5"]
        index = sum(ord(ch) for ch in name) % len(palette)
        return palette[index]

    def _start_boundary(self):
        messagebox.showinfo("Prototype", "Boundary tracing is disabled in prototype mode.")
        return

    def _undo_point(self):
        messagebox.showinfo("Prototype", "Undo point is disabled in prototype mode.")
        return

    def _clear_active_trace(self):
        messagebox.showinfo("Prototype", "Clearing active trace is disabled in prototype mode.")
        return

    def _finish_boundary(self):
        messagebox.showinfo("Prototype", "Finishing boundary is disabled in prototype mode.")
        return

    def _delete_selected_boundary(self):
        messagebox.showinfo("Prototype", "Deleting saved boundaries is disabled in prototype mode.")
        return

    def _clear_all_traces(self):
        messagebox.showinfo("Prototype", "Clearing all traces is disabled in prototype mode.")
        return

    def _rebuild_saved_overlays(self):
        # overlays are disabled in prototype; do nothing
        return

    def _on_vertical_mode_toggled(self):
        messagebox.showinfo("Prototype", "Vertical line mode is disabled in prototype mode.")
        # keep the checkbox in a consistent state (off)
        self.vertical_mode_var.set(False)
        return

    def _center_vertical_line(self):
        messagebox.showinfo("Prototype", "Centering vertical line is disabled in prototype mode.")
        return

    def _clear_vertical_line(self):
        messagebox.showinfo("Prototype", "Clearing vertical line is disabled in prototype mode.")
        return

    def _apply_vertical_line_x(self):
        messagebox.showinfo("Prototype", "Setting vertical line X is disabled in prototype mode.")
        return

    def _on_vertical_line_changed(self, x):
        # vertical line updates are disabled in prototype; ignore
        return

    # ═══════════════════════════════════════════════════════════════════════
    #  Live updates and selection handling
    # ═══════════════════════════════════════════════════════════════════════
    def _on_active_line_changed(self, points):
        # active line changes are disabled in prototype
        return

    def _on_mouse_moved(self, ix, iy, val):
        # pointer/status updates are disabled in prototype
        return

    def _refresh_trace_list(self):
        # saved traces list disabled in prototype
        self.trace_listbox.delete(0, "end")
        return

    def _select_trace_by_name(self, name):
        # selection disabled in prototype
        return

    def _on_saved_boundary_selected(self, _event):
        # saved boundary selection disabled in prototype
        return

    def _update_active_trace_summary(self):
        # active trace summary disabled in prototype
        self.active_trace_var.set("No active boundary")
        return

    def _update_saved_trace_summary(self, name):
        # saved trace summary disabled in prototype
        self.trace_detail_var.set("No saved boundary")
        return

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
