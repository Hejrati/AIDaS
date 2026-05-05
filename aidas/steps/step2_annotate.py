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
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw

from aidas.image_canvas import ImageCanvas
from aidas.utils.io_utils import read_analyze, read_tiff, write_analyze, scale_image
from aidas.utils.ui_utils import apply_app_icon_to


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


class Step2Frame(ttk.Frame):
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

    def __init__(self, parent, preferences=None, source_step=None):
        """Initialize the Step 2 annotation panel.

        Args:
            parent: Tkinter parent widget.
            preferences: User preferences dict (optional).
            source_step: Reference to Step 1 panel for linked image loading (optional).
        """
        super().__init__(parent)

        self.preferences = preferences
        self.source_step = source_step

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
    #  AI Settings Dialog
    # ═══════════════════════════════════════════════════════════════════════
    def _open_ai_settings_dialog(self):
        """Open a separate dialog for AI segmentation configuration.

        This dialog allows users to:
          - Set conda environment name for running oct-segmenter
          - Select segmentation config (.json) file
          - Select neural network model (.h5) file
          - Specify output directory for segmentation results
        """
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("AI Segmentation Settings")
        dialog.geometry("500x350")
        dialog.resizable(True, True)

        # Apply shared helper to propagate the app icon to this dialog
        apply_app_icon_to(dialog)

        main = ttk.Frame(dialog, padding=10)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Conda Environment:", font=(" ", 9, "bold")).pack(anchor="w", pady=(0, 2))
        env_frame = ttk.Frame(main)
        env_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(env_frame, textvariable=self.segmenter_env_var).pack(fill="x")

        ttk.Label(main, text="Config (.json):", font=(" ", 9, "bold")).pack(anchor="w", pady=(6, 2))
        cfg_frame = ttk.Frame(main)
        cfg_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(cfg_frame, textvariable=self.segmenter_config_var).pack(side="left", fill="x", expand=True)
        ttk.Button(cfg_frame, text="Browse", width=10, command=self._browse_segmenter_config).pack(side="right", padx=(4, 0))

        ttk.Label(main, text="Model (.h5):", font=(" ", 9, "bold")).pack(anchor="w", pady=(6, 2))
        model_frame = ttk.Frame(main)
        model_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(model_frame, textvariable=self.segmenter_model_var).pack(side="left", fill="x", expand=True)
        ttk.Button(model_frame, text="Browse", width=10, command=self._browse_segmenter_model).pack(side="right", padx=(4, 0))

        ttk.Label(main, text="Output Folder:", font=(" ", 9, "bold")).pack(anchor="w", pady=(6, 2))
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

        ttk.Label(load, text="Step 2 keeps 16-bit image data when available; MARKED outputs are 8-bit.", wraplength=240, justify="left").pack(
            fill="x",
            pady=(4, 0),
        )

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        self.segmenter_config_var = tk.StringVar(value=self.segmenter_default_config)
        self.segmenter_model_var = tk.StringVar(value=self.segmenter_default_model)
        self.segmenter_output_var = tk.StringVar(value=self._default_segmenter_output_dir())
        self.segmenter_env_var = tk.StringVar(value="oct-segmenter-env")

        self.segmentation_frame = ttk.LabelFrame(self.ctrl, text="Segmentation", padding=3)
        self.segmentation_frame.pack(**pad, pady=2)
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

        ai_buttons = ttk.Frame(workflow)
        ai_buttons.pack(fill="x", pady=(6, 0))
        self.preprocess_button = ttk.Button(
            ai_buttons,
            text="Preprocess Image",
            command=self._preprocess_image_for_ai,
        )
        self.preprocess_button.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.segment_button = ttk.Button(
            ai_buttons,
            text="AI Assist",
            command=self._run_neural_segmentation,
        )
        self.segment_button.pack(side="left", fill="x", expand=True, padx=(2, 0))
        ttk.Button(ai_buttons, text="AI Settings", command=self._open_ai_settings_dialog,).pack(side="right", padx=(2, 0))

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
        ttk.Button(saved_buttons, text="Export CSV", command=self._export_csv).pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 2),
        )
        ttk.Button(saved_buttons, text="Save MARKED Images", command=self._save_marked_images_button).pack(
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

        self._refresh_boundary_lists(auto_select=False)
        self._set_segmentation_frame_enabled(False)

    # ═══════════════════════════════════════════════════════════════════════
    #  Image loading
    # ═══════════════════════════════════════════════════════════════════════
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
        (PNG/JPEG). For multi-frame inputs a single slice is returned
        (the first frame). 16-bit sources are preserved for annotation.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in {".hdr", ".img"}:
            data = read_analyze(path)
            template_data = np.array(data, copy=False)
            if template_data.ndim == 2:
                slices = 1
                height, width = template_data.shape
            elif template_data.ndim == 3:
                slices, height, width = template_data.shape
            else:
                raise ValueError("Analyze image must be 2-D or 3-D.")
            self._input_analyze_template = {
                "slices": int(slices),
                "height": int(height),
                "width": int(width),
                "dtype": np.dtype(template_data.dtype),
            }
        elif ext in {".tif", ".tiff"}:
            data = read_tiff(path)
            self._input_analyze_template = None
        else:
            data = np.array(Image.open(path).convert("L"))
            self._input_analyze_template = None

        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2:
            raise ValueError("Step 2 expects a 2-D grayscale image or a 2-D slice from a stack.")
        return self._image_for_annotation(data)

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
        self._set_segmentation_frame_enabled(True)
        self._update_ai_button_states()
        self.status_var.set(
            "Image loaded. Left-click to place points only after selecting an incomplete boundary."
        )

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
        self._update_boundary_action_buttons()
        self._sync_boundary_canvas_state()

    def _set_active_boundary_target(self, name):
        if name not in BOUNDARY_NAMES or self.image_data is None:
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
            and not self.vertical_mode_var.get()
            and not self._segmenter_running
        )
        self.image_canvas.enable_line(drawing_enabled)
        self.image_canvas.enable_vertical_line(self.vertical_mode_var.get() and not self._segmenter_running)

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
          - Switches to vertical-line drawing mode on the canvas
          - Disables polyline drawing for boundaries
          - Centers the vertical line at image center
          - Enables foveal center nudge controls (left/right buttons, X entry)

        When disabled:
          - Removes the foveal center line from display
          - Clears the FOVEA_BOUNDARY_NAME trace
          - Re-enables polyline drawing for boundaries
          - Disables foveal center controls

        Prevents mode toggle during AI segmentation (_drawing_locked).
        """
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

        self.boundary_workflow_status_var.set(
            f"Incomplete: {len(incomplete_names)} | Completed: {len(completed_names)}/{len(BOUNDARY_NAMES)}"
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
            self.fovea_clear_btn,
        ):
            widget.configure(state=state)

    def _set_segmentation_frame_enabled(self, enabled):
        """Enable/disable all controls in the segmentation frame, except Finish/Revert buttons."""
        state = "normal" if enabled else "disabled"
        # Recursively disable/enable all children in the segmentation frame
        def set_state(widget, s):
            # Skip the Finish and Revert buttons - they manage their own state
            if widget in (self.finish_boundary_btn, self.revert_boundary_btn):
                return
            try:
                widget.configure(state=s)
            except Exception:
                pass
            for child in widget.winfo_children():
                set_state(child, s)
        
        if hasattr(self, "segmentation_frame"):
            set_state(self.segmentation_frame, state)

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
        trace_listbox = getattr(self, "trace_listbox", None)
        if trace_listbox is None:
            return
        trace_listbox.delete(0, "end")
        for name in self.boundary_order:
            trace = self.boundary_traces.get(name)
            if not trace:
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

    def _render_trace_mask(self, trace, width, height, boundary_name=None):
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

    def _apply_mark_values(self, target_image, mark_values):
        height, width = target_image.shape[:2]
        for name, mark_value in mark_values.items():
            trace = self.boundary_traces.get(name)
            if not trace:
                continue
            trace_mask = self._render_trace_mask(trace, width, height, boundary_name=name)
            if trace_mask is None:
                continue
            target_image[trace_mask] = np.uint8(mark_value)

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
                Image.fromarray(base_marked).resize((target_w, target_h), Image.Resampling.NEAREST),
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

    def _save_marked_images_button(self):
        """Button callback to manually save MARKED Analyze volumes.

        Provides user-facing save functionality with confirmation if boundaries
        are incomplete. Handles errors gracefully and shows save location.
        """
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
        if image.dtype == np.uint8:
            return np.array(image, copy=False)
        if getattr(self, "_source_was_8bit", False) and image.dtype == np.uint16:
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
        arr = np.asarray(image)
        if arr.ndim != 2:
            raise ValueError("Step 2 expects a 2-D grayscale image.")
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
        self._source_was_8bit = arr.dtype == np.uint8
        if arr.dtype == np.uint8:
            arr = np.rint(arr.astype(np.float64) * (65535.0 / 255.0)).astype(np.uint16)
        return np.ascontiguousarray(arr)

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

    def _import_segmenter_boundaries(self, csv_path):
        """Import boundary traces from neural network segmentation output.

        Reads the segmentation CSV (generated by oct-segmenter tool) and converts
        each row of pixel coordinates into boundary traces in the current
        annotation image/crop space.

        Steps:
          1. Read CSV with columns: [pixel_x0, pixel_y0, pixel_x1, pixel_y1, ...]
            where rows correspond to the preset boundary order (`BOUNDARY_PRESETS`)
          2. For each boundary, reconstruct the polyline from pixel coordinates
          3. Keep coordinates in the current annotation image/crop space
          4. Add boundary to boundary_traces dict with color and pixels
          5. Mark boundary as complete and update UI
          6. Rebuild canvas overlays to show new boundaries

        Preprocessing rescaling:
          - None here; saved 16-bit originals are cropped/resized to match the
            annotation/MARKED output shape.

        Args:
            csv_path: Path to boundaries CSV file from oct-segmenter output.

        Raises:
            RuntimeError: If CSV cannot be read or has incorrect format.
        """
        try:
            data = np.loadtxt(csv_path, delimiter=",", dtype=float)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read predicted boundaries CSV: {exc}") from exc

        data = np.atleast_2d(data)
        if data.shape[0] < len(BOUNDARY_PRESETS):
            raise RuntimeError(
                f"Predicted CSV has {data.shape[0]} rows; expected at least {len(BOUNDARY_PRESETS)} rows."
            )

        self.boundary_traces.clear()
        self.boundary_order.clear()
        self.image_canvas.clear_active_line()
        self.image_canvas.clear_line_overlays()

        width = int(self.image_data.shape[1])
        height = int(self.image_data.shape[0])
        max_x = min(width, data.shape[1])

        for row_idx, (name, _) in enumerate(BOUNDARY_PRESETS):
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

        # Mark imported boundaries as complete to indicate AI has processed them
        for name in BOUNDARY_NAMES:
            if name in self.boundary_completion_vars:
                self.boundary_completion_vars[name].set(name in self.boundary_traces)
        self._refresh_boundary_lists()
        self._update_boundary_progress_bar()

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

        if self._preprocessing_done:
            self.preprocess_button.configure(text="Remove Preprocess", command=self._remove_preprocess)
            self.preprocess_button.state(["!disabled"])
            self.segment_button.state(["!disabled"])
            return

        if self._image_matches_model_input():
            self.preprocess_button.configure(text="Preprocess Not Needed", command=self._preprocess_image_for_ai)
            self.preprocess_button.state(["disabled"])
            self.segment_button.state(["!disabled"])
            return

        self.preprocess_button.configure(text="Preprocess Image", command=self._preprocess_image_for_ai)
        self.preprocess_button.state(["!disabled"])
        self.segment_button.state(["disabled"])

    def _set_segmentation_running(self, running, status_message=None):
        """Lock/unlock UI controls while AI segmentation is running.

        During segmentation (running=True):
          - Disables all AI buttons (Preprocess, AI Assist)
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
            # If vertical mode is currently on, turn it off during segmentation
            if self.vertical_mode_var.get():
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
            # Start gradually filling progress bar
            if hasattr(self, "segmenter_progress"):
                self.segmenter_progress["value"] = 0
                self._animate_progress_bar()
        else:
            # Stop animation and show actual boundary count
            if hasattr(self, "_progress_animation_job"):
                self.after_cancel(self._progress_animation_job)
            if hasattr(self, "segmenter_progress"):
                self._update_boundary_progress_bar()
            # Update button states after segmentation finishes
            self._update_ai_button_states()

        # Update canvas state to disable/enable drawing based on segmentation status
        self._sync_boundary_canvas_state()

        if status_message:
            self.status_var.set(status_message)

    def _animate_progress_bar(self):
        """Gradually fill progress bar during segmentation to show activity."""
        if not self._segmenter_running or not hasattr(self, "segmenter_progress"):
            return
        
        current = self.segmenter_progress["value"]
        # Gradually increase but slow down as it approaches 6 (leaving room for the final result)
        if current < 6:
            increment = max(0.02, (6 - current) * 0.08)
            self.segmenter_progress["value"] = min(current + increment, 6)
            self._progress_animation_job = self.after(500, self._animate_progress_bar)
        else:
            # Keep it near 6 until segmentation finishes
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
                "no resizing/downsampling. Ready for AI Assist."
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
