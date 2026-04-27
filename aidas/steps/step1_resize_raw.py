"""Step 1 — Load, Resize & Crop Raw OCT Images.

Replicates the ImageJ macro "Step 1 resize raw.txt":
    1.  Open a SDB file (16-bit unsigned, configurable params)
    2.  Display and let user select a crop ROI
    3.  Crop (pixel replication) 
    4.  Save as "Light" (.hdr/.img/.tiff) in the same folder as source 
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from aidas.image_canvas import ImageCanvas
from aidas.utils.io_utils import read_raw_oct, scale_image, write_analyze, save_tiff
from aidas.utils.ui_utils import HoverToolTip

SDB_PREF_KEY = "sdb_dir"
SDB_DEFAULT_DIR = os.path.expanduser("~/Desktop")
DEFAULT_RAW_WIDTH = 768
DEFAULT_RAW_HEIGHT = 1200
DEFAULT_RAW_OFFSET = 1050
DEFAULT_RAW_BIT_DEPTH = 16
DEFAULT_OUTPUT_DIR = SDB_DEFAULT_DIR
CROP_SCALE_X = 3
CROP_SCALE_Y = 1
DEFAULT_ROI_X = 170
DEFAULT_ROI_Y = 585
DEFAULT_ROI_WIDTH = 491
DEFAULT_ROI_HEIGHT = 128


class Step1Frame(ttk.Frame):
    """GUI panel for Step 1: Resize Raw OCT images.

    This view owns all Step 1 controls and state:
    - import parameters for reading `.sdb` data,
    - file discovery/navigation,
    - ROI definition and processing,
    - output saving (Analyze + TIFF),
    - image interaction (zoom/pan/inspection).
    """

    def __init__(self, parent, preferences=None, on_processed_image=None):
        """Initialize the Step 1 panel and construct all widgets.

        Args:
            parent: Parent Tkinter container.
            preferences: Optional preferences object implementing `get` and `set`.
            on_processed_image: Optional callback receiving (image, source_path)
                whenever crop/scale produces a new processed image.
        """
        super().__init__(parent)

        self.preferences = preferences
        self._on_processed_image = on_processed_image

        # ----- state -----
        self.raw_image = None          # original loaded image (H, W)  uint16
        self._source_raw_image = None  # original imported image before width adjustments
        self.processed_image = None    # after crop + scale           int16 (.img-like preview)
        self.current_file = None       # path of opened raw file
        self.raw_import_params = None  # validated import parameters
        self._updating_roi_entries = False
        self._updating_target_size_entries = False
        self._target_size_edit_active = False

        # ----- layout -----
        # Fixed sidebar on the left + expandable image area on the right.
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        # Left — scrollable control panel (content-driven width)
        left_outer = ttk.Frame(main)
        left_outer.pack(side="left", fill="y")

        ctrl_canvas = tk.Canvas(left_outer, highlightthickness=0, bd=0)
        ctrl_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=ctrl_canvas.yview)
        self.ctrl = ttk.Frame(ctrl_canvas)
        self.ctrl.bind("<Configure>",
                       lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        ctrl_window = ctrl_canvas.create_window((0, 0), window=self.ctrl, anchor="nw")
        ctrl_canvas.bind(
            "<Configure>",
            lambda e: ctrl_canvas.itemconfigure(ctrl_window, width=e.width),
        )
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_canvas.pack(side="left", fill="both", expand=True)
        ctrl_scroll.pack(side="right", fill="y")

        # Right — image canvas + status
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        # Image info header at top
        info_frame = ttk.Frame(right, relief="solid", borderwidth=1)
        info_frame.pack(fill="x", padx=2, pady=2)
        self.image_info_var = tk.StringVar(value="No image loaded")
        info_label = ttk.Label(info_frame, textvariable=self.image_info_var,
                              font=("", 10, "bold"), padding=8, anchor="w")
        info_label.pack(fill="x")

        self.image_canvas = ImageCanvas(right,
                                        on_roi_change=self._on_roi_changed,
                                        on_mouse_move=self._on_mouse_moved)
        self.image_canvas.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(
            value="Ready — open an SDB raw OCT file to begin (left-drag ROI, right-drag pan)"
        )
        ttk.Label(right, textvariable=self.status_var,
                  relief="sunken", anchor="w", padding=3).pack(side="bottom", fill="x")

        # Build control widgets
        self._build_controls()

    # ═══════════════════════════════════════════════════════════════════════
    #  Control-panel construction
    # ═══════════════════════════════════════════════════════════════════════
    def _build_controls(self):
        """Create and lay out the full left-side control panel."""
        pad = dict(fill="x", padx=(14, 8))
        numeric_vcmd = (self.register(self._validate_digits_only), "%P")


        # ── SDB Image Parameters ──
        self.sdb_params_frame = ttk.LabelFrame(self.ctrl, text="SDB Image Parameters", padding=1)
        self.sdb_params_frame.pack(**pad, pady=5)

        self.width_var = tk.StringVar(value=str(DEFAULT_RAW_WIDTH))
        self.height_var = tk.StringVar(value=str(DEFAULT_RAW_HEIGHT))
        self.offset_var = tk.StringVar(value=str(DEFAULT_RAW_OFFSET))

        self.width_entry, self.width_minus_btn, self.width_plus_btn, self.width_reset_btn = self._param_button_row(
            self.sdb_params_frame, 0, "Width (px):", self.width_var, DEFAULT_RAW_WIDTH,
            step=1, minimum=1, maximum=10000, validatecommand=numeric_vcmd
        )
        self.height_entry, self.height_minus_btn, self.height_plus_btn, self.height_reset_btn = self._param_button_row(
            self.sdb_params_frame, 1, "Height (px):", self.height_var, DEFAULT_RAW_HEIGHT,
            step=1, minimum=1, maximum=10000, validatecommand=numeric_vcmd
        )
        self.offset_entry, self.offset_minus_btn, self.offset_plus_btn, self.offset_reset_btn = self._param_button_row(
            self.sdb_params_frame, 2, "Offset (bytes):", self.offset_var, DEFAULT_RAW_OFFSET,
            step=2, minimum=0, maximum=10_000_000, validatecommand=numeric_vcmd
        )

        width_entry_hint = (
            "Width value: if smaller than source width, image is cropped; "
            "if larger, image is padded with zeros."
        )
        width_minus_hint = "− Width: decrease by 1 px. May crop image if width becomes smaller than source."
        width_plus_hint = "+ Width: increase by 1 px. Adds zero-padding if width becomes larger than source."
        width_reset_hint = f"↺ Width: reset to default width ({DEFAULT_RAW_WIDTH} px)."
        self._width_tooltips = [
            HoverToolTip(self.width_entry, width_entry_hint),
            HoverToolTip(self.width_minus_btn, width_minus_hint),
            HoverToolTip(self.width_plus_btn, width_plus_hint),
            HoverToolTip(self.width_reset_btn, width_reset_hint),
        ]

        height_entry_hint = (
            "Height value: changes the number of image rows used for import; "
            "too small truncates rows, too large adds zero-filled rows."
        )
        height_minus_hint = "− Height: decrease by 1 px. May truncate bottom rows if below source height."
        height_plus_hint = "+ Height: increase by 1 px. May add zero-filled rows if above source height."
        height_reset_hint = f"↺ Height: reset to default height ({DEFAULT_RAW_HEIGHT} px)."
        self._height_tooltips = [
            HoverToolTip(self.height_entry, height_entry_hint),
            HoverToolTip(self.height_minus_btn, height_minus_hint),
            HoverToolTip(self.height_plus_btn, height_plus_hint),
            HoverToolTip(self.height_reset_btn, height_reset_hint),
        ]

        self.width_var.trace_add("write", lambda *_: self._on_width_changed())
        self.height_var.trace_add("write", lambda *_: self._on_import_param_changed())
        self.offset_var.trace_add("write", lambda *_: self._on_import_param_changed())

        self.endian_var = tk.BooleanVar(value=True)
        self.endian_var.trace_add("write", lambda *_: self._on_import_param_changed())
        self.endian_checkbox = ttk.Checkbutton(
            self.sdb_params_frame,
            text="Little-endian",
            variable=self.endian_var,
        )
        self.endian_checkbox.grid(row=3, column=0, columnspan=2, sticky="w", pady=1)
        self._endian_tooltip = HoverToolTip(self.endian_checkbox, "Can affect visualization for some offsets")

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        # ── SDB Files ──
        sdb = ttk.LabelFrame(self.ctrl, text="SDB Files", padding=3)
        sdb.pack(**pad, pady=2)

        ttk.Label(sdb, text="Input dir:").pack(anchor="w")
        dir_frame = ttk.Frame(sdb)
        dir_frame.pack(fill="x")
        self.sdb_dir_var = tk.StringVar(value=self._initial_sdb_dir())
        ttk.Entry(dir_frame, textvariable=self.sdb_dir_var
                  ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            dir_frame,
            text="⌂",
            width=2,
            command=self._reset_sdb_dir_to_default,
        ).pack(side="right", padx=(2, 0))
        ttk.Button(dir_frame, text="↻", width=3,
                   command=self.refresh_sdb_list).pack(side="right", padx=(2, 0))
        ttk.Button(dir_frame, text="…", width=3,
                   command=self._browse_sdb_dir).pack(side="right")

        filt_frame = ttk.Frame(sdb)
        filt_frame.pack(fill="x", pady=(2, 0))
        ttk.Label(filt_frame, text="Search:").pack(side="left")
        self.sdb_filter_var = tk.StringVar(value="")
        self.sdb_filter_var.trace_add("write", lambda *_: self.refresh_sdb_list())
        ttk.Entry(filt_frame, textvariable=self.sdb_filter_var).pack(
            side="left", fill="x", expand=True, padx=(4, 0))

        lb_frame = ttk.Frame(sdb)
        lb_frame.pack(fill="both", expand=True, pady=(2, 0))
        self.sdb_listbox = tk.Listbox(lb_frame, height=8, selectmode="browse")
        lb_scroll = ttk.Scrollbar(lb_frame, orient="vertical",
                                  command=self.sdb_listbox.yview)
        self.sdb_listbox.configure(yscrollcommand=lb_scroll.set)
        self.sdb_listbox.pack(side="left", fill="both", expand=True)
        lb_scroll.pack(side="right", fill="y")
        self.sdb_listbox.bind("<Double-1>", lambda e: self._open_selected_sdb())
        self.sdb_listbox.bind("<<ListboxSelect>>", self._on_sdb_list_select)

        nav_frame = ttk.Frame(sdb)
        nav_frame.pack(fill="x", pady=(4, 0))
        ttk.Button(nav_frame, text="◀ Prev",
                   command=self._prev_sdb).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(nav_frame, text="Next ▶",
                   command=self._next_sdb).pack(side="right", expand=True, fill="x", padx=(2, 0))

        self._sdb_files = []
        self.refresh_sdb_list()

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        # ── ROI Selection ──
        roi = ttk.LabelFrame(self.ctrl, text="ROI Selection (crop and save)", padding=3)
        roi.pack(**pad, pady=2)
        for col in range(4):
            roi.grid_columnconfigure(col, weight=1)

        self.roi_x_var = tk.StringVar(value="0")
        self.roi_y_var = tk.StringVar(value="0")
        self.roi_w_var = tk.StringVar(value="100")
        self.roi_h_var = tk.StringVar(value="100")
        self.target_w_var = tk.StringVar(value=str(100 * CROP_SCALE_X))
        self.target_h_var = tk.StringVar(value=str(100 * CROP_SCALE_Y))
        self.roi_x_var.trace_add("write", self._on_roi_entry_changed)
        self.roi_y_var.trace_add("write", self._on_roi_entry_changed)
        self.roi_w_var.trace_add("write", self._on_roi_entry_changed)
        self.roi_h_var.trace_add("write", self._on_roi_entry_changed)
        self.target_w_var.trace_add("write", self._on_target_size_entry_changed)
        self.target_h_var.trace_add("write", self._on_target_size_entry_changed)

        ttk.Label(roi, text="Output dir:").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 0)
        )
        save_dir_row = ttk.Frame(roi)
        save_dir_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(2, 6))
        self.outdir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        ttk.Entry(save_dir_row, textvariable=self.outdir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(
            save_dir_row,
            text="⌂",
            width=2,
            command=self._reset_outdir_to_default,
        ).pack(side="right", padx=(2, 0))
        ttk.Button(save_dir_row, text="↻", width=3,
                   command=self._refresh_outdir_from_source).pack(side="right", padx=(2, 0))
        ttk.Button(save_dir_row, text="…", width=3,
                   command=self._browse_outdir).pack(side="right")

        self.roi_entries = []
        for i, (lbl, var, color) in enumerate([
            ("X (Left):", self.roi_x_var, "#DA0404"),
            ("Y (Top):", self.roi_y_var, "#DA0404"),
            ("Source W:", self.roi_w_var, None),
            ("Source H:", self.roi_h_var, None),
        ]):
            r, c = divmod(i, 2)
            if color:
                tk.Label(roi, text=lbl, fg=color).grid(row=r + 2, column=c * 2, sticky="w", pady=1)
            else:
                ttk.Label(roi, text=lbl).grid(row=r + 2, column=c * 2, sticky="w", pady=1)
            entry = ttk.Entry(
                roi,
                textvariable=var,
                width=7,
                validate="key",
                validatecommand=numeric_vcmd,
            )
            entry.grid(
                row=r + 2, column=c * 2 + 1, sticky="e", padx=(0, 8), pady=1)
            self.roi_entries.append(entry)

        ttk.Label(roi, text="Target W:").grid(row=4, column=0, sticky="w", pady=(4, 1))
        self.target_w_entry = ttk.Entry(
            roi,
            textvariable=self.target_w_var,
            width=7,
            validate="key",
            validatecommand=numeric_vcmd,
        )
        self.target_w_entry.grid(row=4, column=1, sticky="e", padx=(0, 8), pady=(4, 1))
        ttk.Label(roi, text="Target H:").grid(row=4, column=2, sticky="w", pady=(4, 1))
        self.target_h_entry = ttk.Entry(
            roi,
            textvariable=self.target_h_var,
            width=7,
            validate="key",
            validatecommand=numeric_vcmd,
        )
        self.target_h_entry.grid(row=4, column=3, sticky="e", padx=(0, 8), pady=(4, 1))
        self.target_size_entries = [self.target_w_entry, self.target_h_entry]

        ttk.Label(
            roi,
            text=f"Scale: target width is source width x{CROP_SCALE_X}; height unchanged.",
            foreground="gray",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(2, 2))

        roi_presets = ttk.Frame(roi)
        roi_presets.grid(row=6, column=0, columnspan=4, sticky="w", pady=(4, 2))
        self.default_roi_btn = ttk.Button(roi_presets, text="Default Region", command=self._set_default_roi)
        self.default_roi_btn.pack(side="left", padx=(0, 4))
        self.entire_roi_btn = ttk.Button(roi_presets, text="Entire Image", command=self._select_all_roi)
        self.entire_roi_btn.pack(side="left", padx=(4, 0))
        #Todo: add AI based auto-ROI detection here
        ttk.Button(roi_presets, text="Auto Select", command=self._set_default_roi, state="disabled").pack(side="left", padx=(4, 0))

        roi_actions = ttk.Frame(roi)
        roi_actions.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(4, 0))

        self.crop_btn = ttk.Button(roi_actions, text="▶  Crop", command=self._crop_and_scale)
        self.crop_btn.pack(fill="x", pady=(0, 2))
        self.undo_crop_btn = ttk.Button(
            roi_actions,
            text="↺ Undo Crop",
            command=self._reset,
            state="disabled",
        )
        self.undo_crop_btn.pack(fill="x", pady=(0, 2))
        self.save_all_btn = ttk.Button(
            roi,
            text="Save All (TIFF, IMG, HDR)",
            command=self._save_all_formats,
            state="disabled",
        )
        self.save_all_btn.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(2, 2))

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        # ── View ──
        view = ttk.LabelFrame(self.ctrl, text="View", padding=3)
        view.pack(**pad, pady=(2, 6))

        zf = ttk.Frame(view)
        zf.pack(fill="x")
        ttk.Button(zf, text="−", width=3, command=self._zoom_out).pack(side="left")
        self.zoom_lbl = ttk.Label(zf, text="100 %", anchor="center")
        self.zoom_lbl.pack(side="left", expand=True)
        ttk.Button(zf, text="+", width=3, command=self._zoom_in).pack(side="right")
        ttk.Button(view, text="Fit to Window",
                   command=self._fit_zoom).pack(fill="x", pady=2)

    def _save_all_formats(self):
        """Save the current cropped image as TIFF, HDR, and IMG in one click."""
        img = self.processed_image
        if img is None:
            messagebox.showwarning("Nothing to save", "Run 'Crop & Scale' first.")
            return
        outdir = self.outdir_var.get()
        if not os.path.isdir(outdir):
            messagebox.showerror("Invalid output folder", f"Folder does not exist: {outdir}")
            return
        base_name = self._build_output_name("light")
        base = os.path.join(outdir, base_name)
        # Save TIFF
        tiff_path = base + ".tif"
        try:
            save_tiff(tiff_path, img)
        except Exception as exc:
            messagebox.showerror("Save error (TIFF)", str(exc))
            return
        # Save Analyze 7.5 (HDR + IMG)
        stack = np.stack([img, img], axis=0)  # shape (2, H, W)
        try:
            hdr_path, img_path = write_analyze(base, stack)
        except Exception as exc:
            messagebox.showerror("Save error (Analyze)", str(exc))
            return
        messagebox.showinfo(
            "Saved",
            f"Saved all formats successfully:\n  {tiff_path}\n  {hdr_path}\n  {img_path}\n\nStack: 2 slices of {img.shape[1]}×{img.shape[0]}  {img.dtype}"
        )
        self.status_var.set(f"Saved → {tiff_path}, {hdr_path}, {img_path}")

    def _update_save_button_state(self):
        """Sync Save/Undo button states with processed image availability."""
        if getattr(self, "save_all_btn", None) is None:
            return
        has_processed = self.processed_image is not None
        self.save_all_btn.configure(state="normal" if has_processed else "disabled")
        if getattr(self, "undo_crop_btn", None) is not None:
            self.undo_crop_btn.configure(state="normal" if has_processed else "disabled")
        if getattr(self, "crop_btn", None) is not None:
            self.crop_btn.configure(state="disabled" if has_processed else "normal")

    @staticmethod
    def _to_uint8_preview(data):
        """Scale any grayscale image to uint8 for display."""
        if data.dtype == np.uint8:
            return np.array(data, copy=False)
        arr = data.astype(np.float64)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _set_widget_tree_state(self, widget, enabled):
        """Recursively enable or disable widgets inside a container."""
        state = "normal" if enabled else "disabled"
        for child in widget.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
            self._set_widget_tree_state(child, enabled)

    def _set_sdb_parameters_enabled(self, enabled):
        """Toggle the SDB import-parameter section as a group."""
        if getattr(self, "sdb_params_frame", None) is None:
            return
        self._set_widget_tree_state(self.sdb_params_frame, enabled)

    def _confirm_discard_processed_image(self, next_path):
        """Ask before replacing an active cropped image with another source."""
        if self.processed_image is None:
            return True

        current_name = os.path.basename(self.current_file) if self.current_file else "current image"
        next_name = os.path.basename(next_path) if next_path else "the selected image"
        return messagebox.askyesno(
            "Discard cropped image?",
            f"A cropped image is currently active for {current_name}.\n\n"
            f"Opening {next_name} will discard the cropped result and reset the view.\n"
            "Continue?",
            icon="warning",
            default="no",
        )

    # helper for param rows
    def _initial_sdb_dir(self):
        """Resolve initial SDB directory.

        Returns:
            str: Default Desktop directory for initial browser location.
        """
        return SDB_DEFAULT_DIR

    def set_sdb_directory(self, directory):
        """Set active SDB directory and persist the value in preferences.

        Args:
            directory: Directory path selected by user.
        """
        self.sdb_dir_var.set(directory)
        self._sync_output_dir_with_source(directory)
        if self.preferences is not None:
            self.preferences.set(SDB_PREF_KEY, directory)

    def _sync_output_dir_with_source(self, source_path):
        """Mirror output folder to the selected source location.

        Args:
            source_path: Source file path or directory path.
        """
        if not source_path:
            return
        target_dir = source_path if os.path.isdir(source_path) else os.path.dirname(source_path)
        if target_dir:
            self.outdir_var.set(target_dir)

    # helper for param rows
    def _param_button_row(
        self,
        parent,
        row,
        label,
        var,
        default_value,
        *,
        step=1,
        minimum=0,
        maximum=10_000_000,
        validatecommand=None,
    ):
        """Create one label+entry row with external minus/plus buttons.

        Args:
            parent: Container where the row is placed.
            row: Grid row index.
            label: Label text.
            var: Tk variable bound to the entry.
            default_value: Value restored by the reset button.
            minimum: Minimum allowed value.
            maximum: Maximum allowed value.

        Returns:
            tuple: (entry, minus_button, plus_button, reset_button)
        """
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=1, sticky="e", pady=1)

        minus_btn = ttk.Button(row_frame, text="−", width=2, command=lambda: self._step_numeric_var(var, -step, minimum, maximum))
        minus_btn.pack(side="left")

        entry_options = {"textvariable": var, "width": 10}
        if validatecommand is not None:
            entry_options["validate"] = "key"
            entry_options["validatecommand"] = validatecommand
        entry = ttk.Entry(row_frame, **entry_options)
        entry.pack(side="left", padx=(4, 4))

        plus_btn = ttk.Button(row_frame, text="+", width=2, command=lambda: self._step_numeric_var(var, step, minimum, maximum))
        plus_btn.pack(side="left")

        reset_btn = ttk.Button(row_frame, text="↺", width=2, command=lambda: self._reset_numeric_var(var, default_value))
        reset_btn.pack(side="left", padx=(4, 0))

        return entry, minus_btn, plus_btn, reset_btn

    @staticmethod
    def _step_numeric_var(var, delta, minimum, maximum):
        try:
            current = int(var.get())
        except ValueError:
            current = minimum
        next_value = max(minimum, min(maximum, current + delta))
        var.set(str(next_value))

    @staticmethod
    def _reset_numeric_var(var, default_value):
        var.set(str(default_value))

    @staticmethod
    def _validate_digits_only(proposed_value):
        """Allow only digits for numeric entries (empty is allowed while editing)."""
        return proposed_value == "" or proposed_value.isdigit()

    # ═══════════════════════════════════════════════════════════════════════
    #  Actions
    # ═══════════════════════════════════════════════════════════════════════

    def _set_default_import_params(self):
        """Restore default SDB import parameters and apply them."""
        self.width_var.set(str(DEFAULT_RAW_WIDTH))
        self.height_var.set(str(DEFAULT_RAW_HEIGHT))
        self.offset_var.set(str(DEFAULT_RAW_OFFSET))
        self.endian_var.set(True)
        return self._apply_import_params()

    def _on_import_param_changed(self):
        """Auto-apply import parameters when UI values change."""
        self._apply_import_params(show_errors=False)

    def _on_width_changed(self):
        """Store the requested width and resize the current image view without rereading the raw file."""
        self._apply_import_params(show_errors=False, skip_reload=True)
        self._apply_width_preview_adjustment()

    def _apply_width_preview_adjustment(self):
        """Crop or pad the loaded image to match the requested width."""
        if self._source_raw_image is None or self.current_file is None:
            return

        try:
            requested_width = int(self.width_var.get())
        except ValueError:
            return

        if requested_width <= 0:
            return

        source = self._source_raw_image
        source_width = int(source.shape[1])
        if requested_width == source_width:
            adjusted = np.array(source, copy=True)
            note = f"Width matches source ({source_width}); no crop/pad applied."
        elif requested_width < source_width:
            crop = source_width - requested_width
            left = crop // 2
            right = left + requested_width
            adjusted = np.array(source[:, left:right], copy=True)
            note = f"Warning: width smaller than source; cropped {crop} px from the image."
        else:
            pad = requested_width - source_width
            left = pad // 2
            right = pad - left
            adjusted = np.pad(source, ((0, 0), (left, right)), mode="constant", constant_values=0)
            note = f"Width larger than source; padded {pad} px with zeros."

        self.raw_image = adjusted
        self.processed_image = None
        self.image_canvas.set_image(adjusted)
        self.image_canvas.enable_roi(True)
        self._set_default_roi()
        self._update_zoom_label()
        self.status_var.set(note)

        filename = os.path.basename(self.current_file)
        self.image_info_var.set(
            f"{filename} |  Dir: {os.path.dirname(self.current_file)}   "
        )

    @staticmethod
    def _offset_noise_score(img):
        """Heuristic score for offset quality: lower means smoother/more plausible image."""
        arr = img.astype(np.float32)
        dx = np.abs(np.diff(arr, axis=1))
        dy = np.abs(np.diff(arr, axis=0))
        return float(np.median(dx) + np.median(dy))

    def _auto_find_offset(self):
        """Search nearby even offsets and pick the least-noisy image alignment."""
        if not self.current_file:
            messagebox.showinfo("No image", "Open an SDB file first.")
            return

        try:
            w = int(self.width_var.get())
            h = int(self.height_var.get())
            off = int(self.offset_var.get())
            le = self.endian_var.get()
        except ValueError:
            messagebox.showerror("Error", "Width/Height/Offset must be valid integers.")
            return

        if w <= 0 or h <= 0 or off < 0:
            messagebox.showerror("Error", "Width/Height must be > 0 and Offset must be >= 0.")
            return

        base = off if off % 2 == 0 else off - 1
        coarse_start = max(0, base - 128)
        coarse_end = base + 128
        coarse_candidates = [o for o in range(coarse_start, coarse_end + 1, 8) if o % 2 == 0]
        if not coarse_candidates:
            coarse_candidates = [max(0, base)]

        self.status_var.set("Scanning nearby offsets...")
        self.update_idletasks()

        best_off = None
        best_score = None

        for cand in coarse_candidates:
            try:
                img = read_raw_oct(
                    self.current_file,
                    width=w,
                    height=h,
                    offset=cand,
                    bit_depth=DEFAULT_RAW_BIT_DEPTH,
                    little_endian=le,
                )
            except (OSError, ValueError, RuntimeError):
                continue
            score = self._offset_noise_score(img)
            if best_score is None or score < best_score:
                best_score = score
                best_off = cand

        if best_off is None:
            messagebox.showerror("Auto offset failed", "Could not evaluate candidate offsets.")
            return

        fine_start = max(0, best_off - 8)
        fine_end = best_off + 8
        for cand in range(fine_start, fine_end + 1, 2):
            try:
                img = read_raw_oct(
                    self.current_file,
                    width=w,
                    height=h,
                    offset=cand,
                    bit_depth=DEFAULT_RAW_BIT_DEPTH,
                    little_endian=le,
                )
            except (OSError, ValueError, RuntimeError):
                continue
            score = self._offset_noise_score(img)
            if score < best_score:
                best_score = score
                best_off = cand

        self.offset_var.set(str(best_off))
        self.status_var.set(f"Auto offset selected: {best_off}")

    def _apply_import_params(self, show_errors=True, skip_reload=False):
        """Validate and store raw import parameters from the form.

        If a file is already open, this method immediately reloads that file
        with the updated values so the UI always reflects active parameters.

        Returns:
            bool: True when parameters are valid (and reload succeeds if needed).
        """
        try:
            w = int(self.width_var.get())
            h = int(self.height_var.get())
            off = int(self.offset_var.get())
            le = self.endian_var.get()
        except ValueError:
            if show_errors:
                messagebox.showerror("Error", "Invalid import parameter (must be integers).")
            else:
                self.status_var.set("Waiting for valid import parameters...")
            return False

        if w <= 0 or h <= 0 or off < 0:
            if show_errors:
                messagebox.showerror("Error", "Width/Height must be > 0 and Offset must be >= 0.")
            else:
                self.status_var.set("Waiting for valid import parameters...")
            return False

        self.raw_import_params = {
            "width": w,
            "height": h,
            "offset": off,
            "bit_depth": DEFAULT_RAW_BIT_DEPTH,
            "little_endian": le,
        }

        # If an image is already open, immediately re-read it with new params.
        # Width changes are handled separately as a display-only crop/pad step.
        if self.current_file and not skip_reload:
            try:
                img = read_raw_oct(self.current_file, **self.raw_import_params)
            except (OSError, ValueError, RuntimeError) as exc:
                messagebox.showerror(
                    "Error reading file",
                    f"Parameters applied, but reloading current image failed:\n{exc}",
                )
                return False
            self._load_image(img, self.current_file)
            self.status_var.set(
                f"Parameters applied and reloaded: {os.path.basename(self.current_file)}"
            )
            return True

        if self.current_file and skip_reload:
            self.status_var.set(
                f"Width stored for display adjustment: {w}px"
            )
            return True

        self.status_var.set(
            f"Import params applied: {w}x{h}, offset {off}, {DEFAULT_RAW_BIT_DEPTH}-bit, "
            f"{'little' if le else 'big'}-endian"
        )
        return True

    def _load_image(self, img, path):
        """Load image data into UI state and refresh display widgets.

        Args:
            img: Loaded image array with shape (H, W).
            path: Source file path for display and output naming.
        """
        self._source_raw_image = np.array(img, copy=True)
        self.raw_image = np.array(img, copy=True)
        self.processed_image = None
        self.current_file = path

        filename = os.path.basename(path)

        # Update the top info display
        self.image_info_var.set(
            f"{filename} |  Dir: {os.path.dirname(path)}   "
        )

        self.image_canvas.set_image(img)
        self.image_canvas.enable_roi(True)
        self._set_default_roi()
        self._update_zoom_label()
        self._update_save_button_state()
        self._set_sdb_parameters_enabled(True)
        self.status_var.set(
            f"Loaded {filename} — left-drag ROI, right-drag pan, then Crop & Scale")

    # ── Open Raw ──
    def _open_raw(self, path=None):
        """Open a `.sdb` raw file and display it.

        Args:
            path: Optional explicit file path. When omitted, shows file picker.
        """
        if self.raw_import_params is None and not self._apply_import_params(skip_reload=True):
            return

        if path is None:
            path = filedialog.askopenfilename(
                title="Select SDB raw OCT file",
                initialdir=self.sdb_dir_var.get() or None,
                filetypes=[("SDB raw", "*.sdb"), ("All files", "*.*")],
            )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext != ".sdb":
            messagebox.showwarning(
                "Unexpected file type",
                "This workflow is intended for .sdb raw files.\n"
                "If this is intentional, rename or convert to .sdb first.",
            )
            return

        if self.processed_image is not None and not self._confirm_discard_processed_image(path):
            return

        # Re-apply before opening so current form values are always used.
        if not self._apply_import_params(skip_reload=True):
            return

        self._sync_output_dir_with_source(path)

        try:
            img = read_raw_oct(path, **self.raw_import_params)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Error reading file", str(exc))
            return

        self._load_image(img, path)

    def _browse_sdb_dir(self):
        """Prompt for SDB folder and refresh the browser list."""
        file_path = filedialog.askopenfilename(
            title="Select SDB file (directory will be used)",
            initialdir=self.sdb_dir_var.get() or None,
            filetypes=[("SDB files", "*.sdb"), ("All files", "*.*")],
        )
        if file_path:
            import os
            directory = os.path.dirname(file_path)
            self.set_sdb_directory(directory)
            self.refresh_sdb_list()

    def _reset_sdb_dir_to_default(self):
        """Reset SDB directory to default (Desktop) and refresh list."""
        target_dir = SDB_DEFAULT_DIR
        self.set_sdb_directory(target_dir)
        self.refresh_sdb_list()
        self.status_var.set(f"SDB directory reset to default: {target_dir}")

    def refresh_sdb_list(self):
        """Scan selected folder and repopulate `.sdb` files list.

        The current search text is applied as a case-insensitive substring filter.
        """
        self.sdb_listbox.delete(0, "end")
        self._sdb_files.clear()
        d = self.sdb_dir_var.get()
        if not d or not os.path.isdir(d):
            return

        filt = self.sdb_filter_var.get().lower().strip()
        files = [f for f in os.listdir(d) if f.lower().endswith(".sdb")]
        files.sort(key=str.lower)

        for f in files:
            if filt and filt not in f.lower():
                continue
            self._sdb_files.append(f)
            self.sdb_listbox.insert("end", f)

        self.status_var.set(f"SDB directory: {d}  |  {len(self._sdb_files)} file(s)")

    def _on_sdb_list_select(self, _event):
        """Update status text when the SDB list selection changes."""
        sel = self.sdb_listbox.curselection()
        if not sel:
            return
        fname = self._sdb_files[sel[0]]
        self.status_var.set(f"Selected SDB: {fname}  (double-click or Open Selected)")

    def _open_selected_sdb(self):
        """Open the currently selected SDB file from the list."""
        sel = self.sdb_listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Select an SDB file from the list first.")
            return
        fname = self._sdb_files[sel[0]]
        full = os.path.join(self.sdb_dir_var.get(), fname)
        self._open_raw(path=full)

    def _prev_sdb(self):
        """Select and open the previous SDB file in the filtered list."""
        if not self._sdb_files:
            return
        sel = self.sdb_listbox.curselection()
        idx = max(0, sel[0] - 1) if sel else 0
        self.sdb_listbox.selection_clear(0, "end")
        self.sdb_listbox.selection_set(idx)
        self.sdb_listbox.see(idx)
        self._open_selected_sdb()

    def _next_sdb(self):
        """Select and open the next SDB file in the filtered list."""
        if not self._sdb_files:
            return
        sel = self.sdb_listbox.curselection()
        idx = min(len(self._sdb_files) - 1, sel[0] + 1) if sel else 0
        self.sdb_listbox.selection_clear(0, "end")
        self.sdb_listbox.selection_set(idx)
        self.sdb_listbox.see(idx)
        self._open_selected_sdb()

    # ── ROI ──
    def _set_default_roi(self):
        """Set ROI to a centered band on the opened image.

        Default behavior: a fixed source crop region using the legacy offset.
        """
        if self.raw_image is None:
            return
        ih, iw = self.raw_image.shape
        x = min(DEFAULT_ROI_X, max(0, iw - 1))
        y = min(DEFAULT_ROI_Y, max(0, ih - 1))
        w = min(DEFAULT_ROI_WIDTH, iw - x)
        h = min(DEFAULT_ROI_HEIGHT, ih - y)
        self._set_roi_and_entries(x, y, w, h)

    def _select_all_roi(self):
        """Set ROI to cover the entire current raw image."""
        if self.raw_image is None:
            return
        ih, iw = self.raw_image.shape
        self._set_roi_and_entries(0, 0, iw, ih)

    def _apply_roi_entries(self):
        """Apply ROI values entered manually in the form fields."""
        try:
            x = int(self.roi_x_var.get())
            y = int(self.roi_y_var.get())
            w = int(self.roi_w_var.get())
            h = int(self.roi_h_var.get())
        except ValueError:
            messagebox.showerror("Error", "ROI values must be integers.")
            return
        self.image_canvas.set_roi((x, y, w, h))

    def _on_roi_entry_changed(self, *_):
        """Apply ROI immediately when entry values become valid integers."""
        if self._updating_roi_entries or self.raw_image is None:
            return
        try:
            x = int(self.roi_x_var.get())
            y = int(self.roi_y_var.get())
            w = int(self.roi_w_var.get())
            h = int(self.roi_h_var.get())
        except ValueError:
            return
        if w <= 0 or h <= 0:
            return
        self._update_target_size_entries(w, h)
        self.image_canvas.set_roi((x, y, w, h))

    def _on_target_size_entry_changed(self, *_):
        """Update source ROI size when the user edits final target dimensions."""
        if self._updating_target_size_entries or self.raw_image is None:
            return
        try:
            x = int(self.roi_x_var.get())
            y = int(self.roi_y_var.get())
            target_w = int(self.target_w_var.get())
            target_h = int(self.target_h_var.get())
        except ValueError:
            return
        if target_w <= 0 or target_h <= 0:
            return

        source_w = max(1, int(round(target_w / CROP_SCALE_X)))
        source_h = max(1, int(round(target_h / CROP_SCALE_Y)))
        ih, iw = self.raw_image.shape
        source_w = min(source_w, max(1, iw - x))
        source_h = min(source_h, max(1, ih - y))

        self._updating_roi_entries = True
        self.roi_w_var.set(str(source_w))
        self.roi_h_var.set(str(source_h))
        self._updating_roi_entries = False
        self._target_size_edit_active = True
        try:
            self.image_canvas.set_roi((x, y, source_w, source_h))
        finally:
            self._target_size_edit_active = False

    def _set_roi_and_entries(self, x, y, w, h):
        """Set ROI in canvas and synchronize ROI entry fields.

        Args:
            x: Left coordinate in image pixels.
            y: Top coordinate in image pixels.
            w: Width in pixels.
            h: Height in pixels.
        """
        self._update_roi_entries(x, y, w, h)
        self.image_canvas.set_roi((x, y, w, h))

    def _update_roi_entries(self, x, y, w, h, update_target=True):
        """Write ROI values into UI entry variables."""
        self._updating_roi_entries = True
        self.roi_x_var.set(str(x))
        self.roi_y_var.set(str(y))
        self.roi_w_var.set(str(w))
        self.roi_h_var.set(str(h))
        self._updating_roi_entries = False
        if update_target:
            self._update_target_size_entries(w, h)

    def _update_target_size_entries(self, w=None, h=None):
        """Keep target size entries aligned with the source ROI dimensions."""
        if getattr(self, "target_w_var", None) is None:
            return
        if w is None or h is None:
            try:
                w = int(self.roi_w_var.get())
                h = int(self.roi_h_var.get())
            except ValueError:
                return

        if w <= 0 or h <= 0:
            return

        final_w = w * CROP_SCALE_X
        final_h = h * CROP_SCALE_Y
        self._updating_target_size_entries = True
        self.target_w_var.set(str(final_w))
        self.target_h_var.set(str(final_h))
        self._updating_target_size_entries = False

    def _on_roi_changed(self, roi):
        """Handle ROI-change callback from the canvas interaction layer.

        Args:
            roi: Tuple `(x, y, w, h)` in image coordinates.
        """
        x, y, w, h = roi
        self._update_roi_entries(
            x,
            y,
            w,
            h,
            update_target=not self._target_size_edit_active,
        )

    def _on_mouse_moved(self, ix, iy, val):
        """Update status with cursor position/value for current image.

        Args:
            ix: X coordinate in image space.
            iy: Y coordinate in image space.
            val: Pixel value at `(ix, iy)`.
        """
        img = self.processed_image if self.processed_image is not None else self.raw_image
        if img is None:
            return
        ih, iw = img.shape[:2]
        z = self.image_canvas.get_zoom()
        self.status_var.set(
            f"({ix}, {iy})  val={val}  |  "
            f"Image: {iw}×{ih} {img.dtype}  |  Zoom: {z * 100:.0f}%"
        )

    # ── Processing ──
    def _crop_and_scale(self):
        """Crop raw image by ROI, apply pixel replication scaling, and display.

        Returns:
            bool: True on successful processing; False when blocked by missing
            prerequisites (image/ROI).
        """
        if self.raw_image is None:
            messagebox.showwarning("No image", "Open a raw file first.")
            return False

        roi = self.image_canvas.get_roi()
        if roi is None:
            messagebox.showwarning("No ROI", "Select a crop region first.")
            return False

        x, y, w, h = roi
        sx, sy = CROP_SCALE_X, CROP_SCALE_Y

        # Crop
        cropped = self.raw_image[y:y + h, x:x + w].copy()

        # Scale (pixel replication)
        scaled = scale_image(cropped, sx=sx, sy=sy)

        # Preview/result should match Analyze .img interpretation (signed 16-bit).
        if scaled.dtype != np.int16:
            scaled = scaled.astype(np.int16, copy=False)

        self.processed_image = np.ascontiguousarray(scaled)
        preview_image = self._to_uint8_preview(self.processed_image)

        # Show the result
        self.image_canvas.enable_roi(False)
        self.image_canvas.set_image(preview_image)
        self._update_zoom_label()
        self._set_sdb_parameters_enabled(False)

        if callable(self._on_processed_image):
            try:
                self._on_processed_image(np.array(self.processed_image, copy=True), self.current_file)
            except Exception:
                # Step 1 must remain usable even if Step 2 sync fails.
                pass

        ih, iw = self.processed_image.shape
        filename = os.path.basename(self.current_file) if self.current_file else "Processed"

        # Update the top info display
        self.image_info_var.set(
            f"✓ Processed: {filename}  |  Size: {iw} × {ih} px  |  "
            f"Type: {self.processed_image.dtype}  |  Range: [{self.processed_image.min()} – {self.processed_image.max()}]  |  "
            f"Cropped {w}×{h} from ({x},{y}), scaled ×{sx}/×{sy}"
        )
        self.status_var.set(
            f"Processed: {w}×{h} → {iw}×{ih}.  "
            f"Save as Light/Dark or Reset to adjust."
        )
        self._update_save_button_state()
        self.default_roi_btn.configure(state="disabled")
        self.entire_roi_btn.configure(state="disabled")
        for entry in self.roi_entries + self.target_size_entries:
            entry.configure(state="disabled")
        return True

    def _crop_scale_and_save_tiff(self):
        """Run crop+scale, then immediately prompt to save TIFF."""
        if self._crop_and_scale():
            self._save_tiff()

    def _reset(self):
        """Restore the loaded raw image view and re-enable ROI editing."""
        if self.raw_image is None:
            return
        self.processed_image = None
        self.image_canvas.set_image(self.raw_image)
        self.image_canvas.enable_roi(True)
        self._set_default_roi()
        self._update_zoom_label()
        self._update_save_button_state()
        self._set_sdb_parameters_enabled(True)
        self.default_roi_btn.configure(state="normal")
        self.entire_roi_btn.configure(state="normal")
        for entry in self.roi_entries + self.target_size_entries:
            entry.configure(state="normal")
        self.status_var.set("Reset — adjust ROI and process again.")

        # Restore the top info display to original image
        img = self.raw_image
        filename = os.path.basename(self.current_file) if self.current_file else "Image"
        self.image_info_var.set(
            f"{filename}  |  Size: {img.shape[1]} × {img.shape[0]} px  |  "
            f"Type: {img.dtype}  |  Range: [{img.min()} – {img.max()}]"
        )

    # ── Save ──
    def _browse_outdir(self):
        """Prompt for output directory used by Analyze saves."""
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.outdir_var.set(d)

    def _reset_outdir_to_default(self):
        """Reset output directory to default Desktop location."""
        self.outdir_var.set(DEFAULT_OUTPUT_DIR)
        self.status_var.set(f"Output directory reset to default: {DEFAULT_OUTPUT_DIR}")

    def _refresh_outdir_from_source(self):
        """Refresh output directory from the current source image path."""
        if self.current_file:
            self._sync_output_dir_with_source(self.current_file)
            self.status_var.set(f"Output directory synced to source: {self.outdir_var.get()}")
            return
        self.status_var.set("No source file loaded yet to sync output directory.")

    def _save_analyze(self, name):
        """Save processed image as Analyze 7.5 two-slice stack.

        Args:
            name: Output role label (for example, `"Light"` or `"Dark"`).
        """
        img = self.processed_image
        if img is None:
            messagebox.showwarning("Nothing to save",
                                   "Run 'Crop & Scale' first.")
            return

        outdir = self.outdir_var.get()
        if not os.path.isdir(outdir):
            messagebox.showerror("Error", f"Output folder does not exist:\n{outdir}")
            return

        base_name = self._build_output_name(name.lower())
        base = os.path.join(outdir, base_name)

        # Create a 2-slice stack (both slices identical) — matches ImageJ workflow
        stack = np.stack([img, img], axis=0)  # shape (2, H, W)

        try:
            hdr_path, img_path = write_analyze(base, stack)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Save error", str(exc))
            return

        messagebox.showinfo("Saved",
                            f"Saved {name} successfully:\n  {hdr_path}\n  {img_path}\n\n"
                            f"Stack: 2 slices of {img.shape[1]}×{img.shape[0]}  {img.dtype}")
        self.status_var.set(f"Saved → {hdr_path}")

    def _save_tiff(self):
        """Save the current processed image (or raw fallback) as TIFF."""
        img = self.processed_image if self.processed_image is not None else self.raw_image
        if img is None:
            messagebox.showwarning("Nothing to save", "Open a file first.")
            return
        default_name = f"{self._build_output_name('light')}.tif"
        path = filedialog.asksaveasfilename(
            title="Save as TIFF",
            defaultextension=".tif",
            initialfile=default_name,
            filetypes=[("TIFF", "*.tif *.tiff")],
        )
        if not path:
            return
        try:
            save_tiff(path, img)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Save error", str(exc))
            return
        self.status_var.set(f"TIFF saved → {path}")

    def _build_output_name(self, suffix):
        """Build output filename from source stem and user-provided suffix.

        Args:
            suffix: Trailing output token (for example `"light"` or `"crop"`).

        Returns:
            str: Filename stem without extension.
        """
        suffix = (suffix or "image").strip().lower()
        return suffix or "image"

    # ── Zoom ──
    def _zoom_in(self):
        """Increase canvas zoom by a fixed multiplier."""
        self.image_canvas.set_zoom(self.image_canvas.get_zoom() * 1.25)
        self._update_zoom_label()

    def _zoom_out(self):
        """Decrease canvas zoom by a fixed divisor."""
        self.image_canvas.set_zoom(self.image_canvas.get_zoom() / 1.25)
        self._update_zoom_label()

    def _fit_zoom(self):
        """Fit the current image into the visible canvas viewport."""
        self.image_canvas.fit_to_window()
        self._update_zoom_label()

    def _update_zoom_label(self):
        """Refresh the visible zoom percentage label."""
        z = self.image_canvas.get_zoom()
        self.zoom_lbl.configure(text=f"{z * 100:.0f} %")
