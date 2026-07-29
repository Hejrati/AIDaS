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

from aidas.canvas.image_canvas import ImageCanvas
from aidas.utils.filesystem import find_sdb_directories, skipped_directories_warning
from aidas.utils.io_utils import read_raw_oct, scale_image, write_analyze, save_tiff
from aidas.utils.ui_layout import COLORS, LAYOUT
from aidas.utils.ui_utils import (
    HoverToolTip,
    NativeNumericSpinbox,
    SidebarStepFrame,
    directory_row,
    load_ui_icon,
)

SDB_PREF_KEY = "sdb_dir"
SDB_DEFAULT_DIR = os.path.abspath(os.path.expanduser("~/Desktop"))
DEFAULT_RAW_WIDTH = 768
DEFAULT_RAW_HEIGHT = 1200
DEFAULT_RAW_OFFSET = 1050
DEFAULT_RAW_BIT_DEPTH = 16
CROP_SCALE_X = 3
CROP_SCALE_Y = 1
DEFAULT_ROI_X = 170
DEFAULT_ROI_Y = 585
DEFAULT_ROI_WIDTH = 491
DEFAULT_ROI_HEIGHT = 128
COMPLETED_ROW_BACKGROUND = "#dff3e4"
COMPLETED_ROW_FOREGROUND = "#1f6b35"
COMPLETED_ROW_SELECTED_BACKGROUND = "#2f7d4a"
SAVED_OUTPUT_FILENAMES = frozenset(("light.tif", "light.hdr", "light.img"))


class Step1Frame(SidebarStepFrame):
    """GUI panel for Step 1: Resize Raw OCT images.

    This view owns all Step 1 controls and state:
    - import parameters for reading `.sdb` data,
    - file discovery/navigation,
    - ROI definition and processing,
    - output saving (Analyze + TIFF),
    - image interaction (zoom/pan/inspection).
    """

    def __init__(
        self,
        parent,
        preferences=None,
        on_processed_image=None,
        on_batch_segment_folders=None,
    ):
        """Initialize the Step 1 panel and construct all widgets.

        Args:
            parent: Parent Tkinter container.
            preferences: Optional preferences object implementing `get` and `set`.
            on_processed_image: Optional callback receiving (image, source_path)
                whenever crop/scale produces a new processed image.
            on_batch_segment_folders: Optional callback receiving folders whose
                saved Light outputs should be batch-segmented in Step 2.
        """
        super().__init__(parent)

        self.preferences = preferences
        self._on_processed_image = on_processed_image
        self._on_batch_segment_folders = on_batch_segment_folders

        # ----- state -----
        self.raw_image = None          # original loaded image (H, W)  uint16
        self._source_raw_image = None  # original imported image before width adjustments
        self.processed_image = None    # after crop + scale           int16 (.img-like preview)
        self.current_file = None       # path of opened raw file
        self._source_roi = None        # ROI coordinates in the source image
        self.raw_import_params = None  # validated import parameters
        self._updating_roi_entries = False
        self._updating_target_size_entries = False
        self._target_size_edit_active = False
        self._syncing_view_roi = False
        self._sdb_directory_files = {}
        self._sdb_directories = []
        self._sdb_directory_labels = []
        self._sdb_files = []
        self._active_sdb_directory = None
        self._cropped_sdb_files = set()
        self._saved_output_directories = set()
        self._sdb_directory_selection_locked = False

        # ----- layout -----
        self.build_standard_layout()

        # Left — scrollable control panel (content-driven width)
        right = self.content

        # Right — processing toolbar + image canvas + ROI toolbar + status
        self.canvas_toolbar = ttk.Frame(
            right,
            style="AIDaS.ContentHeader.TFrame",
            padding=(LAYOUT.space_md, LAYOUT.space_sm),
        )
        self.canvas_toolbar.pack(fill="x", pady=(0, LAYOUT.space_sm))

        self.status_var = tk.StringVar(
            value="Ready — open an SDB raw OCT file to begin (left-drag ROI, right-drag pan)"
        )
        self.add_status_bar(self.status_var, parent=right)

        self.canvas_roi_toolbar = ttk.Frame(
            right,
            style="AIDaS.ContentHeader.TFrame",
            padding=(LAYOUT.space_md, LAYOUT.space_sm),
        )
        self.canvas_roi_toolbar.pack(
            side="bottom",
            fill="x",
            pady=(LAYOUT.space_sm, 0),
        )

        self.image_canvas = ImageCanvas(
            right,
            on_roi_change=self._on_roi_changed,
            on_mouse_move=self._on_mouse_moved,
        )
        self.image_canvas.pack(fill="both", expand=True)

        # Build control widgets
        self._build_controls()

    # ═══════════════════════════════════════════════════════════════════════
    #  Control-panel construction
    # ═══════════════════════════════════════════════════════════════════════
    def _build_controls(self):
        """Create and lay out the full left-side control panel."""
        numeric_vcmd = (self.register(self._validate_digits_only), "%P")

        # ── SDB Image Parameters ──
        # UX Improvement: Increased padding for better section separation
        self.sdb_params_section = self.add_sidebar_section("SDB Image Parameters", pady=(10, 5))
        self.sdb_params_frame = self.sdb_params_section.body
        self.sdb_params_frame.grid_columnconfigure(2, weight=1)

        self.width_var = tk.StringVar(value=str(DEFAULT_RAW_WIDTH))
        self.height_var = tk.StringVar(value=str(DEFAULT_RAW_HEIGHT))
        self.offset_var = tk.StringVar(value=str(DEFAULT_RAW_OFFSET))

        # Update these assignments in _build_controls
        self.width_stepper, self.width_reset_btn = self._param_stepper_row(
            self.sdb_params_frame, 0, "Width (px):", self.width_var, DEFAULT_RAW_WIDTH,
            step=1, minimum=1, maximum=10000, validatecommand=numeric_vcmd
        )
        
        self.height_stepper, self.height_reset_btn = self._param_stepper_row(
            self.sdb_params_frame, 1, "Height (px):", self.height_var, DEFAULT_RAW_HEIGHT,
            step=1, minimum=1, maximum=10000, validatecommand=numeric_vcmd
        )
        
        self.offset_stepper, self.offset_reset_btn = self._param_stepper_row(
            self.sdb_params_frame, 2, "Offset (bytes):", self.offset_var, DEFAULT_RAW_OFFSET,
            step=2, minimum=0, maximum=10_000_000, validatecommand=numeric_vcmd
        )

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
        self.endian_checkbox.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._endian_tooltip = HoverToolTip(self.endian_checkbox, "Can affect visualization for some offsets")

        # ── SDB Work Queue ──
        sdb_section = self.add_sidebar_section("SDB Work Queue", pady=(5, 5))
        sdb = sdb_section.body
        ttk.Label(sdb, text="Parent dir:").pack(anchor="w", pady=(0, 2))
        self.sdb_dir_var = tk.StringVar(value=self._initial_sdb_dir())
        
        dir_frame, _dir_entry, dir_buttons = directory_row(
            sdb,
            self,
            self.sdb_dir_var,
            self._browse_sdb_dir,
            home_command=self._reset_sdb_dir_to_default,
            refresh_command=lambda: self.refresh_sdb_list(preview_first=True),
            browse_tooltip="Choose parent folder to scan",
            home_tooltip="Reset to Desktop",
            refresh_tooltip="Rescan subfolders",
        )
        dir_frame.pack(fill="x", pady=(0, 8))
        self.search_btn = dir_buttons["browse"]
        self.home_btn = dir_buttons["home"]
        self.refresh_btn = dir_buttons["refresh"]

        # Search controls intentionally removed from the Step 1 UI.
        # filt_frame = ttk.Frame(sdb)
        # filt_frame.pack(fill="x", pady=(0, 4))
        # ttk.Label(filt_frame, text="Search:").pack(side="left")
        # self.sdb_filter_var = tk.StringVar(value="")
        # self.sdb_search_entry = ttk.Entry(
        #     filt_frame,
        #     textvariable=self.sdb_filter_var,
        # )
        # self.sdb_search_entry.pack(
        #     side="left", fill="x", expand=True, padx=(6, 0)
        # )

        self.sdb_scan_more_label = ttk.Label(
            sdb,
            text="",
            foreground="#0066cc",
            cursor="hand2",
        )
        self.sdb_scan_more_label.pack(anchor="w", pady=(0, 4))
        self.sdb_scan_tooltip = HoverToolTip(self.sdb_scan_more_label, "")

        self.exclude_saved_outputs_var = tk.BooleanVar(value=False)
        self.exclude_saved_outputs_checkbox = ttk.Checkbutton(
            sdb,
            text="Exclude saved cropped folders",
            variable=self.exclude_saved_outputs_var,
            command=self._render_sdb_directories,
        )
        self.exclude_saved_outputs_checkbox.pack(anchor="w", pady=(0, 4))
        HoverToolTip(
            self.exclude_saved_outputs_checkbox,
            "Hide SDB folders containing light.tif, light.hdr, and light.img",
        )

        folder_header = ttk.Frame(sdb)
        folder_header.pack(fill="x", pady=(2, 2))
        ttk.Label(folder_header, text="Folders containing SDB images:").pack(side="left")
        folder_counts = ttk.Frame(folder_header)
        folder_counts.pack(side="right")
        self.cropped_folder_count_var = tk.StringVar(value="Cropped 0")
        self.uncropped_folder_count_var = tk.StringVar(value="Uncropped 0")
        ttk.Label(
            folder_counts,
            textvariable=self.cropped_folder_count_var,
            foreground=COMPLETED_ROW_FOREGROUND,
            font=("Segoe UI", 8),
        ).pack(side="left")
        ttk.Label(
            folder_counts,
            textvariable=self.uncropped_folder_count_var,
            foreground=COLORS.muted_text,
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(8, 0))
        directory_list_frame = ttk.Frame(sdb)
        directory_list_frame.pack(fill="both", expand=True, pady=(0, 5))
        self.sdb_directory_listbox = tk.Listbox(
            directory_list_frame,
            height=6,
            selectmode="browse",
            relief="flat",
            highlightthickness=1,
        )
        directory_scroll = ttk.Scrollbar(
            directory_list_frame,
            orient="vertical",
            command=self.sdb_directory_listbox.yview,
        )
        self.sdb_directory_listbox.configure(yscrollcommand=directory_scroll.set)
        self.sdb_directory_listbox.pack(side="left", fill="both", expand=True)
        directory_scroll.pack(side="right", fill="y")
        self.sdb_directory_listbox.bind("<<ListboxSelect>>", self._on_sdb_directory_select)

        image_header = ttk.Frame(sdb)
        image_header.pack(fill="x", pady=(2, 2))
        ttk.Label(image_header, text="Images in selected folder:").pack(side="left")
        image_counts = ttk.Frame(image_header)
        image_counts.pack(side="right")
        self.cropped_image_count_var = tk.StringVar(value="Cropped 0")
        self.uncropped_image_count_var = tk.StringVar(value="Uncropped 0")
        ttk.Label(
            image_counts,
            textvariable=self.cropped_image_count_var,
            foreground=COMPLETED_ROW_FOREGROUND,
            font=("Segoe UI", 8),
        ).pack(side="left")
        ttk.Label(
            image_counts,
            textvariable=self.uncropped_image_count_var,
            foreground=COLORS.muted_text,
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(8, 0))
        lb_frame = ttk.Frame(sdb)
        lb_frame.pack(fill="both", expand=True, pady=(0, 6))
        self.sdb_listbox = tk.Listbox(lb_frame, height=5, selectmode="browse", relief="flat", highlightthickness=1)
        lb_scroll = ttk.Scrollbar(lb_frame, orient="vertical", command=self.sdb_listbox.yview)
        self.sdb_listbox.configure(yscrollcommand=lb_scroll.set)
        self.sdb_listbox.pack(side="left", fill="both", expand=True)
        lb_scroll.pack(side="right", fill="y")
        self.sdb_listbox.bind("<Double-1>", lambda e: self._open_selected_sdb())
        self.sdb_listbox.bind("<<ListboxSelect>>", self._on_sdb_list_select)

        # Previous/Next navigation buttons intentionally removed. Images are
        # selected directly from the list above.
        # nav_frame = ttk.Frame(sdb)
        # nav_frame.pack(fill="x")
        # ttk.Button(nav_frame, text="◀ Prev", command=self._prev_sdb).pack(
        #     side="left", expand=True, fill="x", padx=(0, 4)
        # )
        # ttk.Button(nav_frame, text="Next ▶", command=self._next_sdb).pack(
        #     side="right", expand=True, fill="x", padx=(4, 0)
        # )
        ttk.Separator(self.ctrl, orient="horizontal").pack(fill="x", pady=(10, 6))
        self.batch_segment_cropped_btn = ttk.Button(
            self.ctrl,
            text="Go to Step 2 >>",
            command=self._send_cropped_folders_to_step2,
            state="disabled",
        )
        self.batch_segment_cropped_btn.pack(fill="x", padx=2, pady=(0, 8))
        HoverToolTip(
            self.batch_segment_cropped_btn,
            "Open folders with saved Light outputs in Step 2 batch segmentation",
        )

        self.refresh_sdb_list()

        # ── ROI controls below the image canvas ──
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

        roi_toolbar = self.canvas_roi_toolbar
        roi_fields = (
            ("X (Left)", self.roi_x_var, 1, 0, 10000),
            ("Y (Top)", self.roi_y_var, 1, 0, 10000),
            ("Width", self.roi_w_var, 1, 1, 30000),
            ("Height", self.roi_h_var, 1, 1, 30000),
        )
        roi_steppers = []
        for index, (label, var, step, minimum, maximum) in enumerate(roi_fields):
            field = ttk.Frame(roi_toolbar, style="AIDaS.ContentHeader.TFrame")
            field.pack(side="left", padx=(0, 12))
            ttk.Label(
                field,
                text=label,
                style="AIDaS.ContentHeader.TLabel",
            ).pack(anchor="w", pady=(0, 2))
            stepper = self._numeric_stepper(
                field,
                var,
                width=5,
                step=step,
                minimum=minimum,
                maximum=maximum,
                validatecommand=numeric_vcmd,
            )
            stepper.pack(fill="x")
            roi_steppers.append(stepper)

        self.roi_entries = roi_steppers
        self.target_size_entries = []

        ttk.Separator(roi_toolbar, orient="vertical").pack(
            side="left", fill="y", padx=(0, 12), pady=2
        )
        view_group = ttk.Frame(
            roi_toolbar,
            style="AIDaS.ContentHeader.TFrame",
        )
        view_group.pack(side="left")
        ttk.Label(
            view_group,
            text="View",
            style="AIDaS.ContentHeader.TLabel",
        ).pack(anchor="w", pady=(0, 2))
        view_choices = ttk.Frame(
            view_group,
            style="AIDaS.ContentHeader.TFrame",
        )
        view_choices.pack(anchor="w")
        self.view_mode_var = tk.StringVar(value="source")
        self.source_view_radio = ttk.Radiobutton(
            view_choices,
            text="Source",
            value="source",
            variable=self.view_mode_var,
            command=self._on_view_mode_changed,
        )
        self.source_view_radio.pack(side="left", padx=(0, 8))
        self.target_view_radio = ttk.Radiobutton(
            view_choices,
            text="Target",
            value="target",
            variable=self.view_mode_var,
            command=self._on_view_mode_changed,
            state="disabled",
        )
        self.target_view_radio.pack(side="left")

        # Top canvas toolbar: ROI presets, processing actions, and save.
        toolbar = self.canvas_toolbar
        action_icon_size = 16
        self.crop_split_frame = tk.Frame(
            toolbar,
            background="#a7adb3",
            borderwidth=1,
            relief="solid",
        )
        self.crop_split_frame.pack(side="left", padx=(0, 6))
        self.crop_btn_icon = load_ui_icon(
            self, "material-symbols-light--crop.png", size=action_icon_size
        )
        paint_button_options = {
            "background": "#f7f8fa",
            "activebackground": "#e5f3fb",
            "foreground": COLORS.text,
            "activeforeground": COLORS.text,
            "disabledforeground": "#8b9298",
            "font": ("Segoe UI", 9),
            "relief": "flat",
            "borderwidth": 0,
            "highlightthickness": 0,
            "cursor": "hand2",
            "pady": 8,
        }
        self.crop_btn = tk.Button(
            self.crop_split_frame,
            text="Crop & Scale",
            command=self._crop_and_scale,
            image=self.crop_btn_icon,
            compound="left",
            padx=9,
            **paint_button_options,
        )
        self.crop_btn.pack(side="left", fill="y")

        self.crop_split_divider = tk.Frame(
            self.crop_split_frame,
            width=1,
            background="#a7adb3",
        )
        self.crop_split_divider.pack(side="left", fill="y")

        self.crop_options_icon = tk.PhotoImage(master=self, width=16, height=16)
        self.crop_options_btn = tk.Button(
            self.crop_split_frame,
            text="▼",
            image=self.crop_options_icon,
            compound="center",
            state="disabled",
            command=self._show_crop_options_menu,
            padx=5,
            **paint_button_options,
        )
        self.crop_options_menu = tk.Menu(self.crop_options_btn, tearoff=False)
        self.crop_options_menu.add_command(
            label="Default Region",
            command=self._set_default_roi,
        )
        self.crop_options_menu.add_command(
            label="Entire Image",
            command=self._select_all_roi,
        )
        self.crop_options_btn.pack(side="left")
        HoverToolTip(self.crop_options_btn, "Choose a crop region preset")

        self.undo_crop_btn_icon = load_ui_icon(
            self, "grommet-icons--revert.png", size=action_icon_size
        )
        self.undo_crop_btn = ttk.Button(
            toolbar,
            text="Undo",
            command=self._reset,
            state="disabled",
            image=self.undo_crop_btn_icon,
            compound="left",
        )
        self.undo_crop_btn.pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(
            side="left", fill="y", padx=10, pady=2
        )

        self.save_all_btn_icon = load_ui_icon(
            self, "ic--baseline-save.png", size=action_icon_size
        )
        self.save_all_btn = ttk.Button(
            toolbar,
            text="Save",
            command=self._save_all_formats,
            state="disabled",
            image=self.save_all_btn_icon,
            compound="left",
        )
        self.save_all_btn.pack(side="left")
        HoverToolTip(self.save_all_btn, "Save TIFF, IMG, and HDR beside the source SDB image")

        self.save_as_btn = ttk.Button(
            toolbar,
            text="Save As...",
            command=self._save_as,
            state="disabled",
            image=self.save_all_btn_icon,
            compound="left",
        )
        self.save_as_btn.pack(side="left", padx=(6, 0))
        HoverToolTip(self.save_as_btn, "Save the cropped image to a location you choose")

        # # ── View ──
        # view_section = self.add_sidebar_section("View", padding=3, pady=(2, 6))
        # view = view_section.body

        # zf = ttk.Frame(view)
        # zf.pack(fill="x")
        # ttk.Button(zf, text="−", width=3, command=self._zoom_out).pack(side="left")
        # self.zoom_lbl = ttk.Label(zf, text="100 %", anchor="center")
        # self.zoom_lbl.pack(side="left", expand=True)
        # ttk.Button(zf, text="+", width=3, command=self._zoom_in).pack(side="right")
        # ttk.Button(view, text="Fit to Window",
        #            command=self._fit_zoom).pack(fill="x", pady=2)

    @staticmethod
    def _numeric_stepper(parent, var, *, width=8, step=1, minimum=0, maximum=10_000_000, validatecommand=None):
        return NativeNumericSpinbox(
            parent,
            var,
            width=width,
            step=step,
            minimum=minimum,
            maximum=maximum,
            validatecommand=validatecommand,
        )

    def _show_crop_options_menu(self):
        """Open the crop preset menu directly below the split-button arrow."""
        button = self.crop_options_btn
        try:
            self.crop_options_menu.tk_popup(
                button.winfo_rootx(),
                button.winfo_rooty() + button.winfo_height(),
            )
        finally:
            self.crop_options_menu.grab_release()

    def _param_stepper_row(self, parent, row, label, var, default_value, *, step=1, minimum=0, maximum=10_000_000, validatecommand=None):
        """Creates a modern, unified stepper with embedded +/- buttons and a reset icon."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        
        # --- The Main Wrapper Frame ---
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=1, sticky="e", pady=4)

        stepper = NativeNumericSpinbox(
            row_frame,
            var,
            width=6,
            step=step,
            minimum=minimum,
            maximum=maximum,
            validatecommand=validatecommand,
        )
        stepper.pack(side="left", padx=0, anchor="center")

        btn_icon = load_ui_icon(self, "material-symbols-light--refresh-rounded.png")
        reset_btn = tk.Button(
            row_frame,
            image=btn_icon,
            bd=0,
            relief="flat",
            highlightthickness=0,
            cursor="hand2",
            command=lambda: self._reset_numeric_var(var, default_value),
        )
        reset_btn.image = btn_icon
        reset_btn.pack(side="left", padx=(8, 0), anchor="center")

        return stepper, reset_btn


    def _save_all_formats(self):
        """Save TIFF, HDR, and IMG beside the current source SDB image."""
        img = self.processed_image
        if img is None:
            messagebox.showwarning("Nothing to save", "Run 'Crop & Scale' first.")
            return
        outdir = self._source_output_directory()
        if outdir is None:
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

        self._saved_output_directories.add(self._path_key(outdir))
        self._render_sdb_directories()
        self._update_batch_handoff_button_state()

    def _save_as(self):
        """Save the cropped image to a user-selected TIFF or Analyze path."""
        img = self.processed_image
        if img is None:
            messagebox.showwarning("Nothing to save", "Run 'Crop & Scale' first.")
            return

        initial_dir = self._source_output_directory()
        if initial_dir is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save cropped image as",
            initialdir=initial_dir,
            initialfile="light.tif",
            defaultextension=".tif",
            filetypes=(
                ("TIFF image", "*.tif"),
                ("Analyze 7.5 image", "*.img"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        root, extension = os.path.splitext(path)
        try:
            if extension.lower() == ".img":
                stack = np.stack([img, img], axis=0)
                hdr_path, img_path = write_analyze(root, stack)
                saved_paths = (hdr_path, img_path)
            else:
                if extension.lower() not in {".tif", ".tiff"}:
                    path = root + ".tif"
                save_tiff(path, img)
                saved_paths = (path,)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Save As error", str(exc))
            return

        messagebox.showinfo(
            "Saved As",
            "Saved cropped image successfully:\n  " + "\n  ".join(saved_paths),
        )
        self.status_var.set("Saved as: " + ", ".join(saved_paths))
        saved_directory = os.path.dirname(os.path.abspath(saved_paths[0]))
        if self._folder_has_all_saved_outputs(saved_directory):
            self._saved_output_directories.add(self._path_key(saved_directory))
            self._render_sdb_directories()

    def _update_save_button_state(self):
        """Sync Save/Undo button states with processed image availability."""
        if getattr(self, "save_all_btn", None) is None:
            return
        has_processed = self.processed_image is not None
        self.save_all_btn.configure(state="normal" if has_processed else "disabled")
        if getattr(self, "save_as_btn", None) is not None:
            self.save_as_btn.configure(state="normal" if has_processed else "disabled")
        if getattr(self, "undo_crop_btn", None) is not None:
            self.undo_crop_btn.configure(state="normal" if has_processed else "disabled")
        if getattr(self, "crop_btn", None) is not None:
            self.crop_btn.configure(state="disabled" if has_processed else "normal")
        if getattr(self, "crop_options_btn", None) is not None:
            options_enabled = self.raw_image is not None and not has_processed
            self.crop_options_btn.configure(
                state="normal" if options_enabled else "disabled"
            )
        if getattr(self, "target_view_radio", None) is not None:
            self.target_view_radio.configure(
                state="normal" if self.raw_image is not None else "disabled"
            )
        if getattr(self, "source_view_radio", None) is not None:
            source_enabled = self.raw_image is not None and not has_processed
            self.source_view_radio.configure(
                state="normal" if source_enabled else "disabled"
            )

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
        if self.preferences is not None:
            self.preferences.set(SDB_PREF_KEY, directory)

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
        self.view_mode_var.set("source")
        self.image_canvas.set_image(adjusted)
        self.image_canvas.enable_roi(True)
        self._set_default_roi()
        self._update_zoom_label()
        self._update_save_button_state()
        self.status_var.set(note)

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
        self.view_mode_var.set("source")

        filename = os.path.basename(path)

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

        try:
            img = read_raw_oct(path, **self.raw_import_params)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Error reading file", str(exc))
            return

        self._load_image(img, path)

    def _browse_sdb_dir(self):
        """Prompt for a parent folder and recursively discover SDB folders."""
        directory = filedialog.askdirectory(
            title="Select parent folder containing SDB subfolders",
            initialdir=self.sdb_dir_var.get() or None,
        )
        if directory:
            self.set_sdb_directory(directory)
            self.refresh_sdb_list(preview_first=True)

    def _reset_sdb_dir_to_default(self):
        """Reset SDB directory to default (Desktop) and refresh list."""
        target_dir = SDB_DEFAULT_DIR
        self.set_sdb_directory(target_dir)
        self.refresh_sdb_list(preview_first=True)
        self.status_var.set(f"SDB directory reset to default: {target_dir}")

    def refresh_sdb_list(self, preview_first=False):
        """Recursively scan the selected parent and rebuild the work queue."""
        previous_directory = self._active_sdb_directory
        self._active_sdb_directory = None
        self._sdb_directory_files.clear()
        self._saved_output_directories.clear()
        self.sdb_scan_more_label.configure(text="")
        self.sdb_scan_tooltip.text = ""
        root = self.sdb_dir_var.get()
        if not root or not os.path.isdir(root):
            self._render_sdb_directories()
            return

        try:
            directory_files, errors = find_sdb_directories(root)
        except OSError as exc:
            self.status_var.set(f"Could not scan SDB parent directory: {root}")
            self.sdb_scan_more_label.configure(text="More")
            self.sdb_scan_tooltip.text = skipped_directories_warning([(root, str(exc))])
            self._render_sdb_directories()
            return

        self._sdb_directory_files = {
            os.path.abspath(str(directory)): [os.path.abspath(str(path)) for path in files]
            for directory, files in directory_files.items()
        }
        self._saved_output_directories = {
            self._path_key(directory)
            for directory in self._sdb_directory_files
            if self._folder_has_all_saved_outputs(directory)
        }
        if errors:
            self.sdb_scan_more_label.configure(text="Some folders skipped — More")
            self.sdb_scan_tooltip.text = skipped_directories_warning(errors)
        if previous_directory in self._sdb_directory_files:
            self._active_sdb_directory = previous_directory

        self._render_sdb_directories()
        self.status_var.set(
            f"Found {len(self._sdb_directory_files)} folder(s) containing SDB images below {root}"
        )
        if preview_first:
            self._preview_selected_sdb()

    def _render_sdb_directories(self):
        """Render the filtered directory list while preserving its selection."""
        if not hasattr(self, "sdb_directory_listbox"):
            return

        root = os.path.abspath(self.sdb_dir_var.get() or os.curdir)
        directories = []
        for directory, files in self._sdb_directory_files.items():
            if (
                getattr(self, "exclude_saved_outputs_var", None) is not None
                and self.exclude_saved_outputs_var.get()
                and self._directory_has_saved_outputs(directory)
            ):
                continue
            relative = os.path.relpath(directory, root)
            label = "(selected folder)" if relative == os.curdir else relative
            directories.append((directory, label))

        directories.sort(key=lambda item: item[1].lower())
        self._sdb_directories = [directory for directory, _label in directories]
        self._sdb_directory_labels = [label for _directory, label in directories]
        self._update_sdb_folder_count_summary()

        if not self._sdb_directories:
            self._active_sdb_directory = None
            self.sdb_directory_listbox.delete(0, "end")
            self._populate_sdb_files(None)
            self._update_batch_handoff_button_state()
            return

        if self._active_sdb_directory not in self._sdb_directories:
            self._active_sdb_directory = self._sdb_directories[0]
        self._refresh_sdb_directory_rows()
        self._populate_sdb_files(self._active_sdb_directory)
        self._update_batch_handoff_button_state()

    def _refresh_sdb_directory_rows(self):
        """Draw the active-folder symbol and completion colors."""
        self._sdb_directory_selection_locked = True
        self.sdb_directory_listbox.delete(0, "end")
        for index, (directory, label) in enumerate(
            zip(self._sdb_directories, self._sdb_directory_labels)
        ):
            marker = "▶" if directory == self._active_sdb_directory else " "
            self.sdb_directory_listbox.insert(
                "end",
                f"{marker} {label}",
            )
            if self._directory_is_complete(directory):
                self._style_completed_row(self.sdb_directory_listbox, index)

        if self._active_sdb_directory in self._sdb_directories:
            selected_index = self._sdb_directories.index(self._active_sdb_directory)
            self.sdb_directory_listbox.selection_set(selected_index)
            self.sdb_directory_listbox.see(selected_index)
        self.after_idle(self._unlock_sdb_directory_selection)

    def _unlock_sdb_directory_selection(self):
        self._sdb_directory_selection_locked = False

    def _on_sdb_directory_select(self, _event=None):
        """Show SDB images belonging to the selected work-queue folder."""
        if self._sdb_directory_selection_locked:
            return
        selection = self.sdb_directory_listbox.curselection()
        if not selection:
            return
        self._active_sdb_directory = self._sdb_directories[selection[0]]
        self._refresh_sdb_directory_rows()
        self._populate_sdb_files(self._active_sdb_directory, select_first=True)
        self._preview_selected_sdb()

    def _populate_sdb_files(self, directory, select_first=False):
        """Populate the image list for one selected directory."""
        self.sdb_listbox.delete(0, "end")
        self._sdb_files = list(self._sdb_directory_files.get(directory, []))
        folder_saved = self._directory_has_saved_outputs(directory)
        for index, path in enumerate(self._sdb_files):
            self.sdb_listbox.insert("end", os.path.basename(path))
            if folder_saved or self._path_key(path) in self._cropped_sdb_files:
                self._style_completed_row(self.sdb_listbox, index)
        if self._sdb_files and (select_first or self.current_file not in self._sdb_files):
            self.sdb_listbox.selection_set(0)
            self.sdb_listbox.see(0)
        elif self.current_file in self._sdb_files:
            index = self._sdb_files.index(self.current_file)
            self.sdb_listbox.selection_set(index)
            self.sdb_listbox.see(index)
        self._update_sdb_image_count_summary()

    def _update_sdb_folder_count_summary(self):
        """Update completed/remaining counts for all discovered SDB folders."""
        if not hasattr(self, "cropped_folder_count_var"):
            return
        total = len(self._sdb_directory_files)
        cropped = sum(
            self._directory_is_complete(directory)
            for directory in self._sdb_directory_files
        )
        self.cropped_folder_count_var.set(f"Cropped {cropped}")
        self.uncropped_folder_count_var.set(f"Uncropped {total - cropped}")

    def _update_sdb_image_count_summary(self):
        """Update cropped/uncropped counts for the selected SDB folder."""
        if not hasattr(self, "cropped_image_count_var"):
            return
        total = len(self._sdb_files)
        if self._directory_has_saved_outputs(self._active_sdb_directory):
            cropped = total
        else:
            cropped = sum(
                self._path_key(path) in self._cropped_sdb_files
                for path in self._sdb_files
            )
        self.cropped_image_count_var.set(f"Cropped {cropped}")
        self.uncropped_image_count_var.set(f"Uncropped {total - cropped}")

    def _preview_selected_sdb(self):
        """Open the selected folder's first/selected SDB image in the canvas."""
        selection = self.sdb_listbox.curselection()
        if not selection:
            return False
        path = self._sdb_files[selection[0]]
        if self.current_file and self._path_key(path) == self._path_key(self.current_file):
            return True
        self._open_raw(path=path)
        return bool(
            self.current_file
            and self._path_key(path) == self._path_key(self.current_file)
        )

    @staticmethod
    def _path_key(path):
        """Return a stable, case-insensitive key for progress tracking."""
        return os.path.normcase(os.path.abspath(path))

    def _directory_is_complete(self, directory):
        files = self._sdb_directory_files.get(directory, [])
        return self._directory_has_saved_outputs(directory) or (
            bool(files)
            and all(self._path_key(path) in self._cropped_sdb_files for path in files)
        )

    @staticmethod
    def _folder_has_all_saved_outputs(directory):
        """Return whether a folder contains the complete Step 1 Light output set."""
        try:
            with os.scandir(directory) as entries:
                filenames = {
                    entry.name.lower()
                    for entry in entries
                    if entry.is_file()
                }
        except OSError:
            return False
        return SAVED_OUTPUT_FILENAMES.issubset(filenames)

    def _completed_output_folders(self):
        return [
            directory
            for directory in self._sdb_directory_files
            if self._directory_has_saved_outputs(directory)
        ]

    def _directory_has_saved_outputs(self, directory):
        if not directory:
            return False
        return self._path_key(directory) in getattr(
            self, "_saved_output_directories", set()
        )

    def _update_batch_handoff_button_state(self):
        button = getattr(self, "batch_segment_cropped_btn", None)
        if button is None:
            return
        enabled = callable(self._on_batch_segment_folders) and bool(
            self._completed_output_folders()
        )
        button.configure(state="normal" if enabled else "disabled")

    def _send_cropped_folders_to_step2(self):
        """Start Step 2 batch segmentation for all completed Step 1 folders."""
        folders = self._completed_output_folders()
        if not folders:
            messagebox.showwarning(
                "No cropped folders",
                "No SDB folder contains light.tif, light.hdr, and light.img yet.",
            )
            return
        if not callable(self._on_batch_segment_folders):
            messagebox.showerror("Step 2 unavailable", "Step 2 batch segmentation is unavailable.")
            return
        self._on_batch_segment_folders(folders)

    @staticmethod
    def _style_completed_row(listbox, index):
        listbox.itemconfigure(
            index,
            background=COMPLETED_ROW_BACKGROUND,
            foreground=COMPLETED_ROW_FOREGROUND,
            selectbackground=COMPLETED_ROW_SELECTED_BACKGROUND,
            selectforeground="#ffffff",
        )

    def _refresh_sdb_progress_colors(self):
        """Repaint image and folder rows after crop progress changes."""
        for index, path in enumerate(self._sdb_files):
            if self._path_key(path) in self._cropped_sdb_files:
                self._style_completed_row(self.sdb_listbox, index)
        if (
            self._active_sdb_directory in self._sdb_directories
            and self._directory_is_complete(self._active_sdb_directory)
        ):
            index = self._sdb_directories.index(self._active_sdb_directory)
            self._style_completed_row(self.sdb_directory_listbox, index)
        self._update_sdb_folder_count_summary()
        self._update_sdb_image_count_summary()

    def _on_sdb_list_select(self, _event):
        """Update status text when the SDB list selection changes."""
        sel = self.sdb_listbox.curselection()
        if not sel:
            return
        path = self._sdb_files[sel[0]]
        state = "cropped" if self._path_key(path) in self._cropped_sdb_files else "not cropped"
        self.status_var.set(
            f"Selected SDB: {os.path.basename(path)}  ({state}; double-click to open)"
        )

    def _open_selected_sdb(self):
        """Open the currently selected SDB file from the list."""
        sel = self.sdb_listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Select an SDB file from the list first.")
            return
        self._open_raw(path=self._sdb_files[sel[0]])

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
        roi = self._view_to_source_roi((x, y, w, h))
        self._set_roi_and_entries(*roi)

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
        roi = self._clamp_source_roi(
            *self._view_to_source_roi((x, y, w, h))
        )
        if roi is None:
            return
        self._source_roi = roi
        self._update_roi_entries(*roi)
        self._set_canvas_roi_from_source()

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
        roi = self._clamp_source_roi(x, y, w, h)
        if roi is None:
            return
        self._source_roi = roi
        self._update_roi_entries(*roi)
        self._set_canvas_roi_from_source()

    def _source_to_view_roi(self, roi):
        """Convert source ROI coordinates to the active view coordinates."""
        x, y, w, h = roi
        if self.view_mode_var.get() == "target":
            return (
                x * CROP_SCALE_X,
                y * CROP_SCALE_Y,
                w * CROP_SCALE_X,
                h * CROP_SCALE_Y,
            )
        return x, y, w, h

    def _view_to_source_roi(self, roi):
        """Convert active-view ROI coordinates back to source coordinates."""
        x, y, w, h = roi
        if self.view_mode_var.get() == "target":
            return (
                int(round(x / CROP_SCALE_X)),
                int(round(y / CROP_SCALE_Y)),
                max(1, int(round(w / CROP_SCALE_X))),
                max(1, int(round(h / CROP_SCALE_Y))),
            )
        return int(x), int(y), int(w), int(h)

    def _set_canvas_roi_from_source(self):
        """Show the shared source ROI using the active view's coordinates."""
        if self._source_roi is None or self.processed_image is not None:
            return
        self._syncing_view_roi = True
        try:
            self.image_canvas.set_roi(self._source_to_view_roi(self._source_roi))
        finally:
            self._syncing_view_roi = False

    def _clamp_source_roi(self, x, y, w, h):
        """Clamp ROI coordinates against the source image dimensions."""
        if self.raw_image is None:
            return None
        image_height, image_width = self.raw_image.shape[:2]
        x = max(0, min(int(x), max(0, image_width - 1)))
        y = max(0, min(int(y), max(0, image_height - 1)))
        w = max(1, min(int(w), image_width - x))
        h = max(1, min(int(h), image_height - y))
        return x, y, w, h

    def _update_roi_entries(self, x, y, w, h, update_target=True):
        """Write ROI values into UI entry variables."""
        display_x, display_y, display_w, display_h = self._source_to_view_roi(
            (x, y, w, h)
        )
        self._updating_roi_entries = True
        self.roi_x_var.set(str(display_x))
        self.roi_y_var.set(str(display_y))
        self.roi_w_var.set(str(display_w))
        self.roi_h_var.set(str(display_h))
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
        if self._syncing_view_roi:
            return
        self._source_roi = self._clamp_source_roi(
            *self._view_to_source_roi(tuple(roi))
        )
        if self._source_roi is None:
            return
        x, y, w, h = self._source_roi
        self._update_roi_entries(
            x,
            y,
            w,
            h,
            update_target=not self._target_size_edit_active,
        )

    def _build_target_view_image(self):
        """Build the full scaled target work view before cropping."""
        if self.raw_image is None:
            return None
        return np.ascontiguousarray(
            scale_image(self.raw_image, sx=CROP_SCALE_X, sy=CROP_SCALE_Y)
        )

    def _build_processed_target_image(self):
        """Build the final cropped/scaled target from the shared source ROI."""
        if self.raw_image is None or self._source_roi is None:
            return None
        x, y, w, h = self._source_roi
        cropped = self.raw_image[y:y + h, x:x + w].copy()
        target = scale_image(cropped, sx=CROP_SCALE_X, sy=CROP_SCALE_Y)
        if target.dtype != np.int16:
            target = target.astype(np.int16, copy=False)
        return np.ascontiguousarray(target)

    def _on_view_mode_changed(self):
        """Switch editable source/target work views or show the final target."""
        mode = self.view_mode_var.get()
        if self.processed_image is not None and mode != "target":
            self.view_mode_var.set("target")
            mode = "target"
        if mode == "target":
            image = (
                self.processed_image
                if self.processed_image is not None
                else self._build_target_view_image()
            )
            if image is None:
                self.view_mode_var.set("source")
                return
            self.image_canvas.set_image(image)
            self.image_canvas.enable_roi(self.processed_image is None)
            if self.processed_image is None:
                self._set_canvas_roi_from_source()
            label = "Target result" if self.processed_image is not None else "Target work"
        else:
            if self.raw_image is None:
                return
            self.image_canvas.set_image(self.raw_image)
            self.image_canvas.enable_roi(self.processed_image is None)
            self._set_canvas_roi_from_source()
            image = self.raw_image
            label = "Source"

        if self._source_roi is not None:
            self._update_roi_entries(*self._source_roi)
        self._update_zoom_label()
        height, width = image.shape[:2]
        self.status_var.set(f"{label} view — {width}×{height} {image.dtype}")

    def _on_mouse_moved(self, ix, iy, val):
        """Update status with cursor position/value for current image.

        Args:
            ix: X coordinate in image space.
            iy: Y coordinate in image space.
            val: Pixel value at `(ix, iy)`.
        """
        img = self.image_canvas.get_image()
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

        roi = self._source_roi
        if roi is None:
            messagebox.showwarning("No ROI", "Select a crop region first.")
            return False

        x, y, w, h = roi
        self.processed_image = self._build_processed_target_image()
        if self.processed_image is None:
            messagebox.showwarning("No ROI", "Select a crop region first.")
            return False
        self.view_mode_var.set("target")
        # Show the true 16-bit processed image. ImageCanvas creates its own
        # display-only 8-bit view without changing the stored annotation data.
        self.image_canvas.enable_roi(False)
        self.image_canvas.set_image(self.processed_image)
        self._update_zoom_label()
        self._set_sdb_parameters_enabled(False)

        if callable(self._on_processed_image):
            try:
                self._on_processed_image(np.array(self.processed_image, copy=True), self.current_file)
            except Exception:
                # Step 1 must remain usable even if Step 2 sync fails.
                pass

        ih, iw = self.processed_image.shape
        self.status_var.set(
            f"Processed: {w}×{h} → {iw}×{ih}.  "
            f"Save as Light/Dark or Reset to adjust."
        )
        if self.current_file:
            self._cropped_sdb_files.add(self._path_key(self.current_file))
            self._refresh_sdb_progress_colors()
        self._update_save_button_state()
        for entry in self.roi_entries + self.target_size_entries:
            entry.configure(state="disabled")
        return True

    def _crop_scale_and_save_tiff(self):
        """Run crop+scale, then save TIFF beside the source SDB image."""
        if self._crop_and_scale():
            self._save_tiff()

    def _reset(self):
        """Restore the loaded raw image view and re-enable ROI editing."""
        if self.raw_image is None:
            return
        self.processed_image = None
        self.view_mode_var.set("source")
        self.image_canvas.set_image(self.raw_image)
        self.image_canvas.enable_roi(True)
        self._set_default_roi()
        self._update_zoom_label()
        self._update_save_button_state()
        self._set_sdb_parameters_enabled(True)
        for entry in self.roi_entries + self.target_size_entries:
            entry.configure(state="normal")
        self.status_var.set("Reset — adjust ROI and process again.")

    # ── Save ──
    def _source_output_directory(self):
        """Return the current SDB image's folder for automatic saves."""
        if not self.current_file:
            messagebox.showwarning("No source image", "Open an SDB image first.")
            return None
        directory = os.path.dirname(os.path.abspath(self.current_file))
        if not os.path.isdir(directory):
            messagebox.showerror(
                "Invalid source folder",
                f"The source image folder does not exist:\n{directory}",
            )
            return None
        return directory

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

        outdir = self._source_output_directory()
        if outdir is None:
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
        """Save the current image as TIFF beside its source SDB image."""
        show_target = (
            self.view_mode_var.get() == "target"
            and self.processed_image is not None
        )
        img = self.processed_image if show_target else self.raw_image
        if img is None:
            messagebox.showwarning("Nothing to save", "Open a file first.")
            return
        outdir = self._source_output_directory()
        if outdir is None:
            return
        path = os.path.join(outdir, f"{self._build_output_name('light')}.tif")
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
        if hasattr(self, "zoom_lbl"):
            self.zoom_lbl.configure(text=f"{z * 100:.0f} %")
