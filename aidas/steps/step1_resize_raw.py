"""Step 1 — Load, Resize & Crop Raw OCT Images.

Replicates the ImageJ macro "Step 1 resize raw.txt":
    1.  Open a SDB file (16-bit unsigned, configurable params)
    2.  Display and let user select a crop ROI
    3.  Crop (pixel replication) 
    4.  Save as "Light" (.hdr/.img) in the same folder as source
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas
from aidas.utils.bright_pixels import (
    BrightPixelDetection,
    detect_bright_pixels_in_bands,
    smooth_bright_pixels,
)
from aidas.utils.filesystem import find_sdb_directories, skipped_directories_warning
from aidas.utils.io_utils import read_analyze, read_raw_oct, scale_image, write_analyze
from aidas.ui.components import AppButton, AppSplitButton
from aidas.ui.bright_pixel_notice import BrightPixelNotice
from aidas.ui.theme import COLOR_PAIRS, CONTROLS
from aidas.utils.ui_layout import COLORS, LAYOUT
from aidas.utils.ui_utils import (
    HoverToolTip,
    NativeNumericSpinbox,
    SidebarStepFrame,
    directory_row,
    icon_button,
    load_ctk_image,
)

SDB_PREF_KEY = "sdb_dir"
SDB_DEFAULT_DIR = os.path.abspath(os.path.expanduser("~/Desktop"))
DEFAULT_RAW_WIDTH = 768
DEFAULT_RAW_HEIGHT = 1200
DEFAULT_RAW_OFFSET = 1050
DEFAULT_RAW_BIT_DEPTH = 16
CROP_SCALE_X = 3
CROP_SCALE_Y = 1
DEFAULT_ROI_Y = 585
DEFAULT_ROI_HEIGHT = 128
SAVED_OUTPUT_FILENAMES = frozenset(("light.hdr", "light.img"))


class Step1Frame(SidebarStepFrame):
    """GUI panel for Step 1: Resize Raw OCT images.

    This view owns all Step 1 controls and state:
    - import parameters for reading `.sdb` data,
    - file discovery/navigation,
    - ROI definition and processing,
    - output saving (Analyze HDR/IMG pairs),
    - image interaction (zoom/pan/inspection).
    """

    MOUSE_STATUS_INTERVAL_MS = 50

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
        self.default_raw_width = self._configured_sdb_default(
            "sdb_raw_width", DEFAULT_RAW_WIDTH, minimum=1
        )
        self.default_raw_height = self._configured_sdb_default(
            "sdb_raw_height", DEFAULT_RAW_HEIGHT, minimum=1
        )
        self.default_raw_offset = self._configured_sdb_default(
            "sdb_raw_offset", DEFAULT_RAW_OFFSET, minimum=0
        )
        self.default_little_endian = bool(
            True if self.preferences is None else self.preferences.get("sdb_little_endian", True)
        )
        self._on_processed_image = on_processed_image
        self._on_batch_segment_folders = on_batch_segment_folders

        # ----- state -----
        self.raw_image = None          # original loaded image (H, W)  uint16
        self._source_raw_image = None  # original imported image before width adjustments
        self._working_source_raw_image = None  # opt-in corrections; original stays intact
        self._source_bright_pixel_mask = None
        self._raw_bright_pixel_mask = None
        self._source_bright_pixel_threshold = None
        self._bright_pixel_source_checked = False
        self._source_bright_pixels_smoothed = False
        self._bright_pixel_notice = None
        self._bright_pixel_notice_pending = False
        self._bright_pixel_decision = None
        self.processed_image = None    # after crop + scale           int16 (.img-like preview)
        self._processed_image_saved = False
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
        self._saved_crop_previews = {}
        self._crop_notification_popup = None
        self._crop_notification_job = None
        self._saved_output_directories = set()
        self._sdb_directory_selection_locked = False
        self._pending_mouse_status = None
        self._mouse_status_after_id = None

        # ----- layout -----
        self.build_standard_layout()

        # Left — scrollable control panel (content-driven width)
        right = self.content

        # Right — processing toolbar + image canvas + status
        self.canvas_toolbar = ttk.Frame(
            right,
            style="AIDaS.ContentHeader.TFrame",
            padding=(LAYOUT.space_md, LAYOUT.space_sm),
        )
        self.canvas_toolbar.pack(fill="x", pady=(0, LAYOUT.space_sm))

        self.status_var = tk.StringVar(
            value="No image loaded"
        )
        self.add_status_bar(self.status_var, parent=right)

        # self.canvas_roi_toolbar = ttk.Frame(
        #     right,
        #     style="AIDaS.ContentHeader.TFrame",
        #     padding=(LAYOUT.space_md, LAYOUT.space_sm),
        # )
        # self.canvas_roi_toolbar.pack(
        #     side="bottom",
        #     fill="x",
        #     pady=(LAYOUT.space_sm, 0),
        # )

        self.image_canvas = ImageCanvas(
            right,
            on_roi_change=self._on_roi_changed,
            on_mouse_move=self._on_mouse_moved,
            on_zoom_change=self._on_canvas_zoom_changed,
            auto_fit_on_resize=True,
        )
        self.image_canvas.pack(
            fill="both", expand=True, pady=(0, LAYOUT.space_sm)
        )

        # Build control widgets
        self._build_controls()

    def _apply_aidas_theme(self):
        """Refresh the native workspace and progress-row semantic colors."""

        super()._apply_aidas_theme()
        if hasattr(self, "sdb_listbox") and hasattr(self, "sdb_directory_listbox"):
            self._refresh_sdb_progress_colors()

    # ═══════════════════════════════════════════════════════════════════════
    #  Control-panel construction
    # ═══════════════════════════════════════════════════════════════════════
    def _build_controls(self):
        """Create and lay out the full left-side control panel."""
        numeric_vcmd = (self.register(self._validate_digits_only), "%P")

        # Build import controls before the queue so its callbacks can use them.
        # Their section is placed after the queue and collapsed below.
        self.sdb_params_section = self.add_sidebar_section("Advanced import settings", pady=(5, 5))
        self.sdb_params_frame = self.sdb_params_section.body
        self.sdb_params_frame.grid_columnconfigure(2, weight=1)

        self.width_var = tk.StringVar(value=str(self.default_raw_width))
        self.height_var = tk.StringVar(value=str(self.default_raw_height))
        self.offset_var = tk.StringVar(value=str(self.default_raw_offset))

        # Update these assignments in _build_controls
        self.width_stepper, self.width_reset_btn = self._param_stepper_row(
            self.sdb_params_frame, 0, "Width (px):", self.width_var, lambda: self.default_raw_width,
            step=1, minimum=1, maximum=10000, validatecommand=numeric_vcmd
        )
        
        self.height_stepper, self.height_reset_btn = self._param_stepper_row(
            self.sdb_params_frame, 1, "Height (px):", self.height_var, lambda: self.default_raw_height,
            step=1, minimum=1, maximum=10000, validatecommand=numeric_vcmd
        )
        
        self.offset_stepper, self.offset_reset_btn = self._param_stepper_row(
            self.sdb_params_frame, 2, "Offset (bytes):", self.offset_var, lambda: self.default_raw_offset,
            step=2, minimum=0, maximum=10_000_000, validatecommand=numeric_vcmd
        )

        self.width_var.trace_add("write", lambda *_: self._on_width_changed())
        self.height_var.trace_add("write", lambda *_: self._on_import_param_changed())
        self.offset_var.trace_add("write", lambda *_: self._on_import_param_changed())

        self.endian_var = tk.BooleanVar(value=self.default_little_endian)
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
        parent_dir_header = ttk.Frame(sdb)
        parent_dir_header.pack(fill="x", pady=(0, 2))
        ttk.Label(parent_dir_header, text="Parent dir:").pack(side="left")
        self.exclude_saved_outputs_var = tk.BooleanVar(value=False)
        self.exclude_saved_outputs_checkbox = ttk.Checkbutton(
            parent_dir_header,
            text="Exclude saved cropped folders",
            variable=self.exclude_saved_outputs_var,
            command=self._render_sdb_directories,
        )
        self.exclude_saved_outputs_checkbox.pack(side="left", padx=(4, 0))
        HoverToolTip(
            self.exclude_saved_outputs_checkbox,
            "Hide SDB folders containing light.hdr and light.img",
        )
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
            style="AIDaS.Link.TLabel",
            cursor="hand2",
        )
        self.sdb_scan_more_label.pack(anchor="w", pady=(0, 4))
        self.sdb_scan_tooltip = HoverToolTip(self.sdb_scan_more_label, "")

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
            style="AIDaS.Success.TLabel",
            font=("Segoe UI", 8),
        ).pack(side="left")
        ttk.Label(
            folder_counts,
            textvariable=self.uncropped_folder_count_var,
            style="AIDaS.Muted.TLabel",
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
            style="AIDaS.Success.TLabel",
            font=("Segoe UI", 8),
        ).pack(side="left")
        ttk.Label(
            image_counts,
            textvariable=self.uncropped_image_count_var,
            style="AIDaS.Muted.TLabel",
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

        # Keep folder selection and the work queue first. The summary belongs
        # outside the collapsible body so the import settings remain visible.
        self.sdb_params_section.pack(after=sdb_section)
        self.sdb_params_summary_var = tk.StringVar()
        self.sdb_params_summary_label = ttk.Label(
            self.sdb_params_section,
            textvariable=self.sdb_params_summary_var,
            style="AIDaS.Muted.TLabel",
            justify="left",
            wraplength=0,
        )
        self.sdb_params_summary_label.pack(
            fill="x",
            padx=LAYOUT.space_sm,
            pady=(0, LAYOUT.space_sm),
            after=self.sdb_params_section.header,
        )
        self._sdb_params_summary_tooltip = HoverToolTip(
            self.sdb_params_summary_label,
            "W/H: image width and height in pixels. Offset: skipped bytes.\n"
            "LE: little-endian; BE: big-endian.",
        )
        self._update_import_settings_summary()
        self.sdb_params_section.toggle()

        # Navigation is available in the canvas toolbar below.
        # nav_frame = ttk.Frame(sdb)
        # nav_frame.pack(fill="x")
        # ttk.Button(nav_frame, text="◀ Prev", command=self._prev_sdb).pack(
        #     side="left", expand=True, fill="x", padx=(0, 4)
        # )
        # ttk.Button(nav_frame, text="Next ▶", command=self._next_sdb).pack(
        #     side="right", expand=True, fill="x", padx=(4, 0)
        # )

        # Keep the workflow actions outside the scrolling queue so they remain
        # reachable at the minimum supported window height.  The sidebar itself
        # continues to consume the expandable space above this fixed footer.
        self.step_actions_footer = ttk.Frame(
            self.sidebar_shell,
            style="AIDaS.Sidebar.TFrame",
        )
        self.step_actions_footer.pack(
            side="bottom",
            fill="x",
            padx=(LAYOUT.space_sm, LAYOUT.space_xs),
            pady=(0, LAYOUT.space_sm),
            before=self.sidebar,
        )
        ttk.Separator(self.step_actions_footer, orient="horizontal").pack(
            fill="x",
            pady=(LAYOUT.space_xs, 6),
        )
        self.step_actions_frame = ttk.Frame(
            self.step_actions_footer,
            style="AIDaS.Sidebar.TFrame",
        )
        self.step_actions_frame.pack(fill="x", padx=2)
        action_icon_size = 16
        self.batch_segment_cropped_btn_icon = load_ctk_image(
            self, "flat-color-icons--right.png", size=20
        )
        self.batch_segment_cropped_btn = AppButton(
            self.step_actions_frame,
            text="Go to Step 2",
            variant="success",
            command=self._send_cropped_folders_to_step2,
            state="disabled",
            image=self.batch_segment_cropped_btn_icon,
            compound="left",
        )
        self.batch_segment_cropped_btn.pack(fill="x", pady=(4, 0))
        HoverToolTip(
            self.batch_segment_cropped_btn,
            "Continue to Step 2 after every image has been cropped and saved",
        )

        self.refresh_sdb_list()

        # ── ROI controls below the image canvas ──
        # Keep shared crop state for canvas selection and processed previews.
        # Manual ROI / View panel widgets and bindings are commented out.
        self.view_mode_var = tk.StringVar(value="source")
        self.roi_x_var = tk.StringVar(value="0")
        self.roi_y_var = tk.StringVar(value="0")
        self.roi_w_var = tk.StringVar(value="100")
        self.roi_h_var = tk.StringVar(value="100")
        self.target_w_var = tk.StringVar(value=str(100 * CROP_SCALE_X))
        self.target_h_var = tk.StringVar(value=str(100 * CROP_SCALE_Y))
        # self.roi_x_var.trace_add("write", self._on_roi_entry_changed)
        # self.roi_y_var.trace_add("write", self._on_roi_entry_changed)
        # self.roi_w_var.trace_add("write", self._on_roi_entry_changed)
        # self.roi_h_var.trace_add("write", self._on_roi_entry_changed)
        # self.target_w_var.trace_add("write", self._on_target_size_entry_changed)
        # self.target_h_var.trace_add("write", self._on_target_size_entry_changed)

        # roi_toolbar = self.canvas_roi_toolbar
        # roi_fields = (
        #     ("X (Left)", self.roi_x_var, 1, 0, 10000),
        #     ("Y (Top)", self.roi_y_var, 1, 0, 10000),
        #     ("Width", self.roi_w_var, 1, 1, 30000),
        #     ("Height", self.roi_h_var, 1, 1, 30000),
        # )
        # roi_steppers = []
        # for index, (label, var, step, minimum, maximum) in enumerate(roi_fields):
        #     field = ttk.Frame(roi_toolbar, style="AIDaS.ContentHeader.TFrame")
        #     field.pack(side="left", padx=(0, 12))
        #     ttk.Label(
        #         field,
        #         text=label,
        #         style="AIDaS.ContentHeader.TLabel",
        #     ).pack(anchor="w", pady=(0, 2))
        #     stepper = self._numeric_stepper(
        #         field,
        #         var,
        #         width=5,
        #         step=step,
        #         minimum=minimum,
        #         maximum=maximum,
        #         validatecommand=numeric_vcmd,
        #         bg_color=COLOR_PAIRS["surface_subtle"],
        #     )
        #     stepper.pack(fill="x")
        #     roi_steppers.append(stepper)

        # self.roi_entries = roi_steppers
        # self.target_size_entries = []

        # ttk.Separator(roi_toolbar, orient="vertical").pack(
        #     side="left", fill="y", padx=(0, 12), pady=2
        # )
        # view_group = ttk.Frame(
        #     roi_toolbar,
        #     style="AIDaS.ContentHeader.TFrame",
        # )
        # view_group.pack(side="left")
        # ttk.Label(
        #     view_group,
        #     text="View",
        #     style="AIDaS.ContentHeader.TLabel",
        # ).pack(anchor="w", pady=(0, 2))
        # view_choices = ttk.Frame(
        #     view_group,
        #     style="AIDaS.ContentHeader.TFrame",
        # )
        # view_choices.pack(anchor="w")
        # self.view_mode_var = tk.StringVar(value="source")
        # self.source_view_radio = ttk.Radiobutton(
        #     view_choices,
        #     style="AIDaS.ContentHeader.TRadiobutton",
        #     text="Source",
        #     value="source",
        #     variable=self.view_mode_var,
        #     command=self._on_view_mode_changed,
        # )
        # self.source_view_radio.pack(side="left", padx=(0, 8))
        # self.target_view_radio = ttk.Radiobutton(
        #     view_choices,
        #     style="AIDaS.ContentHeader.TRadiobutton",
        #     text="Target",
        #     value="target",
        #     variable=self.view_mode_var,
        #     command=self._on_view_mode_changed,
        #     state="disabled",
        # )
        # self.target_view_radio.pack(side="left")

        # Top canvas toolbar: ROI presets and processing actions.
        toolbar = self.canvas_toolbar
        toolbar.columnconfigure(0, weight=1, uniform="toolbar_sides")
        toolbar.columnconfigure(2, weight=1, uniform="toolbar_sides")
        self.canvas_actions = ttk.Frame(toolbar, style="AIDaS.ContentHeader.TFrame")
        self.canvas_actions.grid(row=0, column=0, sticky="w")
        self.crop_btn_icon = load_ctk_image(
            self, "material-symbols-light--crop.png", size=action_icon_size
        )
        self.crop_split_frame = AppSplitButton(
            self.canvas_actions,
            text="Crop & Scale",
            width=198,
            height=CONTROLS.height_lg,
            command=self._crop_and_scale,
            options_command=self._show_crop_options_menu,
            image=self.crop_btn_icon,
            bg_color=COLOR_PAIRS["surface_subtle"],
        )
        self.crop_split_frame.grid(row=0, column=0)
        self.crop_btn = self.crop_split_frame.action_button
        self.crop_options_btn = self.crop_split_frame.options_button
        self.crop_split_divider = self.crop_split_frame.divider
        self.crop_btn.configure(state="disabled")
        self.crop_options_btn.configure(state="disabled")
        self.crop_options_menu = tk.Menu(self.crop_options_btn, tearoff=False)
        self.crop_options_menu.add_command(
            label="Default Region",
            command=self._set_default_roi,
        )
        self.crop_options_menu.add_command(
            label="Entire Image",
            command=self._select_all_roi,
        )
        HoverToolTip(self.crop_options_btn, "Choose a crop region preset")

        self.undo_crop_btn_icon = load_ctk_image(
            self, "grommet-icons--revert.png", size=action_icon_size
        )
        self.undo_crop_btn = AppButton(
            self.canvas_actions,
            text="Undo",
            variant="primary",
            bg_color=COLOR_PAIRS["surface_subtle"],
            width=96,
            height=CONTROLS.height_lg,
            command=self._reset,
            state="disabled",
            image=self.undo_crop_btn_icon,
            compound="left",
        )
        self.undo_crop_btn.grid(row=0, column=0)
        self.undo_crop_btn.grid_remove()
        HoverToolTip(self.undo_crop_btn, "Remove the saved crop and return to the SDB working image")

        self._processing_controls_cropped = False

        self.canvas_navigation = ttk.Frame(toolbar, style="AIDaS.ContentHeader.TFrame")
        self.canvas_navigation.grid(row=0, column=1)
        self.back_sdb_btn_icon = load_ctk_image(
            self, "flat-color-icons--left.png", size=20
        )
        self.back_sdb_btn = AppButton(
            self.canvas_navigation,
            text="",
            image=self.back_sdb_btn_icon,
            width=CONTROLS.height_lg,
            height=CONTROLS.height_lg,
            command=self._prev_sdb,
            state="disabled",
            bg_color=COLOR_PAIRS["surface_subtle"],
        )
        self.back_sdb_btn.grid(row=0, column=0)
        HoverToolTip(self.back_sdb_btn, "Previous uncropped image")
        self.queue_progress_var = tk.StringVar(value="(0/0)")
        self.queue_progress_label = ttk.Label(
            self.canvas_navigation,
            textvariable=self.queue_progress_var,
            anchor="center",
            style="AIDaS.ContentHeader.TLabel",
        )
        self.queue_progress_label.grid(row=0, column=1, padx=LAYOUT.space_sm)
        HoverToolTip(self.queue_progress_label, "Cropped and saved images / total images")
        self.next_sdb_btn_icon = load_ctk_image(
            self, "flat-color-icons--right.png", size=20
        )
        self.next_sdb_btn = AppButton(
            self.canvas_navigation,
            text="",
            image=self.next_sdb_btn_icon,
            width=CONTROLS.height_lg,
            height=CONTROLS.height_lg,
            command=self._next_sdb,
            state="disabled",
            bg_color=COLOR_PAIRS["surface_subtle"],
        )
        self.next_sdb_btn.grid(row=0, column=2)
        HoverToolTip(self.next_sdb_btn, "Next uncropped image")
        self.bright_pixel_status_var = tk.StringVar(
            value="Bright pixels are checked when an SDB opens."
        )
        self._update_processing_controls()

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
    def _numeric_stepper(
        parent,
        var,
        *,
        width=8,
        step=1,
        minimum=0,
        maximum=10_000_000,
        validatecommand=None,
        bg_color=None,
    ):
        return NativeNumericSpinbox(
            parent,
            var,
            width=width,
            step=step,
            minimum=minimum,
            maximum=maximum,
            validatecommand=validatecommand,
            bg_color=bg_color,
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

        reset_btn = icon_button(
            row_frame,
            self,
            "material-symbols-light--refresh-rounded.png",
            command=lambda: self._reset_numeric_var(
                var,
                default_value() if callable(default_value) else default_value,
            ),
            tooltip=f"Reset {label.rstrip(':').lower()}",
        )
        reset_btn.pack(side="left", padx=(8, 0), anchor="center")

        return stepper, reset_btn


    def _save_processed_image(self):
        """Save the current crop and keep its preview available for Undo."""
        img = self.processed_image
        if img is None:
            messagebox.showwarning("Nothing to save", "Run 'Crop & Scale' first.")
            return False
        outdir = self._source_output_directory()
        if outdir is None:
            return False
        source_file = self.current_file
        base_name = self._build_output_name("light")
        base = os.path.join(outdir, base_name)
        stack = np.stack([img, img], axis=0)  # shape (2, H, W)
        try:
            write_analyze(base, stack)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Save error (Analyze)", str(exc))
            return False

        self._processed_image_saved = True
        if source_file:
            if not hasattr(self, "_saved_crop_previews"):
                self._saved_crop_previews = {}
            self._saved_crop_previews[self._path_key(source_file)] = np.array(img, copy=True)
            self._cropped_sdb_files.add(self._path_key(source_file))
        if self._directory_is_complete(outdir):
            self._saved_output_directories.add(self._path_key(outdir))

        self._refresh_sdb_progress_colors()
        self._update_batch_handoff_button_state()

        self._update_queue_controls()
        return True

    def _update_processing_controls(self):
        """Sync Crop/Undo controls with processed image availability."""
        has_processed = self.processed_image is not None
        has_raw_image = self.raw_image is not None
        desired_states = (
            (getattr(self, "undo_crop_btn", None), has_processed),
            (getattr(self, "crop_btn", None), has_raw_image and not has_processed),
            (getattr(self, "crop_options_btn", None), has_raw_image and not has_processed),
            (getattr(self, "target_view_radio", None), has_raw_image),
            (getattr(self, "source_view_radio", None), has_raw_image and not has_processed),
        )
        for control, enabled in desired_states:
            self._set_control_enabled(control, enabled)
        if hasattr(self, "_processing_controls_cropped") and self._processing_controls_cropped != has_processed:
            if has_processed:
                self.crop_split_frame.grid_remove()
                self.undo_crop_btn.grid()
            else:
                self.crop_split_frame.grid()
                self.undo_crop_btn.grid_remove()
            self._processing_controls_cropped = has_processed
        self._update_queue_controls()
        self._update_batch_handoff_button_state()

    def _dismiss_crop_notification(self):
        job = self.__dict__.get("_crop_notification_job")
        if job is not None:
            self.after_cancel(job)
            self._crop_notification_job = None
        popup = self.__dict__.get("_crop_notification_popup")
        if popup is not None:
            popup.destroy()
            self._crop_notification_popup = None

    def _show_crop_notification(self, message, *, style="AIDaS.Success.TLabel"):
        """Show a small, non-blocking confirmation above the image canvas."""
        if self.__dict__.get("tk") is None:
            return
        self._dismiss_crop_notification()
        popup = ttk.Frame(
            self.content,
            style="AIDaS.ContentHeader.TFrame",
            padding=(LAYOUT.space_lg, LAYOUT.space_sm),
            relief="solid",
            borderwidth=1,
        )
        ttk.Label(popup, text=message, style=style).pack()
        popup.place(
            relx=0.5,
            y=self.canvas_toolbar.winfo_height() + LAYOUT.space_sm,
            anchor="n",
        )
        popup.lift()
        self._crop_notification_popup = popup
        self._crop_notification_job = self.after(2000, self._dismiss_crop_notification)

    def destroy(self):
        self._dismiss_bright_pixel_notice()
        self._dismiss_crop_notification()
        super().destroy()

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

    @staticmethod
    def _control_state(control):
        """Read a CTk/ttk control state without causing a redraw."""
        if control is None:
            return None
        try:
            return str(control.cget("state"))
        except (AttributeError, KeyError, tk.TclError, TypeError, ValueError):
            pass

        # Lightweight test doubles and a few native ttk wrappers expose state
        # as an attribute or a state-token method instead of through ``cget``.
        state_api = getattr(control, "state", None)
        if isinstance(state_api, str):
            return state_api
        if callable(state_api):
            try:
                tokens = tuple(state_api())
            except (tk.TclError, TypeError):
                return None
            return "disabled" if "disabled" in tokens else "normal"
        return None

    @classmethod
    def _set_control_enabled(cls, control, enabled):
        """Configure a semantic control only when its state actually changes."""
        if control is None:
            return False
        desired = "normal" if enabled else "disabled"
        if cls._control_state(control) == desired:
            return False
        try:
            control.configure(state=desired)
        except (AttributeError, tk.TclError, TypeError, ValueError):
            return False
        return True

    def _set_sdb_parameters_enabled(self, enabled):
        """Toggle only interactive SDB parameter controls."""
        control_names = (
            "width_stepper",
            "width_reset_btn",
            "height_stepper",
            "height_reset_btn",
            "offset_stepper",
            "offset_reset_btn",
            "endian_checkbox",
        )
        for name in control_names:
            self._set_control_enabled(getattr(self, name, None), enabled)

    def _set_roi_controls_enabled(self, enabled):
        """Toggle ROI/target editors without reconfiguring unchanged controls."""
        controls = list(getattr(self, "roi_entries", ()))
        controls.extend(getattr(self, "target_size_entries", ()))
        for control in controls:
            self._set_control_enabled(control, enabled)

    def _confirm_discard_processed_image(self, next_path):
        """Ask before replacing an active cropped image with another source."""
        if self.processed_image is None or getattr(self, "_processed_image_saved", False):
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
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
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

    def _configured_sdb_default(self, key, fallback, *, minimum):
        value = fallback if self.preferences is None else self.preferences.get(key, fallback)
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return int(fallback)

    def set_sdb_parameter_defaults(self, *, width, height, offset, little_endian=True):
        """Update reset defaults after they are changed in application settings."""

        self.default_raw_width = max(1, int(width))
        self.default_raw_height = max(1, int(height))
        self.default_raw_offset = max(0, int(offset))
        self.default_little_endian = bool(little_endian)

    @staticmethod
    def _validate_digits_only(proposed_value):
        """Allow only digits for numeric entries (empty is allowed while editing)."""
        return proposed_value == "" or proposed_value.isdigit()

    # ═══════════════════════════════════════════════════════════════════════
    #  Actions
    # ═══════════════════════════════════════════════════════════════════════

    def _set_default_import_params(self):
        """Restore default SDB import parameters and apply them."""
        self.width_var.set(str(self.default_raw_width))
        self.height_var.set(str(self.default_raw_height))
        self.offset_var.set(str(self.default_raw_offset))
        self.endian_var.set(self.default_little_endian)
        return self._apply_import_params()

    def _on_import_param_changed(self):
        """Auto-apply import parameters when UI values change."""
        self._apply_import_params(show_errors=False)

    def _on_width_changed(self):
        """Store the requested width and resize the current image view without rereading the raw file."""
        self._apply_import_params(show_errors=False, skip_reload=True)
        self._apply_width_preview_adjustment()

    def _update_import_settings_summary(self):
        """Show validated import settings even while their controls are hidden."""
        summary = self.__dict__.get("sdb_params_summary_var")
        if summary is None:
            return
        params = self.raw_import_params or {
            "width": self.default_raw_width,
            "height": self.default_raw_height,
            "offset": self.default_raw_offset,
            "bit_depth": DEFAULT_RAW_BIT_DEPTH,
            "little_endian": self.default_little_endian,
        }
        byte_order = "LE" if params["little_endian"] else "BE"
        summary.set(
            f"W: {params['width']:,} · H: {params['height']:,} · "
            f"Offset: {params['offset']:,} B · {params['bit_depth']}-bit · {byte_order}"
        )

    def _apply_width_preview_adjustment(self):
        """Crop or pad the loaded image to match the requested width."""
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
        if self._source_raw_image is None or self.current_file is None:
            return

        try:
            requested_width = int(self.width_var.get())
        except ValueError:
            return

        if requested_width <= 0:
            return

        source = self.__dict__.get("_working_source_raw_image")
        if source is None:
            source = self._source_raw_image
        source_mask = self.__dict__.get("_source_bright_pixel_mask")
        source_width = int(source.shape[1])
        if requested_width == source_width:
            adjusted = np.array(source, copy=True)
            adjusted_mask = source_mask
        elif requested_width < source_width:
            crop = source_width - requested_width
            left = crop // 2
            right = left + requested_width
            adjusted = np.array(source[:, left:right], copy=True)
            adjusted_mask = None if source_mask is None else source_mask[:, left:right]
        else:
            pad = requested_width - source_width
            left = pad // 2
            right = pad - left
            adjusted = np.pad(source, ((0, 0), (left, right)), mode="constant", constant_values=0)
            adjusted_mask = None if source_mask is None else np.pad(
                source_mask, ((0, 0), (left, right)), mode="constant", constant_values=False
            )

        self.raw_image = adjusted
        self._raw_bright_pixel_mask = adjusted_mask
        self.processed_image = None
        self._processed_image_saved = False
        self._set_source_bright_pixel_status()
        self.view_mode_var.set("source")
        self.image_canvas.set_image(adjusted)
        self.image_canvas.enable_roi(True)
        self._set_default_roi()
        self._update_zoom_label()
        self._update_processing_controls()
        self._update_image_status()

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

    def _apply_import_params(self, show_errors=True, skip_reload=False):
        """Validate and store raw import parameters from the form.

        If a file is already open, this method immediately reloads that file
        with the updated values so the UI always reflects active parameters.

        Returns:
            bool: True when parameters are valid (and reload succeeds if needed).
        """
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
        try:
            w = int(self.width_var.get())
            h = int(self.height_var.get())
            off = int(self.offset_var.get())
            le = self.endian_var.get()
        except ValueError:
            if show_errors:
                messagebox.showerror("Error", "Invalid import parameter (must be integers).")
            return False

        if w <= 0 or h <= 0 or off < 0:
            if show_errors:
                messagebox.showerror("Error", "Width/Height must be > 0 and Offset must be >= 0.")
            return False

        self.raw_import_params = {
            "width": w,
            "height": h,
            "offset": off,
            "bit_depth": DEFAULT_RAW_BIT_DEPTH,
            "little_endian": le,
        }
        self._update_import_settings_summary()

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
            return self._load_image(img, self.current_file)

        if self.current_file and skip_reload:
            return True

        return True

    def _load_image(self, img, path, *, check_bright_pixels=True, source_check=None):
        """Load image data into UI state and refresh display widgets.

        Args:
            img: Loaded image array with shape (H, W).
            path: Source file path for display and output naming.
        """
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
        signed = np.asarray(img).astype(np.int16, copy=True)
        detection = None
        prepared = signed
        if source_check is not None:
            # Undo checks and obtains consent before deleting saved outputs.
            # Reuse that result when restoring its raw source afterwards.
            signed, detection, prepared = source_check
        elif check_bright_pixels:
            detection = detect_bright_pixels_in_bands(signed)
            prepared = self._offer_bright_pixel_removal(
                signed, detection=detection, context="SDB image", preview_image=img
            )
            if prepared is None:
                return False

        self._source_raw_image = np.array(img, copy=True)
        self._working_source_raw_image = prepared.astype(img.dtype, copy=True)
        self.raw_image = np.array(self._working_source_raw_image, copy=True)
        self._source_bright_pixel_mask = None if detection is None else detection.mask
        self._raw_bright_pixel_mask = self._source_bright_pixel_mask
        self._source_bright_pixel_threshold = None if detection is None else detection.threshold
        self._bright_pixel_source_checked = check_bright_pixels or source_check is not None
        self._source_bright_pixels_smoothed = prepared is not signed
        self.processed_image = None
        self._processed_image_saved = False
        self.current_file = path
        self._set_source_bright_pixel_status()
        self.view_mode_var.set("source")

        self.image_canvas.set_image(self.raw_image)
        self.image_canvas.enable_roi(True)
        self._set_default_roi()
        self._update_zoom_label()
        self._update_processing_controls()
        self._set_sdb_parameters_enabled(True)
        self._update_image_status()
        return True

    # ── Open Raw ──
    def _saved_crop_for_path(self, path):
        """Retrieve this session's crop, or the folder's saved Light image."""
        cached = getattr(self, "_saved_crop_previews", {}).get(self._path_key(path))
        if cached is not None:
            return np.array(cached, copy=True)
        directory = os.path.dirname(os.path.abspath(path))
        if not self._sdb_image_is_saved(directory, path):
            known_image = any(
                self._path_key(source) == self._path_key(path)
                for sources in getattr(self, "_sdb_directory_files", {}).values()
                for source in sources
            )
            # A shared Light pair may belong to another SDB in this folder.
            if known_image or not self._folder_has_all_saved_outputs(directory):
                return None
        image = read_analyze(os.path.join(directory, "light"))
        if image.ndim == 3:
            image = image[0]
        if image.ndim != 2 or not image.size:
            raise ValueError("The saved crop must contain a 2-D image.")
        return np.ascontiguousarray(image)

    def _load_saved_crop(self, image, path):
        """Display a completed crop; load its raw source only if Undo is used."""
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
        original = np.array(image, copy=True)
        prepared = self._offer_bright_pixel_removal(original, context="saved crop")
        if prepared is None:
            return False
        corrected = prepared is not original
        self.raw_image = None
        self._source_raw_image = None
        self._working_source_raw_image = None
        self._source_bright_pixel_mask = None
        self._raw_bright_pixel_mask = None
        self._source_bright_pixel_threshold = None
        self._bright_pixel_source_checked = False
        self._source_bright_pixels_smoothed = False
        self._source_roi = None
        self.processed_image = np.array(prepared, copy=True)
        self._processed_image_saved = not corrected
        self.current_file = path
        self.view_mode_var.set("target")
        self.image_canvas.enable_roi(False)
        self.image_canvas.set_image(self.processed_image)
        self._set_sdb_parameters_enabled(False)
        self._set_roi_controls_enabled(False)
        self._update_zoom_label()
        self._update_processing_controls()
        self._update_image_status()
        if corrected:
            return self._save_processed_image()
        return True

    def _open_raw(self, path=None, *, show_saved=True):
        """Open a `.sdb` raw file and display it.

        Args:
            path: Optional explicit file path. When omitted, shows file picker.
            show_saved: Prefer a saved crop when the image is already complete.
        """
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
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

        if show_saved:
            try:
                saved_crop = self._saved_crop_for_path(path)
            except (OSError, ValueError, RuntimeError) as exc:
                messagebox.showerror("Error reading cropped image", str(exc))
                return
            if saved_crop is not None:
                return self._load_saved_crop(saved_crop, path)

        # Re-apply before opening so current form values are always used.
        if not self._apply_import_params(skip_reload=True):
            return

        try:
            img = read_raw_oct(path, **self.raw_import_params)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Error reading file", str(exc))
            return

        return self._load_image(img, path)

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

    def refresh_sdb_list(self, preview_first=False):
        """Recursively scan the selected parent and rebuild the work queue."""
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
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
        self._update_queue_controls()

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

    def _sdb_queue_items(self):
        """Return every discovered image in the work queue's display order."""
        directory_files = getattr(self, "_sdb_directory_files", {})
        directories = list(getattr(self, "_sdb_directories", ()))
        known = {self._path_key(directory) for directory in directories}
        directories.extend(
            directory
            for directory in directory_files
            if self._path_key(directory) not in known
        )
        return [
            (directory, index, path)
            for directory in directories
            for index, path in enumerate(directory_files.get(directory, ()))
        ]

    def _sdb_image_is_saved(self, directory, path):
        return self._directory_has_saved_outputs(directory) or self._path_key(path) in getattr(
            self, "_cropped_sdb_files", set()
        )

    def _adjacent_uncropped_sdb(self, current_path, direction):
        """Find an unfinished image in one direction without wrapping."""
        if not current_path:
            return None
        queue = self._sdb_queue_items()
        current_key = self._path_key(current_path)
        for position, (_directory, _index, path) in enumerate(queue):
            if self._path_key(path) == current_key:
                stop = len(queue) if direction > 0 else -1
                for candidate in range(position + direction, stop, direction):
                    directory, index, path = queue[candidate]
                    if not self._sdb_image_is_saved(directory, path):
                        return directory, index, path
                return None
        return None

    def _next_sdb_queue_item(self, current_path):
        """Return the next uncropped image across discovered folders."""
        return self._adjacent_uncropped_sdb(current_path, 1)

    def _previous_sdb_queue_item(self, current_path):
        """Return the previous uncropped image across discovered folders."""
        return self._adjacent_uncropped_sdb(current_path, -1)

    def _sdb_queue_progress(self):
        queue = self._sdb_queue_items()
        saved = sum(self._sdb_image_is_saved(directory, path) for directory, _index, path in queue)
        return saved, len(queue)

    def _update_queue_controls(self):
        """Refresh the saved-image counter and available navigation actions."""
        progress_var = getattr(self, "queue_progress_var", None)
        if progress_var is not None:
            saved, total = self._sdb_queue_progress()
            value = f"({saved}/{total})"
            if progress_var.get() != value:
                progress_var.set(value)
        current_path = getattr(self, "current_file", None)
        self._set_control_enabled(
            getattr(self, "back_sdb_btn", None),
            self._previous_sdb_queue_item(current_path) is not None,
        )
        self._set_control_enabled(
            getattr(self, "next_sdb_btn", None),
            self._next_sdb_queue_item(current_path) is not None,
        )

    def _open_queued_sdb(self, directory, index, path):
        """Select and open a queue item without a discard prompt after saving."""
        self._active_sdb_directory = directory
        self._render_sdb_directories()
        self.sdb_listbox.selection_clear(0, "end")
        self.sdb_listbox.selection_set(index)
        self.sdb_listbox.see(index)
        self._open_raw(path=path)
        return bool(
            self.current_file
            and self._path_key(self.current_file) == self._path_key(path)
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
        enabled = callable(self._on_batch_segment_folders) and self._all_sdb_images_saved()
        self._set_control_enabled(button, enabled)

    def _all_sdb_images_saved(self):
        """Allow handoff only after the whole discovered batch has saved crops."""
        saved, total = self._sdb_queue_progress()
        has_unsaved_crop = getattr(self, "processed_image", None) is not None and not getattr(
            self, "_processed_image_saved", False
        )
        return (
            total > 0
            and saved == total
            and not has_unsaved_crop
            and all(self._directory_has_saved_outputs(directory) for directory in self._sdb_directory_files)
        )

    def _send_cropped_folders_to_step2(self):
        """Start Step 2 batch segmentation for all completed Step 1 folders."""
        if not self._all_sdb_images_saved():
            messagebox.showwarning(
                "Images still need saving",
                "Crop and save every image before continuing to Step 2.",
            )
            return
        folders = self._completed_output_folders()
        if not folders:
            messagebox.showwarning(
                "No cropped folders",
                "No SDB folder contains both light.hdr and light.img yet.",
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
            background=COLORS.success_soft,
            foreground=COLORS.success,
            selectbackground=COLORS.primary,
            selectforeground=COLORS.on_primary,
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
        self._update_image_status()

    def _open_selected_sdb(self):
        """Open the currently selected SDB file from the list."""
        sel = self.sdb_listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Select an SDB file from the list first.")
            return
        self._open_raw(path=self._sdb_files[sel[0]])

    def _prev_sdb(self):
        """Open the previous uncropped image in the work queue."""
        item = self._previous_sdb_queue_item(self.current_file)
        if item is not None:
            self._open_raw(path=item[2], show_saved=False)
            if self.current_file and self._path_key(self.current_file) == self._path_key(item[2]):
                self._active_sdb_directory = item[0]
                self._render_sdb_directories()

    def _next_sdb(self):
        """Open the original SDB for the next uncropped image."""
        item = self._next_sdb_queue_item(self.current_file)
        if item is not None:
            self._open_raw(path=item[2], show_saved=False)
            if self.current_file and self._path_key(self.current_file) == self._path_key(item[2]):
                self._active_sdb_directory = item[0]
                self._render_sdb_directories()

    # ── ROI ──
    def _set_default_roi(self):
        """Set ROI to a full-width horizontal band on the opened image.

        The legacy vertical offset and height are retained while the rectangle
        always spans from the image's left edge to its right edge.
        """
        if self.raw_image is None:
            return
        ih, iw = self.raw_image.shape
        x = 0
        y = min(DEFAULT_ROI_Y, max(0, ih - 1))
        w = iw
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
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
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

    def _build_processed_target_image(self, *, cropped=None):
        """Build the final cropped/scaled target from the shared source ROI."""
        if self.raw_image is None or self._source_roi is None:
            return None
        if cropped is None:
            x, y, w, h = self._source_roi
            cropped = self.raw_image[y:y + h, x:x + w].copy()
        target = scale_image(cropped, sx=CROP_SCALE_X, sy=CROP_SCALE_Y)
        if target.dtype != np.int16:
            target = target.astype(np.int16, copy=False)
        return np.ascontiguousarray(target)

    def _set_bright_pixel_status(self, message):
        """Keep the last histogram result visible beside the crop controls."""
        status = self.__dict__.get("bright_pixel_status_var")
        if status is not None:
            try:
                status.set(message)
            except tk.TclError:
                pass  # Closing the app also cancels a pending canvas notice.

    def _dismiss_bright_pixel_notice(self):
        """Resolve a pending action before removing its in-canvas notice."""
        decision = self.__dict__.get("_bright_pixel_decision")
        if decision is not None:
            try:
                if decision.get() == -1:
                    decision.set(2)
            except tk.TclError:
                pass
        notice = self.__dict__.get("_bright_pixel_notice")
        self._bright_pixel_notice = None
        if notice is not None:
            try:
                notice.destroy()
            except tk.TclError:
                pass

    def _ask_bright_pixel_action(self, detection, image, *, context="crop"):
        """Get consent from an embedded canvas notice without a window or grab.

        Only this Step 1 workspace pauses while the Tk event loop continues.
        The pending image is a temporary preview until the caller commits it.
        """
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return None
        if self.__dict__.get("tk") is None:
            return None

        self._dismiss_crop_notification()
        canvas = self.image_canvas
        old_image = canvas.get_image()
        old_zoom = canvas.get_zoom()
        old_roi = canvas.get_roi()
        old_scroll = (canvas.canvas.xview()[0], canvas.canvas.yview()[0])
        old_syncing = self.__dict__.get("_syncing_view_roi", False)
        old_focus = self.focus_get()
        choice = tk.IntVar(master=self, value=-1)
        self._bright_pixel_decision = choice
        self._bright_pixel_notice_pending = True
        self._syncing_view_roi = True
        guarded = []
        bindings = []
        guard_tag = f"AIDaSBrightPixels_{id(self)}"
        binding_owner = self._root()
        canvas_unmap_binding = None
        outcome = None

        def resolve(value):
            try:
                if choice.get() == -1:
                    choice.set(2 if value is None else int(bool(value)))
            except tk.TclError:
                pass

        try:
            # Prepend a temporary tag to existing workspace controls. Notice
            # buttons are created afterwards and remain active; other steps
            # and the application menu keep their ordinary bindings.
            for sequence in (
                "<ButtonPress>", "<ButtonRelease>", "<KeyPress>",
                "<KeyRelease>", "<MouseWheel>", "<<ListboxSelect>>",
            ):
                binding = self.bind_class(guard_tag, sequence, lambda _event: "break")
                bindings.append((sequence, binding))
            remaining = list(self.winfo_children())
            while remaining:
                widget = remaining.pop()
                remaining.extend(widget.winfo_children())
                tags = widget.bindtags()
                guarded.append((widget, tags))
                widget.bindtags((guard_tag, *tags))

            canvas.set_image(image)
            canvas_unmap_binding = canvas.bind(
                "<Unmap>", lambda _event: resolve(None), add="+"
            )
            notice = BrightPixelNotice(
                canvas.canvas, count=detection.count, context=context, on_decision=resolve
            )
            self._bright_pixel_notice = notice
            notice.place(x=0, y=0, anchor="nw", relwidth=1.0)
            notice.lift()
            notice.bind("<Unmap>", lambda _event: resolve(None), add="+")
            notice.focus_default()
            self.wait_variable(choice)
            outcome = {0: False, 1: True, 2: None}.get(choice.get())
        except tk.TclError:
            # A notice or its parent may be destroyed while awaiting a click.
            outcome = None
        finally:
            self._dismiss_bright_pixel_notice()
            self._bright_pixel_decision = None
            if canvas_unmap_binding:
                try:
                    canvas.unbind("<Unmap>", canvas_unmap_binding)
                except tk.TclError:
                    pass
            for widget, tags in guarded:
                try:
                    widget.bindtags(tags)
                except tk.TclError:
                    pass
            for sequence, binding in bindings:
                try:
                    binding_owner.unbind_class(guard_tag, sequence)
                    # bind_class registers callbacks on the Tk root. Removing
                    # them through this frame leaves stale root callbacks and
                    # can fail when the frame is destroyed during the choice.
                    if binding in (binding_owner.__dict__.get("_tclCommands") or ()):
                        binding_owner.deletecommand(binding)
                except tk.TclError:
                    pass
            try:
                if canvas.winfo_exists():
                    canvas._zoom = old_zoom
                    canvas.set_image(old_image)
                    if old_image is not None:
                        canvas.set_zoom(old_zoom)
                        canvas.set_roi(old_roi)
                        canvas.canvas.xview_moveto(old_scroll[0])
                        canvas.canvas.yview_moveto(old_scroll[1])
                        canvas._render_visible_image()
                    self._update_zoom_label()
                    self._update_image_status()
                if old_focus is not None and old_focus.winfo_exists():
                    old_focus.focus_set()
            except tk.TclError:
                pass
            self._syncing_view_roi = old_syncing
            self._bright_pixel_notice_pending = False
        return outcome

    def _set_source_bright_pixel_status(self, crop_mask=None):
        """Keep the source-level decision visible through ROI and width edits."""
        source_mask = self.__dict__.get("_source_bright_pixel_mask")
        if source_mask is None:
            self._set_bright_pixel_status("Bright pixels will be checked before cropping.")
            return
        count = int(np.count_nonzero(source_mask))
        if not count:
            self._set_bright_pixel_status("SDB histogram checked: no bright outliers found.")
        elif self.__dict__.get("_source_bright_pixels_smoothed", False):
            self._set_bright_pixel_status(
                f"SDB check: smoothed {count:,} bright pixels; other values unchanged."
            )
        else:
            location = "" if crop_mask is None else f" ({np.count_nonzero(crop_mask):,} in this crop)"
            self._set_bright_pixel_status(f"SDB check: kept {count:,} bright pixels{location}.")

    def _offer_bright_pixel_removal(
        self, cropped, *, detection=None, context="crop", preview_image=None
    ):
        """Offer masked smoothing of histogram outliers in the signed crop.

        Return None when the user cancels. Source buffers and the SDB file
        stay untouched; only the copy destined for the saved crop is corrected.
        """
        if detection is None:
            detection = detect_bright_pixels_in_bands(cropped)
        if not detection.count:
            self._set_bright_pixel_status(f"Histogram checked: no bright outliers found in this {context}.")
            return cropped

        self._set_bright_pixel_status(f"Histogram found {detection.count:,} bright pixel candidates.")
        decision = self._ask_bright_pixel_action(
            detection,
            cropped if preview_image is None else preview_image,
            context=context,
        )
        if decision is None:
            self._set_bright_pixel_status("Bright pixel check cancelled; current image kept.")
            return None
        if decision:
            corrected = smooth_bright_pixels(cropped, detection.mask)
            self._set_bright_pixel_status(
                f"Smoothed {detection.count:,} bright pixels; all other values unchanged."
            )
            return corrected
        self._set_bright_pixel_status(f"Kept {detection.count:,} bright pixels unchanged.")
        return cropped

    def _on_view_mode_changed(self):
        """Switch editable source/target work views or show the final target."""
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
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
        else:
            if self.raw_image is None:
                return
            self.image_canvas.set_image(self.raw_image)
            self.image_canvas.enable_roi(self.processed_image is None)
            self._set_canvas_roi_from_source()
            image = self.raw_image

        if self._source_roi is not None:
            self._update_roi_entries(*self._source_roi)
        self._update_zoom_label()
        self._update_image_status()

    def _update_image_status(self):
        """Show image properties only; workflow progress belongs in the UI itself."""
        img = self.image_canvas.get_image()
        if img is None:
            self.status_var.set("No image loaded")
            return
        height, width = img.shape[:2]
        zoom = self.image_canvas.get_zoom()
        self.status_var.set(
            f"Image: {width}×{height} {img.dtype}  |  Zoom: {zoom * 100:.0f}%"
        )

    def _on_canvas_zoom_changed(self, _zoom):
        """Refresh Step 1 status as soon as the canvas zoom changes."""
        self._update_zoom_label()
        self._update_image_status()

    def _on_mouse_moved(self, ix, iy, val):
        """Coalesce high-frequency cursor samples into periodic UI updates.

        Args:
            ix: X coordinate in image space.
            iy: Y coordinate in image space.
            val: Pixel value at `(ix, iy)`.
        """
        self._pending_mouse_status = (ix, iy, val)
        if self._mouse_status_after_id is not None:
            return
        try:
            self._mouse_status_after_id = self.after(
                self.MOUSE_STATUS_INTERVAL_MS,
                self._flush_mouse_status,
            )
        except tk.TclError:
            self._mouse_status_after_id = None

    def _flush_mouse_status(self):
        """Render only the most recent queued mouse-position sample."""

        self._mouse_status_after_id = None
        sample = self._pending_mouse_status
        self._pending_mouse_status = None
        if sample is None:
            return
        ix, iy, val = sample
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
            bool: True once the crop is saved; False on processing or save failure.
        """
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
        if self.raw_image is None:
            messagebox.showwarning("No image", "Open a raw file first.")
            return False

        roi = self._source_roi
        if roi is None:
            messagebox.showwarning("No ROI", "Select a crop region first.")
            return False

        x, y, w, h = roi
        # Analyze output uses signed 16-bit samples. Check that representation
        # before replication: unsigned raw values can wrap into negative tissue
        # intensities while the corrupt bright pixels remain strongly positive.
        cropped = self.raw_image[y:y + h, x:x + w].astype(np.int16, copy=True)
        known_mask = self.__dict__.get("_raw_bright_pixel_mask")
        crop_mask = None if known_mask is None else known_mask[y:y + h, x:x + w]
        if crop_mask is not None:
            detection = detect_bright_pixels_in_bands(cropped)
            new_mask = detection.mask & ~crop_mask
            new_mask.setflags(write=False)
            new_detection = BrightPixelDetection(new_mask, detection.threshold)
            cropped = self._offer_bright_pixel_removal(cropped, detection=new_detection)
            if cropped is not None and not new_detection.count:
                # A file-level Yes or No already covered these candidates.
                self._set_source_bright_pixel_status(crop_mask)
        else:
            cropped = self._offer_bright_pixel_removal(cropped)
        if cropped is None:
            return False

        self.processed_image = self._build_processed_target_image(cropped=cropped)
        if self.processed_image is None:
            messagebox.showwarning("No ROI", "Select a crop region first.")
            return False
        self._processed_image_saved = False
        self.view_mode_var.set("target")
        # Show the true 16-bit processed image. ImageCanvas creates its own
        # display-only 8-bit view without changing the stored annotation data.
        self.image_canvas.enable_roi(False)
        self.image_canvas.set_image(self.processed_image)
        self._update_zoom_label()
        self._set_sdb_parameters_enabled(False)

        self._update_image_status()
        self._update_processing_controls()
        self._set_roi_controls_enabled(False)
        if not self._save_processed_image():
            return False

        if callable(self._on_processed_image):
            try:
                self._on_processed_image(np.array(self.processed_image, copy=True), self.current_file)
            except Exception:
                # Step 1 must remain usable even if Step 2 sync fails.
                pass

        self._show_crop_notification("Saved successfully")
        return True

    def _invalidate_saved_crop(self, path):
        key = self._path_key(path)
        self._cropped_sdb_files.discard(key)
        self._saved_crop_previews.pop(key, None)
        self._saved_output_directories.discard(self._path_key(os.path.dirname(os.path.abspath(path))))
        self._processed_image_saved = False

    def _reset(self):
        """Delete the saved crop and restore the source image for recropping."""
        if self.__dict__.get("_bright_pixel_notice_pending", False):
            return False
        if self.processed_image is None or not self.current_file:
            return False
        original = self.raw_image
        source_check = None
        if original is None:
            if not self._apply_import_params(skip_reload=True):
                return False
            try:
                original = read_raw_oct(self.current_file, **self.raw_import_params)
            except (OSError, ValueError, RuntimeError) as exc:
                messagebox.showerror("Error reading source image", str(exc))
                return False
            signed = original.astype(np.int16, copy=True)
            detection = detect_bright_pixels_in_bands(signed)
            prepared = self._offer_bright_pixel_removal(
                signed, detection=detection, context="SDB image", preview_image=original
            )
            if prepared is None:
                return False
            source_check = (signed, detection, prepared)

        directory = self._source_output_directory()
        if directory is None:
            return False
        try:
            with os.scandir(directory) as entries:
                crop_files = [
                    entry.path for entry in entries
                    if entry.is_file() and entry.name.lower() in SAVED_OUTPUT_FILENAMES
                ]
            for crop_file in crop_files:
                os.remove(crop_file)
        except OSError as exc:
            if not self._folder_has_all_saved_outputs(directory):
                self._invalidate_saved_crop(self.current_file)
                self._render_sdb_directories()
                self._update_processing_controls()
            messagebox.showerror("Could not remove cropped image", str(exc))
            return False

        self._invalidate_saved_crop(self.current_file)
        if self.raw_image is None:
            self._load_image(original, self.current_file, source_check=source_check)
        else:
            self._restore_raw_view()
        self._active_sdb_directory = directory
        self._render_sdb_directories()
        self._update_batch_handoff_button_state()
        self._show_crop_notification("Saved crop removed", style="AIDaS.Danger.TLabel")
        return True

    def _restore_raw_view(self):
        """Restore the loaded raw image without changing its import source."""
        self.processed_image = None
        self._processed_image_saved = False
        self._set_source_bright_pixel_status()
        self.view_mode_var.set("source")
        self.image_canvas.set_image(self.raw_image)
        self.image_canvas.enable_roi(True)
        self._set_default_roi()
        self._update_zoom_label()
        self._update_processing_controls()
        self._set_sdb_parameters_enabled(True)
        self._set_roi_controls_enabled(True)
        self._update_image_status()

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
        self._update_image_status()

    def _zoom_out(self):
        """Decrease canvas zoom by a fixed divisor."""
        self.image_canvas.set_zoom(self.image_canvas.get_zoom() / 1.25)
        self._update_zoom_label()
        self._update_image_status()

    def _fit_zoom(self):
        """Fit the current image into the visible canvas viewport."""
        self.image_canvas.fit_to_window()
        self._update_zoom_label()
        self._update_image_status()

    def _update_zoom_label(self):
        """Refresh the visible zoom percentage label."""
        z = self.image_canvas.get_zoom()
        if hasattr(self, "zoom_lbl"):
            self.zoom_lbl.configure(text=f"{z * 100:.0f} %")
