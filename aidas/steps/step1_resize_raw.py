"""Step 1 — Resize Raw OCT image.

Replicates the ImageJ macro "Step 1 resize raw.txt":
    1.  Open a raw OCT binary file (16-bit unsigned, configurable params)
    2.  Display and let user select a crop ROI
    3.  Crop (pixel replication)
    4.  Duplicate into a 2-slice stack
    5.  Save as Analyze 7.5 (.hdr/.img) — "Light"
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from aidas.image_canvas import ImageCanvas
from aidas.utils.io_utils import read_raw_oct, scale_image, write_analyze, save_tiff

SDB_PREF_KEY = "sdb_dir"
SDB_DEFAULT_DIR = os.path.expanduser("~/Desktop")
DEFAULT_RAW_WIDTH = 768
DEFAULT_RAW_HEIGHT = 1200
DEFAULT_RAW_OFFSET = 1050
DEFAULT_RAW_BIT_DEPTH = 16


class Step1Frame(ttk.Frame):
    """GUI panel for Step 1: Resize Raw OCT images.

    This view owns all Step 1 controls and state:
    - import parameters for reading `.sdb` data,
    - file discovery/navigation,
    - ROI definition and processing,
    - output saving (Analyze + TIFF),
    - image interaction (zoom/pan/inspection).
    """

    def __init__(self, parent, preferences=None):
        """Initialize the Step 1 panel and construct all widgets.

        Args:
            parent: Parent Tkinter container.
            preferences: Optional preferences object implementing `get` and `set`.
        """
        super().__init__(parent)

        self.preferences = preferences

        # ----- state -----
        self.raw_image = None          # original loaded image (H, W)  uint16
        self.processed_image = None    # after crop + scale           uint16
        self.current_file = None       # path of opened raw file
        self.raw_import_params = None  # validated import parameters

        # ----- layout -----
        # Fixed sidebar on the left + expandable image area on the right.
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        # Left — scrollable control panel (content-driven width)
        left_outer = ttk.Frame(main)
        left_outer.pack(side="left", fill="y")

        ctrl_canvas = tk.Canvas(left_outer, highlightthickness=0)
        ctrl_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=ctrl_canvas.yview)
        self.ctrl = ttk.Frame(ctrl_canvas)
        self.ctrl.bind("<Configure>",
                       lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.create_window((0, 0), window=self.ctrl, anchor="nw")
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
        pad = dict(fill="x", padx=25)


        # ── SDB Image Parameters ──
        imp = ttk.LabelFrame(self.ctrl, text="SDB Image Parameters", padding=1)
        imp.pack(**pad, pady=5)

        self.width_var = self._param_row(imp, 0, "Width (px):", str(DEFAULT_RAW_WIDTH))
        self.height_var = self._param_row(imp, 1, "Height (px):", str(DEFAULT_RAW_HEIGHT))
        self.offset_var = self._param_row(imp, 2, "Offset (bytes):", str(DEFAULT_RAW_OFFSET))

        ttk.Label(imp, text="Bit depth:").grid(row=3, column=0, sticky="w", pady=1)
        self.bitdepth_var = tk.StringVar(value=str(DEFAULT_RAW_BIT_DEPTH))
        cb = ttk.Combobox(imp, textvariable=self.bitdepth_var,
                          values=["8", "16"], state="readonly", width=8)
        cb.grid(row=3, column=1, sticky="e", pady=1)

        self.endian_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(imp, text="Little-endian", variable=self.endian_var
                        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=1)

        imp_btns = ttk.Frame(imp)
        imp_btns.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Button(imp_btns, text="Default",
                   command=self._set_default_import_params).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(imp_btns, text="Apply",
                   command=self._apply_import_params).pack(side="right", expand=True, fill="x", padx=(2, 0))

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        # ── SDB Files ──
        sdb = ttk.LabelFrame(self.ctrl, text="SDB Files", padding=3)
        sdb.pack(**pad, pady=2)

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

        btn_frame = ttk.Frame(sdb)
        btn_frame.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_frame, text="Open Selected",
                   command=self._open_selected_sdb).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(btn_frame, text="Refresh",
                   command=self.refresh_sdb_list).pack(side="right")

        nav_frame = ttk.Frame(sdb)
        nav_frame.pack(fill="x", pady=(2, 0))
        ttk.Button(nav_frame, text="◀ Prev",
                   command=self._prev_sdb).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(nav_frame, text="Next ▶",
                   command=self._next_sdb).pack(side="right", expand=True, fill="x", padx=(2, 0))

        self._sdb_files = []
        self.refresh_sdb_list()

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        # ── ROI Selection ──
        roi = ttk.LabelFrame(self.ctrl, text="ROI Selection (crop region)", padding=3)
        roi.pack(**pad, pady=2)

        self.roi_x_var = tk.StringVar(value="0")
        self.roi_y_var = tk.StringVar(value="0")
        self.roi_w_var = tk.StringVar(value="100")
        self.roi_h_var = tk.StringVar(value="100")

        for i, (lbl, var, color) in enumerate([
            ("X (Left):", self.roi_x_var, "#DA0404"),
            ("Y (Top):", self.roi_y_var, "#DA0404"),
            ("Width (W):", self.roi_w_var, None),
            ("Height (H):", self.roi_h_var, None),
        ]):
            r, c = divmod(i, 2)
            if color:
                tk.Label(roi, text=lbl, fg=color).grid(row=r, column=c * 2, sticky="w", pady=1)
            else:
                ttk.Label(roi, text=lbl).grid(row=r, column=c * 2, sticky="w", pady=1)
            ttk.Entry(roi, textvariable=var, width=7).grid(
                row=r, column=c * 2 + 1, sticky="e", padx=(0, 8), pady=1)

        roi_tools = ttk.Frame(roi)
        roi_tools.grid(row=3, column=0, columnspan=4, pady=4)
        ttk.Button(roi_tools, text="Apply", command=self._apply_roi_entries).pack(side="left", padx=2)
        ttk.Button(roi_tools, text="Default ROI", command=self._set_default_roi).pack(side="left", padx=2)
        ttk.Button(roi_tools, text="Select All", command=self._select_all_roi).pack(side="left", padx=2)

        roi_actions = ttk.Frame(roi)
        roi_actions.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        ttk.Button(roi_actions, text="▶  Crop",
               command=self._crop_and_scale).pack(fill="x", pady=(0, 2))
        ttk.Button(roi_actions, text="▶  Crop && Save TIFF…",
               command=self._crop_scale_and_save_tiff).pack(fill="x", pady=2)
        ttk.Button(roi_actions, text="↺  Reset to Original",
               command=self._reset).pack(fill="x", pady=(2, 0))

        ttk.Separator(self.ctrl).pack(**pad, pady=3)

        # ── Save ──
        sav = ttk.LabelFrame(self.ctrl, text="Save Output", padding=3)
        sav.pack(**pad, pady=2)

        ttk.Label(sav, text="Output folder:").pack(anchor="w")
        df = ttk.Frame(sav)
        df.pack(fill="x")
        self.outdir_var = tk.StringVar(value=os.path.expanduser("~/Desktop"))
        ttk.Entry(df, textvariable=self.outdir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(df, text="…", width=3,
                   command=self._browse_outdir).pack(side="right")

        ttk.Button(sav, text="Save as Light (HDR)",
                   command=lambda: self._save_analyze("Light")).pack(fill="x", pady=(6, 2))
        ttk.Button(sav, text="Save as TIFF…",
                   command=self._save_tiff).pack(fill="x", pady=2)

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

    # helper for param rows
    @staticmethod
    def _param_row(parent, row, label, default):
        """Create one label+entry row for raw import parameters.

        Args:
            parent: Container where the row is placed.
            row: Grid row index.
            label: Label text.
            default: Default value string for the entry.

        Returns:
            tk.StringVar bound to the created entry widget.
        """
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=1, sticky="e", pady=1)
        return var

    # ═══════════════════════════════════════════════════════════════════════
    #  Actions
    # ═══════════════════════════════════════════════════════════════════════

    def _set_default_import_params(self):
        """Restore default SDB import parameters and apply them."""
        self.width_var.set(str(DEFAULT_RAW_WIDTH))
        self.height_var.set(str(DEFAULT_RAW_HEIGHT))
        self.offset_var.set(str(DEFAULT_RAW_OFFSET))
        self.bitdepth_var.set(str(DEFAULT_RAW_BIT_DEPTH))
        self.endian_var.set(True)
        return self._apply_import_params()

    def _apply_import_params(self):
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
            bd = int(self.bitdepth_var.get())
            le = self.endian_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid import parameter (must be integers).")
            return False

        if w <= 0 or h <= 0 or off < 0:
            messagebox.showerror("Error", "Width/Height must be > 0 and Offset must be >= 0.")
            return False
        if bd not in (8, 16):
            messagebox.showerror("Error", "Bit depth must be 8 or 16.")
            return False

        self.raw_import_params = {
            "width": w,
            "height": h,
            "offset": off,
            "bit_depth": bd,
            "little_endian": le,
        }

        # If an image is already open, immediately re-read it with new params.
        if self.current_file:
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

        self.status_var.set(
            f"Import params applied: {w}x{h}, offset {off}, {bd}-bit, "
            f"{'little' if le else 'big'}-endian"
        )
        return True

    def _load_image(self, img, path):
        """Load image data into UI state and refresh display widgets.

        Args:
            img: Loaded image array with shape (H, W).
            path: Source file path for display and output naming.
        """
        self.raw_image = img
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
        self.status_var.set(
            f"Loaded {filename} — left-drag ROI, right-drag pan, then Crop & Scale")

    # ── Open Raw ──
    def _open_raw(self, path=None):
        """Open a `.sdb` raw file and display it.

        Args:
            path: Optional explicit file path. When omitted, shows file picker.
        """
        if self.raw_import_params is None and not self._apply_import_params():
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

        # Re-apply before opening so current form values are always used.
        if not self._apply_import_params():
            return

        try:
            img = read_raw_oct(path, **self.raw_import_params)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Error reading file", str(exc))
            return

        self._load_image(img, path)

    def _browse_sdb_dir(self):
        """Prompt for SDB folder and refresh the browser list."""
        d = filedialog.askdirectory(
            title="Select SDB directory",
            initialdir=self.sdb_dir_var.get() or None,
        )
        if d:
            self.set_sdb_directory(d)
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

        Default behavior: full image width and 150 px height, vertically centered.
        """
        if self.raw_image is None:
            return
        ih, iw = self.raw_image.shape
        x = 0
        w = iw
        h = min(155, ih)
        y = 560
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

    def _update_roi_entries(self, x, y, w, h):
        """Write ROI values into UI entry variables."""
        self.roi_x_var.set(str(x))
        self.roi_y_var.set(str(y))
        self.roi_w_var.set(str(w))
        self.roi_h_var.set(str(h))

    def _on_roi_changed(self, roi):
        """Handle ROI-change callback from the canvas interaction layer.

        Args:
            roi: Tuple `(x, y, w, h)` in image coordinates.
        """
        x, y, w, h = roi
        self._update_roi_entries(x, y, w, h)

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
        sx, sy = 3, 1

        # Crop
        cropped = self.raw_image[y:y + h, x:x + w].copy()

        # Scale (pixel replication)
        scaled = scale_image(cropped, sx=sx, sy=sy)

        # Convert to 16-bit (keep as uint16)
        if scaled.dtype != np.uint16:
            scaled = scaled.astype(np.uint16)

        self.processed_image = scaled

        # Show the result
        self.image_canvas.enable_roi(False)
        self.image_canvas.set_image(scaled)
        self._update_zoom_label()

        ih, iw = scaled.shape
        filename = os.path.basename(self.current_file) if self.current_file else "Processed"

        # Update the top info display
        self.image_info_var.set(
            f"✓ Processed: {filename}  |  Size: {iw} × {ih} px  |  "
            f"Type: {scaled.dtype}  |  Range: [{scaled.min()} – {scaled.max()}]  |  "
            f"Cropped {w}×{h} from ({x},{y}), scaled ×{sx}/×{sy}"
        )
        self.status_var.set(
            f"Processed: {w}×{h} → {iw}×{ih}.  "
            f"Save as Light/Dark or Reset to adjust."
        )
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
        default_name = f"{self._build_output_name('crop')}.tif"
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
        if self.current_file:
            stem = os.path.splitext(os.path.basename(self.current_file))[0]
        else:
            stem = "image"
        return f"{stem}_{suffix}"

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
