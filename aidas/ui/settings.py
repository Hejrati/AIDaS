"""Central application, SDB, and R configuration window."""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

from aidas.core.config import STEP4_AUTO_DETECTION_DEFAULTS, validate_step4_auto_detection_preferences
from aidas.ui.settings_layout import SettingsLayoutMixin
from aidas.ui.theme import COLORS, COLOR_PAIRS, TYPOGRAPHY, normalize_appearance_mode, normalize_interface_mode
from aidas.ui.windowing import centered_logical_geometry, synchronize_window_chrome
from aidas.utils.r_script_library import OPTIMIZED_R_SCRIPTS_ENABLED, RScriptChoice
from aidas.utils.ui_utils import apply_app_icon_to


class SettingsDialog(SettingsLayoutMixin, ctk.CTkToplevel):
    """Central application, SDB, and R configuration window."""

    SCRIPT_ROLES = (
        ("main", "Main processing script"),
        ("output", "Output processing script"),
    )
    SCRIPT_OPTIONS = (
        "Original script", "Optimized version", "Optimized deeper layers", "Custom script",
    )

    @classmethod
    def _r_script_options(cls, role):
        if OPTIMIZED_R_SCRIPTS_ENABLED:
            return cls.SCRIPT_OPTIONS
        return ("Original script", "Custom script")

    AUTO_DETECTION_FIELDS = (
        (
            "step4_auto_start_min",
            "Start point minimum threshold",
            "Minimum allowed profile position for the detected start point.",
        ),
        (
            "step4_auto_start_max",
            "Start point maximum threshold",
            "Maximum allowed profile position for the detected start point.",
        ),
        (
            "step4_auto_end_min",
            "End point minimum threshold",
            "Minimum allowed profile position for the detected end point.",
        ),
        (
            "step4_auto_end_max",
            "End point maximum threshold",
            "Maximum allowed profile position for the detected end point.",
        ),
        (
            "step4_auto_savgol_window",
            "Smoothing window (samples)",
            "Number of neighboring samples used by the fixed degree-2 Savitzky-Golay filter. A larger odd number smooths more but can hide narrow features.",
        ),
        (
            "step4_auto_confidence_percent",
            "Acceptance confidence (%)",
            "Minimum score required to accept an ROI automatically. Higher percentages send more ROIs for manual review; the default is 66%.",
        ),
        (
            "step4_auto_min_quadratic_r2",
            "Minimum bell-fit R-squared",
            "Minimum quadratic fit quality for the profile to count as a bell shape. Values closer to 1 require a cleaner bell.",
        ),
        (
            "step4_auto_consistency_tolerance",
            "Consistency tolerance (samples)",
            "Allowed boundary deviation from the median across ROIs, measured in samples. Larger values permit more variation before manual review.",
        ),
    )

    def __init__(
        self,
        parent,
        *,
        preferences,
        interface_mode,
        set_interface_command,
        appearance_mode,
        set_appearance_command,
        step1,
        step3,
        step2=None,
    ):
        super().__init__(parent)
        self.withdraw()
        self.title("AIDaS Settings")
        # Allocate the final viewport before any controls request layout. The
        # centering helper settles idle work, so sizing only at the end first
        # paints a narrow, very tall provisional settings panel.
        self.geometry("920x680")
        self.configure(fg_color=COLOR_PAIRS["application"])
        self.resizable(True, True)
        self.minsize(740, 480)
        self.transient(parent)
        apply_app_icon_to(self)
        self._parent = parent
        self._preferences = preferences
        self._set_interface_command = set_interface_command
        self._set_appearance_command = set_appearance_command
        self._step1 = step1
        self._step2 = step2
        self._step3 = step3
        self._annotation_tutorial_applied_value = bool(
            preferences.get("show_step2_annotation_tutorial", True)
        )
        self.annotation_tutorial_var = tk.BooleanVar(
            master=self, value=self._annotation_tutorial_applied_value,
        )
        self._annotation_tutorial_dirty = False
        self._manual_segmentation_applied_value = bool(
            preferences.get("step2_manual_segmentation_enabled", False)
        )
        self.manual_segmentation_var = tk.BooleanVar(
            master=self, value=self._manual_segmentation_applied_value,
        )
        self.bright_pixel_confirmation_var = tk.BooleanVar(
            master=self,
            value=not (
                preferences.get("step1_bright_pixel_choice", "") in ("fix", "ignore")
                or bool(preferences.get("step1_auto_fix_bright_pixels", False))
            ),
        )
        self._presentation_mode = normalize_interface_mode(interface_mode)
        self._classic_settings = self._presentation_mode == "Classic"
        # CTk option menus flush pending paints while being constructed. Let
        # this window settle its full layout before rendering those surfaces.
        self._aidas_defer_canvas_updates = True
        self._presentation_refresh_after_id = None
        self._presentation_refresh_pending = False
        self._applying_changes = False
        self._modal_activation_after_id = None
        self._script_choices = {"main": [], "output": []}
        self._script_by_label = {"main": {}, "output": {}}
        self._script_custom_choices = {"main": None, "output": None}
        self._script_popups = {}
        self._script_vars = {
            "main": tk.StringVar(master=self),
            "output": tk.StringVar(master=self),
        }
        self._script_status_vars = {
            "main": tk.StringVar(master=self),
            "output": tk.StringVar(master=self),
        }
        self.auto_detection_vars = {
            key: tk.StringVar(
                master=self,
                value=self._settings_number_text(preferences.get(key, default)),
            )
            for key, default in STEP4_AUTO_DETECTION_DEFAULTS.items()
        }

        self._initial_appearance_mode = normalize_appearance_mode(appearance_mode)
        self._scripts_initialized = False
        self._pending_script_roles = set()
        self.sdb_default_vars = {
            key: tk.StringVar(master=self, value=str(preferences.get(key, default)))
            for key, default in (("sdb_raw_width", 768), ("sdb_raw_height", 1200), ("sdb_raw_offset", 1050))
        }
        self.sdb_little_endian_var = tk.BooleanVar(
            master=self, value=bool(preferences.get("sdb_little_endian", True)),
        )
        self.apply_status_var = tk.StringVar(master=self, value="")
        for variable in (*self.sdb_default_vars.values(), *self.auto_detection_vars.values()):
            variable.trace_add("write", self._mark_dirty)
        style = ttk.Style(self)
        style.configure("AIDaS.Settings.TLabel", background=COLORS.surface, foreground=COLORS.text)
        style.configure("AIDaS.Settings.TCheckbutton", background=COLORS.surface)
        self._build_settings_shell()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self.geometry(centered_logical_geometry(self, 920, 680, parent=parent))
        from aidas.ui.rendering import finish_deferred_rendering
        finish_deferred_rendering(self)
        self._settings_ready = True
        self.deiconify()
        synchronize_window_chrome(
            self, background=COLOR_PAIRS["window_chrome"],
            foreground=COLOR_PAIRS["text"], border=COLOR_PAIRS["window_chrome"],
        )
        self._schedule_modal_activation()

    @staticmethod
    def _settings_number_text(value):
        """Keep whole-number settings readable while preserving decimals."""

        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _mark_dirty(self, *_args) -> None:
        self.apply_status_var.set("Unsaved changes")

    def _build_bright_pixel_setting(self, parent, *, modern, row=0):
        options = dict(
            text="Confirm bright-pixel correction (Step 1)",
            variable=self.bright_pixel_confirmation_var,
            command=self._mark_dirty,
        )
        if modern:
            self.bright_pixel_confirmation_switch = ctk.CTkSwitch(
                parent, **options, progress_color=COLOR_PAIRS["primary"],
                bg_color=COLOR_PAIRS["surface"],
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size),
            )
            help_label = ctk.CTkLabel(
                parent, text="When disabled, the saved Fix or Ignore choice is applied automatically.",
                bg_color=COLOR_PAIRS["surface"],
                text_color=COLOR_PAIRS["muted_text"], anchor="w", wraplength=580,
            )
        else:
            self.bright_pixel_confirmation_switch = ttk.Checkbutton(parent, **options)
            help_label = ttk.Label(
                parent, text="When disabled, the saved Fix or Ignore choice is applied automatically.",
                style="AIDaS.Muted.TLabel", wraplength=580,
            )
        self.bright_pixel_confirmation_switch.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=0, pady=(4, 2),
        )
        help_label.grid(
            row=row + 1, column=0, columnspan=2, sticky="w", padx=0, pady=(0, 12),
        )

    def _build_manual_segmentation_setting(self, parent, *, modern: bool, row=0) -> None:
        options = dict(
            text="Enable manual segmentation (Step 2)",
            variable=self.manual_segmentation_var,
            command=self._mark_dirty,
        )
        help_text = (
            "Enables ImageJ-style boundary editing. "
            "Fovea placement remains available."
        )
        if modern:
            self.manual_segmentation_switch = ctk.CTkSwitch(
                parent, **options, progress_color=COLOR_PAIRS["primary"],
                bg_color=COLOR_PAIRS["surface"],
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size),
            )
            help_label = ctk.CTkLabel(
                parent, text=help_text, text_color=COLOR_PAIRS["muted_text"],
                bg_color=COLOR_PAIRS["surface"],
                anchor="w", wraplength=580,
            )
        else:
            self.manual_segmentation_switch = ttk.Checkbutton(parent, **options)
            help_label = ttk.Label(
                parent, text=help_text, style="AIDaS.Muted.TLabel", wraplength=580,
            )
        self.manual_segmentation_switch.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=0, pady=(4, 2),
        )
        help_label.grid(
            row=row + 1, column=0, columnspan=2, sticky="w", padx=0, pady=(0, 12),
        )

    def _apply_manual_segmentation_setting(self) -> None:
        variable = self.__dict__.get("manual_segmentation_var")
        if variable is None:
            return
        enabled = bool(variable.get())
        changed = enabled != self._manual_segmentation_applied_value
        self._preferences.set("step2_manual_segmentation_enabled", enabled)
        if self._step2 is not None and changed:
            self._step2.set_manual_segmentation_enabled(enabled)
        self._manual_segmentation_applied_value = enabled

    def _annotation_tutorial_choice_changed(self) -> None:
        self._annotation_tutorial_dirty = True
        self._mark_dirty()

    def _apply_annotation_tutorial_setting(self) -> None:
        variable = self.__dict__.get("annotation_tutorial_var")
        if variable is None:
            return
        enabled = bool(variable.get())
        changed = enabled != self._annotation_tutorial_applied_value
        self._preferences.set("show_step2_annotation_tutorial", enabled)
        if self._step2 is not None and (changed or self._annotation_tutorial_dirty):
            self._step2.set_annotation_tutorial_enabled(enabled)
        self._annotation_tutorial_applied_value = enabled
        self._annotation_tutorial_dirty = False

    def _interface_choice_changed(self, mode_name: str) -> None:
        """Keep dependent appearance controls honest before Apply is clicked."""

        self._sync_interface_controls(mode_name)
        self._mark_dirty()

    def _sync_interface_controls(
        self,
        interface_mode: object,
        appearance_mode: object | None = None,
    ) -> None:
        """Synchronize General controls after an external or local UI switch."""

        selected = normalize_interface_mode(interface_mode)
        self.interface_menu.set(selected)
        if appearance_mode is not None:
            self.appearance_menu.set(normalize_appearance_mode(appearance_mode))
        self.appearance_menu.configure(
            state=(
                "readonly"
                if getattr(self, "_classic_settings", False) and selected == "Modern"
                else "normal"
                if selected == "Modern"
                else "disabled"
            )
        )

    def _schedule_presentation_refresh(self) -> None:
        """Close Settings after its presentation shell has been replaced."""

        if getattr(self, "_applying_changes", False):
            self._presentation_refresh_pending = True
            return
        if self._presentation_refresh_after_id is not None:
            return

        def refresh() -> None:
            self._presentation_refresh_after_id = None
            parent = self._parent
            self._close()
            try:
                parent._settings_dialog = None
            except (AttributeError, TypeError):
                pass
            try:
                parent.focus_force()
            except (AttributeError, tk.TclError):
                pass

        try:
            self._presentation_refresh_after_id = self.after_idle(refresh)
        except tk.TclError:
            self._presentation_refresh_after_id = None

    def _schedule_modal_activation(self, *, delay_ms: int = 50) -> None:
        """Acquire the modal grab only after Windows has mapped this dialog."""

        pending = getattr(self, "_modal_activation_after_id", None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except tk.TclError:
                pass
            self._modal_activation_after_id = None
        try:
            self.deiconify()
            self._modal_activation_after_id = self.after(
                max(1, int(delay_ms)),
                self._activate_modal_when_visible,
            )
        except (tk.TclError, TypeError, ValueError):
            self._modal_activation_after_id = None

    def _activate_modal_when_visible(self, attempt: int = 0) -> None:
        """Finish a modal handoff without leaving an invisible input grab."""

        self._modal_activation_after_id = None
        try:
            if not self.winfo_exists():
                return
            if not self.winfo_viewable():
                self.deiconify()
                if attempt < 20:
                    self._modal_activation_after_id = self.after(
                        25,
                        lambda: self._activate_modal_when_visible(attempt + 1),
                    )
                return
            self.lift()
            self.grab_set()
            self.focus_force()
            navigation = self.__dict__.get("category_navigation")
            if navigation is not None:
                navigation.focus_set()
        except tk.TclError:
            self._modal_activation_after_id = None

    def _validated_sdb_defaults(self):
        try:
            width = int(self.sdb_default_vars["sdb_raw_width"].get())
            height = int(self.sdb_default_vars["sdb_raw_height"].get())
            offset = int(self.sdb_default_vars["sdb_raw_offset"].get())
            if width < 1 or height < 1 or offset < 0:
                raise ValueError
        except ValueError:
            if "_category_pages" in self.__dict__:
                self._select_category("step1")
            messagebox.showerror(
                "SDB Defaults",
                "Width and height must be positive integers; offset must be zero or greater.",
                parent=self,
            )
            return None
        return width, height, offset, bool(self.sdb_little_endian_var.get())

    def _validated_auto_detection_settings(self):
        variables = getattr(self, "auto_detection_vars", None)
        if not variables:
            return {}
        try:
            return validate_step4_auto_detection_preferences(
                {key: variable.get() for key, variable in variables.items()}
            )
        except ValueError as exc:
            if "_category_pages" in self.__dict__:
                self._select_category("step4")
            messagebox.showerror(
                "Step 4 Auto-detect Settings",
                str(exc),
                parent=self,
            )
            return None

    @staticmethod
    def _compact_script_name(name, limit=48):
        name = str(name)
        if len(name) <= limit:
            return name
        tail = max(14, limit // 2 - 2)
        head = max(12, limit - tail - 1)
        return f"{name[:head]}…{name[-tail:]}"

    @classmethod
    def _script_display_label(cls, choice):
        source = cls._script_version_label(choice)
        if source is not None:
            if choice.is_default:
                source += " (default)"
        else:
            source = "Default" if choice.is_default else str(choice.source).title()
        return f"{source} — {cls._compact_script_name(choice.path.name)}"

    @staticmethod
    def _script_version_label(choice):
        return {
            "original": "Original script",
            "optimized": "Optimized version",
            "optimized_deep_layers": "Optimized deeper layers",
        }.get(getattr(choice, "version", None))

    @classmethod
    def _script_status_label(cls, choice):
        name = cls._compact_script_name(choice.path.name)
        version = cls._script_version_label(choice)
        return f"{version} — {name}" if version is not None else name

    def _refresh_r_script_choices(self, role, *, select_path=None) -> None:
        choices = self._step3.available_r_scripts(role)
        self._script_choices[role] = choices
        by_label = {}
        for version, label in (
            ("original", "Original script"),
            ("optimized", "Optimized version"),
            ("optimized_deep_layers", "Optimized deeper layers"),
        ):
            if label not in self._r_script_options(role):
                continue
            choice = next((choice for choice in choices if choice.version == version), None)
            if choice is not None:
                by_label[label] = choice
        selected_path = Path(select_path).resolve() if select_path is not None else self._step3._selected_r_script_path(role)
        selected = next(
            (choice for choice in choices if selected_path is not None and choice.path.resolve() == selected_path.resolve()),
            None,
        )
        if selected is None and selected_path is not None and selected_path.is_file():
            selected = RScriptChoice("Custom script", selected_path.resolve(), "user")
        if selected is None:
            selected = choices[0] if choices else None

        custom_choices = getattr(self, "_script_custom_choices", {})
        self._script_custom_choices = custom_choices
        label = next((label for label, choice in by_label.items() if choice == selected), "Custom script")
        if selected is not None and label == "Custom script":
            custom_choices[role] = selected
        custom = custom_choices.get(role)
        if custom is not None and custom.path.is_file():
            by_label["Custom script"] = custom
        self._script_by_label[role] = by_label
        self._script_vars[role].set(label)
        self._update_r_script_popup(role)
        if selected is not None:
            state = "Pending" if select_path is not None else "Active"
            self._script_status_vars[role].set(
                f"{state}: {self._script_status_label(selected)}"
            )
            if select_path is not None:
                if "_pending_script_roles" in self.__dict__:
                    self._pending_script_roles.add(role)
                self._mark_dirty()
        else:
            self._script_status_vars[role].set("No runnable script is available")

    def _r_script_check_images(self, popup):
        """Keep a smooth, DPI-sized selection icon in the menu's interpreter."""
        size = max(8, round(16 * ctk.ScalingTracker.get_widget_scaling(popup)))
        color = tuple(value // 257 for value in popup.winfo_rgb(popup.cget("fg")))
        signature = (size, color)
        cached = getattr(popup, "_aidas_script_check_icons", None)
        if cached is not None and cached[0] == signature:
            return cached[1]

        supersampling = 8
        image = Image.new("RGBA", (size * supersampling,) * 2)
        unit = size * supersampling / 16
        points = [(x * unit, y * unit) for x, y in ((3, 8), (6.5, 11.5), (13, 4.5))]
        stroke = max(1, round(1.7 * unit))
        draw = ImageDraw.Draw(image)
        draw.line(points, fill=color, width=stroke, joint="curve")
        radius = stroke / 2
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        rendered = image.resize((size, size), Image.Resampling.LANCZOS)
        icons = (
            ImageTk.PhotoImage(rendered, master=popup),
            ImageTk.PhotoImage(Image.new("RGBA", (size, size)), master=popup),
        )
        popup._aidas_script_check_icons = (signature, icons)
        if not getattr(popup, "_aidas_script_check_cleanup", False):
            popup.bind("<Destroy>", lambda _event: popup.__dict__.pop(
                "_aidas_script_check_icons", None,
            ), add="+")
            popup._aidas_script_check_cleanup = True
        return icons

    def _update_r_script_popup(self, role) -> None:
        popup = getattr(self, "_script_popups", {}).get(role)
        if popup is None:
            return
        popup.delete(0, "end")
        check_icon, spacer = self._r_script_check_images(popup)
        selected = self._script_vars[role].get()
        for label in self._r_script_options(role):
            if label == "Custom script":
                popup.add_separator()
                display = "Select custom script…"
                enabled = True
            else:
                display = label
                enabled = label in self._script_by_label[role]
            checked = label == selected and label in self._script_by_label[role]
            popup.add_command(
                label=display,
                image=check_icon if checked else spacer,
                compound="left",
                state="normal" if enabled else "disabled",
                command=lambda selected_label=label, selected_role=role: self._choose_r_script_option(
                    selected_role, selected_label
                ),
            )

    def _choose_r_script_option(self, role, label) -> None:
        if self._step3._busy:
            messagebox.showwarning(
                "Step 3 is running",
                "R script settings cannot be changed until the current Step 3 batch finishes.",
                parent=self,
            )
            return
        if label == "Custom script":
            self._add_r_script(role)
            return
        if label not in self._script_by_label[role]:
            return
        self._script_vars[role].set(label)
        self._select_r_script(role)

    def _select_r_script(self, role) -> None:
        if self._step3._busy:
            messagebox.showwarning(
                "Step 3 is running",
                "R script settings cannot be changed until the current Step 3 batch finishes.",
                parent=self,
            )
            self._refresh_r_script_choices(role)
            return
        choice = self._script_by_label[role].get(self._script_vars[role].get())
        if choice is None:
            return
        if "_pending_script_roles" in self.__dict__:
            self._pending_script_roles.add(role)
        self._script_status_vars[role].set(
            f"Pending: {self._script_status_label(choice)} — click Apply"
        )
        self._update_r_script_popup(role)
        self._mark_dirty()

    def _add_r_script(self, role) -> None:
        if self._step3._busy:
            messagebox.showwarning(
                "Step 3 is running",
                "Wait for the current Step 3 batch to finish before selecting a custom R script.",
                parent=self,
            )
            return
        selected = filedialog.askopenfilename(
            parent=self,
            title="Select custom main processing R script" if role == "main" else "Select custom output R script",
            filetypes=(("R scripts", "*.R"), ("All files", "*.*")),
        )
        if not selected:
            return
        try:
            imported = self._step3.import_r_script_for_role(Path(selected), role)
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "Select Custom R Script", f"Could not select the custom R script.\n{exc}", parent=self
            )
            return
        self._refresh_r_script_choices(role, select_path=imported)

    def _accept_changes(self) -> None:
        if self._apply_changes():
            self._close()

    def _apply_changes(self) -> bool:
        sdb_defaults = self._validated_sdb_defaults()
        if sdb_defaults is None:
            return False
        auto_detection_settings = self._validated_auto_detection_settings()
        if auto_detection_settings is None:
            return False
        selected_scripts = {}
        for role, _label in self.SCRIPT_ROLES:
            pending_roles = self.__dict__.get("_pending_script_roles")
            if pending_roles is not None and role not in pending_roles:
                continue
            choice = self._script_by_label[role].get(self._script_vars[role].get())
            if choice is None:
                messagebox.showerror(
                    "R Script Settings",
                    f"Choose an available {role} R script before applying settings.",
                    parent=self,
                )
                return False
            selected_scripts[role] = choice
        if selected_scripts and self._step3._busy:
            script_changed = any(
                self._step3._selected_r_script_path(role) is None
                or self._step3._selected_r_script_path(role).resolve() != choice.path.resolve()
                for role, choice in selected_scripts.items()
            )
            if script_changed:
                messagebox.showwarning(
                    "Step 3 is running",
                    "R script changes cannot be applied until the current Step 3 batch finishes.",
                    parent=self,
                )
                return False

        # Switching or reselecting the interface synchronizes these controls
        # with the app's current values. Preserve the user's choices before
        # any callbacks can replace them with the previously saved appearance.
        selected_interface = normalize_interface_mode(self.interface_menu.get())
        selected_appearance = normalize_appearance_mode(self.appearance_menu.get())
        self._apply_annotation_tutorial_setting()
        self._apply_manual_segmentation_setting()
        width, height, offset, little_endian = sdb_defaults
        remembered_choice = self._preferences.get("step1_bright_pixel_choice", "")
        if self.bright_pixel_confirmation_var.get():
            remembered_choice = ""
        elif remembered_choice not in ("fix", "ignore"):
            remembered_choice = "fix"
        self._preferences.set("step1_bright_pixel_choice", remembered_choice)
        self._preferences.set("step1_auto_fix_bright_pixels", remembered_choice == "fix")
        self._preferences.set("sdb_raw_width", width)
        self._preferences.set("sdb_raw_height", height)
        self._preferences.set("sdb_raw_offset", offset)
        self._preferences.set("sdb_little_endian", little_endian)
        for key, value in auto_detection_settings.items():
            self._preferences.set(key, value)
        self._step1.set_sdb_parameter_defaults(
            width=width,
            height=height,
            offset=offset,
            little_endian=little_endian,
        )
        for role, choice in selected_scripts.items():
            self._step3.select_r_script(role, choice.path)
            self._script_status_vars[role].set(
                f"Active: {self._script_status_label(choice)}"
            )
            self._update_r_script_popup(role)
        if "_pending_script_roles" in self.__dict__:
            self._pending_script_roles.difference_update(selected_scripts)
        self._applying_changes = True
        try:
            applied_interface = self._set_interface_command(selected_interface)
            if applied_interface is None:
                applied_interface = getattr(
                    self._parent,
                    "interface_mode",
                    selected_interface,
                )
            applied_interface = normalize_interface_mode(applied_interface)
            self.interface_menu.set(applied_interface)
            if applied_interface == selected_interface == "Modern":
                self._set_appearance_command(selected_appearance)
                self.appearance_menu.set(selected_appearance)
            self._sync_interface_controls(
                applied_interface,
                selected_appearance,
            )
            if applied_interface == selected_interface:
                self.apply_status_var.set("All settings applied")
            else:
                self.apply_status_var.set(
                    f"{applied_interface} remains active; interface change was not applied"
                )
        finally:
            self._applying_changes = False
            if getattr(self, "_presentation_refresh_pending", False):
                self._presentation_refresh_pending = False
                self._schedule_presentation_refresh()
        return applied_interface == selected_interface

    def _close(self) -> None:
        render_id = self.__dict__.get("_category_render_after_id")
        if render_id is not None:
            self.after_cancel(render_id)
            self._category_render_after_id = None
        pending = getattr(self, "_presentation_refresh_after_id", None)
        if pending is not None:
            self.after_cancel(pending)
            self._presentation_refresh_after_id = None
        activation_id = getattr(self, "_modal_activation_after_id", None)
        self._modal_activation_after_id = None
        if activation_id is not None:
            try:
                self.after_cancel(activation_id)
            except tk.TclError:
                pass
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
