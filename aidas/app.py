"""Main Tkinter application and branded information windows for AIDaS."""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from PIL import Image

from aidas import __version__
from aidas.core.config import Config
from aidas.core.display import (
    centered_decorated_position,
    centered_position,
    enable_per_monitor_dpi_awareness,
    fractional_size_of_bounds,
    work_area_bounds,
)
from aidas.core.single_instance import SingleInstanceGuard
from aidas.services.update_service import launch_installer
from aidas.services.update_ui import UpdateController
from aidas.ui.components import AppButton, AppStatusBar, WorkflowHeader
from aidas.ui.menu_bar import ApplicationMenuBar
from aidas.ui.splash import SplashWindow
from aidas.ui.theme import (
    APPEARANCE_MODES,
    COLOR_PAIRS,
    SHAPES,
    TYPOGRAPHY,
    apply_appearance_mode,
    normalize_appearance_mode,
    refresh_native_widgets,
)
from aidas.ui.title_bar import create_custom_windows_title_bar, reassert_client_size
from aidas.ui.windowing import (
    centered_logical_geometry,
    logical_window_size,
    physical_window_size,
    synchronize_window_chrome,
)
from aidas.utils.ui_layout import LAYOUT
from aidas.utils.ui_utils import (
    apply_app_icon_to,
    resource_path,
)


APP_TITLE = "AIDaS"
APP_SUBTITLE = "OCT Image Processing"
LAB_ACRONYM = "MVPRL"
LAB_NAME = "Machine Vision and Pattern Recognition Lab"
LAB_URL = "https://mvprl.cs.wayne.edu"
LAB_URL_TEXT = "mvprl.cs.wayne.edu"
UNIVERSITY_NAME = "Wayne State University"
COPYRIGHT_NOTICE = (
    "Copyright (c) 2026 Machine Vision and Pattern Recognition Lab, "
    "Wayne State University. Licensed under GNU AGPL v3 or later."
)
LAB_DESCRIPTION = (
    "Established in 2002, Machine Vision and Pattern Recognition Lab aims at\n"
    "performing research in Deep Learning, Data Mining and Multimedia Content\n"
    "Analysis."
)

SPLASH_MINIMUM_MS = 700


def _center_geometry(window: tk.Misc, width: int, height: int, *, parent=None) -> str:
    """Center a CTk window whose width and height are logical UI units."""

    return centered_logical_geometry(window, width, height, parent=parent)


class AboutDialog(ctk.CTkToplevel):
    """Branded, fixed-layout About window using the shared design system."""

    PREFERRED_WIDTH = 540
    PREFERRED_HEIGHT = 570
    MAX_SCREEN_FRACTION = 0.9

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.withdraw()
        self.title("About AIDaS")
        self.configure(fg_color=COLOR_PAIRS["application"])
        self.resizable(False, False)
        self.transient(parent)
        apply_app_icon_to(self)

        bounds = work_area_bounds(self, parent=parent)
        available_width = max(1, bounds[2] - bounds[0])
        available_height = max(1, bounds[3] - bounds[1])
        preferred_physical_width, preferred_physical_height = physical_window_size(
            self,
            self.PREFERRED_WIDTH,
            self.PREFERRED_HEIGHT,
        )
        physical_width = min(
            preferred_physical_width,
            round(available_width * self.MAX_SCREEN_FRACTION),
        )
        physical_height = min(
            preferred_physical_height,
            round(available_height * self.MAX_SCREEN_FRACTION),
        )
        dialog_width, dialog_height = logical_window_size(
            self,
            physical_width,
            physical_height,
        )
        content_wrap = max(80, dialog_width - 96)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        content = ctk.CTkFrame(
            self,
            fg_color=COLOR_PAIRS["surface"],
            corner_radius=SHAPES.corner_radius_lg,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
        )
        content.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))
        content.grid_columnconfigure(0, weight=1)
        self._wrapped_about_labels: list[ctk.CTkLabel] = []

        logo_path = resource_path(os.path.join("assets", "aidas.png"))
        with Image.open(logo_path) as logo:
            logo_source = logo.convert("RGBA").copy()
        self.logo_image = ctk.CTkImage(
            light_image=logo_source,
            dark_image=logo_source,
            size=(96, 96),
        )
        ctk.CTkLabel(content, text="", image=self.logo_image).grid(row=0, column=0, pady=(18, 4))

        def add_label(*, text: str, font, text_color=None, **options) -> ctk.CTkLabel:
            label = ctk.CTkLabel(
                content,
                text=text,
                font=font,
                text_color=text_color or COLOR_PAIRS["text"],
                justify=options.pop("justify", "center"),
                wraplength=content_wrap,
                **options,
            )
            self._wrapped_about_labels.append(label)
            return label

        add_label(
            text=APP_TITLE,
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.title_size,
                weight=TYPOGRAPHY.bold_weight,
            ),
        ).grid(row=1, column=0)
        add_label(
            text=f"{APP_SUBTITLE}  ·  Version {__version__}",
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size),
        ).grid(row=2, column=0, pady=(2, 14))
        add_label(
            text=LAB_ACRONYM,
            text_color=COLOR_PAIRS["institution"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.subtitle_size,
                weight=TYPOGRAPHY.bold_weight,
            ),
        ).grid(row=3, column=0)
        add_label(
            text=LAB_NAME,
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        ).grid(row=4, column=0, pady=(3, 0))

        link = add_label(
            text=LAB_URL_TEXT,
            text_color=COLOR_PAIRS["link"],
            cursor="hand2",
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
                underline=True,
            ),
        )
        link.grid(row=5, column=0, pady=(10, 0))
        link.bind("<Button-1>", lambda _event: webbrowser.open_new_tab(LAB_URL))

        add_label(
            text=" ".join(LAB_DESCRIPTION.splitlines()),
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size),
        ).grid(row=6, column=0, padx=24, pady=(16, 0))
        add_label(
            text=UNIVERSITY_NAME,
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        ).grid(row=7, column=0, pady=(14, 0))
        add_label(
            text=COPYRIGHT_NOTICE,
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.caption_size),
        ).grid(row=8, column=0, padx=24, pady=(10, 0))
        add_label(
            text=f"Python {sys.version.split()[0]}  ·  CustomTkinter {ctk.__version__}",
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.mono_family, size=TYPOGRAPHY.caption_size),
        ).grid(row=9, column=0, pady=(10, 18))

        button_panel = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        button_panel.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))
        AppButton(
            button_panel,
            text="Close",
            width=112,
            variant="primary",
            command=self._close,
        ).pack(side="right")

        self.bind("<Configure>", self._resize_about_content, add="+")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self.bind("<Return>", lambda _event: self._close())
        self.geometry(_center_geometry(self, dialog_width, dialog_height, parent=parent))
        self.deiconify()
        synchronize_window_chrome(
            self,
            background=COLOR_PAIRS["window_chrome"],
            foreground=COLOR_PAIRS["text"],
            border=COLOR_PAIRS["window_chrome"],
        )
        self.grab_set()
        self.focus_force()

    def _resize_about_content(self, event) -> None:
        if event.widget is not self:
            return
        logical_width, _ = logical_window_size(self, event.width, 1)
        wraplength = max(80, logical_width - 104)
        for label in self._wrapped_about_labels:
            label.configure(wraplength=wraplength)

    def _close(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()


class SettingsDialog(ctk.CTkToplevel):
    """Central application, SDB, and R configuration window."""

    SCRIPT_ROLES = (
        ("main", "Main processing script"),
        ("output", "Output processing script"),
    )

    def __init__(
        self,
        parent,
        *,
        preferences,
        appearance_mode,
        set_appearance_command,
        step1,
        step3,
    ):
        super().__init__(parent)
        self.withdraw()
        self.title("AIDaS Settings")
        self.configure(fg_color=COLOR_PAIRS["application"])
        self.resizable(False, False)
        self.transient(parent)
        apply_app_icon_to(self)
        self._parent = parent
        self._preferences = preferences
        self._set_appearance_command = set_appearance_command
        self._step1 = step1
        self._step3 = step3
        self._r_setup_wizard = None
        self._script_choices = {"main": [], "output": []}
        self._script_by_label = {"main": {}, "output": {}}
        self._script_vars = {
            "main": tk.StringVar(master=self),
            "output": tk.StringVar(master=self),
        }
        self._script_status_vars = {
            "main": tk.StringVar(master=self),
            "output": tk.StringVar(master=self),
        }

        self.settings_panel = ctk.CTkScrollableFrame(
            self,
            fg_color=COLOR_PAIRS["surface"],
            corner_radius=SHAPES.corner_radius_lg,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
            scrollbar_button_color=COLOR_PAIRS["border_strong"],
            scrollbar_button_hover_color=COLOR_PAIRS["primary"],
        )
        self.settings_panel.pack(fill="both", expand=True, padx=16, pady=(16, 8))
        self.settings_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.settings_panel,
            text="Settings",
            anchor="w",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.heading_size,
                weight=TYPOGRAPHY.bold_weight,
            ),
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))

        general = self._settings_section("General", row=1)
        self.appearance_menu = ctk.CTkOptionMenu(
            general,
            values=list(APPEARANCE_MODES),
            command=self._mark_dirty,
            width=140,
            height=34,
            corner_radius=SHAPES.corner_radius_sm,
            fg_color=COLOR_PAIRS["button"],
            button_color=COLOR_PAIRS["primary"],
            button_hover_color=COLOR_PAIRS["primary_hover"],
            text_color=COLOR_PAIRS["text"],
            dropdown_fg_color=COLOR_PAIRS["surface_elevated"],
            dropdown_hover_color=COLOR_PAIRS["primary_soft"],
            dropdown_text_color=COLOR_PAIRS["text"],
        )
        self._labeled_control(general, 0, "Appearance", self.appearance_menu)
        self.appearance_menu.set(appearance_mode)
        self.update_checks_var = tk.BooleanVar(
            master=self,
            value=bool(preferences.get("check_for_updates", True)),
        )
        self.update_switch = ctk.CTkSwitch(
            general,
            text="Check automatically for application updates",
            variable=self.update_checks_var,
            command=self._mark_dirty,
            onvalue=True,
            offvalue=False,
            progress_color=COLOR_PAIRS["primary"],
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size),
        )
        self.update_switch.grid(row=1, column=0, columnspan=2, sticky="w", padx=14, pady=(8, 12))

        sdb = self._settings_section("Default SDB image parameters", row=2)
        self.sdb_default_vars = {
            "sdb_raw_width": tk.StringVar(value=str(preferences.get("sdb_raw_width", 768))),
            "sdb_raw_height": tk.StringVar(value=str(preferences.get("sdb_raw_height", 1200))),
            "sdb_raw_offset": tk.StringVar(value=str(preferences.get("sdb_raw_offset", 1050))),
        }
        for row, (key, label) in enumerate(
            (
                ("sdb_raw_width", "Width (px)"),
                ("sdb_raw_height", "Height (px)"),
                ("sdb_raw_offset", "Offset (bytes)"),
            )
        ):
            entry = ctk.CTkEntry(
                sdb,
                textvariable=self.sdb_default_vars[key],
                width=140,
                height=32,
                fg_color=COLOR_PAIRS["surface_elevated"],
                border_color=COLOR_PAIRS["border_strong"],
                text_color=COLOR_PAIRS["text"],
            )
            self._labeled_control(sdb, row, label, entry)
        self.sdb_little_endian_var = tk.BooleanVar(
            master=self,
            value=bool(preferences.get("sdb_little_endian", True)),
        )
        ctk.CTkSwitch(
            sdb,
            text="Little-endian byte order",
            variable=self.sdb_little_endian_var,
            command=self._mark_dirty,
            onvalue=True,
            offvalue=False,
            progress_color=COLOR_PAIRS["primary"],
            text_color=COLOR_PAIRS["text"],
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=14, pady=(8, 4))
        sdb_actions = ctk.CTkFrame(sdb, fg_color="transparent", corner_radius=0)
        sdb_actions.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(8, 12))
        ctk.CTkLabel(
            sdb_actions,
            text="Changes are saved with the Apply button below.",
            text_color=COLOR_PAIRS["muted_text"],
            anchor="w",
        ).pack(side="left")

        r_environment = self._settings_section("R environment", row=3)
        ctk.CTkLabel(
            r_environment,
            text=f"Install or verify R {step3.R_REQUIRED_VERSION} and all required packages.",
            anchor="w",
            text_color=COLOR_PAIRS["muted_text"],
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(6, 8))
        AppButton(
            r_environment,
            text="Install / set up R and packages",
            variant="primary",
            command=self._open_r_setup,
            width=230,
        ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 12))

        scripts = self._settings_section("Step 3 R scripts", row=4)
        ctk.CTkLabel(
            scripts,
            text="Choose each active script from its list, then click Apply.",
            anchor="w",
            text_color=COLOR_PAIRS["muted_text"],
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(4, 8))
        self.script_menus = {}
        for row, (role, label) in enumerate(self.SCRIPT_ROLES):
            role_frame = ctk.CTkFrame(scripts, fg_color="transparent", corner_radius=0)
            role_frame.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(6, 10))
            role_frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                role_frame,
                text=label,
                anchor="w",
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.body_size,
                    weight=TYPOGRAPHY.semibold_weight,
                ),
            ).grid(row=0, column=0, columnspan=2, sticky="ew")
            menu = ctk.CTkOptionMenu(
                role_frame,
                variable=self._script_vars[role],
                values=["No R scripts found"],
                command=lambda _label, selected_role=role: self._select_r_script(selected_role),
                height=34,
                fg_color=COLOR_PAIRS["button"],
                button_color=COLOR_PAIRS["primary"],
                button_hover_color=COLOR_PAIRS["primary_hover"],
                text_color=COLOR_PAIRS["text"],
                dropdown_fg_color=COLOR_PAIRS["surface_elevated"],
                dropdown_hover_color=COLOR_PAIRS["primary_soft"],
                dropdown_text_color=COLOR_PAIRS["text"],
            )
            menu.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            self.script_menus[role] = menu
            AppButton(
                role_frame,
                text="Add R script...",
                variant="secondary",
                command=lambda selected_role=role: self._add_r_script(selected_role),
                width=132,
            ).grid(row=1, column=1, padx=(8, 0), pady=(4, 0))
            ctk.CTkLabel(
                role_frame,
                textvariable=self._script_status_vars[role],
                anchor="w",
                text_color=COLOR_PAIRS["muted_text"],
            ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
            self._refresh_r_script_choices(role)

        self.footer = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.footer.pack(fill="x", padx=16, pady=(0, 14))
        self.apply_status_var = tk.StringVar(master=self, value="")
        for variable in self.sdb_default_vars.values():
            variable.trace_add("write", self._mark_dirty)
        ctk.CTkLabel(
            self.footer,
            textvariable=self.apply_status_var,
            anchor="w",
            text_color=COLOR_PAIRS["muted_text"],
        ).pack(side="left")
        AppButton(
            self.footer,
            text="Close",
            variant="secondary",
            command=self._close,
            width=104,
        ).pack(side="right")
        AppButton(
            self.footer,
            text="Apply",
            variant="primary",
            command=self._apply_changes,
            width=104,
        ).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self.geometry(_center_geometry(self, 720, 720, parent=parent))
        self.deiconify()
        synchronize_window_chrome(
            self,
            background=COLOR_PAIRS["window_chrome"],
            foreground=COLOR_PAIRS["text"],
            border=COLOR_PAIRS["window_chrome"],
        )
        self.grab_set()
        self.focus_force()

    def _settings_section(self, title, *, row):
        section = ctk.CTkFrame(
            self.settings_panel,
            fg_color=COLOR_PAIRS["surface_subtle"],
            corner_radius=SHAPES.corner_radius_md,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
        )
        section.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 10))
        section.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            section,
            text=title,
            anchor="w",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.subtitle_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 4))
        body = ctk.CTkFrame(section, fg_color="transparent", corner_radius=0)
        body.grid(row=1, column=0, columnspan=2, sticky="ew")
        body.grid_columnconfigure(1, weight=1)
        return body

    @staticmethod
    def _labeled_control(parent, row, label, control):
        ctk.CTkLabel(
            parent,
            text=label,
            anchor="w",
            text_color=COLOR_PAIRS["text"],
        ).grid(row=row, column=0, sticky="w", padx=(14, 18), pady=6)
        control.grid(row=row, column=1, sticky="e", padx=(0, 14), pady=6)

    def _mark_dirty(self, *_args) -> None:
        self.apply_status_var.set("Unsaved changes")

    def _validated_sdb_defaults(self):
        try:
            width = int(self.sdb_default_vars["sdb_raw_width"].get())
            height = int(self.sdb_default_vars["sdb_raw_height"].get())
            offset = int(self.sdb_default_vars["sdb_raw_offset"].get())
            if width < 1 or height < 1 or offset < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "SDB Defaults",
                "Width and height must be positive integers; offset must be zero or greater.",
                parent=self,
            )
            return None
        return width, height, offset, bool(self.sdb_little_endian_var.get())

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
        source = "Default" if choice.is_default else str(choice.source).title()
        return f"{source} — {cls._compact_script_name(choice.path.name)}"

    def _refresh_r_script_choices(self, role, *, select_path=None) -> None:
        choices = self._step3.available_r_scripts(role)
        self._script_choices[role] = choices
        by_label = {}
        labels = []
        for choice in choices:
            label = self._script_display_label(choice)
            if label in by_label:
                label = f"{label} ({len(labels) + 1})"
            labels.append(label)
            by_label[label] = choice
        self._script_by_label[role] = by_label
        labels = labels or ["No R scripts found"]
        self.script_menus[role].configure(values=labels)
        selected_path = Path(select_path).resolve() if select_path is not None else self._step3._selected_r_script_path(role)
        selected = next(
            (choice for choice in choices if selected_path is not None and choice.path.resolve() == selected_path.resolve()),
            choices[0] if choices else None,
        )
        label = next(
            (display for display, choice in by_label.items() if choice == selected),
            labels[0],
        )
        self._script_vars[role].set(label)
        if selected is not None:
            state = "Pending" if select_path is not None else "Active"
            self._script_status_vars[role].set(
                f"{state}: {self._compact_script_name(selected.path.name)}"
            )
            if select_path is not None:
                self._mark_dirty()
        else:
            self._script_status_vars[role].set("No runnable script is available")

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
        self._script_status_vars[role].set(
            f"Pending: {self._compact_script_name(choice.path.name)} — click Apply"
        )
        self._mark_dirty()

    def _add_r_script(self, role) -> None:
        if self._step3._busy:
            messagebox.showwarning(
                "Step 3 is running",
                "Wait for the current Step 3 batch to finish before adding an R script.",
                parent=self,
            )
            return
        selected = filedialog.askopenfilename(
            parent=self,
            title="Add main processing R script" if role == "main" else "Add output R script",
            filetypes=(("R scripts", "*.R"), ("All files", "*.*")),
        )
        if not selected:
            return
        try:
            imported = self._step3.import_r_script_for_role(Path(selected), role)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Add R Script", f"Could not add the R script.\n{exc}", parent=self)
            return
        self._refresh_r_script_choices(role, select_path=imported)

    def _apply_changes(self) -> None:
        sdb_defaults = self._validated_sdb_defaults()
        if sdb_defaults is None:
            return
        selected_scripts = {}
        for role, _label in self.SCRIPT_ROLES:
            choice = self._script_by_label[role].get(self._script_vars[role].get())
            if choice is None:
                messagebox.showerror(
                    "R Script Settings",
                    f"Choose an available {role} R script before applying settings.",
                    parent=self,
                )
                return
            selected_scripts[role] = choice
        if self._step3._busy:
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
                return

        width, height, offset, little_endian = sdb_defaults
        self._preferences.set("check_for_updates", bool(self.update_checks_var.get()))
        self._preferences.set("sdb_raw_width", width)
        self._preferences.set("sdb_raw_height", height)
        self._preferences.set("sdb_raw_offset", offset)
        self._preferences.set("sdb_little_endian", little_endian)
        self._step1.set_sdb_parameter_defaults(
            width=width,
            height=height,
            offset=offset,
            little_endian=little_endian,
        )
        for role, choice in selected_scripts.items():
            self._step3.select_r_script(role, choice.path)
            self._script_status_vars[role].set(
                f"Active: {self._compact_script_name(choice.path.name)}"
            )
        self._set_appearance_command(self.appearance_menu.get())
        self.appearance_menu.set(normalize_appearance_mode(self.appearance_menu.get()))
        self.apply_status_var.set("All settings applied")

    def _open_r_setup(self) -> None:
        if self._r_setup_wizard is not None:
            return
        if self._step3._busy:
            messagebox.showwarning(
                "Step 3 is running",
                "Wait for the current Step 3 batch to finish before changing the R environment.",
                parent=self,
            )
            return
        from aidas.steps.step3_flatten import RSetupWizard

        self.settings_panel.pack_forget()
        self.footer.pack_forget()
        self.resizable(True, True)
        self._r_setup_wizard = RSetupWizard(
            self._step3,
            self,
            on_finish=lambda _result: self._refresh_r_setup_state(),
            close_command=self._close_r_setup,
        )
        self._r_setup_wizard.pack(fill="both", expand=True)
        self.geometry(_center_geometry(self, 900, 650, parent=self._parent))

    def _refresh_r_setup_state(self) -> None:
        self._step3._refresh_input_status()

    def _close_r_setup(self) -> None:
        wizard = self._r_setup_wizard
        self._r_setup_wizard = None
        if wizard is not None:
            wizard.destroy()
        self.settings_panel.pack(fill="both", expand=True, padx=16, pady=(16, 8))
        self.footer.pack(fill="x", padx=16, pady=(0, 14))
        self.resizable(False, False)
        self.geometry(_center_geometry(self, 720, 720, parent=self._parent))

    def _close(self) -> None:
        if self._r_setup_wizard is not None and self._r_setup_wizard.busy:
            return
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()


class AIDaSApp(ctk.CTk):
    """Root application window."""

    # The main window owns a client-drawn Windows caption.  Disabling CTk's
    # withdraw/restyle title-bar routine prevents it from restoring WS_CAPTION
    # during later appearance changes. Dialogs retain CTk's native behavior.
    _deactivate_windows_window_header_manipulation = True

    def __init__(self) -> None:
        enable_per_monitor_dpi_awareness()
        ctk.set_default_color_theme("dark-blue")
        ctk.set_appearance_mode("System")
        super().__init__()
        self.withdraw()
        self.title("AIDaS — Retinal Image Processing")
        self.configure(fg_color=COLOR_PAIRS["application"])
        bounds = work_area_bounds(self)
        physical_width, physical_height = fractional_size_of_bounds(
            bounds, LAYOUT.screen_fraction
        )
        app_width, app_height = logical_window_size(
            self,
            physical_width,
            physical_height,
        )
        self._startup_window_size = (app_width, app_height)
        self.geometry(f"{app_width}x{app_height}")
        self.minsize(
            min(LAYOUT.minimum_width, app_width),
            min(LAYOUT.minimum_height, app_height),
        )
        self._about_dialog = None
        self._settings_dialog = None
        self._set_app_icon()

        self._splash_started_at = time.monotonic()
        self._splash = SplashWindow(
            self,
            logo_path=resource_path(os.path.join("assets", "aidas.png")),
            title=APP_TITLE,
            subtitle=APP_SUBTITLE,
            affiliation=f"{LAB_ACRONYM}  ·  {UNIVERSITY_NAME}",
            lab_name=LAB_NAME,
            copyright_notice=COPYRIGHT_NOTICE,
        )
        self._set_splash_progress(3, "Starting AIDaS...")

        try:
            self._build_application()
        except Exception:
            try:
                splash = self._splash
                if splash is not None and splash.winfo_exists():
                    splash.destroy()
            finally:
                self.destroy()
            raise

        self._set_splash_progress(100, "Ready")
        elapsed_ms = int((time.monotonic() - self._splash_started_at) * 1000)
        delay_ms = max(0, SPLASH_MINIMUM_MS - elapsed_ms)
        self._startup_finished = False
        self._finish_startup_after_id = self.after(delay_ms, self._finish_startup)

    def _set_splash_progress(self, value: float, message: str) -> None:
        """Paint one startup stage while the main window is still hidden."""
        splash = getattr(self, "_splash", None)
        if splash is None or not splash.winfo_exists():
            return
        splash.set_progress(value, message)
        self.update_idletasks()
        self.update()

    def _build_application(self) -> None:
        """Build the main UI while the splash remains visible."""
        # Keep scientific and imaging imports behind the splash and expose each
        # expensive stage instead of leaving the startup window motionless.
        self._set_splash_progress(8, "Loading Step 1 image tools...")
        from aidas.steps.step1_resize_raw import Step1Frame

        self._set_splash_progress(18, "Loading Step 2 canvas and AI tools...")
        from aidas.steps.step2_annotate import Step2Frame

        self._set_splash_progress(30, "Loading Step 3 flattening tools...")
        from aidas.steps.step3_flatten import Step3Frame

        self._set_splash_progress(42, "Loading Step 4 analysis tools...")
        from aidas.steps.step4_analyze_isez import Step4Frame

        self._set_splash_progress(50, "Loading preferences...")
        self.preferences = Config()
        self._set_splash_progress(54, "Applying the interface theme...")
        self.style = ttk.Style(self)
        self.appearance_mode = normalize_appearance_mode(
            self.preferences.get("appearance_mode", self.preferences.get("theme", "System"))
        )
        apply_appearance_mode(
            self.appearance_mode,
            root=self,
            style=self.style,
        )

        self._set_splash_progress(58, "Starting application services...")
        self.update_controller = UpdateController(
            self,
            preferences=self.preferences,
            current_version=__version__,
            status_callback=self._set_status_message,
            restart_blocker_callback=self._update_restart_blocker,
            install_callback=self._queue_update_install,
        )
        self.window_title_bar = create_custom_windows_title_bar(
            self,
            title=self.title(),
            logo_path=self._resource_path(os.path.join("assets", "aidas.png")),
        )
        if self.window_title_bar is not None:
            self.window_title_bar.pack(side="top", fill="x")
            # SWP_FRAMECHANGED can alter Tk's client dimensions. Preserve the
            # logical startup size before the rest of the shell is composed.
            width, height = self._startup_window_size
            reassert_client_size(self, width, height)
        self._build_menu()
        self.bind_all("<Alt-F4>", lambda _event: self.destroy())

        self._set_splash_progress(62, "Creating the application workspace...")
        self.header = WorkflowHeader(
            self,
            version=__version__,
            on_step_selected=self._select_workflow_step,
            on_settings_selected=self._show_settings,
            on_help_selected=self._show_about,
            logo_path=self._resource_path(os.path.join("assets", "aidas.png")),
            settings_icon_path=self._resource_path(
                os.path.join("assets", "iconify-fluent-color--settings-32.png")
            ),
            help_icon_path=self._resource_path(
                os.path.join("assets", "iconify-fluent-color--question-circle-32.png")
            ),
        )
        self.header.pack(side="top", fill="x")
        self.status_bar = AppStatusBar(
            self,
            text=f"AIDaS v{__version__} — ready",
        )
        self.status_bar.pack(side="bottom", fill="x")
        # Preserve the historical label attribute used by update callbacks.
        self.status = self.status_bar.label

        self.notebook = ttk.Notebook(self, style="AIDaS.TNotebook")
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_workflow_tab_changed, add="+")

        self._set_splash_progress(66, "Preparing Step 1 - Load, Resize & Crop...")
        self.step1 = Step1Frame(
            self.notebook,
            preferences=self.preferences,
            on_processed_image=self._on_step1_processed_image,
            on_batch_segment_folders=self._on_step1_batch_segment_folders,
        )
        self.notebook.add(self.step1, text="  Step 1 — Load, Resize & Crop  ")

        self._set_splash_progress(74, "Preparing Step 2 - Annotate and Segment...")
        self.step2 = Step2Frame(
            self.notebook,
            preferences=self.preferences,
            source_step=self.step1,
            on_output_folder_changed=self._on_step2_output_folder_changed,
            on_continue_to_step3=self._on_step2_continue_to_step3,
        )
        self.notebook.add(self.step2, text="  Step 2 — Annotate and Segment  ")

        self._set_splash_progress(83, "Preparing Step 3 - Flatten Retina...")
        self.step3 = Step3Frame(self.notebook, preferences=self.preferences)
        self.notebook.add(self.step3, text="  Step 3 — Flatten Retina  ")

        self._set_splash_progress(91, "Preparing Step 4 - Analyze ISEZ...")
        self.step4 = Step4Frame(self.notebook, preferences=self.preferences)
        self.notebook.add(self.step4, text="  Step 4 — Analyze ISEZ  ")

        self._set_splash_progress(97, "Finalizing the main window...")
        self.header.select_step(0)
        refresh_native_widgets(self)
        self._last_effective_appearance = ctk.get_appearance_mode()
        self._appearance_watch_after_id = self.after(1500, self._watch_system_appearance)

    def _finish_startup(self) -> None:
        """Close the splash and reveal the fully initialized main window."""
        if self._startup_finished:
            return
        self._startup_finished = True
        self._finish_startup_after_id = None
        splash = getattr(self, "_splash", None)
        if splash is not None and splash.winfo_exists():
            splash.destroy()
        self._splash = None
        width, height = self._startup_window_size
        self.geometry(_center_geometry(self, width, height))
        self.deiconify()
        self.update_idletasks()
        if self.window_title_bar is not None:
            # Some Windows builds recalculate the captionless frame when it is
            # first mapped. Reassert the requested client size before centering.
            reassert_client_size(self, width, height)
        self._center_window(account_for_decorations=True)
        self.lift()
        self.update()
        synchronize_window_chrome(
            self,
            background=COLOR_PAIRS["window_chrome"],
            foreground=COLOR_PAIRS["text"],
            border=COLOR_PAIRS["window_chrome"],
        )
        self.focus_force()
        self.after(1500, self.update_controller.check_automatically)

    @staticmethod
    def _resource_path(relative_path: str) -> str:
        """Resolve a resource path for source runs and PyInstaller bundles."""
        return resource_path(relative_path)

    def _build_menu(self) -> None:
        menu_bar = getattr(self, "menu_bar", None)
        if menu_bar is not None:
            menu_bar.set_appearance(self.appearance_mode)
            return

        self.menu_bar = ApplicationMenuBar(
            self,
            appearance_modes=APPEARANCE_MODES,
            current_appearance=self.appearance_mode,
            set_appearance_command=self._set_theme,
            browse_sdb_command=self._menu_browse_sdb,
            check_updates_command=self.update_controller.check_now,
            about_command=self._show_about,
            exit_command=self.destroy,
        )
        self.menu_bar.pack(side="top", fill="x")
        # Preserve the historical attribute for integrations that only need a
        # handle to the application menu surface.
        self.menubar = self.menu_bar

    def _set_status_message(self, message: str) -> None:
        """Show a transient application-level status without assuming startup is complete."""
        status = getattr(self, "status", None)
        if status is not None:
            status.configure(text=f"AIDaS v{__version__} — {message}")

    def _select_workflow_step(self, index: int) -> None:
        """Select a workflow page without exposing its container to the header."""

        notebook = getattr(self, "notebook", None)
        if notebook is None:
            return
        try:
            notebook.select(int(index))
        except (tk.TclError, TypeError, ValueError):
            return

    def _on_workflow_tab_changed(self, _event=None) -> None:
        """Synchronize navigation and render deferred work for the active step."""

        notebook = getattr(self, "notebook", None)
        header = getattr(self, "header", None)
        if notebook is None:
            return
        try:
            selected_index = notebook.index(notebook.select())
        except (tk.TclError, TypeError, ValueError):
            return

        if header is not None:
            try:
                header.select_step(selected_index)
            except (tk.TclError, TypeError, ValueError):
                pass

        step2 = getattr(self, "step2", None)
        if step2 is None:
            return
        try:
            step2_selected = selected_index == notebook.index(step2)
        except (tk.TclError, TypeError, ValueError):
            return
        if step2_selected:
            step2.render_pending_external_image()

    def _watch_system_appearance(self) -> None:
        """Keep retained ttk/native widgets synced with OS appearance changes."""

        current = ctk.get_appearance_mode()
        if self.appearance_mode == "System" and current != self._last_effective_appearance:
            apply_appearance_mode("System", root=self, style=self.style)
            step3 = getattr(self, "step3", None)
            if step3 is not None:
                self.after_idle(step3.refresh_appearance)
        self._last_effective_appearance = current
        self._appearance_watch_after_id = self.after(1500, self._watch_system_appearance)

    def _update_restart_blocker(self) -> str | None:
        """Describe work that must finish before replacing the application."""
        step2 = getattr(self, "step2", None)
        if step2 is not None and getattr(step2, "_segmenter_running", False):
            return "Step 2 AI segmentation is still running."

        step3 = getattr(self, "step3", None)
        if step3 is not None:
            if getattr(step3, "_busy", False):
                return "Step 3 R batch processing is still running."
            setup_panel = getattr(step3, "r_setup_panel", None)
            if setup_panel is not None and getattr(setup_panel, "busy", False):
                return "Step 3 R or package setup is still running."
        return None

    def _queue_update_install(self, installer_path) -> None:
        """Close the UI; main() starts Setup after releasing the app mutex."""
        self._pending_update_installer = installer_path
        self.destroy()

    def _set_app_icon(self) -> None:
        """Set the taskbar icon if available; never fail startup if missing."""
        ico_path = self._resource_path(os.path.join("assets", "aidas.ico"))
        png_path = self._resource_path(os.path.join("assets", "aidas.png"))

        if os.path.isfile(ico_path):
            self._icon_ico_path = ico_path
            try:
                self.iconbitmap(ico_path)
                try:
                    self.iconbitmap(default=ico_path)
                except tk.TclError:
                    pass
            except tk.TclError:
                pass

        if os.path.isfile(png_path):
            try:
                image = tk.PhotoImage(file=png_path)
                self.iconphoto(True, image)
                self._icon_image_ref = image
            except tk.TclError:
                pass

    def _center_window(self, *, account_for_decorations: bool = False) -> None:
        """Center the standard application window in the active work area."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        if not account_for_decorations:
            x, y = centered_position(work_area_bounds(self), width, height)
            self.geometry(f"{x:+d}{y:+d}")
            return

        frame_left = max(0, self.winfo_rootx() - self.winfo_x())
        frame_top = max(0, self.winfo_rooty() - self.winfo_y())
        x, y = centered_decorated_position(
            work_area_bounds(self),
            width,
            height,
            frame_left=frame_left,
            frame_top=frame_top,
        )
        self.geometry(f"{x:+d}{y:+d}")

    def _menu_browse_sdb(self) -> None:
        self.notebook.select(0)
        directory = filedialog.askdirectory(
            title="Select parent folder containing SDB subfolders",
            initialdir=self.step1.sdb_dir_var.get() or None,
        )
        if directory:
            self.step1.set_sdb_directory(directory)
            self.step1.refresh_sdb_list(preview_first=True)

    def _set_theme(self, theme_name: str) -> None:
        """Apply one unified CTk/native appearance and save the preference."""

        self.appearance_mode = apply_appearance_mode(
            theme_name,
            root=self,
            style=self.style,
        )
        self.preferences.set("appearance_mode", self.appearance_mode)
        self._last_effective_appearance = ctk.get_appearance_mode()
        self._build_menu()
        self.status.configure(
            text=f"AIDaS v{__version__} — appearance changed to {self.appearance_mode}"
        )
        self.after_idle(lambda: refresh_native_widgets(self))
        step3 = getattr(self, "step3", None)
        if step3 is not None:
            self.after_idle(step3.refresh_appearance)

    def _on_step1_processed_image(self, image, source_path) -> None:
        """Receive a Step 1 crop without repainting a hidden Step 2 page."""

        step2 = getattr(self, "step2", None)
        if step2 is None:
            return
        notebook = getattr(self, "notebook", None)
        defer_render = True
        if notebook is not None:
            try:
                defer_render = notebook.index(notebook.select()) != notebook.index(step2)
            except (tk.TclError, TypeError, ValueError):
                pass
        step2.load_external_image(
            image,
            source_path=source_path,
            defer_render=defer_render,
        )

    def _on_step1_batch_segment_folders(self, folders) -> None:
        """Open Step 2 and batch-segment completed Step 1 folders."""
        step2 = getattr(self, "step2", None)
        if step2 is None:
            return
        self.notebook.select(step2)
        self.update_idletasks()
        step2.start_batch_segmentation_for_folders(folders)

    def _on_step2_output_folder_changed(self, folder) -> None:
        """Keep Step 3 pointed at Step 2's MARKED output folder."""
        if getattr(self, "step3", None) is not None:
            self.step3.set_input_folder(folder)

    def _on_step2_continue_to_step3(self, folders) -> None:
        """Open Step 3 with the exact nasal/temporal folders saved in Step 2."""
        step3 = getattr(self, "step3", None)
        if step3 is None or not folders:
            return
        self.notebook.select(step3)
        self.update_idletasks()
        step3.open_batch_folders(folders)

    def _show_about(self) -> None:
        """Open one modal About window, or focus the existing one."""
        dialog = self._about_dialog
        try:
            if dialog is not None and dialog.winfo_exists():
                dialog.lift()
                dialog.focus_force()
                return
        except tk.TclError:
            pass
        self._about_dialog = AboutDialog(self)

    def _show_settings(self) -> None:
        """Open one settings window, or focus the existing instance."""

        dialog = self._settings_dialog
        try:
            if dialog is not None and dialog.winfo_exists():
                dialog.lift()
                dialog.focus_force()
                return
        except tk.TclError:
            pass
        self._settings_dialog = SettingsDialog(
            self,
            preferences=self.preferences,
            appearance_mode=self.appearance_mode,
            set_appearance_command=self._set_theme,
            step1=self.step1,
            step3=self.step3,
        )


def _show_native_notice(title: str, message: str, *, error: bool = False) -> None:
    """Show a startup notice without constructing a second Tk application."""
    if os.name == "nt":
        icon_flag = 0x10 if error else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
        ctypes.windll.user32.MessageBoxW(None, message, title, icon_flag | 0x00010000)
    else:
        print(f"{title}: {message}", file=sys.stderr)


def main() -> int:
    """Run AIDaS if no other desktop instance owns the process guard."""
    guard = SingleInstanceGuard()
    try:
        if not guard.acquire():
            _show_native_notice(APP_TITLE, "AIDaS is already running.")
            return 0
    except OSError as exc:
        _show_native_notice(
            APP_TITLE,
            f"AIDaS could not verify that only one instance is running.\n\n{exc}",
            error=True,
        )
        return 1

    app = None
    pending_update = None
    interrupted = False
    try:
        app = AIDaSApp()
        try:
            app.mainloop()
        except KeyboardInterrupt:
            # Stopping a VS Code debug session (or pressing Ctrl+C in a
            # terminal) interrupts Tk's blocking event loop.  Treat that as a
            # normal user-requested shutdown instead of emitting a traceback.
            interrupted = True
            try:
                app.destroy()
            except tk.TclError:
                pass
        if not interrupted:
            pending_update = getattr(app, "_pending_update_installer", None)
    finally:
        guard.close()

    if interrupted:
        return 0

    if pending_update is not None:
        try:
            launch_installer(pending_update)
        except Exception as exc:
            _show_native_notice(
                "AIDaS Update",
                f"The verified update was downloaded, but Windows could not start the installer.\n\n{exc}",
                error=True,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
