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
from PIL import Image, ImageTk

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
from aidas.ui.classic import build_classic_application_menu
from aidas.ui.components import AppButton, AppStatusBar, WorkflowHeader
from aidas.ui.menu_bar import ApplicationMenuBar
from aidas.ui.splash import SplashWindow
from aidas.ui.theme import (
    APPEARANCE_MODES,
    COLOR_PAIRS,
    COLORS,
    INTERFACE_MODES,
    SHAPES,
    TYPOGRAPHY,
    apply_appearance_mode,
    normalize_appearance_mode,
    normalize_interface_mode,
    refresh_interface_widgets,
    refresh_native_widgets,
    set_interface_mode,
)
from aidas.ui.title_bar import (
    cache_native_window_handle,
    create_custom_windows_title_bar,
    reassert_client_size,
)
from aidas.ui.windowing import (
    centered_logical_geometry,
    logical_window_size,
    physical_window_size,
    synchronize_window_chrome,
)
from aidas.utils.ui_layout import LAYOUT
from aidas.utils.ui_utils import (
    action_button,
    apply_app_icon_to,
    load_color_close_ctk_icon,
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

    def __init__(self, parent: tk.Misc, *, interface_mode="Modern") -> None:
        super().__init__(parent)
        self.withdraw()
        self.title("About AIDaS")
        self._presentation_mode = normalize_interface_mode(interface_mode)
        self._classic_about = self._presentation_mode == "Classic"
        self.configure(
            fg_color=(
                COLOR_PAIRS["surface"]
                if self._classic_about
                else COLOR_PAIRS["application"]
            )
        )
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

        if self._classic_about:
            self._build_classic_about(content_wrap)
            self.protocol("WM_DELETE_WINDOW", self._close)
            self.bind("<Escape>", lambda _event: self._close())
            self.bind("<Return>", lambda _event: self._close())
            self.geometry(
                _center_geometry(
                    self,
                    dialog_width,
                    dialog_height,
                    parent=parent,
                )
            )
            self.deiconify()
            synchronize_window_chrome(
                self,
                background=COLOR_PAIRS["window_chrome"],
                foreground=COLOR_PAIRS["text"],
                border=COLOR_PAIRS["window_chrome"],
            )
            self.grab_set()
            self.ok_button.focus_set()
            return

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
            text=f"{APP_SUBTITLE} - Version {__version__}",
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
        ).grid(row=8, column=0, padx=24, pady=(10, 18))

        button_panel = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        button_panel.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))
        self.ok_button = AppButton(
            button_panel,
            text="OK",
            width=96,
            variant="primary",
            command=self._close,
        )
        self.ok_button.pack(side="right")

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
        self.ok_button.focus_set()

    def _build_classic_about(self, content_wrap: int) -> None:
        """Build an opaque native About surface for the Classic interface."""

        self.classic_footer = ttk.Frame(
            self,
            style="AIDaS.Content.TFrame",
            padding=(12, 8),
        )
        self.classic_footer.pack(side="bottom", fill="x")
        ttk.Separator(self).pack(side="bottom", fill="x")

        self.classic_content = ttk.Frame(
            self,
            style="AIDaS.Content.TFrame",
            padding=(28, 16, 28, 10),
        )
        self.classic_content.pack(side="top", fill="both", expand=True)
        self.classic_content.grid_columnconfigure(0, weight=1)

        logo_path = resource_path(os.path.join("assets", "aidas.png"))
        with Image.open(logo_path) as logo:
            logo_source = logo.convert("RGBA").resize(
                (88, 88),
                Image.Resampling.LANCZOS,
            )
        self.logo_image = ImageTk.PhotoImage(logo_source, master=self)
        ttk.Label(
            self.classic_content,
            image=self.logo_image,
            anchor="center",
        ).grid(row=0, column=0, pady=(2, 4))

        def add_label(
            row,
            text,
            *,
            font=None,
            style=None,
            pady=(0, 0),
            wraplength=content_wrap,
        ):
            options = {
                "text": text,
                "anchor": "center",
                "justify": "center",
                "wraplength": wraplength,
            }
            if font is not None:
                options["font"] = font
            if style is not None:
                options["style"] = style
            label = ttk.Label(self.classic_content, **options)
            label.grid(row=row, column=0, sticky="ew", pady=pady)
            return label

        add_label(
            1,
            APP_TITLE,
            font=(TYPOGRAPHY.family, TYPOGRAPHY.title_size, "bold"),
        )
        add_label(
            2,
            f"{APP_SUBTITLE} - Version {__version__}",
            style="AIDaS.Muted.TLabel",
            pady=(2, 14),
        )
        add_label(
            3,
            LAB_ACRONYM,
            font=(TYPOGRAPHY.family, TYPOGRAPHY.subtitle_size, "bold"),
        )
        add_label(
            4,
            LAB_NAME,
            font=(TYPOGRAPHY.family, TYPOGRAPHY.body_size, "bold"),
            pady=(3, 0),
        )
        link = add_label(
            5,
            LAB_URL_TEXT,
            style="AIDaS.Link.TLabel",
            pady=(10, 0),
        )
        link.configure(cursor="hand2")
        link.bind("<Button-1>", lambda _event: webbrowser.open_new_tab(LAB_URL))
        add_label(
            6,
            " ".join(LAB_DESCRIPTION.splitlines()),
            style="AIDaS.Muted.TLabel",
            pady=(16, 0),
        )
        add_label(
            7,
            UNIVERSITY_NAME,
            font=(TYPOGRAPHY.family, TYPOGRAPHY.body_size, "bold"),
            pady=(14, 0),
        )
        add_label(
            8,
            COPYRIGHT_NOTICE,
            style="AIDaS.Muted.TLabel",
            pady=(10, 8),
        )

        self.ok_button = ttk.Button(
            self.classic_footer,
            text="OK",
            width=12,
            command=self._close,
        )
        self.ok_button.pack(side="right")

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
        interface_mode,
        set_interface_command,
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
        self._set_interface_command = set_interface_command
        self._set_appearance_command = set_appearance_command
        self._step1 = step1
        self._step3 = step3
        self._presentation_mode = normalize_interface_mode(interface_mode)
        self._classic_settings = self._presentation_mode == "Classic"
        self._presentation_refresh_after_id = None
        self._presentation_refresh_pending = False
        self._applying_changes = False
        self._classic_mousewheel_binding_id = None
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

        if self._classic_settings:
            self._build_classic_settings(
                preferences=preferences,
                interface_mode=interface_mode,
                appearance_mode=appearance_mode,
                step3=step3,
            )
            self.protocol("WM_DELETE_WINDOW", self._close)
            self.bind("<Escape>", lambda _event: self._close())
            self.geometry(_center_geometry(self, 720, 720, parent=parent))
            self.deiconify()
            self.grab_set()
            self.focus_force()
            return

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
        self.interface_menu = ctk.CTkOptionMenu(
            general,
            values=list(INTERFACE_MODES),
            command=self._interface_choice_changed,
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
        self._labeled_control(general, 0, "Interface", self.interface_menu)
        self.interface_menu.set(normalize_interface_mode(interface_mode))
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
        self._labeled_control(general, 1, "Appearance", self.appearance_menu)
        self.appearance_menu.set(appearance_mode)
        self._sync_interface_controls(interface_mode, appearance_mode)
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
        self.update_switch.grid(row=2, column=0, columnspan=2, sticky="w", padx=14, pady=(8, 12))

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
        close_icon = load_color_close_ctk_icon(self)
        AppButton(
            self.footer,
            text="Close",
            variant="secondary",
            command=self._close,
            width=104,
            image=close_icon,
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

    def _build_classic_settings(
        self,
        *,
        preferences,
        interface_mode,
        appearance_mode,
        step3,
    ) -> None:
        """Build a native property-sheet surface for the Classic interface."""

        self.settings_panel = ttk.Frame(self, padding=(10, 10, 10, 0))
        self.settings_panel.pack(fill="both", expand=True)
        self.settings_panel.rowconfigure(0, weight=1)
        self.settings_panel.columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            self.settings_panel,
            background=COLORS.application,
            highlightthickness=1,
            highlightbackground=COLORS.border,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(
            self.settings_panel,
            orient="vertical",
            command=canvas.yview,
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        content = ttk.Frame(canvas, padding=14)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")

        def update_scroll_region(_event=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except tk.TclError:
                pass

        def fit_content_width(event):
            try:
                canvas.itemconfigure(content_window, width=max(1, event.width))
            except tk.TclError:
                pass

        content.bind("<Configure>", update_scroll_region, add="+")
        canvas.bind("<Configure>", fit_content_width, add="+")
        def scroll_classic_settings(event):
            # Descendant widgets receive the toplevel bindtag, whereas a
            # canvas-only binding stops working as soon as the pointer is over
            # a label, entry, or combobox.  Ignore the binding while the
            # embedded R setup wizard has replaced the settings property sheet.
            try:
                if self.settings_panel.winfo_manager() != "pack":
                    return None
                direction = -1 if event.delta > 0 else 1
                canvas.yview_scroll(direction, "units")
            except tk.TclError:
                return None
            return "break"

        self._classic_mousewheel_binding_id = self.bind(
            "<MouseWheel>",
            scroll_classic_settings,
            add="+",
        )

        ttk.Label(
            content,
            text="Settings",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        general = ttk.LabelFrame(content, text="General", padding=10)
        general.pack(fill="x", pady=(0, 10))
        general.columnconfigure(1, weight=1)
        ttk.Label(general, text="Interface").grid(row=0, column=0, sticky="w", pady=4)
        self.interface_menu = ttk.Combobox(
            general,
            values=list(INTERFACE_MODES),
            state="readonly",
            width=18,
        )
        self.interface_menu.set(normalize_interface_mode(interface_mode))
        self.interface_menu.grid(row=0, column=1, sticky="e", pady=4)
        self.interface_menu.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._interface_choice_changed(self.interface_menu.get()),
            add="+",
        )
        ttk.Label(general, text="Appearance").grid(row=1, column=0, sticky="w", pady=4)
        self.appearance_menu = ttk.Combobox(
            general,
            values=list(APPEARANCE_MODES),
            state="disabled",
            width=18,
        )
        self.appearance_menu.set(normalize_appearance_mode(appearance_mode))
        self.appearance_menu.grid(row=1, column=1, sticky="e", pady=4)
        self.appearance_menu.bind(
            "<<ComboboxSelected>>",
            self._mark_dirty,
            add="+",
        )
        self.update_checks_var = tk.BooleanVar(
            master=self,
            value=bool(preferences.get("check_for_updates", True)),
        )
        self.update_switch = ttk.Checkbutton(
            general,
            text="Check automatically for application updates",
            variable=self.update_checks_var,
            command=self._mark_dirty,
        )
        self.update_switch.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 2))

        sdb = ttk.LabelFrame(content, text="Default SDB image parameters", padding=10)
        sdb.pack(fill="x", pady=(0, 10))
        sdb.columnconfigure(1, weight=1)
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
            ttk.Label(sdb, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(
                sdb,
                textvariable=self.sdb_default_vars[key],
                width=18,
                justify="right",
            ).grid(row=row, column=1, sticky="e", pady=4)
        self.sdb_little_endian_var = tk.BooleanVar(
            master=self,
            value=bool(preferences.get("sdb_little_endian", True)),
        )
        ttk.Checkbutton(
            sdb,
            text="Little-endian byte order",
            variable=self.sdb_little_endian_var,
            command=self._mark_dirty,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 2))

        r_environment = ttk.LabelFrame(content, text="R environment", padding=10)
        r_environment.pack(fill="x", pady=(0, 10))
        ttk.Label(
            r_environment,
            text=f"Install or verify R {step3.R_REQUIRED_VERSION} and all required packages.",
        ).pack(anchor="w", pady=(0, 8))
        action_button(
            r_environment,
            self,
            "Install / set up R and packages",
            self._open_r_setup,
            "package",
            style="AIDaS.PrimaryAction.TButton",
        ).pack(anchor="w")

        scripts = ttk.LabelFrame(content, text="Step 3 R scripts", padding=10)
        scripts.pack(fill="x")
        scripts.columnconfigure(0, weight=1)
        ttk.Label(
            scripts,
            text="Choose each active script from its list, then click Apply.",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.script_menus = {}
        for row, (role, label) in enumerate(self.SCRIPT_ROLES):
            base_row = 1 + row * 3
            ttk.Label(scripts, text=label).grid(
                row=base_row,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(5, 2),
            )
            menu = ttk.Combobox(
                scripts,
                textvariable=self._script_vars[role],
                values=["No R scripts found"],
                state="readonly",
                width=62,
            )
            menu.grid(row=base_row + 1, column=0, sticky="ew")
            menu.bind(
                "<<ComboboxSelected>>",
                lambda _event, selected_role=role: self._select_r_script(selected_role),
                add="+",
            )
            self.script_menus[role] = menu
            ttk.Button(
                scripts,
                text="Add R script...",
                command=lambda selected_role=role: self._add_r_script(selected_role),
            ).grid(row=base_row + 1, column=1, padx=(8, 0))
            ttk.Label(
                scripts,
                textvariable=self._script_status_vars[role],
                style="AIDaS.Muted.TLabel",
            ).grid(row=base_row + 2, column=0, columnspan=2, sticky="w", pady=(2, 4))
            self._refresh_r_script_choices(role)

        self.footer = ttk.Frame(self, padding=(10, 8, 10, 10))
        self.footer.pack(fill="x")
        self.apply_status_var = tk.StringVar(master=self, value="")
        ttk.Label(
            self.footer,
            textvariable=self.apply_status_var,
            style="AIDaS.Muted.TLabel",
        ).pack(side="left", fill="x", expand=True)
        action_button(
            self.footer,
            self,
            "Close",
            self._close,
            "close",
        ).pack(side="right")
        action_button(
            self.footer,
            self,
            "Apply",
            self._apply_changes,
            "confirm",
            style="AIDaS.PrimaryAction.TButton",
        ).pack(side="right", padx=(0, 8))
        for variable in self.sdb_default_vars.values():
            variable.trace_add("write", self._mark_dirty)
        self._sync_interface_controls(interface_mode, appearance_mode)

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
        """Reopen Settings in the active shell after the current Apply returns."""

        if getattr(self, "_applying_changes", False):
            self._presentation_refresh_pending = True
            return
        if self._presentation_refresh_after_id is not None:
            return

        def refresh() -> None:
            self._presentation_refresh_after_id = None
            parent = self._parent
            show_settings = getattr(parent, "_show_settings", None)
            self._close()
            try:
                parent._settings_dialog = None
            except (AttributeError, TypeError):
                pass
            if callable(show_settings):
                show_settings()

        try:
            self._presentation_refresh_after_id = self.after_idle(refresh)
        except tk.TclError:
            self._presentation_refresh_after_id = None

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
        selected_interface = normalize_interface_mode(self.interface_menu.get())
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
                self._set_appearance_command(self.appearance_menu.get())
                self.appearance_menu.set(
                    normalize_appearance_mode(self.appearance_menu.get())
                )
            self._sync_interface_controls(
                applied_interface,
                self.appearance_menu.get(),
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
        if self._classic_settings:
            self.settings_panel.pack(fill="both", expand=True)
            self.footer.pack(fill="x")
        else:
            self.settings_panel.pack(fill="both", expand=True, padx=16, pady=(16, 8))
            self.footer.pack(fill="x", padx=16, pady=(0, 14))
        self.resizable(False, False)
        self.geometry(_center_geometry(self, 720, 720, parent=self._parent))

    def _close(self) -> None:
        if self._r_setup_wizard is not None and self._r_setup_wizard.busy:
            return
        binding_id = getattr(self, "_classic_mousewheel_binding_id", None)
        if binding_id is not None:
            try:
                self.unbind("<MouseWheel>", binding_id)
            except tk.TclError:
                pass
            self._classic_mousewheel_binding_id = None
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
        # Resolve the presentation before constructing even the splash so a
        # Classic launch never flashes modern chrome or rounded design tokens.
        self.interface_mode = set_interface_mode(
            Config.peek("interface_mode", "Modern"),
            redraw=False,
        )
        self.requested_interface_mode = self.interface_mode
        super().__init__()
        self._interface_refresh_after_id = None
        self._interface_switching = False
        self._normal_logical_client_size = None
        self._pending_normal_client_size = None
        self._normal_size_restore_after_id = None
        tk.Misc.bind_all(
            self,
            "<Map>",
            self._on_interface_widget_mapped,
            add="+",
        )
        self.bind(
            "<Configure>",
            self._remember_normal_client_size,
            add="+",
        )
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
        self._queue_interface_widget_refresh(include_splash=True)
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
        self.interface_mode = set_interface_mode(
            self.preferences.get("interface_mode", "Modern"),
            redraw=False,
        )
        self.requested_interface_mode = self.interface_mode
        self._set_splash_progress(54, "Applying the interface theme...")
        self.style = ttk.Style(self)
        self.appearance_mode = normalize_appearance_mode(
            self.preferences.get("appearance_mode", self.preferences.get("theme", "System"))
        )
        apply_appearance_mode(
            "Light" if self.interface_mode == "Classic" else self.appearance_mode,
            root=self,
            style=self.style,
            force_ctk_redraw=True,
        )
        self._queue_interface_widget_refresh(include_splash=True)

        self._set_splash_progress(58, "Starting application services...")
        self.update_controller = UpdateController(
            self,
            preferences=self.preferences,
            current_version=__version__,
            status_callback=self._set_status_message,
            restart_blocker_callback=self._update_restart_blocker,
            install_callback=self._queue_update_install,
        )
        self.window_title_bar = None
        self.menu_bar = None
        self._modern_menu_bar_cache = None
        self.classic_menu = None
        self.menubar = None
        if self.interface_mode == "Classic":
            # Resolve the unchanged HWND behind the splash so the first live
            # move to custom Modern chrome does not flush the entire finished
            # workflow tree merely to discover the native handle.
            cache_native_window_handle(self)
        self._install_modern_title_bar()
        if self.window_title_bar is not None:
            # SWP_FRAMECHANGED can alter Tk's client dimensions. Preserve the
            # logical startup size before the rest of the shell is composed.
            width, height = self._startup_window_size
            reassert_client_size(self, width, height)
        self._build_menu()
        self.bind_all("<Alt-F4>", lambda _event: self.destroy())

        self._set_splash_progress(62, "Creating the application workspace...")
        self.header = None
        self.status_bar = None
        self.status = None
        self._modern_header_cache = None
        self._modern_status_bar_cache = None
        self._build_workflow_header()
        self._build_status_surface(f"AIDaS v{__version__} — ready")

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
            is_step3_folder_active=self._is_step3_folder_active,
            get_step3_core_usage=self._step3_core_usage,
        )
        self.notebook.add(self.step2, text="  Step 2 — Annotate and Segment  ")

        self._set_splash_progress(83, "Preparing Step 3 - Flatten Retina...")
        self.step3 = Step3Frame(
            self.notebook,
            preferences=self.preferences,
            get_step2_core_usage=self._step2_core_usage,
        )
        self.notebook.add(self.step3, text="  Step 3 — Flatten Retina  ")

        self._set_splash_progress(91, "Preparing Step 4 - Analyze ISEZ...")
        self.step4 = Step4Frame(self.notebook, preferences=self.preferences)
        self.notebook.add(self.step4, text="  Step 4 — Analyze ISEZ  ")

        self._set_splash_progress(97, "Finalizing the main window...")
        if self.interface_mode == "Classic":
            # Build the reusable Modern-only surfaces while the startup splash
            # is already present.  A user's first Classic -> Modern selection
            # then swaps presentation immediately instead of constructing the
            # popup canvases and header for the first time on that click.
            self._prime_modern_shell_cache()
        if self.header is not None:
            self.header.select_step(0)
        refresh_native_widgets(self)
        self._queue_interface_widget_refresh(include_splash=True)
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
        if self.interface_mode == "Modern":
            synchronize_window_chrome(
                self,
                background=COLOR_PAIRS["window_chrome"],
                foreground=COLOR_PAIRS["text"],
                border=COLOR_PAIRS["window_chrome"],
            )
        self._queue_interface_widget_refresh()
        self.focus_force()
        self.after(1500, self.update_controller.check_automatically)

    @staticmethod
    def _resource_path(relative_path: str) -> str:
        """Resolve a resource path for source runs and PyInstaller bundles."""
        return resource_path(relative_path)

    def _queue_interface_widget_refresh(self, *, include_splash: bool = False) -> None:
        """Debounce one retained-widget interface refresh onto Tk's idle queue."""

        pending = self.__dict__.get("_interface_refresh_after_id")
        if pending is not None:
            try:
                self.after_cancel(pending)
            except (AttributeError, tk.TclError):
                pass
            self._interface_refresh_after_id = None

        def refresh() -> None:
            self._interface_refresh_after_id = None
            try:
                refresh_interface_widgets(self)
            except (AttributeError, tk.TclError):
                return
            if not include_splash:
                return
            splash = self.__dict__.get("_splash")
            try:
                if splash is not None and splash.winfo_exists():
                    refresh_interface_widgets(splash)
            except (AttributeError, tk.TclError):
                pass

        try:
            self._interface_refresh_after_id = self.after_idle(refresh)
        except (AttributeError, tk.TclError):
            refresh()

    def _on_interface_widget_mapped(self, _event=None) -> None:
        """Settle lazily mapped startup, dialog, and workflow CTk surfaces."""

        # Shell replacement maps several retained CTk surfaces at once.  The
        # switch transaction queues one consolidated pass after its geometry
        # work, so running an idle refresh for every intermediate map would
        # make the menu command block while hundreds of canvases redraw.
        if self.__dict__.get("_interface_switching", False):
            return
        self._queue_interface_widget_refresh(
            include_splash=self.__dict__.get("_splash") is not None
        )

    def _remember_normal_client_size(self, event=None) -> None:
        """Remember the last resizable client size for maximized UI switches."""

        if event is not None and getattr(event, "widget", self) is not self:
            return
        if self.__dict__.get("_interface_switching", False):
            return
        try:
            if str(self.state()).lower() != "normal":
                return
            pending = self.__dict__.get("_pending_normal_client_size")
            if pending is not None:
                if self.__dict__.get("_normal_size_restore_after_id") is None:

                    def restore_pending_size() -> None:
                        self._normal_size_restore_after_id = None
                        target = self.__dict__.get("_pending_normal_client_size")
                        if target is None or self.__dict__.get(
                            "_interface_switching", False
                        ):
                            return
                        try:
                            if str(self.state()).lower() != "normal":
                                return
                            # Clear before geometry/update_idletasks so the
                            # resulting Configure events cannot enqueue the
                            # same correction recursively.
                            self._pending_normal_client_size = None
                            title_bar = self.__dict__.get("window_title_bar")
                            if title_bar is not None:
                                physical_size = physical_window_size(self, *target)
                                title_bar.controller.resize_window(
                                    int(self.winfo_x()),
                                    int(self.winfo_y()),
                                    *physical_size,
                                )
                            else:
                                reassert_client_size(self, *target)
                            self._normal_logical_client_size = target
                        except (
                            AttributeError,
                            tk.TclError,
                            TypeError,
                            ValueError,
                        ):
                            self._pending_normal_client_size = target

                    # Wait for Windows/Tk to finish leaving the maximized
                    # state.  Running geometry correction in the same idle
                    # drain can race the title-bar state synchronizer and keep
                    # ``update()`` processing Configure events indefinitely.
                    self._normal_size_restore_after_id = self.after(
                        100,
                        restore_pending_size,
                    )
                return
            width = int(getattr(event, "width", self.winfo_width()))
            height = int(getattr(event, "height", self.winfo_height()))
            if width <= 1 or height <= 1:
                return
            self._normal_logical_client_size = logical_window_size(
                self,
                width,
                height,
            )
        except (AttributeError, tk.TclError, TypeError, ValueError):
            pass

    def _pack_shell_surface(self, widget: tk.Misc, *, side: str) -> None:
        """Pack shell chrome before the retained workflow notebook when present."""

        options = {"side": side, "fill": "x"}
        notebook = self.__dict__.get("notebook")
        try:
            if notebook is not None and notebook.winfo_manager() == "pack":
                options["before"] = notebook
        except (AttributeError, tk.TclError):
            pass
        widget.pack(**options)

    def _install_modern_title_bar(self):
        """Install a fresh custom caption, retaining native chrome as fallback."""

        self.window_title_bar = None
        if self.interface_mode != "Modern":
            return None
        title_bar = create_custom_windows_title_bar(
            self,
            title=self.title(),
            logo_path=self._resource_path(os.path.join("assets", "aidas.png")),
        )
        self.window_title_bar = title_bar
        if title_bar is not None:
            self._pack_shell_surface(title_bar, side="top")
        return title_bar

    def _restore_native_title_bar(self) -> bool:
        """Restore the system caption before discarding custom window controls."""

        title_bar = self.__dict__.get("window_title_bar")
        if title_bar is None:
            return True
        try:
            restored = title_bar.controller.restore_native_caption()
        except (AttributeError, tk.TclError):
            restored = False
        if not restored:
            return False
        try:
            title_bar.destroy()
        except tk.TclError:
            pass
        self.window_title_bar = None
        return True

    def _new_modern_workflow_header(self):
        """Construct the Modern workflow header without packing it."""

        return WorkflowHeader(
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

    def _build_workflow_header(self):
        """Create Modern navigation without owning or rebuilding workflow pages."""

        self.header = None
        if self.interface_mode != "Modern":
            return None
        cached = self.__dict__.get("_modern_header_cache")
        try:
            if cached is not None and cached.winfo_exists():
                self.header = cached
                self._pack_shell_surface(cached, side="top")
                return cached
        except (AttributeError, tk.TclError):
            self._modern_header_cache = None
        header = self._new_modern_workflow_header()
        self.header = header
        self._modern_header_cache = header
        self._pack_shell_surface(header, side="top")
        return header

    def _build_status_surface(self, text: str):
        """Create the status presentation for the active interface."""

        if self.interface_mode == "Modern":
            status_bar = self.__dict__.get("_modern_status_bar_cache")
            try:
                if status_bar is not None and not status_bar.winfo_exists():
                    status_bar = None
            except (AttributeError, tk.TclError):
                status_bar = None
            if status_bar is None:
                status_bar = AppStatusBar(self, text=str(text))
                self._modern_status_bar_cache = status_bar
            else:
                status_bar.label.configure(text=str(text))
            self.status_bar = status_bar
            self.status = status_bar.label
            self._pack_shell_surface(status_bar, side="bottom")
            return status_bar

        self.status_bar = None
        status = ttk.Label(
            self,
            text=str(text),
            style="AIDaS.Status.TLabel",
            anchor="w",
        )
        self.status = status
        self._pack_shell_surface(status, side="bottom")
        return status

    def _current_status_text(self) -> str:
        status = self.__dict__.get("status")
        try:
            return str(status.cget("text"))
        except (AttributeError, tk.TclError):
            return f"AIDaS v{__version__} — ready"

    def _destroy_status_surface(self) -> None:
        status_bar = self.__dict__.get("status_bar")
        status = self.__dict__.get("status")
        try:
            if status_bar is not None:
                status_bar.pack_forget()
                self._modern_status_bar_cache = status_bar
            elif status is not None:
                status.destroy()
        except tk.TclError:
            pass
        self.status_bar = None
        self.status = None

    def _destroy_workflow_header(self) -> None:
        header = self.__dict__.get("header")
        if header is not None:
            try:
                header.pack_forget()
                self._modern_header_cache = header
            except tk.TclError:
                pass
        self.header = None

    def _destroy_application_menus(self) -> None:
        """Remove both menu implementations and all of their root bindings."""

        menu_bar = self.__dict__.get("menu_bar")
        if menu_bar is not None:
            try:
                menu_bar.suspend()
                self._modern_menu_bar_cache = menu_bar
            except tk.TclError:
                pass
        classic_menu = self.__dict__.get("classic_menu")
        if classic_menu is not None:
            try:
                classic_menu.destroy()
            except tk.TclError:
                pass
        self.menu_bar = None
        self.classic_menu = None
        self.menubar = None

    def _selected_workflow_index(self) -> int:
        notebook = self.__dict__.get("notebook")
        try:
            return int(notebook.index(notebook.select()))
        except (AttributeError, tk.TclError, TypeError, ValueError):
            return 0

    def _sync_settings_interface_controls(self) -> None:
        dialog = self.__dict__.get("_settings_dialog")
        selected_interface = self.interface_mode
        if self.__dict__.get("_interface_switching", False):
            selected_interface = self.requested_interface_mode
        try:
            if dialog is not None and dialog.winfo_exists():
                dialog._sync_interface_controls(
                    selected_interface,
                    self.appearance_mode,
                )
                if (
                    not self.__dict__.get("_interface_switching", False)
                    and normalize_interface_mode(
                        getattr(dialog, "_presentation_mode", selected_interface)
                    )
                    != normalize_interface_mode(self.interface_mode)
                ):
                    dialog._schedule_presentation_refresh()
        except (AttributeError, tk.TclError):
            pass

    def _new_modern_menu_bar(self):
        """Construct the reusable Modern menu surface without packing it."""

        return ApplicationMenuBar(
            self,
            appearance_modes=APPEARANCE_MODES,
            current_appearance=self.appearance_mode,
            set_appearance_command=self._set_theme,
            interface_modes=INTERFACE_MODES,
            current_interface=self.interface_mode,
            set_interface_command=self._set_interface,
            browse_sdb_command=self._menu_browse_sdb,
            check_updates_command=self.update_controller.check_now,
            about_command=self._show_about,
            exit_command=self.destroy,
        )

    def _prime_modern_shell_cache(self) -> None:
        """Prepare inactive Modern chrome without changing the Classic shell."""

        if self.interface_mode != "Classic":
            return

        if self.__dict__.get("_modern_menu_bar_cache") is None:
            try:
                menu_bar = self._new_modern_menu_bar()
                menu_bar.suspend()
                self._modern_menu_bar_cache = menu_bar
            except (AttributeError, tk.TclError):
                self._modern_menu_bar_cache = None

        if self.__dict__.get("_modern_header_cache") is None:
            try:
                self._modern_header_cache = self._new_modern_workflow_header()
            except (AttributeError, tk.TclError):
                self._modern_header_cache = None

        if self.__dict__.get("_modern_status_bar_cache") is None:
            try:
                self._modern_status_bar_cache = AppStatusBar(
                    self,
                    text=self._current_status_text(),
                )
            except (AttributeError, tk.TclError):
                self._modern_status_bar_cache = None

    def _build_menu(self) -> None:
        if self.interface_mode == "Classic":
            menu_bar = self.__dict__.get("menu_bar")
            if menu_bar is not None:
                try:
                    menu_bar.suspend()
                    self._modern_menu_bar_cache = menu_bar
                except tk.TclError:
                    pass
                self.menu_bar = None
            classic_menu = self.__dict__.get("classic_menu")
            if classic_menu is None:
                self.classic_menu = build_classic_application_menu(
                    self,
                    interface_modes=INTERFACE_MODES,
                    current_interface=self.interface_mode,
                    set_interface_command=self._set_interface,
                    appearance_modes=APPEARANCE_MODES,
                    current_appearance=self.appearance_mode,
                    set_appearance_command=self._set_theme,
                    browse_sdb_command=self._menu_browse_sdb,
                    settings_command=self._show_settings,
                    check_updates_command=self.update_controller.check_now,
                    about_command=self._show_about,
                    exit_command=self.destroy,
                )
            else:
                classic_menu.set_interface(self.interface_mode)
                classic_menu.set_appearance(self.appearance_mode)
            self.menubar = self.classic_menu.menubar
            return

        classic_menu = self.__dict__.get("classic_menu")
        if classic_menu is not None:
            try:
                classic_menu.destroy()
            except tk.TclError:
                pass
            self.classic_menu = None
        menu_bar = self.__dict__.get("menu_bar")
        if menu_bar is not None:
            menu_bar.set_appearance(self.appearance_mode)
            menu_bar.set_interface(self.interface_mode)
            self.menubar = menu_bar
            return

        cached_menu = self.__dict__.get("_modern_menu_bar_cache")
        try:
            if cached_menu is not None and cached_menu.winfo_exists():
                self.menu_bar = cached_menu
                cached_menu.resume()
                cached_menu.set_appearance(self.appearance_mode)
                cached_menu.set_interface(self.interface_mode)
                self._pack_shell_surface(cached_menu, side="top")
                self.menubar = cached_menu
                return
        except (AttributeError, tk.TclError):
            try:
                if cached_menu is not None:
                    cached_menu.destroy()
            except (AttributeError, tk.TclError):
                pass
            self._modern_menu_bar_cache = None

        self.menu_bar = self._new_modern_menu_bar()
        self._pack_shell_surface(self.menu_bar, side="top")
        self._modern_menu_bar_cache = self.menu_bar
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
        if (
            self.interface_mode == "Modern"
            and self.appearance_mode == "System"
            and current != self._last_effective_appearance
        ):
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

    def _set_interface(self, mode_name: str) -> str:
        """Switch presentation shells immediately around retained workflows."""

        selected = normalize_interface_mode(mode_name)
        previous = normalize_interface_mode(
            self.__dict__.get("interface_mode", "Modern")
        )
        if self.__dict__.get("_interface_switching", False):
            requested = normalize_interface_mode(
                self.__dict__.get("requested_interface_mode", previous)
            )
            menu = self.__dict__.get("menu_bar") or self.__dict__.get("classic_menu")
            if menu is not None:
                menu.set_interface(requested)
            self._sync_settings_interface_controls()
            return requested
        if selected == previous:
            self.requested_interface_mode = previous
            self._build_menu()
            self._sync_settings_interface_controls()
            return previous

        selected_index = self._selected_workflow_index()
        status_text = self._current_status_text()
        logical_client_size = None
        window_state = "normal"
        try:
            self.update_idletasks()
            logical_client_size = logical_window_size(
                self,
                self.winfo_width(),
                self.winfo_height(),
            )
            window_state = str(self.state()).lower()
        except (AttributeError, tk.TclError, TypeError, ValueError):
            pass
        was_zoomed = window_state == "zoomed"
        current_title_bar = self.__dict__.get("window_title_bar")
        if current_title_bar is not None:
            try:
                was_zoomed = was_zoomed or current_title_bar.controller.is_maximized()
            except (AttributeError, tk.TclError):
                pass
        if not was_zoomed and logical_client_size is not None:
            self._normal_logical_client_size = logical_client_size
        normal_client_size = self.__dict__.get("_normal_logical_client_size")

        self._interface_switching = True
        self.requested_interface_mode = selected
        try:
            # A captionless window must never lose its in-client controls.
            # Confirm the native frame first and leave the active shell intact
            # when Windows cannot complete that transition.
            if selected == "Classic" and not self._restore_native_title_bar():
                self.requested_interface_mode = previous
                menu = self.__dict__.get("menu_bar")
                if menu is not None:
                    menu.set_interface(previous)
                self._set_status_message(
                    "the native window frame could not be restored; "
                    "Modern remains active"
                )
                return previous

            self._destroy_application_menus()
            self._destroy_workflow_header()
            self._destroy_status_surface()

            self.interface_mode = set_interface_mode(selected, redraw=False)
            active_appearance = (
                "Light" if self.interface_mode == "Classic" else self.appearance_mode
            )
            apply_appearance_mode(
                active_appearance,
                root=self,
                style=self.style,
                force_ctk_redraw=True,
                defer_ctk_ms=25,
            )
            self._last_effective_appearance = ctk.get_appearance_mode()

            if self.interface_mode == "Modern":
                self._install_modern_title_bar()
            self._build_menu()
            self._build_workflow_header()
            self._build_status_surface(status_text)

            notebook = self.__dict__.get("notebook")
            if notebook is not None:
                try:
                    notebook.select(selected_index)
                except (tk.TclError, TypeError, ValueError):
                    pass
            if self.header is not None:
                self.header.select_step(selected_index)

            self._sync_settings_interface_controls()
            step3 = self.__dict__.get("step3")
            if step3 is not None:
                self.after(50, step3.refresh_appearance)

            try:
                self.update_idletasks()
                if was_zoomed:
                    # Keep the window continuously zoomed.  Changing frame
                    # families changes the outer rectangle needed for the
                    # saved normal client size, so correct it lazily when the
                    # user eventually restores the window.
                    if normal_client_size is not None:
                        self._pending_normal_client_size = normal_client_size
                    title_bar = self.__dict__.get("window_title_bar")
                    if title_bar is not None:
                        self.after_idle(
                            title_bar.controller.correct_maximized_bounds
                        )
                elif logical_client_size is not None:
                    reassert_client_size(self, *logical_client_size)
            except (AttributeError, tk.TclError, TypeError, ValueError):
                pass

            synchronize_window_chrome(
                self,
                background=COLOR_PAIRS["window_chrome"],
                foreground=COLOR_PAIRS["text"],
                border=COLOR_PAIRS["window_chrome"],
            )
            # Commit the preference only after the replacement shell exists.
            self.preferences.set("interface_mode", self.interface_mode)
            self.requested_interface_mode = self.interface_mode
            self._set_status_message(f"{self.interface_mode} interface is active")
            # Queue the expensive retained-widget corner pass only after all
            # synchronous frame/geometry work.  It will paint on the next Tk
            # loop turn instead of delaying the menu command itself.
            self._queue_interface_widget_refresh()
            return self.interface_mode
        except Exception as exc:
            # Shell replacement is transactional: reconstruct the prior shell
            # and presentation while retaining the same notebook/workflows.
            try:
                self._destroy_application_menus()
                self._destroy_workflow_header()
                self._destroy_status_surface()
                if previous == "Classic":
                    self._restore_native_title_bar()
                self.interface_mode = set_interface_mode(previous, redraw=False)
                self.requested_interface_mode = previous
                prior_appearance = (
                    "Light" if previous == "Classic" else self.appearance_mode
                )
                apply_appearance_mode(
                    prior_appearance,
                    root=self,
                    style=self.style,
                    force_ctk_redraw=True,
                    defer_ctk_ms=25,
                )
                if previous == "Modern":
                    self._install_modern_title_bar()
                self._build_menu()
                self._build_workflow_header()
                self._build_status_surface(status_text)
                notebook = self.__dict__.get("notebook")
                if notebook is not None:
                    notebook.select(selected_index)
                if self.header is not None:
                    self.header.select_step(selected_index)
                self._queue_interface_widget_refresh()
            except Exception:
                self.interface_mode = previous
                self.requested_interface_mode = previous
            self._set_status_message(
                f"interface change was not applied ({type(exc).__name__}); "
                f"{previous} remains active"
            )
            return previous
        finally:
            self._interface_switching = False
            self._sync_settings_interface_controls()

    def _set_theme(self, theme_name: str) -> None:
        """Apply Modern appearance, or retain it while Classic stays light."""

        self.appearance_mode = normalize_appearance_mode(theme_name)
        active_appearance = (
            "Light" if self.interface_mode == "Classic" else self.appearance_mode
        )
        apply_appearance_mode(
            active_appearance,
            root=self,
            style=self.style,
        )
        self.preferences.set("appearance_mode", self.appearance_mode)
        self._last_effective_appearance = ctk.get_appearance_mode()
        self._build_menu()
        if self.interface_mode == "Classic":
            status_text = (
                f"AIDaS v{__version__} — Modern appearance saved as "
                f"{self.appearance_mode}"
            )
        else:
            status_text = (
                f"AIDaS v{__version__} — appearance changed to {self.appearance_mode}"
            )
        self.status.configure(text=status_text)
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

    def _is_step3_folder_active(self, folder) -> bool:
        """Return whether a live R batch is using a prospective Step 2 output folder."""
        step3 = getattr(self, "step3", None)
        return bool(step3 is not None and step3.is_folder_active(folder))

    def _step3_core_usage(self) -> int:
        step3 = getattr(self, "step3", None)
        if step3 is None:
            return 0
        return int(step3.active_core_allocation())

    def _step2_core_usage(self) -> int:
        step2 = getattr(self, "step2", None)
        if step2 is None:
            return 0
        return int(step2.active_core_allocation())

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
        self._about_dialog = AboutDialog(
            self,
            interface_mode=self.interface_mode,
        )

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
            interface_mode=self.interface_mode,
            set_interface_command=self._set_interface,
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
