"""Category-based Settings layout with panels constructed on first use."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

from aidas.ui.components import AppButton, AppOptionMenu
from aidas.ui.scrolling import SettingsScrollableFrame
from aidas.ui.theme import (
    APPEARANCE_MODES, COLOR_PAIRS, COLORS, INTERFACE_MODES, SHAPES, TYPOGRAPHY,
    refresh_interface_widgets,
)
from aidas.utils.ui_utils import HoverToolTip


class SettingsLayoutMixin:
    """Office-style navigation shared by the Modern and Classic interfaces."""

    CATEGORIES = (
        ("general", "General"),
        ("step1", "Step 1: SDB Parameters"),
        ("step2", "Step 2: Segmentation"),
        ("step3", "Step 3: R Scripts"),
        ("step4", "Step 4: ROI Analysis"),
    )
    CATEGORY_DESCRIPTIONS = {
        "general": "Personalize the appearance and behavior of AIDaS.",
        "step1": "Set the default image dimensions and bright-pixel behavior.",
        "step2": "Configure boundary editing and the annotation tutorial.",
        "step3": "Choose the R scripts used for processing and output.",
        "step4": "Configure automatic ROI detection and review criteria.",
    }

    def _build_settings_shell(self):
        self._category_pages = {}
        self._selected_category = None
        self._category_render_after_id = None
        self._settings_ready = False
        self.settings_panel = self._surface_frame(self)
        self.settings_panel.pack(fill="both", expand=True)
        self.settings_panel.grid_rowconfigure(0, weight=1)
        self.settings_panel.grid_columnconfigure(2, weight=1)

        navigation = self._surface_frame(self.settings_panel, "surface_subtle")
        navigation.grid(row=0, column=0, sticky="nsew")
        navigation.grid_rowconfigure(1, weight=1)
        self._page_label(navigation, "Options", heading=True).grid(
            row=0, column=0, sticky="w", padx=26, pady=(22, 14),
        )
        self.category_navigation = ttk.Treeview(
            navigation, show="tree", selectmode="browse", height=len(self.CATEGORIES),
            style="AIDaS.SettingsNav.Treeview", takefocus=True,
        )
        self.category_navigation.column("#0", width=194, stretch=False)
        self.category_navigation.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.category_navigation._apply_aidas_theme = self._style_category_navigation
        self._style_category_navigation()
        for key, label in self.CATEGORIES:
            self.category_navigation.insert("", "end", iid=key, text=label)
        self.category_navigation.bind("<<TreeviewSelect>>", self._on_category_selected)
        self.category_navigation.bind("<Return>", self._focus_category_content)
        ttk.Separator(self.settings_panel, orient="vertical").grid(
            row=0, column=1, sticky="ns",
        )
        self.category_content = self._surface_frame(self.settings_panel)
        self.category_content.grid(row=0, column=2, sticky="nsew")

        self.footer = self._surface_frame(self, "surface_subtle")
        self.footer.pack(fill="x")
        ttk.Separator(self.footer, orient="horizontal").pack(fill="x")
        actions = self._surface_frame(self.footer, "surface_subtle")
        actions.pack(fill="x", padx=18, pady=12)
        actions.grid_columnconfigure(0, weight=1)
        self._page_label(
            actions, textvariable=self.apply_status_var, muted=True, wraplength=330,
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.apply_button = AppButton(
            actions, text="Apply", command=self._apply_changes, width=88,
            bg_color=COLOR_PAIRS["surface_subtle"],
        )
        self.apply_button.grid(row=0, column=3)
        self.close_button = AppButton(
            actions, text="Cancel", command=self._close, width=88,
            bg_color=COLOR_PAIRS["surface_subtle"],
        )
        self.close_button.grid(row=0, column=2, padx=8)
        self.ok_button = AppButton(
            actions, text="OK", variant="primary", command=self._accept_changes, width=88,
            bg_color=COLOR_PAIRS["surface_subtle"],
        )
        self.ok_button.grid(row=0, column=1)
        self._select_category("general")

    def _style_category_navigation(self):
        style = ttk.Style(self)
        font = (TYPOGRAPHY.family, 10)
        scale = self._get_window_scaling()
        inset = round(16 * scale)
        label_width = max(int(self.tk.call("font", "measure", font, label)) for _key, label in self.CATEGORIES)
        self.category_navigation.column("#0", width=max(round(204 * scale), label_width + 2 * inset))
        row_height = max(round(46 * scale), int(self.tk.call("font", "metrics", font, "-linespace")) + 20)
        element = self._rounded_navigation_element(style, scale)
        style.configure(
            "AIDaS.SettingsNav.Treeview", background=COLORS.surface_subtle,
            fieldbackground=COLORS.surface_subtle, foreground=COLORS.text,
            borderwidth=0, relief="flat", rowheight=row_height, font=font,
        )
        style.map(
            "AIDaS.SettingsNav.Treeview",
            background=[("selected", COLORS.surface_subtle)],
            foreground=[("selected", COLORS.primary)],
        )
        style.layout("AIDaS.SettingsNav.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.layout("AIDaS.SettingsNav.Treeview.Item", [
            (element, {"sticky": "nswe", "children": [
                ("Treeitem.text", {"sticky": "w"}),
            ]}),
        ])
        style.configure("AIDaS.Settings.TLabel", background=COLORS.surface, foreground=COLORS.text)
        style.configure("AIDaS.Settings.TCheckbutton", background=COLORS.surface)

    def _rounded_navigation_element(self, style, scale):
        """Keep small rounded row images alive across dialog and theme changes."""
        root = self._root()
        cache = root.__dict__.setdefault("_aidas_settings_tab_images", {})
        size = max(32, round(40 * scale))
        key = (style.theme_use(), size)
        record = cache.get(key)
        palette = (
            COLORS.surface_subtle, COLORS.button, COLORS.border,
            COLORS.primary_soft, COLORS.primary,
        )
        if record is None or record["palette"] != palette:
            supersampling = 4
            gap = max(3, round(4 * scale))
            radius = max(6, round(8 * scale))
            sources = []
            for fill, border, stroke in (
                (COLORS.button, COLORS.border, 1),
                (COLORS.primary_soft, COLORS.primary, 1),
                (COLORS.primary_soft, COLORS.primary, 2),
            ):
                source = Image.new("RGB", (size * supersampling,) * 2, COLORS.surface_subtle)
                draw = ImageDraw.Draw(source)
                draw.rounded_rectangle(
                    (0, gap * supersampling, size * supersampling - 1,
                     (size - gap) * supersampling - 1),
                    radius=radius * supersampling, fill=fill, outline=border,
                    width=max(1, round(stroke * scale)) * supersampling,
                )
                sources.append(source.resize((size, size), Image.Resampling.LANCZOS))
            if record is None:
                images = tuple(ImageTk.PhotoImage(source, master=root) for source in sources)
                element = f"AIDaS.SettingsTab{len(cache)}.background"
                style.element_create(
                    element, "image", images[0],
                    ("selected", "focus", images[2]), ("selected", images[1]),
                    border=(radius + 1, radius + gap + 1),
                    padding=(round(16 * scale), round(8 * scale)), sticky="nswe",
                )
                record = {"element": element, "images": images}
                cache[key] = record
            else:
                for image, source in zip(record["images"], sources):
                    image.paste(source)
            record["palette"] = palette
        return record["element"]

    def _on_category_selected(self, _event=None):
        selected = self.category_navigation.selection()
        if selected:
            self._select_category(selected[0])

    def _select_category(self, key):
        if key == self._selected_category:
            return
        if key not in dict(self.CATEGORIES):
            raise ValueError(f"Unknown settings category: {key}")
        if self._selected_category is not None:
            self._category_pages[self._selected_category].pack_forget()
        if key not in self._category_pages:
            self._aidas_defer_canvas_updates = True
            page = SettingsScrollableFrame(
                self.category_content, fg_color=COLOR_PAIRS["surface"],
                bg_color=COLOR_PAIRS["surface"],
                corner_radius=0, border_width=0,
                scrollbar_button_color=COLOR_PAIRS["border_strong"],
                scrollbar_button_hover_color=COLOR_PAIRS["primary"],
            )
            page.grid_columnconfigure(0, weight=1)
            self._category_pages[key] = page
            self._page_label(page, dict(self.CATEGORIES)[key], heading=True).grid(
                row=0, column=0, sticky="w", padx=24, pady=(24, 6),
            )
            self._page_label(
                page, self.CATEGORY_DESCRIPTIONS[key], muted=True, wraplength=580,
            ).grid(row=1, column=0, sticky="ew", padx=24, pady=(0, 18))
            getattr(self, f"_build_{key}_page")(page)
            refresh_interface_widgets(page)
        self._category_pages[key].pack(fill="both", expand=True)
        self._selected_category = key
        self.category_navigation.selection_set(key)
        self.category_navigation.focus(key)
        self.category_navigation.see(key)
        if (self._settings_ready and self._aidas_defer_canvas_updates
                and self._category_render_after_id is None):
            self._category_render_after_id = self.after_idle(self._finish_category_rendering)

    def _finish_category_rendering(self):
        self._category_render_after_id = None
        from aidas.ui.rendering import finish_deferred_rendering
        finish_deferred_rendering(self)

    def _focus_category_content(self, _event=None):
        page = self._category_pages[self._selected_category]
        focus = page.tk_focusNext()
        if focus is not None:
            focus.focus_set()
        return "break"

    @staticmethod
    def _surface_frame(parent, token="surface"):
        frame = tk.Frame(parent, background=getattr(COLORS, token), borderwidth=0)
        frame._apply_aidas_theme = lambda: frame.configure(background=getattr(COLORS, token))
        return frame

    def _page_label(self, parent, text="", *, heading=False, muted=False, **options):
        token = "muted_text" if muted else "text"
        label = tk.Label(
            parent, text=text, anchor="w", justify="left", borderwidth=0,
            font=(TYPOGRAPHY.family, 14 if heading else 10, "bold" if heading else "normal"),
            background=tk.Misc.cget(parent, "background"), foreground=getattr(COLORS, token), **options,
        )

        def apply_theme():
            label.configure(background=tk.Misc.cget(parent, "background"), foreground=getattr(COLORS, token))

        label._apply_aidas_theme = apply_theme
        return label

    def _page_section(self, page, title, *, row):
        section = self._surface_frame(page)
        section.grid(row=row, column=0, sticky="ew", padx=24, pady=(0, 20))
        section.grid_columnconfigure(0, weight=1)
        self._page_label(section, title).grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Separator(section, orient="horizontal").grid(row=1, column=0, sticky="ew")
        body = self._surface_frame(section)
        body.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        body.grid_columnconfigure(0, weight=1)
        return body

    def _page_field(self, parent, row, label, control, *, help_text=None):
        holder = self._surface_frame(parent)
        holder.grid(row=row, column=0, sticky="w", padx=(0, 16), pady=7)
        self._page_label(holder, label).pack(side="left")
        if help_text:
            help_label = self._page_label(holder, "  \u24d8", muted=True, cursor="hand2")
            help_label.pack(side="left")
            HoverToolTip(help_label, help_text)
            HoverToolTip(control, help_text)
        control.grid(row=row, column=1, sticky="e", pady=7)

    def _page_entry(self, parent, variable):
        if self._classic_settings:
            return ttk.Entry(parent, textvariable=variable, width=18, justify="right")
        return ctk.CTkEntry(
            parent, textvariable=variable, width=140, height=32, justify="right",
            bg_color=COLOR_PAIRS["surface"],
            fg_color=COLOR_PAIRS["surface_elevated"], border_color=COLOR_PAIRS["border_strong"],
            text_color=COLOR_PAIRS["text"],
        )

    def _page_check(self, parent, text, variable, command):
        if self._classic_settings:
            return ttk.Checkbutton(
                parent, text=text, variable=variable, command=command,
                style="AIDaS.Settings.TCheckbutton",
            )
        return ctk.CTkSwitch(
            parent, text=text, variable=variable, command=command,
            bg_color=COLOR_PAIRS["surface"],
            progress_color=COLOR_PAIRS["primary"], text_color=COLOR_PAIRS["text"],
            font=(TYPOGRAPHY.family, 12),
        )

    def _page_menu(self, parent, values, *, variable=None, command=None):
        if self._classic_settings:
            menu = ttk.Combobox(parent, values=values, textvariable=variable, state="readonly", width=18)
            if command is not None:
                menu.bind("<<ComboboxSelected>>", lambda _event: command(menu.get()))
            return menu
        return AppOptionMenu(
            parent, values=list(values), variable=variable, command=command,
            bg_color=COLOR_PAIRS["surface"],
            width=160, height=34, corner_radius=SHAPES.corner_radius_sm,
            fg_color=COLOR_PAIRS["button"], button_color=COLOR_PAIRS["primary"],
            button_hover_color=COLOR_PAIRS["primary_hover"], text_color=COLOR_PAIRS["text"],
            dropdown_fg_color=COLOR_PAIRS["surface_elevated"],
            dropdown_hover_color=COLOR_PAIRS["primary_soft"], dropdown_text_color=COLOR_PAIRS["text"],
        )

    def _build_general_page(self, page):
        appearance = self._page_section(page, "Appearance", row=2)
        self.interface_menu = self._page_menu(
            appearance, INTERFACE_MODES, command=self._interface_choice_changed,
        )
        self.interface_menu.set(self._presentation_mode)
        self._page_field(appearance, 0, "Interface", self.interface_menu)
        self.appearance_menu = self._page_menu(
            appearance, APPEARANCE_MODES, command=self._mark_dirty,
        )
        self.appearance_menu.set(self._initial_appearance_mode)
        self._page_field(appearance, 1, "Appearance", self.appearance_menu)
        self._sync_interface_controls(self._presentation_mode, self._initial_appearance_mode)
        updates = self._page_section(page, "Application Updates", row=3)
        self.update_switch = self._page_check(
            updates, "Check for updates whenever AIDaS opens", self.update_checks_var,
            self._mark_dirty,
        )
        self.update_switch.grid(row=0, column=0, columnspan=2, sticky="w", pady=6)

    def _build_step1_page(self, page):
        parameters = self._page_section(page, "Default SDB Image Parameters", row=2)
        for row, (key, label) in enumerate((
            ("sdb_raw_width", "Width (px)"), ("sdb_raw_height", "Height (px)"),
            ("sdb_raw_offset", "Offset (bytes)"),
        )):
            self._page_field(parameters, row, label, self._page_entry(parameters, self.sdb_default_vars[key]))
        self._page_check(
            parameters, "Little-endian byte order", self.sdb_little_endian_var, self._mark_dirty,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 4))
        correction = self._page_section(page, "Bright-Pixel Correction", row=3)
        self._build_bright_pixel_setting(correction, modern=not self._classic_settings, row=0)

    def _build_step2_page(self, page):
        editing = self._page_section(page, "Boundary Editing", row=2)
        self._build_manual_segmentation_setting(editing, modern=not self._classic_settings, row=0)
        help_section = self._page_section(page, "Annotation Tutorial", row=3)
        self.annotation_tutorial_switch = self._page_check(
            help_section, "Enable tutorial", self.annotation_tutorial_var,
            self._annotation_tutorial_choice_changed,
        )
        self.annotation_tutorial_switch.grid(row=0, column=0, columnspan=2, sticky="w", pady=6)
        self._page_label(
            help_section, "Show guidance when manual boundary editing is enabled.",
            muted=True, wraplength=580,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def _build_step3_page(self, page):
        scripts = self._page_section(page, "R Script Selection", row=2)
        self.script_menus = {}
        for row, (role, label) in enumerate(self.SCRIPT_ROLES):
            role_frame = self._surface_frame(scripts)
            role_frame.grid(row=row, column=0, sticky="ew", pady=(0, 24))
            role_frame.grid_columnconfigure(0, weight=1)
            self._page_label(role_frame, label).grid(row=0, column=0, sticky="w", pady=(0, 8))
            if self._classic_settings:
                menu = ttk.Menubutton(role_frame, textvariable=self._script_vars[role], width=32)
                popup = tk.Menu(menu, tearoff=False)
                menu.configure(menu=popup)
                popup.configure(postcommand=lambda selected_role=role: self._update_r_script_popup(selected_role))
            else:
                menu = self._page_menu(
                    role_frame, self._r_script_options(role), variable=self._script_vars[role],
                    command=lambda label, selected_role=role: self._choose_r_script_option(selected_role, label),
                )
                popup = menu._dropdown_menu
                open_dropdown = menu._open_dropdown_menu

                def open_script_choices(selected_role=role, open_dropdown=open_dropdown):
                    self._update_r_script_popup(selected_role)
                    open_dropdown()

                menu._open_dropdown_menu = open_script_choices
            menu.grid(row=1, column=0, sticky="w")
            self.script_menus[role] = menu
            self._script_popups[role] = popup
            self._page_label(
                role_frame, textvariable=self._script_status_vars[role], muted=True, wraplength=580,
            ).grid(row=2, column=0, sticky="ew", pady=(8, 0))
            self._refresh_r_script_choices(role)
        self._scripts_initialized = True

    def _build_step4_page(self, page):
        detection = self._page_section(page, "Automatic ROI Detection", row=2)
        for row, (key, label, help_text) in enumerate(self.AUTO_DETECTION_FIELDS):
            self._page_field(
                detection, row, label, self._page_entry(detection, self.auto_detection_vars[key]),
                help_text=help_text,
            )
