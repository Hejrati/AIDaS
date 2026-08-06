"""Central design tokens and appearance management for the AIDaS UI.

CustomTkinter owns the modern application chrome.  Native Tk and ttk remain
available for scientific canvases, data grids, and menus that have no direct
CustomTkinter equivalent; this module keeps both widget families on the same
semantic palette.
"""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from types import MappingProxyType
from typing import Mapping, TypeAlias

import customtkinter as ctk

from aidas.ui.windowing import synchronize_window_chrome


ColorPair: TypeAlias = tuple[str, str]
APPEARANCE_MODES = ("System", "Light", "Dark")


COLOR_PAIRS: Mapping[str, ColorPair] = MappingProxyType(
    {
        "application": ("#EEF3F8", "#0B1118"),
        # Keep the native caption and in-app menu visually related without
        # merging them into one flat black/white strip.
        "window_chrome": ("#D4DFE9", "#111B25"),
        "menu_bar": ("#EAF0F5", "#1A2733"),
        "surface": ("#FFFFFF", "#141D27"),
        "surface_subtle": ("#F5F8FB", "#1A2530"),
        "surface_elevated": ("#FFFFFF", "#202D39"),
        # Neutral controls need their own fill. Reusing surface_elevated made
        # buttons white-on-white in light mode, hiding their clickable area.
        "button": ("#E5EDF5", "#202D39"),
        "button_hover": ("#D7E7F2", "#17384D"),
        "sidebar": ("#F8FAFC", "#111923"),
        "border": ("#D7E0E9", "#293847"),
        "border_strong": ("#B8C5D1", "#3B4C5C"),
        "text": ("#152235", "#F2F6FA"),
        "muted_text": ("#637487", "#9FB0C0"),
        "disabled_text": ("#98A5B2", "#667787"),
        "primary": ("#0B67A3", "#3294D1"),
        "primary_hover": ("#085584", "#55A9DD"),
        "primary_soft": ("#DDEFFA", "#17384D"),
        "on_primary": ("#FFFFFF", "#07131C"),
        "success": ("#157A43", "#4FD18B"),
        "success_hover": ("#106534", "#72DFA2"),
        "success_soft": ("#E3F5EA", "#173B2A"),
        "warning": ("#A35A00", "#F2B84B"),
        "warning_soft": ("#FFF1D6", "#493716"),
        "danger": ("#B4232E", "#FF7581"),
        "danger_hover": ("#8F1D26", "#FF98A1"),
        "danger_soft": ("#FCE7E9", "#49242A"),
        "link": ("#0969A8", "#65B8EC"),
        "institution": ("#C0002B", "#FF6F82"),
        "selection": ("#CBE8F8", "#244B63"),
        "canvas": ("#101820", "#080D12"),
        "accent": ("#0B67A3", "#3294D1"),
        "accent_hover": ("#085584", "#55A9DD"),
        "accent_soft": ("#DDEFFA", "#17384D"),
    }
)


@dataclass(frozen=True)
class TypographyTokens:
    family: str = "Segoe UI"
    mono_family: str = "Cascadia Mono"
    caption_size: int = 11
    body_size: int = 12
    subtitle_size: int = 14
    heading_size: int = 16
    title_size: int = 20
    display_size: int = 32
    normal_weight: str = "normal"
    semibold_weight: str = "bold"
    bold_weight: str = "bold"


@dataclass(frozen=True)
class ShapeTokens:
    corner_radius_sm: int = 6
    corner_radius_md: int = 10
    corner_radius_lg: int = 14
    border_width: int = 1
    focus_width: int = 2


@dataclass(frozen=True)
class ControlTokens:
    height_sm: int = 28
    height_md: int = 34
    height_lg: int = 40
    padding_x: int = 12
    padding_y: int = 8
    gap: int = 8
    icon_size: int = 18
    tree_row_height: int = 30
    scrollbar_width: int = 12


@dataclass(frozen=True)
class ThemeTokens:
    colors: Mapping[str, ColorPair]
    typography: TypographyTokens
    shapes: ShapeTokens
    controls: ControlTokens


TYPOGRAPHY = TypographyTokens()
SHAPES = ShapeTokens()
CONTROLS = ControlTokens()
THEME = ThemeTokens(COLOR_PAIRS, TYPOGRAPHY, SHAPES, CONTROLS)


def normalize_appearance_mode(value: object) -> str:
    """Return a supported CTk mode, safely mapping legacy ttk theme names."""

    text = str(value or "").strip().lower()
    return {"system": "System", "light": "Light", "dark": "Dark"}.get(text, "System")


def resolve_color(value: str | ColorPair, appearance_mode: str | None = None) -> str:
    """Resolve a CTk dual color to the currently displayed native-widget color."""

    if isinstance(value, str):
        return value
    if len(value) != 2:
        raise ValueError("A CustomTkinter color pair must contain light and dark colors")
    mode = normalize_appearance_mode(appearance_mode) if appearance_mode else ctk.get_appearance_mode()
    if mode == "System":
        # AppearanceModeTracker has already resolved the operating-system mode.
        mode = ctk.get_appearance_mode()
    return value[1] if str(mode).lower() == "dark" else value[0]


class _ResolvedColorTokens:
    """Backward-compatible dynamic color access for existing Tk/ttk modules."""

    def __getattr__(self, name: str) -> str:
        try:
            return resolve_color(COLOR_PAIRS[name])
        except KeyError as exc:
            raise AttributeError(name) from exc

    def pair(self, name: str) -> ColorPair:
        """Return the CTk-ready light/dark value for a semantic color name."""

        try:
            return COLOR_PAIRS[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


COLORS = _ResolvedColorTokens()


def configure_ttk_styles(style: ttk.Style, appearance_mode: str | None = None) -> None:
    """Synchronize retained ttk widgets with the active CustomTkinter palette."""

    mode = normalize_appearance_mode(appearance_mode) if appearance_mode else ctk.get_appearance_mode()
    color = lambda name: resolve_color(COLOR_PAIRS[name], mode)
    available = style.theme_names()
    if "clam" in available and style.theme_use() != "clam":
        style.theme_use("clam")

    body_font = (TYPOGRAPHY.family, 9)
    strong_font = (TYPOGRAPHY.family, 9, "bold")
    style.configure(
        ".",
        font=body_font,
        background=color("surface"),
        foreground=color("text"),
        bordercolor=color("border"),
        lightcolor=color("border"),
        darkcolor=color("border"),
        troughcolor=color("surface_subtle"),
    )
    style.configure("TFrame", background=color("surface"), borderwidth=0)
    style.configure("TLabel", background=color("surface"), foreground=color("text"))
    style.map(
        "TLabel",
        # clam supplies a light disabled-label background internally. Override
        # only that state so disabled status text remains part of the active
        # AIDaS surface without masking specialized enabled backgrounds.
        background=[("disabled", color("surface"))],
        foreground=[("disabled", color("disabled_text"))],
    )
    style.configure("TCheckbutton", background=color("surface"), foreground=color("text"), padding=(2, 3))
    style.configure("TRadiobutton", background=color("surface"), foreground=color("text"), padding=(2, 3))
    for widget_style in ("TCheckbutton", "TRadiobutton"):
        style.map(
            widget_style,
            background=[("active", color("surface"))],
            foreground=[("disabled", color("disabled_text")), ("!disabled", color("text"))],
            indicatorcolor=[("selected", color("primary")), ("!selected", color("surface_elevated"))],
        )

    # Radio controls placed in a subtle toolbar should visually inherit that
    # surface instead of drawing a white rectangle around their text.
    style.configure(
        "AIDaS.ContentHeader.TRadiobutton",
        background=color("surface_subtle"),
        foreground=color("text"),
        padding=(2, 3),
    )
    style.map(
        "AIDaS.ContentHeader.TRadiobutton",
        background=[("active", color("surface_subtle"))],
        foreground=[("disabled", color("disabled_text")), ("!disabled", color("text"))],
        indicatorcolor=[("selected", color("primary")), ("!selected", color("surface_elevated"))],
    )

    style.configure(
        "TLabelframe",
        background=color("surface"),
        bordercolor=color("border"),
        borderwidth=1,
        relief="solid",
    )
    style.configure(
        "TLabelframe.Label",
        background=color("surface"),
        foreground=color("text"),
        font=strong_font,
        padding=(3, 1),
    )
    style.configure(
        "TButton",
        background=color("button"),
        foreground=color("text"),
        bordercolor=color("border_strong"),
        focusthickness=1,
        focuscolor=color("primary"),
        padding=(11, 7),
        relief="flat",
    )
    style.map(
        "TButton",
        background=[
            ("disabled", color("surface_subtle")),
            ("pressed", color("button_hover")),
            ("active", color("button_hover")),
        ],
        foreground=[("disabled", color("disabled_text")), ("!disabled", color("text"))],
        bordercolor=[("focus", color("primary")), ("!focus", color("border_strong"))],
    )
    # Icon actions retain the standard 34 px control height. Text-and-icon
    # buttons need less vertical padding than text-only ttk buttons because
    # their 20 px glyph determines the content height. Icon-only controls use
    # narrower symmetric padding so their requested size is a true square.
    for style_name, padding in (
        ("AIDaS.Action.TButton", (11, 4)),
        ("AIDaS.Icon.TButton", (2, 4)),
    ):
        style.configure(
            style_name,
            background=color("button"),
            foreground=color("text"),
            bordercolor=color("border_strong"),
            focusthickness=1,
            focuscolor=color("primary"),
            padding=padding,
            relief="flat",
        )
        style.map(
            style_name,
            background=[
                ("disabled", color("surface_subtle")),
                ("pressed", color("button_hover")),
                ("active", color("button_hover")),
            ],
            foreground=[("disabled", color("disabled_text")), ("!disabled", color("text"))],
            bordercolor=[("focus", color("primary")), ("!focus", color("border_strong"))],
        )
    for style_name, base_color, hover_color in (
        ("AIDaS.PrimaryAction.TButton", "primary", "primary_hover"),
        ("AIDaS.DangerAction.TButton", "danger", "danger_hover"),
    ):
        style.configure(
            style_name,
            background=color(base_color),
            foreground=color("on_primary"),
            bordercolor=color(base_color),
            focusthickness=1,
            focuscolor=color(base_color),
            padding=(11, 4),
            relief="flat",
        )
        style.map(
            style_name,
            background=[
                ("disabled", color("surface_subtle")),
                ("pressed", color(hover_color)),
                ("active", color(hover_color)),
            ],
            foreground=[("disabled", color("disabled_text")), ("!disabled", color("on_primary"))],
            bordercolor=[("disabled", color("border")), ("!disabled", color(base_color))],
        )
    for entry_style in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(
            entry_style,
            fieldbackground=color("surface_elevated"),
            background=color("surface_elevated"),
            foreground=color("text"),
            insertcolor=color("text"),
            bordercolor=color("border_strong"),
            arrowcolor=color("muted_text"),
            padding=(7, 5),
        )
        style.map(
            entry_style,
            fieldbackground=[("disabled", color("surface_subtle")), ("readonly", color("surface_subtle"))],
            foreground=[("disabled", color("disabled_text")), ("!disabled", color("text"))],
            bordercolor=[("focus", color("primary")), ("!focus", color("border_strong"))],
        )

    style.configure(
        "Treeview",
        rowheight=CONTROLS.tree_row_height,
        background=color("surface"),
        fieldbackground=color("surface"),
        foreground=color("text"),
        bordercolor=color("border"),
        relief="flat",
    )
    style.map(
        "Treeview",
        background=[("selected", color("primary"))],
        foreground=[("selected", color("on_primary"))],
    )
    style.configure(
        "Treeview.Heading",
        background=color("surface_subtle"),
        foreground=color("text"),
        font=strong_font,
        padding=(8, 7),
        relief="flat",
    )
    style.map("Treeview.Heading", background=[("active", color("primary_soft"))])
    style.configure(
        "TScrollbar",
        background=color("border_strong"),
        troughcolor=color("surface_subtle"),
        bordercolor=color("surface_subtle"),
        arrowcolor=color("muted_text"),
        relief="flat",
        width=CONTROLS.scrollbar_width,
    )
    style.map("TScrollbar", background=[("active", color("primary"))])
    style.configure(
        "TProgressbar",
        background=color("primary"),
        troughcolor=color("surface_subtle"),
        bordercolor=color("surface_subtle"),
        lightcolor=color("primary"),
        darkcolor=color("primary"),
    )
    style.configure("TSeparator", background=color("border"))

    style.configure("TNotebook", background=color("application"), borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure(
        "TNotebook.Tab",
        background=color("surface_subtle"),
        foreground=color("muted_text"),
        font=strong_font,
        padding=(12, 7),
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", color("surface")), ("active", color("primary_soft"))],
        foreground=[("selected", color("primary")), ("active", color("text"))],
    )

    style.configure("AIDaS.App.TFrame", background=color("application"))
    style.configure("AIDaS.Workspace.TFrame", background=color("application"))
    style.configure("AIDaS.Sidebar.TFrame", background=color("sidebar"))
    style.configure("AIDaS.Content.TFrame", background=color("surface"))
    style.configure("AIDaS.Section.TFrame", background=color("surface"))
    style.configure("AIDaS.ContentHeader.TFrame", background=color("surface_subtle"))
    style.configure(
        "AIDaS.ContentHeader.TLabel",
        background=color("surface_subtle"),
        foreground=color("text"),
        font=(TYPOGRAPHY.family, 10, "bold"),
    )
    style.map(
        "AIDaS.ContentHeader.TLabel",
        background=[("disabled", color("surface_subtle"))],
    )
    style.configure(
        "AIDaS.Status.TLabel",
        background=color("surface_subtle"),
        foreground=color("muted_text"),
        padding=(12, 6),
        relief="flat",
    )
    style.map(
        "AIDaS.Status.TLabel",
        background=[("disabled", color("surface_subtle"))],
    )
    style.configure("AIDaS.TNotebook", background=color("application"), borderwidth=0, tabmargins=(0, 0, 0, 0))
    # The app supplies its own CTk workflow navigation; the retained Notebook
    # remains the compatibility container used by all cross-step callbacks.
    try:
        style.layout("AIDaS.TNotebook.Tab", [])
    except tk.TclError:
        pass

    for style_name in ("Accent.TButton", "AIDaS.Primary.TButton"):
        style.configure(
            style_name,
            background=color("primary"),
            foreground=color("on_primary"),
            bordercolor=color("primary"),
            font=strong_font,
            padding=(12, 7),
        )
        style.map(
            style_name,
            background=[("pressed", color("primary_hover")), ("active", color("primary_hover"))],
            foreground=[("disabled", color("disabled_text")), ("!disabled", color("on_primary"))],
        )

    semantic_styles = {
        "AIDaS.Muted.TLabel": "muted_text",
        "AIDaS.Link.TLabel": "link",
        "AIDaS.Success.TLabel": "success",
        "AIDaS.Warning.TLabel": "warning",
        "AIDaS.Danger.TLabel": "danger",
    }
    for style_name, color_name in semantic_styles.items():
        style.configure(style_name, background=color("surface"), foreground=color(color_name))

    # Step 3's retained ttk setup wizard participates in live appearance
    # switching even when it was created before the mode changed.
    style.configure("WizardSubtitle.TLabel", foreground=color("muted_text"))
    style.configure("WizardStepDone.TLabel", foreground=color("success"))
    style.configure("WizardSuccess.TLabel", foreground=color("success"))
    style.configure("WizardMissing.TLabel", foreground=color("danger"))
    style.configure("WizardNeutral.TLabel", foreground=color("muted_text"))


def _walk_widgets(widget: tk.Misc):
    for child in widget.winfo_children():
        yield child
        yield from _walk_widgets(child)


def refresh_native_widgets(root: tk.Misc) -> None:
    """Refresh native widgets and shared composites after an appearance change."""

    surface = COLORS.surface
    subtle = COLORS.surface_subtle
    text = COLORS.text
    muted = COLORS.muted_text
    border = COLORS.border_strong
    selected = COLORS.primary
    on_selected = COLORS.on_primary

    for widget in (root, *_walk_widgets(root)):
        callback = getattr(widget, "_apply_aidas_theme", None)
        if callable(callback):
            try:
                callback()
            except tk.TclError:
                pass
        try:
            if isinstance(widget, tk.Listbox):
                widget.configure(
                    background=surface,
                    foreground=text,
                    selectbackground=selected,
                    selectforeground=on_selected,
                    highlightbackground=border,
                    highlightcolor=COLORS.primary,
                )
            elif isinstance(widget, tk.Text):
                widget.configure(
                    background=subtle,
                    foreground=text,
                    insertbackground=text,
                    selectbackground=selected,
                    selectforeground=on_selected,
                    highlightbackground=border,
                )
            elif isinstance(widget, tk.Menu):
                widget.configure(
                    background=surface,
                    foreground=text,
                    activebackground=COLORS.primary_soft,
                    activeforeground=text,
                    disabledforeground=muted,
                    borderwidth=0,
                )
            elif isinstance(widget, ttk.Treeview):
                widget.tag_configure("locked", foreground=muted)
                widget.tag_configure("installed", foreground=COLORS.success)
                widget.tag_configure("missing", foreground=COLORS.danger)
                widget.tag_configure("neutral", foreground=muted)
        except tk.TclError:
            pass


def apply_appearance_mode(
    mode: object,
    *,
    root: tk.Misc | None = None,
    style: ttk.Style | None = None,
) -> str:
    """Apply one appearance mode to CTk plus all retained native widgets."""

    normalized = normalize_appearance_mode(mode)
    ctk.set_appearance_mode(normalized)
    if style is not None:
        configure_ttk_styles(style, normalized)
    if root is not None:
        try:
            root.configure(fg_color=COLOR_PAIRS["application"])
        except (tk.TclError, ValueError, TypeError):
            try:
                root.configure(background=COLORS.application)
            except tk.TclError:
                pass
        refresh_native_widgets(root)
        titlebar_windows = [root]
        titlebar_windows.extend(
            widget
            for widget in _walk_widgets(root)
            if isinstance(widget, tk.Toplevel)
        )
        for window in titlebar_windows:
            synchronize_window_chrome(
                window,
                background=COLOR_PAIRS["window_chrome"],
                foreground=COLOR_PAIRS["text"],
                border=COLOR_PAIRS["window_chrome"],
            )
        try:
            root.event_generate("<<AIDaSThemeChanged>>", when="tail")
        except tk.TclError:
            pass
    return normalized


__all__ = [
    "APPEARANCE_MODES",
    "COLORS",
    "COLOR_PAIRS",
    "CONTROLS",
    "SHAPES",
    "THEME",
    "TYPOGRAPHY",
    "apply_appearance_mode",
    "configure_ttk_styles",
    "normalize_appearance_mode",
    "refresh_native_widgets",
    "resolve_color",
]
