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


class _LiveColorPair(list[str]):
    """A two-color CTk value whose identity survives interface switches.

    CustomTkinter keeps the color sequence supplied at construction time.  A
    normal dictionary replacement therefore leaves already-created widgets on
    the old palette.  These tiny mutable pairs let us update every semantic
    color in place while retaining tuple-friendly equality for callers and
    tests that treat theme values as ordinary color pairs.
    """

    def __init__(self, values) -> None:
        super().__init__(str(value) for value in values)
        if len(self) != 2:
            raise ValueError("A live color pair must contain two colors")

    def __eq__(self, other) -> bool:
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return super().__eq__(other)

    def __ne__(self, other) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result


ColorPair: TypeAlias = tuple[str, str] | list[str]
APPEARANCE_MODES = ("System", "Light", "Dark")
INTERFACE_MODES = ("Modern", "Classic")


MODERN_COLOR_PAIRS: Mapping[str, ColorPair] = MappingProxyType(
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

# The Classic palette is taken from the final v2 interface immediately before
# the v3 redesign.  Both members of each pair are intentionally the same:
# classic mode mirrors the original light, native-desktop presentation while
# the appearance preference remains available for a later return to Modern.
CLASSIC_COLOR_PAIRS: Mapping[str, ColorPair] = MappingProxyType(
    {
        "application": ("#E9EEF3", "#E9EEF3"),
        "window_chrome": ("#F0F0F0", "#F0F0F0"),
        "menu_bar": ("#F0F0F0", "#F0F0F0"),
        "surface": ("#FFFFFF", "#FFFFFF"),
        "surface_subtle": ("#F5F7FA", "#F5F7FA"),
        "surface_elevated": ("#FFFFFF", "#FFFFFF"),
        "button": ("#F0F0F0", "#F0F0F0"),
        "button_hover": ("#E5F1F9", "#E5F1F9"),
        "sidebar": ("#FFFFFF", "#FFFFFF"),
        "border": ("#C9D2DC", "#C9D2DC"),
        "border_strong": ("#A7ADB3", "#A7ADB3"),
        "text": ("#17212B", "#17212B"),
        "muted_text": ("#5D6B78", "#5D6B78"),
        "disabled_text": ("#8B9298", "#8B9298"),
        "primary": ("#0B5F9E", "#0B5F9E"),
        "primary_hover": ("#084B7D", "#084B7D"),
        "primary_soft": ("#E5F1F9", "#E5F1F9"),
        "on_primary": ("#FFFFFF", "#FFFFFF"),
        "success": ("#1F6B35", "#1F6B35"),
        "success_hover": ("#18562A", "#18562A"),
        "success_soft": ("#DFF3E4", "#DFF3E4"),
        "warning": ("#8A5200", "#8A5200"),
        "warning_soft": ("#FFF1D6", "#FFF1D6"),
        "danger": ("#A1262F", "#A1262F"),
        "danger_hover": ("#811E26", "#811E26"),
        "danger_soft": ("#F7E4E6", "#F7E4E6"),
        "link": ("#0066CC", "#0066CC"),
        "institution": ("#C0002B", "#C0002B"),
        "selection": ("#CBE8F8", "#CBE8F8"),
        "canvas": ("#101820", "#101820"),
        "accent": ("#0B5F9E", "#0B5F9E"),
        "accent_hover": ("#084B7D", "#084B7D"),
        "accent_soft": ("#E5F1F9", "#E5F1F9"),
    }
)

_ACTIVE_COLOR_PAIRS: dict[str, ColorPair] = {
    name: _LiveColorPair(value) for name, value in MODERN_COLOR_PAIRS.items()
}
COLOR_PAIRS: Mapping[str, ColorPair] = MappingProxyType(_ACTIVE_COLOR_PAIRS)
_active_interface_mode = "Modern"


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


def normalize_interface_mode(value: object) -> str:
    """Return a supported presentation mode, defaulting safely to Modern."""

    text = str(value or "").strip().lower()
    return {"modern": "Modern", "classic": "Classic"}.get(text, "Modern")


def get_interface_mode() -> str:
    """Return the presentation mode currently supplying the design tokens."""

    return _active_interface_mode


def set_interface_mode(value: object, *, redraw: bool = True) -> str:
    """Select live design tokens without replacing their shared identities."""

    global _active_interface_mode
    normalized = normalize_interface_mode(value)
    palette = CLASSIC_COLOR_PAIRS if normalized == "Classic" else MODERN_COLOR_PAIRS
    for name, value in palette.items():
        live_pair = _ACTIVE_COLOR_PAIRS[name]
        live_pair[:] = value
    _active_interface_mode = normalized
    # Changing Light to Light is intentionally a no-op in CTk.  Force its
    # registered widgets to redraw because the stable color pairs changed
    # beneath them even when the appearance index did not.
    if redraw:
        try:
            ctk.AppearanceModeTracker.update_callbacks()
        except (AttributeError, tk.TclError):
            pass
    return normalized


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
    classic = get_interface_mode() == "Classic"
    color = lambda name: resolve_color(COLOR_PAIRS[name], mode)
    available = style.theme_names()
    if classic:
        preferred_theme = next(
            (name for name in ("xpnative", "vista", "clam") if name in available),
            available[0],
        )
    else:
        preferred_theme = "clam" if "clam" in available else available[0]
    if style.theme_use() != preferred_theme:
        style.theme_use(preferred_theme)

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
        padding=(9, 5) if classic else (11, 7),
        relief="raised" if classic else "flat",
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
    # equal padding on every side so their glyph stays centered in a square.
    for style_name, padding in (
        ("AIDaS.Action.TButton", (11, 4)),
        ("AIDaS.Icon.TButton", (4, 4)),
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
    style.configure("AIDaS.Icon.TButton", anchor="center")
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
        rowheight=24 if classic else CONTROLS.tree_row_height,
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
        padding=(7, 5) if classic else (8, 7),
        relief="raised" if classic else "flat",
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

    notebook_margins = (8, 8, 8, 0) if classic else (0, 0, 0, 0)
    tab_padding = (14, 8) if classic else (12, 7)
    style.configure(
        "TNotebook",
        background=color("application"),
        borderwidth=0,
        tabmargins=notebook_margins,
    )
    style.configure(
        "TNotebook.Tab",
        background=color("surface_subtle"),
        foreground=color("muted_text"),
        font=strong_font,
        padding=tab_padding,
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
        padding=(10, 5) if classic else (12, 6),
        relief="sunken" if classic else "flat",
    )
    style.map(
        "AIDaS.Status.TLabel",
        background=[("disabled", color("surface_subtle"))],
    )
    style.configure(
        "AIDaS.TNotebook",
        background=color("application"),
        borderwidth=0,
        tabmargins=notebook_margins,
    )
    style.configure(
        "AIDaS.TNotebook.Tab",
        background=color("surface_subtle"),
        foreground=color("muted_text"),
        font=strong_font,
        padding=tab_padding,
        borderwidth=0,
    )
    style.map(
        "AIDaS.TNotebook.Tab",
        background=[("selected", color("surface")), ("active", color("primary_soft"))],
        foreground=[("selected", color("primary")), ("active", color("text"))],
    )
    # Modern supplies its own workflow header, while Classic restores the
    # visible four-tab navigation used by AIDaS v2.
    try:
        style.layout(
            "AIDaS.TNotebook.Tab",
            style.layout("TNotebook.Tab") if classic else [],
        )
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


def refresh_interface_widgets(root: tk.Misc) -> int:
    """Flatten or restore existing CTk corners for the active interface.

    Color tokens update live because every widget holds the same mutable color
    pair.  Corner radii are scalar values, so Classic temporarily stores each
    widget's exact Modern value and restores it on the return trip.  This also
    handles widgets created while Classic is active: their canonical radius is
    captured before the widget is flattened.
    """

    classic = get_interface_mode() == "Classic"
    changed = 0
    for widget in (root, *_walk_widgets(root)):
        try:
            current = widget.cget("corner_radius")
        except (AttributeError, KeyError, tk.TclError, TypeError, ValueError):
            continue
        snapshot_name = "_aidas_modern_corner_radius"
        if classic:
            if not hasattr(widget, snapshot_name):
                try:
                    setattr(widget, snapshot_name, current)
                except (AttributeError, TypeError):
                    continue
            if current == 0:
                continue
            target = 0
        else:
            if not hasattr(widget, snapshot_name):
                continue
            target = getattr(widget, snapshot_name)
            if current == target:
                try:
                    delattr(widget, snapshot_name)
                except AttributeError:
                    pass
                continue
        try:
            widget.configure(corner_radius=target)
            if not classic:
                try:
                    delattr(widget, snapshot_name)
                except AttributeError:
                    pass
            changed += 1
        except (AttributeError, KeyError, tk.TclError, TypeError, ValueError):
            pass
    return changed


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
            elif isinstance(widget, tk.Menu) and get_interface_mode() == "Modern":
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
    force_ctk_redraw: bool = False,
    defer_ctk_ms: int = 0,
) -> str:
    """Apply one appearance mode to CTk plus all retained native widgets."""

    normalized = normalize_appearance_mode(mode)
    pending_name = "_aidas_deferred_appearance_after_id"
    if root is not None:
        pending = getattr(root, pending_name, None)
        if pending is not None:
            try:
                root.after_cancel(pending)
            except (AttributeError, tk.TclError):
                pass
            try:
                setattr(root, pending_name, None)
            except (AttributeError, TypeError):
                pass
    if style is not None:
        # ttk styling is cheap and must be ready before the replacement shell
        # is constructed. The expensive CTk callback fan-out may be deferred
        # until after the menu command has returned and the shell can paint.
        configure_ttk_styles(style, normalized)

    def finish() -> None:
        if root is not None:
            try:
                setattr(root, pending_name, None)
            except (AttributeError, TypeError):
                pass
        previous_effective = ctk.get_appearance_mode()
        ctk.set_appearance_mode(normalized)
        if normalized == "System":
            # CTk's public System setter only changes who owns the mode; its
            # periodic detector updates the effective Light/Dark value later.
            # Resolve it now so retained ttk widgets and the replacement shell
            # agree on the first frame after a Classic -> Modern switch.
            try:
                ctk.AppearanceModeTracker.init_appearance_mode()
            except (AttributeError, tk.TclError):
                pass
        if force_ctk_redraw and ctk.get_appearance_mode() == previous_effective:
            try:
                ctk.AppearanceModeTracker.update_callbacks()
            except (AttributeError, tk.TclError):
                pass
        if style is not None and normalized == "System":
            # The eager pass above establishes the correct Notebook layout;
            # this pass resolves colors against CTk's now-current OS mode.
            configure_ttk_styles(style, normalized)
        if root is None:
            return
        if hasattr(root, "_last_effective_appearance"):
            try:
                root._last_effective_appearance = ctk.get_appearance_mode()
            except (AttributeError, TypeError):
                pass
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

    if root is not None and int(defer_ctk_ms) > 0:
        try:
            pending = root.after(max(1, int(defer_ctk_ms)), finish)
            setattr(root, pending_name, pending)
            return normalized
        except (AttributeError, tk.TclError, TypeError, ValueError):
            pass
    finish()
    return normalized


__all__ = [
    "APPEARANCE_MODES",
    "CLASSIC_COLOR_PAIRS",
    "COLORS",
    "COLOR_PAIRS",
    "CONTROLS",
    "INTERFACE_MODES",
    "MODERN_COLOR_PAIRS",
    "SHAPES",
    "THEME",
    "TYPOGRAPHY",
    "apply_appearance_mode",
    "configure_ttk_styles",
    "get_interface_mode",
    "normalize_appearance_mode",
    "normalize_interface_mode",
    "refresh_interface_widgets",
    "refresh_native_widgets",
    "resolve_color",
    "set_interface_mode",
]
