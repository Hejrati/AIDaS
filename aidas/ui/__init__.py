"""Reusable visual components and theme primitives for AIDaS."""

from aidas.ui.menu_bar import ApplicationMenuBar
from aidas.ui.tabs import ClosableTabView
from aidas.ui.title_bar import (
    CustomWindowsTitleBar,
    WindowsCaptionController,
    create_custom_windows_title_bar,
    reassert_client_size,
)
from aidas.ui.theme import (
    APPEARANCE_MODES,
    COLORS,
    COLOR_PAIRS,
    CONTROLS,
    SHAPES,
    THEME,
    TYPOGRAPHY,
    apply_appearance_mode,
    configure_ttk_styles,
    normalize_appearance_mode,
    refresh_native_widgets,
    resolve_color,
)
from aidas.ui.windowing import (
    apply_windows_titlebar_colors,
    centered_logical_geometry,
    centered_physical_geometry,
    logical_window_size,
    physical_window_size,
    synchronize_window_chrome,
)

__all__ = [
    "ApplicationMenuBar",
    "ClosableTabView",
    "CustomWindowsTitleBar",
    "WindowsCaptionController",
    "create_custom_windows_title_bar",
    "reassert_client_size",
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
    "apply_windows_titlebar_colors",
    "centered_logical_geometry",
    "centered_physical_geometry",
    "logical_window_size",
    "physical_window_size",
    "synchronize_window_chrome",
]
