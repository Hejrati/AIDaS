"""CustomTkinter-aware window sizing and centering helpers.

Monitor APIs and ``winfo_*`` report physical pixels. CustomTkinter geometry
expects logical dimensions and applies its own DPI scaling, while x/y positions
remain physical. Keeping that conversion here prevents double-scaled windows.
"""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from typing import Sequence

from aidas.core.display import centered_position, work_area_bounds


WindowColor = str | Sequence[str]
_GA_ROOT = 2
_DWMWA_COLOR_NONE = 0xFFFFFFFE


def _resolved_window_color(window: tk.Misc, color: WindowColor) -> str:
    """Resolve a CTk color pair without importing the theme module here."""

    if isinstance(color, str):
        return color
    resolver = getattr(window, "_apply_appearance_mode", None)
    if callable(resolver):
        return str(resolver(color))
    return str(color[0])


def _hex_to_colorref(color: str) -> int:
    """Convert ``#RRGGBB`` to the COLORREF layout used by Windows DWM."""

    value = str(color).strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return red | (green << 8) | (blue << 16)


def _is_dark_color(color: str) -> bool:
    value = str(color).strip().lstrip("#")
    if len(value) != 6:
        return False
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance < 128


def _native_border_colorref(window: tk.Misc, border_hex: str) -> int:
    """Return a themed border, or Win11's no-border sentinel for custom frames."""

    if getattr(window, "_aidas_suppress_native_border", False):
        return _DWMWA_COLOR_NONE
    return _hex_to_colorref(border_hex)


def apply_windows_titlebar_colors(
    window: tk.Misc,
    *,
    background: WindowColor,
    foreground: WindowColor,
    border: WindowColor | None = None,
) -> bool:
    """Match a decorated Windows caption to the active AIDaS palette.

    Windows 11 supports explicit caption, text, and border colors. Older
    Windows releases fall back to the immersive light/dark caption flag.
    Failures are intentionally non-fatal so Tk remains portable.
    """

    if not sys.platform.startswith("win"):
        return False

    try:
        window.update_idletasks()
        background_hex = _resolved_window_color(window, background)
        foreground_hex = _resolved_window_color(window, foreground)
        border_hex = _resolved_window_color(window, border or background)

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        get_ancestor = user32.GetAncestor
        get_ancestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        get_ancestor.restype = ctypes.c_void_p
        get_parent = user32.GetParent
        get_parent.argtypes = (ctypes.c_void_p,)
        get_parent.restype = ctypes.c_void_p
        widget_handle = ctypes.c_void_p(int(window.winfo_id()))
        # Tk's ``winfo_id`` points at its client child and ``GetParent`` may
        # only reach an intermediate wrapper. DWM caption attributes must be
        # applied to the decorated root HWND itself.
        window_handle = (
            get_ancestor(widget_handle, _GA_ROOT)
            or get_parent(widget_handle)
            or widget_handle.value
        )

        dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
        set_attribute = dwmapi.DwmSetWindowAttribute
        set_attribute.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
        )
        set_attribute.restype = ctypes.c_long

        immersive = ctypes.c_int(1 if _is_dark_color(background_hex) else 0)
        result = set_attribute(
            window_handle,
            20,  # DWMWA_USE_IMMERSIVE_DARK_MODE
            ctypes.byref(immersive),
            ctypes.sizeof(immersive),
        )
        if result != 0:
            set_attribute(
                window_handle,
                19,  # pre-20H1 compatibility value
                ctypes.byref(immersive),
                ctypes.sizeof(immersive),
            )

        results = []
        for attribute, color in (
            (34, border_hex),  # DWMWA_BORDER_COLOR
            (35, background_hex),  # DWMWA_CAPTION_COLOR
            (36, foreground_hex),  # DWMWA_TEXT_COLOR
        ):
            if attribute == 34 and getattr(
                window, "_aidas_suppress_native_border", False
            ):
                # Windows 11 build 22000+ supports COLOR_NONE. Windows 10
                # rejects it harmlessly and uses the custom client frame.
                colorref = ctypes.c_uint(
                    _native_border_colorref(window, border_hex)
                )
            else:
                colorref = ctypes.c_uint(_hex_to_colorref(color))
            results.append(
                set_attribute(
                    window_handle,
                    attribute,
                    ctypes.byref(colorref),
                    ctypes.sizeof(colorref),
                )
                == 0
            )
        return all(results)
    except (AttributeError, OSError, TypeError, ValueError, tk.TclError):
        return False


def synchronize_window_chrome(
    window: tk.Misc,
    *,
    background: WindowColor,
    foreground: WindowColor,
    border: WindowColor | None = None,
) -> None:
    """Apply caption colors now and once more after CTk's own redraw."""

    def apply() -> None:
        apply_windows_titlebar_colors(
            window,
            background=background,
            foreground=foreground,
            border=border,
        )

    apply()
    try:
        window.after(100, apply)
    except tk.TclError:
        pass


def logical_window_size(
    window: tk.Misc,
    physical_width: int,
    physical_height: int,
) -> tuple[int, int]:
    """Convert physical-pixel dimensions to CTk logical window dimensions."""

    reverse = getattr(window, "_reverse_window_scaling", None)
    if callable(reverse):
        return (
            max(1, int(reverse(max(1, int(physical_width))))),
            max(1, int(reverse(max(1, int(physical_height))))),
        )
    return max(1, int(physical_width)), max(1, int(physical_height))


def physical_window_size(
    window: tk.Misc,
    logical_width: int,
    logical_height: int,
) -> tuple[int, int]:
    """Convert logical CTk dimensions to physical pixels used for centering."""

    apply = getattr(window, "_apply_window_scaling", None)
    if callable(apply):
        return (
            max(1, int(apply(max(1, int(logical_width))))),
            max(1, int(apply(max(1, int(logical_height))))),
        )
    return max(1, int(logical_width)), max(1, int(logical_height))


def centered_logical_geometry(
    window: tk.Misc,
    logical_width: int,
    logical_height: int,
    *,
    bounds: tuple[int, int, int, int] | None = None,
    parent: tk.Misc | None = None,
) -> str:
    """Build CTk geometry from logical dimensions and physical bounds."""

    if bounds is None:
        bounds = work_area_bounds(window, parent=parent)
    physical_width, physical_height = physical_window_size(
        window,
        logical_width,
        logical_height,
    )
    x, y = centered_position(bounds, physical_width, physical_height)
    return f"{int(logical_width)}x{int(logical_height)}{x:+d}{y:+d}"


def centered_physical_geometry(
    window: tk.Misc,
    physical_width: int,
    physical_height: int,
    *,
    bounds: tuple[int, int, int, int] | None = None,
    parent: tk.Misc | None = None,
) -> str:
    """Build CTk geometry when requested dimensions are physical pixels."""

    if bounds is None:
        bounds = work_area_bounds(window, parent=parent)
    logical_width, logical_height = logical_window_size(
        window,
        physical_width,
        physical_height,
    )
    x, y = centered_position(bounds, physical_width, physical_height)
    return f"{logical_width}x{logical_height}{x:+d}{y:+d}"


__all__ = [
    "apply_windows_titlebar_colors",
    "centered_logical_geometry",
    "centered_physical_geometry",
    "logical_window_size",
    "physical_window_size",
    "synchronize_window_chrome",
]
