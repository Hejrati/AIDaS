"""DPI-aware monitor selection and window positioning helpers."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import tkinter as tk


MonitorBounds = tuple[int, int, int, int]


class _Point(ctypes.Structure):
    _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))


class _Rect(ctypes.Structure):
    _fields_ = (
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    )


class _MonitorInfo(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", _Rect),
        ("rcWork", _Rect),
        ("dwFlags", wintypes.DWORD),
    )


def _rect_bounds(rect: _Rect) -> MonitorBounds | None:
    bounds = int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def _monitor_work_area(user32, monitor) -> MonitorBounds | None:
    if not monitor:
        return None
    get_monitor_info = user32.GetMonitorInfoW
    get_monitor_info.argtypes = (wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo))
    get_monitor_info.restype = wintypes.BOOL
    info = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
    if not get_monitor_info(monitor, ctypes.byref(info)):
        return None
    return _rect_bounds(info.rcWork)


def enable_per_monitor_dpi_awareness() -> None:
    """Enable the best Windows DPI mode available before Tk creates a window."""
    if os.name != "nt":
        return

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        set_context = user32.SetProcessDpiAwarenessContext
        set_context.argtypes = (wintypes.HANDLE,)
        set_context.restype = wintypes.BOOL
        if set_context(ctypes.c_void_p(-4)):  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            return
    except (AttributeError, OSError):
        pass

    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        set_awareness = shcore.SetProcessDpiAwareness
        set_awareness.argtypes = (ctypes.c_int,)
        set_awareness.restype = ctypes.c_long
        if set_awareness(2) == 0:  # PROCESS_PER_MONITOR_DPI_AWARE
            return
    except (AttributeError, OSError):
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def monitor_work_area_at_pointer() -> MonitorBounds | None:
    """Return the Windows working area for the display containing the pointer."""
    if os.name != "nt":
        return None

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        get_cursor_pos = user32.GetCursorPos
        get_cursor_pos.argtypes = (ctypes.POINTER(_Point),)
        get_cursor_pos.restype = wintypes.BOOL
        monitor_from_point = user32.MonitorFromPoint
        monitor_from_point.argtypes = (_Point, wintypes.DWORD)
        monitor_from_point.restype = wintypes.HMONITOR

        point = _Point()
        if not get_cursor_pos(ctypes.byref(point)):
            return None
        monitor = monitor_from_point(point, 2)  # MONITOR_DEFAULTTONEAREST
        return _monitor_work_area(user32, monitor)
    except (AttributeError, OSError, ValueError):
        return None


def monitor_work_area_for_window(window) -> MonitorBounds | None:
    """Return the usable display area containing a Tk window's native frame."""
    if os.name != "nt":
        return None

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        get_ancestor = user32.GetAncestor
        get_ancestor.argtypes = (wintypes.HWND, wintypes.UINT)
        get_ancestor.restype = wintypes.HWND
        monitor_from_window = user32.MonitorFromWindow
        monitor_from_window.argtypes = (wintypes.HWND, wintypes.DWORD)
        monitor_from_window.restype = wintypes.HMONITOR
        client_handle = int(window.winfo_toplevel().winfo_id())
        frame_handle = get_ancestor(client_handle, 2) or client_handle  # GA_ROOT
        monitor = monitor_from_window(frame_handle, 2)  # MONITOR_DEFAULTTONEAREST
        return _monitor_work_area(user32, monitor)
    except (AttributeError, OSError, TypeError, ValueError, tk.TclError):
        return None


def _primary_monitor_work_area() -> MonitorBounds | None:
    """Preserve taskbar space even when cursor and window queries fail."""
    if os.name != "nt":
        return None
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        system_parameters_info = user32.SystemParametersInfoW
        system_parameters_info.argtypes = (
            wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT,
        )
        system_parameters_info.restype = wintypes.BOOL
        area = _Rect()
        if system_parameters_info(48, 0, ctypes.byref(area), 0):  # SPI_GETWORKAREA
            return _rect_bounds(area)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def centered_position(bounds: MonitorBounds, width: int, height: int) -> tuple[int, int]:
    """Center a window within bounds, including monitors with negative origins."""
    left, top, right, bottom = (int(value) for value in bounds)
    available_width = max(0, right - left)
    available_height = max(0, bottom - top)
    x = left + max(0, (available_width - int(width)) // 2)
    y = top + max(0, (available_height - int(height)) // 2)
    return x, y


def centered_decorated_position(
    bounds: MonitorBounds,
    client_width: int,
    client_height: int,
    *,
    frame_left: int = 0,
    frame_top: int = 0,
    frame_right: int | None = None,
    frame_bottom: int | None = None,
) -> tuple[int, int]:
    """Center the complete OS window while Tk sizes its client area.

    Tk's geometry width and height describe the client area, while its x and y
    position describe the decorated window.  Accounting for the title bar and
    borders prevents an apparently low/right startup position on Windows.
    """

    left = max(0, int(frame_left))
    top = max(0, int(frame_top))
    right = left if frame_right is None else max(0, int(frame_right))
    bottom = left if frame_bottom is None else max(0, int(frame_bottom))
    outer_width = max(1, int(client_width)) + left + right
    outer_height = max(1, int(client_height)) + top + bottom
    return centered_position(bounds, outer_width, outer_height)


def work_area_bounds(window, *, parent=None, monitor_window=None) -> MonitorBounds:
    """Return parent bounds or usable monitor bounds, preferring an owner if given."""
    window.update_idletasks()
    if parent is not None and parent.winfo_viewable():
        return (
            parent.winfo_rootx(),
            parent.winfo_rooty(),
            parent.winfo_rootx() + parent.winfo_width(),
            parent.winfo_rooty() + parent.winfo_height(),
        )

    if monitor_window is not None:
        bounds = monitor_work_area_for_window(monitor_window)
        if bounds is not None:
            return bounds

    bounds = monitor_work_area_at_pointer()
    if bounds is not None:
        return bounds

    # GetCursorPos can fail for a locked or noninteractive desktop. Native
    # window and primary-monitor queries still exclude the Windows taskbar.
    bounds = monitor_work_area_for_window(window)
    if bounds is not None:
        return bounds

    bounds = _primary_monitor_work_area()
    if bounds is not None:
        return bounds

    left = int(window.winfo_vrootx())
    top = int(window.winfo_vrooty())
    return (
        left,
        top,
        left + int(window.winfo_vrootwidth()),
        top + int(window.winfo_vrootheight()),
    )


def fit_size_to_bounds(
    bounds: MonitorBounds,
    width: int,
    height: int,
    *,
    maximum_fraction: float = 0.9,
) -> tuple[int, int, float]:
    """Scale a design size down uniformly until it fits usable bounds."""
    left, top, right, bottom = (int(value) for value in bounds)
    available_width = max(1, right - left)
    available_height = max(1, bottom - top)
    fraction = max(0.1, min(float(maximum_fraction), 1.0))
    source_width = max(1, int(width))
    source_height = max(1, int(height))
    scale = min(
        1.0,
        available_width * fraction / source_width,
        available_height * fraction / source_height,
    )
    return (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
        scale,
    )


def fractional_size_of_bounds(
    bounds: MonitorBounds,
    fraction: float,
) -> tuple[int, int]:
    """Return a fixed fraction of the available width and height."""

    left, top, right, bottom = (int(value) for value in bounds)
    available_width = max(1, right - left)
    available_height = max(1, bottom - top)
    safe_fraction = max(0.1, min(float(fraction), 1.0))
    return (
        max(1, round(available_width * safe_fraction)),
        max(1, round(available_height * safe_fraction)),
    )


def centered_geometry(window, width: int, height: int, *, parent=None) -> str:
    """Return Tk geometry centered on a parent or the pointer's current display."""
    bounds = work_area_bounds(window, parent=parent)

    x, y = centered_position(bounds, width, height)
    return f"{int(width)}x{int(height)}{x:+d}{y:+d}"
