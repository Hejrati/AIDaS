"""DPI-aware monitor selection and window positioning helpers."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os


MonitorBounds = tuple[int, int, int, int]


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

    class Point(ctypes.Structure):
        _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))

    class Rect(ctypes.Structure):
        _fields_ = (
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        )

    class MonitorInfo(ctypes.Structure):
        _fields_ = (
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", Rect),
            ("rcWork", Rect),
            ("dwFlags", wintypes.DWORD),
        )

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        get_cursor_pos = user32.GetCursorPos
        get_cursor_pos.argtypes = (ctypes.POINTER(Point),)
        get_cursor_pos.restype = wintypes.BOOL
        monitor_from_point = user32.MonitorFromPoint
        monitor_from_point.argtypes = (Point, wintypes.DWORD)
        monitor_from_point.restype = wintypes.HMONITOR
        get_monitor_info = user32.GetMonitorInfoW
        get_monitor_info.argtypes = (wintypes.HMONITOR, ctypes.POINTER(MonitorInfo))
        get_monitor_info.restype = wintypes.BOOL

        point = Point()
        if not get_cursor_pos(ctypes.byref(point)):
            return None
        monitor = monitor_from_point(point, 2)  # MONITOR_DEFAULTTONEAREST
        if not monitor:
            return None

        info = MonitorInfo(cbSize=ctypes.sizeof(MonitorInfo))
        if not get_monitor_info(monitor, ctypes.byref(info)):
            return None
        area = info.rcWork
        return int(area.left), int(area.top), int(area.right), int(area.bottom)
    except (AttributeError, OSError, ValueError):
        return None


def centered_position(bounds: MonitorBounds, width: int, height: int) -> tuple[int, int]:
    """Center a window within bounds, including monitors with negative origins."""
    left, top, right, bottom = (int(value) for value in bounds)
    available_width = max(0, right - left)
    available_height = max(0, bottom - top)
    x = left + max(0, (available_width - int(width)) // 2)
    y = top + max(0, (available_height - int(height)) // 2)
    return x, y


def work_area_bounds(window, *, parent=None) -> MonitorBounds:
    """Return the usable bounds for a parent or the pointer's current display."""
    window.update_idletasks()
    if parent is not None and parent.winfo_viewable():
        return (
            parent.winfo_rootx(),
            parent.winfo_rooty(),
            parent.winfo_rootx() + parent.winfo_width(),
            parent.winfo_rooty() + parent.winfo_height(),
        )

    bounds = monitor_work_area_at_pointer()
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


def centered_geometry(window, width: int, height: int, *, parent=None) -> str:
    """Return Tk geometry centered on a parent or the pointer's current display."""
    bounds = work_area_bounds(window, parent=parent)

    x, y = centered_position(bounds, width, height)
    return f"{int(width)}x{int(height)}{x:+d}{y:+d}"
