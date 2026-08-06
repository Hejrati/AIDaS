"""Interactive Windows smoke test for the captionless AIDaS main frame."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import time
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidas.app import AIDaSApp
from aidas.ui.title_bar import (
    _RETAINED_NATIVE_STYLES,
    _WS_CAPTION,
    _WS_THICKFRAME,
)
from aidas.ui.windowing import physical_window_size


_GWL_EXSTYLE = -20
_WS_EX_TOOLWINDOW = 0x00000080


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


def _pump(window, seconds: float = 0.25) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        window.update()
        time.sleep(0.01)


def _rect_tuple(rect: _Rect) -> tuple[int, int, int, int]:
    return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)


def _client_screen_rect(user32, handle: int) -> tuple[int, int, int, int]:
    client = _Rect()
    assert user32.GetClientRect(handle, ctypes.byref(client))
    upper_left = _Point(client.left, client.top)
    lower_right = _Point(client.right, client.bottom)
    assert user32.ClientToScreen(handle, ctypes.byref(upper_left))
    assert user32.ClientToScreen(handle, ctypes.byref(lower_right))
    return upper_left.x, upper_left.y, lower_right.x, lower_right.y


def _monitor_work_rect(user32, handle: int) -> tuple[int, int, int, int]:
    monitor = user32.MonitorFromWindow(handle, 2)
    assert monitor
    info = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
    assert user32.GetMonitorInfoW(monitor, ctypes.byref(info))
    return _rect_tuple(info.rcWork)


def main() -> int:
    if not sys.platform.startswith("win"):
        print("CUSTOM_TITLE_BAR_SKIPPED (Windows only)")
        return 0

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    get_window_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    get_window_long.argtypes = (ctypes.c_void_p, ctypes.c_int)
    get_window_long.restype = ctypes.c_ssize_t
    user32.GetClientRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Rect))
    user32.GetClientRect.restype = ctypes.c_bool
    user32.GetWindowRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Rect))
    user32.GetWindowRect.restype = ctypes.c_bool
    user32.ClientToScreen.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Point))
    user32.ClientToScreen.restype = ctypes.c_bool
    user32.MonitorFromWindow.argtypes = (ctypes.c_void_p, wintypes.DWORD)
    user32.MonitorFromWindow.restype = wintypes.HMONITOR
    user32.GetMonitorInfoW.argtypes = (
        wintypes.HMONITOR,
        ctypes.POINTER(_MonitorInfo),
    )
    user32.GetMonitorInfoW.restype = ctypes.c_bool
    user32.IsIconic.argtypes = (ctypes.c_void_p,)
    user32.IsIconic.restype = ctypes.c_bool
    app = AIDaSApp()
    try:
        app._finish_startup()
        _pump(app)
        title_bar = app.window_title_bar
        assert title_bar is not None, "The Windows custom title bar was not installed."
        controller = title_bar.controller
        assert controller.handle is not None
        assert controller._api is not None
        handle = controller.handle

        style = controller._api.get_style(handle)
        assert style & _WS_CAPTION == 0, "WS_CAPTION was restored unexpectedly."
        assert style & _WS_THICKFRAME == 0, (
            "WS_THICKFRAME was restored and can paint an unthemed outer border."
        )
        assert style & _RETAINED_NATIVE_STYLES == _RETAINED_NATIVE_STYLES, (
            "Native system/minimize/maximize commands were not preserved."
        )
        extended_style = int(get_window_long(handle, _GWL_EXSTYLE)) & 0xFFFFFFFF
        assert extended_style & _WS_EX_TOOLWINDOW == 0, (
            "The main window became a tool window and may disappear from the taskbar."
        )

        expected_width, expected_height = physical_window_size(
            app, *app._startup_window_size
        )
        assert abs(app.winfo_width() - expected_width) <= 1
        assert abs(app.winfo_height() - expected_height) <= 1

        outer = _Rect()
        assert user32.GetWindowRect(handle, ctypes.byref(outer))
        outer_rect = _rect_tuple(outer)
        client_rect = _client_screen_rect(user32, handle)
        assert all(
            abs(client_value - outer_value) <= 1
            for client_value, outer_value in zip(client_rect, outer_rect)
        ), (
            "The client does not fill the captionless outer window: "
            f"client={client_rect}, outer={outer_rect}."
        )

        normal_width = outer_rect[2] - outer_rect[0]
        normal_height = outer_rect[3] - outer_rect[1]
        resized_width = normal_width - 32
        resized_height = normal_height - 24
        southeast_edge = SimpleNamespace(
            x_root=outer_rect[2] - 1,
            y_root=outer_rect[3] - 1,
            widget=app,
        )
        assert title_bar._start_client_resize(southeast_edge), (
            "The client-side southeast resize edge was not detected."
        )
        title_bar._drag_client_resize(
            SimpleNamespace(
                x_root=southeast_edge.x_root - 32,
                y_root=southeast_edge.y_root - 24,
                widget=app,
            )
        )
        title_bar._end_client_resize()
        _pump(app)
        resized_outer = _Rect()
        assert user32.GetWindowRect(handle, ctypes.byref(resized_outer))
        resized_rect = _rect_tuple(resized_outer)
        assert abs((resized_rect[2] - resized_rect[0]) - resized_width) <= 1
        assert abs((resized_rect[3] - resized_rect[1]) - resized_height) <= 1
        assert controller.resize_window(
            outer_rect[0], outer_rect[1], normal_width, normal_height
        )
        _pump(app)

        normal_geometry = app.geometry()
        controller.toggle_maximize()
        _pump(app)
        assert controller.is_maximized(), "Native maximize did not enter zoomed state."
        work_rect = _monitor_work_rect(user32, handle)
        client_rect = _client_screen_rect(user32, handle)
        assert all(
            abs(client_value - work_value) <= 1
            for client_value, work_value in zip(client_rect, work_rect)
        ), (
            "Maximized client area does not match the monitor work area: "
            f"client={client_rect}, work={work_rect}."
        )
        assert (int(get_window_long(handle, _GWL_EXSTYLE)) & 0xFFFFFFFF) == extended_style

        controller.toggle_maximize()
        _pump(app)
        assert not controller.is_maximized(), "Native restore left the window zoomed."
        assert abs(app.winfo_width() - expected_width) <= 1
        assert abs(app.winfo_height() - expected_height) <= 1

        controller.minimize()
        _pump(app)
        assert bool(user32.IsIconic(handle)), "Native minimize did not iconify the app."
        controller.restore()
        _pump(app)
        assert not bool(user32.IsIconic(handle)), "The app did not restore from minimize."
        assert (int(get_window_long(handle, _GWL_EXSTYLE)) & 0xFFFFFFFF) == extended_style

        print("CUSTOM_TITLE_BAR_OK")
        print("style", hex(style), "extended_style", hex(extended_style))
        print("normal_geometry", normal_geometry)
        print("work_rect", work_rect, "maximized_client_rect", client_rect)
        return 0
    finally:
        try:
            app.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
