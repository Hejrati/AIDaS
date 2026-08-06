"""Reusable Windows title bar that keeps native window behavior.

The component removes the system-drawn caption and resize frame from a normal
Tk top-level, then supplies a small client-side resize interaction. It does not
use ``overrideredirect``, so the taskbar entry, system menu, minimize/maximize
styles, keyboard window management, and native caption drag remain available.
On unsupported platforms or unexpected window styles, installation fails
closed and callers can keep the operating-system caption.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import tkinter as tk
from typing import Protocol

import customtkinter as ctk
from PIL import Image

from aidas.ui.theme import COLOR_PAIRS, TYPOGRAPHY


_GWL_STYLE = -16
_GA_ROOT = 2

_WS_CAPTION = 0x00C00000
_WS_THICKFRAME = 0x00040000
_WS_SYSMENU = 0x00080000
_WS_MINIMIZEBOX = 0x00020000
_WS_MAXIMIZEBOX = 0x00010000
_REQUIRED_NATIVE_STYLES = (
    _WS_THICKFRAME | _WS_SYSMENU | _WS_MINIMIZEBOX | _WS_MAXIMIZEBOX
)
_RETAINED_NATIVE_STYLES = _WS_SYSMENU | _WS_MINIMIZEBOX | _WS_MAXIMIZEBOX

_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SWP_FRAMECHANGED = 0x0020

_WM_CLOSE = 0x0010
_WM_NCLBUTTONDOWN = 0x00A1
_HTCAPTION = 2
_CLIENT_RESIZE_MARGIN = 6


class _Rect(ctypes.Structure):
    _fields_ = (
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    )


class _Point(ctypes.Structure):
    _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))


class _MonitorInfo(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", _Rect),
        ("rcWork", _Rect),
        ("dwFlags", wintypes.DWORD),
    )


def _outer_rect_for_client_work_area(
    work_rect: tuple[int, int, int, int],
    outer_rect: tuple[int, int, int, int],
    client_rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return ``x, y, width, height`` placing the client on ``work_rect``."""

    left_margin = client_rect[0] - outer_rect[0]
    top_margin = client_rect[1] - outer_rect[1]
    right_margin = outer_rect[2] - client_rect[2]
    bottom_margin = outer_rect[3] - client_rect[3]
    return (
        work_rect[0] - left_margin,
        work_rect[1] - top_margin,
        work_rect[2] - work_rect[0] + left_margin + right_margin,
        work_rect[3] - work_rect[1] + top_margin + bottom_margin,
    )


def _resized_window_rect(
    start_rect: tuple[int, int, int, int],
    edge: str,
    delta_x: int,
    delta_y: int,
    minimum_width: int,
    minimum_height: int,
) -> tuple[int, int, int, int]:
    """Resize ``x, y, width, height`` from one or two client-side edges."""

    left, top, width, height = (int(value) for value in start_rect)
    right = left + max(1, width)
    bottom = top + max(1, height)
    minimum_width = max(1, int(minimum_width))
    minimum_height = max(1, int(minimum_height))

    if "w" in edge:
        left = min(left + int(delta_x), right - minimum_width)
    elif "e" in edge:
        right = max(right + int(delta_x), left + minimum_width)
    if "n" in edge:
        top = min(top + int(delta_y), bottom - minimum_height)
    elif "s" in edge:
        bottom = max(bottom + int(delta_y), top + minimum_height)
    return left, top, right - left, bottom - top


class _NativeWindowAPI(Protocol):
    """Small injectable WinAPI surface used by the controller and tests."""

    def root_handle(self, window: tk.Misc) -> int: ...

    def get_style(self, handle: int) -> int: ...

    def set_style(self, handle: int, style: int) -> bool: ...

    def refresh_frame(self, handle: int) -> bool: ...

    def resize_window(
        self,
        handle: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bool: ...

    def begin_caption_drag(self, handle: int) -> None: ...

    def is_zoomed(self, handle: int) -> bool: ...

    def fit_maximized_to_work_area(self, handle: int) -> bool: ...

    def close_window(self, handle: int) -> None: ...


class _WindowsAPI:
    """Pointer-size-safe wrappers for the few user32 calls we need."""

    def __init__(self) -> None:
        if not sys.platform.startswith("win"):
            raise OSError("Windows window APIs are unavailable")

        # Frame/style calls can synchronously dispatch Tk window messages.
        # PyDLL intentionally keeps the GIL held during that re-entrancy.
        user32 = ctypes.PyDLL("user32", use_last_error=True)
        self._user32 = user32

        self._get_ancestor = user32.GetAncestor
        self._get_ancestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        self._get_ancestor.restype = ctypes.c_void_p

        self._get_parent = user32.GetParent
        self._get_parent.argtypes = (ctypes.c_void_p,)
        self._get_parent.restype = ctypes.c_void_p

        try:
            self._get_window_long = user32.GetWindowLongPtrW
            self._set_window_long = user32.SetWindowLongPtrW
            long_result = ctypes.c_ssize_t
        except AttributeError:  # 32-bit Windows
            self._get_window_long = user32.GetWindowLongW
            self._set_window_long = user32.SetWindowLongW
            long_result = ctypes.c_long
        self._get_window_long.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self._get_window_long.restype = long_result
        self._set_window_long.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            long_result,
        )
        self._set_window_long.restype = long_result
        self._long_result = long_result

        self._set_window_pos = user32.SetWindowPos
        self._set_window_pos.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        )
        self._set_window_pos.restype = ctypes.c_bool

        self._release_capture = user32.ReleaseCapture
        self._release_capture.argtypes = ()
        self._release_capture.restype = ctypes.c_bool

        self._is_zoomed = user32.IsZoomed
        self._is_zoomed.argtypes = (ctypes.c_void_p,)
        self._is_zoomed.restype = ctypes.c_bool

        self._get_window_rect = user32.GetWindowRect
        self._get_window_rect.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_Rect),
        )
        self._get_window_rect.restype = ctypes.c_bool

        self._get_client_rect = user32.GetClientRect
        self._get_client_rect.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_Rect),
        )
        self._get_client_rect.restype = ctypes.c_bool

        self._client_to_screen = user32.ClientToScreen
        self._client_to_screen.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_Point),
        )
        self._client_to_screen.restype = ctypes.c_bool

        self._monitor_from_window = user32.MonitorFromWindow
        self._monitor_from_window.argtypes = (ctypes.c_void_p, wintypes.DWORD)
        self._monitor_from_window.restype = wintypes.HMONITOR

        self._get_monitor_info = user32.GetMonitorInfoW
        self._get_monitor_info.argtypes = (
            wintypes.HMONITOR,
            ctypes.POINTER(_MonitorInfo),
        )
        self._get_monitor_info.restype = ctypes.c_bool

        self._post_message = user32.PostMessageW
        self._post_message.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
        )
        self._post_message.restype = ctypes.c_bool

    def root_handle(self, window: tk.Misc) -> int:
        window.update_idletasks()
        widget_handle = ctypes.c_void_p(int(window.winfo_id()))
        handle = (
            self._get_ancestor(widget_handle, _GA_ROOT)
            or self._get_parent(widget_handle)
            or widget_handle.value
        )
        if not handle:
            raise OSError("Unable to resolve the native root window")
        return int(handle)

    def get_style(self, handle: int) -> int:
        # Window styles are 32-bit bitfields even when LONG_PTR is 64-bit.
        return int(self._get_window_long(handle, _GWL_STYLE)) & 0xFFFFFFFF

    def set_style(self, handle: int, style: int) -> bool:
        self._set_window_long(
            handle,
            _GWL_STYLE,
            self._long_result(int(style) & 0xFFFFFFFF),
        )
        return self.get_style(handle) == (int(style) & 0xFFFFFFFF)

    def refresh_frame(self, handle: int) -> bool:
        return bool(
            self._set_window_pos(
                handle,
                None,
                0,
                0,
                0,
                0,
                _SWP_NOMOVE
                | _SWP_NOSIZE
                | _SWP_NOZORDER
                | _SWP_NOACTIVATE
                | _SWP_FRAMECHANGED,
            )
        )

    def resize_window(
        self,
        handle: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bool:
        """Resize the borderless outer window in physical screen pixels."""

        return bool(
            self._set_window_pos(
                handle,
                None,
                int(x),
                int(y),
                max(1, int(width)),
                max(1, int(height)),
                _SWP_NOZORDER | _SWP_NOACTIVATE,
            )
        )

    def begin_caption_drag(self, handle: int) -> None:
        # Delegating the move loop to Windows preserves Aero Snap, Win+Arrow,
        # restore-on-drag, and multi-monitor DPI transitions. PostMessage is
        # deliberately asynchronous: SendMessage enters Windows' modal move
        # loop inside ctypes and can reenter Tk without a valid Python thread.
        self._release_capture()
        self._post_message(handle, _WM_NCLBUTTONDOWN, _HTCAPTION, 0)

    def is_zoomed(self, handle: int) -> bool:
        return bool(self._is_zoomed(handle))

    def fit_maximized_to_work_area(self, handle: int) -> bool:
        """Keep a captionless maximized client above the Windows taskbar."""

        monitor = self._monitor_from_window(handle, 2)  # nearest monitor
        if not monitor:
            return False
        info = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
        if not self._get_monitor_info(monitor, ctypes.byref(info)):
            return False
        work = info.rcWork
        outer = _Rect()
        client = _Rect()
        if not self._get_window_rect(handle, ctypes.byref(outer)):
            return False
        if not self._get_client_rect(handle, ctypes.byref(client)):
            return False
        client_upper_left = _Point(client.left, client.top)
        client_lower_right = _Point(client.right, client.bottom)
        if not self._client_to_screen(handle, ctypes.byref(client_upper_left)):
            return False
        if not self._client_to_screen(handle, ctypes.byref(client_lower_right)):
            return False

        work_rect = (
            int(work.left),
            int(work.top),
            int(work.right),
            int(work.bottom),
        )
        client_screen_rect = (
            int(client_upper_left.x),
            int(client_upper_left.y),
            int(client_lower_right.x),
            int(client_lower_right.y),
        )
        if client_screen_rect == work_rect:
            return True

        # SetWindowPos sizes the outer rectangle. Expand rcWork by the current
        # non-client margins so the *client* rectangle lands exactly on rcWork.
        target = _outer_rect_for_client_work_area(
            work_rect,
            (
                int(outer.left),
                int(outer.top),
                int(outer.right),
                int(outer.bottom),
            ),
            client_screen_rect,
        )
        return bool(
            self._set_window_pos(
                handle,
                None,
                *target,
                _SWP_NOZORDER | _SWP_NOACTIVATE,
            )
        )

    def close_window(self, handle: int) -> None:
        # WM_CLOSE respects Tk's WM_DELETE_WINDOW protocol when one exists.
        self._post_message(handle, _WM_CLOSE, 0, 0)


class WindowsCaptionController:
    """Remove native frame painting while retaining normal window commands."""

    def __init__(
        self,
        window: tk.Misc,
        *,
        native_api: _NativeWindowAPI | None = None,
    ) -> None:
        self.window = window
        self._api = native_api
        self.handle: int | None = None
        self.original_style: int | None = None
        self.style: int | None = None
        self.installed = False

    def install(self) -> bool:
        """Install once, returning false without mutating unsupported windows."""

        if self.installed:
            return True
        if self._api is None:
            if not sys.platform.startswith("win"):
                return False
            try:
                self._api = _WindowsAPI()
            except Exception:
                return False

        handle: int | None = None
        original_style: int | None = None
        mutation_attempted = False
        try:
            handle = self._api.root_handle(self.window)
            original_style = self._api.get_style(handle)
            if original_style & _REQUIRED_NATIVE_STYLES != _REQUIRED_NATIVE_STYLES:
                return False
            # On Windows 10, retaining WS_THICKFRAME after removing the
            # caption leaves a system-painted black strip that cannot be
            # themed. Client-side edge resizing replaces that visual frame.
            captionless_style = original_style & ~(_WS_CAPTION | _WS_THICKFRAME)
            # Mark the attempt first: SetWindowLongPtr can mutate the style and
            # still surface an exception or an unexpected verification result.
            mutation_attempted = True
            if not self._api.set_style(handle, captionless_style):
                self._rollback_failed_install(handle, original_style)
                return False
            if not self._api.refresh_frame(handle):
                self._rollback_failed_install(handle, original_style)
                return False
        except Exception:
            if mutation_attempted and handle is not None and original_style is not None:
                self._rollback_failed_install(handle, original_style)
            return False

        self.handle = handle
        self.original_style = original_style
        self.style = captionless_style
        self.installed = True
        try:
            setattr(self.window, "_aidas_suppress_native_border", True)
        except (AttributeError, TypeError):
            pass
        return True

    def _rollback_failed_install(
        self,
        handle: int,
        original_style: int,
    ) -> None:
        """Best-effort rollback that never masks the original install failure."""

        if self._api is None:
            return
        try:
            self._api.set_style(handle, original_style)
            self._api.refresh_frame(handle)
        except Exception:
            pass

    def restore_native_caption(self) -> bool:
        """Restore the exact pre-install style, useful for reusable hosts."""

        if (
            not self.installed
            or self._api is None
            or self.handle is None
            or self.original_style is None
        ):
            return False
        try:
            restored = self._api.set_style(self.handle, self.original_style)
            refreshed = self._api.refresh_frame(self.handle)
        except Exception:
            return False
        if restored and refreshed:
            self.installed = False
            self.style = self.original_style
            try:
                setattr(self.window, "_aidas_suppress_native_border", False)
            except (AttributeError, TypeError):
                pass
            return True
        return False

    def begin_drag(self) -> None:
        if self.installed and self._api is not None and self.handle is not None:
            self._api.begin_caption_drag(self.handle)

    def resize_window(self, x: int, y: int, width: int, height: int) -> bool:
        """Apply a client-edge resize without reintroducing a native frame."""

        if not self.installed or self._api is None or self.handle is None:
            return False
        try:
            return self._api.resize_window(self.handle, x, y, width, height)
        except Exception:
            return False

    def minimize(self) -> None:
        if self.installed:
            try:
                self.window.iconify()
            except (AttributeError, tk.TclError):
                pass

    def toggle_maximize(self) -> None:
        if self.installed and self._api is not None and self.handle is not None:
            maximizing = not self._api.is_zoomed(self.handle)
            try:
                self.window.state("zoomed" if maximizing else "normal")
            except (AttributeError, tk.TclError):
                return
            if maximizing:
                self.correct_maximized_bounds()
                try:
                    self.window.after_idle(self.correct_maximized_bounds)
                except (AttributeError, tk.TclError):
                    pass

    def restore(self) -> None:
        """Restore safely through Tk instead of reentrant user32 ShowWindow."""

        if not self.installed:
            return
        try:
            if self.is_maximized():
                self.window.state("normal")
            self.window.deiconify()
        except (AttributeError, tk.TclError):
            pass

    def is_maximized(self) -> bool:
        if not self.installed or self._api is None or self.handle is None:
            return False
        return self._api.is_zoomed(self.handle)

    def correct_maximized_bounds(self) -> bool:
        """Constrain Windows' borderless zoom rectangle to monitor work area."""

        if (
            not self.installed
            or self._api is None
            or self.handle is None
            or not self._api.is_zoomed(self.handle)
        ):
            return False
        return self._api.fit_maximized_to_work_area(self.handle)

    def close(self) -> None:
        if self.installed and self._api is not None and self.handle is not None:
            self._api.close_window(self.handle)


class CustomWindowsTitleBar(ctk.CTkFrame):
    """Palette-aware client title bar backed by native window operations."""

    HEIGHT = 36
    _RESIZE_CURSORS = {
        "n": "sb_v_double_arrow",
        "s": "sb_v_double_arrow",
        "e": "sb_h_double_arrow",
        "w": "sb_h_double_arrow",
        "nw": "size_nw_se",
        "se": "size_nw_se",
        "ne": "size_ne_sw",
        "sw": "size_ne_sw",
    }

    def __init__(
        self,
        master: tk.Misc,
        *,
        controller: WindowsCaptionController,
        title: str,
        logo_path: str | None = None,
    ) -> None:
        super().__init__(
            master,
            height=self.HEIGHT,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["window_chrome"],
        )
        self.controller = controller
        self.pack_propagate(False)
        self._configure_binding: str | None = None
        self._state_sync_after_id: str | None = None
        self._resize_bindings: list[tuple[str, str]] = []
        self._resize_drag: dict[str, object] | None = None
        self._cursor_widget = None
        self._cursor_before_resize = ""

        drag_surface = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        drag_surface.pack(side="left", fill="both", expand=True)

        self.logo_image = None
        if logo_path and Path(logo_path).is_file():
            with Image.open(logo_path) as source:
                logo = source.convert("RGBA").copy()
            self.logo_image = ctk.CTkImage(
                light_image=logo,
                dark_image=logo,
                size=(18, 18),
            )
            logo_label = ctk.CTkLabel(
                drag_surface,
                text="",
                image=self.logo_image,
                width=24,
            )
            logo_label.pack(side="left", padx=(8, 2))
            self._bind_drag_surface(logo_label)

        self.title_label = ctk.CTkLabel(
            drag_surface,
            text=str(title),
            anchor="w",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.caption_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        )
        self.title_label.pack(side="left", fill="y", padx=(4, 8))
        self._bind_drag_surface(self)
        self._bind_drag_surface(drag_surface)
        self._bind_drag_surface(self.title_label)

        controls = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        controls.pack(side="right", fill="y")
        self.minimize_button = self._window_button(
            controls,
            text="\u2212",
            command=self.controller.minimize,
            font_size=17,
        )
        self.minimize_button.pack(side="left", fill="y")
        self.maximize_button = self._window_button(
            controls,
            text="\u25a1",
            command=self._toggle_maximize,
            font_size=17,
        )
        self.maximize_button.pack(side="left", fill="y")
        self.close_button = self._window_button(
            controls,
            text="\u00d7",
            command=self.controller.close,
            font_size=20,
            close=True,
        )
        self.close_button.pack(side="left", fill="y")

        self._configure_binding = master.bind(
            "<Configure>", self._queue_window_state_sync, add="+"
        )
        for sequence, callback in (
            ("<Motion>", self._track_resize_cursor),
            ("<ButtonPress-1>", self._begin_client_resize),
            ("<B1-Motion>", self._drag_client_resize),
            ("<ButtonRelease-1>", self._end_client_resize),
            ("<Escape>", self._cancel_client_resize),
        ):
            binding = master.bind(sequence, callback, add="+")
            if binding:
                self._resize_bindings.append((sequence, binding))
        self._sync_window_state()

    def _window_button(
        self,
        master: tk.Misc,
        *,
        text: str,
        command,
        font_size: int,
        close: bool = False,
    ) -> ctk.CTkButton:
        button = ctk.CTkButton(
            master,
            text=text,
            command=command,
            width=48,
            height=self.HEIGHT,
            corner_radius=0,
            border_width=0,
            fg_color="transparent",
            hover_color=(
                COLOR_PAIRS["danger_hover"]
                if close
                else COLOR_PAIRS["surface_subtle"]
            ),
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=font_size),
        )
        button.bind("<Return>", lambda _event: (command(), "break")[1], add="+")
        button.bind("<space>", lambda _event: (command(), "break")[1], add="+")
        return button

    def _bind_drag_surface(self, widget: tk.Misc) -> None:
        widget.bind("<ButtonPress-1>", self._begin_drag, add="+")
        widget.bind("<Double-Button-1>", self._toggle_maximize, add="+")

    def _begin_drag(self, _event=None) -> str:
        if _event is not None and self._start_client_resize(_event):
            return "break"
        self.controller.begin_drag()
        return "break"

    def _resize_edge_at(self, pointer_x: int, pointer_y: int) -> str | None:
        """Return the client edge under one screen-coordinate pointer."""

        if self.controller.is_maximized():
            return None
        try:
            left = int(self.master.winfo_rootx())
            top = int(self.master.winfo_rooty())
            width = max(1, int(self.master.winfo_width()))
            height = max(1, int(self.master.winfo_height()))
            scaler = getattr(self.master, "_apply_window_scaling", None)
            margin = (
                max(3, int(scaler(_CLIENT_RESIZE_MARGIN)))
                if callable(scaler)
                else _CLIENT_RESIZE_MARGIN
            )
        except (AttributeError, tk.TclError, TypeError, ValueError):
            return None

        relative_x = int(pointer_x) - left
        relative_y = int(pointer_y) - top
        if not (0 <= relative_x < width and 0 <= relative_y < height):
            return None
        vertical = ""
        if relative_y < margin:
            vertical = "n"
        elif relative_y >= height - margin:
            vertical = "s"
        horizontal = ""
        if relative_x < margin:
            horizontal = "w"
        elif relative_x >= width - margin:
            horizontal = "e"
        edge = vertical + horizontal
        return edge or None

    def _minimum_physical_size(self) -> tuple[int, int]:
        try:
            minimum_width, minimum_height = self.master.minsize()
        except (AttributeError, tk.TclError, TypeError):
            return 1, 1
        scaler = getattr(self.master, "_apply_window_scaling", None)
        if callable(scaler):
            return (
                max(1, int(scaler(minimum_width))),
                max(1, int(scaler(minimum_height))),
            )
        return max(1, int(minimum_width)), max(1, int(minimum_height))

    def _start_client_resize(self, event) -> bool:
        edge = self._resize_edge_at(event.x_root, event.y_root)
        if edge is None:
            return False
        try:
            self._resize_drag = {
                "edge": edge,
                "pointer": (int(event.x_root), int(event.y_root)),
                "rect": (
                    int(self.master.winfo_x()),
                    int(self.master.winfo_y()),
                    int(self.master.winfo_width()),
                    int(self.master.winfo_height()),
                ),
                "minimum": self._minimum_physical_size(),
            }
            self.master.grab_set()
        except (AttributeError, tk.TclError, TypeError, ValueError):
            self._resize_drag = None
            return False
        self._set_resize_cursor(getattr(event, "widget", None), edge)
        return True

    def _begin_client_resize(self, event=None):
        if event is not None and self._start_client_resize(event):
            return "break"
        return None

    def _drag_client_resize(self, event=None):
        drag = self._resize_drag
        if drag is None or event is None:
            return None
        pointer_x, pointer_y = drag["pointer"]
        minimum_width, minimum_height = drag["minimum"]
        target = _resized_window_rect(
            drag["rect"],
            str(drag["edge"]),
            int(event.x_root) - int(pointer_x),
            int(event.y_root) - int(pointer_y),
            int(minimum_width),
            int(minimum_height),
        )
        self.controller.resize_window(*target)
        return "break"

    def _end_client_resize(self, event=None):
        if self._resize_drag is None:
            return None
        self._resize_drag = None
        try:
            self.master.grab_release()
        except (AttributeError, tk.TclError):
            pass
        if event is not None:
            self._track_resize_cursor(event)
        return "break"

    def _cancel_client_resize(self, _event=None):
        if self._resize_drag is None:
            return None
        self._end_client_resize()
        self._set_resize_cursor(None, None)
        return "break"

    def _track_resize_cursor(self, event=None):
        if self._resize_drag is not None or event is None:
            return None
        edge = self._resize_edge_at(event.x_root, event.y_root)
        self._set_resize_cursor(getattr(event, "widget", None), edge)
        return None

    def _set_resize_cursor(self, widget, edge: str | None) -> None:
        desired = self._RESIZE_CURSORS.get(edge or "", "")
        if widget is self._cursor_widget and desired:
            try:
                widget.configure(cursor=desired)
            except (AttributeError, tk.TclError, ValueError):
                pass
            return
        if self._cursor_widget is not None:
            try:
                self._cursor_widget.configure(cursor=self._cursor_before_resize)
            except (AttributeError, tk.TclError, ValueError):
                pass
        self._cursor_widget = None
        self._cursor_before_resize = ""
        if widget is None or not desired:
            return
        try:
            previous = str(widget.cget("cursor"))
            widget.configure(cursor=desired)
        except (AttributeError, tk.TclError, ValueError):
            return
        self._cursor_widget = widget
        self._cursor_before_resize = previous

    def _toggle_maximize(self, _event=None) -> str:
        self.controller.toggle_maximize()
        self._queue_window_state_sync()
        return "break"

    def _sync_window_state(self, _event=None) -> None:
        self._state_sync_after_id = None
        try:
            self.controller.correct_maximized_bounds()
            self.maximize_button.configure(
                text="\u2750" if self.controller.is_maximized() else "\u25a1"
            )
        except tk.TclError:
            pass

    def _queue_window_state_sync(self, _event=None) -> None:
        # A toplevel's bind tag is also present on descendant widgets, so a
        # binding registered on the root can receive every child's Configure
        # event. Only root moves/resizes/state changes affect caption state.
        if _event is not None and getattr(_event, "widget", None) is not self.master:
            return
        if self._state_sync_after_id is not None:
            return
        try:
            self._state_sync_after_id = self.after_idle(self._sync_window_state)
        except tk.TclError:
            self._state_sync_after_id = None

    def set_title(self, title: str) -> None:
        self.title_label.configure(text=str(title))

    def destroy(self) -> None:
        self._cancel_client_resize()
        self._set_resize_cursor(None, None)
        if self._state_sync_after_id is not None:
            try:
                self.after_cancel(self._state_sync_after_id)
            except tk.TclError:
                pass
            self._state_sync_after_id = None
        if self._configure_binding:
            try:
                self.master.unbind("<Configure>", self._configure_binding)
            except tk.TclError:
                pass
            self._configure_binding = None
        try:
            for sequence, binding in self._resize_bindings:
                self.master.unbind(sequence, binding)
        except tk.TclError:
            pass
        self._resize_bindings.clear()
        super().destroy()


def reassert_client_size(
    window: tk.Misc,
    width: int,
    height: int,
    *,
    attempts: int = 3,
) -> tuple[int, int]:
    """Restore a requested Tk client size after ``SWP_FRAMECHANGED``.

    Caption removal changes how some Tk/Windows versions translate ``wm
    geometry`` to the client rectangle (commonly by one resize-border pixel on
    each side). Measuring and correcting the request is safer than assuming a
    fixed border size and naturally adapts to DPI scaling.
    """

    target_width = max(1, int(width))
    target_height = max(1, int(height))
    apply_scaling = getattr(window, "_apply_window_scaling", None)
    reverse_scaling = getattr(window, "_reverse_window_scaling", None)
    if callable(apply_scaling):
        target_physical_width = max(1, int(apply_scaling(target_width)))
        target_physical_height = max(1, int(apply_scaling(target_height)))
    else:
        target_physical_width = target_width
        target_physical_height = target_height
    requested_width = target_width
    requested_height = target_height
    actual_width = max(1, int(window.winfo_width()))
    actual_height = max(1, int(window.winfo_height()))
    for _ in range(max(1, int(attempts))):
        window.geometry(f"{requested_width}x{requested_height}")
        window.update_idletasks()
        actual_width = max(1, int(window.winfo_width()))
        actual_height = max(1, int(window.winfo_height()))
        width_error = target_physical_width - actual_width
        height_error = target_physical_height - actual_height
        # CTk truncates scaled geometry to integer physical pixels. A one-pixel
        # remainder can therefore be unrepresentable at fractional DPI scales.
        if abs(width_error) <= 1 and abs(height_error) <= 1:
            break

        def logical_correction(physical_error: int) -> int:
            if abs(physical_error) <= 1:
                return 0
            if callable(reverse_scaling):
                correction = int(reverse_scaling(abs(physical_error)))
            else:
                correction = abs(physical_error)
            correction = max(1, correction)
            return correction if physical_error > 0 else -correction

        requested_width = max(
            1, requested_width + logical_correction(width_error)
        )
        requested_height = max(
            1, requested_height + logical_correction(height_error)
        )
    return actual_width, actual_height


def create_custom_windows_title_bar(
    window: tk.Misc,
    *,
    title: str,
    logo_path: str | None = None,
) -> CustomWindowsTitleBar | None:
    """Install and build the title bar, or leave the native caption intact."""

    controller = WindowsCaptionController(window)
    if not controller.install():
        return None
    try:
        return CustomWindowsTitleBar(
            window,
            controller=controller,
            title=title,
            logo_path=logo_path,
        )
    except Exception:
        controller.restore_native_caption()
        raise


__all__ = [
    "CustomWindowsTitleBar",
    "WindowsCaptionController",
    "create_custom_windows_title_bar",
    "reassert_client_size",
]
