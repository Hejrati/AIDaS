"""An owned, click-through arrow between controls in different widgets."""

from __future__ import annotations

import ctypes
import math
import sys
import tkinter as tk


class _WindowsOverlay:
    """Keep the arrow out of activation, hit testing, and the taskbar."""

    FLAGS = 0x00080000 | 0x00000020 | 0x08000000 | 0x00000080

    def __init__(self, window):
        # Style and position calls can synchronously re-enter Tk handlers.
        self.user32 = ctypes.PyDLL("user32", use_last_error=True)
        ancestor = self.user32.GetAncestor
        ancestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        ancestor.restype = ctypes.c_void_p
        self.handle = ancestor(int(window.winfo_id()), 2)
        if not self.handle:
            raise OSError("The guidance window has no native wrapper")
        try:
            self.get_style = self.user32.GetWindowLongPtrW
            self.set_style = self.user32.SetWindowLongPtrW
            result_type = ctypes.c_ssize_t
        except AttributeError:  # 32-bit Windows
            self.get_style = self.user32.GetWindowLongW
            self.set_style = self.user32.SetWindowLongW
            result_type = ctypes.c_long
        self.get_style.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self.get_style.restype = result_type
        self.set_style.argtypes = (ctypes.c_void_p, ctypes.c_int, result_type)
        self.set_style.restype = result_type
        self.set_position = self.user32.SetWindowPos
        self.set_position.argtypes = (
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint,
        )
        self.set_position.restype = ctypes.c_bool
        style = int(self.get_style(self.handle, -20))
        self.set_style(self.handle, -20, (style | self.FLAGS) & ~0x00040000)
        if int(self.get_style(self.handle, -20)) & self.FLAGS != self.FLAGS:
            raise OSError("Unable to make guidance click-through")

    def position(self, x, y, width, height):
        # Native coordinates also work on monitors left/above the main screen;
        # negative Tk geometry offsets instead count from the opposite edge.
        flags = 0x0010 | 0x0004 | 0x0200  # NOACTIVATE, NOZORDER, NOOWNERZORDER
        if not self.set_position(self.handle, None, x, y, width, height, flags):
            raise OSError("Unable to position guidance")


class GuidanceArrow(tk.Toplevel):
    """Draw a temporary arrow using exact screen-coordinate endpoints.

    The caller controls its lifetime and updates its endpoints when controls or
    image contents move. Window moves retain the most recent owner-relative
    endpoints. Unsupported platforms leave the arrow hidden.
    """

    _TRANSPARENT = "#010203"
    _MARGIN = 14

    def __init__(self, master):
        super().__init__(master, takefocus=False)
        self.withdraw()
        self._destroyed = False
        self._owner = master.winfo_toplevel()
        self._native = None
        self._endpoints = None
        self._color = "#ffd500"
        self._last_drawing = None
        self._last_size = None
        self._rendering = False
        self._bindings = []
        self.overrideredirect(True)
        self.transient(self._owner)
        self.configure(background=self._TRANSPARENT)
        self.canvas = tk.Canvas(
            self, background=self._TRANSPARENT, borderwidth=0,
            highlightthickness=0, takefocus=False,
        )
        self.canvas.pack(fill="both", expand=True)
        self._arrow = self.canvas.create_line(
            0, 0, 0, 0, 0, 0, smooth=True, splinesteps=16,
            arrow=tk.LAST, arrowshape=(10, 12, 5), width=2,
            fill=self._color,
        )
        if sys.platform.startswith("win"):
            try:
                self.attributes("-transparentcolor", self._TRANSPARENT)
                self.update_idletasks()
                self._native = _WindowsOverlay(self)
            except (tk.TclError, OSError, AttributeError):
                self._native = None
        for sequence, callback in (
            ("<Configure>", self._on_owner_configure),
            ("<Unmap>", self._on_owner_unmap),
        ):
            binding = tk.Misc.bind(self._owner, sequence, callback, add="+")
            self._bindings.append((self._owner, sequence, binding))
        if master is not self._owner:
            binding = tk.Misc.bind(master, "<Unmap>", self._on_master_unmap, add="+")
            self._bindings.append((master, "<Unmap>", binding))

    @property
    def supported(self):
        return self._native is not None and not self._destroyed

    def show(self, start_rootxy, end_rootxy, *, color="#ffd500"):
        """Show an arrow, returning whether click-through drawing is supported."""
        if not self.supported:
            return False
        values = tuple(float(value) for point in (start_rootxy, end_rootxy) for value in point)
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            self.hide()
            return False
        owner_x, owner_y = self._owner.winfo_rootx(), self._owner.winfo_rooty()
        self._endpoints = (
            values[0] - owner_x, values[1] - owner_y,
            values[2] - owner_x, values[3] - owner_y,
        )
        self._color = color
        return self._render()

    def _render(self):
        if self._rendering or self._endpoints is None or not self.supported:
            return False
        if not self._owner.winfo_viewable():
            self.hide()
            return False
        self._rendering = True
        try:
            owner_x, owner_y = self._owner.winfo_rootx(), self._owner.winfo_rooty()
            x1, y1, x2, y2 = self._endpoints
            x1, x2 = x1 + owner_x, x2 + owner_x
            y1, y2 = y1 + owner_y, y2 + owner_y
            drawing = (x1, y1, x2, y2, self._color)
            if drawing == self._last_drawing and self.winfo_ismapped():
                return True
            left = math.floor(min(x1, x2)) - self._MARGIN
            top = math.floor(min(y1, y2)) - self._MARGIN
            width = math.ceil(max(x1, x2)) - left + self._MARGIN
            height = math.ceil(max(y1, y2)) - top + self._MARGIN
            bend_y = y2 + (y1 - y2) * 0.35
            self.canvas.coords(
                self._arrow, x1 - left, y1 - top,
                x1 - left, bend_y - top, x2 - left, y2 - top,
            )
            self.canvas.itemconfigure(self._arrow, fill=self._color)
            if (width, height) != self._last_size:
                self.geometry(f"{width}x{height}")
                self.update_idletasks()
                self._last_size = (width, height)
            self._native.position(left, top, width, height)
            if not self.winfo_ismapped():
                # Tk uses SHOWNOACTIVATE for an overrideredirect window.
                self.deiconify()
            self._native.position(left, top, width, height)
            self._last_drawing = drawing
            return True
        except (tk.TclError, OSError):
            self.hide()
            return False
        finally:
            self._rendering = False

    def hide(self):
        self._endpoints = None
        if not self._destroyed:
            try:
                self.withdraw()
            except tk.TclError:
                pass

    def _on_owner_configure(self, event):
        if event.widget is self._owner:
            self._render()

    def _on_owner_unmap(self, event):
        if event.widget is self._owner:
            self.hide()

    def _on_master_unmap(self, event):
        if event.widget is self.master:
            self.hide()

    def destroy(self):
        if self._destroyed:
            return
        self._destroyed = True
        self._endpoints = None
        for widget, sequence, binding in self._bindings:
            try:
                tk.Misc.unbind(widget, sequence, binding)
            except tk.TclError:
                pass
        self._bindings.clear()
        super().destroy()
