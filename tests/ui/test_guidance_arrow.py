"""Native guidance across sibling widgets without changing mouse behavior."""

import ctypes
from ctypes import wintypes
import sys
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock, patch

from aidas.ui.guidance_arrow import GuidanceArrow


@unittest.skipUnless(sys.platform.startswith("win"), "Windows layered-window behavior")
class GuidanceArrowTests(unittest.TestCase):
    def setUp(self):
        self.root = None
        try:
            self.root = tk.Tk()
            self.root.withdraw()
            self.root.geometry("360x240+80+80")
            self.root.attributes("-alpha", 0.01)
            self.root.deiconify()
            self.entry = ttk.Entry(self.root)
            self.entry.pack(pady=12)
            self.button = ttk.Button(self.root, text="Set Fovea")
            self.button.pack(pady=8)
            self.root.update()
        except tk.TclError as exc:
            if self.root is not None:
                self.root.destroy()
            self.skipTest(str(exc))
        self.addCleanup(self.root.destroy)
        self.entry.focus_force()
        self.root.update()
        self.arrow = GuidanceArrow(self.root)
        self.addCleanup(self.arrow.destroy)
        self.assertTrue(self.arrow.supported)
        self.root.update()
        self.assertEqual(self.root.focus_get(), self.entry)

    def _endpoints(self):
        start = (self.root.winfo_rootx() + 31.25, self.root.winfo_rooty() + 195.75)
        end = (
            self.button.winfo_rootx() + self.button.winfo_width() / 2,
            self.button.winfo_rooty() + self.button.winfo_height() / 2,
        )
        return start, end

    def _drawn_root_endpoints(self):
        coordinates = self.arrow.canvas.coords(self.arrow._arrow)
        root_x = self.arrow.canvas.winfo_rootx()
        root_y = self.arrow.canvas.winfo_rooty()
        return (
            (coordinates[0] + root_x, coordinates[1] + root_y),
            (coordinates[-2] + root_x, coordinates[-1] + root_y),
        )

    def test_exact_subpixel_origin_and_button_center_survive_owner_movement(self):
        start, end = self._endpoints()
        self.assertTrue(self.arrow.show(start, end))
        self.root.update()
        self.assertEqual(self._drawn_root_endpoints(), (start, end))
        before = (self.root.winfo_rootx(), self.root.winfo_rooty())
        self.root.geometry("360x240+140+160")
        self.root.update()
        delta = (self.root.winfo_rootx() - before[0], self.root.winfo_rooty() - before[1])
        shifted = tuple((x + delta[0], y + delta[1]) for x, y in (start, end))
        self.assertEqual(self._drawn_root_endpoints(), shifted)

    def test_visible_arrow_pixels_pass_native_hit_testing_to_the_button(self):
        start, end = self._endpoints()
        self.arrow.show(start, end)
        self.root.update()
        native = self.arrow._native
        style = int(native.get_style(native.handle, -20))
        self.assertEqual(style & 0x08080020, 0x08080020)
        self.assertFalse(style & 0x00000008, "Guidance must not be globally topmost")
        hit_test = native.user32.WindowFromPoint
        hit_test.argtypes = (wintypes.POINT,)
        hit_test.restype = ctypes.c_void_p
        hit = hit_test(wintypes.POINT(round(end[0]), round(end[1])))
        self.assertEqual(hit, self.button.winfo_id())

    def test_show_hide_and_cleanup_preserve_focus_bindings_and_timer_state(self):
        self.entry.focus_force()
        self.root.update()
        before_focus = self.root.focus_get()
        get_foreground = self.arrow._native.user32.GetForegroundWindow
        get_foreground.argtypes = ()
        get_foreground.restype = ctypes.c_void_p
        before_foreground = get_foreground()
        before_jobs = self.root.tk.call("after", "info")
        handler = Mock()
        binding = self.root.bind("<Configure>", handler, add="+")
        self.arrow.show(*self._endpoints())
        self.root.update()
        self.assertEqual(self.root.focus_get(), before_focus)
        self.assertEqual(get_foreground(), before_foreground)
        self.assertEqual(self.root.tk.call("after", "info"), before_jobs)
        self.assertIsNone(self.root.grab_current())
        self.arrow.hide()
        self.root.update()
        self.assertFalse(self.arrow.winfo_ismapped())
        self.arrow.destroy()
        self.arrow.destroy()
        self.assertIn(binding, self.root.bind("<Configure>"))
        self.root.geometry("360x240+180+180")
        self.root.update()
        self.assertTrue(handler.called)

    def test_owner_unmap_hides_the_arrow_and_does_not_reopen_it_on_restore(self):
        self.arrow.show(*self._endpoints())
        self.root.update()
        self.assertTrue(self.arrow.winfo_ismapped())
        self.root.withdraw()
        self.root.update()
        self.assertFalse(self.arrow.winfo_ismapped())
        self.root.deiconify()
        self.root.update()
        self.assertFalse(self.arrow.winfo_ismapped())

    def test_unsupported_platform_never_shows_an_opaque_window(self):
        with patch("aidas.ui.guidance_arrow.sys.platform", "linux"):
            unsupported = GuidanceArrow(self.root)
        self.addCleanup(unsupported.destroy)
        self.assertFalse(unsupported.supported)
        self.assertFalse(unsupported.show(*self._endpoints()))
        self.root.update()
        self.assertFalse(unsupported.winfo_ismapped())

    def test_hiding_the_image_workspace_hides_the_arrow_while_owner_stays_visible(self):
        workspace = tk.Canvas(self.root, width=150, height=40)
        workspace.pack()
        self.root.update()
        arrow = GuidanceArrow(workspace)
        self.addCleanup(arrow.destroy)
        handler = Mock()
        binding = workspace.bind("<Unmap>", handler, add="+")
        arrow.show(*self._endpoints())
        self.root.update()
        self.assertTrue(arrow.winfo_ismapped())
        workspace.pack_forget()
        self.root.update()
        self.assertTrue(self.root.winfo_viewable())
        self.assertFalse(arrow.winfo_ismapped())
        workspace.pack()
        self.root.update()
        self.assertFalse(arrow.winfo_ismapped())
        arrow.destroy()
        self.assertIn(binding, workspace.bind("<Unmap>"))


if __name__ == "__main__":
    unittest.main()
