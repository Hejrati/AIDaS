"""Tutorial scrolling keeps its native contract while coalescing paints."""

from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.ui.scrolling import DeferredScrollbar
from aidas.ui.tutorial import _TutorialScrollableFrame


class TutorialScrollbarTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.attributes("-alpha", 0)
        self.root.geometry("600x400+0+0")
        self.addCleanup(self.root.destroy)

    def test_native_grid_and_scrollbar_bounds_match_at_fractional_scales_and_after_transition(self):
        previous_scaling = ctk.ScalingTracker.widget_scaling
        self.addCleanup(ctk.set_widget_scaling, previous_scaling)
        self.root.geometry("800x400+0+0")
        self.root.grid_columnconfigure(0, weight=1, uniform="panels")
        self.root.grid_columnconfigure(1, weight=1, uniform="panels")
        self.root.grid_rowconfigure(0, weight=1)

        def compare(original, optimized):
            self.root.update()
            before, after = original._scrollbar, optimized._scrollbar
            before_grid = {key: value for key, value in before.grid_info().items() if key != "in"}
            after_grid = {key: value for key, value in after.grid_info().items() if key != "in"}
            self.assertEqual(after_grid, before_grid)
            self.assertEqual(
                (after.winfo_x(), after.winfo_y(), after.winfo_width(), after.winfo_height()),
                (before.winfo_x(), before.winfo_y(), before.winfo_width(), before.winfo_height()),
            )

        for scaling in (1.25, 1.5, 2.0):
            with self.subTest(scaling=scaling):
                ctk.set_widget_scaling(scaling)
                options = dict(width=220, height=180, corner_radius=7, border_width=2)
                original = ctk.CTkScrollableFrame(self.root, **options)
                optimized = _TutorialScrollableFrame(self.root, **options)
                original.grid(row=0, column=0, sticky="nsew")
                optimized.grid(row=0, column=1, sticky="nsew")
                try:
                    compare(original, optimized)
                    if scaling == 1.25:
                        ctk.set_widget_scaling(1.5)
                        compare(original, optimized)
                finally:
                    original.destroy()
                    optimized.destroy()

    def test_scrollbar_preserves_native_options_fractions_and_wheel_distance(self):
        options = dict(width=360, height=140, corner_radius=7, border_width=2,
            scrollbar_fg_color="#123456", scrollbar_button_color="#345678",
            scrollbar_button_hover_color="#56789a")
        original = ctk.CTkScrollableFrame(self.root, **options)
        optimized = _TutorialScrollableFrame(self.root, **options)
        original.pack(side="left", fill="both", expand=True)
        optimized.pack(side="left", fill="both", expand=True)
        self.root.update()
        before, after = original._scrollbar, optimized._scrollbar
        self.assertIsInstance(after, DeferredScrollbar)
        for name in ("width", "height", "corner_radius", "border_spacing", "minimum_pixel_length",
                     "bg_color", "fg_color", "button_color", "button_hover_color", "hover", "orientation"):
            with self.subTest(option=name):
                self.assertEqual(after.cget(name), before.cget(name))
        for values in ((0.0, 1.0), (0.1, 0.6), (0.5, 1.0)):
            before.set(*values)
            after.set(*values)
            self.assertEqual(after.get(), before.get())

        for panel in (original, optimized):
            canvas = panel._parent_canvas
            canvas.configure(scrollregion=(0, 0, 250, 2000), yscrollincrement=10)
            canvas.yview_moveto(0.3)
        self.root.update()
        for panel in (original, optimized):
            panel._scrollbar._mouse_scroll_event(SimpleNamespace(delta=-120))
        self.assertEqual(optimized._parent_canvas.yview(), original._parent_canvas.yview())

    def test_destroy_cancels_the_queued_thumb_paint(self):
        panel = _TutorialScrollableFrame(self.root)
        scrollbar = panel._scrollbar
        scrollbar.set(0.1, 0.8)
        pending = scrollbar._redraw_after_id
        self.assertIsNotNone(pending)
        self.assertIn(pending, self.root.tk.call("after", "info"))

        panel.destroy()

        self.assertNotIn(pending, self.root.tk.call("after", "info"))


if __name__ == "__main__":
    unittest.main()
