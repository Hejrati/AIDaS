from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import unittest

import customtkinter as ctk

from aidas.ui.scrolling_text import ScrollingText


class ScrollingTextTests(unittest.TestCase):
    def setUp(self):
        self.scaling = ctk.ScalingTracker.widget_scaling
        self.root = ctk.CTk()
        self.root.withdraw()
        self.root.attributes("-alpha", 0)
        self.root.geometry("840x140+0+0")
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(fill="x")
        self.actions = ttk.Frame(self.toolbar)
        self.actions.pack(side="right", fill="y")
        for label in ("Set Fovea", "Exit"):
            ctk.CTkButton(self.actions, text=label, width=120).pack(side="right", padx=4)
        self.value = tk.StringVar(value="Specify fovea (1/5): C:/" + "long folder/" * 30 + "Light.img")
        self.prompt = ScrollingText(self.toolbar, textvariable=self.value, padding=4)
        self.prompt.pack(side="left", fill="x", expand=True)
        self.root.deiconify()
        self.root.update()

    def tearDown(self):
        ctk.set_widget_scaling(self.scaling)
        self.root.destroy()

    def test_path_stays_clipped_before_buttons_at_each_scale(self):
        for scale in (1, 1.25, 1.5, 2):
            ctk.set_widget_scaling(scale)
            self.root.update()
            with self.subTest(scale=scale):
                self.assertLessEqual(self.prompt.winfo_rootx() + self.prompt.winfo_width(),
                                     self.actions.winfo_rootx())
                self.assertEqual(self.prompt._canvas.winfo_width(), self.prompt.winfo_width())
                self.assertGreater(self.prompt._overflow, 0)
                self.assertIsNotNone(self.prompt._scroll_job)

    def test_scroll_reaches_both_ends_and_reverses(self):
        prompt = self.prompt
        prompt._stop()
        prompt._offset = prompt._overflow - 1
        prompt._tick()
        self.assertEqual(prompt._offset, prompt._overflow)
        self.assertEqual(prompt._direction, -1)
        right = prompt._canvas.bbox(prompt._text_id)[2]
        self.assertLessEqual(right, prompt.winfo_width())
        prompt._stop()
        prompt._tick()
        self.assertLess(prompt._offset, prompt._overflow)
        prompt._stop()
        prompt._offset = 1
        prompt._tick()
        self.assertEqual(prompt._offset, 0)
        self.assertEqual(prompt._direction, 1)

    def test_animation_stops_when_short_hidden_or_destroyed_and_new_paths_restart(self):
        self.prompt.pack_forget()
        self.root.update()
        self.assertIsNone(self.prompt._scroll_job)
        self.prompt.pack(side="left", fill="x", expand=True)
        self.root.update()
        self.assertIsNotNone(self.prompt._scroll_job)
        self.prompt._offset = 100
        self.value.set("Another " + "long folder/" * 30)
        self.assertEqual(self.prompt._offset, 0)
        self.value.set("Specify fovea")
        self.assertIsNone(self.prompt._scroll_job)
        self.assertEqual(self.prompt._offset, 0)
        self.value.set("Long address " * 100)
        job = self.prompt._scroll_job
        self.assertIsNotNone(job)
        self.prompt.destroy()
        self.assertNotIn(job, self.root.tk.call("after", "info"))
        self.value.set("The old widget must no longer observe this variable")


if __name__ == "__main__":
    unittest.main()
