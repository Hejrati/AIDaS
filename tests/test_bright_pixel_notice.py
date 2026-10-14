from __future__ import annotations

import tkinter as tk
import unittest
from unittest.mock import Mock

import customtkinter as ctk

from aidas.ui.bright_pixel_notice import BrightPixelNotice
from aidas.ui.theme import (
    COLOR_PAIRS,
    apply_appearance_mode,
    get_interface_mode,
    resolve_color,
    set_interface_mode,
)


class BrightPixelNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = None
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
            cls.root.attributes("-alpha", 0)
            cls.root.geometry("400x320+10000+10000")
            cls.root.deiconify()
        except tk.TclError as exc:
            if cls.root is not None:
                cls.root.destroy()
                cls.root = None
            raise unittest.SkipTest(f"A transparent Tk test window is unavailable: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        if cls.root is not None:
            cls.root.destroy()
            cls.root = None

    def setUp(self):
        self.parent = tk.Canvas(self.root, highlightthickness=0, borderwidth=0)
        self.parent.pack(fill="both", expand=True)

    def tearDown(self):
        if self.parent.winfo_exists():
            self.parent.destroy()

    def _notice(self, callback):
        notice = BrightPixelNotice(
            self.parent, count=4, context="SDB image", on_decision=callback
        )
        notice.place(x=0, y=0, relwidth=1)
        self.root.update()
        self.assertEqual(notice.winfo_rootx(), self.parent.winfo_rootx())
        self.assertEqual(notice.winfo_rooty(), self.parent.winfo_rooty())
        self.assertEqual(notice.winfo_width(), self.parent.winfo_width())
        return notice

    def test_buttons_resolve_each_choice_once(self):
        for name, decision in (
            ("fix_button", True),
            ("cancel_button", None),
        ):
            with self.subTest(button=name):
                callback = Mock()
                notice = self._notice(callback)
                self.assertEqual(notice.fix_button.cget("text"), "Fix")
                self.assertEqual(notice.cancel_button.cget("text"), "Cancel")
                self.assertFalse(hasattr(notice, "smooth_button"))
                self.assertFalse(hasattr(notice, "keep_button"))
                getattr(notice, name).invoke()
                notice._finish(None)
                notice.destroy()
                callback.assert_called_once_with(decision)

    def test_destroying_parent_cancels_an_unanswered_choice(self):
        callback = Mock()
        self._notice(callback)
        self.parent.destroy()
        callback.assert_called_once_with(None)

    def test_notice_is_embedded_and_escape_cancels(self):
        callback = Mock()
        notice = self._notice(callback)
        self.assertIs(notice.master, self.parent)
        self.assertEqual(notice.winfo_toplevel(), self.root)
        self.assertIsNone(self.root.grab_current())
        self.root.focus_force()
        notice.focus_default()
        self.root.update()
        focused = self.root.focus_get()
        self.assertTrue(
            focused is notice.cancel_button or str(focused).startswith(f"{notice.cancel_button}.")
        )
        notice.cancel_button._text_label.focus_force()
        self.root.update()
        notice.cancel_button._text_label.event_generate("<Escape>")
        self.root.update()
        callback.assert_called_once_with(None)

    def test_colors_follow_modes_and_actions_fit_a_narrow_canvas(self):
        original_interface = get_interface_mode()
        original_appearance = ctk.get_appearance_mode()
        original_source = ctk.AppearanceModeTracker.appearance_mode_set_by
        try:
            notice = self._notice(Mock())
            for interface, appearance in (
                ("Modern", "Light"),
                ("Modern", "Dark"),
                ("Classic", "Dark"),
            ):
                with self.subTest(interface=interface, appearance=appearance):
                    set_interface_mode(interface)
                    apply_appearance_mode(appearance, root=self.root)
                    self.root.update()
                    self.assertEqual(
                        notice.cget("background"), resolve_color(COLOR_PAIRS["warning_soft"])
                    )
                    self.assertEqual(
                        notice._body.cget("foreground"), resolve_color(COLOR_PAIRS["text"])
                    )
                    self.assertEqual(
                        resolve_color(notice.fix_button.cget("fg_color")), resolve_color(COLOR_PAIRS["success"])
                    )
                    self.assertEqual(
                        resolve_color(notice.cancel_button.cget("fg_color")), resolve_color(COLOR_PAIRS["danger"])
                    )
                    self.assertEqual(notice.fix_button.winfo_rooty(), notice.cancel_button.winfo_rooty())
                    self.assertLess(
                        notice.fix_button.winfo_rootx() + notice.fix_button.winfo_width(),
                        notice.cancel_button.winfo_rootx(),
                    )
                    self.assertLessEqual(
                        notice.cancel_button.winfo_rootx() + notice.cancel_button.winfo_width(),
                        notice.winfo_rootx() + notice.winfo_width() - 12,
                    )
                    self.assertLessEqual(int(notice._body.cget("wraplength")), notice.winfo_width())
                    self.assertLess(notice.winfo_height(), self.parent.winfo_height())
        finally:
            set_interface_mode(original_interface)
            ctk.set_appearance_mode("System" if original_source == "system" else original_appearance)


if __name__ == "__main__":
    unittest.main()
