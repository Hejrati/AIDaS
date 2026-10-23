from __future__ import annotations

import tkinter as tk
import unittest
from unittest.mock import Mock, patch

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
        notice.show_modal()
        self.root.update()
        self.assertIs(self.root.grab_current(), notice)
        return notice

    def test_buttons_resolve_each_choice_once(self):
        for name, decision in (
            ("fix_button", True),
            ("ignore_button", False),
        ):
            with self.subTest(button=name):
                callback = Mock()
                notice = self._notice(callback)
                self.assertEqual(notice.fix_button.cget("text"), "Fix")
                self.assertEqual(notice.ignore_button.cget("text"), "Ignore")
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

    def test_notice_is_modal_and_escape_cancels(self):
        callback = Mock()
        notice = self._notice(callback)
        self.assertIs(notice.master, self.parent)
        self.assertIs(notice.winfo_toplevel(), notice)
        self.assertEqual(str(notice.transient()), str(self.root))
        self.assertIs(self.root.grab_current(), notice)
        notice.focus_default()
        self.root.update()
        focused = self.root.focus_get()
        self.assertTrue(
            focused is notice.ignore_button or str(focused).startswith(f"{notice.ignore_button}.")
        )
        notice.ignore_button._text_label.focus_force()
        self.root.update()
        notice.ignore_button._text_label.event_generate("<Escape>")
        self.root.update()
        callback.assert_called_once_with(None)
        notice.destroy()
        self.assertIsNone(self.root.grab_current())

    def test_window_close_ignores_without_remembering_and_releases_modal_grab(self):
        callback = Mock()
        notice = self._notice(callback)
        notice.auto_fix_checkbox.toggle()
        self.root.tk.call(notice.protocol("WM_DELETE_WINDOW"))
        callback.assert_called_once_with(False)
        self.assertFalse(notice.auto_fix_var.get())
        notice.destroy()
        self.assertIsNone(self.root.grab_current())

    def test_remember_choice_tooltip_explains_how_to_restore_confirmation(self):
        notice = self._notice(Mock())
        self.assertEqual(notice.auto_fix_checkbox.cget("text"), "Remember my choice")
        tooltip = notice._remember_tooltip
        self.assertIn("Settings > Step 1: SDB Parameters", tooltip.text)
        self.assertIn("Fix or Ignore", tooltip.text)
        with patch.object(tooltip, "_pointer_is_over_widget", return_value=True):
            tooltip._show_now()
            self.root.update()
            self.assertIsNotNone(tooltip.tipwindow)
            self.assertIs(self.root.grab_current(), notice)
        notice.destroy()
        self.assertIsNone(tooltip.tipwindow)

    def test_popup_uses_default_native_window_decorations(self):
        notice = self._notice(Mock())
        self.assertFalse(notice.overrideredirect())
        self.assertTrue(notice.protocol("WM_DELETE_WINDOW"))
        notice.destroy()

    def test_colors_follow_modes_and_actions_fit_a_narrow_popup(self):
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
                        notice.cget("background"), resolve_color(COLOR_PAIRS["surface_elevated"])
                    )
                    self.assertEqual(
                        notice._body.cget("foreground"), resolve_color(COLOR_PAIRS["text"])
                    )
                    self.assertEqual(
                        resolve_color(notice.fix_button.cget("fg_color")), resolve_color(COLOR_PAIRS["primary"])
                    )
                    self.assertEqual(
                        resolve_color(notice.ignore_button.cget("fg_color")), resolve_color(COLOR_PAIRS["button"])
                    )
                    self.assertEqual(notice._icon.cget("foreground"), resolve_color(COLOR_PAIRS["danger"]))
                    self.assertEqual(notice._title.cget("foreground"), resolve_color(COLOR_PAIRS["text"]))
                    checkbox_center = notice.auto_fix_checkbox.winfo_rooty() + notice.auto_fix_checkbox.winfo_height() / 2
                    button_center = notice.fix_button.winfo_rooty() + notice.fix_button.winfo_height() / 2
                    self.assertLessEqual(abs(checkbox_center - button_center), 1)
                    self.assertLess(
                        notice.auto_fix_checkbox.winfo_rootx() + notice.auto_fix_checkbox.winfo_width(),
                        notice.fix_button.winfo_rootx(),
                    )
                    self.assertEqual(notice.fix_button.winfo_rooty(), notice.ignore_button.winfo_rooty())
                    self.assertLess(
                        notice.fix_button.winfo_rootx() + notice.fix_button.winfo_width(),
                        notice.ignore_button.winfo_rootx(),
                    )
                    self.assertLessEqual(
                        notice.ignore_button.winfo_rootx() + notice.ignore_button.winfo_width(),
                        notice.winfo_rootx() + notice.winfo_width() - 12,
                    )
                    self.assertLessEqual(int(notice._body.cget("wraplength")), notice.winfo_width())
                    self.assertEqual(notice._actions.grid_info()["sticky"], "e")
                    self.assertLessEqual(
                        notice.ignore_button.winfo_rooty() + notice.ignore_button.winfo_height(),
                        notice.winfo_rooty() + notice.winfo_height() - 12,
                    )
        finally:
            set_interface_mode(original_interface)
            ctk.set_appearance_mode("System" if original_source == "system" else original_appearance)


if __name__ == "__main__":
    unittest.main()
