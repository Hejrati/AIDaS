"""Mouse and asynchronous action feedback for the shared CTk button."""

import tkinter as tk
import unittest
from unittest.mock import Mock

import customtkinter as ctk
from PIL import Image

from aidas.ui.components import AppButton


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class ButtonPressFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.previous_appearance = ctk.get_appearance_mode()
        self.previous_source = ctk.AppearanceModeTracker.appearance_mode_set_by
        ctk.set_appearance_mode("Light")
        self.addCleanup(self._restore_appearance)
        try:
            self.root = _QuietRoot()
            self.root.geometry("500x240")
            self.root.attributes("-alpha", 0)
        except tk.TclError as error:
            self.skipTest(f"A transparent Tk test window is unavailable: {error}")
        self.addCleanup(self.root.destroy)
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)

    def _restore_appearance(self):
        ctk.set_appearance_mode(self.previous_appearance)
        if self.previous_source == "system":
            ctk.set_appearance_mode("System")

    def _button(self, **kwargs):
        button = AppButton(self.root, text="Next", raised=True, **kwargs)
        button.pack()
        self.root.update()
        button._on_leave()
        return button

    def _fill(self, button):
        return button._canvas.itemcget("inner_parts", "fill")

    def _bevel(self, button):
        return tuple(
            (button._canvas.itemcget(item, "fill"), button._canvas.itemcget(item, "width"))
            for item in button._canvas.find_withtag("aidas_depth")
        )

    def _click(self, button, *, child=None):
        child = child if child is not None else button._canvas
        child.event_generate("<Enter>")
        child.event_generate("<ButtonPress-1>", x=5, y=5)
        child.event_generate("<ButtonRelease-1>", x=5, y=5)
        self.root.update()

    def test_mouse_down_visibly_depresses_and_mouse_up_invokes_once(self):
        command = Mock()
        button = self._button(command=command)
        button._canvas.event_generate("<Enter>")
        hovering_fill = self._fill(button)
        raised_bevel = self._bevel(button)
        button._canvas.event_generate("<ButtonPress-1>", x=5, y=5)
        self.assertNotEqual(self._fill(button), hovering_fill)
        self.assertNotEqual(self._bevel(button), raised_bevel)
        command.assert_not_called()
        button._canvas.event_generate("<ButtonRelease-1>", x=5, y=5)
        self.assertFalse(button._is_visually_pressed())
        command.assert_called_once_with()
        self.assertEqual(self.callback_errors, [])

    def test_async_command_holds_feedback_across_release_disable_leave_and_click_timer(self):
        calls = []

        def load_next():
            calls.append("load")
            button.set_pressed(True)
            button.configure(state="disabled")

        button = self._button(command=load_next)
        self._click(button)
        self.assertEqual(calls, ["load"])
        self.assertEqual(button.cget("state"), "disabled")
        self.assertTrue(button._is_visually_pressed())
        held_fill = self._fill(button)
        held_bevel = self._bevel(button)
        button._on_leave()
        button._click_animation()
        button.configure(text="Loading…")
        self.assertEqual(self._fill(button), held_fill)
        self.assertEqual(self._bevel(button), held_bevel)
        button.set_pressed(False)
        self.assertFalse(button._is_visually_pressed())
        self.assertNotEqual(self._fill(button), held_fill)
        self.assertEqual(self._bevel(button), ())
        self.assertEqual(self.callback_errors, [])

    def test_release_restores_current_enabled_palette_and_raised_bevel(self):
        button = self._button()
        normal_fill = self._fill(button)
        normal_bevel = self._bevel(button)
        enabled_palette = dict(button._enabled_palette)
        button.set_pressed(True)
        button.configure(state="disabled")
        button.configure(state="normal")
        button.set_pressed(False)
        self.assertEqual(self._fill(button), normal_fill)
        self.assertEqual(self._bevel(button), normal_bevel)
        self.assertEqual(button._enabled_palette, enabled_palette)

    def test_palette_and_appearance_changes_during_hold_are_restored_after_release(self):
        button = self._button()
        button.set_pressed(True)
        button.configure(state="disabled")
        button.configure(fg_color=("#A8B8C8", "#405060"), hover_color=("#C0D0E0", "#607080"))
        ctk.set_appearance_mode("Dark")
        button.configure(state="normal")
        self.assertTrue(button._is_visually_pressed())
        button._on_enter()
        held_fill = self._fill(button)
        button.set_pressed(False)
        self.assertEqual(self._fill(button), "#607080")
        self.assertNotEqual(held_fill, self._fill(button))
        button._on_leave()
        self.assertEqual(self._fill(button), "#405060")
        self.assertEqual(button.cget("fg_color"), ("#A8B8C8", "#405060"))

    def test_disabled_mouse_and_programmatic_activation_do_not_invoke_or_depress(self):
        command = Mock()
        button = self._button(command=command, state="disabled")
        normal_fill = self._fill(button)
        self._click(button)
        button.invoke()
        self.assertFalse(button._is_visually_pressed())
        self.assertEqual(self._fill(button), normal_fill)
        command.assert_not_called()

    def test_releasing_mouse_press_without_hold_restores_hover_palette(self):
        button = self._button()
        button._on_enter()
        hovering_fill = self._fill(button)
        button._on_press()
        button.set_pressed(False)
        self.assertFalse(button._is_visually_pressed())
        self.assertEqual(self._fill(button), hovering_fill)

    def test_leaving_or_disabling_cancels_transient_mouse_press(self):
        button = self._button()
        button._on_enter()
        button._on_press()
        button._on_leave()
        self.assertFalse(button._is_visually_pressed())
        button._on_press()
        button.configure(state="disabled")
        self.assertFalse(button._is_visually_pressed())
        button.configure(state="normal")
        self.assertFalse(button._is_visually_pressed())

    def test_new_labels_receive_mouse_press_feedback_without_duplicate_commands(self):
        command = Mock()
        button = self._button(command=command)
        button.configure(text="")
        button.configure(text="Next ROI")
        button._text_label.event_generate("<Enter>")
        button._text_label.event_generate("<ButtonPress-1>", x=3, y=3)
        self.assertTrue(button._is_visually_pressed())
        button._text_label.event_generate("<ButtonRelease-1>", x=3, y=3)
        self.assertFalse(button._is_visually_pressed())
        command.assert_called_once_with()

        icon = ctk.CTkImage(light_image=Image.new("RGB", (12, 12), "blue"), size=(12, 12))
        button.configure(image=icon)
        self.root.update()
        self._click(button, child=button._image_label)
        self.assertEqual(command.call_count, 2)

    def test_redraw_preserves_custom_mouse_bindings_and_unbind_restores_press_feedback(self):
        button = self._button()
        extra_press = Mock()
        button.bind("<ButtonPress-1>", extra_press)
        button.configure(border_width=2)
        button._canvas.event_generate("<ButtonPress-1>", x=3, y=3)
        extra_press.assert_called_once()
        self.assertTrue(button._is_visually_pressed())
        button.set_pressed(False)
        button.unbind("<ButtonPress-1>")
        button._canvas.event_generate("<ButtonPress-1>", x=3, y=3)
        self.assertTrue(button._is_visually_pressed())
        extra_press.assert_called_once()

    def test_feedback_api_does_not_invoke_and_is_safe_after_destroy(self):
        command = Mock()
        button = self._button(command=command)
        button.set_pressed(True)
        button.set_pressed(False)
        command.assert_not_called()
        button.set_pressed(True)
        button.destroy()
        button.set_pressed(False)
        self.root.update()
        command.assert_not_called()
        self.assertEqual(self.callback_errors, [])

    def test_mouse_command_can_destroy_its_button_without_a_pending_animation(self):
        calls = []

        def close_view():
            calls.append("closed")
            button.destroy()

        button = self._button(command=close_view)
        self._click(button)
        self.assertEqual(calls, ["closed"])
        for job in self.root.tk.call("after", "info"):
            script = str(self.root.tk.call("after", "info", job))
            self.assertNotIn("_click_animation", script)
        self.assertEqual(self.callback_errors, [])


if __name__ == "__main__":
    unittest.main()
