"""Visible navigation feedback through the real Step 4 zoom editor."""

import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk
import numpy as np

from aidas.steps.step4_analyze_isez import RESULTS_HEADERS, Step4ProfileZoomDialog


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class Step4NavigationFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.root = _QuietRoot()
        self.root.attributes("-alpha", 0)
        self.root.geometry("400x180+0+0")
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)
        self.root.update()
        self.profile = np.sin(np.linspace(0, 10, 100)) + 2
        self.navigate = mock.Mock(side_effect=self._profile_options)
        self.dialog = None

    def tearDown(self):
        self.root.destroy()
        self.assertFalse(self.callback_errors, f"Unexpected Tk callback errors: {self.callback_errors}")

    def _profile_options(self, index):
        return {
            "roi_index": index, "roi_suffix": f"{index + 1:02d}",
            "profile": self.profile, "start": 20, "end": 70,
            "adjust_bounds": False,
        }

    def _open_dialog(self, *, feedback_ms=None, navigation=True):
        size_window = Step4ProfileZoomDialog._size_and_center_on_active_screen

        def invisible_size(dialog):
            dialog.attributes("-alpha", 0)
            size_window(dialog)

        with mock.patch.object(Step4ProfileZoomDialog, "_size_and_center_on_active_screen", invisible_size):
            self.dialog = Step4ProfileZoomDialog(
                self.root, **self._profile_options(0),
                on_preview=lambda *_args, **_kwargs: {k: 1.0 for k in RESULTS_HEADERS[1:]},
                on_apply=mock.Mock(return_value=True), on_save=mock.Mock(return_value=True),
                on_close=mock.Mock(), navigation_indices=[0, 3, 5],
                on_navigate=self.navigate if navigation else None,
            )
        if feedback_ms is not None:
            self.dialog.NAVIGATION_FEEDBACK_MS = feedback_ms
        self.root.update()
        return self.dialog

    def _wait_for(self, predicate, message="Navigation did not finish"):
        deadline = time.monotonic() + 3
        while not predicate():
            self.root.update()
            if time.monotonic() >= deadline:
                self.fail(message)
            time.sleep(0.005)

    def _wait_for_navigation(self):
        self._wait_for(lambda: not self.dialog._actions_busy and
                       self.dialog.__dict__.get("_navigation_feedback_button") is None)

    def _key(self, key):
        chart = self.dialog.canvas.get_tk_widget()
        chart.focus_force()
        self.root.update()
        chart.event_generate(f"<KeyPress-{key}>")
        chart.event_generate(f"<KeyRelease-{key}>")

    def test_mouse_press_is_visible_and_holds_until_destination_is_drawn(self):
        dialog = self._open_dialog()
        button = dialog.next_plot_button
        pointer_target = button._text_label
        observations = []

        def load(index):
            observations.append(("load", button._is_visually_pressed(), dialog._actions_busy))
            return self._profile_options(index)

        draw = dialog.canvas.draw

        def draw_target(*args, **kwargs):
            if dialog.roi_index == 3:
                observations.append(("draw", button._is_visually_pressed(), dialog._actions_busy))
            return draw(*args, **kwargs)

        self.navigate.side_effect = load
        with mock.patch.object(dialog.canvas, "draw", side_effect=draw_target):
            pointer_target.event_generate("<Enter>")
            pointer_target.event_generate("<ButtonPress-1>", x=3, y=3)
            self.assertTrue(button._is_visually_pressed())
            self.navigate.assert_not_called()
            pointer_target.event_generate("<ButtonRelease-1>", x=3, y=3)
            self.assertTrue(button._is_visually_pressed())
            self.assertTrue(dialog._actions_busy)
            self.navigate.assert_not_called()
            self._wait_for_navigation()

        self.assertEqual(dialog.roi_index, 3)
        self.navigate.assert_called_once_with(3)
        self.assertIn(("load", True, True), observations)
        self.assertIn(("draw", True, True), observations)
        self.assertFalse(button._is_visually_pressed())
        self.assertEqual(button.cget("state"), "normal")

    def test_arrow_keys_press_matching_button_and_release_after_loading(self):
        dialog = self._open_dialog()
        for key, button, target in (("Right", dialog.next_plot_button, 3),
                                    ("Left", dialog.previous_plot_button, 0)):
            with self.subTest(key=key):
                seen = []
                self.navigate.side_effect = lambda index: (
                    seen.append(button._is_visually_pressed()) or self._profile_options(index)
                )
                self._key(key)
                self.assertTrue(button._is_visually_pressed())
                self.assertTrue(dialog._actions_busy)
                self._wait_for_navigation()
                self.assertEqual(dialog.roi_index, target)
                self.assertEqual(seen, [True])
                self.assertFalse(button._is_visually_pressed())
        self.assertEqual(self.navigate.call_args_list, [mock.call(3), mock.call(0)])

    def test_repeated_keys_before_load_do_not_skip_a_profile(self):
        dialog = self._open_dialog(feedback_ms=0)
        chart = dialog.canvas.get_tk_widget()
        chart.focus_force()
        self.root.update()

        for _ in range(8):
            chart.event_generate("<KeyPress-Right>")
        chart.event_generate("<KeyRelease-Right>")
        self.navigate.assert_not_called()
        self.assertTrue(dialog.next_plot_button._is_visually_pressed())
        self._wait_for_navigation()

        self.navigate.assert_called_once_with(3)
        self.assertEqual(dialog.roi_index, 3)

    def test_browsing_feedback_preserves_unapplied_drafts(self):
        dialog = self._open_dialog(feedback_ms=0)
        dialog._move_boundary("start", 25, refresh_measurements=True)
        self._key("Right")
        self._wait_for_navigation()
        dialog._move_boundary("end", 65, refresh_measurements=True)
        self._key("Left")
        self._wait_for_navigation()
        self.assertEqual((dialog._start, dialog._end), (25, 70))
        self.assertTrue(dialog._has_pending_changes())
        self._key("Right")
        self._wait_for_navigation()
        self.assertEqual((dialog._start, dialog._end), (20, 65))
        self.assertTrue(dialog._has_pending_changes())
        dialog._on_apply_callback.assert_not_called()
        dialog._on_save_callback.assert_not_called()

    def test_old_release_cannot_clear_feedback_for_a_new_navigation(self):
        dialog = self._open_dialog(feedback_ms=10000)
        dialog.next_plot_button.invoke()
        self.root.update()
        self.assertEqual(dialog.roi_index, 3)
        self.assertFalse(dialog._actions_busy)
        old_token = dialog._navigation_feedback_token
        self.assertTrue(dialog.next_plot_button._is_visually_pressed())
        dialog.NAVIGATION_FEEDBACK_MS = 0
        dialog.previous_plot_button.invoke()
        self.assertFalse(dialog.next_plot_button._is_visually_pressed())
        self.assertTrue(dialog.previous_plot_button._is_visually_pressed())

        # A release callback already dispatched for the old image is harmless.
        dialog._release_navigation_feedback(old_token)
        self.assertTrue(dialog.previous_plot_button._is_visually_pressed())
        self._wait_for_navigation()
        self.assertEqual(dialog.roi_index, 0)
        self.assertFalse(dialog.previous_plot_button._is_visually_pressed())
        self.assertEqual(self.navigate.call_args_list, [mock.call(3), mock.call(0)])

    def test_pending_navigation_does_not_accept_boundary_edits(self):
        dialog = self._open_dialog(feedback_ms=0)
        dialog.next_plot_button.invoke()
        self.assertTrue(dialog._actions_busy)
        with mock.patch.object(dialog, "_move_boundary") as move:
            event = SimpleNamespace(button=1, inaxes=dialog.ax, xdata=35)
            dialog._on_press(event)
            dialog._on_motion(event)
            dialog._on_release(event)
            move.assert_not_called()
        self._wait_for_navigation()
        self.assertEqual(dialog.roi_index, 3)

    def test_endpoints_and_unavailable_navigation_never_latch_feedback(self):
        dialog = self._open_dialog(feedback_ms=0)
        self._key("Left")
        self.navigate.assert_not_called()
        self.assertFalse(dialog.previous_plot_button._is_visually_pressed())
        self.assertIsNone(dialog.__dict__.get("_navigation_feedback_button"))
        for target in (3, 5):
            self._key("Right")
            self._wait_for_navigation()
            self.assertEqual(dialog.roi_index, target)
        self._key("Right")
        self.assertEqual(self.navigate.call_count, 2)
        self.assertEqual(dialog.next_plot_button.cget("state"), "disabled")
        self.assertFalse(dialog.next_plot_button._is_visually_pressed())
        dialog._on_navigate_callback = None
        self._key("Left")
        self.assertEqual(self.navigate.call_count, 2)
        self.assertFalse(dialog.previous_plot_button._is_visually_pressed())
        self.assertIsNone(dialog.__dict__.get("_navigation_feedback_button"))

    def test_missing_destination_clears_feedback_and_restores_controls(self):
        dialog = self._open_dialog()
        self.navigate.side_effect = None
        self.navigate.return_value = None
        dialog.next_plot_button.invoke()
        self.assertTrue(dialog.next_plot_button._is_visually_pressed())
        self._wait_for_navigation()
        self.assertEqual(dialog.roi_index, 0)
        self.assertFalse(dialog.next_plot_button._is_visually_pressed())
        self.assertEqual(dialog.next_plot_button.cget("state"), "normal")
        self.navigate.assert_called_once_with(3)

    def test_callback_error_releases_feedback_and_allows_retry(self):
        dialog = self._open_dialog(feedback_ms=0)
        self.navigate.side_effect = RuntimeError("Could not load target profile")
        dialog.next_plot_button.invoke()
        self._wait_for_navigation()
        self.assertEqual(len(self.callback_errors), 1)
        self.assertIs(self.callback_errors[0][0], RuntimeError)
        self.callback_errors.clear()
        self.assertEqual(dialog.roi_index, 0)
        self.assertFalse(dialog.next_plot_button._is_visually_pressed())
        self.assertEqual(dialog.next_plot_button.cget("state"), "normal")
        self.navigate.side_effect = self._profile_options
        dialog.next_plot_button.invoke()
        self._wait_for_navigation()
        self.assertEqual(dialog.roi_index, 3)

    def test_close_before_deferred_load_cancels_navigation(self):
        dialog = self._open_dialog()
        dialog.next_plot_button.invoke()
        self.assertTrue(dialog._actions_busy)
        dialog.close()
        self.root.update()
        self.navigate.assert_not_called()
        self.assertFalse(dialog.winfo_exists())

    def test_close_during_release_delay_cancels_feedback_callback(self):
        dialog = self._open_dialog(feedback_ms=250)
        dialog.next_plot_button.invoke()
        self._wait_for(lambda: dialog.roi_index == 3)
        dialog.close()
        finished = tk.BooleanVar(master=self.root, value=False)
        self.root.after(300, lambda: finished.set(True))
        self.root.wait_variable(finished)
        self.assertFalse(dialog.winfo_exists())
        self.navigate.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
