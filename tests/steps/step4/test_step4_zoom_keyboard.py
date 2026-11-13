"""Keyboard navigation through the real Step 4 zoom editor."""

from types import SimpleNamespace
import unittest
from unittest import mock
from tkinter import ttk

import customtkinter as ctk
import numpy as np

from aidas.steps.step4_analyze_isez import RESULTS_HEADERS, Step4ProfileZoomDialog


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class Step4ZoomKeyboardTests(unittest.TestCase):
    def setUp(self):
        self.root = _QuietRoot()
        self.root.attributes("-alpha", 0)
        self.root.geometry("400x180+0+0")
        self.root.update()
        self.profile = np.sin(np.linspace(0, 10, 100)) + 2
        self.navigate = mock.Mock(side_effect=self._profile_options)
        self.apply = mock.Mock(return_value=True)
        self.save = mock.Mock(return_value=True)
        self.closed = mock.Mock()
        self.dialog = None

    def tearDown(self):
        self.root.destroy()

    def _profile_options(self, index):
        return {
            "roi_index": index, "roi_suffix": f"{index + 1:02d}",
            "profile": self.profile, "start": 20, "end": 70,
            "adjust_bounds": False,
        }

    def _open_dialog(self, *, navigation=True):
        size_window = Step4ProfileZoomDialog._size_and_center_on_active_screen

        def invisible_size(dialog):
            dialog.attributes("-alpha", 0)
            size_window(dialog)

        with mock.patch.object(Step4ProfileZoomDialog, "_size_and_center_on_active_screen", invisible_size):
            self.dialog = Step4ProfileZoomDialog(
                self.root, **self._profile_options(0),
                on_preview=lambda *_args, **_kwargs: {k: 1.0 for k in RESULTS_HEADERS[1:]},
                on_apply=self.apply, on_save=self.save, on_close=self.closed,
                navigation_indices=[0, 3, 5],
                on_navigate=self.navigate if navigation else None,
            )
        self.root.update()
        return self.dialog

    def _press(self, widget, key):
        widget.focus_force()
        self.root.update()
        widget.event_generate(f"<KeyPress-{key}>")
        widget.event_generate(f"<KeyRelease-{key}>")
        self.root.update()

    def test_canvas_arrows_navigate_once_and_skip_missing_profiles(self):
        dialog = self._open_dialog()
        chart = dialog.canvas.get_tk_widget()
        matplotlib_key = mock.Mock()
        dialog.canvas.mpl_connect("key_press_event", matplotlib_key)

        self._press(chart, "Right")
        self.assertEqual(dialog.roi_index, 3)
        self.assertEqual(dialog.plot_number_label.cget("text"), "ROI 04")
        self.navigate.assert_called_once_with(3)
        self._press(chart, "Left")
        self.assertEqual(dialog.roi_index, 0)
        self.assertEqual(self.navigate.call_args_list, [mock.call(3), mock.call(0)])
        matplotlib_key.assert_not_called()
        self.apply.assert_not_called()
        self.save.assert_not_called()
        self.assertFalse(dialog._closing)

    def test_arrows_work_when_action_button_or_its_child_has_focus(self):
        dialog = self._open_dialog()

        self._press(dialog.cancel_button, "Right")
        self.assertEqual(dialog.roi_index, 3)
        self._press(dialog.cancel_button._text_label, "Right")
        self.assertEqual(dialog.roi_index, 5)
        self._press(dialog.next_plot_button._text_label, "Left")
        self.assertEqual(dialog.roi_index, 3)
        self.assertEqual(self.navigate.call_args_list, [mock.call(3), mock.call(5), mock.call(3)])
        self.closed.assert_not_called()

    def test_arrows_stop_at_first_and_last_available_profiles(self):
        dialog = self._open_dialog()
        chart = dialog.canvas.get_tk_widget()

        self._press(chart, "Left")
        self.navigate.assert_not_called()
        self._press(chart, "Right")
        self._press(chart, "Right")
        self.assertEqual(dialog.roi_index, 5)
        self._press(chart, "Right")
        self._press(chart, "Right")
        self.assertEqual(self.navigate.call_args_list, [mock.call(3), mock.call(5)])
        self.assertEqual(dialog.next_plot_button.cget("state"), "disabled")

    def test_arrow_navigation_preserves_each_unapplied_draft(self):
        dialog = self._open_dialog()
        chart = dialog.canvas.get_tk_widget()
        dialog._move_boundary("start", 25, refresh_measurements=True)

        self._press(chart, "Right")
        dialog._move_boundary("end", 65, refresh_measurements=True)
        self._press(chart, "Left")
        self.assertEqual((dialog._start, dialog._end), (25, 70))
        self.assertTrue(dialog._has_pending_changes())
        self._press(chart, "Right")
        self.assertEqual((dialog._start, dialog._end), (20, 65))
        self.assertTrue(dialog._has_pending_changes())
        self.apply.assert_not_called()
        self.save.assert_not_called()

    def test_busy_or_unavailable_navigation_does_not_change_profile(self):
        dialog = self._open_dialog()
        chart = dialog.canvas.get_tk_widget()
        dialog._actions_busy = True
        self._press(chart, "Right")
        self.assertEqual(dialog.roi_index, 0)
        self.navigate.assert_not_called()
        dialog._actions_busy = False
        dialog._on_navigate_callback = None
        self._press(chart, "Right")
        self.assertEqual(dialog.roi_index, 0)
        self.navigate.assert_not_called()

    def test_arrows_stay_within_editor_and_close_removes_bindings(self):
        outside = ttk.Entry(self.root)
        outside.pack()
        dialog = self._open_dialog()
        tag = dialog._profile_navigation_key_tag
        self.assertNotIn(tag, outside.bindtags())
        self._press(outside, "Right")
        self.navigate.assert_not_called()
        dialog.close()
        self.root.update()

        for sequence in ("<Left>", "<Right>"):
            self.assertEqual(self.root.bind_class(tag, sequence), "")
        self._press(outside, "Right")
        # An already-dispatched key callback also becomes harmless after close.
        dialog._on_profile_navigation_key(SimpleNamespace(keysym="Right"))
        self.navigate.assert_not_called()
        self.closed.assert_called_once_with(dialog)


if __name__ == "__main__":
    unittest.main()
