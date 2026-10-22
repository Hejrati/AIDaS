import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aidas.steps.step4_analyze_isez import Step4Frame


class Step4ProfileNavigationTests(unittest.TestCase):
    """Navigation changes the active preview without closing or saving it."""

    def _frame(self):
        frame = Step4Frame.__new__(Step4Frame)
        frame.image = object()
        frame.rois = [SimpleNamespace(suffix=f"{index + 1:02d}") for index in range(6)]
        frame.current_roi_idx = 0
        frame.profile_clicks = [3.0, 60.0]
        frame.roi_clicks = {"04": [12.0, 180.0]}
        frame.completed = {
            "01": SimpleNamespace(start=2, end=65, manual_bounds=False),
            "04": SimpleNamespace(start=10, end=190, manual_bounds=True),
            "06": SimpleNamespace(start=20, end=75, manual_bounds=False),
        }
        frame._roi_adjust_bounds = {"04": False}
        frame._current_profile = np.arange(80, dtype=float)
        frame._profile_zoom_dialog = mock.Mock(roi_index=0)
        frame._opening_profile_zoom = False
        frame._auto_save_after_id = None
        frame.start_var = mock.Mock()
        frame.end_var = mock.Mock()
        frame._select_roi_in_list = mock.Mock()
        frame._render_current_roi = mock.Mock()
        frame._sync_active_batch_roi_state = mock.Mock()
        frame._close_profile_zoom = mock.Mock()
        frame._save_current_roi = mock.Mock()
        frame._save_profile_zoom_clicks = mock.Mock()
        frame._apply_profile_zoom_clicks = mock.Mock()
        frame._profile_zoom_measurements = mock.Mock()
        frame._profile_zoom_closed = mock.Mock()
        frame._queue_roi_auto_save = mock.Mock()
        frame.after_cancel = mock.Mock()
        frame._roi_overview_result = mock.Mock(
            side_effect=lambda index: frame.completed.get(frame.rois[index].suffix)
        )
        return frame

    @staticmethod
    def _profile_for(_image, roi):
        return np.arange(200 if roi.suffix == "04" else 80, dtype=float)

    def test_opening_zoom_exposes_only_available_profiles_with_gaps(self):
        frame = self._frame()
        with (
            mock.patch("aidas.steps.step4_analyze_isez.intensity_profile", side_effect=self._profile_for),
            mock.patch("aidas.steps.step4_analyze_isez.Step4ProfileZoomDialog") as constructor,
        ):
            frame._profile_zoom_dialog = None
            frame._open_profile_zoom(0)

        options = constructor.call_args.kwargs
        self.assertEqual(options["navigation_indices"], [0, 3, 5])
        self.assertEqual(options["on_navigate"], frame._navigate_profile_zoom)
        frame._save_current_roi.assert_not_called()
        frame._apply_profile_zoom_clicks.assert_not_called()

    def test_current_profile_options_keep_applied_bounds_and_saved_reference_separate(self):
        frame = self._frame()
        with mock.patch(
            "aidas.steps.step4_analyze_isez.intensity_profile", side_effect=self._profile_for
        ):
            options = frame._profile_zoom_options(0)

        self.assertEqual((options["start"], options["end"]), (3.0, 60.0))
        self.assertEqual(options["saved_bounds"], (2, 65))
        self.assertTrue(options["adjust_bounds"])
        self.assertTrue(options["saved_adjust_bounds"])
        self.assertEqual(options["roi_suffix"], "01")
        self.assertEqual(options["roi_index"], 0)

    def test_other_profile_options_use_its_own_selection_and_manual_mode(self):
        frame = self._frame()
        with mock.patch(
            "aidas.steps.step4_analyze_isez.intensity_profile", side_effect=self._profile_for
        ):
            options = frame._profile_zoom_options(3)

        self.assertEqual((options["start"], options["end"]), (12.0, 180.0))
        self.assertEqual(options["saved_bounds"], (10, 190))
        self.assertFalse(options["adjust_bounds"])
        self.assertFalse(options["saved_adjust_bounds"])
        self.assertEqual(frame.current_roi_idx, 0)
        self.assertEqual(frame.profile_clicks, [3.0, 60.0])

    def test_navigation_activates_target_before_preview_and_preserves_window(self):
        frame = self._frame()
        dialog = frame._profile_zoom_dialog
        observed = []

        def render_target():
            observed.append((frame.current_roi_idx, len(frame._current_profile)))
            self.assertEqual(frame.profile_clicks, [12.0, 180.0])

        frame._render_current_roi.side_effect = render_target
        with mock.patch(
            "aidas.steps.step4_analyze_isez.intensity_profile", side_effect=self._profile_for
        ):
            options = frame._navigate_profile_zoom(3)

        self.assertEqual(observed, [(3, 200)])
        self.assertEqual(options["roi_index"], frame.current_roi_idx)
        self.assertEqual((options["start"], options["end"]), (12.0, 180.0))
        self.assertIs(frame._profile_zoom_dialog, dialog)
        self.assertEqual(frame.roi_clicks["01"], [3.0, 60.0])
        frame._select_roi_in_list.assert_called_once_with()
        frame._close_profile_zoom.assert_not_called()
        dialog.close.assert_not_called()
        frame._save_current_roi.assert_not_called()
        frame._save_profile_zoom_clicks.assert_not_called()
        frame._apply_profile_zoom_clicks.assert_not_called()
        frame._queue_roi_auto_save.assert_not_called()

    def test_navigation_uses_target_profile_length_for_entry_values(self):
        frame = self._frame()
        with mock.patch(
            "aidas.steps.step4_analyze_isez.intensity_profile", side_effect=self._profile_for
        ):
            frame._navigate_profile_zoom(3)

        self.assertEqual(frame._current_profile.size, 200)
        frame.start_var.set.assert_called_with("12")
        frame.end_var.set.assert_called_with("180")

    def test_invalid_or_unavailable_target_leaves_active_editor_unchanged(self):
        for target in (-1, 1, 6):
            with self.subTest(target=target):
                frame = self._frame()
                dialog = frame._profile_zoom_dialog
                with mock.patch(
                    "aidas.steps.step4_analyze_isez.intensity_profile",
                    side_effect=self._profile_for,
                ):
                    self.assertIsNone(frame._navigate_profile_zoom(target))
                self.assertEqual(frame.current_roi_idx, 0)
                self.assertEqual(frame.profile_clicks, [3.0, 60.0])
                self.assertIs(frame._profile_zoom_dialog, dialog)
                frame._select_roi_in_list.assert_not_called()
                frame._render_current_roi.assert_not_called()
                frame._close_profile_zoom.assert_not_called()

    def test_navigation_without_image_leaves_editor_unchanged(self):
        frame = self._frame()
        frame.image = None

        self.assertIsNone(frame._navigate_profile_zoom(3))

        self.assertEqual(frame.current_roi_idx, 0)
        frame._render_current_roi.assert_not_called()
        frame._save_current_roi.assert_not_called()


if __name__ == "__main__":
    unittest.main()
