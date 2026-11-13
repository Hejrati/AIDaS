"""Manual review commits must win over asynchronous automatic detection."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aidas.steps import step4_analyze_isez as step4


class Step4ReviewCorrectionsTests(unittest.TestCase):
    @staticmethod
    def _frame(index=20):
        frame = step4.Step4Frame.__new__(step4.Step4Frame)
        profile = np.full(140, 20.0)
        profile[16:80] += 80 * np.sin(np.linspace(0, np.pi, 64)) ** 2
        frame.image = np.repeat((profile * 3)[:, None], 4, axis=1)
        frame.rois = [step4.ISezROI(f"{i + 1:02d}", 1, 4, 1, 140) for i in range(20)]
        frame.rois.append(step4.ISezROI("fovea", 1, 4, 1, 140))
        frame.current_roi_idx = index
        frame.profile_clicks = [2.0, 8.0]
        frame.completed = {}
        frame._roi_adjust_bounds = {}
        frame.roi_clicks = {}
        frame.excluded_rois = set()
        frame.auto_detection_reviews = {frame.rois[index].suffix: "Review required"}
        frame.batch_roi_notebook = None
        frame.canvas = None
        frame.status_var = mock.Mock()
        frame._profile_zoom_dialog = None
        frame._auto_detecting = False
        frame._auto_saving_roi = False
        frame._current_profile = profile
        for name in (
            "_refresh_roi_list", "_start_roi_update_animation", "_select_roi_in_list",
            "_draw_roi_overview_grid", "_update_build_stack_button_state",
            "_open_automatic_profile_review", "_close_profile_zoom",
            "_sync_entry_vars_from_clicks", "_render_current_roi",
        ):
            setattr(frame, name, mock.Mock())
        return frame

    @staticmethod
    def _dialog():
        dialog = step4.Step4ProfileZoomDialog.__new__(step4.Step4ProfileZoomDialog)
        dialog.roi_index = 20
        dialog._start, dialog._end = 25, 110
        dialog._applied_bounds = dialog._saved_bounds = (25, 110)
        dialog._adjust_bounds = dialog._applied_adjust_bounds = dialog._saved_adjust_bounds = False
        dialog._actions_busy = False
        dialog._closing = False
        dialog.save_button = mock.Mock()
        dialog.apply_button = mock.Mock()
        dialog._on_apply_callback = mock.Mock(return_value=True)
        dialog._on_save_callback = mock.Mock(return_value=True)
        dialog._on_close_callback = mock.Mock()
        dialog.destroy = mock.Mock()
        return dialog

    def test_apply_and_save_confirm_foveola_and_clear_review_after_measurement(self):
        for action in ("_apply_profile_zoom_clicks", "_save_profile_zoom_clicks"):
            with self.subTest(action=action):
                frame = self._frame()
                self.assertTrue(getattr(frame, action)(20, 25, 110, adjust_bounds=False))
                self.assertNotIn("fovea", frame.auto_detection_reviews)
                self.assertEqual(frame.current_roi_idx, 20)
                saved = frame.completed["fovea"]
                self.assertEqual((saved.start, saved.end), (25, 110))
                self.assertTrue(saved.manual_bounds)
                np.testing.assert_array_equal(saved.baseline_x, [25, 110])
                frame._refresh_roi_list.assert_called_once_with()
                frame._draw_roi_overview_grid.assert_called_once_with()

    def test_failed_popup_apply_keeps_review_flag_and_unconfirmed_result(self):
        frame = self._frame()
        with mock.patch.object(step4, "analyze_and_save_roi", side_effect=ValueError("invalid profile")), \
                mock.patch.object(step4.messagebox, "showerror"):
            self.assertFalse(frame._apply_profile_zoom_clicks(20, 25, 110))
        self.assertIn("fovea", frame.auto_detection_reviews)
        self.assertNotIn("fovea", frame.completed)

    def test_unchanged_flagged_bounds_can_be_confirmed_in_popup(self):
        frame = self._frame()
        frame.profile_clicks = [25.0, 110.0]
        frame._roi_adjust_bounds["fovea"] = False
        options = frame._profile_zoom_options(20)
        self.assertTrue(options["requires_confirmation"])
        dialog = self._dialog()
        dialog._requires_confirmation = options["requires_confirmation"]
        dialog._on_apply_callback = frame._apply_profile_zoom_clicks
        self.assertTrue(dialog._has_pending_changes())
        dialog._apply()
        self.assertFalse(dialog._has_pending_changes())
        self.assertNotIn("fovea", frame.auto_detection_reviews)
        self.assertIn("fovea", frame.completed)
        dialog.destroy.assert_not_called()

    def test_new_review_roi_stays_selected_after_main_plot_auto_save(self):
        frame = self._frame(index=0)
        frame.profile_clicks = [25.0, 110.0]
        frame._roi_adjust_bounds["01"] = False
        frame._auto_save_current_roi()
        self.assertEqual(frame.current_roi_idx, 0)
        self.assertNotIn("01", frame.auto_detection_reviews)
        self.assertIn("01", frame.completed)
        self.assertEqual(frame.profile_clicks, [25.0, 110.0])

    @staticmethod
    def _worker_result():
        return step4.AutoDetectionRunResult(
            completed={"01": SimpleNamespace(start=70, end=100)},
            reviews={"fovea": "Must be manual", "02": "Uncertain detection"},
            roi_clicks={"01": [70.0, 100.0], "02": [71.0, 101.0]},
            removed_clicks={"03"}, review_count=2, current_roi_idx=1,
        )

    def test_late_detection_preserves_popup_commit_and_current_roi(self):
        frame = self._frame()
        frame._auto_detecting = True
        frame._auto_detection_start_roi_idx = 20
        self.assertTrue(frame._apply_profile_zoom_clicks(20, 25, 110))
        saved = frame.completed["fovea"]
        frame._apply_auto_detection_run(self._worker_result())
        self.assertIs(frame.completed["fovea"], saved)
        self.assertNotIn("fovea", frame.auto_detection_reviews)
        self.assertEqual(frame.profile_clicks, [25.0, 110.0])
        self.assertFalse(frame._should_adjust_roi_bounds("fovea"))
        self.assertEqual(frame.current_roi_idx, 20)
        self.assertIn("01", frame.completed)
        self.assertIn("02", frame.auto_detection_reviews)
        frame._open_automatic_profile_review.assert_not_called()

    def test_late_detection_cannot_replace_or_remove_manual_boundary_edits(self):
        for index in (0, 2):
            with self.subTest(index=index):
                frame = self._frame(index=index)
                suffix = frame.rois[index].suffix
                frame._auto_detecting = True
                frame._auto_detection_start_roi_idx = index
                frame.ax_profile, frame.ax_roi_grid = object(), object()
                frame.profile_clicks = []
                frame.disable_line_positioning_var = SimpleNamespace(get=lambda: True)
                frame.profile_status_var = mock.Mock()
                frame._on_profile_click(SimpleNamespace(inaxes=frame.ax_profile, xdata=27.0, button=1))
                frame._apply_auto_detection_run(self._worker_result())
                self.assertEqual(frame.current_roi_idx, index)
                self.assertEqual(frame.roi_clicks[suffix], [27.0])
                self.assertEqual(frame.profile_clicks, [27.0])
                self.assertNotIn(suffix, frame.completed)
                self.assertFalse(frame._should_adjust_roi_bounds(suffix))
                frame._open_automatic_profile_review.assert_not_called()

    def test_detection_does_not_move_selected_roi_when_user_navigated(self):
        frame = self._frame(index=5)
        frame._auto_detection_start_roi_idx = 0
        frame._apply_auto_detection_run(self._worker_result())
        self.assertEqual(frame.current_roi_idx, 5)
        frame._open_automatic_profile_review.assert_not_called()

    def test_excluded_roi_cannot_be_edited_saved_or_restored_by_worker(self):
        frame = self._frame(index=0)
        frame.excluded_rois = {"01"}
        frame.profile_clicks = [25.0, 110.0]
        frame.roi_clicks["01"] = [25.0, 110.0]
        frame._roi_adjust_bounds["01"] = False
        self.assertFalse(frame._apply_profile_zoom_clicks(0, 35, 100))
        self.assertFalse(frame._save_current_roi(auto_advance=False, apply_entry_values=False))
        frame._apply_auto_detection_run(self._worker_result())
        self.assertEqual(frame.roi_clicks["01"], [25.0, 110.0])
        self.assertNotIn("01", frame.completed)
        self.assertFalse(frame._should_adjust_roi_bounds("01"))

    def test_detector_skips_excluded_profiles_and_reports_full_progress(self):
        rois = [step4.ISezROI("01", 1, 4), step4.ISezROI("02", 1, 4)]
        progress = mock.Mock()
        detection = step4.ProfileBoundaryDetection(
            start=70, peak=85, end=100, confidence=0.95, accepted=True,
            prominence=8.0, quadratic_r2=0.9, reason="accepted",
        )
        with mock.patch.object(step4, "intensity_profile", return_value=np.ones(140)) as profile, \
                mock.patch.object(step4, "detect_profile_boundaries", return_value=detection), \
                mock.patch.object(step4, "analyze_and_save_roi", return_value=SimpleNamespace(start=70, end=100)):
            result = step4.Step4Frame._compute_auto_detection_run(
                object(), rois, set(), step4.STEP4_AUTO_DETECTION_DEFAULTS,
                progress_callback=progress, excluded_suffixes={"01"},
            )
        self.assertEqual(profile.call_count, 1)
        self.assertEqual(set(result.completed), {"02"})
        self.assertNotIn("01", result.reviews)
        self.assertNotIn("01", result.roi_clicks)
        self.assertEqual(progress.call_args, mock.call(4, 4))

    def test_popup_exclusion_closes_before_toggle_without_saving_draft(self):
        dialog = self._dialog()
        dialog._start = 30
        order = []
        dialog._on_close_callback.side_effect = lambda _dialog: order.append("closed")
        dialog._on_exclude_callback = mock.Mock(side_effect=lambda: order.append("excluded"))
        dialog._exclude_roi()
        dialog._exclude_roi()
        self.assertEqual(order, ["closed", "excluded"])
        dialog._on_apply_callback.assert_not_called()
        dialog._on_save_callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
