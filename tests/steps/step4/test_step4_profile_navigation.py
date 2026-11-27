import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aidas.steps.step4_analyze_isez import AutoDetectionRunResult, Step4Frame


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

    def _automatic_review_frame(self):
        frame = self._frame()
        frame.rois = [SimpleNamespace(suffix=f"{index + 1:02d}") for index in range(21)]
        frame.rois[20].suffix = "fovea"
        frame.current_roi_idx = 20
        frame._automatic_profile_review_ready = True
        frame._profile_zoom_dialog = None
        frame._remember_current_roi_clicks = mock.Mock()
        frame._load_current_roi_clicks = mock.Mock()
        frame._open_profile_zoom = mock.Mock()
        return frame

    def test_automatic_review_starts_at_roi_one_and_opens_only_once(self):
        frame = self._automatic_review_frame()
        frame.auto_detection_reviews = {"fovea": "Manual"}

        frame._open_automatic_profile_review()
        # Closing the editor, rendering, and revisiting the same file must not
        # repeatedly force the review window back onto the user.
        frame._profile_zoom_dialog = None
        frame._open_automatic_profile_review()

        self.assertEqual(frame.current_roi_idx, 0)
        frame._open_profile_zoom.assert_called_once_with(0, navigation_indices=[0, 3, 5])
        frame._save_current_roi.assert_not_called()
        frame._queue_roi_auto_save.assert_not_called()

    def test_automatic_preview_only_browses_finished_peripheral_plots(self):
        frame = self._automatic_review_frame()
        frame.excluded_rois = {"01"}
        frame.roi_clicks["02"] = [10.0, 60.0]
        frame.auto_detection_reviews = {"03": "Uncertain"}
        frame.completed.update({"03": frame.completed["04"], "fovea": frame.completed["04"]})
        del frame._open_profile_zoom

        with (
            mock.patch("aidas.steps.step4_analyze_isez.intensity_profile", side_effect=self._profile_for),
            mock.patch("aidas.steps.step4_analyze_isez.Step4ProfileZoomDialog") as constructor,
        ):
            frame._open_automatic_profile_review()

        options = constructor.call_args.kwargs
        self.assertEqual(options["roi_index"], 3)
        self.assertEqual(options["navigation_indices"], [3, 5])
        self.assertIsNone(options["on_foveola"])
        self.assertTrue(frame._automatic_profile_review_opened)

    def test_flagged_profile_without_preview_opens_with_empty_boundaries(self):
        frame = self._frame()
        frame.auto_detection_reviews = {"02": "Uncertain", "03": "Manual"}
        with mock.patch("aidas.steps.step4_analyze_isez.intensity_profile", side_effect=self._profile_for):
            options = frame._profile_zoom_options(2)

        self.assertTrue(options["review_required"])
        self.assertTrue(options["review_has_more"])
        self.assertIsNone(options["start"])
        self.assertIsNone(options["end"])

    def test_confirmed_review_opens_next_at_idle_only_in_same_context(self):
        for switch_image in (False, True):
            with self.subTest(switch_image=switch_image):
                frame = self._frame()
                dialog = frame._profile_zoom_dialog
                dialog.review_confirmed = True
                del frame._profile_zoom_closed
                frame._roi_context_generation = 1
                frame.auto_detection_reviews = {"03": "Manual"}
                frame.after_idle = mock.Mock()
                frame._open_profile_zoom = mock.Mock()

                frame._profile_zoom_closed(dialog)
                self.assertIsNone(frame._profile_zoom_dialog)
                if switch_image:
                    frame._roi_context_generation += 1
                frame.after_idle.call_args.args[0]()

                if switch_image:
                    frame._open_profile_zoom.assert_not_called()
                else:
                    frame._open_profile_zoom.assert_called_once_with(2)

    def test_new_detection_run_reopens_review_after_the_previous_review_closed(self):
        frame = self._automatic_review_frame()
        frame.batch_roi_notebook = None
        frame.auto_detection_reviews = {}
        frame.status_var = mock.Mock()
        frame._refresh_roi_list = mock.Mock()
        result = AutoDetectionRunResult(
            completed={}, reviews={}, roi_clicks={}, removed_clicks=set(),
            review_count=0, current_roi_idx=20,
        )

        frame._apply_auto_detection_run(result)
        frame._apply_auto_detection_run(result)

        preview = mock.call(0, navigation_indices=[0, 3, 5])
        self.assertEqual(frame._open_profile_zoom.call_args_list, [preview, preview])
        self.assertEqual(frame.current_roi_idx, 0)
        frame._save_current_roi.assert_not_called()

    def test_automatic_review_skips_missing_previews_and_never_opens_fovea(self):
        frame = self._automatic_review_frame()
        frame.completed = {"04": object(), "fovea": object()}

        frame._open_automatic_profile_review()

        frame._open_profile_zoom.assert_called_once_with(3, navigation_indices=[3])
        frame._automatic_profile_review_opened = False
        frame.completed = {"fovea": object()}
        frame.auto_detection_reviews = {"fovea": "Manual"}
        frame._open_profile_zoom.reset_mock()
        frame._open_automatic_profile_review()
        frame._open_profile_zoom.assert_not_called()

        frame._review_next_plot()
        frame._open_profile_zoom.assert_called_once_with(20)

    def test_automatic_review_waits_for_loading_and_respects_finished_or_destroyed_files(self):
        for name in ("_stack_building", "_stack_build_complete", "_step4_destroying", "_batch_roi_open_cancel"):
            with self.subTest(name=name):
                frame = self._automatic_review_frame()
                setattr(frame, name, True)

                frame._open_automatic_profile_review()

                frame._open_profile_zoom.assert_not_called()
                self.assertFalse(frame.__dict__.get("_automatic_profile_review_opened", False))

    def test_automatic_review_has_independent_once_only_state_per_file_tab(self):
        frame = self._automatic_review_frame()
        frame.batch_roi_notebook = object()
        frame.batch_roi_tab_states = {
            "first": {"_automatic_profile_review_ready": True},
            "second": {"_automatic_profile_review_ready": True},
            "done": {"_automatic_profile_review_ready": True, "complete": True},
        }
        for key in ("first", "second", "first", "done"):
            frame._active_batch_roi_tab = key
            frame._open_automatic_profile_review()

        preview = mock.call(0, navigation_indices=[0, 3, 5])
        self.assertEqual(frame._open_profile_zoom.call_args_list, [preview, preview])

    def test_foveola_action_selects_manual_roi_without_saving(self):
        frame = self._automatic_review_frame()
        frame.status_var = mock.Mock()
        frame.current_roi_idx = 19

        frame._continue_to_foveola()

        self.assertEqual(frame.current_roi_idx, 20)
        frame._close_profile_zoom.assert_called_once_with()
        frame._load_current_roi_clicks.assert_called_once_with()
        frame._render_current_roi.assert_called_once_with()
        frame._save_current_roi.assert_not_called()
        frame._queue_roi_auto_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
