"""Excluded ROIs remain attached to their image and keep export positions."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step4_analyze_isez import Step4Frame


class Step4ExclusionTests(unittest.TestCase):
    def setUp(self):
        frame = self.frame = object.__new__(Step4Frame)
        frame.rois = [SimpleNamespace(suffix="01"), SimpleNamespace(suffix="02")]
        frame.image = object()
        frame.current_path = Path("subject/nasal/_flat_LIGHT.hdr")
        frame.current_roi_idx = 0
        frame.completed = {}
        frame.excluded_rois = set()
        frame.auto_detection_reviews = {"01": "Needs review"}
        frame.roi_clicks = {}
        frame.profile_clicks = [78.0, 101.0]
        frame.batch_roi_notebook = None
        frame.batch_roi_tab_states = {}
        frame._active_batch_roi_tab = None
        frame._auto_save_after_id = "pending-save"
        frame._auto_detecting = False
        frame._stack_building = False
        frame._stack_build_complete = False
        frame.status_var = mock.Mock()
        frame.profile_status_var = mock.Mock()
        frame.stats_var = mock.Mock()
        frame.after_cancel = mock.Mock()
        for name in (
            "_close_profile_zoom", "_cancel_roi_update_animation", "_refresh_roi_list",
            "_render_current_roi", "_sync_active_batch_roi_state",
            "_update_continue_to_step5_button_state",
        ):
            setattr(frame, name, mock.Mock())

    def test_exclusion_is_reversible_and_preserves_saved_work_and_current_selection(self):
        frame = self.frame
        saved = frame.completed["01"] = object()
        frame._toggle_current_roi_exclusion()
        self.assertEqual(frame.excluded_rois, {"01"})
        self.assertEqual(frame.roi_clicks["01"], [78.0, 101.0])
        self.assertIs(frame.completed["01"], saved)
        self.assertEqual(frame.current_roi_idx, 0)
        frame.after_cancel.assert_called_once_with("pending-save")
        self.assertIsNone(frame._auto_save_after_id)
        self.assertTrue(all(value == "" for value in frame._roi_metric_values(frame.rois[0])))
        self.assertTrue(frame._can_change_roi_exclusion())

        frame._toggle_current_roi_exclusion()
        self.assertEqual(frame.excluded_rois, set())
        self.assertIs(frame.completed["01"], saved)
        self.assertEqual(frame.auto_detection_reviews["01"], "Needs review")
        self.assertEqual(frame.current_roi_idx, 0)
        self.assertTrue(frame._can_change_roi_exclusion())

    def test_excluding_missing_result_satisfies_only_its_own_readiness_slot(self):
        frame = self.frame
        frame._toggle_current_roi_exclusion()
        self.assertFalse(frame._all_rois_completed())
        frame.completed["02"] = object()
        self.assertTrue(frame._all_rois_completed())
        frame._toggle_current_roi_exclusion()
        self.assertFalse(frame._all_rois_completed())

    def test_all_excluded_rois_are_ready_without_fabricated_results(self):
        frame = self.frame
        frame.excluded_rois = {"01", "02"}
        self.assertTrue(frame._all_rois_completed())
        self.assertEqual(frame.completed, {})
        self.assertIsNone(frame._roi_overview_result(0))
        frame._update_profile_status(None)
        self.assertIn("blank in Excel", frame.profile_status_var.set.call_args.args[0])

    def test_busy_or_finished_files_cannot_change_exclusions(self):
        frame = self.frame
        for flag in ("_stack_building", "_stack_build_complete", "_batch_roi_loading_surface"):
            with self.subTest(flag=flag):
                setattr(frame, flag, True)
                frame._toggle_current_roi_exclusion()
                self.assertEqual(frame.excluded_rois, set())
                self.assertFalse(frame._can_change_roi_exclusion())
                delattr(frame, flag)

    def test_exclusion_marks_roi_edited_during_pending_detection(self):
        frame = self.frame
        frame._auto_detecting = True
        frame._auto_detection_edited_suffixes = set()
        frame._toggle_current_roi_exclusion()
        self.assertIn("01", frame._auto_detection_edited_suffixes)

    def test_toggling_another_tile_keeps_selection_and_pending_save_for_current_roi(self):
        frame = self.frame
        frame._toggle_roi_exclusion(1)
        self.assertEqual(frame.excluded_rois, {"02"})
        self.assertEqual(frame.current_roi_idx, 0)
        self.assertEqual(frame.profile_clicks, [78.0, 101.0])
        self.assertEqual(frame._auto_save_after_id, "pending-save")
        frame.after_cancel.assert_not_called()

    def test_exclusions_do_not_leak_between_active_and_inactive_tab_readiness(self):
        frame = self.frame
        frame._active_batch_roi_tab = "nasal"
        frame.excluded_rois = {"01"}
        frame.completed = {"02": object()}
        nasal = {"complete": False, "completed": {}}
        temporal = {"complete": False, "completed": {"02": object()}}
        self.assertTrue(frame._batch_tab_stack_ready("nasal", nasal))
        self.assertFalse(frame._batch_tab_stack_ready("temporal", temporal))
        temporal["excluded_rois"] = {"01"}
        self.assertTrue(frame._batch_tab_stack_ready("temporal", temporal))
        temporal["complete"] = True
        self.assertFalse(frame._batch_tab_stack_ready("temporal", temporal))

    def test_tab_snapshot_copies_exclusions_so_later_changes_do_not_mutate_it(self):
        frame = self.frame
        frame._active_batch_roi_tab = "nasal"
        state = frame.batch_roi_tab_states["nasal"] = {}
        frame.excluded_rois = {"01"}
        for field in (
            "volume", "canvas", "figure", "ax_profile", "ax_roi_grid", "empty_placeholder",
            "_current_profile", "_last_results_dir",
        ):
            setattr(frame, field, None)
        frame.current_stem = "_flat_LIGHT"
        frame._roi_adjust_bounds = {}
        frame.output_dir_var = SimpleNamespace(get=lambda: "subject/nasal")
        Step4Frame._sync_active_batch_roi_state(frame)
        frame.excluded_rois.clear()
        self.assertEqual(state["excluded_rois"], {"01"})


if __name__ == "__main__":
    unittest.main()
