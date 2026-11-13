from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from aidas.steps.step2_annotate import Step2Frame
from aidas.utils.io_utils import read_analyze, write_analyze


class Step2DeletedOutputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.frame = Step2Frame.__new__(Step2Frame)
        frame = self.frame
        frame.current_file = str(self.folder / "Light.img")
        frame.image_data = np.arange(32, dtype=np.int16).reshape(4, 8)
        write_analyze(str(self.folder / "Light"), np.stack([frame.image_data] * 2))
        frame._current_source_snapshot = frame._source_file_snapshot(frame.current_file)
        frame.boundary_traces = {"RPE": {"points": [(0, 1), (7, 1)]}}
        frame.boundary_order = ["RPE"]
        frame.fovea_x = 4
        frame._input_analyze_template = None
        frame._source_was_8bit = False
        frame.boundary_completion_vars = {}
        frame.is_step3_folder_active = None
        frame._batch_result_states = {}
        frame._active_batch_result_tab = None
        frame._sync_active_batch_result_state = mock.Mock()
        frame._all_required_boundaries_complete = mock.Mock(return_value=True)
        frame.status_var = mock.Mock()
        marked = np.stack([frame.image_data] * 2).astype(np.uint8)
        frame._build_current_marked_volume = mock.Mock(return_value=marked)
        frame._build_marked_volume = mock.Mock(return_value=marked)

    def test_deleted_source_prevents_every_cached_save_and_side_folder_creation(self):
        self.frame._save_current_marked_orientation_pair()
        for side in ("nasal", "temporal"):
            shutil.rmtree(self.folder / side)
        for suffix in (".hdr", ".img"):
            (self.folder / ("Light" + suffix)).unlink()
        for name in ("_save_current_marked_orientation_pair", "_save_current_marked_image",
                     "_save_marked_images", "_save_light_image"):
            with self.subTest(save=name), self.assertRaisesRegex(RuntimeError, "removed or changed"):
                getattr(self.frame, name)()
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_partial_analyze_pair_cannot_restore_side_folders(self):
        (self.folder / "Light.hdr").unlink()
        with self.assertRaisesRegex(RuntimeError, "removed or changed"):
            self.frame._save_current_marked_orientation_pair()
        self.assertEqual([path.name for path in self.folder.iterdir()], ["Light.img"])

    def test_replaced_source_requires_fresh_editor_but_can_be_processed_again(self):
        replacement = self.frame.image_data + 100
        for suffix in (".hdr", ".img"):
            (self.folder / ("Light" + suffix)).unlink()
        write_analyze(str(self.folder / "Light"), np.stack([replacement] * 2))
        with self.assertRaisesRegex(RuntimeError, "removed or changed"):
            self.frame._save_current_marked_orientation_pair()
        self.assertFalse((self.folder / "nasal").exists())
        self.frame.image_data = replacement
        self.frame._current_source_snapshot = self.frame._source_file_snapshot(self.frame.current_file)
        self.frame._save_current_marked_orientation_pair()
        self.assertTrue((self.folder / "nasal" / "Light_MARKED.img").is_file())
        np.testing.assert_array_equal(read_analyze(str(self.folder / "Light"))[0], replacement)

    def test_deleting_only_outputs_allows_deliberate_resave_with_live_source(self):
        first = self.frame._save_current_marked_orientation_pair()
        expected = {str(path.relative_to(self.folder)): path.read_bytes()
                    for side in ("nasal", "temporal") for path in (self.folder / side).iterdir()}
        for side in ("nasal", "temporal"):
            shutil.rmtree(self.folder / side)
        self.assertEqual(self.frame._save_current_marked_orientation_pair(), first)
        for name, content in expected.items():
            self.assertEqual((self.folder / name).read_bytes(), content)

    def test_inactive_tab_cannot_save_deleted_source_and_preserves_active_context(self):
        frame = self.frame
        state = frame._capture_current_editor_state()
        frame._batch_result_states["old"] = state
        frame._active_batch_result_tab = "active"
        frame.current_file = "active source"
        frame._current_source_snapshot = ("active snapshot",)
        (self.folder / "Light.img").unlink()
        with self.assertRaisesRegex(RuntimeError, "removed or changed"):
            frame._save_batch_result_state("old", save_orientation_pair=True)
        self.assertEqual(frame.current_file, "active source")
        self.assertEqual(frame._current_source_snapshot, ("active snapshot",))
        self.assertFalse((self.folder / "nasal").exists())

    def test_pending_step1_handoff_is_discarded_if_its_crop_was_deleted(self):
        frame = self.frame
        frame.batch_segmentation_panel = None
        frame._show_image = mock.Mock()
        frame.load_external_image(frame.image_data, frame.current_file, defer_render=True)
        (self.folder / "Light.img").unlink()
        self.assertFalse(frame.render_pending_external_image())
        self.assertIsNone(frame._pending_external_image)
        frame._show_image.assert_not_called()

    def test_scan_unlocks_after_side_results_are_deleted_and_checks_both_pair_members(self):
        frame = self.frame
        frame._save_current_marked_orientation_pair()
        self.assertTrue(frame._step2_batch_segmentation_row(self.folder)["locked"])
        (self.folder / "nasal" / "Light_MARKED.hdr").unlink()
        self.assertFalse(frame._step2_batch_segmentation_row(self.folder)["locked"])
        for side in ("nasal", "temporal"):
            shutil.rmtree(self.folder / side)
        row = frame._step2_batch_segmentation_row(self.folder)
        self.assertTrue(row["include"])
        self.assertEqual(row["status"], "Ready")
        self.assertFalse((self.folder / "nasal").exists())

    def test_saving_original_light_updates_its_snapshot_for_another_explicit_save(self):
        retained = self.frame._capture_current_editor_state()
        self.frame._batch_result_states["retained"] = retained
        self.frame._single_editor_state = self.frame._capture_current_editor_state()
        self.frame._save_marked_images()
        self.assertEqual(retained["source_snapshot"], self.frame._current_source_snapshot)
        self.assertEqual(self.frame._single_editor_state["source_snapshot"], self.frame._current_source_snapshot)
        self.frame._save_marked_images()
        self.frame._save_batch_result_state("retained", save_orientation_pair=True)
        self.assertTrue((self.folder / "temporal" / "Light_MARKED.hdr").is_file())

    def test_source_removed_during_ai_prediction_does_not_become_a_saveable_result(self):
        frame = self.frame
        frame.after = lambda _delay, callback: callback()
        for name in ("_aidas_worker_command", "_aidas_worker_env", "_hidden_subprocess_kwargs",
                     "_set_segmenter_progress_value", "_set_segmentation_status", "_append_segmenter_log",
                     "_set_runtime_ai_device_status", "_write_segmenter_log_file",
                     "_on_aidas_batch_segmenter_worker_done"):
            setattr(frame, name, mock.Mock())

        def prediction(*_args, **_kwargs):
            (self.folder / "Light.img").unlink()
            return {"boundaries": np.ones((6, 8))}

        with mock.patch("aidas.steps.step2_annotate.AIWorkerClient") as worker_class:
            worker = worker_class.return_value
            worker.startup_result = {}
            worker.predict.side_effect = prediction
            frame._run_aidas_batch_segmenter_worker(
                [frame.current_file], "model.onnx", "auto", 0, 1,
                {frame._image_pair_key(frame.current_file): 4}, {},
            )
        result = frame._on_aidas_batch_segmenter_worker_done.call_args.args[0]
        self.assertEqual(result["results"], [])
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("removed or changed", result["failures"][0]["error"])
        self.assertFalse((self.folder / "nasal").exists())


if __name__ == "__main__":
    unittest.main()
