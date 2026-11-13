from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from aidas.steps.step3_flatten import Step3Frame
from aidas.utils.io_utils import write_analyze


class Step3DeletedOutputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.frame = Step3Frame.__new__(Step3Frame)
        frame = self.frame
        frame._r_cancel_event = threading.Event()
        frame._register_r_process = mock.Mock()
        frame._unregister_r_process = mock.Mock()
        frame._terminate_r_process = mock.Mock()
        frame._batch_panel_update = mock.Mock()
        frame.after = lambda _delay, callback: callback()
        frame._r_env = lambda **_kwargs: {}
        frame._r_worker_limit = lambda: 1
        frame._r_threads_per_process = lambda _workers: 1
        self.finished = []
        frame._on_batch_r_done = self.finished.extend
        frame._write_r_run_log = lambda output, *_args: self.root / "test-run.log"

    def _inputs(self, name):
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        for basename, dtype in (("Light", np.int16), ("Light_MARKED", np.uint8)):
            write_analyze(str(folder / basename), np.ones((2, 4, 8), dtype=dtype))
        return folder

    def test_deleted_queued_folder_is_revalidated_and_fresh_inputs_run_again(self):
        frame = self.frame
        for mode in (frame.R_OUTPUT_MODE_PARALLEL, frame.R_OUTPUT_MODE_SEQUENTIAL):
            with self.subTest(mode=mode):
                folder = self._inputs(mode)
                # The selector previously found these valid inputs.
                frame._r_script_config_for_folder(folder)
                shutil.rmtree(folder)
                frame._run_r_script_for_config = mock.Mock(return_value={
                    "folder": folder, "returncode": 0, "outcome": "completed"})
                frame._run_main_r_script_for_config = frame._run_r_script_for_config
                frame._run_output_r_script_for_config = frame._run_r_script_for_config
                frame._finalize_r_script_result = lambda _config, result, **_kwargs: result
                self.finished.clear()
                frame._batch_r_worker(Path("Rscript"), Path("main.R"), Path("out.R"),
                                      [folder], workers=1, output_mode=mode)
                frame._run_r_script_for_config.assert_not_called()
                self.assertFalse(folder.exists())
                self.assertEqual(self.finished[0]["outcome"], "failed")
                self.assertIn("Missing Step 3 inputs", self.finished[0]["stderr"])
                self._inputs(mode)
                self.finished.clear()
                frame._batch_r_worker(Path("Rscript"), Path("main.R"), Path("out.R"),
                                      [folder], workers=1, output_mode=mode)
                self.assertEqual(self.finished[0]["outcome"], "completed")

    def test_main_and_output_jobs_reject_a_source_deleted_after_validation(self):
        frame = self.frame
        folder = self._inputs("nasal")
        config = frame._r_script_config_for_folder(folder)
        shutil.rmtree(folder)
        with mock.patch("aidas.steps.step3_flatten.subprocess.Popen") as popen:
            main = frame._run_main_r_script_for_config("Rscript", "main.R", config)
            output = frame._run_output_r_script_for_config(
                "Rscript", "output.R", config, {"returncode": 0})
        popen.assert_not_called()
        for result in (main, output):
            self.assertEqual(result["outcome"], "failed")
            self.assertIn("removed or changed", result["stderr"])
        self.assertFalse(folder.exists())

    def test_replaced_inputs_do_not_start_output_for_the_old_main_result(self):
        folder = self._inputs("temporal")
        config = self.frame._r_script_config_for_folder(folder)
        shutil.rmtree(folder)
        self._inputs("temporal")
        with mock.patch("aidas.steps.step3_flatten.subprocess.Popen") as popen:
            result = self.frame._run_output_r_script_for_config(
                "Rscript", "output.R", config, {"returncode": 0})
        popen.assert_not_called()
        self.assertEqual(result["outcome"], "failed")

    def test_active_r_process_is_stopped_if_its_inputs_disappear_and_can_retry(self):
        folder = self._inputs("active")
        config = self.frame._r_script_config_for_folder(folder)
        process = mock.Mock(stdout=iter(()))
        process.wait.return_value = 0
        process.poll.return_value = 0

        def launch(*_args, **_kwargs):
            (folder / "Light.img").unlink()
            return process

        with mock.patch("aidas.steps.step3_flatten.subprocess.Popen", side_effect=launch):
            code, error, outcome = self.frame._run_supervised_r_command(
                ["Rscript"], folder, {}, mock.Mock(), input_snapshot=config["_input_snapshot"])
        self.assertEqual((code, outcome), (1, "failed"))
        self.assertIn("removed or changed", error)
        self.frame._terminate_r_process.assert_called_once_with(process)
        self.frame._unregister_r_process.assert_called_once_with(process)
        self._inputs("active")
        config = self.frame._r_script_config_for_folder(folder)
        process.stdout = iter(())
        with mock.patch("aidas.steps.step3_flatten.subprocess.Popen", return_value=process):
            self.assertEqual(self.frame._run_supervised_r_command(
                ["Rscript"], folder, {}, mock.Mock(), input_snapshot=config["_input_snapshot"]),
                (0, "", "completed"))

    def test_deleted_inputs_cannot_be_published_as_successful_results(self):
        folder = self._inputs("finished")
        config = self.frame._r_script_config_for_folder(folder)
        shutil.rmtree(folder)
        with mock.patch("aidas.services.step4_compiler.create_cleaned_light_thickness_workbook") as export:
            result = self.frame._finalize_r_script_result(
                config, {"returncode": 0}, validate_exports=True)
        export.assert_not_called()
        self.assertEqual(result["outcome"], "failed")
        self.assertIn("removed or changed", result["stderr"])
        self.assertFalse(folder.exists())

    def _prepare_handoff(self, folders):
        frame = self.frame
        for folder in folders:
            for suffix in (".hdr", ".img"):
                (folder / ("_flat_LIGHT" + suffix)).write_bytes(b"flattened")
        frame.step4_result_folders = list(folders)
        frame._busy = False
        frame.on_continue_to_step4 = mock.Mock()
        frame.continue_to_step4_button = object()
        frame._set_control_enabled = mock.Mock()
        frame.progress_text_var = mock.Mock()
        frame._refresh_input_status = mock.Mock()
        return frame

    def test_reentry_clears_deleted_flattened_completion_before_refreshing_inputs(self):
        folder = self._inputs("completed")
        frame = self._prepare_handoff([folder])
        (folder / "_flat_LIGHT.img").unlink()
        notifications = []
        frame._refresh_input_status.side_effect = lambda: notifications.append("refresh")
        with mock.patch("aidas.steps.step3_flatten.notify_workflow_invalidated",
                        side_effect=lambda *_args: notifications.append("invalidated")):
            frame.on_show()
            frame.on_show()
        self.assertEqual(notifications, ["invalidated", "refresh", "refresh"])
        self.assertEqual(frame.step4_result_folders, [])
        frame._set_control_enabled.assert_called_with(frame.continue_to_step4_button, False)
        self.assertIsNotNone(frame._find_input_paths(folder)["LIGHT"])

    def test_partial_deleted_batch_invalidates_completion_and_hands_off_remaining_folders(self):
        first = self._inputs("nasal")
        remaining = self._inputs("temporal")
        frame = self._prepare_handoff([first, remaining])
        (first / "_flat_LIGHT.hdr").unlink()
        with mock.patch("aidas.steps.step3_flatten.notify_workflow_invalidated") as invalidated:
            frame._continue_to_step4()
        invalidated.assert_called_once_with(frame, 3)
        self.assertEqual(frame.step4_result_folders, [remaining.resolve()])
        frame.on_continue_to_step4.assert_called_once_with([str(remaining.resolve())])
        frame._set_control_enabled.assert_called_with(frame.continue_to_step4_button, True)

    def test_deleted_handoff_disables_continue_and_invalidates_previous_completion(self):
        folder = self._inputs("deleted")
        frame = self._prepare_handoff([folder])
        shutil.rmtree(folder)
        with mock.patch("aidas.steps.step3_flatten.notify_workflow_invalidated") as invalidated, \
                mock.patch("aidas.steps.step3_flatten.messagebox.showwarning") as warning:
            frame._continue_to_step4()
        invalidated.assert_called_once_with(frame, 3)
        warning.assert_called_once()
        frame.on_continue_to_step4.assert_not_called()
        self.assertEqual(frame.step4_result_folders, [])
        frame._set_control_enabled.assert_called_with(frame.continue_to_step4_button, False)

    def test_fresh_and_historical_sessions_do_not_invalidate_valid_completion(self):
        frame = self._prepare_handoff([])
        with mock.patch("aidas.steps.step3_flatten.notify_workflow_invalidated") as invalidated:
            frame.on_show()
            historical = self.root / "historical"
            historical.mkdir()
            self._prepare_handoff([historical])
            frame.on_show()
            frame._continue_to_step4()
        invalidated.assert_not_called()
        frame.on_continue_to_step4.assert_called_once_with([str(historical.resolve())])


if __name__ == "__main__":
    unittest.main()
