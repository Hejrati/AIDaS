from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from aidas.steps.step3_flatten import RBatchRunPanel, RBatchSelectionPanel, Step3Frame


class _FakeProcess:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self._returncode = returncode

    def wait(self):
        return self._returncode

    def poll(self):
        return self._returncode


class _BlockingOutput:
    def __init__(self, stopped):
        self.stopped = stopped

    def __iter__(self):
        return self

    def __next__(self):
        self.stopped.wait()
        raise StopIteration


class _BlockingProcess:
    pid = None

    def __init__(self):
        self.stopped = threading.Event()
        self.stdout = _BlockingOutput(self.stopped)
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.stopped.set()

    def kill(self):
        self.returncode = -9
        self.stopped.set()

    def wait(self, timeout=None):
        if not self.stopped.wait(timeout):
            raise TimeoutError("process did not stop")
        return self.returncode


class Step3RScriptExecutionTests(unittest.TestCase):
    @staticmethod
    def _make_frame():
        frame = Step3Frame.__new__(Step3Frame)
        frame.r_package_library_path = None
        frame.after = lambda _delay, callback: callback()
        frame._batch_panel_update = lambda *_args, **_kwargs: None
        frame._r_cancel_event = threading.Event()
        frame._r_process_lock = threading.Lock()
        frame._active_r_processes = set()
        return frame

    def test_selected_main_and_output_scripts_run_in_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            log_dir = root / "logs"
            input_dir.mkdir()
            output_dir.mkdir()
            log_dir.mkdir()
            main_script = root / "custom-main.R"
            output_script = root / "custom-output.R"
            main_script.write_text("# main", encoding="utf-8")
            output_script.write_text("# output", encoding="utf-8")

            frame = self._make_frame()

            r_config = {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "reference_dark": "DARK_MARKED",
                "reference_light": "Light_MARKED",
                "to_process_dark": "DARK",
                "to_process_light": "LIGHT",
                "image_index_light": "1:2",
                "image_index_dark": "1:2",
                "pixel_width": "3.89",
            }
            commands = []

            def fake_popen(command, **_kwargs):
                commands.append(command)
                if len(commands) == 2:
                    (output_dir / "_thickness_vs_distance_from_fovea_DARK.txt").write_text(
                        "dark", encoding="utf-8"
                    )
                    (output_dir / "_thickness_vs_distance_from_fovea_LIGHT.txt").write_text(
                        "light", encoding="utf-8"
                    )
                return _FakeProcess([])

            with mock.patch("aidas.steps.step3_flatten.subprocess.Popen", side_effect=fake_popen), mock.patch(
                "aidas.steps.step3_flatten.app_log_dir", return_value=log_dir
            ):
                result = frame._run_r_script_for_config(
                    Path("Rscript.exe"),
                    main_script,
                    output_script,
                    r_config,
                )

            self.assertEqual(result["returncode"], 0)
            self.assertEqual(len(commands), 2)
            self.assertEqual(Path(commands[0][2]), main_script)
            self.assertEqual(commands[1][2], "-e")
            self.assertIn(output_script.resolve().as_posix(), commands[1][3])
            self.assertIn(Step3Frame.R_WORKSPACE_FILES[1], commands[1][3])

    def test_step3_requires_only_the_app_light_inputs(self):
        self.assertEqual(
            tuple(label for label, *_rest in Step3Frame.REQUIRED_INPUTS),
            ("Light_MARKED", "LIGHT"),
        )

    def test_r_config_maps_light_into_the_legacy_dark_slots(self):
        frame = self._make_frame()
        input_paths = {
            "Light_MARKED": r"C:\input\subject_Light_MARKED",
            "LIGHT": r"C:\input\subject_LIGHT",
        }
        input_info = {
            "Light_MARKED": {"shape": (3, 128, 1473), "bits": 8},
            "LIGHT": {"shape": (3, 128, 1473), "bits": 16},
        }
        frame._find_input_paths = lambda _folder: input_paths
        frame._read_input_stack_info = lambda _paths: input_info

        config = frame._r_script_config_for_folder(Path("subject"))

        self.assertEqual(config["reference_dark"], "DARK_MARKED")
        self.assertEqual(config["reference_light"], "subject_Light_MARKED")
        self.assertEqual(config["to_process_dark"], "DARK")
        self.assertEqual(config["to_process_light"], "subject_LIGHT")
        self.assertEqual(config["image_index_dark"], "1,2,3")
        self.assertEqual(config["image_index_light"], "1,2,3")

    def test_shape_validation_requires_matching_light_and_marked_stacks(self):
        valid = {
            "Light_MARKED": {"shape": (3, 128, 1473)},
            "LIGHT": {"shape": (3, 128, 1473)},
        }
        self.assertEqual(Step3Frame._validate_input_stack_shapes(valid)["LIGHT"], (3, 128, 1473))

        invalid = dict(valid)
        invalid["LIGHT"] = {"shape": (2, 128, 1473)}
        with self.assertRaisesRegex(ValueError, "Step 3 inputs must all have the same"):
            Step3Frame._validate_input_stack_shapes(invalid)

    def test_silent_process_is_stopped_when_timeout_expires(self):
        frame = self._make_frame()
        process = _BlockingProcess()
        popen_options = {}

        def fake_popen(_command, **kwargs):
            popen_options.update(kwargs)
            return process

        with mock.patch("aidas.steps.step3_flatten.subprocess.Popen", side_effect=fake_popen), mock.patch(
            "aidas.steps.step3_flatten.time.monotonic", side_effect=(0.0, 2.0)
        ):
            returncode, error, outcome = frame._run_supervised_r_command(
                ["Rscript.exe", "silent.R"],
                ".",
                {},
                1,
                lambda _line: None,
            )

        self.assertEqual(returncode, 124)
        self.assertEqual(outcome, "timed_out")
        self.assertIn("timeout", error)
        self.assertIs(popen_options["stdin"], subprocess.DEVNULL)
        if os.name == "nt":
            self.assertTrue(
                popen_options["creationflags"]
                & getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
            )
        self.assertTrue(process.stopped.is_set())

    def test_user_cancellation_stops_a_silent_process(self):
        frame = self._make_frame()
        frame._r_cancel_event.set()
        process = _BlockingProcess()

        with mock.patch("aidas.steps.step3_flatten.subprocess.Popen", return_value=process):
            returncode, error, outcome = frame._run_supervised_r_command(
                ["Rscript.exe", "interactive.R"],
                ".",
                {},
                60,
                lambda _line: None,
            )

        self.assertEqual(returncode, 130)
        self.assertEqual(outcome, "cancelled")
        self.assertIn("Cancelled", error)
        self.assertTrue(process.stopped.is_set())

    def test_nonzero_r_exit_is_reported_as_failure(self):
        frame = self._make_frame()
        process = _FakeProcess(["R error\n"], returncode=7)
        lines = []

        with mock.patch("aidas.steps.step3_flatten.subprocess.Popen", return_value=process):
            returncode, error, outcome = frame._run_supervised_r_command(
                ["Rscript.exe", "bad.R"],
                ".",
                {},
                60,
                lines.append,
            )

        self.assertEqual(returncode, 7)
        self.assertEqual(outcome, "failed")
        self.assertEqual(error, "")
        self.assertEqual(lines, ["R error\n"])

    def test_batch_coordinator_failure_still_releases_the_busy_lifecycle(self):
        frame = self._make_frame()
        finished = []
        frame._on_batch_r_done = finished.extend
        folders = [Path("folder-a"), Path("folder-b")]

        with mock.patch(
            "aidas.steps.step3_flatten.concurrent.futures.ThreadPoolExecutor",
            side_effect=RuntimeError("executor unavailable"),
        ):
            frame._batch_r_worker(
                Path("Rscript.exe"),
                Path("main.R"),
                Path("output.R"),
                folders,
                workers=2,
                timeout_seconds=60,
            )

        self.assertEqual([result["folder"] for result in finished], folders)
        self.assertTrue(all(result["outcome"] == "failed" for result in finished))
        self.assertTrue(all("executor unavailable" in result["stderr"] for result in finished))

    def test_stop_requires_confirmation_before_cancelling_batch(self):
        panel = RBatchRunPanel.__new__(RBatchRunPanel)
        panel.stop_requested = False
        panel.stop_button = mock.Mock()
        panel.restart_button = mock.Mock()
        panel.summary_var = mock.Mock()
        panel.step_frame = mock.Mock()

        with mock.patch("aidas.steps.step3_flatten.messagebox.askyesno", return_value=False):
            panel._cancel_batch()
        panel.step_frame._cancel_batch_r_runs.assert_not_called()

        with mock.patch("aidas.steps.step3_flatten.messagebox.askyesno", return_value=True):
            panel._cancel_batch()
        panel.step_frame._cancel_batch_r_runs.assert_called_once_with()
        self.assertTrue(panel.stop_requested)

    def test_restart_waits_for_active_batch_to_stop(self):
        frame = Step3Frame.__new__(Step3Frame)
        frame._busy = True
        frame._pending_batch_restart = None
        frame.status_var = mock.Mock()
        frame._cancel_batch_r_runs = mock.Mock()

        frame._restart_batch_r_runs(
            [Path("folder-a")],
            2,
            Path("main.R"),
            Path("output.R"),
            60,
        )

        frame._cancel_batch_r_runs.assert_called_once_with()
        self.assertIsNotNone(frame._pending_batch_restart)
        self.assertEqual(frame._pending_batch_restart[0], [Path("folder-a")])

    def test_completed_batch_starts_pending_restart_without_an_event_loop_gap(self):
        restart = (
            [Path("folder-a")],
            2,
            Path("main.R"),
            Path("output.R"),
            60,
        )
        panel = mock.Mock()
        panel.close_when_finished = False
        frame = Step3Frame.__new__(Step3Frame)
        frame._busy = True
        frame._active_r_folder_keys = {frame._folder_key("folder-a")}
        frame._pending_batch_restart = restart
        frame.r_batch_run_panel = panel
        frame.progress_text_var = mock.Mock()
        frame.status_var = mock.Mock()
        frame.info_var = mock.Mock()
        frame._set_process_buttons = mock.Mock()
        frame._start_batch_r_runs = mock.Mock()
        frame.after = mock.Mock()

        frame._on_batch_r_done([])

        frame._start_batch_r_runs.assert_called_once_with(
            *restart,
            allow_existing_rdata=True,
        )
        frame.after.assert_not_called()

    def test_cpu_worker_limit_uses_processors_available_to_the_process(self):
        with mock.patch(
            "aidas.steps.step3_flatten.os.process_cpu_count",
            create=True,
            return_value=12,
        ):
            self.assertEqual(Step3Frame._cpu_worker_limit(), 12)
            self.assertEqual(Step3Frame._r_worker_limit(), 11)

    def test_batch_reserves_a_processor_for_step2_and_the_ui(self):
        panel = RBatchSelectionPanel.__new__(RBatchSelectionPanel)
        panel.step_frame = mock.Mock()
        panel.step_frame._r_worker_limit.return_value = 7

        self.assertEqual(panel._max_worker_count(), 7)
        self.assertEqual(panel._max_worker_count(20), 7)
        self.assertEqual(panel._max_worker_count(3), 3)
        self.assertEqual(
            panel._worker_limit_text(7),
            "Max: 7 (1 CPU reserved)",
        )

    def test_r_process_thread_budget_prevents_nested_cpu_oversubscription(self):
        with mock.patch.object(Step3Frame, "_cpu_worker_limit", return_value=8):
            self.assertEqual(Step3Frame._r_threads_per_process(1), 7)
            self.assertEqual(Step3Frame._r_threads_per_process(2), 3)
            self.assertEqual(Step3Frame._r_threads_per_process(7), 1)

        frame = self._make_frame()
        frame._default_r_package_library = lambda: None
        env = frame._r_env(thread_limit=2)
        for name in (
            "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual(env[name], "2")


if __name__ == "__main__":
    unittest.main()
