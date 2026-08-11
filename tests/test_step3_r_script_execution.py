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

    @staticmethod
    def _batch_config(folder):
        folder = Path(folder)
        return {
            "input_dir": str(folder),
            "output_dir": str(folder),
            "reference_dark": "DARK_MARKED",
            "reference_light": "Light_MARKED",
            "to_process_dark": "DARK",
            "to_process_light": "LIGHT",
            "image_index_light": "1:2",
            "image_index_dark": "1:2",
            "pixel_width": "3.89",
        }

    def _make_batch_frame(self):
        frame = self._make_frame()
        finished = []
        frame._folder_has_r_data = lambda _folder: False
        frame._r_script_config_for_folder = self._batch_config
        frame._write_r_run_log = lambda output_dir, *_args: Path(output_dir) / "step3.log"
        frame._r_env = lambda thread_limit=None: {}
        frame._r_worker_limit = lambda: 8
        frame._r_threads_per_process = lambda _workers: 1
        frame._on_batch_r_done = finished.extend
        return frame, finished

    @staticmethod
    def _write_required_exports(folder):
        folder = Path(folder)
        for suffix in ("DARK", "LIGHT"):
            (folder / f"_thickness_vs_distance_from_fovea_{suffix}.txt").write_text(
                suffix.lower(),
                encoding="utf-8",
            )

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

    def test_parallel_output_is_default_and_keeps_the_per_folder_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folders = [Path(temp_dir) / "folder-a", Path(temp_dir) / "folder-b"]
            for folder in folders:
                folder.mkdir()

            frame, finished = self._make_batch_frame()
            events = []
            event_lock = threading.Lock()
            main_barrier = threading.Barrier(2)
            output_barrier = threading.Barrier(2)
            output_started = threading.Event()
            output_activity = {"active": 0, "maximum": 0}

            def record(stage, action, folder):
                with event_lock:
                    events.append((stage, action, folder.name))

            def fake_run(command, cwd, _env, _timeout, _on_line):
                folder = Path(cwd)
                is_output = "-e" in command
                if not is_output:
                    record("main", "start", folder)
                    main_barrier.wait(timeout=3)
                    if folder.name == "folder-b":
                        # The default per-folder pipeline lets folder A start its
                        # output while folder B is still finishing its main script.
                        output_started.wait(timeout=0.5)
                    record("main", "end", folder)
                    return 0, "", "completed"

                record("output", "start", folder)
                output_started.set()
                with event_lock:
                    output_activity["active"] += 1
                    output_activity["maximum"] = max(
                        output_activity["maximum"],
                        output_activity["active"],
                    )
                try:
                    output_barrier.wait(timeout=3)
                    self._write_required_exports(folder)
                finally:
                    with event_lock:
                        output_activity["active"] -= 1
                record("output", "end", folder)
                return 0, "", "completed"

            frame._run_supervised_r_command = fake_run
            frame._batch_r_worker(
                Path("Rscript.exe"),
                Path("main.R"),
                Path("output.R"),
                folders,
                workers=2,
                timeout_seconds=60,
            )

            self.assertLess(
                events.index(("output", "start", "folder-a")),
                events.index(("main", "end", "folder-b")),
            )
            for folder in folders:
                with self.subTest(folder=folder.name):
                    self.assertLess(
                        events.index(("main", "end", folder.name)),
                        events.index(("output", "start", folder.name)),
                    )
            self.assertEqual(output_activity["maximum"], 2)
            self.assertEqual(len(finished), 2)
            self.assertTrue(all(result["outcome"] == "completed" for result in finished))

    def test_sequential_output_keeps_main_parallel_and_outputs_nonoverlapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folders = [Path(temp_dir) / "folder-a", Path(temp_dir) / "folder-b"]
            for folder in folders:
                folder.mkdir()

            frame, finished = self._make_batch_frame()
            events = []
            event_lock = threading.Lock()
            main_barrier = threading.Barrier(2)
            output_started = threading.Event()
            second_output_started = threading.Event()
            output_activity = {"active": 0, "maximum": 0, "calls": 0, "overlapped": False}

            def record(stage, action, folder):
                with event_lock:
                    events.append((stage, action, folder.name))

            def fake_run(command, cwd, _env, _timeout, _on_line):
                folder = Path(cwd)
                is_output = "-e" in command
                if not is_output:
                    record("main", "start", folder)
                    main_barrier.wait(timeout=3)
                    if folder.name == "folder-b":
                        output_started.wait(timeout=0.5)
                    record("main", "end", folder)
                    return 0, "", "completed"

                output_started.set()
                record("output", "start", folder)
                with event_lock:
                    output_activity["calls"] += 1
                    call_number = output_activity["calls"]
                    output_activity["active"] += 1
                    output_activity["maximum"] = max(
                        output_activity["maximum"],
                        output_activity["active"],
                    )
                    if call_number == 2:
                        second_output_started.set()
                if call_number == 1:
                    output_activity["overlapped"] = second_output_started.wait(timeout=0.5)
                self._write_required_exports(folder)
                with event_lock:
                    output_activity["active"] -= 1
                record("output", "end", folder)
                return 0, "", "completed"

            frame._run_supervised_r_command = fake_run
            frame._batch_r_worker(
                Path("Rscript.exe"),
                Path("main.R"),
                Path("output.R"),
                folders,
                workers=2,
                timeout_seconds=60,
                output_mode="sequential",
            )

            last_main_end = max(
                index
                for index, event in enumerate(events)
                if event[0:2] == ("main", "end")
            )
            first_output_start = min(
                index
                for index, event in enumerate(events)
                if event[0:2] == ("output", "start")
            )
            self.assertLess(last_main_end, first_output_start)
            self.assertEqual(output_activity["calls"], 2)
            self.assertEqual(output_activity["maximum"], 1)
            self.assertFalse(output_activity["overlapped"])
            self.assertEqual(len(finished), 2)
            self.assertTrue(all(result["outcome"] == "completed" for result in finished))

    def test_failed_main_script_skips_output_for_only_that_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folders = [Path(temp_dir) / "folder-a", Path(temp_dir) / "folder-b"]
            for folder in folders:
                folder.mkdir()

            frame, finished = self._make_batch_frame()
            main_barrier = threading.Barrier(2)
            output_folders = []

            def fake_run(command, cwd, _env, _timeout, _on_line):
                folder = Path(cwd)
                if "-e" not in command:
                    main_barrier.wait(timeout=3)
                    if folder.name == "folder-a":
                        return 7, "main failed", "failed"
                    return 0, "", "completed"
                output_folders.append(folder.name)
                self._write_required_exports(folder)
                return 0, "", "completed"

            frame._run_supervised_r_command = fake_run
            frame._batch_r_worker(
                Path("Rscript.exe"),
                Path("main.R"),
                Path("output.R"),
                folders,
                workers=2,
                timeout_seconds=60,
                output_mode="sequential",
            )

            results_by_folder = {Path(result["folder"]).name: result for result in finished}
            self.assertEqual(output_folders, ["folder-b"])
            self.assertEqual(results_by_folder["folder-a"]["outcome"], "failed")
            self.assertEqual(results_by_folder["folder-b"]["outcome"], "completed")

    def test_cancellation_skips_queued_sequential_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folders = [Path(temp_dir) / f"folder-{name}" for name in ("a", "b", "c")]
            for folder in folders:
                folder.mkdir()

            frame, finished = self._make_batch_frame()
            main_barrier = threading.Barrier(3)
            output_folders = []

            def fake_run(command, cwd, _env, _timeout, _on_line):
                folder = Path(cwd)
                if "-e" not in command:
                    main_barrier.wait(timeout=3)
                    return 0, "", "completed"
                output_folders.append(folder.name)
                if len(output_folders) == 1:
                    frame._r_cancel_event.set()
                    return 130, "Cancelled by user.", "cancelled"
                self._write_required_exports(folder)
                return 0, "", "completed"

            frame._run_supervised_r_command = fake_run
            frame._batch_r_worker(
                Path("Rscript.exe"),
                Path("main.R"),
                Path("output.R"),
                folders,
                workers=3,
                timeout_seconds=60,
                output_mode="sequential",
            )

            self.assertEqual(len(output_folders), 1)
            self.assertEqual(len(finished), 3)
            self.assertEqual(
                {result["outcome"] for result in finished},
                {"cancelled"},
            )

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

    def test_selection_forwards_the_selected_output_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            folders = [root / "folder-a", root / "folder-b"]
            main_script = root / "main.R"
            output_script = root / "output.R"
            main_script.write_text("# main", encoding="utf-8")
            output_script.write_text("# output", encoding="utf-8")

            panel = RBatchSelectionPanel.__new__(RBatchSelectionPanel)
            panel.table = mock.Mock()
            panel.table.selected_rows.return_value = [
                {"folder": folder} for folder in folders
            ]
            panel.step_frame = mock.Mock()
            panel.step_frame._selected_r_script_path.side_effect = [main_script, output_script]
            panel.step_frame._normalize_r_output_mode.side_effect = (
                Step3Frame._normalize_r_output_mode
            )
            panel.workers_var = mock.Mock()
            panel.workers_var.get.return_value = 2
            panel.timeout_var = mock.Mock()
            panel.timeout_var.get.return_value = 5
            panel.output_mode_var = mock.Mock()
            panel.output_mode_var.get.return_value = "sequential"
            panel._max_worker_count = lambda _ready_count: 2

            panel._run_selected()

            panel.step_frame._start_batch_r_runs.assert_called_once_with(
                folders,
                2,
                main_script,
                output_script,
                300,
                output_mode="sequential",
            )

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
            output_mode="sequential",
        )

        frame._cancel_batch_r_runs.assert_called_once_with()
        self.assertIsNotNone(frame._pending_batch_restart)
        self.assertEqual(frame._pending_batch_restart[0], [Path("folder-a")])
        self.assertEqual(frame._pending_batch_restart[-1], "sequential")

    def test_run_panel_restart_preserves_the_output_mode(self):
        panel = RBatchRunPanel.__new__(RBatchRunPanel)
        panel.folders = [Path("folder-a")]
        panel.workers = 2
        panel.main_script_path = Path("main.R")
        panel.output_script_path = Path("output.R")
        panel.timeout_seconds = 60
        panel.output_mode = "sequential"
        panel.restart_button = mock.Mock()
        panel.stop_button = mock.Mock()
        panel.close_button = mock.Mock()
        panel.summary_var = mock.Mock()
        panel.step_frame = mock.Mock()
        panel.step_frame._busy = False

        with mock.patch("aidas.steps.step3_flatten.messagebox.askyesno", return_value=True):
            panel._restart_batch()

        call = panel.step_frame._restart_batch_r_runs.call_args
        self.assertEqual(
            call.args[:5],
            (
                panel.folders,
                2,
                Path("main.R"),
                Path("output.R"),
                60,
            ),
        )
        forwarded_mode = call.kwargs.get(
            "output_mode",
            call.args[5] if len(call.args) > 5 else None,
        )
        self.assertEqual(forwarded_mode, "sequential")

    def test_completed_batch_starts_pending_restart_without_an_event_loop_gap(self):
        restart = (
            [Path("folder-a")],
            2,
            Path("main.R"),
            Path("output.R"),
            60,
            "sequential",
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
            self.assertEqual(Step3Frame._r_worker_limit(), 12)

    def test_batch_reports_job_limit_from_the_shared_core_budget(self):
        panel = RBatchSelectionPanel.__new__(RBatchSelectionPanel)
        panel.step_frame = mock.Mock()
        panel.step_frame._available_r_worker_limit.return_value = 7
        panel.step_frame._step2_core_allocation.return_value = 0

        self.assertEqual(panel._max_worker_count(), 7)
        self.assertEqual(panel._max_worker_count(20), 7)
        self.assertEqual(panel._max_worker_count(3), 3)
        self.assertEqual(
            panel._worker_limit_text(7),
            "Max jobs: 7 (shared with Step 2)",
        )
        self.assertEqual(panel._worker_limit_text(1), "Max jobs: 1 (shared with Step 2)")

        panel.step_frame._step2_core_allocation.return_value = 3
        self.assertEqual(
            panel._worker_limit_text(4),
            "Max jobs: 4 (3 cores used by Step 2)",
        )

    def test_r_process_thread_budget_prevents_nested_cpu_oversubscription(self):
        with mock.patch.object(Step3Frame, "_cpu_worker_limit", return_value=8):
            self.assertEqual(Step3Frame._r_threads_per_process(1), 1)
            self.assertEqual(Step3Frame._r_threads_per_process(2), 1)
            self.assertEqual(Step3Frame._r_threads_per_process(8), 1)

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

    def test_step3_allocation_subtracts_active_step2_cores(self):
        frame = Step3Frame.__new__(Step3Frame)
        frame.get_step2_core_usage = lambda: 3
        frame._busy = True
        frame._active_r_core_allocation = 4

        with mock.patch.object(Step3Frame, "_cpu_worker_limit", return_value=8):
            self.assertEqual(frame._step2_core_allocation(), 3)
            self.assertEqual(frame._available_r_worker_limit(), 5)
            self.assertEqual(frame.active_core_allocation(), 4)
            frame._busy = False
            self.assertEqual(frame.active_core_allocation(), 0)


if __name__ == "__main__":
    unittest.main()
