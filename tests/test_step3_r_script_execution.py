from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from aidas.steps.step3_flatten import Step3Frame


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
            self.assertIn(str(output_script).replace("\\", "/"), commands[1][3])
            self.assertIn(Step3Frame.R_WORKSPACE_FILES[1], commands[1][3])

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


if __name__ == "__main__":
    unittest.main()
