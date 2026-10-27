"""Known companions provide previews once; custom scripts keep their behavior."""

from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from aidas.steps.step3_flatten import Step3Frame


class Step3PreviewPipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = Step3Frame.__new__(Step3Frame)
        self.frame.after = lambda _delay, callback: callback()
        self.frame._batch_panel_update = mock.Mock()
        self.frame._r_cancel_event = threading.Event()
        self.frame._r_env = lambda thread_limit=None: {}
        self.frame._write_r_run_log = lambda directory, *_args: Path(directory) / "run.log"
        self.frame._folder_has_r_data = lambda _folder: False
        self.frame._r_worker_limit = lambda: 1
        self.frame._r_threads_per_process = lambda _workers: 1

    def _pair(self, deep):
        main = self.frame.R_DEEP_LAYERS_SCRIPT_NAME if deep else self.frame.R_OPTIMIZED_SCRIPT_NAME
        output = self.frame.R_DEEP_LAYERS_OUTPUT_SCRIPT_NAME if deep else self.frame.R_OPTIMIZED_OUTPUT_SCRIPT_NAME
        return (self.frame._script_path("main").with_name(main),
                self.frame._script_path("output").with_name(output))

    @staticmethod
    def _config(directory):
        return {
            "input_dir": str(directory), "output_dir": str(directory),
            "reference_dark": "DARK_MARKED", "reference_light": "LIGHT_MARKED",
            "to_process_dark": "DARK", "to_process_light": "LIGHT",
            "image_index_light": "1,2", "image_index_dark": "1,2", "pixel_width": "3.89",
        }

    def _run(self, directory, main, output, output_mode=None):
        commands = []

        def execute(command, cwd, environment, on_line):
            commands.append((command, environment))
            if "-e" in command:
                for channel in ("DARK", "LIGHT"):
                    (Path(cwd) / f"_thickness_vs_distance_from_fovea_{channel}.txt").touch()
            return 0, "", "completed"

        self.frame._run_supervised_r_command = execute
        config = self._config(directory)
        if output_mode is None:
            result = self.frame._run_r_script_for_config(Path("Rscript.exe"), main, output, config)
            self.assertEqual(result["returncode"], 0)
        else:
            results = []
            self.frame._r_script_config_for_folder = lambda _folder: config
            self.frame._on_batch_r_done = results.extend
            self.frame._batch_r_worker(
                Path("Rscript.exe"), main, output, [directory], 1, output_mode=output_mode,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["returncode"], 0)
        self.assertEqual(len(commands), 2)
        self.assertNotIn("_defer_border_plots", config)
        return commands

    def test_both_bundled_pairs_render_once_in_direct_and_both_batch_schedules(self):
        for deep in (False, True):
            for schedule in (None, "parallel", "sequential"):
                with self.subTest(deep=deep, schedule=schedule), tempfile.TemporaryDirectory() as temporary:
                    commands = self._run(Path(temporary), *self._pair(deep), output_mode=schedule)
                    for _command, environment in commands:
                        self.assertEqual(environment["AIDAS_STEP3_DEFER_BORDER_PLOTS"], "1")
                        self.assertEqual(environment["AIDAS_STEP3_RASTER"], "1")
                    self.assertEqual(commands[0][1].get("AIDAS_STEP3_DEEP_ONLY"), "1" if deep else None)

    def test_custom_companion_does_not_suppress_main_previews(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            main, _output = self._pair(False)
            commands = self._run(directory, main, directory / "custom-output.R")
            for _command, environment in commands:
                self.assertNotIn("AIDAS_STEP3_DEFER_BORDER_PLOTS", environment)
                self.assertNotIn("AIDAS_STEP3_RASTER", environment)

    def test_custom_copies_with_bundled_filenames_keep_standalone_behavior(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            main, output = self._pair(False)
            commands = self._run(directory, directory / main.name, directory / output.name)
            for _command, environment in commands:
                self.assertNotIn("AIDAS_STEP3_DEFER_BORDER_PLOTS", environment)
                self.assertNotIn("AIDAS_STEP3_RASTER", environment)

    def test_original_scripts_and_mixed_pairs_keep_their_existing_rendering(self):
        for main, output in (
            (self.frame._script_path("main").with_name(self.frame.R_SCRIPT_NAME),
             self.frame._script_path("output").with_name(self.frame.R_OUTPUT_SCRIPT_NAME)),
            (self._pair(False)[0], self._pair(True)[1]),
        ):
            with self.subTest(main=main, output=output), tempfile.TemporaryDirectory() as temporary:
                commands = self._run(Path(temporary), main, output)
                for _command, environment in commands:
                    self.assertNotIn("AIDAS_STEP3_DEFER_BORDER_PLOTS", environment)
                    self.assertNotIn("AIDAS_STEP3_RASTER", environment)


if __name__ == "__main__":
    unittest.main()
