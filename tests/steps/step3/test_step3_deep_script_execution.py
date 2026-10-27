from __future__ import annotations

import concurrent.futures
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from aidas.steps.step3_flatten import Step3Frame


class Step3DeepScriptExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame = Step3Frame.__new__(Step3Frame)
        self.frame.after = lambda _delay, callback: callback()
        self.frame._batch_panel_update = lambda *_args, **_kwargs: None
        self.frame._r_cancel_event = threading.Event()
        self.frame._r_env = lambda thread_limit=None: {"EXTERNAL_SETTING": "preserved"}
        self.frame._write_r_run_log = lambda output_dir, *_args: Path(output_dir) / "step3.log"
        self.frame._script_path = lambda role="main": self.root / (
            self.frame.R_OPTIMIZED_SCRIPT_NAME if role == "main"
            else self.frame.R_OPTIMIZED_OUTPUT_SCRIPT_NAME
        )

    @staticmethod
    def _config(folder):
        return {
            "input_dir": str(folder), "output_dir": str(folder),
            "reference_dark": "DARK_MARKED", "reference_light": "Light_MARKED",
            "to_process_dark": "DARK", "to_process_light": "Light",
            "image_index_light": "1:2", "image_index_dark": "1:2",
            "pixel_width": "3.89",
        }

    def _main(self):
        return self.root / self.frame.R_DEEP_LAYERS_SCRIPT_NAME

    def _output(self):
        return self.root / self.frame.R_DEEP_LAYERS_OUTPUT_SCRIPT_NAME

    def _run_main(self, selected):
        return self.frame._run_main_r_script_for_config(
            Path("Rscript.exe"), selected, self._config(self.root),
        )

    def _run_output(self, selected, main_result):
        return self.frame._run_output_r_script_for_config(
            Path("Rscript.exe"), selected, self._config(self.root), main_result,
        )

    def test_only_exact_bundled_deep_main_enables_reduced_processing(self):
        self.frame._run_supervised_r_command = mock.Mock(return_value=(0, "", "completed"))
        choices = (
            (self._main(), True),
            (self.root / self.frame.R_SCRIPT_NAME, False),
            (self.root / self.frame.R_OPTIMIZED_SCRIPT_NAME, False),
            (self.root / "custom" / self._main().name, False),
        )
        for selected, reduced in choices:
            with self.subTest(selected=selected):
                result = self._run_main(selected)
                env = self.frame._run_supervised_r_command.call_args.args[2]
                self.assertEqual(env.get("AIDAS_STEP3_DEEP_ONLY"), "1" if reduced else None)
                self.assertEqual(env["EXTERNAL_SETTING"], "preserved")
                self.assertEqual(result["_deep_layers_only"], reduced)

    def test_deep_setting_does_not_mutate_a_shared_environment(self):
        shared_env = {"AIDAS_STEP3_DEEP_ONLY": "0", "EXTERNAL_SETTING": "preserved"}
        self.frame._r_run_env_for_config = lambda *_args: shared_env
        barrier = threading.Barrier(2)
        environments = {}
        lock = threading.Lock()

        def run(command, _cwd, env, _on_line):
            barrier.wait(timeout=3)
            with lock:
                environments[Path(command[2]).name] = dict(env)
            return 0, "", "completed"

        self.frame._run_supervised_r_command = run
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self._run_main, selected) for selected in (
                self._main(), self.root / "custom.R",
            )]
            for future in futures:
                self.assertEqual(future.result(timeout=5)["returncode"], 0)

        self.assertEqual(environments[self._main().name]["AIDAS_STEP3_DEEP_ONLY"], "1")
        self.assertEqual(environments["custom.R"]["AIDAS_STEP3_DEEP_ONLY"], "0")
        self.assertEqual(shared_env["AIDAS_STEP3_DEEP_ONLY"], "0")

    def test_custom_scripts_keep_explicit_external_deep_setting(self):
        self.frame._r_env = lambda thread_limit=None: {"AIDAS_STEP3_DEEP_ONLY": "1"}
        self.frame._run_supervised_r_command = mock.Mock(return_value=(0, "", "completed"))
        self._run_main(self.root / "custom.R")
        env = self.frame._run_supervised_r_command.call_args.args[2]
        self.assertEqual(env["AIDAS_STEP3_DEEP_ONLY"], "1")

    def test_deep_main_requires_exact_matching_bundled_companion(self):
        self.frame._run_supervised_r_command = mock.Mock(return_value=(0, "", "completed"))
        main_result = self._run_main(self._main())
        for incompatible in (
            self.root / self.frame.R_OUTPUT_SCRIPT_NAME,
            self.root / self.frame.R_OPTIMIZED_OUTPUT_SCRIPT_NAME,
            self.root / "custom" / self._output().name,
        ):
            with self.subTest(incompatible=incompatible):
                result = self._run_output(incompatible, main_result)
                self.assertEqual(result["returncode"], 1)
                self.assertEqual(result["outcome"], "failed")
                self.assertIn("matching", result["stderr"])
                self.assertIn("Optimized deeper layers", result["stderr"])
                self.assertEqual(result["cmd"], main_result["cmd"])
                self.assertEqual(self.frame._run_supervised_r_command.call_count, 1)

        result = self._run_output(self._output(), main_result)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(self.frame._run_supervised_r_command.call_count, 2)
        output_cmd = self.frame._run_supervised_r_command.call_args.args[0]
        self.assertIn(self._output().as_posix(), output_cmd[3])

    def test_new_companion_remains_usable_after_old_main(self):
        self.frame._run_supervised_r_command = mock.Mock(return_value=(0, "", "completed"))
        main_result = self._run_main(self.root / self.frame.R_SCRIPT_NAME)
        result = self._run_output(self._output(), main_result)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(self.frame._run_supervised_r_command.call_count, 2)
        output_env = self.frame._run_supervised_r_command.call_args.args[2]
        self.assertNotIn("AIDAS_STEP3_DEEP_ONLY", output_env)

    def test_cancellation_takes_priority_over_pair_validation(self):
        self.frame._r_cancel_event.set()
        self.frame._run_supervised_r_command = mock.Mock()
        main = self._run_main(self._main())
        output = self._run_output(self.root / "incompatible.R", {"_deep_layers_only": True})
        self.assertEqual(main["outcome"], "cancelled")
        self.assertEqual(output["outcome"], "cancelled")
        self.assertEqual(output["returncode"], 130)
        self.frame._run_supervised_r_command.assert_not_called()

    def test_incompatible_pair_is_rejected_before_ui_setup_or_worker_launch(self):
        self.frame._busy = False
        self._main().write_text("# deep main", encoding="utf-8")
        incompatible = self.root / self.frame.R_OUTPUT_SCRIPT_NAME
        incompatible.write_text("# original output", encoding="utf-8")
        self.frame._ensure_r_ready_with_wizard = mock.Mock()
        with mock.patch("aidas.steps.step3_flatten.messagebox.showerror") as error:
            self.frame._start_batch_r_runs(
                [self.root], 1, main_script_path=self._main(), output_script_path=incompatible,
            )
        self.assertIn("matching", error.call_args.args[1])
        self.frame._ensure_r_ready_with_wizard.assert_not_called()
        self.assertFalse(self.frame._busy)

    def test_direct_pipeline_pair_error_starts_no_process(self):
        self.frame._run_supervised_r_command = mock.Mock()
        result = self.frame._run_r_script_for_config(
            Path("Rscript.exe"), self._main(), self.root / self.frame.R_OUTPUT_SCRIPT_NAME,
            self._config(self.root),
        )
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["cmd"], [])
        self.assertIn("matching", result["stderr"])
        self.frame._run_supervised_r_command.assert_not_called()

    def test_both_batch_schedules_preserve_deep_mode_and_validate_pair(self):
        for mode in (Step3Frame.R_OUTPUT_MODE_PARALLEL, Step3Frame.R_OUTPUT_MODE_SEQUENTIAL):
            for compatible in (False, True):
                with self.subTest(mode=mode, compatible=compatible):
                    folders = [self.root / f"{mode}-{compatible}-{index}" for index in range(2)]
                    for folder in folders:
                        folder.mkdir()
                    self.frame._r_worker_limit = lambda: 2
                    self.frame._r_threads_per_process = lambda _workers: 1
                    self.frame._folder_has_r_data = lambda _folder: False
                    self.frame._r_script_config_for_folder = self._config
                    finished = []
                    self.frame._on_batch_r_done = finished.extend

                    def run(command, cwd, env, _on_line):
                        if "-e" in command:
                            self.assertTrue(compatible)
                            self.assertIn(self._output().as_posix(), command[3])
                            for channel in ("DARK", "Light"):
                                (Path(cwd) / f"_thickness_vs_distance_from_fovea_{channel}.txt").write_text(
                                    "retained thickness", encoding="utf-8",
                                )
                        else:
                            self.assertEqual(env["AIDAS_STEP3_DEEP_ONLY"], "1")
                        return 0, "", "completed"

                    self.frame._run_supervised_r_command = mock.Mock(side_effect=run)
                    self.frame._batch_r_worker(
                        Path("Rscript.exe"), self._main(),
                        self._output() if compatible else self.root / self.frame.R_OUTPUT_SCRIPT_NAME,
                        folders, workers=2, output_mode=mode,
                    )
                    self.assertEqual(len(finished), 2)
                    self.assertEqual(self.frame._run_supervised_r_command.call_count, 4 if compatible else 0)
                    self.assertTrue(all(result["returncode"] == (0 if compatible else 1) for result in finished))
                    if not compatible:
                        self.assertTrue(all(result["cmd"] == [] for result in finished))
                        self.assertTrue(all("matching" in result["stderr"] for result in finished))


if __name__ == "__main__":
    unittest.main()
