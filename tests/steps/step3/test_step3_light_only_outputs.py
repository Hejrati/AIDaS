"""App runs save LIGHT results and can still read historical paired results."""

from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from aidas.steps.step3_flatten import Step3Frame


class Step3LightOnlyOutputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.frame = Step3Frame.__new__(Step3Frame)
        self.frame.after = lambda _delay, callback: callback()
        self.frame._batch_panel_update = mock.Mock()
        self.frame._r_cancel_event = threading.Event()
        self.frame._write_r_run_log = lambda folder, *_args: Path(folder) / "run.log"
        self.config = {
            "input_dir": str(self.folder), "output_dir": str(self.folder),
            "reference_dark": "DARK_MARKED", "reference_light": "Light_MARKED",
            "to_process_dark": "DARK", "to_process_light": "LIGHT",
            "image_index_light": "1,2", "image_index_dark": "1,2",
            "pixel_width": "3.89",
        }
        self.light = np.arange(40.0).reshape((4, 5, 2))

    @staticmethod
    def _write_array(folder, name, data):
        data = np.asarray(data, dtype="<f8")
        (folder / (name + ".bin")).write_bytes(data.ravel(order="F").tobytes())
        (folder / (name + ".shape")).write_text(
            ",".join(map(str, data.shape)), encoding="utf-8",
        )

    def _assert_light_preview(self, results):
        np.testing.assert_array_equal(
            results["flattened_light"], np.transpose(self.light, (2, 0, 1)),
        )
        np.testing.assert_array_equal(
            results["final_light"], np.transpose(self.light, (2, 0, 1)),
        )

    def test_app_enables_light_only_even_when_the_external_environment_disables_it(self):
        self.frame._r_env = lambda thread_limit=None: {
            "AIDAS_LIGHT_ONLY": "0", "EXTERNAL_SETTING": "preserved",
        }
        environment = self.frame._r_run_env_for_config(self.config, 1)

        self.assertEqual(environment["AIDAS_LIGHT_ONLY"], "1")
        self.assertEqual(environment["EXTERNAL_SETTING"], "preserved")

    def test_success_requires_the_light_thickness_export_without_a_dark_export(self):
        (self.folder / "_thickness_vs_distance_from_fovea_LIGHT.txt").write_text(
            "retained LIGHT measurements", encoding="utf-8",
        )
        result = self.frame._finalize_r_script_result(
            self.config,
            {"folder": self.folder, "returncode": 0, "outcome": "completed"},
            validate_exports=True,
        )

        self.assertEqual(result["returncode"], 0, result.get("stdout"))
        self.assertEqual(result["outcome"], "completed")
        self.assertFalse(any("dark" in path.name.casefold() for path in self.folder.iterdir()))

    def test_missing_light_thickness_still_fails_when_a_dark_export_exists(self):
        (self.folder / "_thickness_vs_distance_from_fovea_DARK.txt").touch()
        result = self.frame._finalize_r_script_result(
            self.config,
            {"folder": self.folder, "returncode": 0, "outcome": "completed"},
            validate_exports=True,
        )

        self.assertNotEqual(result["returncode"], 0)
        self.assertIn("_thickness_vs_distance_from_fovea_LIGHT.txt", result["stdout"])

    def test_success_publishes_the_cleaned_table_under_the_original_light_filename(self):
        path = self.folder / "_thickness_vs_distance_from_fovea_LIGHT.txt"
        path.write_text(
            "Distance\tWhole\tELM\tONL\n"
            "-2\tNA\tNA\t99\n"
            "-1\tNA\t10.250\tNA\n"
            "0\tNA\tNA\tNA\n"
            "1\tNA\t20.500\t40.750\n",
            encoding="utf-8",
        )

        result = self.frame._finalize_r_script_result(
            self.config, {"returncode": 0, "outcome": "completed"},
            validate_exports=True,
        )

        self.assertEqual(result["returncode"], 0, result.get("stdout"))
        self.assertEqual(path.read_text(encoding="utf-8").splitlines(), [
            "Distance\tWhole\tELM\tONL",
            "-1\tNA\t10.250\tNA",
            "0\tNA\tNA\tNA",
            "1\tNA\t20.500\t40.750",
        ])
        self.assertEqual([saved.name for saved in self.folder.iterdir()], [path.name])

    def test_cleanup_failure_does_not_report_a_successful_r_run(self):
        path = self.folder / "_thickness_vs_distance_from_fovea_LIGHT.txt"
        original = b"Distance\tWhole\tELM\n1\tNA\tNA\n"
        path.write_bytes(original)
        with mock.patch(
            "aidas.services.step4_compiler.clean_light_thickness_file",
            side_effect=OSError("cannot replace thickness file"),
        ):
            result = self.frame._finalize_r_script_result(
                self.config, {"returncode": 0, "outcome": "completed"},
                validate_exports=True,
            )

        self.assertNotEqual(result["returncode"], 0)
        self.assertEqual(result["outcome"], "failed")
        self.assertIn("Could not save cleaned LIGHT thickness export", result["stdout"])
        self.assertEqual(path.read_bytes(), original)

    def test_current_result_previews_and_workspaces_use_light_names(self):
        names = [filename for _title, filename in Step3Frame.RESULT_IMAGE_ROWS]
        names.extend(Step3Frame.R_WORKSPACE_FILES)

        self.assertTrue(names)
        self.assertFalse(any("dark" in name.casefold() for name in names), names)

    def test_array_export_loads_and_is_discovered_without_dark_arrays(self):
        export = self.folder / Step3Frame.R_ARRAY_EXPORT_DIR
        export.mkdir()
        self._write_array(export, "FLATTENED_LIGHT_RETINA_RRC", self.light)
        self._write_array(export, "FLATTENED_MARKERS_RRC", np.ones((4, 5)))
        expected_files = set(export.iterdir())

        results = self.frame._load_r_array_export(self.folder)

        self._assert_light_preview(results)
        np.testing.assert_array_equal(results["final_grand_mean"], self.light.mean(axis=2))
        self.assertTrue(Step3Frame._folder_has_r_results(self.folder))
        self.assertEqual(set(export.iterdir()), expected_files)

    def test_analyze_preview_loads_without_dark_files_or_creating_them(self):
        (self.folder / "_flat_LIGHT.hdr").write_bytes(b"header")
        (self.folder / "_flat_LIGHT.img").write_bytes(b"image")
        expected_files = set(self.folder.iterdir())
        with mock.patch(
            "aidas.steps.step3_flatten._load_analyze_volume_r_layout",
            return_value=self.light,
        ) as load:
            results = self.frame._load_r_analyze_results(self.folder)

        self._assert_light_preview(results)
        np.testing.assert_array_equal(results["final_grand_mean"], self.light.mean(axis=2))
        self.assertTrue(all("dark" not in str(call.args[0]).casefold() for call in load.call_args_list))
        self.assertEqual(set(self.folder.iterdir()), expected_files)

    def test_new_and_legacy_workspaces_load_without_rewriting_the_saved_files(self):
        for names in (
            ("_flat_LIGHT.RData", "_done_LIGHT.RData"),
            ("DARK__and__LIGHT__flat.RData", "_done_DARK__and__LIGHT.RData"),
        ):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                for name in names:
                    (folder / name).write_bytes(b"saved R workspace")
                flat_data = {
                    "FLATTENED.LIGHT.RETINA.RRC": self.light,
                    "FLATTENED.MARKERS.RRC": np.ones((4, 5)),
                }
                if "DARK" in names[0]:
                    flat_data["FLATTENED.DARK.RETINA.RRC"] = self.light + 100
                pyreadr = SimpleNamespace(read_r=mock.Mock(side_effect=[flat_data, {}]))
                before = {path.name: path.read_bytes() for path in folder.iterdir()}

                with mock.patch.dict(sys.modules, {"pyreadr": pyreadr}):
                    results = self.frame._load_r_workspace_results(folder)

                self._assert_light_preview(results)
                self.assertTrue(Step3Frame._folder_has_r_results(folder))
                self.assertEqual(
                    {path.name: path.read_bytes() for path in folder.iterdir()}, before,
                )
                self.assertEqual(
                    [Path(call.args[0]).name for call in pyreadr.read_r.call_args_list],
                    list(names),
                )

    def test_output_script_uses_an_available_done_workspace_without_requiring_flat_data(self):
        cases = (
            ("current done only", ("_done_LIGHT.RData",), "_done_LIGHT.RData"),
            ("legacy done only", ("_done_DARK__and__LIGHT.RData",),
             "_done_DARK__and__LIGHT.RData"),
            ("current done and legacy pair",
             ("_done_LIGHT.RData", "DARK__and__LIGHT__flat.RData", "_done_DARK__and__LIGHT.RData"),
             "_done_LIGHT.RData"),
        )
        self.frame._r_env = lambda thread_limit=None: {}
        self.frame._run_supervised_r_command = mock.Mock(return_value=(0, "", "completed"))
        for label, filenames, expected in cases:
            with self.subTest(label=label):
                folder = self.folder / label
                folder.mkdir()
                for filename in filenames:
                    (folder / filename).write_bytes(b"existing R workspace")
                config = dict(self.config, input_dir=str(folder), output_dir=str(folder))

                result = self.frame._run_output_r_script_for_config(
                    Path("Rscript.exe"), folder / "custom-output.R", config,
                    {"returncode": 0, "cmd": [], "stdout": ""},
                )

                self.assertEqual(result["returncode"], 0)
                command = self.frame._run_supervised_r_command.call_args.args[0]
                expression = command[command.index("-e") + 1]
                self.assertIn(f"load({self.frame._r_string((folder / expected).resolve())})", expression)
                self.assertEqual({path.name for path in folder.iterdir()}, set(filenames))


if __name__ == "__main__":
    unittest.main()
