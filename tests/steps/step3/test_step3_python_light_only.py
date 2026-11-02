"""The standalone Python follow-on exporter saves only original LIGHT data."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main as translation


class PythonLightOnlyExportTests(unittest.TestCase):
    def test_input_loader_supports_light_only_data_and_preserves_legacy_distinct_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            light, marked = np.ones((2, 3, 1)), np.full((2, 3, 1), 2)
            dark, dark_marked = light + 3, marked + 4
            values = {"LIGHT.hdr": light, "LIGHT_MARKED.hdr": marked,
                      "DARK.hdr": dark, "DARK_MARKED.hdr": dark_marked}
            with patch.object(translation, "load_analyze_volume_r_layout",
                              side_effect=lambda path: values[path.name]):
                result = translation.load_input_volumes(root)
                self.assertIs(result["DARK"], light)
                self.assertIs(result["REF_DARK"], marked)
                for filename in ("DARK.hdr", "DARK_MARKED.hdr"):
                    (root / filename).touch()
                result = translation.load_input_volumes(root)
                self.assertIs(result["DARK"], dark)
                self.assertIs(result["REF_DARK"], dark_marked)

    def test_follow_on_exports_keep_light_values_with_and_without_legacy_dark_arrays(self):
        for with_dark in (False, True):
            with self.subTest(legacy_dark_arrays=with_dark), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                light = np.arange(24, dtype=np.float64).reshape((4, 3, 2))
                flat = {"flattened_light_retina_rrc": light}
                done = {
                    f"R_{border}_POSITION_LIGHT": np.full((2851, 1), value)
                    for border, value in (("RPE", 400.125), ("OLM", 350.25),
                                          ("ONL_OPL", 300.5), ("INL_IPL", 250.75),
                                          ("RNFL_GCL", 200.875), ("VITREOUS_RETINA", 100.0625))
                }
                if with_dark:
                    flat["flattened_dark_retina_rrc"] = light + 1000
                    done.update({name.replace("LIGHT", "DARK"): value + 500
                                 for name, value in tuple(done.items())})
                flat_path, done_path = root / "_flat_LIGHT.npz", root / "_done_LIGHT.npz"
                np.savez_compressed(flat_path, **flat)
                np.savez_compressed(done_path, **done)
                output = root / "output"
                output.mkdir()
                obsolete = output / "_flat_DARK.img"
                obsolete.write_bytes(b"obsolete generated volume")

                def save_preview(flattened, positions, path, title):
                    np.testing.assert_array_equal(flattened, light)
                    self.assertEqual(title, "Tissue Borders LIGHT")
                    Path(path).write_bytes(b"LIGHT preview")

                with patch.object(translation, "save_tissue_border_plot", side_effect=save_preview) as preview:
                    results = translation.run_more_outputs_from_step3_npz(flat_path, done_path, output)

                self.assertEqual(preview.call_count, 1)
                self.assertFalse(obsolete.exists())
                self.assertEqual(set(results), {"light_plot", "light_thickness"})
                self.assertEqual({path.name for path in output.iterdir()}, {
                    "_tissueBorders__LIGHT.png", "_thickness_vs_distance_from_fovea_LIGHT.txt"})
                expected = translation.build_thickness_export(
                    *(done[f"R_{border}_POSITION_LIGHT"] for border in
                      ("RPE", "OLM", "ONL_OPL", "INL_IPL", "RNFL_GCL", "VITREOUS_RETINA")))
                actual = np.loadtxt(results["light_thickness"], dtype=str, delimiter="\t")
                self.assertEqual(actual.shape, (2852, 8))
                np.testing.assert_array_equal(actual[0], expected[:, 0])
                np.testing.assert_allclose(actual[1:].astype(float), expected[:, 1:].T.astype(float),
                                           rtol=0, atol=0)

    def test_failed_follow_on_export_preserves_existing_dark_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flat_path, done_path = root / "_flat_LIGHT.npz", root / "_done_LIGHT.npz"
            np.savez_compressed(flat_path, flattened_light_retina_rrc=np.zeros((2, 2, 1)))
            np.savez_compressed(done_path, unrelated=np.zeros((2, 1)))
            output = root / "output"
            output.mkdir()
            old_output = output / "_flat_DARK.img"
            old_output.write_bytes(b"keep until a complete new export succeeds")
            with self.assertRaises(KeyError):
                translation.run_more_outputs_from_step3_npz(flat_path, done_path, output)
            self.assertTrue(old_output.is_file())


if __name__ == "__main__":
    unittest.main()
