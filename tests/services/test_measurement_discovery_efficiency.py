"""Measurement ranking/order is retained while irrelevant paths are skipped."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from aidas.services import step4_compiler as compiler
from aidas.services.step4_results import encode_step4_measurements


class MeasurementDiscoveryEfficiencyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_ignored_plot_files_do_not_require_path_resolution(self):
        for index in range(100):
            (self.root / f"unrelated_{index:03}.png").touch()
        selected = self.root / "Results.xlsx"
        selected.touch()
        with mock.patch.object(compiler, "Path", wraps=Path) as constructor:
            actual = compiler.find_results_file(self.root)
        self.assertEqual(actual, str(selected))
        self.assertLessEqual(constructor.call_count, 2)

    def test_natural_depth_and_legacy_priorities_are_unchanged(self):
        for name in ("Results10.xlsx", "Results2.xlsx", "other.xlsx", "~$Results1.xlsx",
                     "_thickness_vs_distance_from_fovea_LIGHT2.xlsx", "_fovea_LIGHT.xlsx"):
            (self.root / name).touch()
        self.assertEqual(compiler.find_results_file(self.root), str(self.root / "Results2.xlsx"))
        nested = self.root / "nested"
        nested.mkdir()
        legacy = nested / "rr_MCPAR.xlsx"
        legacy.touch()
        self.assertEqual(compiler.find_results_file(self.root), str(legacy))
        top = self.root / "rr_mcpar.xlsx"
        top.touch()
        self.assertEqual(compiler.find_results_file(self.root), str(top))

    def test_valid_tiff_metadata_precedes_workbooks_and_plain_tiff_does_not(self):
        workbook = self.root / "rr_MCPAR.xlsx"
        workbook.touch()
        stack = self.root / "ROI_TO_MOVE_STCK.TIFF"
        with Image.new("L", (2, 2)) as image:
            image.save(stack)
            self.assertEqual(compiler.find_results_file(self.root), str(workbook))
            values = [{name: .12345678901234567 for name in
                       ("Major", "Minor", "Angle", "Circ.", "AR", "Round", "Solidity")}]
            image.save(stack, description=encode_step4_measurements(values))
        self.assertEqual(compiler.find_results_file(self.root), str(stack))

    def test_missing_paths_and_walk_errors_keep_existing_behavior(self):
        self.assertIsNone(compiler.find_results_file(None))
        self.assertIsNone(compiler.find_results_file(self.root / "missing"))
        with mock.patch.object(compiler, "walk_unique_directories", side_effect=PermissionError("unreadable")):
            with self.assertRaisesRegex(PermissionError, "unreadable"):
                compiler.find_results_file(self.root)


if __name__ == "__main__":
    unittest.main()
