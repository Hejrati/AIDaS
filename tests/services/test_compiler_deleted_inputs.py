"""An active compiler must not publish measurements from a deleted run."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from openpyxl import Workbook, load_workbook

from aidas.services.step4_compiler import (
    compile_step4_results,
    create_cleaned_light_thickness_workbook,
    find_fovea_file,
)


class CompilerDeletedInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.subject = self.root / "Group" / "Subject"
        self.side = self.subject / "nasal"
        self.side.mkdir(parents=True)
        self.measurement = self.side / "rr_MCPAR.xlsx"
        self._write_measurement(12)
        self.output = self.root / "compiled.xlsx"
        self.output.write_bytes(b"previous final workbook")
        self.original = self.subject / "original.sdb"
        self.original.write_bytes(b"unchanged original image")

    def _write_measurement(self, value):
        workbook = Workbook()
        try:
            workbook.active.cell(1, 7, "Round")
            workbook.active.cell(2, 7, value)
            workbook.save(self.measurement)
        finally:
            workbook.close()

    def _before_save(self, action):
        def progress(_completed, _total, message):
            if message == "Saving workbook...":
                action()
        return progress

    def _assert_previous_preserved(self):
        self.assertEqual(self.output.read_bytes(), b"previous final workbook")
        self.assertEqual(self.original.read_bytes(), b"unchanged original image")
        self.assertEqual(list(self.root.glob(".*.tmp.xlsx")), [])

    def test_deleted_side_after_extraction_preserves_previous_final_workbook(self):
        with self.assertRaisesRegex(RuntimeError, "changed or was removed"):
            compile_step4_results(
                self.root, self.output,
                progress_callback=self._before_save(lambda: shutil.rmtree(self.side)),
            )
        self._assert_previous_preserved()
        self.assertFalse(self.side.exists())

    def test_deleted_side_during_serialization_never_publishes_stale_workbook(self):
        save = Workbook.save

        def save_then_delete(workbook, destination):
            save(workbook, destination)
            shutil.rmtree(self.side)

        with mock.patch.object(Workbook, "save", save_then_delete):
            with self.assertRaisesRegex(RuntimeError, "changed or was removed"):
                compile_step4_results(self.root, self.output)
        self._assert_previous_preserved()
        self.assertFalse(self.side.exists())

    def test_rebuilt_measurement_rejects_old_values_and_next_run_uses_new_values(self):
        with self.assertRaisesRegex(RuntimeError, "changed or was removed"):
            compile_step4_results(
                self.root, self.output,
                progress_callback=self._before_save(lambda: self._write_measurement(86)),
            )
        self._assert_previous_preserved()

        compile_step4_results(self.root, self.output)

        workbook = load_workbook(self.output, data_only=True)
        try:
            self.assertEqual(workbook["rrMCP-AR"]["C3"].value, 86)
        finally:
            workbook.close()

    def test_serialization_failure_preserves_previous_file_and_removes_partial_temp(self):
        def broken_save(_workbook, destination):
            destination.write(b"partial workbook")
            raise OSError("disk full")

        with mock.patch.object(Workbook, "save", broken_save):
            with self.assertRaisesRegex(OSError, "disk full"):
                compile_step4_results(self.root, self.output)
        self._assert_previous_preserved()

    def test_locked_output_preserves_previous_file_and_removes_temp(self):
        with mock.patch("aidas.services.step4_compiler.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaisesRegex(PermissionError, "locked"):
                compile_step4_results(self.root, self.output)
        self._assert_previous_preserved()

    def test_fovea_source_deleted_after_extraction_rejects_old_values(self):
        fovea = self.side / "_fovea_light_profiles_LIGHT.txt"
        fovea.write_text("RPEtoOLM 55\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed or was removed"):
            compile_step4_results(
                self.root, self.output,
                progress_callback=self._before_save(fovea.unlink),
            )
        self._assert_previous_preserved()

    def test_historical_subject_root_fovea_source_is_also_guarded(self):
        fovea = self.subject / "_fovea_light_profiles_LIGHT.txt"
        fovea.write_text("RPEtoOLM 55\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed or was removed"):
            compile_step4_results(
                self.root, self.output,
                progress_callback=self._before_save(fovea.unlink),
            )
        self._assert_previous_preserved()

    def test_unselected_fovea_source_does_not_invalidate_fovea_excluded_export(self):
        fovea = self.side / "_fovea_light_profiles_LIGHT.txt"
        fovea.write_text("RPEtoOLM 55\n", encoding="utf-8")

        compile_step4_results(
            self.root, self.output, include_fovea=False,
            progress_callback=self._before_save(fovea.unlink),
        )

        workbook = load_workbook(self.output, data_only=True)
        try:
            self.assertEqual(workbook["rrMCP-AR"]["C3"].value, 12)
        finally:
            workbook.close()

    def test_normal_cleaned_outputs_do_not_invalidate_original_input_snapshot(self):
        raw = self.side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
        original_text = "Distance\tWhole\tELM\tONL\n0\t0\t21\t34\n"
        raw.write_text(original_text, encoding="utf-8")
        obsolete = self.side / "_thickness_vs_distance_from_fovea_LIGHT2.txt"
        obsolete.write_text("old cleaned output", encoding="utf-8")
        cleaned = obsolete.with_suffix(".xlsx")
        self.assertIsNone(find_fovea_file(self.side))

        compile_step4_results(self.root, self.output)
        compile_step4_results(self.root, self.output)

        self.assertTrue(cleaned.is_file())
        self.assertFalse(obsolete.exists())
        self.assertEqual(raw.read_text(encoding="utf-8"), original_text)
        self.assertEqual(list(self.root.glob(".*.tmp.xlsx")), [])

    def test_fovea_discovery_ignores_thickness_tables_even_when_their_names_match(self):
        thickness = self.side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
        thickness.write_text("not a fovea profile", encoding="utf-8")
        profile = self.side / "nested" / "_fovea_light_profiles_LIGHT.txt"
        profile.parent.mkdir()
        profile.write_text("RPEtoOLM 45\n", encoding="utf-8")

        self.assertEqual(Path(find_fovea_file(self.side)), profile)

    def test_cleaned_thickness_is_not_recreated_after_original_text_deleted_mid_save(self):
        source = self.side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
        source.write_text("Distance\tWhole\tELM\tONL\n0\t0\t21\t34\n", encoding="utf-8")
        before = set(self.side.iterdir()) - {source}
        save = Workbook.save

        def save_then_delete(workbook, destination):
            save(workbook, destination)
            source.unlink()

        with mock.patch.object(Workbook, "save", save_then_delete):
            with self.assertRaisesRegex(RuntimeError, "changed or was removed"):
                create_cleaned_light_thickness_workbook(source)

        self.assertEqual(set(self.side.iterdir()), before)
        self._assert_previous_preserved()


if __name__ == "__main__":
    unittest.main()
