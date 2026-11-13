"""The compiled rrMCP-AR sheet retains every saved ROI measurement."""

from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook, load_workbook
from PIL import Image

from aidas.services.step4_compiler import compile_step4_results, get_rrmcpar_values
from aidas.services.step4_results import encode_step4_measurements


def _write_stack(side: Path, values: list[float | None], *, excluded_rois=()) -> None:
    side.mkdir(parents=True, exist_ok=True)
    rows = [
        None if value is None else {
            "Major": 10.0,
            "Minor": 5.0,
            "Angle": 0.0,
            "Circ.": 0.4,
            "AR": 2.0,
            "Round": value,
            "Solidity": 0.9,
        }
        for value in values
    ]
    image = Image.new("RGB", (2, 2))
    try:
        image.save(
            side / "ROI_to_move_stck.tif",
            tiffinfo={270: encode_step4_measurements(rows, excluded_roi_indices=excluded_rois)},
        )
    finally:
        image.close()


class RrMcpArRoiExportTests(unittest.TestCase):
    def test_excluded_first_middle_and_foveola_rois_stay_blank_in_their_original_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = root / "Control" / "Subject_1"
            nasal = [index / 100 for index in range(1, 22)]
            temporal = [index / 100 for index in range(31, 52)]
            excluded = (1, 11, 21)
            expected_nasal = [None if index in excluded else value for index, value in enumerate(nasal, 1)]
            _write_stack(subject / "nasal", nasal, excluded_rois=excluded)
            _write_stack(subject / "temporal", temporal)
            self.assertEqual(get_rrmcpar_values(subject / "nasal" / "ROI_to_move_stck.tif"), expected_nasal)

            for include_fovea in (False, True):
                with self.subTest(include_fovea=include_fovea):
                    output = root / "compiled.xlsx"
                    compile_step4_results(root, output, include_fovea=include_fovea)
                    workbook = load_workbook(output, data_only=True)
                    try:
                        sheet = workbook["rrMCP-AR"]
                        self.assertEqual([sheet.cell(row, 2).value for row in range(3, 24)], list(range(1, 22)))
                        self.assertEqual([sheet.cell(row, 3).value for row in range(3, 24)], expected_nasal)
                        self.assertEqual([sheet.cell(row, 4).value for row in range(3, 24)], temporal)
                    finally:
                        workbook.close()

    def test_all_excluded_subject_keeps_numbered_blank_rows_before_the_next_subject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            group = root / "Control"
            for side in ("nasal", "temporal"):
                _write_stack(group / "Subject_1" / side, [None] * 21, excluded_rois=range(1, 22))
            _write_stack(group / "Subject_2" / "nasal", [0.37] * 21)

            output = root / "compiled.xlsx"
            result = compile_step4_results(root, output)
            self.assertEqual(result.measures[0].included, 2)
            self.assertEqual(result.measures[0].skipped, 0)
            workbook = load_workbook(output, data_only=True)
            try:
                sheet = workbook["rrMCP-AR"]
                self.assertEqual(sheet["A3"].value, "Subject_1")
                self.assertEqual([sheet.cell(row, 2).value for row in range(3, 24)], list(range(1, 22)))
                self.assertTrue(all(sheet.cell(row, column).value is None for row in range(3, 24) for column in (3, 4)))
                self.assertEqual(sheet["A24"].value, "Subject_2")
                self.assertEqual(sheet["B24"].value, 1)
                self.assertEqual(sheet["C44"].value, 0.37)
            finally:
                workbook.close()

    def test_rrmcpar_exclusions_leave_thickness_and_fovea_values_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = root / "Control" / "Subject_1"
            for side_name, thickness, fovea in (("nasal", 12.5, 31.5), ("temporal", 24.5, 42.5)):
                side = subject / side_name
                _write_stack(side, [0.5] * 21)
                (side / "_thickness_vs_distance_from_fovea_LIGHT.txt").write_text(
                    "Distance\tWhole\tELM\tONL\n" + "".join(
                        f"{index}\t0\t{thickness}\t8.5\n" for index in range(100)
                    ), encoding="utf-8",
                )
                (side / "_fovea_light_profiles_LIGHT.txt").write_text(f"RPEtoOLM {fovea}\n", encoding="utf-8")

            output = root / "compiled.xlsx"
            compile_step4_results(root, output)
            workbook = load_workbook(output, data_only=True)
            try:
                before = list(workbook["ELM-RPE"].values)
            finally:
                workbook.close()
            for side_name in ("nasal", "temporal"):
                _write_stack(subject / side_name, [None] * 21, excluded_rois=range(1, 22))
            compile_step4_results(root, output)
            workbook = load_workbook(output, data_only=True)
            try:
                self.assertEqual(list(workbook["ELM-RPE"].values), before)
                self.assertEqual(workbook["ELM-RPE"]["C3"].value, 31.5)
                self.assertEqual(workbook["ELM-RPE"]["D3"].value, 42.5)
                self.assertEqual(workbook["ELM-RPE"]["C4"].value, 12.5)
                self.assertEqual(workbook["ELM-RPE"]["D4"].value, 24.5)
            finally:
                workbook.close()

    def test_all_21_tiff_rois_are_exported_for_either_fovea_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nasal = [index / 100 for index in range(1, 21)] + [0.519]
            temporal = [index / 100 for index in range(31, 51)] + [0.937]
            subject = root / "Control" / "Subject_1"
            _write_stack(subject / "nasal", nasal)
            _write_stack(subject / "temporal", temporal)

            for include_fovea in (False, True):
                with self.subTest(include_fovea=include_fovea):
                    output = root / f"compiled_{include_fovea}.xlsx"
                    compile_step4_results(root, output, include_fovea=include_fovea)
                    workbook = load_workbook(output, data_only=True)
                    try:
                        sheet = workbook["rrMCP-AR"]
                        self.assertEqual(sheet.max_row, 23)
                        self.assertEqual(
                            [sheet.cell(row, 2).value for row in range(3, 24)],
                            list(range(1, 22)),
                        )
                        self.assertEqual(
                            [sheet.cell(row, 3).value for row in range(3, 24)], nasal,
                        )
                        self.assertEqual(
                            [sheet.cell(row, 4).value for row in range(3, 24)], temporal,
                        )
                        log_rows = [
                            row for row in workbook["Extraction_log"].iter_rows(
                                min_row=2, values_only=True,
                            ) if row[0] == "rrMCP-AR"
                        ]
                        self.assertEqual(len(log_rows), 1)
                        self.assertEqual(log_rows[0][6:9], (21, 21, "OK"))
                    finally:
                        workbook.close()

    def test_legacy_workbook_exports_values_after_roi_20_and_preserves_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            side = root / "Control" / "Subject_1" / "nasal"
            side.mkdir(parents=True)
            values = [0.0] + [index / 100 for index in range(1, 21)] + [0.987654321]
            source = Workbook()
            try:
                source.active.cell(1, 7, "Round")
                for row, value in enumerate(values, start=2):
                    source.active.cell(row, 7, value)
                source.save(side / "rr_MCPAR.xlsx")
            finally:
                source.close()

            output = root / "compiled.xlsx"
            compile_step4_results(root, output)
            workbook = load_workbook(output, data_only=True)
            try:
                sheet = workbook["rrMCP-AR"]
                self.assertEqual(sheet.max_row, 24)
                self.assertEqual(
                    [sheet.cell(row, 3).value for row in range(3, 25)], values,
                )
                self.assertEqual(sheet["B24"].value, 22)
                self.assertTrue(all(sheet.cell(row, 4).value is None for row in range(3, 25)))
            finally:
                workbook.close()

    def test_mixed_subject_counts_keep_ids_numbering_and_missing_sides_aligned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            group = root / "Control"
            _write_stack(group / "Subject_1" / "nasal", [0.1] * 20)
            _write_stack(group / "Subject_1" / "temporal", [0.2] * 20)
            _write_stack(group / "Subject_2" / "nasal", [0.3] * 21)
            _write_stack(group / "Subject_2" / "temporal", [0.4] * 20)
            _write_stack(group / "Subject_3" / "temporal", [0.5] * 21)

            output = root / "compiled.xlsx"
            compile_step4_results(root, output, include_fovea=False)
            workbook = load_workbook(output, data_only=True)
            try:
                sheet = workbook["rrMCP-AR"]
                self.assertEqual(sheet.max_row, 64)
                for first_row, subject_id, count in (
                    (3, "Subject_1", 20),
                    (23, "Subject_2", 21),
                    (44, "Subject_3", 21),
                ):
                    self.assertEqual(sheet.cell(first_row, 1).value, subject_id)
                    self.assertEqual(
                        [sheet.cell(first_row + offset, 2).value for offset in range(count)],
                        list(range(1, count + 1)),
                    )
                    self.assertTrue(all(
                        sheet.cell(first_row + offset, 1).value is None
                        for offset in range(1, count)
                    ))
                self.assertEqual(sheet["C43"].value, 0.3)
                self.assertIsNone(sheet["D43"].value)
                self.assertTrue(all(sheet.cell(row, 3).value is None for row in range(44, 65)))
                self.assertEqual(sheet["D64"].value, 0.5)
            finally:
                workbook.close()


if __name__ == "__main__":
    unittest.main()
