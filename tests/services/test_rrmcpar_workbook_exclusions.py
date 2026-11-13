"""Side workbooks retain excluded ROI positions without a companion TIFF."""

from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook, load_workbook

from aidas.services.step4_compiler import (
    compile_step4_results,
    get_first20_col_g_values,
    get_rrmcpar_values,
)
from aidas.services.step4_results import STEP4_RESULTS_HEADERS


def _write_side_workbook(side: Path, values: list[float | None]) -> Path:
    side.mkdir(parents=True, exist_ok=True)
    path = side / "rr_MCPAR.xlsx"
    workbook = Workbook()
    try:
        sheet = workbook.active
        sheet.append(STEP4_RESULTS_HEADERS)
        for roi_index, value in enumerate(values, 1):
            sheet.append(
                [roi_index] + [None] * 7 if value is None
                else [roi_index, 10, 5, 0, 0.4, 2, value, 0.9]
            )
        workbook.save(path)
    finally:
        workbook.close()
    return path


class RrMcpArWorkbookExclusionTests(unittest.TestCase):
    def test_indexed_workbook_keeps_first_middle_and_foveola_exclusions_and_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = [None, 0.0] + [index / 100 for index in range(3, 22)]
            values[10] = values[20] = None
            source = _write_side_workbook(Path(temporary), values)

            self.assertEqual(get_rrmcpar_values(source), values)
            self.assertEqual(get_first20_col_g_values(source), values[:20])
            self.assertFalse(list(Path(temporary).rglob("*.tif")))

    def test_indexed_all_excluded_workbook_retains_all_roi_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = _write_side_workbook(Path(temporary), [None] * 21)
            self.assertEqual(get_rrmcpar_values(source), [None] * 21)

    def test_numeric_text_roi_indices_and_trailing_formatting_preserve_blanks(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = _write_side_workbook(Path(temporary), [None, 0.0, None])
            workbook = load_workbook(source)
            try:
                sheet = workbook.active
                for roi_index in range(1, 4):
                    sheet.cell(roi_index + 1, 1, f"{roi_index:02d}")
                sheet.cell(3, 7, "   ")
                sheet.cell(30, 7).number_format = "0.000"
                workbook.save(source)
            finally:
                workbook.close()
            self.assertEqual(get_rrmcpar_values(source), [None, None, None])

    def test_old_unindexed_workbooks_keep_compact_column_g_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Results.xlsx"
            for full_headers in (False, True):
                with self.subTest(full_headers=full_headers):
                    workbook = Workbook()
                    try:
                        sheet = workbook.active
                        if full_headers:
                            sheet.append(STEP4_RESULTS_HEADERS)
                        else:
                            sheet.cell(1, 7, "Round")
                        for row, value in enumerate((None, 0.0, " ", 0.51, None, 0.93), 2):
                            sheet.cell(row, 7, value)
                        workbook.save(source)
                    finally:
                        workbook.close()
                    self.assertEqual(get_rrmcpar_values(source), [0.0, 0.51, 0.93])

    def test_other_numeric_first_column_layout_keeps_historical_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Results.xlsx"
            workbook = Workbook()
            try:
                sheet = workbook.active
                sheet.append(["Distance", None, None, None, None, None, "Round"])
                for row, value in enumerate((None, 0.0, None, 0.73), 1):
                    sheet.cell(row + 1, 1, row)
                    sheet.cell(row + 1, 7, value)
                workbook.save(source)
            finally:
                workbook.close()
            self.assertEqual(get_rrmcpar_values(source), [0.0, 0.73])

    def test_workbook_only_compilation_preserves_exclusions_for_either_fovea_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = root / "Control" / "Subject_1"
            nasal = [index / 100 for index in range(1, 22)]
            temporal = [index / 100 for index in range(31, 52)]
            for index in (0, 10, 20):
                nasal[index] = None
            nasal[1] = 0.0
            for side_name, values in (("nasal", nasal), ("temporal", temporal)):
                _write_side_workbook(subject / side_name, values)

            for include_fovea in (False, True):
                with self.subTest(include_fovea=include_fovea):
                    output = root / f"compiled_{include_fovea}.xlsx"
                    result = compile_step4_results(root, output, include_fovea=include_fovea)
                    self.assertEqual(result.measures[0].included, 1)
                    workbook = load_workbook(output, data_only=True)
                    try:
                        sheet = workbook["rrMCP-AR"]
                        self.assertEqual(sheet.max_row, 23)
                        self.assertEqual([sheet.cell(row, 2).value for row in range(3, 24)], list(range(1, 22)))
                        self.assertEqual([sheet.cell(row, 3).value for row in range(3, 24)], nasal)
                        self.assertEqual([sheet.cell(row, 4).value for row in range(3, 24)], temporal)
                        log_row = next(workbook["Extraction_log"].iter_rows(min_row=2, max_row=2, values_only=True))
                        self.assertTrue(log_row[4].endswith("rr_MCPAR.xlsx"))
                        self.assertTrue(log_row[5].endswith("rr_MCPAR.xlsx"))
                        self.assertEqual(log_row[6:9], (21, 21, "OK"))
                    finally:
                        workbook.close()
            self.assertFalse(list(root.rglob("*.tif")))

    def test_all_excluded_workbook_subject_does_not_shift_the_next_subject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            group = root / "Control"
            for side_name in ("nasal", "temporal"):
                _write_side_workbook(group / "Subject_1" / side_name, [None] * 21)
            _write_side_workbook(group / "Subject_2" / "nasal", [0.37] * 21)

            result = compile_step4_results(root, root / "compiled.xlsx")
            self.assertEqual((result.measures[0].included, result.measures[0].skipped), (2, 0))
            workbook = load_workbook(result.output_path, data_only=True)
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


if __name__ == "__main__":
    unittest.main()
