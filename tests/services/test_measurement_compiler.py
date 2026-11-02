from __future__ import annotations

import inspect
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from openpyxl import Workbook, load_workbook
from PIL import Image

from aidas.services.step4_results import encode_step4_measurements

from aidas.services.step4_compiler import (
    DEFAULT_OUTPUT_FILENAME,
    MeasureCompilation,
    clean_light_thickness_file,
    create_cleaned_light_thickness_workbook,
    compile_step4_results,
    extract_elmrpe_onl_from_subject,
    find_light_files,
    find_subject_folders,
    find_results_file,
    find_thickness_file,
    make_light2_from_light,
    get_first20_col_g_values,
    get_column_values,
    read_tsv_rows,
    read_thickness_rows,
)


def _write_results(path: Path, first_value: int) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "in"
    worksheet.cell(1, 7).value = "Round"
    for offset in range(20):
        worksheet.cell(offset + 2, 7).value = first_value + offset
    workbook.save(path)
    workbook.close()


def _write_thickness(path: Path, elm_value: float, onl_value: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("Distance\tWhole\tELM\tONL\n")
        for distance in range(-1150, 1150):
            handle.write(f"{distance}\t0\t{elm_value}\t{onl_value}\n")


def _write_fovea(path: Path, value: float) -> None:
    path.write_text(f"foveaLIGHT 1 2\nRPEtoOLM {value} {value}\n", encoding="utf-8")


def _build_subject(root: Path) -> Path:
    subject = root / "LE" / "7_0_0"
    nasal = subject / "nasal"
    temporal = subject / "temporal"
    nasal.mkdir(parents=True)
    temporal.mkdir(parents=True)

    _write_results(nasal / "rr_MCPAR.xlsx", 1)
    _write_results(temporal / "Results.xlsx", 101)
    _write_thickness(nasal / "_thickness_vs_distance_from_fovea_LIGHT.txt", 10.0, 20.0)
    _write_thickness(temporal / "_thickness_vs_distance_from_fovea_LIGHT.txt", 30.0, 40.0)
    _write_fovea(nasal / "_fovea_light_profiles_LIGHT.txt", 55.77)
    _write_fovea(temporal / "_fovea_light_profiles_LIGHT.txt", 66.88)
    return subject


class Step4CompilerTests(unittest.TestCase):
    def test_embedded_tiff_measurements_replace_side_workbooks_without_changing_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subject = root / "Control" / "Subject_1"
            expected = {}
            for side_name, first in (("nasal", 1), ("temporal", 101)):
                side = subject / side_name
                side.mkdir(parents=True)
                rows = [dict(zip(
                    ("Major", "Minor", "Angle", "Circ.", "AR", "Round", "Solidity"),
                    (10.1, 5.2, -4.3, 0.4, 2.5, first + index + 0.1234567890123, 0.7),
                )) for index in range(25)]
                source = side / "ROI_to_move_stck.tif"
                image = Image.new("RGB", (2, 2))
                image.save(source, tiffinfo={270: encode_step4_measurements(rows)})
                image.close()
                expected[side_name] = [row["Round"] for row in rows[:20]]
                _write_thickness(side / "_thickness_vs_distance_from_fovea_LIGHT.txt", 30, 40)
                self.assertEqual(Path(find_results_file(side)), source)
                self.assertEqual(get_first20_col_g_values(source), expected[side_name])

            output = root / "compiled.xlsx"
            result = compile_step4_results(root, output, include_fovea=False)
            self.assertEqual(result.group_ids, {"Control": ("Subject_1",)})
            workbook = load_workbook(output, data_only=True)
            try:
                for column, side_name in ((2, "nasal"), (3, "temporal")):
                    values = [workbook["rrMCP-AR"].cell(row, column).value for row in range(3, 23)]
                    for actual, original in zip(values, expected[side_name]):
                        self.assertAlmostEqual(actual, original, places=12)
                self.assertTrue(workbook["Extraction_log"]["E2"].value.endswith("ROI_to_move_stck.tif"))
            finally:
                workbook.close()
            self.assertEqual(list(root.rglob("rr_MCPAR.xlsx")), [])

    def test_valid_embedded_tiff_precedes_legacy_workbook_but_ordinary_tiff_does_not(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "rr_MCPAR.xlsx"
            _write_results(legacy, 999)
            image = Image.new("L", (2, 2))
            source = root / "ROI_to_move_stck.tif"
            image.save(source)
            self.assertEqual(Path(find_results_file(root)), legacy)
            rows = [{"Major": 10, "Minor": 5, "Angle": 0, "Circ.": 0.3, "AR": 2, "Round": 0.8, "Solidity": 0.9}]
            image.save(source, tiffinfo={270: encode_step4_measurements(rows)})
            image.close()
            self.assertEqual(Path(find_results_file(root)), source)
            self.assertEqual(get_first20_col_g_values(source), [0.8])

    def test_compiler_uses_public_sheet_reordering_api(self):
        source = inspect.getsource(compile_step4_results)

        self.assertIn("workbook.move_sheet", source)
        self.assertNotIn("workbook._sheets", source)

    def test_full_numeric_subject_names_are_preserved_and_sorted_naturally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for subject_id in ("1000", "999"):
                nasal = root / "LE" / f"{subject_id}_0_0" / "nasal"
                nasal.mkdir(parents=True)
                _write_results(nasal / "Results.xlsx", 1)

            folders = find_subject_folders(root)
            result = compile_step4_results(root, root / "compiled.xlsx")

            self.assertEqual(set(folders["LE"]), {"999_0_0", "1000_0_0"})
            self.assertEqual(result.group_ids, {"LE": ("999_0_0", "1000_0_0")})

    def test_light_cleanup_preserves_original_and_removes_temporary_file_on_os_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "LIGHT.txt"
            source.write_bytes(b"Distance\tWhole\tELM\tONL\r\n-1\t0\tNA\t9\r\n0\t0\t1\t2\r\n")
            original = source.read_bytes()
            legacy = source.with_name("LIGHT2.txt")
            legacy.write_bytes(b"keep until Excel is saved")

            for operation in ("Workbook.save", "os.replace"):
                with self.subTest(operation=operation), mock.patch(
                    "aidas.services.step4_compiler." + operation,
                    side_effect=OSError("disk full"),
                ):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        clean_light_thickness_file(source)

                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(legacy.read_bytes(), b"keep until Excel is saved")
                self.assertEqual(set(Path(temp_dir).iterdir()), {source, legacy})

    def test_failed_cleaned_excel_replacement_preserves_raw_and_previous_workbook(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "_thickness_vs_distance_from_fovea_LIGHT.txt"
            source.write_bytes(b"Distance\tWhole\tELM\tONL\r\n0\t0\t1\t2\r\n")
            destination = Path(create_cleaned_light_thickness_workbook(source))
            previous = destination.read_bytes()
            source.write_bytes(b"Distance\tWhole\tELM\tONL\r\n0\t0\t3\t4\r\n")
            original = source.read_bytes()
            legacy = source.with_name("_thickness_vs_distance_from_fovea_LIGHT2.txt")
            legacy.write_bytes(b"obsolete cleaned table")
            legacy_original = legacy.read_bytes()

            with mock.patch("aidas.services.step4_compiler.os.replace", side_effect=PermissionError("locked")):
                with self.assertRaisesRegex(PermissionError, "locked"):
                    create_cleaned_light_thickness_workbook(source)

            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(destination.read_bytes(), previous)
            self.assertEqual(legacy.read_bytes(), legacy_original)
            self.assertEqual(set(Path(temp_dir).iterdir()), {source, destination, legacy})

    def test_successful_raw_to_excel_conversion_removes_only_obsolete_sibling_light2_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "_thickness_vs_distance_from_fovea_LIGHT.txt"
            original = b"Distance\tWhole\tELM\tONL\r\n0\t0\t1\t2\r\n"
            source.write_bytes(original)
            legacy = source.with_name("_thickness_vs_distance_from_fovea_LIGHT2.txt")
            legacy.write_bytes(b"obsolete cleaned table")
            unrelated = source.with_name("another_LIGHT2.txt")
            unrelated.write_bytes(b"keep unrelated input")

            destination = Path(create_cleaned_light_thickness_workbook(source))

            self.assertTrue(destination.is_file())
            self.assertFalse(legacy.exists())
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(unrelated.read_bytes(), b"keep unrelated input")
            # The idempotent existing-workbook path also retires an old copy.
            legacy.write_bytes(b"another obsolete copy")
            workbook_bytes = destination.read_bytes()
            self.assertEqual(create_cleaned_light_thickness_workbook(source), str(destination))
            self.assertFalse(legacy.exists())
            self.assertEqual(destination.read_bytes(), workbook_bytes)
            self.assertEqual(source.read_bytes(), original)

    def test_locked_obsolete_light2_text_does_not_drop_valid_compiled_measurements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            side = root / "Control" / "Subject_1" / "nasal"
            side.mkdir(parents=True)
            _write_results(side / "rr_MCPAR.xlsx", 1)
            source = side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
            original = b"Distance\tWhole\tELM\tONL\r\n0\t0\t12.5\t25.5\r\n"
            source.write_bytes(original)
            legacy = source.with_name("_thickness_vs_distance_from_fovea_LIGHT2.txt")
            legacy.write_bytes(b"locked obsolete table")
            real_unlink = Path.unlink

            def unlink(path, *args, **kwargs):
                if path == legacy:
                    raise PermissionError("file is open")
                return real_unlink(path, *args, **kwargs)

            output = root / "compiled.xlsx"
            with mock.patch.object(Path, "unlink", unlink), self.assertLogs(
                "aidas.services.step4_compiler", level="WARNING",
            ):
                compile_step4_results(root, output, include_fovea=False)

            workbook = load_workbook(output, data_only=True)
            try:
                self.assertEqual(workbook["ELM-RPE"]["B3"].value, 12.5)
                self.assertEqual(workbook["ONL"]["B3"].value, 25.5)
                self.assertEqual(workbook["Extraction_log"]["I3"].value, "OK")
                self.assertEqual(workbook["Extraction_log"]["I4"].value, "OK")
            finally:
                workbook.close()
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(legacy.read_bytes(), b"locked obsolete table")
            self.assertEqual(read_thickness_rows(legacy.with_suffix(".xlsx"))[1], [0, 0, 12.5, 25.5])

    def test_light_cleanup_saves_numeric_excel_with_fixed_elm_column_and_keeps_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "_thickness_vs_distance_from_fovea_LIGHT.txt"
            source.write_text(
                "Distance\tWhole\tELM\tONL\n"
                "-3\t0\tNA\t999\n"
                "-2\t0\t20\tNA\n"
                "-1\t0\t30\t300\n"
                "0\t0\tNA\t400\n"
                "1\t0\tNA\t500\n"
                "0\t0\tNA\t600\n"
                "2\t0\t40\t700\n"
                "3\tshort\n",
                encoding="utf-8",
            )

            original = source.read_bytes()
            modified = source.stat().st_mtime_ns
            destination = source.with_name("_thickness_vs_distance_from_fovea_LIGHT2.xlsx")
            self.assertEqual(make_light2_from_light(source, 4), str(destination))
            self.assertEqual(read_thickness_rows(destination), [
                ["Distance", "Whole", "ELM", "ONL"],
                [-2, 0, 20, None],
                [-1, 0, 30, 300],
                [0, 0, None, 400],
                [2, 0, 40, 700],
            ])
            cleaned = destination.read_bytes()
            with mock.patch("aidas.services.step4_compiler.Workbook.save") as write:
                self.assertEqual(clean_light_thickness_file(source), str(destination))
            write.assert_not_called()
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(destination.read_bytes(), cleaned)
            self.assertEqual(source.stat().st_mtime_ns, modified)
            self.assertEqual(set(Path(temp_dir).iterdir()), {source, destination})

    def test_already_clean_light_retains_exact_bytes_without_a_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "LIGHT.txt"
            original = b"Distance\tWhole\tELM\tONL\r\n-1\t0\t1\t2\r\n0\t0\tNA\t3\r\n"
            source.write_bytes(original)

            destination = Path(clean_light_thickness_file(source))
            self.assertEqual(destination.name, "LIGHT2.xlsx")
            self.assertEqual(read_thickness_rows(destination)[2], [0, 0, None, 3])
            self.assertEqual(source.read_bytes(), original)

    def test_nonfinite_tokens_round_trip_as_text_without_changing_legacy_parser_meaning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "_thickness_vs_distance_from_fovea_LIGHT.txt"
            original = b"Distance\tWhole\tELM\tONL\r\n-1\t0\tNaN\tInf\r\n0\t0\tNA\t-Inf\r\n1\t0\t10\tNA\r\n"
            source.write_bytes(original)

            destination = Path(create_cleaned_light_thickness_workbook(source))

            self.assertEqual(read_thickness_rows(destination)[1:], [
                [-1, 0, "NaN", "Inf"], [0, 0, None, "-Inf"], [1, 0, 10, None],
            ])
            elm_values = get_column_values(destination, 3)
            onl_values = get_column_values(destination, 4)
            self.assertTrue(math.isnan(elm_values[0]))
            self.assertEqual(elm_values[1], 10)
            self.assertEqual(onl_values, [math.inf, -math.inf])
            self.assertEqual(source.read_bytes(), original)

    def test_current_light_is_preferred_over_stale_light2_in_direct_and_recursive_paths(self):
        cases = (
            ("nasal", "nasal"),
            ("nasal", "nasal/exports/final"),
            ("rrMCPAR", "nasal/exports/final"),
            ("nasal", "other exports"),
        )
        for legacy_directory, current_directory in cases:
            with self.subTest(legacy=legacy_directory, current=current_directory), \
                    tempfile.TemporaryDirectory() as temp_dir:
                subject = Path(temp_dir)
                legacy = subject / legacy_directory / "_thickness_vs_distance_from_fovea_LIGHT2.txt"
                current = subject / current_directory / "_thickness_vs_distance_from_fovea_LIGHT.txt"
                legacy.parent.mkdir(parents=True, exist_ok=True)
                current.parent.mkdir(parents=True, exist_ok=True)
                legacy.write_text("stale", encoding="utf-8")
                current.write_text("current", encoding="utf-8")

                self.assertEqual(Path(find_thickness_file(subject)), current)
                nasal, _temporal = find_light_files(subject)
                self.assertEqual(Path(nasal), current)

    def test_cleaned_excel_is_preferred_to_legacy_text_without_modifying_either(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subject = Path(temp_dir)
            side = subject / "nasal"
            side.mkdir()
            legacy = side / "_thickness_vs_distance_from_fovea_LIGHT2.txt"
            legacy.write_text("Distance\tWhole\tELM\tONL\n0\t0\t999\t999\n", encoding="utf-8")
            excel = side / "_thickness_vs_distance_from_fovea_LIGHT2.xlsx"
            workbook = Workbook()
            workbook.active.append(["Distance", "Whole", "ELM", "ONL"])
            workbook.active.append([0, 0, 12.1234567890123, 34.1234567890123])
            workbook.save(excel)
            workbook.close()
            original = {path: path.read_bytes() for path in (excel, legacy)}

            self.assertEqual(Path(find_thickness_file(subject)), excel)
            self.assertEqual(find_light_files(subject), (str(excel), None))
            for measure, expected in (("ELM-RPE", 12.1234567890123), ("ONL", 34.1234567890123)):
                values, _temporal, path, _other, count, _other_count = (
                    extract_elmrpe_onl_from_subject(subject, measure, False)
                )
                self.assertAlmostEqual(values[0], expected, places=12)
                self.assertEqual(path, str(excel))
                self.assertEqual(count, 1)
            self.assertEqual({path: path.read_bytes() for path in original}, original)

    def test_new_raw_export_refreshes_cleaned_excel_and_ignores_stale_legacy_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subject = Path(temp_dir)
            side = subject / "nasal"
            side.mkdir()
            source = side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
            source.write_text("Distance\tWhole\tELM\tONL\n0\t0\t1\t2\n", encoding="utf-8")
            destination = Path(create_cleaned_light_thickness_workbook(source))
            (side / "_thickness_vs_distance_from_fovea_LIGHT2.txt").write_text("stale", encoding="utf-8")
            source.write_bytes(b"Distance\tWhole\tELM\tONL\r\n0\t0\t9.1234567890123\t10.1234567890123\r\n")
            original = source.read_bytes()

            values, _temporal, filename, _other, count, _other_count = (
                extract_elmrpe_onl_from_subject(subject, "ONL", False)
            )

            self.assertAlmostEqual(values[0], 10.1234567890123, places=12)
            self.assertEqual(Path(filename), destination)
            self.assertEqual(count, 1)
            self.assertEqual(source.read_bytes(), original)
            self.assertAlmostEqual(read_thickness_rows(destination)[1][2], 9.1234567890123, places=12)
            self.assertFalse((side / "_thickness_vs_distance_from_fovea_LIGHT2.txt").exists())

    def test_cleaned_thickness_excel_never_creates_a_step4_subject(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            side = root / "Control" / "Subject" / "nasal"
            side.mkdir(parents=True)
            source = side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
            _write_thickness(source, 10, 20)
            create_cleaned_light_thickness_workbook(source)

            self.assertIsNone(find_results_file(side))
            self.assertEqual(find_subject_folders(root), {})

    def test_explicit_conversion_of_legacy_light2_retains_all_already_cleaned_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "_thickness_vs_distance_from_fovea_LIGHT2.txt"
            original = b"Distance\tWhole\tELM\tONL\r\n-1\t0\tNA\t100\r\n0\t0\t20\t200\r\n"
            source.write_bytes(original)

            destination = Path(create_cleaned_light_thickness_workbook(source))

            self.assertEqual(destination.name, "_thickness_vs_distance_from_fovea_LIGHT2.xlsx")
            self.assertEqual(read_thickness_rows(destination)[1:], [[-1, 0, None, 100], [0, 0, 20, 200]])
            self.assertEqual(source.read_bytes(), original)

    def test_current_temporal_light_is_preferred_over_direct_legacy_light2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subject = Path(temp_dir)
            side = subject / "temporal"
            current = side / "exports" / "_thickness_vs_distance_from_fovea_LIGHT.txt"
            current.parent.mkdir(parents=True)
            current.write_text("current", encoding="utf-8")
            (side / "_thickness_vs_distance_from_fovea_LIGHT2.txt").write_text("stale", encoding="utf-8")

            _nasal, temporal = find_light_files(subject)

            self.assertEqual(Path(temporal), current)

    def test_legacy_light2_remains_a_read_only_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subject = Path(temp_dir)
            side = subject / "nasal"
            side.mkdir()
            legacy = side / "_thickness_vs_distance_from_fovea_LIGHT2.txt"
            original = b"Distance\tWhole\tELM\tONL\r\n-1\t0\tNA\t100\r\n0\t0\t20\t200\r\n"
            legacy.write_bytes(original)

            self.assertEqual(Path(find_thickness_file(subject)), legacy)
            self.assertEqual(clean_light_thickness_file(legacy), str(legacy))
            blocks, _temporal, filename, _other, count, _other_count = extract_elmrpe_onl_from_subject(
                subject, "ONL", False,
            )

            self.assertEqual(Path(filename), legacy)
            self.assertEqual(blocks[0], 150)
            self.assertEqual(count, 2)
            self.assertEqual(legacy.read_bytes(), original)
            self.assertEqual(list(side.iterdir()), [legacy])

    def test_legacy_thickness_is_preferred_to_unrelated_current_light_profiles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subject = Path(temp_dir)
            side = subject / "nasal"
            side.mkdir()
            legacy = side / "_thickness_vs_distance_from_fovea_LIGHT2.txt"
            original = b"Distance\tWhole\tELM\tONL\n-1\t0\t10\t20\n0\t0\t30\t40\n"
            legacy.write_bytes(original)
            profile = side / "_fovea_light_profiles_LIGHT.txt"
            _write_fovea(profile, 55.77)
            profile_bytes = profile.read_bytes()

            self.assertEqual(Path(find_thickness_file(side)), legacy)
            nasal, _temporal = find_light_files(subject)
            self.assertEqual(Path(nasal), legacy)
            blocks, _temporal, filename, _other, count, _other_count = extract_elmrpe_onl_from_subject(
                subject, "ONL", False,
            )

            self.assertEqual(Path(filename), legacy)
            self.assertEqual(blocks[0], 30)
            self.assertEqual(count, 2)
            self.assertEqual(legacy.read_bytes(), original)
            self.assertEqual(profile.read_bytes(), profile_bytes)

    def test_profile_only_folders_are_not_cleaned_or_used_as_thickness_tables(self):
        for filename in ("_light_profiles_LIGHT.txt", "_fovea_light_profiles_LIGHT.txt",
                         "_fovea_light_LIGHT.txt", "_fovea_light_profiles_LIGHT2.txt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp_dir:
                subject = Path(temp_dir)
                side = subject / "nasal"
                side.mkdir()
                profile = side / filename
                original = b"foveaLIGHT\t1\t2\r\nRPEtoOLM\tNA\tNA\r\n"
                profile.write_bytes(original)

                self.assertIsNone(find_thickness_file(side))
                self.assertEqual(find_light_files(subject), (None, None))
                for measurement in ("ELM-RPE", "ONL"):
                    nasal, temporal, nasal_file, temporal_file, nasal_count, temporal_count = (
                        extract_elmrpe_onl_from_subject(subject, measurement, False)
                    )
                    self.assertTrue(all(value is None for value in nasal + temporal))
                    self.assertIsNone(nasal_file)
                    self.assertIsNone(temporal_file)
                    self.assertEqual((nasal_count, temporal_count), (0, 0))
                self.assertEqual(profile.read_bytes(), original)
                self.assertEqual(list(side.iterdir()), [profile])

    def test_mixed_missing_elm_and_onl_rows_keep_former_values_and_save_cleaned_excel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            side = root / "Control" / "Subject_1" / "nasal"
            side.mkdir(parents=True)
            _write_results(side / "rr_MCPAR.xlsx", 1)
            source = side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
            source.write_text(
                "Distance\tWhole\tELM\tONL\n"
                "-3\t0\tNA\t999\n"
                "-2\t0\t20\tNA\n"
                "-1\t0\t30\t300\n"
                "0\t0\tNA\t400\n"
                "1\t0\tNA\t500\n"
                "0\t0\tNA\t600\n"
                "2\t0\t40\t700\n",
                encoding="utf-8",
            )
            output = root / "compiled.xlsx"
            original = source.read_bytes()

            compile_step4_results(root, output, include_fovea=False)

            workbook = load_workbook(output, data_only=True)
            try:
                self.assertEqual(workbook["ELM-RPE"]["B3"].value, 30)
                self.assertAlmostEqual(workbook["ONL"]["B3"].value, (300 + 400 + 700) / 3)
                log = workbook["Extraction_log"]
                self.assertEqual(log["G3"].value, 3)
                self.assertEqual(log["G4"].value, 3)
            finally:
                workbook.close()
            self.assertEqual(
                sorted(path.name for path in side.iterdir()),
                ["_thickness_vs_distance_from_fovea_LIGHT.txt", "_thickness_vs_distance_from_fovea_LIGHT2.xlsx", "rr_MCPAR.xlsx"],
            )
            self.assertEqual(source.read_bytes(), original)

    def test_compile_matches_legacy_layout_and_reads_current_step4_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subject = _build_subject(root)
            output = root / DEFAULT_OUTPUT_FILENAME
            log_messages = []
            progress_events = []

            result = compile_step4_results(
                root,
                output,
                include_fovea=True,
                log_callback=log_messages.append,
                progress_callback=lambda completed, total, message: progress_events.append(
                    (completed, total, message)
                ),
            )

            self.assertEqual(result.output_path, output)
            self.assertEqual(result.le_ids, ("7_0_0",))
            self.assertEqual(result.re_ids, ())
            self.assertEqual(
                result.measures,
                (
                    MeasureCompilation("rrMCP-AR", 1, 0),
                    MeasureCompilation("ELM-RPE", 1, 0),
                    MeasureCompilation("ONL", 1, 0),
                ),
            )
            self.assertTrue(any("LE subject(s) found: 7_0_0" in line for line in log_messages))
            self.assertTrue(log_messages[-1].endswith(str(output)))
            self.assertEqual(progress_events[0], (0, 1, "Scanning input folders..."))
            self.assertEqual(progress_events[1][:2], (1, 5))
            self.assertEqual(
                [event[:2] for event in progress_events if event[2].startswith("Processed ")],
                [(2, 5), (3, 5), (4, 5)],
            )
            self.assertEqual(progress_events[-2], (4, 5, "Saving workbook..."))
            self.assertEqual(progress_events[-1], (5, 5, "Saved compiled workbook."))
            self.assertEqual(
                [completed for completed, _total, _message in progress_events],
                sorted(completed for completed, _total, _message in progress_events),
            )

            workbook = load_workbook(output, data_only=True)
            try:
                self.assertEqual(
                    workbook.sheetnames,
                    ["rrMCP-AR", "ELM-RPE", "ONL", "Extraction_log"],
                )

                rr_sheet = workbook["rrMCP-AR"]
                self.assertEqual(rr_sheet.max_row, 22)
                self.assertEqual(rr_sheet["A1"].value, "LE")
                self.assertIsNone(rr_sheet["E1"].value)
                self.assertEqual(
                    [rr_sheet.cell(2, column).value for column in range(1, 8)],
                    ["ID", "Nasal", "Temporal", None, None, None, None],
                )
                self.assertEqual(rr_sheet["A3"].value, "7_0_0")
                self.assertEqual(rr_sheet["B3"].value, 1)
                self.assertEqual(rr_sheet["B22"].value, 20)
                self.assertEqual(rr_sheet["C3"].value, 101)
                self.assertEqual(rr_sheet["C22"].value, 120)
                self.assertTrue(rr_sheet["A3"].font.bold)
                self.assertEqual(rr_sheet["A3"].alignment.horizontal, "center")
                self.assertEqual(rr_sheet["A3"].alignment.vertical, "center")
                self.assertEqual(rr_sheet["B3"].number_format, "0.000")
                for column in ("A", "B", "C"):
                    self.assertEqual(rr_sheet.column_dimensions[column].width, 12.0)

                elm_sheet = workbook["ELM-RPE"]
                self.assertEqual(elm_sheet.max_row, 26)
                self.assertEqual(elm_sheet["A3"].value, "7_0_0")
                self.assertAlmostEqual(elm_sheet["B3"].value, 55.77)
                self.assertAlmostEqual(elm_sheet["C3"].value, 66.88)
                self.assertEqual(elm_sheet["B4"].value, 10.0)
                self.assertEqual(elm_sheet["B26"].value, 10.0)
                self.assertEqual(elm_sheet["C4"].value, 30.0)

                onl_sheet = workbook["ONL"]
                self.assertEqual(onl_sheet.max_row, 25)
                self.assertEqual(onl_sheet["A3"].value, "7_0_0")
                self.assertEqual(onl_sheet["B3"].value, 20.0)
                self.assertEqual(onl_sheet["B25"].value, 20.0)
                self.assertEqual(onl_sheet["C3"].value, 40.0)

                log_sheet = workbook["Extraction_log"]
                self.assertEqual(log_sheet.max_row, 4)
                self.assertEqual(log_sheet.max_column, 9)
                self.assertEqual(log_sheet["A2"].value, "rrMCP-AR")
                self.assertEqual(log_sheet["B2"].value, "LE")
                self.assertEqual(log_sheet["C2"].value, "7_0_0")
                self.assertEqual(log_sheet["I2"].value, "OK")
                self.assertTrue(str(log_sheet["E2"].value).endswith("rr_MCPAR.xlsx"))
                self.assertEqual(log_sheet["G2"].value, 20)
                self.assertEqual(log_sheet["G3"].value, 2300)
                self.assertTrue(log_sheet["A1"].font.bold)
                self.assertEqual(log_sheet["A1"].alignment.horizontal, "center")
                for column in "ABCDEFGHI":
                    self.assertEqual(log_sheet.column_dimensions[column].width, 28.0)
            finally:
                workbook.close()

            self.assertFalse(
                (subject / "nasal" / "_thickness_vs_distance_from_fovea_LIGHT2.txt").is_file()
            )
            self.assertFalse(
                (subject / "temporal" / "_thickness_vs_distance_from_fovea_LIGHT2.txt").is_file()
            )

    def test_fovea_can_be_excluded_from_elmrpe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_subject(root)
            output = root / "without_fovea.xlsx"

            compile_step4_results(root, output, include_fovea=False)

            workbook = load_workbook(output, data_only=True)
            try:
                elm_sheet = workbook["ELM-RPE"]
                self.assertEqual(elm_sheet.max_row, 25)
                self.assertEqual(elm_sheet["B3"].value, 10.0)
                self.assertEqual(elm_sheet["B25"].value, 10.0)
            finally:
                workbook.close()

    def test_current_step4_workbook_takes_precedence_over_legacy_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            _write_results(folder / "Results.xlsx", 100)
            current = folder / "rr_MCPAR.xlsx"
            _write_results(current, 1)

            self.assertEqual(Path(find_results_file(folder)), current)

    def test_skipped_subject_still_advances_each_measurement_progress_unit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nasal = root / "LE" / "8_0_0" / "nasal"
            nasal.mkdir(parents=True)
            workbook = Workbook()
            workbook.active.title = "in"
            workbook.active.cell(1, 7).value = "Round"
            workbook.save(nasal / "Results.xlsx")
            workbook.close()
            output = root / "skipped.xlsx"
            progress_events = []

            result = compile_step4_results(
                root,
                output,
                progress_callback=lambda completed, total, message: progress_events.append(
                    (completed, total, message)
                ),
            )

            self.assertEqual(
                result.measures,
                (
                    MeasureCompilation("rrMCP-AR", 0, 1),
                    MeasureCompilation("ELM-RPE", 0, 1),
                    MeasureCompilation("ONL", 0, 1),
                ),
            )
            self.assertEqual(
                [event[:2] for event in progress_events if event[2].startswith("Processed ")],
                [(2, 5), (3, 5), (4, 5)],
            )
            self.assertEqual(progress_events[-1][:2], (5, 5))

    def test_save_failure_does_not_report_full_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_subject(root)
            output = root / "locked.xlsx"
            progress_events = []

            with mock.patch(
                "aidas.services.step4_compiler.Workbook.save",
                side_effect=PermissionError("workbook is locked"),
            ):
                with self.assertRaisesRegex(PermissionError, "workbook is locked"):
                    compile_step4_results(
                        root,
                        output,
                        progress_callback=lambda completed, total, message: (
                            progress_events.append((completed, total, message))
                        ),
                    )

            self.assertEqual(progress_events[-1], (4, 5, "Saving workbook..."))
            self.assertFalse(
                any(completed == total for completed, total, _message in progress_events)
            )

    def test_empty_or_flat_folder_is_rejected_instead_of_saving_empty_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_results(root / "Results.xlsx", 1)
            output = root / "empty.xlsx"

            with self.assertRaisesRegex(ValueError, "No subject folders"):
                compile_step4_results(root, output)

            self.assertFalse(output.exists())

    def test_recursive_groups_nested_workbooks_and_extra_subject_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = {
                "Control": ("Subject_2", "Subject_10"),
                "Treatment": ("Subject_2",),
                "Validation": ("001_extra",),
            }
            sources = {}
            for group, ids in expected.items():
                for subject_id in ids:
                    subject = root / "Experiment" / "Visit 1" / group / subject_id
                    for side, value in (("Nasal", 1), ("TEMPORAL", 101)):
                        result_dir = subject / side / "exports" / "final"
                        result_dir.mkdir(parents=True)
                        source = result_dir / "measurements.xlsx"
                        _write_results(source, value)
                        sources[group, subject_id, side] = str(source)
                    images = subject / "Images" / "Previews"
                    images.mkdir(parents=True)
                    (images / "scan.png").write_bytes(b"preview")
                    notes = subject / "Notes"
                    notes.mkdir()
                    (notes / "notes.txt").write_text("notes", encoding="utf-8")

            # These branches must never create subjects or group blocks.
            thickness_only = root / "Other" / "TextOnly" / "nasal"
            thickness_only.mkdir(parents=True)
            _write_thickness(thickness_only / "_thickness_vs_distance_from_fovea_LIGHT.txt", 8, 9)
            unrelated = root / "Other" / "Unrelated" / "Documents"
            unrelated.mkdir(parents=True)
            _write_results(unrelated / "notes.xlsx", 900)
            locked = root / "Other" / "LockOnly" / "temporal"
            locked.mkdir(parents=True)
            (locked / "~$rr_MCPAR.xlsx").write_bytes(b"lock")
            output = root / DEFAULT_OUTPUT_FILENAME
            result = compile_step4_results(root, output)

            self.assertEqual(result.group_ids, expected)
            self.assertEqual(result.measures[0], MeasureCompilation("rrMCP-AR", 4, 0))
            with_workbook = load_workbook(output, data_only=True)
            try:
                sheet = with_workbook["rrMCP-AR"]
                for column, (group, ids) in zip((1, 5, 9), expected.items()):
                    self.assertEqual(sheet.cell(1, column).value, group)
                    self.assertEqual(sheet.cell(2, column).value, "ID")
                    for index, subject_id in enumerate(ids):
                        row = 3 + index * 20
                        self.assertEqual(sheet.cell(row, column).value, subject_id)
                        self.assertEqual(sheet.cell(row, column + 1).value, 1)
                        self.assertEqual(sheet.cell(row, column + 2).value, 101)
                log = with_workbook["Extraction_log"]
                self.assertEqual(log["B1"].value, "Group")
                rr_rows = [row for row in log.iter_rows(min_row=2, values_only=True) if row[0] == "rrMCP-AR"]
                self.assertEqual(len(rr_rows), 4)
                for row in rr_rows:
                    self.assertEqual(row[4], sources[row[1], row[2], "Nasal"])
                    self.assertEqual(row[5], sources[row[1], row[2], "TEMPORAL"])
            finally:
                with_workbook.close()

            # Re-running must not discover the combined output as an input.
            self.assertEqual(compile_step4_results(root, output).group_ids, expected)

    def test_duplicate_group_and_subject_names_are_reported_without_losing_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for batch in ("Batch1", "Batch2"):
                folder = root / batch / "Control" / "Subject_A" / "nasal"
                folder.mkdir(parents=True)
                _write_results(folder / "rr_MCPAR.xlsx", 1)
            output = root / "combined.xlsx"
            with self.assertRaisesRegex(ValueError, "Duplicate subject 'Subject_A' in group 'Control'"):
                compile_step4_results(root, output)
            self.assertFalse(output.exists())

    def test_temporal_only_subject_keeps_nasal_measurements_blank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            side = root / "Temporal group" / "Subject_A" / "temporal"
            side.mkdir(parents=True)
            _write_results(side / "rr_MCPAR.xlsx", 101)
            _write_thickness(side / "_thickness_vs_distance_from_fovea_LIGHT.txt", 30, 40)
            output = root / "compiled.xlsx"
            result = compile_step4_results(root, output, include_fovea=False)
            self.assertEqual(result.group_ids, {"Temporal group": ("Subject_A",)})
            workbook = load_workbook(output, data_only=True)
            try:
                for measure, value in (("rrMCP-AR", 101), ("ELM-RPE", 30), ("ONL", 40)):
                    self.assertIsNone(workbook[measure]["B3"].value)
                    self.assertEqual(workbook[measure]["C3"].value, value)
            finally:
                workbook.close()

    def test_full_names_that_start_with_equals_are_written_as_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            side = root / "=Control" / "=Subject_A" / "nasal"
            side.mkdir(parents=True)
            _write_results(side / "rr_MCPAR.xlsx", 1)
            output = root / "compiled.xlsx"
            compile_step4_results(root, output)
            workbook = load_workbook(output, data_only=False)
            try:
                for cell, value in (
                    (workbook["rrMCP-AR"]["A1"], "=Control"),
                    (workbook["rrMCP-AR"]["A3"], "=Subject_A"),
                    (workbook["Extraction_log"]["B2"], "=Control"),
                    (workbook["Extraction_log"]["C2"], "=Subject_A"),
                ):
                    self.assertEqual(cell.value, value)
                    self.assertEqual(cell.data_type, "s")
            finally:
                workbook.close()

    def test_nested_current_workbook_is_preferred_to_direct_legacy_workbook(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            _write_results(folder / "Results.xlsx", 100)
            nested = folder / "exports"
            nested.mkdir()
            current = nested / "rr_MCPAR.xlsx"
            _write_results(current, 1)
            self.assertEqual(Path(find_results_file(folder)), current)

    def test_output_cannot_replace_an_input_workbook(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = _build_subject(root) / "nasal" / "rr_MCPAR.xlsx"
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "different from the input"):
                compile_step4_results(root, source)
            self.assertEqual(source.read_bytes(), original)

    def test_output_cannot_replace_existing_or_planned_cleaned_thickness_excel(self):
        for source_type in ("raw", "raw_and_excel", "excel", "legacy_text"):
            with self.subTest(source_type=source_type), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                side = root / "Control" / "Subject_1" / "nasal"
                side.mkdir(parents=True)
                _write_results(side / "rr_MCPAR.xlsx", 1)
                source = side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
                source.write_bytes(b"Distance\tWhole\tELM\tONL\r\n0\t0\t3\t4\r\n")
                output = side / "_thickness_vs_distance_from_fovea_LIGHT2.xlsx"
                if source_type in ("raw_and_excel", "excel"):
                    create_cleaned_light_thickness_workbook(source)
                if source_type == "excel":
                    source.unlink()
                elif source_type == "legacy_text":
                    source.rename(output.with_suffix(".txt"))
                before = {path.name: path.read_bytes() for path in side.iterdir()}

                with self.assertRaisesRegex(ValueError, "different from the input thickness"):
                    compile_step4_results(root, output)

                self.assertEqual({path.name: path.read_bytes() for path in side.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
