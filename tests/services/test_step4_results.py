"""Step 4 measurement metadata preserves values without decoding TIFF pixels."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from openpyxl import Workbook, load_workbook
from PIL import Image

from aidas.services.step4_results import (
    STEP4_RESULTS_HEADERS,
    encode_step4_measurements,
    has_step4_measurements,
    read_step4_measurements,
    write_step4_measurement_workbook,
)


class Step4ResultMetadataTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "ROI_to_move_stck.tif"
        self.measurements = [
            dict(zip(STEP4_RESULTS_HEADERS[1:], (228.364, 34.072, 71.276, .227, 6.702, .149, .939))),
            dict(zip(STEP4_RESULTS_HEADERS[1:], (130.295, 27.441, 58.563, .357, 4.748, .211, .978))),
        ]

    def _write(self, description=None):
        image = Image.new("L", (16, 12), 71)
        try:
            if description is None:
                image.save(self.path)
            else:
                image.save(self.path, description=description)
        finally:
            image.close()

    def test_measurements_round_trip_in_original_column_order(self):
        encoded = encode_step4_measurements(self.measurements)
        self.assertEqual(json.loads(encoded)["aidas_step4_measurements"]["version"], 1)
        self._write(encoded)
        expected = [
            [index, *(row[name] for name in STEP4_RESULTS_HEADERS[1:])]
            for index, row in enumerate(self.measurements, start=1)
        ]

        self.assertEqual(read_step4_measurements(self.path), expected)
        self.assertTrue(has_step4_measurements(self.path))
        # Windows cannot rename a TIFF that a reader left open.
        self.path.rename(self.path.with_name("moved.tif"))

    def test_reading_metadata_never_loads_the_plot_pixels(self):
        self._write(encode_step4_measurements(self.measurements))
        with mock.patch.object(Image.Image, "load", side_effect=AssertionError("Decoded TIFF pixels")):
            self.assertEqual(read_step4_measurements(self.path)[0][6], .149)

    def test_ordinary_tiff_and_missing_file_are_not_measurement_sources(self):
        for description in (None, "ImageJ=1.54\nimages=2\n", "not JSON", "[]"):
            with self.subTest(description=description):
                self._write(description)
                self.assertIsNone(read_step4_measurements(self.path))
        self.path.unlink()
        self.assertIsNone(read_step4_measurements(self.path))

    def test_invalid_measurement_records_are_rejected(self):
        valid = json.loads(encode_step4_measurements(self.measurements))
        for update in (
            {"version": 2},
            {"version": True},
            {"columns": ["Round"]},
            {"rows": []},
            {"rows": [[1, .149]]},
            {"rows": [[2, 228.364, 34.072, 71.276, .227, 6.702, .149, .939]]},
            {"rows": [[1, 228.364, 34.072, 71.276, .227, 6.702, float("nan"), .939]]},
            {"rows": [[1, 228.364, 34.072, 71.276, .227, 6.702, True, .939]]},
        ):
            with self.subTest(update=update):
                payload = {"aidas_step4_measurements": {**valid["aidas_step4_measurements"], **update}}
                self._write(json.dumps(payload))
                self.assertFalse(has_step4_measurements(self.path))

    def test_writer_rejects_empty_or_nonfinite_measurements(self):
        with self.assertRaises(ValueError):
            encode_step4_measurements([])
        self.measurements[0]["Round"] = float("inf")
        with self.assertRaises(ValueError):
            encode_step4_measurements(self.measurements)

    def test_excluded_rows_round_trip_as_blanks_without_shifting_the_next_roi(self):
        rows = [None, self.measurements[0], self.measurements[1], None]
        encoded = encode_step4_measurements(rows, excluded_roi_indices=(1, 4))
        metadata = json.loads(encoded)["aidas_step4_measurements"]
        self.assertEqual(metadata["version"], 2)
        self.assertEqual(metadata["excluded_rois"], [1, 4])
        self._write(encoded)

        decoded = read_step4_measurements(self.path)
        self.assertEqual([row[0] for row in decoded], [1, 2, 3, 4])
        self.assertEqual(decoded[0], [1] + [None] * 7)
        self.assertEqual(decoded[3], [4] + [None] * 7)
        self.assertEqual([row[6] for row in decoded], [None, .149, .211, None])
        self.assertTrue(has_step4_measurements(self.path))

    def test_exclusion_marker_overrides_original_numeric_measurements(self):
        metadata = json.loads(encode_step4_measurements(self.measurements))
        metadata["aidas_step4_measurements"].update(version=2, excluded_rois=[1])
        self._write(json.dumps(metadata))

        self.assertEqual(read_step4_measurements(self.path)[0], [1] + [None] * 7)
        self.assertEqual(read_step4_measurements(self.path)[1][6], .211)

    def test_all_excluded_measurements_are_valid_even_without_measured_rows(self):
        self._write(encode_step4_measurements([None] * 21, excluded_roi_indices=range(1, 22)))

        self.assertEqual(
            read_step4_measurements(self.path),
            [[index] + [None] * 7 for index in range(1, 22)],
        )

    def test_writer_rejects_invalid_exclusion_indices(self):
        for excluded in ((0,), (-1,), (3,), (True,), (1.0,), ("1",), (1, 1)):
            with self.subTest(excluded=excluded), self.assertRaises(ValueError):
                encode_step4_measurements(self.measurements, excluded_roi_indices=excluded)

    def test_null_measurements_require_an_explicit_exclusion(self):
        valid = json.loads(encode_step4_measurements(self.measurements))["aidas_step4_measurements"]
        for version, exclusions in ((1, {}), (2, {"excluded_rois": []}), (2, {"excluded_rois": [2]})):
            with self.subTest(version=version, exclusions=exclusions):
                rows = [[1] + [None] * 7, valid["rows"][1]]
                self._write(json.dumps({"aidas_step4_measurements": {
                    **valid, "version": version, "rows": rows, **exclusions,
                }}))
                self.assertIsNone(read_step4_measurements(self.path))
        with self.assertRaises((TypeError, ValueError)):
            encode_step4_measurements([None, self.measurements[1]], excluded_roi_indices=(2,))

    def test_reader_rejects_invalid_exclusion_markers_and_unsupported_versions(self):
        valid = json.loads(encode_step4_measurements(self.measurements))["aidas_step4_measurements"]
        for update in (
            {"version": 2},
            {"version": 2, "excluded_rois": None},
            {"version": 2, "excluded_rois": "1"},
            {"version": 2, "excluded_rois": [0]},
            {"version": 2, "excluded_rois": [3]},
            {"version": 2, "excluded_rois": [True]},
            {"version": 2, "excluded_rois": [1.0]},
            {"version": 2, "excluded_rois": [1, 1]},
            {"version": 2, "excluded_rois": [[]]},
            {"version": 1, "excluded_rois": [1]},
            {"version": 3, "excluded_rois": [1]},
        ):
            with self.subTest(update=update):
                self._write(json.dumps({"aidas_step4_measurements": {**valid, **update}}))
                self.assertIsNone(read_step4_measurements(self.path))


class Step4MeasurementWorkbookTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "rr_MCPAR.xlsx"
        self.measurement = dict(zip(
            STEP4_RESULTS_HEADERS[1:],
            (228.364, 34.072, 71.276, .227, 6.702, .149, .939),
        ))

    def _read(self):
        workbook = load_workbook(self.path, read_only=True, data_only=True)
        try:
            self.assertEqual(workbook.sheetnames, ["in"])
            return list(workbook.active.iter_rows(values_only=True))
        finally:
            workbook.close()

    def _assert_only_output_remains(self):
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_restores_historical_layout_and_exact_tiff_measurements(self):
        precise = {name: value / 7 for name, value in self.measurement.items()}
        rows = [self.measurement, precise]
        encoded_rows = json.loads(encode_step4_measurements(rows))["aidas_step4_measurements"]["rows"]

        write_step4_measurement_workbook(self.path, rows)

        self.assertEqual(self._read(), [STEP4_RESULTS_HEADERS, *(tuple(row) for row in encoded_rows)])
        self.assertEqual(self._read()[1][6], self.measurement["Round"])
        self._assert_only_output_remains()
        # No writer or reader may retain a Windows handle to the final file.
        self.path.rename(self.directory / "renamed.xlsx")

    def test_excluded_first_middle_and_foveola_slots_stay_in_place(self):
        rows = [dict(self.measurement) for _ in range(21)]
        for index in (0, 9, 20):
            rows[index] = None

        write_step4_measurement_workbook(self.path, rows, excluded_roi_indices=(1, 10, 21))

        saved = self._read()
        self.assertEqual(len(saved), 22)
        self.assertEqual([row[0] for row in saved[1:]], list(range(1, 22)))
        for index in (1, 10, 21):
            self.assertEqual(saved[index], (index, *([None] * 7)))
        for index in (2, 9, 11, 20):
            self.assertEqual(saved[index][6], self.measurement["Round"])

    def test_excluded_existing_measurements_are_omitted_and_all_excluded_is_valid(self):
        for rows in ([dict(self.measurement) for _ in range(21)], [None] * 21):
            with self.subTest(rows=rows):
                write_step4_measurement_workbook(self.path, rows, excluded_roi_indices=range(1, 22))
                self.assertEqual(
                    self._read(),
                    [STEP4_RESULTS_HEADERS, *((index, *([None] * 7)) for index in range(1, 22))],
                )
                self._assert_only_output_remains()

    def test_rebuild_replaces_existing_workbook_including_old_exclusion_slots(self):
        write_step4_measurement_workbook(self.path, [None, self.measurement], excluded_roi_indices=(1,))
        changed = {**self.measurement, "Round": .211}

        write_step4_measurement_workbook(self.path, [changed])

        self.assertEqual(self._read(), [STEP4_RESULTS_HEADERS, (1, *changed.values())])
        self._assert_only_output_remains()

    def test_failed_serialization_preserves_old_file_closes_writer_and_removes_temporary(self):
        write_step4_measurement_workbook(self.path, [self.measurement])
        original = self.path.read_bytes()
        workbook = Workbook()

        def fail_after_partial_write(stream):
            stream.write(b"partial workbook")
            raise OSError("The disk is full")

        with mock.patch("openpyxl.Workbook", return_value=workbook), \
                mock.patch.object(workbook, "save", side_effect=fail_after_partial_write), \
                mock.patch.object(workbook, "close", wraps=workbook.close) as close:
            with self.assertRaisesRegex(OSError, "disk is full"):
                write_step4_measurement_workbook(self.path, [self.measurement])

        close.assert_called_once_with()
        self.assertEqual(self.path.read_bytes(), original)
        self._assert_only_output_remains()

    def test_locked_destination_preserves_old_file_and_removes_temporary(self):
        write_step4_measurement_workbook(self.path, [self.measurement])
        original = self.path.read_bytes()
        replacement = {**self.measurement, "Round": .211}

        with mock.patch("aidas.services.step4_results.os.replace", side_effect=PermissionError("Excel has the file open")):
            with self.assertRaisesRegex(PermissionError, "Excel has the file open"):
                write_step4_measurement_workbook(self.path, [replacement])

        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self._read()[1][6], .149)
        self._assert_only_output_remains()

    def test_new_output_failure_leaves_no_partial_workbook(self):
        with mock.patch("aidas.services.step4_results.os.replace", side_effect=OSError("Cannot replace output")):
            with self.assertRaisesRegex(OSError, "Cannot replace output"):
                write_step4_measurement_workbook(self.path, [self.measurement])

        self.assertEqual(list(self.directory.iterdir()), [])

    def test_invalid_input_cannot_replace_existing_workbook(self):
        write_step4_measurement_workbook(self.path, [self.measurement])
        original = self.path.read_bytes()
        for rows, exclusions in (
            ([], ()),
            ([None], ()),
            ([self.measurement], (0,)),
            ([{**self.measurement, "Round": float("inf")}], ()),
        ):
            with self.subTest(rows=rows, exclusions=exclusions), self.assertRaises((TypeError, ValueError)):
                write_step4_measurement_workbook(self.path, rows, excluded_roi_indices=exclusions)
            self.assertEqual(self.path.read_bytes(), original)
            self._assert_only_output_remains()


if __name__ == "__main__":
    unittest.main()
