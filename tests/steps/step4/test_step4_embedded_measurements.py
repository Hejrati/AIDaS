"""Stack exports retain exact pixels and matching per-folder workbooks."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from openpyxl import load_workbook
from PIL import Image

from aidas.services.step4_results import (
    STEP4_RESULTS_FILENAME,
    STEP4_RESULTS_HEADERS,
    encode_step4_measurements,
    read_step4_measurements,
)
from aidas.steps import step4_analyze_isez as module


class Step4EmbeddedMeasurementTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        sample = np.arange(450)
        profile = 100 + 200 * np.exp(-((sample - 88) / 8) ** 2)
        self.image = np.repeat(profile[:, None], 800, axis=1)
        self.results = [
            module.analyze_and_save_roi(
                self.image, module.ISezROI(f"{index:02}", left, left + 10),
                start_click=76, end_click=100,
            )
            for index, left in enumerate((101, 301), start=1)
        ]

    def _plain_stack(self, *, description=None, filename=STEP4_RESULTS_FILENAME):
        image = Image.new("L", (16, 12), 71)
        try:
            if description is None:
                image.save(self.directory / filename)
            else:
                image.save(self.directory / filename, description=description)
        finally:
            image.close()

    def _assert_workbook_matches(self, rows):
        workbook = load_workbook(self.directory / "rr_MCPAR.xlsx", data_only=True, read_only=True)
        try:
            saved = list(workbook.active.iter_rows(values_only=True))
        finally:
            workbook.close()
        self.assertEqual(saved[0], STEP4_RESULTS_HEADERS)
        self.assertEqual(len(saved) - 1, len(rows))
        for actual, expected in zip(saved[1:], rows):
            self.assertEqual(actual[0], expected[0])
            for actual_value, expected_value in zip(actual[1:], expected[1:]):
                if expected_value is None:
                    self.assertIsNone(actual_value)
                else:
                    self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_build_writes_two_tiffs_and_workbook_with_unchanged_pixels_and_exact_measurements(self):
        expected_frames = [module.make_isez_plot_image(result) for result in self.results]
        expected_rows = [
            [index, *(row[name] for name in STEP4_RESULTS_HEADERS[1:])]
            for index, frame in enumerate(expected_frames, start=1)
            for row in [module.imagej_shape_measurements_from_frame(frame)]
        ]

        module.Step4Frame._write_stack_output_files(self.image, self.results, self.directory)

        self.assertEqual(
            {path.name for path in self.directory.iterdir()},
            {"MAX_Stack.tif", STEP4_RESULTS_FILENAME, "rr_MCPAR.xlsx"},
        )
        self.assertEqual(read_step4_measurements(self.directory / STEP4_RESULTS_FILENAME), expected_rows)
        self._assert_workbook_matches(expected_rows)
        stack = Image.open(self.directory / STEP4_RESULTS_FILENAME)
        try:
            self.assertEqual(stack.n_frames, len(expected_frames))
            for index, expected in enumerate(expected_frames):
                stack.seek(index)
                np.testing.assert_array_equal(np.asarray(stack), np.asarray(expected))
        finally:
            stack.close()
            for image in expected_frames:
                image.close()
        projection = Image.open(self.directory / "MAX_Stack.tif")
        try:
            np.testing.assert_array_equal(
                np.asarray(projection),
                module.max_stack_projection_image(self.image, [result.roi for result in self.results]),
            )
        finally:
            projection.close()
        self.assertTrue(module.Step4Frame._step4_outputs_complete(self.directory))
        self.assertEqual(module.Step4Frame._step4_output_count(self.directory), 2)

    def test_rebuild_replaces_former_workbook_with_current_tiff_measurements(self):
        old_workbook = self.directory / "rr_MCPAR.xlsx"
        old_workbook.write_bytes(b"old generated measurements")

        module.Step4Frame._write_stack_output_files(self.image, self.results, self.directory)

        first_rows = read_step4_measurements(self.directory / STEP4_RESULTS_FILENAME)
        self._assert_workbook_matches(first_rows)

        module.Step4Frame._write_stack_output_files(
            self.image, self.results, self.directory, excluded_roi_indices=(1,),
        )

        rebuilt_rows = read_step4_measurements(self.directory / STEP4_RESULTS_FILENAME)
        self.assertNotEqual(rebuilt_rows, first_rows)
        self._assert_workbook_matches(rebuilt_rows)

    def test_unprocessed_exclusions_preserve_included_frames_and_measurement_positions(self):
        ordered = [None, self.results[0], None, self.results[1], None]
        excluded = (1, 3, 5)
        expected_frames = [module.make_isez_plot_image(result) for result in self.results]
        try:
            expected_measurements = [module.imagej_shape_measurements_from_frame(frame) for frame in expected_frames]
            module.Step4Frame._write_stack_output_files(
                self.image, ordered, self.directory, excluded_roi_indices=excluded,
            )

            rows = read_step4_measurements(self.directory / STEP4_RESULTS_FILENAME)
            self._assert_workbook_matches(rows)
            self.assertEqual([row[0] for row in rows], list(range(1, 6)))
            for index in excluded:
                self.assertEqual(rows[index - 1], [index] + [None] * 7)
            for index, expected in zip((2, 4), expected_measurements):
                self.assertEqual(rows[index - 1], [index, *(expected[name] for name in STEP4_RESULTS_HEADERS[1:])])

            with Image.open(self.directory / STEP4_RESULTS_FILENAME) as stack:
                self.assertEqual(stack.n_frames, 5)
                for frame_index, expected in zip((1, 3), expected_frames):
                    stack.seek(frame_index)
                    np.testing.assert_array_equal(np.asarray(stack), np.asarray(expected))
                for index in excluded:
                    stack.seek(index - 1)
                    self.assertEqual(stack.size, module.IMAGEJ_ISEZ_CANVAS_SIZE)
                    # Exclusion labels occupy the top; no measured curve or
                    # axis is invented in the remaining neutral plot frame.
                    self.assertTrue(np.all(np.asarray(stack)[80:] == 255))
            with Image.open(self.directory / "MAX_Stack.tif") as projection:
                np.testing.assert_array_equal(
                    np.asarray(projection),
                    module.max_stack_projection_image(self.image, [result.roi for result in self.results]),
                )
            self.assertTrue(module.Step4Frame._step4_outputs_complete(self.directory))
        finally:
            for frame in expected_frames:
                frame.close()

    def test_excluding_a_saved_result_suppresses_its_measurements_and_overlay(self):
        module.Step4Frame._write_stack_output_files(
            self.image, self.results, self.directory, excluded_roi_indices=(1,),
        )

        rows = read_step4_measurements(self.directory / STEP4_RESULTS_FILENAME)
        self._assert_workbook_matches(rows)
        self.assertEqual(rows[0], [1] + [None] * 7)
        self.assertIsNotNone(rows[1][6])
        with Image.open(self.directory / STEP4_RESULTS_FILENAME) as stack:
            self.assertEqual(stack.n_frames, 2)
            self.assertTrue(np.all(np.asarray(stack)[80:] == 255))
        with Image.open(self.directory / "MAX_Stack.tif") as projection:
            np.testing.assert_array_equal(
                np.asarray(projection),
                module.max_stack_projection_image(self.image, [self.results[1].roi]),
            )

    def test_all_21_excluded_rois_write_complete_blank_measurement_outputs(self):
        module.Step4Frame._write_stack_output_files(
            self.image, [None] * 21, self.directory, excluded_roi_indices=range(1, 22),
        )

        self.assertEqual(
            {path.name for path in self.directory.iterdir()},
            {"MAX_Stack.tif", STEP4_RESULTS_FILENAME, "rr_MCPAR.xlsx"},
        )
        self.assertEqual(
            read_step4_measurements(self.directory / STEP4_RESULTS_FILENAME),
            [[index] + [None] * 7 for index in range(1, 22)],
        )
        self._assert_workbook_matches([[index] + [None] * 7 for index in range(1, 22)])
        with Image.open(self.directory / STEP4_RESULTS_FILENAME) as stack:
            self.assertEqual(stack.n_frames, 21)
            for index in range(21):
                stack.seek(index)
                self.assertEqual(stack.size, module.IMAGEJ_ISEZ_CANVAS_SIZE)
                self.assertTrue(np.all(np.asarray(stack)[80:] == 255))
        with Image.open(self.directory / "MAX_Stack.tif") as projection:
            np.testing.assert_array_equal(np.asarray(projection), module.to_uint8_display(self.image))
        self.assertTrue(module.Step4Frame._step4_outputs_complete(self.directory))
        self.assertEqual(module.Step4Frame._step4_output_count(self.directory), 2)

    def test_unconfirmed_included_roi_fails_before_creating_or_overwriting_outputs(self):
        new_directory = self.directory / "new"
        with self.assertRaisesRegex(ValueError, "included ROI"):
            module.Step4Frame._write_stack_output_files(
                self.image, [self.results[0], None], new_directory, excluded_roi_indices=(1,),
            )
        self.assertFalse(new_directory.exists())

        previous = {
            STEP4_RESULTS_FILENAME: b"previous stack",
            "MAX_Stack.tif": b"previous projection",
            "rr_MCPAR.xlsx": b"previous measurements",
        }
        for name, content in previous.items():
            (self.directory / name).write_bytes(content)
        with self.assertRaisesRegex(ValueError, "included ROI"):
            module.Step4Frame._write_stack_output_files(self.image, [None, self.results[1]], self.directory)
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.directory.iterdir()}, previous,
        )

    def test_failed_tiff_write_preserves_the_existing_measurement_workbook(self):
        old_workbook = self.directory / "rr_MCPAR.xlsx"
        old_workbook.write_bytes(b"old generated measurements")
        with mock.patch.object(Image.Image, "save", side_effect=OSError("Disk full")):
            with self.assertRaisesRegex(OSError, "Disk full"):
                module.Step4Frame._write_stack_output_files(self.image, self.results, self.directory)

        self.assertEqual(old_workbook.read_bytes(), b"old generated measurements")

    def test_failed_metadata_validation_preserves_existing_workbook(self):
        old_workbook = self.directory / "rr_MCPAR.xlsx"
        old_workbook.write_bytes(b"old generated measurements")
        with mock.patch.object(module, "has_step4_measurements", return_value=False):
            with self.assertRaisesRegex(ValueError, "did not retain"):
                module.Step4Frame._write_stack_output_files(self.image, self.results, self.directory)

        self.assertTrue(old_workbook.exists())

    def test_locked_workbook_does_not_report_success_or_change_existing_workbook(self):
        old_workbook = self.directory / "rr_MCPAR.xlsx"
        old_workbook.write_bytes(b"previous workbook")
        with mock.patch.object(
            module, "write_step4_measurement_workbook", side_effect=PermissionError("Workbook is locked"),
        ):
            with self.assertRaisesRegex(PermissionError, "locked"):
                module.Step4Frame._write_stack_output_files(self.image, self.results, self.directory)

        self.assertEqual(old_workbook.read_bytes(), b"previous workbook")

    def test_plain_stack_requires_measurement_metadata_or_historical_workbook(self):
        self._plain_stack()
        (self.directory / "MAX_Stack.tif").touch()
        self.assertFalse(module.Step4Frame._step4_outputs_complete(self.directory))
        self.assertEqual(module.Step4Frame._step4_output_count(self.directory), 1)

        old_workbook = self.directory / "rr_MCPAR.xlsx"
        old_workbook.write_bytes(b"old generated measurements")
        self.assertTrue(module.Step4Frame._step4_outputs_complete(self.directory))
        self.assertEqual(old_workbook.read_bytes(), b"old generated measurements")

    def test_tiff_extension_variant_is_recognized(self):
        rows = [{name: .149 for name in STEP4_RESULTS_HEADERS[1:]}]
        self._plain_stack(
            description=encode_step4_measurements(rows), filename="ROI_to_move_stck.tiff",
        )
        (self.directory / "MAX_Stack.tif").touch()

        self.assertTrue(module.Step4Frame._step4_outputs_complete(self.directory))

    def test_single_folder_continue_button_requires_embedded_measurements(self):
        self._plain_stack()
        (self.directory / "MAX_Stack.tif").touch()
        frame = object.__new__(module.Step4Frame)
        frame.continue_to_step5_button = mock.Mock()
        frame.batch_roi_notebook = None
        frame.current_path = self.directory / "_flat_LIGHT.hdr"
        frame.image = self.image
        frame.output_dir_var = SimpleNamespace(get=lambda: str(self.directory))
        frame._stack_build_complete = True
        frame._completion_progress_counts = mock.Mock(return_value=(1, 1))

        frame._update_continue_to_step5_button_state()
        frame.continue_to_step5_button.state.assert_called_with(["disabled"])
        rows = [{name: .149 for name in STEP4_RESULTS_HEADERS[1:]}]
        self._plain_stack(description=encode_step4_measurements(rows))
        frame._update_continue_to_step5_button_state()
        frame.continue_to_step5_button.state.assert_called_with(["!disabled"])

    def test_batch_scan_reports_two_outputs_and_keeps_incomplete_stack_selectable(self):
        self._plain_stack()
        (self.directory / "MAX_Stack.tif").touch()
        (self.directory / "_flat_LIGHT.hdr").touch()
        (self.directory / "_flat_LIGHT.img").touch()
        frame = object.__new__(module.Step4Frame)
        with mock.patch.object(
            module, "find_candidate_directories", return_value=({self.directory}, 1, []),
        ):
            rows, *_ = frame._scan_batch_roi_folders(self.directory)
            self.assertEqual(rows[0]["outputs"], "1/2")
            self.assertTrue(rows[0]["include"])
            (self.directory / "rr_MCPAR.xlsx").touch()
            rows, *_ = frame._scan_batch_roi_folders(self.directory)
            self.assertEqual(rows[0]["outputs"], "2/2")
            self.assertFalse(rows[0]["include"])


if __name__ == "__main__":
    unittest.main()
