"""Step 4 measurement metadata preserves values without decoding TIFF pixels."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from PIL import Image

from aidas.services.step4_results import (
    STEP4_RESULTS_HEADERS,
    encode_step4_measurements,
    has_step4_measurements,
    read_step4_measurements,
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
        self._write(encode_step4_measurements(self.measurements))
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


if __name__ == "__main__":
    unittest.main()
