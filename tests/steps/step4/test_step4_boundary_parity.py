"""Exact parity with the boundary tracer frozen before the performance audit."""

from __future__ import annotations

import itertools
import unittest
from unittest import mock

import numpy as np

from aidas.steps import step4_analyze_isez as step4


def reference_boundary_points(mask):
    """Original complete tracer, including its redundant coordinate sort."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    order = np.lexsort((xs, ys))
    start = (int(ys[order[0]]), int(xs[order[0]]))
    directions = [(0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    height, width = mask.shape
    point = start
    backtrack = 4
    boundary = []
    for _ in range(mask.size * 4):
        boundary.append(point)
        found = False
        for offset in range(8):
            idx = (backtrack + 1 + offset) % 8
            dy, dx = directions[idx]
            candidate = (point[0] + dy, point[1] + dx)
            if 0 <= candidate[0] < height and 0 <= candidate[1] < width and mask[candidate]:
                backtrack = (idx + 4) % 8
                point = candidate
                found = True
                break
        if not found or (point == start and len(boundary) > 1):
            break
    return np.asarray([(x, y) for y, x in boundary], dtype=np.float64)


class Step4BoundaryParityTests(unittest.TestCase):
    def _assert_trace_exact(self, mask):
        before = mask.tobytes(order="C")
        expected = reference_boundary_points(mask)
        actual = step4._imagej_boundary_points(mask)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(order="C"), expected.tobytes(order="C"))
        self.assertEqual(step4._imagej_perimeter(actual).hex(), step4._imagej_perimeter(expected).hex())
        self.assertEqual(mask.tobytes(order="C"), before)

    def test_every_three_by_three_mask_retains_exact_trace_and_perimeter(self):
        for bits in itertools.product((False, True), repeat=9):
            mask = np.array(bits, dtype=bool).reshape(3, 3)
            for source in (mask, np.asfortranarray(mask), mask[:, ::-1], mask.T):
                self._assert_trace_exact(source)

    def test_disconnected_hollow_diagonal_empty_and_noncontiguous_masks(self):
        disconnected = np.zeros((33, 47), dtype=bool)
        disconnected[2:7, 30:40] = True
        disconnected[18:29, 3:15] = True
        hollow = np.zeros((33, 47), dtype=bool)
        hollow[3:29, 4:40] = True
        hollow[6:25, 8:35] = False
        masks = {
            "empty": np.empty((0, 0), dtype=bool),
            "empty_rows": np.empty((0, 7), dtype=bool),
            "empty_columns": np.empty((7, 0), dtype=bool),
            "blank": np.zeros((13, 17), dtype=bool),
            "single": np.pad(np.ones((1, 1), dtype=bool), ((3, 4), (5, 6))),
            "full": np.ones((17, 31), dtype=bool),
            "disconnected": disconnected,
            "hollow": hollow,
            "diagonal": np.eye(31, dtype=bool),
        }
        for name, mask in masks.items():
            for dtype in (np.bool_, np.uint8, np.int16, np.float64):
                values = mask.astype(dtype)
                for source in (values, np.asfortranarray(values), values[:, ::-1], values.T):
                    with self.subTest(case=name, dtype=np.dtype(dtype).str, strides=source.strides):
                        self._assert_trace_exact(source)

    def test_numeric_truth_values_select_the_original_start(self):
        mask = np.array([[0, 0, -3, 0], [0, np.nan, 2, 0], [0, 0, 0, np.inf]])
        for source in (mask, mask[:, ::-1], np.asfortranarray(mask), mask.T):
            self._assert_trace_exact(source)

    def test_generated_roi_frames_preserve_raw_geometry_and_saved_measurements(self):
        samples = np.arange(1, 152, dtype=np.float64)
        for start, end, peak in ((76, 100, 88), (73, 102, 86), (70, 105, 90)):
            profile = 12 + 30 * np.exp(-((samples - peak) / 8) ** 2)
            image = np.repeat((3 * profile)[:, None], 80, axis=1)
            result = step4.analyze_and_save_roi(
                image, step4.ISezROI("01", 7, 27, low=1, high=151),
                start_click=start, end_click=end,
            )
            with step4.make_isez_plot_image(result) as frame:
                mask = step4._imagej_wand_mask(frame)
                self._assert_trace_exact(mask)
                expected_points = reference_boundary_points(mask)
                actual_points = step4._imagej_boundary_points(mask)
                expected_perimeter = step4._imagej_perimeter(expected_points)
                actual_perimeter = step4._imagej_perimeter(actual_points)
                self.assertEqual(actual_perimeter.hex(), expected_perimeter.hex())
                area = float(np.count_nonzero(mask))
                expected_circularity = min(1.0, 4 * step4.math.pi * area / (expected_perimeter * expected_perimeter))
                actual_circularity = min(1.0, 4 * step4.math.pi * area / (actual_perimeter * actual_perimeter))
                self.assertEqual(actual_circularity.hex(), expected_circularity.hex())
                with mock.patch.object(step4, "_imagej_boundary_points", side_effect=reference_boundary_points):
                    expected = step4.imagej_shape_measurements_from_frame(frame)
                actual = step4.imagej_shape_measurements_from_frame(frame)
                self.assertEqual({key: value.hex() for key, value in actual.items()},
                                 {key: value.hex() for key, value in expected.items()})


if __name__ == "__main__":
    unittest.main()
