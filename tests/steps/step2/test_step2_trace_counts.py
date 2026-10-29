"""Keep fast annotation summaries identical to the actual rasterized traces."""

import random
import unittest
from unittest import mock

from aidas.steps.step2_annotate import (
    BOUNDARY_NAMES,
    _polyline_pixel_count,
    _polyline_pixels,
)
from tests.support.annotation import make_annotation_frame


class PolylinePixelCountTests(unittest.TestCase):
    def assert_matches_pixels(self, points):
        self.assertEqual(_polyline_pixel_count(points), len(_polyline_pixels(points)))

    def test_empty_single_and_repeated_vertices(self):
        for points in (
            [], [(7, 11)], [(7, 11), (7, 11)],
            [(7, 11), (7, 11), (12, 20), (12, 20)],
        ):
            with self.subTest(points=points):
                self.assert_matches_pixels(points)

    def test_horizontal_vertical_diagonal_and_all_octants(self):
        for endpoint in (
            (17, 0), (-17, 0), (0, 17), (0, -17),
            (17, 17), (-17, 17), (17, -17), (-17, -17),
            (17, 5), (5, 17), (-17, 5), (-5, 17),
            (17, -5), (5, -17), (-17, -5), (-5, -17),
        ):
            with self.subTest(endpoint=endpoint):
                self.assert_matches_pixels([(0, 0), endpoint])
                self.assert_matches_pixels([endpoint, (0, 0), endpoint])

    def test_non_integer_vertices_use_the_same_truncation(self):
        self.assert_matches_pixels([
            (-0.9, 0.9), (1.9, -1.9), (1.1, -1.1), (-21.75, 30.8),
        ])

    def test_random_polylines_match_exactly(self):
        randomizer = random.Random(4102)
        for index in range(300):
            points = [
                (randomizer.uniform(-80, 80), randomizer.uniform(-80, 80))
                for _ in range(randomizer.randrange(25))
            ]
            with self.subTest(index=index):
                self.assert_matches_pixels(points)

    def test_summary_callbacks_preserve_text_without_rasterizing(self):
        frame = make_annotation_frame()
        name = BOUNDARY_NAMES[0]
        frame.active_boundary = name
        points = [(0, 0), (2999, 255), (0, 255), (2999, 0)]
        expected = f"Active: {name} | 4 vertices | {len(_polyline_pixels(points))} pixel(s) on line"
        frame.image_canvas.get_active_line.return_value = points
        with mock.patch(
            "aidas.steps.step2_annotate._polyline_pixels",
            side_effect=AssertionError("Summaries should not rasterize the line"),
        ):
            frame._on_active_line_changed(points)
            self.assertEqual(frame.active_trace_var.get(), expected)
            frame._update_active_trace_summary()
            self.assertEqual(frame.active_trace_var.get(), expected)


if __name__ == "__main__":
    unittest.main()
