from __future__ import annotations

import itertools
import unittest

from aidas.canvas.annotation_labels import (
    LabelSpec,
    PolylineAnchorIndex,
    layout_annotation_labels,
    visible_polyline_anchor,
)


def _intersects(first, second, gap=0):
    return (
        first[0] < second[2] + gap
        and first[2] + gap > second[0]
        and first[1] < second[3] + gap
        and first[3] + gap > second[1]
    )


class AnnotationLabelLayoutTests(unittest.TestCase):
    def assert_readable(self, boxes, specs, viewport, obstacles=()):
        by_key = {spec.key: spec for spec in specs}
        self.assertEqual(set(boxes), set(by_key))
        for key, (left, top, right, bottom) in boxes.items():
            self.assertGreaterEqual(left, viewport[0])
            self.assertGreaterEqual(top, viewport[1])
            self.assertLessEqual(right, viewport[2])
            self.assertLessEqual(bottom, viewport[3])
            self.assertAlmostEqual(right - left, by_key[key].width)
            self.assertAlmostEqual(bottom - top, by_key[key].height)
            for obstacle in obstacles:
                self.assertFalse(_intersects(boxes[key], obstacle, gap=4))
        for first, second in itertools.combinations(boxes.values(), 2):
            self.assertFalse(_intersects(first, second, gap=4))

    def test_close_retinal_traces_form_a_readable_staircase_in_anatomical_order(self):
        specs = [LabelSpec(index, (25, 90 + index * 2), 40, 15) for index in range(6)]

        boxes = layout_annotation_labels(specs, (0, 0, 300, 200))

        self.assert_readable(boxes, specs, (0, 0, 300, 200))
        for earlier, later in zip(specs, specs[1:]):
            self.assertGreater(boxes[later.key][0], boxes[earlier.key][0])
            self.assertGreaterEqual(boxes[later.key][1], boxes[earlier.key][3] + 4)

    def test_short_viewport_adds_columns_instead_of_overlapping_or_hiding_labels(self):
        specs = [LabelSpec(index, (5, 80 + index), 50, 20) for index in range(12)]

        boxes = layout_annotation_labels(specs, (0, 0, 300, 200))

        self.assert_readable(boxes, specs, (0, 0, 300, 200))
        self.assertGreater(boxes[6][0], boxes[5][0])
        self.assertLess(boxes[6][1], boxes[5][1])

    def test_narrow_viewport_reduces_horizontal_steps_without_changing_label_size(self):
        specs = [LabelSpec(index, (1, 50 + index), 68, 15) for index in range(6)]

        boxes = layout_annotation_labels(specs, (0, 0, 80, 180))

        self.assert_readable(boxes, specs, (0, 0, 80, 180))

    def test_a_fovea_label_is_reserved_as_an_obstacle(self):
        specs = [LabelSpec(index, (4, 38 + index * 2), 45, 16) for index in range(6)]
        obstacles = [(0, 16, 78, 65)]

        boxes = layout_annotation_labels(specs, (0, 0, 300, 200), obstacles=obstacles)

        self.assert_readable(boxes, specs, (0, 0, 300, 200), obstacles)

    def test_panning_translates_layout_and_anchors_together(self):
        specs = [LabelSpec(index, (5, 80 + index * 3), 40, 16) for index in range(6)]
        original = layout_annotation_labels(specs, (0, 0, 300, 200))
        shifted_specs = [LabelSpec(item.key, (item.anchor[0] + 150, item.anchor[1] + 75), item.width, item.height) for item in specs]

        shifted = layout_annotation_labels(shifted_specs, (150, 75, 450, 275))

        for key, bounds in original.items():
            self.assertEqual(shifted[key], tuple(value + (150 if index % 2 == 0 else 75) for index, value in enumerate(bounds)))

    def test_input_order_does_not_change_identically_anchored_labels(self):
        specs = [LabelSpec(key, (4, 30), 45, 18) for key in ("RPE", "ILM", "ONL")]

        forward = layout_annotation_labels(specs, (0, 0, 300, 200))
        backward = layout_annotation_labels(reversed(specs), (0, 0, 300, 200))

        self.assertEqual(forward, backward)
        self.assertEqual(list(forward), ["ILM", "ONL", "RPE"])

    def test_empty_or_physically_oversized_labels_have_no_boxes(self):
        self.assertEqual(layout_annotation_labels([], (0, 0, 300, 200)), {})
        self.assertEqual(layout_annotation_labels([LabelSpec("large", (0, 0), 400, 20)], (0, 0, 300, 200)), {})
        self.assertEqual(layout_annotation_labels([LabelSpec("flat", (0, 0), 20, 0)], (0, 0, 300, 200)), {})


class VisiblePolylineAnchorTests(unittest.TestCase):
    viewport = (10, 20, 110, 120)

    def test_crossing_segment_without_any_visible_vertex_is_interpolated(self):
        self.assertEqual(visible_polyline_anchor([(-40, 30), (160, 110)], self.viewport), (10, 50))

    def test_segment_enters_through_horizontal_viewport_edge(self):
        self.assertEqual(visible_polyline_anchor([(30, -20), (70, 60)], self.viewport), (50, 20))

    def test_nonmonotonic_trace_uses_leftmost_visible_point_without_mutation(self):
        points = [(90, 40), (20, 70), (60, 100), (0, 100)]
        original = list(points)

        self.assertEqual(visible_polyline_anchor(points, self.viewport), (10, 100))
        self.assertEqual(visible_polyline_anchor(reversed(points), self.viewport), (10, 100))
        self.assertEqual(points, original)

    def test_vertical_trace_uses_topmost_visible_point(self):
        self.assertEqual(visible_polyline_anchor([(40, 150), (40, 0)], self.viewport), (40, 20))

    def test_a_single_visible_vertex_or_corner_touch_is_visible(self):
        self.assertEqual(visible_polyline_anchor([(10, 20)], self.viewport), (10, 20))
        self.assertEqual(visible_polyline_anchor([(0, 30), (20, 10)], self.viewport), (10, 20))
        self.assertEqual(visible_polyline_anchor([(10, 20), (10, 20)], self.viewport), (10, 20))

    def test_wholly_invisible_traces_have_no_anchor(self):
        for points in ([], [(1, 1)], [(0, 10), (120, 10)], [(0, 0), (5, 100)]):
            with self.subTest(points=points):
                self.assertIsNone(visible_polyline_anchor(points, self.viewport))

    def test_invalid_vertex_does_not_join_unrelated_segments(self):
        self.assertIsNone(visible_polyline_anchor([(-50, 70), (float("nan"), 50), (160, 70)], self.viewport))


class PolylineAnchorIndexTests(unittest.TestCase):
    def test_index_matches_general_clipping_for_every_trace_shape(self):
        traces = [
            [],
            [(50, 50)],
            [(-40, 30), (160, 110)],
            [(30, -20), (70, 60)],
            [(90, 40), (20, 70), (60, 100), (0, 100)],
            [(40, 150), (40, 0)],
            [(10, 100), (10, 50), (10, 20), (40, 50)],
            [(0, 100), (10, 80), (10, 50), (10, 90), (20, 100)],
            [(-50, 70), (float("nan"), 50), (160, 70)],
        ]
        viewports = [(10, 20, 110, 120), (0, 0, 300, 200), (20, 20, 20, 100), (50, 40, 60, 60), (100, 100, 150, 150)]
        for points in traces:
            for trace in (points, list(reversed(points))):
                index = PolylineAnchorIndex(trace)
                for viewport in viewports:
                    with self.subTest(trace=trace, viewport=viewport):
                        self.assertEqual(index.anchor(viewport), visible_polyline_anchor(trace, viewport))

    def test_rebuilding_after_geometry_edit_uses_the_new_trace(self):
        points = [[0, 50], [200, 50]]
        before = PolylineAnchorIndex(points)
        points[0][1] = 100
        points[1][1] = 100
        after = PolylineAnchorIndex(points)

        self.assertEqual(before.anchor((10, 20, 110, 120)), (10, 50))
        self.assertEqual(after.anchor((10, 20, 110, 120)), (10, 100))

    def test_dense_trace_pan_and_vertical_crop_match_general_clipping(self):
        points = [(x, 40 + (x % 97)) for x in range(2133)]
        index = PolylineAnchorIndex(points)
        for viewport in ((900, 0, 1200, 200), (900, 100, 1200, 120), (900, 500, 1200, 700), (3000, 0, 3300, 200)):
            with self.subTest(viewport=viewport):
                self.assertEqual(index.anchor(viewport), visible_polyline_anchor(points, viewport))


if __name__ == "__main__":
    unittest.main()
