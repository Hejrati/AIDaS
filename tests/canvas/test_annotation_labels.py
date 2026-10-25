from __future__ import annotations

import math
from types import SimpleNamespace
import tkinter as tk
from tkinter import font as tkfont
import unittest
from unittest import mock

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas


class AnnotationLabelLayoutTests(unittest.TestCase):
    """Check rendered label geometry against the saved retinal traces."""

    def setUp(self):
        self.root = tk.Tk()
        self.addCleanup(self.root.destroy)
        self.root.attributes("-alpha", 0)
        self.root.geometry("900x520")
        self.changed = mock.Mock()
        self.edited = mock.Mock()
        self.widget = ImageCanvas(
            self.root, on_line_change=self.changed, on_line_edit=self.edited
        )
        self.widget.pack(fill="both", expand=True)
        self.root.update()
        self.image = np.arange(400 * 2133, dtype=np.uint16).reshape(400, 2133)
        self.widget.set_image(self.image)
        self.overlays = [
            {
                "label": label,
                "color": color,
                "points": [(x, 182 + index * 4 + x / 150) for x in (0, 650, 1400, 2132)],
            }
            for index, (label, color) in enumerate(
                (
                    ("ILM", "#a050ff"),
                    ("NFL", "#ff9800"),
                    ("GCL", "#7bc62d"),
                    ("IPL", "#f569b6"),
                    ("OPL", "#00bde5"),
                    ("RPE", "#ffd500"),
                )
            )
        ]
        self.widget.set_line_overlays(self.overlays)
        self.original = self.widget.get_line_overlays()
        self.root.update()

    def _settle_resize(self):
        settled = tk.BooleanVar(self.root, False)
        self.root.after(self.widget.RESIZE_REDRAW_DEBOUNCE_MS + 40, settled.set, True)
        self.root.wait_variable(settled)
        self.root.update_idletasks()

    def _labels(self):
        canvas = self.widget.canvas
        return {
            canvas.itemcget(item, "text"): item
            for item in self._visible_items("annotation-label-text")
        }

    def _visible_items(self, tag):
        canvas = self.widget.canvas
        return [
            item
            for item in canvas.find_withtag(tag)
            if canvas.itemcget(item, "state") != "hidden"
        ]

    def _color_matches(self, actual, expected):
        return self.root.winfo_rgb(actual) == self.root.winfo_rgb(expected)

    def _label_boxes(self, overlays):
        canvas = self.widget.canvas
        texts = self._labels()
        self.assertEqual(set(texts), {item["label"] for item in overlays})
        self.assertEqual(len(self._visible_items("annotation-label-text")), len(overlays))
        rectangles = [
            item
            for item in self._visible_items("annotation-label")
            if canvas.type(item) == "rectangle"
        ]
        self.assertEqual(len(rectangles), len(overlays))
        boxes = {}
        for overlay in overlays:
            label = overlay["label"]
            candidates = [
                item
                for item in rectangles
                if self._color_matches(canvas.itemcget(item, "outline"), overlay["color"])
            ]
            self.assertEqual(len(candidates), 1, label)
            box = canvas.bbox(candidates[0])
            text_box = canvas.bbox(texts[label])
            self.assertIsNotNone(box, label)
            self.assertIsNotNone(text_box, label)
            self.assertLessEqual(box[0], text_box[0], label)
            self.assertLessEqual(box[1], text_box[1], label)
            self.assertGreaterEqual(box[2], text_box[2], label)
            self.assertGreaterEqual(box[3], text_box[3], label)
            boxes[label] = box
        return boxes

    def _assert_readable(self, overlays=None):
        if overlays is None:
            overlays = self.widget.get_line_overlays()
        canvas = self.widget.canvas
        boxes = self._label_boxes(overlays)
        left, top = canvas.canvasx(0), canvas.canvasy(0)
        right = left + canvas.winfo_width()
        bottom = top + canvas.winfo_height()
        for label, (x0, y0, x1, y1) in boxes.items():
            self.assertGreaterEqual(x0, left, label)
            self.assertGreaterEqual(y0, top, label)
            self.assertLessEqual(x1, right, label)
            self.assertLessEqual(y1, bottom, label)
        for index, (label, box) in enumerate(boxes.items()):
            for other_label, other in list(boxes.items())[index + 1 :]:
                overlap_x = min(box[2], other[2]) - max(box[0], other[0])
                overlap_y = min(box[3], other[3]) - max(box[1], other[1])
                self.assertFalse(
                    overlap_x > 0 and overlap_y > 0,
                    f"{label} {box} overlaps {other_label} {other}",
                )
        return boxes

    @staticmethod
    def _distance_to_segment(point, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_squared = dx * dx + dy * dy
        fraction = 0 if not length_squared else max(
            0,
            min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared),
        )
        return math.hypot(point[0] - start[0] - fraction * dx, point[1] - start[1] - fraction * dy)

    def _assert_leaders_match_traces(self, overlays=None):
        if overlays is None:
            overlays = self.widget.get_line_overlays()
        canvas = self.widget.canvas
        boxes = self._label_boxes(overlays)
        links = self._visible_items("annotation-label-link")
        self.assertEqual(len(links), len(overlays))
        for overlay in overlays:
            candidates = [
                item
                for item in links
                if self._color_matches(canvas.itemcget(item, "fill"), overlay["color"])
            ]
            self.assertEqual(len(candidates), 1, overlay["label"])
            coordinates = canvas.coords(candidates[0])
            endpoints = (tuple(coordinates[:2]), tuple(coordinates[-2:]))
            trace = [self.widget._i2c(*point) for point in overlay["points"]]
            segments = list(zip(trace, trace[1:])) or [(trace[0], trace[0])]
            distances = [
                min(self._distance_to_segment(endpoint, *segment) for segment in segments)
                for endpoint in endpoints
            ]
            anchor_index = 0 if distances[0] <= distances[1] else 1
            self.assertLessEqual(distances[anchor_index], 1.1, overlay["label"])
            label_endpoint = endpoints[1 - anchor_index]
            x0, y0, x1, y1 = boxes[overlay["label"]]
            edges = [
                ((x0, y0), (x1, y0)),
                ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)),
                ((x0, y1), (x0, y0)),
            ]
            self.assertLessEqual(
                min(self._distance_to_segment(label_endpoint, *edge) for edge in edges),
                3,
                overlay["label"],
            )

    def test_crowded_traces_have_readable_staircase_labels_and_matching_leaders(self):
        boxes = self._assert_readable()
        self._assert_leaders_match_traces()
        # The six traces are only four source pixels apart. Their labels need
        # different rows and horizontal steps, rather than a pile at x=0.
        centers = [(box[0], (box[1] + box[3]) / 2) for box in boxes.values()]
        self.assertGreater(len({x for x, _y in centers}), 1)
        self.assertEqual(len({y for _x, y in centers}), len(boxes))
        self.assertEqual([y for _x, y in centers], sorted(y for _x, y in centers))
        self.assertEqual(self.widget.get_line_overlays(), self.original)
        np.testing.assert_array_equal(self.widget.get_image(), self.image)
        self.edited.assert_not_called()

    def test_zoom_scrollbars_and_drag_pan_keep_labels_attached_to_visible_traces(self):
        widget = self.widget
        for zoom in (1.2, 2.2):
            widget.set_zoom(zoom)
            widget._scroll_x("moveto", 0.38)
            widget._scroll_y("moveto", 0.3)
            self._assert_readable()
            self._assert_leaders_match_traces()
        before_left = widget.canvas.canvasx(0)
        widget._on_pan_start(SimpleNamespace(x=400, y=250))
        widget._on_pan_motion(SimpleNamespace(x=280, y=250))
        widget._on_pan_end(None)
        self.assertNotEqual(widget.canvas.canvasx(0), before_left)
        self._assert_readable()
        self._assert_leaders_match_traces()
        self.assertEqual(widget.get_line_overlays(), self.original)
        self.edited.assert_not_called()

    def test_zooming_in_enlarges_readable_labels_while_preserving_trace_associations(self):
        font_sizes = []
        label_heights = []
        for zoom in (0.5, 1, 2, 4):
            with self.subTest(zoom=zoom):
                self.widget.set_zoom(zoom)
                self.widget._scroll_x("moveto", 0.38)
                self.widget._scroll_y("moveto", 0.3)
                boxes = self._assert_readable()
                self._assert_leaders_match_traces()
                text = self._labels()["ILM"]
                font = tkfont.Font(
                    root=self.root,
                    font=self.widget.canvas.itemcget(text, "font"),
                )
                font_sizes.append(abs(font.actual("size")))
                label_heights.append(boxes["ILM"][3] - boxes["ILM"][1])
                self.assertEqual(self.widget.get_line_overlays(), self.original)
        self.assertTrue(
            all(before < after for before, after in zip(font_sizes, font_sizes[1:])),
            font_sizes,
        )
        self.assertTrue(
            all(before < after for before, after in zip(label_heights, label_heights[1:])),
            label_heights,
        )
        np.testing.assert_array_equal(self.widget.get_image(), self.image)
        self.edited.assert_not_called()

    def test_resize_repositions_labels_inside_the_current_view_without_changing_annotations(self):
        self.widget.set_zoom(1.5)
        self.widget._scroll_x("moveto", 0.4)
        self.widget._scroll_y("moveto", 0.22)
        for geometry in ("460x300", "1100x650", "700x440"):
            with self.subTest(geometry=geometry):
                self.root.geometry(geometry)
                self.root.update()
                self._settle_resize()
                self._assert_readable()
                self._assert_leaders_match_traces()
        self.assertEqual(self.widget.get_line_overlays(), self.original)
        np.testing.assert_array_equal(self.widget.get_image(), self.image)

    def test_editing_a_trace_refreshes_its_label_and_leader_without_changing_other_traces(self):
        widget = self.widget
        widget.enable_line()
        widget.set_editable_line("ILM")
        self._assert_readable()
        canvas = widget.canvas
        source = widget.get_line_overlays()[0]["points"][1]
        cx, cy = widget._i2c(*source)
        viewport_x, viewport_y = cx - canvas.canvasx(0), cy - canvas.canvasy(0)
        widget._on_press(SimpleNamespace(x=viewport_x, y=viewport_y, state=0))
        widget._on_drag_motion(SimpleNamespace(x=viewport_x, y=viewport_y - 25))
        widget._on_release(None)
        updated = widget.get_line_overlays()
        self.assertNotEqual(updated[0]["points"], self.original[0]["points"])
        self.assertEqual(updated[1:], self.original[1:])
        self.edited.assert_called_once_with("ILM", updated[0]["points"])
        self._assert_readable()
        self._assert_leaders_match_traces()
        widget.set_editable_line(None)
        widget.enable_line(False)
        self._assert_readable()

    def test_active_trace_joins_the_layout_without_overlapping_saved_layer_labels(self):
        widget = self.widget
        widget.enable_line()
        widget.set_active_line([(0, 195), (650, 199), (1400, 205), (2132, 209)])
        active = {
            "label": "active",
            "color": widget._line_color,
            "points": widget.get_active_line(),
        }
        self._assert_readable([*widget.get_line_overlays(), active])
        self._assert_leaders_match_traces([*widget.get_line_overlays(), active])
        self.assertEqual(widget.get_line_overlays(), self.original)
        widget.clear_active_line()
        self._assert_readable()
        self._assert_leaders_match_traces()

    def test_panning_beyond_a_short_trace_removes_its_label_then_restores_it(self):
        overlay = dict(label="Short trace", color="#44bb77", points=[(0, 190), (250, 194)])
        widget = self.widget
        widget.set_line_overlays([overlay])
        widget.set_zoom(1)
        widget._scroll_x("moveto", 0)
        self._assert_readable([overlay])
        widget._scroll_x("moveto", 0.6)
        self.assertFalse(self._labels())
        self.assertFalse(self._visible_items("annotation-label-link"))
        widget._scroll_x("moveto", 0)
        self._assert_readable([overlay])
        self._assert_leaders_match_traces([overlay])

    def test_vertical_annotation_and_single_point_keep_their_own_labels(self):
        overlays = [
            {
                "label": "Fovea-Center",
                "color": "#5699ff",
                "points": [(1066, 20), (1066, 380)],
            },
            {"label": "Point", "color": "#66cc88", "points": [(30, 190)]},
        ]
        self.widget.set_line_overlays(overlays)
        self._assert_readable(overlays)
        self._assert_leaders_match_traces(overlays)
        self.assertEqual(self.widget.get_line_overlays(), overlays)

    def test_clearing_annotations_or_replacing_image_leaves_no_stale_label_items(self):
        self._assert_readable()
        self.widget.clear_line_overlays()
        for tag in ("annotation-label", "annotation-label-text", "annotation-label-link"):
            self.assertFalse(self.widget.canvas.find_withtag(tag), tag)
        self.widget.set_line_overlays(self.overlays)
        self._assert_readable()
        self.widget.set_image(np.zeros((80, 100), dtype=np.uint8))
        for tag in ("annotation-label", "annotation-label-text", "annotation-label-link"):
            self.assertFalse(self.widget.canvas.find_withtag(tag), tag)


if __name__ == "__main__":
    unittest.main()
