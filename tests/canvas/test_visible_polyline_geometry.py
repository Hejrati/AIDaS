"""Clip only painted monotonic trace geometry; retain every scientific vertex."""

from copy import deepcopy
import tkinter as tk
import unittest
from unittest import mock

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas


class _Canvas:
    def __init__(self, left=1100.25, width=400):
        self.left = left
        self.width = width
        self.items = {}
        self.coordinate_updates = 0

    def canvasx(self, x):
        return self.left + x

    def winfo_width(self):
        return self.width

    def create_line(self, *coordinates, **options):
        item = len(self.items) + 1
        self.items[item] = {"coordinates": list(coordinates), **options}
        return item

    def coords(self, item, *coordinates):
        if coordinates:
            self.coordinate_updates += 1
            self.items[item]["coordinates"] = list(coordinates)
        return self.items[item]["coordinates"]

    def itemconfigure(self, item, **options):
        self.items[item].update(options)


class VisiblePolylineGeometryTests(unittest.TestCase):
    def _widget(self, points, *, left=1100.25, width=400, show_vertices=False):
        widget = ImageCanvas.__new__(ImageCanvas)
        widget.canvas = _Canvas(left, width)
        widget._data = np.arange(1200, dtype=np.uint16).reshape(30, 40)
        widget._img_id = 1
        widget._zoom = 2.375
        widget._img_offset_x, widget._img_offset_y = 13.25, -47.5
        widget._line_width = 2
        widget._dense_polyline_geometry = []
        widget._draw_polyline(points, "red", show_vertices=show_vertices)
        return widget

    def _assert_all_crossing_segments_retained(self, widget, record):
        full = record["coordinates"]
        left = widget.canvas.canvasx(0)
        right = left + widget.canvas.winfo_width()
        span = record["span"]
        intersecting = [index for index in range(len(full) - 1)
                        if max(full[index, 0], full[index + 1, 0]) >= left
                        and min(full[index, 0], full[index + 1, 0]) <= right]
        if intersecting:
            self.assertIsNotNone(span)
            for index in intersecting:
                self.assertLessEqual(span[0], index)
                self.assertGreater(span[1], index + 1)
        if span is not None:
            self.assertEqual(widget.canvas.coords(record["item_id"]),
                             full[span[0]:span[1]].ravel().tolist())

    def test_ascending_descending_and_repeated_x_traces_keep_every_intersecting_segment(self):
        ascending = [(x / 2.0, 100.0 + x % 11) for x in range(4000)]
        for points in (ascending, list(reversed(ascending)),
                       [(float(x // 3), 100.0 + x % 11) for x in range(4000)]):
            with self.subTest(descending=points[0][0] > points[-1][0]):
                original = list(points)
                widget = self._widget(points)
                record, = widget._dense_polyline_geometry
                self.assertLess(len(widget.canvas.coords(record["item_id"])), len(points) * 2)
                for left in (1100.25, 2700.5, -1000.0, 100000.0, 0.0):
                    widget.canvas.left = left
                    widget._refresh_visible_polyline_geometry()
                    self._assert_all_crossing_segments_retained(widget, record)
                self.assertEqual(points, original)

    def test_long_crossing_segment_is_retained_when_both_endpoints_are_outside_view(self):
        points = [(0.0, 100.0), (3000.0, 130.0),
                  *[(3000.0 + index, 130.0) for index in range(1, 127)]]
        widget = self._widget(points)
        record, = widget._dense_polyline_geometry
        self._assert_all_crossing_segments_retained(widget, record)
        coordinates = widget.canvas.coords(record["item_id"])
        self.assertLess(coordinates[0], widget.canvas.left)
        self.assertGreater(coordinates[-2], widget.canvas.left + widget.canvas.width)
        self.assertEqual(record["span"], (0, 2))

    def test_pan_reuses_line_item_reveals_old_geometry_and_skips_unchanged_spans(self):
        points = [(float(x), 100.0 + x % 7) for x in range(4000)]
        widget = self._widget(points, show_vertices=True)
        original_pixels = widget._data.copy()
        record, = widget._dense_polyline_geometry
        item_id = record["item_id"]
        handles = list(widget._line_handle_geometry)
        widget._refresh_visible_polyline_geometry()
        self.assertEqual(widget.canvas.coordinate_updates, 0)
        widget.canvas.left = 6500.0
        widget._refresh_visible_polyline_geometry()
        self.assertEqual(record["item_id"], item_id)
        self._assert_all_crossing_segments_retained(widget, record)
        self.assertEqual(widget.canvas.items[item_id]["state"], "normal")
        widget.canvas.left = 100000.0
        widget._refresh_visible_polyline_geometry()
        self.assertIsNone(record["span"])
        self.assertEqual(widget.canvas.items[item_id]["state"], "hidden")
        widget.canvas.left = 0.0
        widget._refresh_visible_polyline_geometry()
        self.assertEqual(widget.canvas.items[item_id]["state"], "normal")
        self._assert_all_crossing_segments_retained(widget, record)
        self.assertEqual(widget._line_handle_geometry, handles)
        for (cx, cy, _color), index in zip(handles, widget._line_handle_indices(points)):
            self.assertEqual((cx, cy), widget._i2c(*points[index]))
        np.testing.assert_array_equal(widget._data, original_pixels)

    def test_short_manual_and_nonmonotonic_traces_keep_full_original_painted_geometry(self):
        for points in (
            [(float(x), 100.0 + x % 7) for x in range(127)],
            [(float(x % 30), float(x % 71)) for x in range(1000)],
        ):
            with self.subTest(vertices=len(points)):
                original = deepcopy(points)
                widget = self._widget(points)
                self.assertEqual(widget._dense_polyline_geometry, [])
                self.assertEqual(widget.canvas.coords(1), widget._polyline_canvas_coordinates(points))
                widget._refresh_visible_polyline_geometry()
                self.assertEqual(widget.canvas.coordinate_updates, 0)
                self.assertEqual(points, original)


class VisiblePolylineCanvasTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk is unavailable: {error}")
        self.addCleanup(self.root.destroy)
        self.root.attributes("-alpha", 0)
        self.root.geometry("900x520")
        self.edited = mock.Mock()
        self.widget = ImageCanvas(self.root, on_line_edit=self.edited)
        self.widget.pack(fill="both", expand=True)
        self.root.update()
        self.image = np.arange(500 * 4500, dtype=np.uint16).reshape(500, 4500)
        self.widget.set_image(self.image)

    def test_pan_reveals_original_dense_vertices_with_stable_item_and_unchanged_source_data(self):
        points = [(float(x), 200.0 + x % 9) for x in range(4500)]
        self.widget.set_line_overlays([{"points": points, "color": "red"}])
        original = self.widget.get_line_overlays()
        self.widget.set_zoom(4.0)
        record, = self.widget._dense_polyline_geometry
        item_id = record["item_id"]
        spans = []
        for fraction in (0.0, 0.3, 0.7, 1.0):
            self.widget._scroll_x("moveto", fraction)
            self.root.update_idletasks()
            spans.append(record["span"])
            start, stop = record["span"]
            self.assertEqual(self.widget.canvas.coords(item_id),
                             record["coordinates"][start:stop].ravel().tolist())
            self.assertLess(stop - start, len(points) // 2)
        self.assertEqual(len(set(spans)), 4)
        self.assertEqual(spans[-1][1], len(points))
        self.assertEqual(self.widget.get_line_overlays(), original)
        np.testing.assert_array_equal(self.widget.get_image(), self.image)
        self.edited.assert_not_called()

    def test_crossing_endpoints_outside_view_and_nonmonotonic_manual_shape_remain_complete(self):
        points = [(0.0, 200.0), (3500.0, 200.0),
                  *[(3500.0 + index * 7.0, 200.0) for index in range(1, 127)]]
        self.widget.set_line_overlays([{"points": points, "color": "blue"}])
        self.widget.set_zoom(4.0)
        record, = self.widget._dense_polyline_geometry
        coordinates = self.widget.canvas.coords(record["item_id"])
        left = self.widget.canvas.canvasx(0)
        right = left + self.widget.canvas.winfo_width()
        self.assertLess(coordinates[0], left)
        self.assertGreater(coordinates[-2], right)
        self.assertEqual(record["span"], (0, 2))

        manual = [(100.0 + index % 11, 180.0 + index % 17) for index in range(200)]
        self.widget.set_line_overlays([{"points": manual, "color": "green"}])
        self.assertEqual(self.widget._dense_polyline_geometry, [])
        lines = [item for item in self.widget.canvas.find_withtag("overlay")
                 if self.widget.canvas.type(item) == "line"]
        self.assertEqual(len(lines), 1)
        self.assertEqual(self.widget.canvas.coords(lines[0]), self.widget._polyline_canvas_coordinates(manual))
        self.edited.assert_not_called()
        np.testing.assert_array_equal(self.widget.get_image(), self.image)


if __name__ == "__main__":
    unittest.main()
