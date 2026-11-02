"""Headless checks for reusable anchor geometry and exact overlay projection."""

from copy import deepcopy
import unittest
from unittest import mock

import numpy as np

from aidas.canvas.annotation_labels import PolylineAnchorIndex, visible_polyline_anchor
from aidas.canvas.image_canvas import ImageCanvas


class AnnotationAnchorCacheTests(unittest.TestCase):
    def _canvas(self):
        widget = ImageCanvas.__new__(ImageCanvas)
        widget.canvas = mock.Mock()
        widget._data = np.zeros((500, 700), dtype=np.uint8)
        widget._img_id = 1
        widget._roi_items = []
        widget._annotation_anchor_cache = {}
        widget._line_color = "yellow"
        widget._active_line = [(5.0, 6.0), (10.0, 8.0)]
        widget._line_overlays = [
            {"label": "Dense AI", "color": "red",
             "points": [(float(x), 100.0 + x / 17.0) for x in range(600)]},
            {"label": "Manual", "color": "blue",
             "points": [(0.0, 20.0), (30.0, 15.0), (12.0, 40.0)]},
        ]
        for name in ("_clear_vertical_line_overlay", "_draw_roi", "_draw_line_overlays",
                     "_draw_active_line", "_draw_vertical_line", "_update_annotation_labels",
                     "_hide_annotation_labels", "_refresh_visible_line_handles"):
            setattr(widget, name, mock.Mock())
        return widget

    def test_zoom_redraw_reuses_indexes_but_in_place_geometry_edits_rebuild_only_changed_trace(self):
        widget = self._canvas()
        original = deepcopy(widget._line_overlays)
        with mock.patch("aidas.canvas.image_canvas.PolylineAnchorIndex", wraps=PolylineAnchorIndex) as constructor:
            widget._redraw_overlays()
            initial = {key: geometry for key, _, _, geometry in widget._annotation_label_sources}
            self.assertEqual(constructor.call_count, 3)
            for zoom in (0.7, 1.2, 3.4, 10.0):
                widget._zoom = zoom
                widget._redraw_overlays()
                self.assertEqual(constructor.call_count, 3)
                for key, _, _, geometry in widget._annotation_label_sources:
                    self.assertIs(geometry, initial[key])
            self.assertEqual(widget._line_overlays, original)

            points = widget._line_overlays[0]["points"]
            points[100] = (100.0, 240.0)
            widget._redraw_overlays()
            changed = {key: geometry for key, _, _, geometry in widget._annotation_label_sources}
            self.assertEqual(constructor.call_count, 4)
            self.assertIsNot(changed["saved-0"], initial["saved-0"])
            self.assertIs(changed["saved-1"], initial["saved-1"])
            self.assertIs(changed["active"], initial["active"])
            for viewport in ((95, 90, 110, 260), (0, 0, 650, 400)):
                self.assertEqual(changed["saved-0"].anchor(viewport),
                                 visible_polyline_anchor(points, viewport))

            widget._active_line.append((14.0, 15.0))
            widget._line_overlays[1]["points"].reverse()
            widget._redraw_overlays()
            self.assertEqual(constructor.call_count, 6)

    def test_removed_labels_and_replaced_images_do_not_retain_cached_geometries(self):
        widget = self._canvas()
        widget._redraw_overlays()
        self.assertEqual(set(widget._annotation_anchor_cache), {"saved-0", "saved-1", "active"})
        widget._active_line.clear()
        widget._line_overlays.pop()
        widget._redraw_overlays()
        self.assertEqual(set(widget._annotation_anchor_cache), {"saved-0"})
        for name in ("_cancel_pending_zoom", "_redraw", "_refresh_cursor_for_mode"):
            setattr(widget, name, mock.Mock())
        widget.set_image(None)
        self.assertEqual(widget._annotation_anchor_cache, {})
        self.assertEqual(widget._line_overlays, [])


class PolylineCanvasProjectionTests(unittest.TestCase):
    def test_projection_matches_every_original_vertex_exactly_without_changing_source_pixels(self):
        widget = ImageCanvas.__new__(ImageCanvas)
        random = np.random.default_rng(728)
        image = random.integers(0, 65535, size=(80, 130), dtype=np.uint16)
        widget._data = image
        original_image = image.copy()
        for count in (1, 2, 16, 127, 128, 2133):
            points = [tuple(map(float, point)) for point in random.uniform(-20, 800, size=(count, 2))]
            original_points = list(points)
            for zoom, offsets in ((0.02, (0.0, 0.0)), (0.7, (123.5, -37.25)),
                                  (1.23456789, (-17.1, 45.75)), (30.0, (999.125, -8.375))):
                with self.subTest(vertices=count, zoom=zoom, offsets=offsets):
                    widget._zoom = zoom
                    widget._img_offset_x, widget._img_offset_y = offsets
                    expected = [coordinate for point in points for coordinate in widget._i2c(*point)]
                    actual = widget._polyline_canvas_coordinates(points)
                    self.assertEqual(actual, expected)
                    self.assertEqual(len(actual), 2 * count)
                    self.assertEqual(points, original_points)
                    np.testing.assert_array_equal(widget._data, original_image)


if __name__ == "__main__":
    unittest.main()
