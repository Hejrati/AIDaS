"""Viewport handle rendering retains the full annotation editing geometry."""

import unittest

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas


class _Canvas:
    def __init__(self):
        self.left = self.top = 0
        self.items = {}
        self.created = 0
        self.lowered = []
        self.configured = 0

    def winfo_width(self):
        return 500

    def winfo_height(self):
        return 200

    def canvasx(self, x):
        return self.left + x

    def canvasy(self, y):
        return self.top + y

    def create_rectangle(self, *coords, **options):
        self.created += 1
        self.items[self.created] = {"coords": coords, **options}
        return self.created

    def coords(self, item, *coords):
        self.items[item]["coords"] = coords

    def itemconfigure(self, item, **options):
        self.configured += 1
        self.items[item].update(options)

    def find_withtag(self, tag):
        return tuple(item for item, value in self.items.items() if tag in value.get("tags", ()))

    def find_all(self):
        return tuple(self.items)

    def tag_lower(self, *args):
        self.lowered.append(args)


class VisibleLineHandleTests(unittest.TestCase):
    def setUp(self):
        self.widget = ImageCanvas.__new__(ImageCanvas)
        self.widget.canvas = _Canvas()
        self.widget._data = np.zeros((200, 200000), dtype=np.uint8)
        self.widget._img_id = 1
        self.widget._zoom = 1
        self.widget._img_offset_x = self.widget._img_offset_y = 0
        self.widget._line_handle_geometry = [
            (index * 20.0, 100.0, "yellow" if index == 0 else "white")
            for index in range(10000)
        ]
        self.widget._line_handle_positions = np.asarray(
            [point[:2] for point in self.widget._line_handle_geometry], dtype=float,
        )

    def _visible(self):
        return [item for item in self.widget.canvas.items.values() if item.get("state") == "normal"]

    def test_only_visible_handles_are_created_without_changing_global_geometry(self):
        original = list(self.widget._line_handle_geometry)
        points = [point[:2] for point in original]
        global_handles = self.widget._line_handle_indices(points)

        self.widget._refresh_visible_line_handles()

        self.assertEqual(len(self._visible()), 26)
        self.assertEqual(self.widget.canvas.created, 26)
        self.assertEqual(self._visible()[0]["fill"], "yellow")
        self.assertEqual(self.widget._line_handle_geometry, original)
        self.assertEqual(self.widget._line_handle_indices(points), global_handles)
        self.assertEqual(len(global_handles), len(points))

    def test_pan_reuses_pool_for_newly_visible_handles_and_keeps_edit_hit_testing(self):
        self.widget._refresh_visible_line_handles()
        item_ids = tuple(self.widget._line_handle_items)
        self.widget.canvas.left = 20000

        self.widget._refresh_visible_line_handles()

        self.assertEqual(tuple(self.widget._line_handle_items), item_ids)
        self.assertEqual(self.widget.canvas.created, 26)
        self.assertEqual(self._visible()[0]["coords"], (19997.0, 97.0, 20003.0, 103.0))
        points = [point[:2] for point in self.widget._line_handle_geometry]
        self.assertEqual(self.widget._hit_line_vertex(points, 20000, 100), 1000)

    def test_image_outside_view_hides_handles_and_returning_restores_them(self):
        self.widget._refresh_visible_line_handles()
        self.widget._img_id = None
        self.widget._refresh_visible_line_handles()
        self.assertEqual(self._visible(), [])

        self.widget._img_id = 1
        self.widget._refresh_visible_line_handles()
        self.assertEqual(len(self._visible()), 26)

    def test_unchanged_view_does_not_reconfigure_handles(self):
        self.widget._refresh_visible_line_handles()
        configured = self.widget.canvas.configured

        self.widget._refresh_visible_line_handles()

        self.assertEqual(self.widget.canvas.configured, configured)

    def test_new_handles_remain_below_marker_and_annotation_labels(self):
        marker = self.widget.canvas.create_rectangle(0, 0, 1, 1, tags=("vertical_overlay",))
        self.widget.canvas.create_rectangle(0, 0, 1, 1, tags=("annotation-label",))

        self.widget._refresh_visible_line_handles()

        self.assertEqual(self.widget.canvas.lowered, [("line-handle", marker)])

    def test_missing_geometry_fields_are_safe_for_an_empty_canvas(self):
        widget = ImageCanvas.__new__(ImageCanvas)
        widget.canvas = _Canvas()
        widget._data = None
        widget._img_id = None

        widget._refresh_visible_line_handles()

        self.assertEqual(widget._line_handle_items, [])


if __name__ == "__main__":
    unittest.main()
