from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest import mock

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas


class FoveaCanvasGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.addCleanup(self.root.destroy)
        self.root.geometry("800x500")
        self.changed = mock.Mock()
        self.widget = ImageCanvas(self.root, on_vertical_line_change=self.changed)
        self.widget.pack(fill="both", expand=True)
        self.root.update()
        self.widget.set_image(np.arange(100 * 600, dtype=np.uint16).reshape(100, 600))
        self.widget.enable_vertical_line()
        self.widget.set_vertical_line_x(250)
        self.changed.reset_mock()

    def _arrow_coordinates(self):
        arrows = self.widget.canvas.find_withtag("fovea-guidance")
        self.assertEqual(len(arrows), 1)
        return self.widget.canvas.coords(arrows[0])

    def test_label_moves_in_both_directions_without_moving_the_marker_or_data(self):
        widget = self.widget
        canvas = widget.canvas
        original_bounds = widget.fovea_label_bounds()
        marker = canvas.find_withtag("fovea-marker")
        marker_coordinates = canvas.coords(marker)
        image = widget.get_image().copy()

        # Fractional frames must not accumulate rounding drift before reset.
        for offset in (14, -14, 0.4, -0.7, 0.3, 0):
            widget.set_fovea_label_offset(offset)
            bounds = widget.fovea_label_bounds()
            self.assertAlmostEqual(bounds[0] - original_bounds[0], offset, delta=1)
            self.assertEqual(bounds[1::2], original_bounds[1::2])
            self.assertEqual(canvas.find_withtag("fovea-marker"), marker)
            self.assertEqual(canvas.coords(marker), marker_coordinates)
            self.assertEqual(widget.get_vertical_line_x(), 250)
        self.assertEqual(widget.fovea_label_bounds(), original_bounds)
        np.testing.assert_array_equal(widget.get_image(), image)
        self.changed.assert_not_called()

    def test_label_offset_survives_actual_marker_drag_and_zoom(self):
        widget = self.widget
        widget.set_fovea_label_offset(12)
        marker_x = widget.canvas.coords("fovea-marker")[0]
        viewport_x = marker_x - widget.canvas.canvasx(0)
        widget._on_press(SimpleNamespace(x=viewport_x, y=250))
        widget._on_drag_motion(SimpleNamespace(x=viewport_x + 25, y=250))
        widget._on_release(None)
        moved_x = widget.get_vertical_line_x()
        self.assertGreater(moved_x, 250)
        self.assertTrue(self.changed.called)

        for zoom in (widget.get_zoom(), widget.get_zoom() * 1.3):
            widget.set_zoom(zoom)
            moved_bounds = widget.fovea_label_bounds()
            self.assertEqual(widget._fovea_label_offset, 12)
            widget.set_fovea_label_offset(0)
            resting_bounds = widget.fovea_label_bounds()
            self.assertEqual(moved_bounds[0] - resting_bounds[0], 12)
            self.assertEqual(widget.get_vertical_line_x(), moved_x)
            widget.set_fovea_label_offset(12)

    def test_label_stays_readable_at_both_viewport_edges(self):
        widget = self.widget
        for marker_x, offset in ((0, -100), (599, 100)):
            widget.set_vertical_line_x(marker_x)
            widget.set_fovea_label_offset(offset)
            left, _top, right, _bottom = widget.fovea_label_bounds()
            self.assertGreaterEqual(left, widget.canvas.canvasx(0))
            self.assertLessEqual(right, widget.canvas.canvasx(widget.canvas.winfo_width()))
            self.assertEqual(len(widget.canvas.find_withtag("fovea-label")), 2)
            self.assertEqual(widget.get_vertical_line_x(), marker_x)

    def test_arrow_starts_at_marker_top_through_animation_drag_zoom_and_scroll(self):
        widget = self.widget
        canvas = widget.canvas
        widget.set_fovea_guidance_target(600)
        original_arrow = self._arrow_coordinates()
        marker_coordinates = canvas.coords("fovea-marker")
        self.assertEqual(original_arrow[:2], marker_coordinates[:2])
        for offset in (14, -14, 0):
            widget.set_fovea_label_offset(offset)
            animated_arrow = self._arrow_coordinates()
            self.assertEqual(animated_arrow[:2], marker_coordinates[:2])
            self.assertEqual(animated_arrow[-2:], original_arrow[-2:])
            self.assertEqual(canvas.coords("fovea-marker"), marker_coordinates)
        self.changed.assert_not_called()

        viewport_x = marker_coordinates[0] - canvas.canvasx(0)
        widget._on_press(SimpleNamespace(x=viewport_x, y=250))
        widget._on_drag_motion(SimpleNamespace(x=viewport_x + 25, y=250))
        widget._on_release(None)
        moved_x = widget.get_vertical_line_x()
        self.assertGreater(moved_x, 250)
        self.assertEqual(self._arrow_coordinates()[:2], canvas.coords("fovea-marker")[:2])
        self.assertNotEqual(self._arrow_coordinates()[0], original_arrow[0])
        self.assertTrue(self.changed.called)
        self.changed.reset_mock()

        widget.set_zoom(widget.get_zoom() * 2)
        arrow = self._arrow_coordinates()
        self.assertEqual(arrow[:2], canvas.coords("fovea-marker")[:2])
        self.assertAlmostEqual(arrow[-2], canvas.canvasx(0) + 600)
        self.assertAlmostEqual(arrow[-1], canvas.canvasy(0) + 6)
        widget._scroll_x("moveto", 0.15)
        arrow = self._arrow_coordinates()
        self.assertEqual(arrow[:2], canvas.coords("fovea-marker")[:2])
        self.assertAlmostEqual(arrow[-2], canvas.canvasx(0) + 600)

        widget.set_fovea_guidance_target(None)
        self.assertFalse(canvas.find_withtag("fovea-guidance"))
        widget.set_zoom(widget.get_zoom() / 2)
        self.assertFalse(canvas.find_withtag("fovea-guidance"))
        self.assertEqual(widget.get_vertical_line_x(), moved_x)
        self.changed.assert_not_called()

    def test_hidden_or_offscreen_marker_has_no_arrow_or_stale_bounds(self):
        widget = self.widget
        widget.set_fovea_guidance_target(600)
        self._arrow_coordinates()
        widget.enable_vertical_line(False)
        self.assertIsNone(widget.fovea_label_bounds())
        self.assertFalse(widget.canvas.find_withtag("fovea-guidance"))
        widget.enable_vertical_line()
        self._arrow_coordinates()
        widget.set_zoom(widget.get_zoom() * 3)
        # The label may remain visible at the edge after the marker scrolls
        # outside.  Guidance must follow visibility of the actual marker top.
        marker_x = widget.canvas.coords("fovea-marker")[0]
        draw_width = widget._draw_region[2]
        widget._scroll_x("moveto", (marker_x + 20) / draw_width)
        self.assertFalse(widget.canvas.find_withtag("fovea-guidance"))
        widget._scroll_x("moveto", 1)
        self.assertFalse(widget.canvas.find_withtag("fovea-guidance"))
        widget.set_image(None)
        self.assertIsNone(widget.fovea_label_bounds())
        self.assertFalse(widget.canvas.find_withtag("fovea-guidance"))

    def test_external_renderer_receives_exact_screen_origin_and_hidden_updates(self):
        widget = self.widget
        canvas = widget.canvas
        renderer = mock.Mock()
        image = widget.get_image().copy()

        def assert_origin():
            marker_x, marker_y = canvas.coords("fovea-marker")[:2]
            expected = (
                canvas.winfo_rootx() + marker_x - canvas.canvasx(0),
                canvas.winfo_rooty() + marker_y - canvas.canvasy(0),
            )
            self.assertEqual(renderer.call_args, mock.call(expected))
            self.assertFalse(canvas.find_withtag("fovea-guidance"))

        widget.set_fovea_guidance_target(600, renderer=renderer)
        assert_origin()
        widget.set_fovea_label_offset(14)
        assert_origin()
        self.assertEqual(widget.get_vertical_line_x(), 250)
        self.changed.assert_not_called()

        previous_calls = renderer.call_count
        viewport_x = canvas.coords("fovea-marker")[0] - canvas.canvasx(0)
        widget._on_press(SimpleNamespace(x=viewport_x, y=250))
        widget._on_drag_motion(SimpleNamespace(x=viewport_x + 25, y=250))
        widget._on_release(None)
        self.assertGreater(renderer.call_count, previous_calls)
        moved_x = widget.get_vertical_line_x()
        self.assertGreater(moved_x, 250)
        assert_origin()
        self.changed.reset_mock()

        previous_calls = renderer.call_count
        widget.set_zoom(widget.get_zoom() * 2)
        self.assertGreater(renderer.call_count, previous_calls)
        assert_origin()
        previous_calls = renderer.call_count
        widget._scroll_x("moveto", 0.15)
        self.assertGreater(renderer.call_count, previous_calls)
        assert_origin()
        widget._scroll_x("moveto", 1)
        self.assertEqual(renderer.call_args, mock.call(None))
        widget._scroll_x("moveto", 0.15)
        assert_origin()
        widget.enable_vertical_line(False)
        self.assertEqual(renderer.call_args, mock.call(None))
        self.assertEqual(widget.get_vertical_line_x(), moved_x)
        widget.enable_vertical_line()
        assert_origin()
        self.changed.assert_not_called()

        widget.clear_vertical_line()
        self.assertEqual(renderer.call_args, mock.call(None))
        widget.set_vertical_line_x(moved_x)
        assert_origin()
        replacement = mock.Mock()
        widget.set_fovea_guidance_target(600, renderer=replacement)
        self.assertEqual(renderer.call_args, mock.call(None))
        self.assertIsNotNone(replacement.call_args.args[0])
        widget.set_fovea_guidance_target(None)
        self.assertEqual(replacement.call_args, mock.call(None))
        replacement.reset_mock()
        self.changed.reset_mock()
        widget.set_zoom(widget.get_zoom() / 2)
        replacement.assert_not_called()
        self.changed.assert_not_called()
        self.assertEqual(widget.get_vertical_line_x(), moved_x)
        np.testing.assert_array_equal(widget.get_image(), image)


if __name__ == "__main__":
    unittest.main()
