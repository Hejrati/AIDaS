import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas


class CanvasTabPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.geometry('800x400')
        self.widget = ImageCanvas(self.root)
        self.widget.pack(fill='both', expand=True)
        self.root.update()
        self.widget.set_image(np.arange(200 * 1500, dtype=np.uint16).reshape(200, 1500))
        self.root.update()

    def tearDown(self):
        self.root.destroy()

    def test_restoring_dense_lines_keeps_items_and_accepts_edits(self):
        points = [(x, 50 + x % 7) for x in range(1500)]
        overlays = [dict(points=points, color='red', label='Boundary')]
        self.widget.set_line_overlays(overlays)
        items = self.widget.canvas.find_all()
        with patch.object(self.widget, '_clean_line_points', wraps=self.widget._clean_line_points) as clean:
            self.widget.set_line_overlays(overlays)
            clean.assert_not_called()
        self.assertEqual(items, self.widget.canvas.find_all())
        points[100] = (100, 80)
        self.widget.set_line_overlays(overlays)
        self.assertEqual(self.widget._line_overlays[0]['points'][100], (100, 80))
        overlays[0]['handle_indices'] = [0, 100, 1499]
        self.widget.set_line_overlays(overlays)
        self.assertEqual(self.widget.get_line_handle_indices('Boundary'), [0, 100, 1499])

    def test_fit_reuses_image_but_updates_zoom_and_resize(self):
        self.widget.fit_to_window()
        image_id = self.widget._img_id
        with patch.object(self.widget, '_redraw', wraps=self.widget._redraw) as redraw:
            self.widget.fit_to_window()
            redraw.assert_not_called()
            self.assertEqual(self.widget._img_id, image_id)
            self.widget._zoom *= 2
            self.widget._redraw()
            redraw.reset_mock()
            self.widget.fit_to_window()
            redraw.assert_called_once()
        self.root.geometry('950x500')
        self.root.update()
        self.widget.fit_to_window()
        self.assertEqual(self.widget._base_size[2:],
                         (self.widget.canvas.winfo_width(), self.widget.canvas.winfo_height()))

    def test_empty_active_line_does_not_redraw_saved_lines(self):
        with patch.object(self.widget, '_redraw_overlays') as redraw:
            self.widget.clear_active_line()
            redraw.assert_not_called()
            self.widget.set_active_line([(20, 30), (50, 60)])
            redraw.reset_mock()
            self.widget.clear_active_line()
            redraw.assert_called_once()

    def test_high_zoom_draws_visible_handles_and_pan_reuses_their_items(self):
        points = [(float(x), 100.0) for x in range(1500)]
        self.widget.set_line_overlays([dict(points=points, color="yellow", label="Boundary")])
        self.widget.set_editable_line("Boundary")
        self.widget.set_zoom(10)
        self.root.update()
        original = self.widget.get_line_overlays()
        handles = self.widget._line_handle_indices(self.widget._line_overlays[0]["points"])

        def visible_handles():
            return [item for item in self.widget.canvas.find_withtag("line-handle")
                    if self.widget.canvas.itemcget(item, "state") != "hidden"]

        visible = visible_handles()
        self.assertTrue(visible)
        self.assertLess(len(visible), len(handles) // 4)
        first_bounds = self.widget.canvas.coords(visible[0])
        self.widget.canvas.xview_moveto(.7)
        self.widget._render_visible_image()
        self.root.update()

        panned = visible_handles()
        self.assertTrue(panned)
        self.assertNotEqual(self.widget.canvas.coords(panned[0]), first_bounds)
        self.assertEqual(self.widget.get_line_overlays(), original)
        self.assertEqual(self.widget._line_handle_indices(self.widget._line_overlays[0]["points"]), handles)
        for item in panned:
            x1, y1, x2, y2 = self.widget.canvas.coords(item)
            self.assertIsNotNone(self.widget._hit_line_vertex(points, (x1 + x2) / 2, (y1 + y2) / 2))


if __name__ == '__main__':
    unittest.main()
