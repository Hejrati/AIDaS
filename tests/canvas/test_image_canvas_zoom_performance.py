"""Zoom frame work stays bounded while image-space annotations stay exact."""

from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageTk

from aidas.canvas.image_canvas import ImageCanvas, RESAMPLE_NEAREST


class ImageCanvasZoomPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.addCleanup(self.root.destroy)
        self.root.attributes("-alpha", 0)
        self.root.geometry("720x440")
        self.zoom_changed = mock.Mock()
        self.roi_changed = mock.Mock()
        self.line_changed = mock.Mock()
        self.line_edited = mock.Mock()
        self.fovea_changed = mock.Mock()
        self.widget = ImageCanvas(
            self.root,
            on_zoom_change=self.zoom_changed,
            on_roi_change=self.roi_changed,
            on_line_change=self.line_changed,
            on_line_edit=self.line_edited,
            on_vertical_line_change=self.fovea_changed,
        )
        self.widget.pack(fill="both", expand=True)
        self.root.update()
        self.image = np.arange(240 * 4000, dtype=np.uint16).reshape(240, 4000)
        self.widget.set_image(self.image)
        self.root.update()
        self.zoom_changed.reset_mock()

    def _center_image_point(self):
        canvas = self.widget.canvas
        return self.widget._c2i(
            canvas.canvasx(canvas.winfo_width() / 2),
            canvas.canvasy(canvas.winfo_height() / 2),
        )

    def _flush_queued_zoom(self):
        # Run the queued frame deterministically without letting its timer fire
        # twice, or relying on machine speed to reach the 16 ms deadline.
        job = self.widget._zoom_redraw_after_id
        self.assertIsNotNone(job)
        self.widget.after_cancel(job)
        self.widget._apply_pending_zoom()

    def _assert_cancelled(self, job):
        self.assertIsNotNone(job)
        with self.assertRaises(tk.TclError):
            self.root.tk.call("after", "info", job)

    def test_wheel_burst_paints_one_frame_and_notifies_after_restoring_center(self):
        widget = self.widget
        widget.set_zoom(2)
        widget._scroll_x("moveto", 0.25)
        widget._scroll_y("moveto", 0.05)
        focus = self._center_image_point()
        snapshots = []
        widget._cb_zoom = lambda zoom: snapshots.append((
            zoom, self._center_image_point(), widget.canvas.coords(widget._img_id),
        ))
        with mock.patch.object(widget, "after", wraps=widget.after) as schedule, mock.patch(
            "aidas.canvas.image_canvas.ImageTk.PhotoImage", wraps=ImageTk.PhotoImage,
        ) as photo:
            for _ in range(4):
                widget._on_wheel(SimpleNamespace(num=None, delta=30))
            self.assertEqual(schedule.call_count, 1)
            self.assertEqual(schedule.call_args.args[0], widget.ZOOM_REDRAW_DEBOUNCE_MS)
            self.assertAlmostEqual(widget._pending_zoom, 2 * widget.ZOOM_STEP_FACTOR)
            self.assertEqual(widget.get_zoom(), 2)
            photo.assert_not_called()
            self.assertEqual(snapshots, [])

            self._flush_queued_zoom()

            self.assertEqual(photo.call_count, 1)
        self.assertEqual(len(snapshots), 1)
        self.assertAlmostEqual(snapshots[0][0], 2.5)
        for expected, actual in zip(focus, snapshots[0][1]):
            # Tk scroll origins use whole canvas pixels.
            self.assertAlmostEqual(expected, actual, delta=1 / widget.get_zoom())
        self.assertEqual(snapshots[0][2], widget.canvas.coords(widget._img_id))
        self.assertIsNone(widget._pending_zoom)
        self.assertIsNone(widget._pending_zoom_focus)
        self.assertIsNone(widget._zoom_redraw_after_id)

    def test_public_set_zoom_remains_synchronous_and_paints_one_tile(self):
        with mock.patch(
            "aidas.canvas.image_canvas.ImageTk.PhotoImage", wraps=ImageTk.PhotoImage,
        ) as photo:
            self.widget.set_zoom(2.75)
            self.assertEqual(photo.call_count, 1)
        self.assertEqual(self.widget.get_zoom(), 2.75)
        self.zoom_changed.assert_called_once_with(2.75)
        self.assertIsNone(self.widget._zoom_redraw_after_id)
        self.assertIsNotNone(self.widget._photo)

    def test_queued_zoom_obeys_limits_and_explicit_zoom_or_fit_cancels_it(self):
        widget = self.widget
        for requested, expected in ((1000, 30.0), (-1, 0.02)):
            with self.subTest(requested=requested):
                widget._queue_zoom(requested)
                self._flush_queued_zoom()
                self.assertEqual(widget.get_zoom(), expected)
        for action in (lambda: widget.set_zoom(1.5), widget.fit_to_window):
            with self.subTest(action=action):
                widget._queue_zoom(7)
                job = widget._zoom_redraw_after_id
                action()
                self._assert_cancelled(job)
                zoom = widget.get_zoom()
                widget._apply_pending_zoom()
                self.assertEqual(widget.get_zoom(), zoom)
                self.assertIsNone(widget._pending_zoom)
                self.assertIsNone(widget._pending_zoom_focus)

    def test_visible_tile_pixels_match_the_existing_nearest_neighbor_crop(self):
        widget = self.widget
        for zoom in (0.37, 1.3, 2.75, 30):
            with self.subTest(zoom=zoom):
                widget.set_zoom(zoom)
                widget._scroll_x("moveto", 0.31)
                widget._scroll_y("moveto", 0.12)
                canvas = widget.canvas
                view_x, view_y = canvas.canvasx(0), canvas.canvasy(0)
                width, height = canvas.winfo_width(), canvas.winfo_height()
                left, top = widget._img_offset_x, widget._img_offset_y
                image_height, image_width = widget.get_image().shape
                x0 = max(0, int(np.floor((max(view_x, left) - left) / zoom)))
                y0 = max(0, int(np.floor((max(view_y, top) - top) / zoom)))
                x1 = min(image_width, int(np.ceil((min(view_x + width, left + image_width * zoom) - left) / zoom)))
                y1 = min(image_height, int(np.ceil((min(view_y + height, top + image_height * zoom) - top) / zoom)))
                tile_size = (max(1, round((x1 - x0) * zoom)), max(1, round((y1 - y0) * zoom)))
                with Image.fromarray(widget._display_data) as full_image:
                    expected = full_image.crop((x0, y0, x1, y1)).resize(tile_size, RESAMPLE_NEAREST)
                actual = ImageTk.getimage(widget._photo).convert("L")
                try:
                    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
                    self.assertEqual(widget.canvas.coords(widget._img_id), [left + x0 * zoom, top + y0 * zoom])
                    self.assertLessEqual(actual.width, width + 2 * zoom + 1)
                    self.assertLessEqual(actual.height, height + 2 * zoom + 1)
                finally:
                    actual.close()
                    expected.close()

    def test_zoom_and_pan_preserve_original_pixels_roi_boundaries_and_fovea(self):
        widget = self.widget
        widget.enable_roi()
        widget.set_roi((25, 20, 1200, 150))
        overlays = [
            dict(label=f"Layer {index}", color="#ffd500", points=[
                (x, 50 + index * 15 + x % 7) for x in range(4000)
            ])
            for index in range(4)
        ]
        widget.set_line_overlays(overlays)
        widget.set_editable_line("Layer 0")
        widget.enable_line()
        widget.set_active_line([(50, 25), (70, 35), (90, 25)])
        widget.enable_vertical_line()
        widget.set_vertical_line_x(1200)
        original_image = widget.get_image().copy()
        original_overlays = widget.get_line_overlays()
        original_active = widget.get_active_line()
        original_roi = widget.get_roi()
        for callback in (self.roi_changed, self.line_changed, self.line_edited, self.fovea_changed):
            callback.reset_mock()

        for zoom in (2.0, 0.37, 4.0):
            widget.set_zoom(zoom)
            widget._scroll_x("moveto", 0.25)
            widget._scroll_y("moveto", 0.1)
            widget._on_pan_start(SimpleNamespace(x=400, y=220))
            widget._on_pan_motion(SimpleNamespace(x=340, y=200))
            widget._on_pan_end(None)
            np.testing.assert_array_equal(widget.get_image(), original_image)
            np.testing.assert_array_equal(self.image, original_image)
            self.assertEqual(widget.get_line_overlays(), original_overlays)
            self.assertEqual(widget.get_active_line(), original_active)
            self.assertEqual(widget.get_roi(), original_roi)
            self.assertEqual(widget.get_vertical_line_x(), 1200)
            for ix, iy in ((0, 0), (1200, 110), (3999, 239)):
                actual = widget._c2i(*widget._i2c(ix, iy))
                self.assertAlmostEqual(actual[0], ix)
                self.assertAlmostEqual(actual[1], iy)
        for callback in (self.roi_changed, self.line_changed, self.line_edited, self.fovea_changed):
            callback.assert_not_called()

    def test_replacing_image_cancels_zoom_from_the_previous_image(self):
        widget = self.widget
        widget._queue_zoom(8)
        job = widget._zoom_redraw_after_id
        replacement = np.arange(40 * 80, dtype=np.uint16).reshape(40, 80)

        widget.set_image(replacement)

        self._assert_cancelled(job)
        expected_zoom = widget.get_zoom()
        widget._apply_pending_zoom()
        self.assertEqual(widget.get_zoom(), expected_zoom)
        np.testing.assert_array_equal(widget.get_image(), replacement)
        self.assertIsNone(widget._pending_zoom)

    def test_models_changed_while_image_is_offscreen_rebuild_latest_overlays_on_return(self):
        widget = self.widget
        canvas = widget.canvas
        widget.set_zoom(1)
        widget._scroll_x("moveto", 0)
        widget.set_line_overlays([dict(
            label="Old boundary", color="#cc0000",
            points=[(x, 40 + x % 3) for x in range(4000)],
        )])
        widget.set_editable_line("Old boundary")
        widget.enable_line()
        widget.set_active_line([(50, 90), (150, 100), (250, 90)])

        def move_to_blank_area():
            # A large scroll region exposes a viewport containing no source
            # pixels, independently of the image's normal scroll limits.
            canvas.configure(scrollregion=(0, 0, widget._draw_region[2], 4000))
            widget._scroll_y("moveto", 0.8)
            self.assertIsNone(widget._img_id)

        def return_to_image():
            with mock.patch.object(widget, "_redraw_overlays", wraps=widget._redraw_overlays) as redraw:
                widget._scroll_y("moveto", 0)
            redraw.assert_called_once_with()
            self.assertIsNotNone(widget._img_id)

        move_to_blank_area()
        updated_points = [(x, 140 + x % 5) for x in range(4000)]
        updated_active = [(50, 190), (150, 195), (250, 190)]
        widget.clear_line_overlays()
        widget.set_line_overlays([dict(
            label="New boundary", color="#00aa55", points=updated_points,
        )])
        widget.set_editable_line("New boundary")
        widget.clear_active_line()
        widget.set_active_line(updated_active)
        self.assertIsNone(widget._img_id)

        return_to_image()

        self.assertEqual(widget.get_line_overlays(), [dict(
            label="New boundary", color="#00aa55", points=updated_points,
        )])
        self.assertEqual(widget.get_active_line(), updated_active)
        texts = {canvas.itemcget(item, "text") for item in canvas.find_withtag("annotation-label-text")}
        self.assertEqual(texts, {"New boundary", "active"})
        boundary_items = [
            item for item in canvas.find_withtag("overlay")
            if canvas.type(item) == "line"
            and canvas.itemcget(item, "fill") == "#00aa55"
            and "annotation-label-link" not in canvas.gettags(item)
        ]
        self.assertEqual(len(boundary_items), 1)
        coordinates = canvas.coords(boundary_items[0])
        self.assertGreaterEqual(len(coordinates), 4)
        for cx, cy in zip(coordinates[::2], coordinates[1::2]):
            ix, iy = widget._c2i(cx, cy)
            self.assertEqual((round(ix), round(iy)), updated_points[round(ix)])
        visible_handles = [
            item for item in canvas.find_withtag("line-handle")
            if canvas.itemcget(item, "state") != "hidden"
        ]
        self.assertTrue(visible_handles)
        expected_handles = set(updated_points + updated_active)
        for item in visible_handles:
            x0, y0, x1, y1 = canvas.coords(item)
            ix, iy = widget._c2i((x0 + x1) / 2, (y0 + y1) / 2)
            self.assertIn((round(ix), round(iy)), expected_handles)

        # Clearing while offscreen must also replace old projected snapshots;
        # returning cannot resurrect either the boundary or the active trace.
        move_to_blank_area()
        widget.clear_line_overlays()
        widget.clear_active_line()
        widget.set_editable_line(None)

        return_to_image()

        self.assertEqual(widget.get_line_overlays(), [])
        self.assertEqual(widget.get_active_line(), [])
        self.assertFalse(canvas.find_withtag("overlay"))
        np.testing.assert_array_equal(widget.get_image(), self.image)

    def test_destroy_cancels_pending_zoom_and_resize_callbacks(self):
        widget = self.widget
        widget._queue_zoom(3)
        zoom_job = widget._zoom_redraw_after_id
        widget._schedule_resize_redraw()
        resize_job = widget._resize_redraw_after_id

        widget.destroy()

        self._assert_cancelled(zoom_job)
        self._assert_cancelled(resize_job)


if __name__ == "__main__":
    unittest.main()
