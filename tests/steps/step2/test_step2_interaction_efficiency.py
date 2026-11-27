"""Retain annotation state while avoiding redundant work during navigation."""

from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest import mock

import customtkinter as ctk
import numpy as np

from aidas.steps.step2_annotate import (
    BOUNDARY_NAMES, FOVEA_BOUNDARY_NAME, Step2Frame, _polyline_pixels,
)
from aidas.utils.io_utils import write_analyze
from tests.support.annotation import make_annotation_frame


class Step2InteractionEfficiencyTests(unittest.TestCase):
    def test_vertical_marker_pixels_match_the_original_raster(self):
        for x in (-10, 0, 99.8):
            for height in (-1, 0, 1, 177, 1200):
                with self.subTest(x=x, height=height):
                    trace = Step2Frame._vertical_line_trace(x, height)
                    self.assertEqual(trace["pixels"], _polyline_pixels(trace["points"]))

    def test_fovea_drag_updates_trace_and_status_without_rescanning_batch_actions(self):
        frame = make_annotation_frame()
        frame.image_canvas.get_zoom.return_value = 1
        frame._last_status_mouse_sample = None
        frame.revert_line_btn = mock.Mock()
        before = frame._copy_trace_dict(frame.boundary_traces)

        frame._on_vertical_line_changed(7)

        self.assertEqual(frame.fovea_x_entry_var.get(), "7")
        self.assertEqual(frame.boundary_traces[FOVEA_BOUNDARY_NAME], frame._vertical_line_trace(7, 8))
        self.assertIn("Fovea x: 7", frame.status_var.get())
        self.assertNotEqual(frame.boundary_traces, before)
        frame._update_boundary_action_buttons.assert_not_called()

    def test_active_tab_guard_still_loads_a_replaced_canvas(self):
        frame = make_annotation_frame()
        canvas = mock.Mock()
        state = {"canvas": canvas, "input": "replacement.img"}
        frame._active_batch_result_tab = "tab"
        frame._batch_result_states = {"tab": state}
        frame._batch_result_tab_canvases = {}
        frame._load_editor_state = mock.Mock()

        frame._activate_batch_result_tab("tab")

        frame._sync_active_batch_result_state.assert_called_once()
        self.assertIs(frame._load_editor_state.call_args.args[0], state)
        self.assertIs(frame._load_editor_state.call_args.args[1], canvas)


class Step2TabInteractionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = ctk.CTk()
            self.addCleanup(self.root.destroy)
            self.root.attributes("-alpha", 0)
            self.root.geometry("1024x680+0+0")
        except tk.TclError as exc:
            self.skipTest(f"A transparent Tk test window is unavailable: {exc}")
        with mock.patch.object(Step2Frame, "_start_ai_device_probe"):
            self.frame = Step2Frame(self.root)
        self.frame.pack(fill="both", expand=True)
        self.root.update()
        self.image = np.arange(64 * 240, dtype=np.uint16).reshape(64, 240)
        rows = np.repeat(np.arange(6)[:, None] * 7 + 5, 240, axis=1)
        self.traces, _ = Step2Frame._boundary_traces_from_rows(rows, self.image.shape)
        self.frame._open_aidas_batch_results_viewer([
            {"input": f"image-{index}.img", "image": self.image, "traces": self.traces, "fovea_x": 120}
            for index in range(3)
        ])
        self.root.update()

    def test_only_visited_tabs_render_and_revisits_keep_the_fitted_canvas_and_overlays(self):
        frame = self.frame
        notebook = frame.batch_results_notebook
        first, second, third = notebook.tabs()
        self.assertEqual(set(frame._batch_result_tab_canvases), {first})
        notebook.select(second)
        self.root.update()
        self.assertEqual(set(frame._batch_result_tab_canvases), {first, second})
        notebook.select(third)
        self.root.update()

        for tab in notebook.tabs():
            with self.subTest(tab=tab):
                canvas = frame._batch_result_tab_canvases[tab]
                with (
                    mock.patch.object(canvas, "set_line_overlays", wraps=canvas.set_line_overlays) as overlays,
                    mock.patch.object(canvas, "_redraw", wraps=canvas._redraw) as redraw,
                    mock.patch.object(canvas, "update_idletasks", wraps=canvas.update_idletasks) as settle,
                ):
                    notebook.select(tab)
                    self.root.update()
                overlays.assert_not_called()
                redraw.assert_not_called()
                settle.assert_not_called()
                self.assertEqual([item["label"] for item in canvas._line_overlays], BOUNDARY_NAMES)
                self.assertEqual(canvas.get_vertical_line_x(), 120)
                self.assertEqual(len(canvas.canvas.find_withtag("fovea-marker")), 1)
                self.assertIn(FOVEA_BOUNDARY_NAME, frame._batch_result_states[tab]["traces"])

    def _write_batch_sources(self, directory):
        frame = self.frame
        results = []
        for index in range(2):
            folder = Path(directory) / f"image-{index}"
            folder.mkdir()
            base = folder / "light"
            write_analyze(str(base), self.image.astype(np.int16))
            path = base.with_suffix(".img")
            results.append({
                "input": str(path), "source_snapshot": frame._source_file_snapshot(path),
                "image": self.image, "traces": self.traces, "fovea_x": 120,
            })
        return results

    def test_save_all_writes_an_unvisited_tab_without_opening_its_canvas(self):
        frame = self.frame
        with tempfile.TemporaryDirectory() as directory:
            frame._open_aidas_batch_results_viewer(self._write_batch_sources(directory))
            self.root.update()
            self.assertEqual(len(frame._batch_result_tab_canvases), 1)
            with mock.patch.object(frame, "_create_batch_result_canvas", wraps=frame._create_batch_result_canvas) as create:
                saved = frame._save_all_batch_result_tabs()
            create.assert_not_called()
            self.assertEqual(len(saved), 2)
            for output_pair in saved:
                for output in output_pair:
                    self.assertTrue(Path(output + ".hdr").is_file())
                    self.assertTrue(Path(output + ".img").is_file())

    def test_save_all_rejects_a_replaced_source_in_an_unvisited_tab(self):
        frame = self.frame
        with tempfile.TemporaryDirectory() as directory:
            results = self._write_batch_sources(directory)
            frame._open_aidas_batch_results_viewer(results)
            changed = Path(results[1]["input"])
            write_analyze(str(changed.with_suffix("")), self.image[:32, :120].astype(np.int16))

            with mock.patch("aidas.steps.step2_annotate.messagebox.showerror") as error:
                saved = frame._save_all_batch_result_tabs()

            self.assertEqual(len(saved), 1)
            error.assert_called_once()
            self.assertIn("removed or changed after segmentation", error.call_args.args[1])
            self.assertFalse((changed.parent / "nasal").exists())
            self.assertFalse((changed.parent / "temporal").exists())
            self.assertIsNone(frame.image_data)

    def test_revisited_tab_refits_to_the_resized_viewport(self):
        frame = self.frame
        notebook = frame.batch_results_notebook
        first, second, _third = notebook.tabs()
        first_canvas = frame.image_canvas
        before_zoom = first_canvas.get_zoom()
        notebook.select(second)
        self.root.geometry("1400x760+0+0")
        self.root.update()
        notebook.select(first)
        self.root.update()

        height, width = self.image.shape
        expected = 0.95 * min(first_canvas.canvas.winfo_width() / width,
                              first_canvas.canvas.winfo_height() / height)
        self.assertNotEqual(expected, before_zoom)
        self.assertAlmostEqual(first_canvas.get_zoom(), expected)

    def test_quick_selection_paints_only_the_latest_unvisited_tab(self):
        frame = self.frame
        notebook = frame.batch_results_notebook
        _first, second, third = notebook.tabs()
        notebook.select(second)
        notebook.select(third)
        self.root.update()

        self.assertIsNone(frame._batch_result_tab_canvases[second]._img_id)
        self.assertIsNotNone(frame._batch_result_tab_canvases[third]._img_id)
        self.assertIsNone(frame._batch_result_paint_after_id)
        notebook.select(second)
        self.root.update()
        self.assertIsNotNone(frame._batch_result_tab_canvases[second]._img_id)

    def test_reselecting_active_tab_keeps_unfinished_line_zoom_and_annotations(self):
        frame = self.frame
        frame.set_manual_segmentation_enabled(True)
        name = BOUNDARY_NAMES[0]
        del frame.boundary_traces[name]
        frame.boundary_order.remove(name)
        frame._set_completion_from_traces()
        frame._start_boundary_for_name(name)
        frame.image_canvas.set_active_line([(0, 4), (40, 6)])
        frame.image_canvas.set_zoom(1.5)
        self.root.update()
        snapshot = frame._annotation_snapshot()
        points = frame.image_canvas.get_active_line()
        zoom = frame.image_canvas.get_zoom()

        with mock.patch.object(frame, "_load_editor_state", wraps=frame._load_editor_state) as load:
            frame.batch_results_notebook.select(frame._active_batch_result_tab)
            self.root.update()

        load.assert_not_called()
        self.assertEqual(frame.active_boundary, name)
        self.assertEqual(frame.image_canvas.get_active_line(), points)
        self.assertEqual(frame.image_canvas.get_zoom(), zoom)
        self.assertEqual(frame._annotation_snapshot(), snapshot)


if __name__ == "__main__":
    unittest.main()
