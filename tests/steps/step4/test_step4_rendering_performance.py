from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from aidas.steps import step4_analyze_isez as module


def plot_frame():
    frame = module.Step4Frame.__new__(module.Step4Frame)
    samples = np.arange(1, 152, dtype=float)
    profile = np.full(samples.size, 12.0)
    bell = (samples >= 76) & (samples <= 100)
    profile[bell] = 25 - .12 * (samples[bell] - 88) ** 2
    frame.image = np.repeat(profile[:, None] * 3, 63, axis=1)
    frame.rois = [module.ISezROI(str(i), i * 3 + 1, i * 3 + 3, low=1, high=151) for i in range(21)]
    frame.completed = {roi.suffix: module.analyze_and_save_roi(
        frame.image, roi, start_click=76, end_click=100) for roi in frame.rois[:20]}
    frame.roi_clicks = {suffix: [float(result.start), float(result.end)]
                        for suffix, result in frame.completed.items()}
    frame.auto_detection_reviews = {frame.rois[20].suffix: "manual"}
    frame._roi_adjust_bounds = {}
    frame._flashing_updated_rois = set()
    frame.current_roi_idx = 20
    frame.profile_clicks = []
    frame.figure = Figure(figsize=(8, 5), dpi=100)
    frame.canvas = FigureCanvasAgg(frame.figure)
    grid = frame.figure.add_gridspec(2, 1, height_ratios=[1.35, .95])
    frame.ax_roi_grid = frame.figure.add_subplot(grid[0, 0])
    frame.ax_profile = frame.figure.add_subplot(grid[1, 0])
    frame.figure.subplots_adjust(left=.065, right=.965, bottom=.085, top=.985, hspace=.13)
    frame.canvas._aidas_roi_grid_axes = frame.ax_roi_grid
    frame.canvas.mpl_connect("draw_event", frame._cache_roi_grid_background)
    widget = Mock()
    widget.winfo_manager.return_value = "pack"
    frame.canvas.get_tk_widget = lambda: widget
    frame._ensure_plot_canvas = lambda: None
    frame._place_line_positioning_checkbox = Mock()
    frame._update_profile_status = lambda profile: None
    frame._update_confirm_button_state = lambda: None
    return frame


class Step4RenderingPerformanceTests(unittest.TestCase):
    def test_metrics_follow_the_result_across_tabs_and_refresh_after_edits(self):
        frame = plot_frame()
        roi = frame.rois[0]
        first = frame.completed
        other_tab = {roi.suffix: module.analyze_and_save_roi(
            frame.image, roi, start_click=77, end_click=100)}
        with patch.object(module, "imagej_shape_measurements_from_frame", return_value={"AR": 2.0}) as measure:
            expected = frame._roi_metric_values(roi)
            frame.completed = other_tab
            frame._roi_metric_values(roi)
            frame.completed = first
            self.assertEqual(frame._roi_metric_values(roi), expected)
            self.assertEqual(measure.call_count, 2)
            # Changes in an array must invalidate even an otherwise identical
            # result object; swapping the result also starts with a fresh cache.
            frame.completed[roi.suffix].normalized_y[0] += 1
            frame._roi_metric_values(roi)
            self.assertEqual(measure.call_count, 3)
            frame.completed[roi.suffix] = module.analyze_and_save_roi(
                frame.image, roi, start_click=76, end_click=101)
            frame._roi_metric_values(roi)
            self.assertEqual(measure.call_count, 4)

    def test_reused_grid_and_profile_match_fresh_render_after_selection_and_theme_changes(self):
        reused = plot_frame()
        reused._render_current_roi()
        boxes = tuple(reused.ax_roi_grid.patches)
        profile_line = reused.ax_profile._aidas_profile_artists[0]
        palette = module._plot_palette()
        light = dict(palette, axes="#ffffff", figure="#f0f0f0", text="#111111", line="#111111")
        for index, colors in ((0, palette), (7, light), (20, palette)):
            fresh = plot_frame()
            for frame in (fresh, reused):
                frame.current_roi_idx = index
                frame.profile_clicks = list(frame.roi_clicks.get(frame.rois[index].suffix, []))
                frame._flashing_updated_rois = {frame.rois[2].suffix}
                frame.auto_detection_reviews[frame.rois[3].suffix] = "review"
                with patch.object(module, "_plot_palette", return_value=colors):
                    frame._render_current_roi()
            np.testing.assert_array_equal(reused.canvas.buffer_rgba(), fresh.canvas.buffer_rgba())
            self.assertEqual(tuple(reused.ax_roi_grid.patches), boxes)
            self.assertIs(reused.ax_profile._aidas_profile_artists[0], profile_line)
            self.assertEqual(len(reused.ax_roi_grid.patches), 21)
            self.assertEqual(len(reused.ax_profile.lines), 3)

    def test_hover_blits_only_the_grid_and_restores_its_original_pixels(self):
        frame = plot_frame()
        frame._render_current_roi()
        before = np.asarray(frame.canvas.buffer_rgba()).copy()
        x, y = frame.ax_roi_grid.transData.transform((.5, 2.5))
        with patch.object(frame.canvas, "draw_idle") as full_draw, \
                patch.object(frame.canvas, "blit", wraps=frame.canvas.blit) as blit:
            frame._on_plot_motion(MouseEvent("motion_notify_event", frame.canvas, x, y))
            self.assertTrue(frame.ax_roi_grid._aidas_hover_hint.get_visible())
            self.assertFalse(np.array_equal(before, frame.canvas.buffer_rgba()))
            frame._hide_roi_grid_hint()
            full_draw.assert_not_called()
            self.assertEqual(blit.call_count, 2)
            np.testing.assert_array_equal(frame.canvas.buffer_rgba(), before)

    def test_hidden_canvas_configure_does_not_resize_the_active_figure(self):
        frame = module.Step4Frame.__new__(module.Step4Frame)
        frame.canvas = Mock()
        frame.figure = Mock()
        frame._place_line_positioning_checkbox = Mock()
        frame._on_plot_canvas_configure(SimpleNamespace(widget=object(), width=100, height=100))
        frame.figure.set_size_inches.assert_not_called()
        frame.canvas.draw_idle.assert_not_called()
        frame._place_line_positioning_checkbox.assert_not_called()
        frame._on_plot_canvas_configure(SimpleNamespace(widget=frame.canvas.get_tk_widget()))
        frame._place_line_positioning_checkbox.assert_called_once()

    def test_hover_cleanup_includes_text_outside_narrow_grid_edges(self):
        frame = plot_frame()
        frame.figure.set_size_inches(3, 5)
        frame._render_current_roi()
        before = np.asarray(frame.canvas.buffer_rgba()).copy()
        x, y = frame.ax_roi_grid.transData.transform((6.5, 2.5))
        frame._on_plot_motion(MouseEvent("motion_notify_event", frame.canvas, x, y))
        self.assertTrue(frame.ax_roi_grid._aidas_hover_hint.get_visible())
        self.assertEqual(frame.ax_roi_grid._aidas_hover_hint.get_text(), "Double-click to zoom")
        frame._hide_roi_grid_hint()
        np.testing.assert_array_equal(frame.canvas.buffer_rgba(), before)


if __name__ == "__main__":
    unittest.main()
