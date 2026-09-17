import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import customtkinter as ctk
import numpy as np

from aidas.steps.step4_analyze_isez import (
    Step4Frame,
    Step4ProfileZoomDialog,
    _focused_profile_limits,
    _nearest_local_minimum,
    _nearest_profile_sample,
    _profile_selection_boundary,
    _profile_zoom_window_geometry,
    _plot_palette,
    _updated_profile_bounds,
)
from aidas.ui.theme import COLOR_PAIRS


class Step4ProfileZoomHelpersTests(unittest.TestCase):
    def tearDown(self):
        ctk.set_appearance_mode("System")

    def test_plot_palette_tracks_dark_mode_without_losing_contrast(self):
        ctk.set_appearance_mode("Dark")
        palette = _plot_palette()

        self.assertEqual(palette["figure"], COLOR_PAIRS["surface"][1])
        self.assertEqual(palette["axes"], COLOR_PAIRS["surface_subtle"][1])
        self.assertEqual(palette["line"], COLOR_PAIRS["text"][1])
        self.assertNotEqual(palette["line"], palette["axes"])

    def test_zoom_dialog_uses_app_icon_and_shared_rounded_buttons(self):
        source = inspect.getsource(Step4ProfileZoomDialog.__init__)

        self.assertIn("apply_app_icon_to(self)", source)
        self.assertEqual(source.count("= AppButton("), 2)
        self.assertIn('variant="primary"', source)
        self.assertIn('variant="secondary"', source)
        self.assertIn('"flat-color-icons--checkmark.png"', source)
        self.assertIn("load_color_close_ctk_icon(self, size=20)", source)

    def test_zoom_dialog_drags_boundary_lines_without_radio_buttons(self):
        init_source = inspect.getsource(Step4ProfileZoomDialog.__init__)
        motion_source = inspect.getsource(Step4ProfileZoomDialog._on_motion)

        self.assertNotIn("ttk.Radiobutton", init_source)
        self.assertNotIn("boundary_var", init_source)
        self.assertIn("Ellipse fitting measurements", init_source)
        self.assertIn('mpl_connect("button_press_event", self._on_press)', init_source)
        self.assertIn('mpl_connect("button_release_event", self._on_release)', init_source)
        self.assertIn("_drag_boundary", motion_source)

    def test_focused_limits_add_margin_around_selection(self):
        self.assertEqual(_focused_profile_limits(100, 20, 40), (15, 45))

    def test_focused_limits_clamp_margin_to_profile(self):
        self.assertEqual(_focused_profile_limits(30, 2, 10), (1, 15))
        self.assertEqual(_focused_profile_limits(30, 29, 25), (20, 30))

    def test_nearest_sample_uses_one_based_index_and_exact_value(self):
        profile = np.array([2.5, 4.25, 8.75, 16.0])
        self.assertEqual(_nearest_profile_sample(profile, 2.2), (2, 4.25))
        self.assertEqual(_nearest_profile_sample(profile, 99), (4, 16.0))

    def test_click_snaps_to_the_minimum_in_a_fixed_nearby_window(self):
        profile = np.array([9.0, 6.0, 4.0, 7.0, 2.0, 8.0, 5.0])

        self.assertEqual(
            _nearest_local_minimum(profile, 3.2, search_radius=2),
            (5, 2.0),
        )

    def test_equal_nearby_minima_use_click_distance_then_lower_index(self):
        profile = np.array([8.0, 1.0, 7.0, 1.0, 9.0])

        self.assertEqual(
            _nearest_local_minimum(profile, 3.0, search_radius=2),
            (2, 1.0),
        )

    def test_main_profile_clicks_snap_and_queue_automatic_save(self):
        profile = np.full(30, 10.0)
        profile[4] = 2.0
        profile[13] = 0.5
        profile[23] = 1.0
        frame = Step4Frame.__new__(Step4Frame)
        frame.ax_roi_grid = object()
        frame.ax_profile = object()
        frame.image = object()
        frame._current_profile = profile
        frame.profile_clicks = []
        frame.profile_status_var = mock.Mock()
        frame._close_profile_zoom = mock.Mock()
        frame._sync_entry_vars_from_clicks = mock.Mock()
        frame._remember_current_roi_clicks = mock.Mock()
        frame._render_current_roi = mock.Mock()
        frame._auto_save_current_roi = mock.Mock()
        frame.after_idle = mock.Mock()

        def click(x_value):
            frame._on_profile_click(
                SimpleNamespace(inaxes=frame.ax_profile, xdata=x_value, button=1)
            )

        click(6.0)
        click(22.0)

        self.assertEqual(frame.profile_clicks, [5.0, 24.0])
        self.assertEqual(frame._render_current_roi.call_count, 2)
        frame._auto_save_current_roi.assert_not_called()
        frame.after_idle.assert_called_once()
        callback = frame.after_idle.call_args.args[0]
        self.assertIs(callback, frame._auto_save_current_roi)

        callback()

        frame._auto_save_current_roi.assert_called_once_with()

    def test_saving_uses_grid_animation_without_plot_overlays(self):
        save_source = inspect.getsource(Step4Frame._save_current_roi)
        data_source = inspect.getsource(Step4Frame._save_current_roi_data)
        grid_source = inspect.getsource(Step4Frame._draw_roi_overview_grid)
        zoom_apply_source = inspect.getsource(Step4ProfileZoomDialog._apply)

        self.assertNotIn("_show_plot_activity", save_source)
        self.assertNotIn('"Saving', save_source)
        self.assertNotIn('"Saving', zoom_apply_source)
        self.assertIn("self._start_roi_update_animation(roi.suffix)", data_source)
        self.assertIn('face = palette["success_soft"]', grid_source)
        self.assertIn('face = palette["warning_soft"]', grid_source)
        self.assertIn("facecolor=face", grid_source)

    def test_dragged_boundary_snaps_to_local_minimum_on_release(self):
        profile = np.full(20, 9.0)
        profile[11] = 1.5
        dialog = Step4ProfileZoomDialog.__new__(Step4ProfileZoomDialog)
        dialog.profile = profile
        dialog.ax = object()
        dialog._drag_boundary = "end"
        dialog._start = 4
        dialog._end = 16
        dialog.cursor_var = mock.Mock()
        dialog.canvas = mock.Mock()
        dialog._move_boundary = mock.Mock()

        dialog._on_release(
            SimpleNamespace(inaxes=dialog.ax, xdata=14.0)
        )

        dialog._move_boundary.assert_called_once_with(
            "end",
            12,
            refresh_measurements=True,
        )
        self.assertIsNone(dialog._drag_boundary)
        dialog.canvas.draw_idle.assert_called_once_with()

    def test_end_click_save_uses_fast_path_and_advances_new_roi(self):
        frame = Step4Frame.__new__(Step4Frame)
        frame._auto_saving_roi = False
        frame.image = object()
        frame.profile_clicks = [5.0, 24.0]
        frame.completed = {}
        frame._current_roi_suffix = mock.Mock(return_value="01")
        frame._save_current_roi = mock.Mock(return_value=True)

        frame._auto_save_current_roi()

        frame._save_current_roi.assert_called_once_with(
            auto_advance=True,
            apply_entry_values=False,
        )
        self.assertFalse(frame._auto_saving_roi)

    def test_selected_boundary_uses_straight_line_between_endpoint_values(self):
        profile = np.array([2.0, 6.0, 10.0, 4.0, 8.0])
        x_values, curve, baseline = _profile_selection_boundary(profile, 2, 5)
        np.testing.assert_array_equal(x_values, [2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(curve, [6.0, 10.0, 4.0, 8.0])
        np.testing.assert_allclose(baseline, [6.0, 6.6666667, 7.3333333, 8.0])

    def test_updating_start_cannot_cross_end(self):
        self.assertEqual(
            _updated_profile_bounds(10, 20, boundary="start", sample=25, n_points=40),
            (19, 20),
        )

    def test_updating_end_cannot_cross_start(self):
        self.assertEqual(
            _updated_profile_bounds(10, 20, boundary="end", sample=5, n_points=40),
            (10, 11),
        )

    def test_updating_bounds_rejects_unknown_boundary(self):
        with self.assertRaisesRegex(ValueError, "Unknown profile boundary"):
            _updated_profile_bounds(10, 20, boundary="middle", sample=15, n_points=40)

    def test_popup_size_is_dynamic_and_centered_on_monitor(self):
        self.assertEqual(
            _profile_zoom_window_geometry((0, 0, 1920, 1040)),
            (1100, 811, 410, 114),
        )
        self.assertEqual(
            _profile_zoom_window_geometry((-1920, 0, 0, 1040)),
            (1100, 811, -1510, 114),
        )

    def test_popup_shrinks_to_fit_a_small_monitor(self):
        self.assertEqual(
            _profile_zoom_window_geometry((0, 0, 640, 480)),
            (576, 432, 32, 24),
        )


if __name__ == "__main__":
    unittest.main()
