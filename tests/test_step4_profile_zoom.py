import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import customtkinter as ctk
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from aidas.steps.step4_analyze_isez import (
    IMAGEJ_PLOT_BOX,
    ISezROI,
    MATLAB_ISEZ_Y_LIMITS,
    RESULTS_HEADERS,
    Step4Frame,
    Step4ProfileZoomDialog,
    analyze_and_save_roi,
    intensity_profile,
    _focused_profile_limits,
    _nearest_local_minimum,
    _nearest_profile_sample,
    _profile_selection_boundary,
    _profile_selection_plot_data,
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
        self.assertEqual(source.count("= AppButton("), 3)
        self.assertIn('text="Save and Exit"', source)
        self.assertIn('text="Apply"', source)
        self.assertIn('text="Cancel"', source)
        self.assertIn('variant="success"', source)
        self.assertIn('variant="primary"', source)
        self.assertIn('variant="danger"', source)
        self.assertNotIn('"flat-color-icons--checkmark.png"', source)
        self.assertNotIn("load_color_close_ctk_icon", source)
        apply_button_source = source.split("self.apply_button = AppButton(", 1)[1].split(
            "self.apply_button.pack", 1
        )[0]
        self.assertNotIn("image=", apply_button_source)
        self.assertNotIn("compound=", apply_button_source)
        self.assertNotIn("Manual edits use", source)
        self.assertNotIn("without automatic minimum snapping", source)
        footer_source = source.split("footer = ttk.Frame(", 1)[1].split(
            "chart_frame = ttk.Frame(", 1
        )[0]
        self.assertNotIn("ttk.Label(", footer_source)
        dialog_source = inspect.getsource(Step4ProfileZoomDialog)
        self.assertNotIn("cursor_var", dialog_source)
        self.assertNotIn("apply_status_var", dialog_source)

    @staticmethod
    def _zoom_frame():
        frame = Step4Frame.__new__(Step4Frame)
        frame.image = object()
        frame.current_roi_idx = 3
        frame.profile_clicks = [2.0, 8.0]
        frame.rois = [SimpleNamespace(suffix=f"{index + 1:02d}") for index in range(4)]
        frame.completed = {}
        frame._roi_adjust_bounds = {}
        frame._sync_entry_vars_from_clicks = mock.Mock()
        frame._remember_current_roi_clicks = mock.Mock()
        frame._render_current_roi = mock.Mock()
        frame._save_current_roi = mock.Mock(return_value=True)
        return frame

    @staticmethod
    def _zoom_dialog(*, changed=False):
        dialog = Step4ProfileZoomDialog.__new__(Step4ProfileZoomDialog)
        dialog.roi_index = 3
        dialog._start, dialog._end = 4, 19
        dialog._saved_bounds = (2, 8) if changed else (4, 19)
        dialog._applied_bounds = dialog._saved_bounds
        dialog._adjust_bounds = not changed
        dialog._saved_adjust_bounds = True
        dialog._applied_adjust_bounds = True
        dialog._actions_busy = False
        dialog._drag_boundary = None
        dialog._drag_view = None
        dialog._closing = False
        dialog.save_button = mock.Mock()
        dialog.apply_button = mock.Mock()
        dialog._on_apply_callback = mock.Mock(return_value=True)
        dialog._on_save_callback = mock.Mock(return_value=True)
        dialog._on_close_callback = mock.Mock()
        dialog.destroy = mock.Mock()
        return dialog

    @staticmethod
    def _flat_tailed_bell_profile():
        profile = np.full(140, 20.0)
        profile[16:80] += 80 * np.sin(np.linspace(0, np.pi, 64)) ** 2
        return profile

    def test_profile_zoom_apply_updates_boundaries_without_saving(self):
        frame = self._zoom_frame()

        self.assertTrue(frame._apply_profile_zoom_clicks(3, 4, 19))

        self.assertEqual(frame.profile_clicks, [4.0, 19.0])
        frame._sync_entry_vars_from_clicks.assert_called_once_with()
        frame._remember_current_roi_clicks.assert_called_once_with()
        frame._render_current_roi.assert_called_once_with()
        frame._save_current_roi.assert_not_called()
        self.assertEqual(frame.current_roi_idx, 3)
        self.assertFalse(frame._roi_adjust_bounds["04"])

    def test_profile_zoom_save_commits_the_new_boundaries_without_advancing(self):
        frame = self._zoom_frame()

        def commit(**options):
            self.assertEqual(frame.profile_clicks, [4.0, 19.0])
            frame._sync_entry_vars_from_clicks.assert_called_once_with()
            frame._remember_current_roi_clicks.assert_called_once_with()
            return True

        frame._save_current_roi.side_effect = commit

        self.assertTrue(frame._save_profile_zoom_clicks(3, 4, 19))

        frame._save_current_roi.assert_called_once_with(auto_advance=False, apply_entry_values=False)
        frame._render_current_roi.assert_called_once_with()
        self.assertEqual(frame.current_roi_idx, 3)

    def test_profile_zoom_cannot_save_an_inactive_roi(self):
        frame = self._zoom_frame()
        with mock.patch("aidas.steps.step4_analyze_isez.messagebox.showwarning") as warning:
            self.assertFalse(frame._save_profile_zoom_clicks(2, 4, 19))

        warning.assert_called_once()
        self.assertEqual(frame.profile_clicks, [2.0, 8.0])
        frame._save_current_roi.assert_not_called()
        frame._render_current_roi.assert_not_called()

    def test_dialog_apply_keeps_open_and_does_not_save(self):
        dialog = self._zoom_dialog(changed=True)

        def apply(index, start, end, *, adjust_bounds):
            self.assertEqual((index, start, end), (3, 4, 19))
            self.assertFalse(adjust_bounds)
            for button in (dialog.save_button, dialog.apply_button):
                self.assertEqual(button.state.call_args, mock.call(["disabled"]))
            return True

        dialog._on_apply_callback.side_effect = apply
        dialog._apply()

        dialog._on_apply_callback.assert_called_once_with(3, 4, 19, adjust_bounds=False)
        dialog._on_save_callback.assert_not_called()
        dialog.destroy.assert_not_called()
        dialog._on_close_callback.assert_not_called()
        self.assertFalse(dialog._closing)
        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["disabled"]))
        self.assertEqual(dialog._saved_bounds, (2, 8))
        self.assertEqual(dialog._applied_bounds, (4, 19))
        self.assertTrue(dialog._saved_adjust_bounds)
        self.assertFalse(dialog._applied_adjust_bounds)
        dialog._apply()
        dialog._save()
        dialog._on_apply_callback.assert_called_once_with(3, 4, 19, adjust_bounds=False)
        dialog._on_save_callback.assert_not_called()

    def test_dialog_save_stays_open_on_failure_and_closes_once_on_success(self):
        dialog = self._zoom_dialog(changed=True)
        outcomes = iter((False, True))

        def save(index, start, end, *, adjust_bounds):
            self.assertEqual((index, start, end), (3, 4, 19))
            self.assertFalse(adjust_bounds)
            for button in (dialog.save_button, dialog.apply_button):
                self.assertEqual(button.state.call_args, mock.call(["disabled"]))
            return next(outcomes)

        dialog._on_save_callback.side_effect = save
        dialog._save()

        dialog.destroy.assert_not_called()
        dialog._on_close_callback.assert_not_called()
        self.assertFalse(dialog._closing)
        self.assertEqual(dialog._saved_bounds, (2, 8))
        self.assertEqual(dialog._applied_bounds, (2, 8))
        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["!disabled"]))

        dialog._save()
        dialog.close()

        self.assertEqual(dialog._on_save_callback.call_count, 2)
        dialog._on_apply_callback.assert_not_called()
        dialog.destroy.assert_called_once_with()
        dialog._on_close_callback.assert_called_once_with(dialog)
        self.assertEqual(dialog._saved_bounds, (4, 19))
        self.assertEqual(dialog._applied_bounds, (4, 19))
        self.assertFalse(dialog._saved_adjust_bounds)
        self.assertFalse(dialog._applied_adjust_bounds)

    def test_dialog_cancel_closes_once_without_applying_or_saving(self):
        dialog = self._zoom_dialog()

        dialog.close()
        dialog.close()

        dialog._on_apply_callback.assert_not_called()
        dialog._on_save_callback.assert_not_called()
        dialog.destroy.assert_called_once_with()
        dialog._on_close_callback.assert_called_once_with(dialog)

    def test_dialog_callback_errors_restore_both_actions(self):
        for action in ("apply", "save"):
            with self.subTest(action=action):
                dialog = self._zoom_dialog(changed=True)
                callback = getattr(dialog, f"_on_{action}_callback")
                callback.side_effect = RuntimeError("Callback failed")

                with self.assertRaisesRegex(RuntimeError, "Callback failed"):
                    getattr(dialog, f"_{action}")()

                for button in (dialog.save_button, dialog.apply_button):
                    self.assertEqual(button.state.call_args, mock.call(["!disabled"]))
                dialog.destroy.assert_not_called()
                dialog._on_close_callback.assert_not_called()
                self.assertFalse(dialog._actions_busy)
                self.assertEqual(dialog._saved_bounds, (2, 8))
                self.assertEqual(dialog._applied_bounds, (2, 8))

    def test_dialog_unchanged_or_busy_actions_do_not_invoke_callbacks(self):
        for busy in (False, True):
            with self.subTest(busy=busy):
                dialog = self._zoom_dialog(changed=busy)
                dialog._actions_busy = busy
                dialog._update_action_states()
                for button in (dialog.save_button, dialog.apply_button):
                    self.assertEqual(button.state.call_args, mock.call(["disabled"]))

                dialog._apply()
                dialog._save()

                dialog._on_apply_callback.assert_not_called()
                dialog._on_save_callback.assert_not_called()
                dialog.destroy.assert_not_called()

    def test_dragging_changes_actions_immediately_and_reverting_cleans_them(self):
        dialog = self._zoom_dialog()
        dialog.profile = np.arange(24, dtype=np.float64)
        dialog._start_line = mock.Mock()
        dialog._end_line = mock.Mock()
        dialog._update_closed_boundary = mock.Mock()
        dialog._refresh_measurements = mock.Mock()

        dialog._move_boundary("start", 5, refresh_measurements=False)
        self.assertFalse(dialog._adjust_bounds)

        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["!disabled"]))
        dialog._refresh_measurements.assert_not_called()

        dialog._move_boundary("start", 4, refresh_measurements=False)
        self.assertTrue(dialog._adjust_bounds)

        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["disabled"]))

    def test_reverting_an_applied_change_keeps_both_actions_active_until_confirmed(self):
        for action in ("apply", "save"):
            with self.subTest(action=action):
                dialog = self._zoom_dialog()
                dialog.profile = np.arange(24, dtype=np.float64)
                dialog._start_line = mock.Mock()
                dialog._end_line = mock.Mock()
                dialog._update_closed_boundary = mock.Mock()
                dialog._move_boundary("start", 5, refresh_measurements=False)
                dialog._apply()
                self.assertEqual(dialog._applied_bounds, (5, 19))
                for button in (dialog.save_button, dialog.apply_button):
                    self.assertEqual(button.state.call_args, mock.call(["disabled"]))

                dialog._move_boundary("start", 6, refresh_measurements=False)
                for button in (dialog.save_button, dialog.apply_button):
                    self.assertEqual(button.state.call_args, mock.call(["!disabled"]))
                dialog._move_boundary("start", 5, refresh_measurements=False)
                for button in (dialog.save_button, dialog.apply_button):
                    self.assertEqual(button.state.call_args, mock.call(["disabled"]))

                dialog._move_boundary("start", 4, refresh_measurements=False)
                self.assertTrue(dialog._adjust_bounds)

                for button in (dialog.save_button, dialog.apply_button):
                    self.assertEqual(button.state.call_args, mock.call(["!disabled"]))
                getattr(dialog, f"_{action}")()
                for button in (dialog.save_button, dialog.apply_button):
                    self.assertEqual(button.state.call_args, mock.call(["disabled"]))
                self.assertEqual(dialog._applied_bounds, dialog._saved_bounds)
                self.assertTrue(dialog._applied_adjust_bounds)
                if action == "apply":
                    dialog._on_save_callback.assert_not_called()
                    dialog.destroy.assert_not_called()
                else:
                    dialog._on_save_callback.assert_called_once_with(
                        3, 4, 19, adjust_bounds=True
                    )
                    dialog.destroy.assert_called_once_with()

    def test_failed_apply_keeps_both_actions_dirty(self):
        dialog = self._zoom_dialog(changed=True)
        dialog._on_apply_callback.return_value = False

        dialog._apply()

        self.assertEqual(dialog._saved_bounds, (2, 8))
        self.assertEqual(dialog._applied_bounds, (2, 8))
        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["!disabled"]))
        dialog.destroy.assert_not_called()

    def test_adjustment_mode_is_part_of_saved_and_applied_state(self):
        dialog = self._zoom_dialog()
        dialog._adjust_bounds = False
        dialog._update_action_states()
        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["!disabled"]))

        dialog._apply()

        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["disabled"]))
        dialog._adjust_bounds = True
        dialog._update_action_states()
        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["!disabled"]))
        dialog._adjust_bounds = False
        dialog._update_action_states()
        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["disabled"]))
        dialog._adjust_bounds = True
        dialog._apply()
        for button in (dialog.save_button, dialog.apply_button):
            self.assertEqual(button.state.call_args, mock.call(["disabled"]))

    def test_dragged_bell_baselines_use_exact_samples_and_preview_mode(self):
        profile = self._flat_tailed_bell_profile()
        for start in (20, 25, 30):
            automatic = _profile_selection_plot_data(profile, start, 110)
            self.assertEqual(automatic["adjusted_start"], 17)

        dialog = self._zoom_dialog()
        dialog.profile = profile
        dialog._start, dialog._end = 35, 110
        dialog._saved_bounds = dialog._applied_bounds = (35, 110)
        dialog.figure = Figure()
        FigureCanvasAgg(dialog.figure)
        dialog.ax = dialog.figure.add_subplot(111)
        dialog.canvas = mock.Mock()
        dialog.measurement_vars = {name: mock.Mock() for name in RESULTS_HEADERS[1:]}
        dialog._on_preview_callback = mock.Mock(return_value={name: 1.0 for name in RESULTS_HEADERS[1:]})
        dialog._draw_profile()
        dialog.figure.canvas.draw()
        press_x = dialog.ax.transData.transform((35, 0))[0]
        dialog._on_press(SimpleNamespace(inaxes=dialog.ax, xdata=35.0, x=press_x, button=1))

        for start in (20, 25, 30):
            dialog._on_motion(SimpleNamespace(inaxes=dialog.ax, xdata=float(start)))
            self.assertEqual(dialog._start, start)
            self.assertFalse(dialog._adjust_bounds)
            np.testing.assert_array_equal(dialog._baseline_line.get_xdata(), [start, 110])
            self.assertEqual(dialog._plot_data["adjusted_start"], start)
        dialog._on_preview_callback.assert_not_called()

        dialog._on_release(SimpleNamespace(inaxes=dialog.ax, xdata=30.0))

        dialog._on_preview_callback.assert_called_once_with(3, 30, 110, adjust_bounds=False)
        np.testing.assert_array_equal(dialog._baseline_line.get_xdata(), [30, 110])
        np.testing.assert_array_equal(dialog.profile, profile)

    def test_saved_manual_bounds_survive_overview_and_reopening(self):
        frame = self._zoom_frame()
        profile = self._flat_tailed_bell_profile()
        frame.image = np.repeat((profile * 3)[:, None], 4, axis=1)
        frame.rois = [ISezROI(f"{index + 1:02d}", left=1, right=4, low=1, high=140) for index in range(4)]
        frame.roi_clicks = {}
        frame.auto_detection_reviews = {}
        frame.batch_roi_notebook = None
        frame.canvas = None
        frame.status_var = mock.Mock()
        frame._profile_zoom_dialog = None
        for attribute in (
            "_refresh_roi_list", "_start_roi_update_animation", "_select_roi_in_list",
            "_draw_roi_overview_grid", "_update_build_stack_button_state",
        ):
            setattr(frame, attribute, mock.Mock())
        frame._save_current_roi = Step4Frame._save_current_roi.__get__(frame, Step4Frame)

        self.assertTrue(frame._save_profile_zoom_clicks(3, 25, 110, adjust_bounds=False))

        saved = frame.completed["04"]
        self.assertTrue(saved.manual_bounds)
        self.assertEqual(saved.adjusted_start, 25)
        np.testing.assert_array_equal(saved.baseline_x, [25, 110])
        frame._roi_adjust_bounds.clear()
        overview = frame._roi_overview_result(3)
        self.assertTrue(overview.manual_bounds)
        np.testing.assert_array_equal(overview.baseline_x, saved.baseline_x)
        np.testing.assert_allclose(overview.normalized_y, saved.normalized_y)

        with mock.patch("aidas.steps.step4_analyze_isez.Step4ProfileZoomDialog") as constructor:
            frame._open_profile_zoom(3)

        self.assertFalse(constructor.call_args.kwargs["adjust_bounds"])
        self.assertFalse(constructor.call_args.kwargs["saved_adjust_bounds"])
        self.assertEqual(constructor.call_args.kwargs["saved_bounds"], (25, 110))
        frame._set_roi_adjustment("04", True)
        automatic = frame._roi_overview_result(3)
        self.assertFalse(automatic.manual_bounds)
        self.assertEqual(automatic.adjusted_start, 17)

    def test_reopening_an_applied_unsaved_selection_keeps_confirmed_bounds(self):
        frame = self._zoom_frame()
        frame.rois = [SimpleNamespace(suffix=f"{index + 1:02d}") for index in range(4)]
        confirmed = SimpleNamespace(start=2, end=8)
        frame.completed = {"04": confirmed}
        frame._roi_overview_result = mock.Mock(return_value=SimpleNamespace(start=4, end=19))
        frame._profile_zoom_measurements = mock.Mock()
        dialog = self._zoom_dialog(changed=True)
        dialog._on_apply_callback = frame._apply_profile_zoom_clicks
        dialog._on_close_callback = frame._profile_zoom_closed
        frame._profile_zoom_dialog = dialog

        dialog._apply()
        dialog.close()

        self.assertEqual(frame.profile_clicks, [4.0, 19.0])
        self.assertIs(frame.completed["04"], confirmed)
        self.assertIsNone(frame._profile_zoom_dialog)
        with (
            mock.patch("aidas.steps.step4_analyze_isez.intensity_profile", return_value=np.arange(24)),
            mock.patch("aidas.steps.step4_analyze_isez.Step4ProfileZoomDialog") as constructor,
        ):
            frame._open_profile_zoom(3)

        constructor.assert_called_once()
        options = constructor.call_args.kwargs
        self.assertEqual((options["start"], options["end"]), (4.0, 19.0))
        self.assertEqual(options["saved_bounds"], (2, 8))
        self.assertFalse(options["adjust_bounds"])
        self.assertTrue(options["saved_adjust_bounds"])
        frame._save_current_roi.assert_not_called()

    def _normalized_zoom_dialog(self):
        dialog = self._zoom_dialog()
        positions = np.arange(140, dtype=np.float64)
        image = np.repeat((3 * (15 + 80 * np.exp(-((positions - 80) / 8) ** 2) + positions * 0.12))[:, None], 4, axis=1)
        roi = ISezROI("01", left=1, right=4, low=1, high=140)
        original = intensity_profile(image, roi)
        dialog.profile = original.copy()
        dialog._start, dialog._end = 70, 95
        dialog._saved_bounds = dialog._applied_bounds = (70, 95)
        dialog.figure = Figure()
        FigureCanvasAgg(dialog.figure)
        dialog.ax = dialog.figure.add_subplot(111)
        dialog.canvas = mock.Mock()
        dialog._refresh_measurements = mock.Mock()

        dialog._draw_profile()
        return dialog, image, roi, original

    def test_zoom_matches_saved_normalized_result_and_preserves_raw_profile(self):
        dialog, image, roi, original = self._normalized_zoom_dialog()
        previous_limits = dialog.ax.get_xlim()
        for start in (70, 40, 70):
            if start != dialog._start:
                dialog._move_boundary("start", start, refresh_measurements=False)
            self.assertEqual(dialog._adjust_bounds, start == 70)
            saved = analyze_and_save_roi(
                image, roi, start_click=start, end_click=95, adjust_bounds=dialog._adjust_bounds,
            )
            np.testing.assert_allclose(dialog._boundary_curve.get_xdata(), saved.normalized_x)
            np.testing.assert_allclose(dialog._boundary_curve.get_ydata(), saved.normalized_y)
            np.testing.assert_allclose(dialog._baseline_line.get_xdata(), saved.baseline_x)
            np.testing.assert_allclose(dialog._baseline_line.get_ydata(), saved.baseline_y)
            left, right = dialog.ax.get_xlim()
            self.assertLessEqual(left, previous_limits[0])
            self.assertGreaterEqual(right, previous_limits[1])
            previous_limits = (left, right)
            context_x = dialog._context_curve.get_xdata()
            context_y = dialog._context_curve.get_ydata()
            self.assertLessEqual(context_x[0], max(1, min(start, saved.adjusted_start) - 20))
            self.assertGreaterEqual(context_x[-1], min(original.size, max(95, saved.adjusted_end) + 20))
            overlap = np.searchsorted(context_x, saved.normalized_x)
            np.testing.assert_array_equal(context_x[overlap], saved.normalized_x)
            np.testing.assert_allclose(context_y[overlap], saved.normalized_y)
            self.assertLess(dialog._context_curve.get_linewidth(), dialog._boundary_curve.get_linewidth())
            self.assertLess(dialog._context_curve.get_zorder(), dialog._boundary_curve.get_zorder())
            low, high = dialog.ax.get_ylim()
            self.assertLessEqual(low, MATLAB_ISEZ_Y_LIMITS[0])
            self.assertGreaterEqual(high, MATLAB_ISEZ_Y_LIMITS[1])
            selected = (
                (saved.normalized_x >= min(saved.baseline_x))
                & (saved.normalized_x <= max(saved.baseline_x))
                & np.isfinite(saved.normalized_y)
            )
            self.assertLessEqual(low, float(np.min(saved.normalized_y[selected])) - 5)
            self.assertGreaterEqual(high, float(np.max(saved.normalized_y[selected])) + 5)
            self._assert_readable_plot_frame(dialog)
            self.assertTrue(np.all(np.diff(dialog.ax.get_yticks()) % 20 == 0))
            np.testing.assert_array_equal(dialog.profile, original)

            index = int(np.argmin(np.abs(saved.normalized_y - 50)))
            sample = int(saved.normalized_x[index])
            normalized_value = float(saved.normalized_y[index])
            self.assertGreater(abs(normalized_value - original[sample - 1]), 1)
            dialog._on_motion(SimpleNamespace(inaxes=dialog.ax, xdata=float(sample)))
            self.assertEqual(dialog._hover_annotation.xy, (sample, normalized_value))
            self.assertAlmostEqual(dialog._crosshair_y.get_ydata()[0], normalized_value)
            self.assertIn(f"Normalized intensity: {normalized_value:.4f}", dialog._hover_annotation.get_text())

            earlier_sample = int(context_x[2])
            self.assertLess(earlier_sample, saved.normalized_x[0])
            dialog._on_motion(SimpleNamespace(inaxes=dialog.ax, xdata=float(earlier_sample)))
            self.assertEqual(dialog._hover_annotation.xy, (earlier_sample, float(context_y[2])))
            self.assertTrue(dialog._crosshair_y.get_visible())
            stable_limits = dialog.ax.get_xlim()
            dialog._update_closed_boundary()
            self.assertEqual(dialog.ax.get_xlim(), stable_limits)

    def test_drag_keeps_sample_mapping_fixed_until_release_refits_context(self):
        dialog, image, roi, _original = self._normalized_zoom_dialog()
        dialog.figure.canvas.draw()
        samples = np.array([(70, 0), (80, 50), (95, 0)], dtype=np.float64)
        pixels_before = dialog.ax.transData.transform(samples)
        view_before = (dialog.ax.get_xlim(), dialog.ax.get_ylim(), dialog.ax.get_box_aspect())

        dialog._on_press(SimpleNamespace(inaxes=dialog.ax, xdata=70.0, x=pixels_before[0, 0], button=1))

        self.assertEqual(dialog._drag_boundary, "start")
        for sample in (60.0, 50.0):
            dialog._on_motion(SimpleNamespace(inaxes=dialog.ax, xdata=sample))
            dialog.figure.canvas.draw()
            self.assertEqual((dialog.ax.get_xlim(), dialog.ax.get_ylim(), dialog.ax.get_box_aspect()), view_before)
            np.testing.assert_allclose(dialog.ax.transData.transform(samples), pixels_before)
            self.assertEqual(dialog._start, int(sample))

        dialog._on_release(SimpleNamespace(inaxes=dialog.ax, xdata=50.0))

        self.assertIsNone(dialog._drag_boundary)
        self.assertIsNone(dialog._drag_view)
        self.assertLessEqual(dialog.ax.get_xlim()[0], 30)
        self.assertGreaterEqual(dialog.ax.get_xlim()[1], view_before[0][1])
        dialog.figure.canvas.draw()
        self.assertFalse(np.allclose(dialog.ax.transData.transform(samples), pixels_before))
        saved = analyze_and_save_roi(image, roi, start_click=50, end_click=95, adjust_bounds=False)
        np.testing.assert_allclose(dialog._boundary_curve.get_ydata(), saved.normalized_y)
        np.testing.assert_allclose(dialog._baseline_line.get_ydata(), saved.baseline_y)
        self._assert_readable_plot_frame(dialog)

    def _assert_readable_plot_frame(self, dialog):
        dialog.figure.canvas.draw()
        box = dialog.ax.get_window_extent()
        x0, y0, x1, y1 = IMAGEJ_PLOT_BOX
        self.assertAlmostEqual(box.height / box.width, (y1 - y0) / (x1 - x0))
        self.assertGreater(box.width, dialog.figure.bbox.width * 0.5)
        return box.width

    def test_zoom_limits_include_selected_outliers_without_collapsing_plot_frame(self):
        dialog = self._zoom_dialog()
        dialog.figure = Figure()
        FigureCanvasAgg(dialog.figure)
        dialog.ax = dialog.figure.add_subplot(111)
        dialog._plot_data = {
            "center": 0,
            "x": np.array([-100, -55, 0, 60, 100, np.nan, np.inf]),
            "y": np.array([1e7, -43, 50, 156, -1e7, 1e12, -np.inf]),
            "baseline_x": np.array([-70, 70, np.nan]),
            "baseline_y": np.array([-84, 190, np.nan]),
        }
        dialog._context_y = np.array([-1e10, 1e10])

        dialog._set_normalized_plot_limits()

        left, right = dialog.ax.get_xlim()
        low, high = dialog.ax.get_ylim()
        self.assertTrue(np.all(np.isfinite([left, right, low, high])))
        self.assertLessEqual(left, -71)
        self.assertGreaterEqual(right, 71)
        self.assertLessEqual(low, -89)
        self.assertGreaterEqual(high, 195)
        self.assertGreater(low, -1000)
        self.assertLess(high, 1000)
        self.assertEqual(left % 10, 0)
        self.assertEqual(right % 10, 0)
        self.assertEqual(low % 20, 0)
        self.assertEqual(high % 20, 0)
        for ticks, base_step in ((dialog.ax.get_xticks(), 10), (dialog.ax.get_yticks(), 20)):
            self.assertLessEqual(ticks.size - 1, 10)
            steps = np.diff(ticks)
            self.assertTrue(np.all(steps > 0))
            self.assertTrue(np.all(steps % base_step == 0))
        self._assert_readable_plot_frame(dialog)

    def test_repeated_tail_edits_keep_a_wide_plot_despite_large_padding_and_context(self):
        samples = np.arange(1, 141, dtype=np.float64)
        profile = 10 + 30 * np.exp(-0.5 * ((samples - 80) / 8) ** 2)
        image = np.repeat((profile * 3)[:, None], 4, axis=1)
        roi = ISezROI("01", left=1, right=4, low=1, high=140)
        dialog = self._zoom_dialog()
        dialog.profile = intensity_profile(image, roi)
        original = dialog.profile.copy()
        dialog._start, dialog._end = 30, 40
        dialog._saved_bounds = dialog._applied_bounds = (30, 40)
        dialog._adjust_bounds = dialog._saved_adjust_bounds = dialog._applied_adjust_bounds = False
        dialog.figure = Figure()
        FigureCanvasAgg(dialog.figure)
        dialog.ax = dialog.figure.add_subplot(111)
        dialog.canvas = mock.Mock()
        dialog._refresh_measurements = mock.Mock()
        dialog._draw_profile()
        original_width = self._assert_readable_plot_frame(dialog)

        for start, end in ((30, 40), (35, 45), (30, 40), (35, 45)):
            moves = (("end", end), ("start", start)) if end > dialog._end else (("start", start), ("end", end))
            for boundary, sample in moves:
                current = dialog._start if boundary == "start" else dialog._end
                if current == sample:
                    continue
                dialog.figure.canvas.draw()
                pixel_x = dialog.ax.transData.transform((current, 0))[0]
                dialog._on_press(SimpleNamespace(inaxes=dialog.ax, xdata=float(current), x=pixel_x, button=1))
                self.assertEqual(dialog._drag_boundary, boundary)
                dialog._on_motion(SimpleNamespace(inaxes=dialog.ax, xdata=float(sample)))
                dialog._on_release(SimpleNamespace(inaxes=dialog.ax, xdata=float(sample)))
                np.testing.assert_array_equal(dialog._baseline_line.get_xdata(), [dialog._start, dialog._end])

            saved = analyze_and_save_roi(image, roi, start_click=start, end_click=end, adjust_bounds=False)
            selected = (saved.normalized_x >= start) & (saved.normalized_x <= end)
            self.assertGreaterEqual(float(np.min(saved.normalized_y[selected])), 0)
            self.assertLessEqual(float(np.max(saved.normalized_y[selected])), 100)
            self.assertGreater(float(np.max(saved.normalized_y)), 120)
            self.assertGreater(float(np.max(dialog._context_y)), 1000)
            self.assertEqual(dialog.ax.get_ylim(), MATLAB_ISEZ_Y_LIMITS)
            self.assertAlmostEqual(self._assert_readable_plot_frame(dialog), original_width)
            np.testing.assert_array_equal(dialog._baseline_line.get_xdata(), [start, end])
            np.testing.assert_allclose(dialog._boundary_curve.get_ydata(), saved.normalized_y)
            np.testing.assert_allclose(dialog._baseline_line.get_ydata(), saved.baseline_y)
            np.testing.assert_array_equal(dialog.profile, original)

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
        frame.current_roi_idx = 0
        frame.rois = [SimpleNamespace(suffix="01")]
        frame._roi_adjust_bounds = {"01": False}
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
        self.assertTrue(frame._roi_adjust_bounds["01"])
        self.assertEqual(frame._render_current_roi.call_count, 2)
        frame._auto_save_current_roi.assert_not_called()
        frame.after_idle.assert_called_once()
        callback = frame.after_idle.call_args.args[0]
        self.assertTrue(callable(callback))

        callback()

        frame._auto_save_current_roi.assert_called_once_with()

    @staticmethod
    def _line_positioning_frame(*, disabled=True, profile=None):
        if profile is None:
            profile = np.full(30, 10.0)
            profile[4], profile[13], profile[23] = 2.0, 0.5, 1.0
        frame = Step4Frame.__new__(Step4Frame)
        frame.ax_roi_grid = object()
        frame.ax_profile = object()
        frame.image = object()
        frame._current_profile = np.asarray(profile, dtype=np.float64)
        frame.current_roi_idx = 0
        frame.rois = [SimpleNamespace(suffix="01")]
        frame._roi_adjust_bounds = {}
        frame.profile_clicks = []
        frame.roi_clicks = {}
        frame.completed = {}
        frame.disable_line_positioning_var = mock.Mock()
        frame.disable_line_positioning_var.get.return_value = disabled
        frame.profile_status_var = mock.Mock()
        frame.status_var = mock.Mock()
        frame._close_profile_zoom = mock.Mock()
        frame._sync_entry_vars_from_clicks = mock.Mock()
        frame._render_current_roi = mock.Mock()
        frame._auto_save_current_roi = mock.Mock()
        frame.after_idle = mock.Mock()
        frame.after_cancel = mock.Mock()
        return frame

    @staticmethod
    def _click_main_profile(frame, x_value):
        frame._on_profile_click(
            SimpleNamespace(inaxes=frame.ax_profile, xdata=x_value, button=1)
        )

    def test_disabling_line_positioning_keeps_first_and_second_clicked_samples(self):
        frame = self._line_positioning_frame()

        with mock.patch(
            "aidas.steps.step4_analyze_isez._nearest_local_minimum",
            side_effect=AssertionError("Manual positioning must not search for a minimum."),
        ):
            self._click_main_profile(frame, 6.2)
            self._click_main_profile(frame, 22.3)

        self.assertEqual(frame.profile_clicks, [6.0, 22.0])
        self.assertEqual(frame.roi_clicks["01"], [6.0, 22.0])
        self.assertFalse(frame._should_adjust_roi_bounds("01"))
        self.assertEqual(frame._render_current_roi.call_count, 2)
        frame.after_idle.assert_called_once()
        frame.after_idle.call_args.args[0]()
        frame._auto_save_current_roi.assert_called_once_with()

    def test_disabling_line_positioning_corrects_either_existing_boundary_exactly(self):
        for click, expected in ((8.2, [8.0, 24.0]), (21.2, [5.0, 21.0])):
            with self.subTest(click=click):
                frame = self._line_positioning_frame()
                frame.profile_clicks = [5.0, 24.0]
                frame._set_roi_adjustment("01", True)

                self._click_main_profile(frame, click)

                self.assertEqual(frame.profile_clicks, expected)
                self.assertEqual(frame.roi_clicks["01"], expected)
                self.assertFalse(frame._should_adjust_roi_bounds("01"))
                frame.after_idle.assert_called_once()
                frame.after_idle.call_args.args[0]()
                frame._auto_save_current_roi.assert_called_once_with()

    def test_checked_line_positions_survive_preview_and_actual_roi_save(self):
        profile = self._flat_tailed_bell_profile()
        frame = self._line_positioning_frame(profile=profile)
        frame.image = np.repeat((profile * 3)[:, None], 4, axis=1)
        frame.rois = [ISezROI("01", left=1, right=4, low=1, high=140)]
        frame.auto_detection_reviews = {}
        frame.batch_roi_notebook = None
        frame.canvas = None
        for attribute in (
            "_refresh_roi_list", "_start_roi_update_animation", "_select_roi_in_list",
            "_draw_roi_overview_grid", "_update_build_stack_button_state",
        ):
            setattr(frame, attribute, mock.Mock())
        automatic = analyze_and_save_roi(
            frame.image, frame.rois[0], start_click=25, end_click=110,
        )
        self.assertNotEqual(automatic.adjusted_start, 25)

        self._click_main_profile(frame, 25)
        self._click_main_profile(frame, 110)
        with mock.patch(
            "aidas.steps.step4_analyze_isez.adjust_isez_bounds",
            side_effect=AssertionError("Manual positions must not be adjusted later."),
        ):
            preview = frame._roi_overview_result(0)
            self.assertIsNotNone(preview)
            self.assertTrue(frame._save_current_roi(auto_advance=False, apply_entry_values=False))
            saved = frame.completed["01"]

        self.assertTrue(saved.manual_bounds)
        self.assertEqual((saved.start, saved.end), (25, 110))
        self.assertEqual((saved.adjusted_start, saved.adjusted_end), (25, 110))
        np.testing.assert_array_equal(saved.baseline_x, [25, 110])
        np.testing.assert_array_equal(preview.baseline_x, saved.baseline_x)
        np.testing.assert_allclose(preview.normalized_y, saved.normalized_y)
        frame._roi_adjust_bounds.clear()
        self.assertFalse(frame._should_adjust_roi_bounds("01"))

    def test_toggling_only_changes_future_placement_and_restores_automatic_minimum(self):
        frame = self._line_positioning_frame()
        self._click_main_profile(frame, 6.2)
        self._click_main_profile(frame, 22.3)
        before = list(frame.profile_clicks)
        remembered = list(frame.roi_clicks["01"])
        frame._render_current_roi.reset_mock()
        frame.after_idle.reset_mock()
        frame._update_profile_status = mock.Mock()

        for disabled in (False, True, False):
            frame.disable_line_positioning_var.get.return_value = disabled
            frame._on_line_positioning_changed()
            self.assertEqual(frame.profile_clicks, before)
            self.assertEqual(frame.roi_clicks["01"], remembered)
            self.assertFalse(frame._should_adjust_roi_bounds("01"))
            frame.after_idle.assert_not_called()
            frame._auto_save_current_roi.assert_not_called()
        self._click_main_profile(frame, 6.2)

        self.assertEqual(frame.profile_clicks, [5.0, 22.0])
        self.assertTrue(frame._should_adjust_roi_bounds("01"))
        frame.after_idle.assert_called_once()
        frame.after_idle.call_args.args[0]()
        frame._auto_save_current_roi.assert_called_once_with()

    @staticmethod
    def _grid_frame():
        frame = Step4Frame.__new__(Step4Frame)
        frame.figure = Figure(figsize=(6, 3), dpi=100)
        FigureCanvasAgg(frame.figure)
        frame.ax_roi_grid = frame.figure.add_subplot(111)
        frame.ax_roi_grid.set_xlim(0, 7)
        frame.ax_roi_grid.set_ylim(0, 3)
        frame.figure.canvas.draw()
        frame.ax_profile = object()
        frame.image = object()
        frame.rois = [SimpleNamespace(suffix=f"{index + 1:02d}") for index in range(16)]
        frame.current_roi_idx = 0
        frame.profile_clicks = [6.0, 30.0]
        frame.roi_clicks = {"02": [12.0, 42.0]}
        frame.completed = {"15": SimpleNamespace(start=25, end=65)}
        frame._current_profile = np.arange(100, dtype=np.float64)
        frame.start_var = mock.Mock()
        frame.end_var = mock.Mock()
        frame.roi_table = mock.Mock()
        frame.after_idle = mock.Mock()
        frame._profile_zoom_dialog = None
        frame._roi_overview_result = mock.Mock(
            side_effect=lambda index: object() if index in (0, 1, 14) else None
        )
        frame._open_profile_zoom = mock.Mock()
        frame._render_current_roi = mock.Mock()
        return frame

    @staticmethod
    def _grid_click(frame, x, y, *, button=1, double=False):
        pixel_x, pixel_y = frame.ax_roi_grid.transData.transform((x, y))
        event = MouseEvent(
            "button_press_event", frame.figure.canvas, pixel_x, pixel_y,
            button=button, dblclick=double,
        )
        frame._on_profile_click(event)

    def test_grid_single_click_selects_and_loads_profile_without_opening_zoom(self):
        frame = self._grid_frame()
        previous_zoom = SimpleNamespace(roi_index=0, close=mock.Mock())
        frame._profile_zoom_dialog = previous_zoom
        rendered = []
        frame._render_current_roi.side_effect = lambda: rendered.append(
            (frame.current_roi_idx, list(frame.profile_clicks))
        )

        self._grid_click(frame, 1.5, 2.5)

        self.assertEqual(frame.current_roi_idx, 1)
        self.assertEqual(frame.profile_clicks, [12.0, 42.0])
        self.assertEqual(frame.roi_clicks["01"], [6.0, 30.0])
        frame.roi_table.selection_set.assert_called_once_with("1")
        frame.start_var.set.assert_called_once_with("12")
        frame.end_var.set.assert_called_once_with("42")
        previous_zoom.close.assert_called_once_with()
        self.assertIsNone(frame._profile_zoom_dialog)

        current_zoom = SimpleNamespace(roi_index=1, close=mock.Mock())
        frame._profile_zoom_dialog = current_zoom
        self._grid_click(frame, 1.5, 2.5)
        self.assertIs(frame._profile_zoom_dialog, current_zoom)
        current_zoom.close.assert_not_called()

        self._grid_click(frame, 0.5, 0.5)
        self.assertEqual(frame.current_roi_idx, 14)
        self.assertEqual(frame.profile_clicks, [25.0, 65.0])
        self.assertEqual(frame.roi_clicks["02"], [12.0, 42.0])
        frame.roi_table.selection_set.assert_called_with("14")
        current_zoom.close.assert_called_once_with()
        self.assertEqual(rendered, [
            (1, [12.0, 42.0]), (1, [12.0, 42.0]), (14, [25.0, 65.0]),
        ])
        frame._open_profile_zoom.assert_not_called()

    def test_grid_double_click_opens_only_the_selected_drawn_preview(self):
        for index, has_preview in ((1, True), (2, False)):
            with self.subTest(index=index):
                frame = self._grid_frame()

                def open_zoom(selected):
                    self.assertEqual(selected, frame.current_roi_idx)
                    self.assertEqual(frame.profile_clicks, [12.0, 42.0])
                    frame.roi_table.selection_set.assert_called_with(str(selected))

                frame._open_profile_zoom.side_effect = open_zoom
                self._grid_click(frame, index + 0.5, 2.5)
                frame._open_profile_zoom.assert_not_called()
                self._grid_click(frame, index + 0.5, 2.5, double=True)

                self.assertEqual(frame.current_roi_idx, index)
                self.assertEqual(frame._render_current_roi.call_count, 2)
                if has_preview:
                    frame._open_profile_zoom.assert_called_once_with(index)
                else:
                    frame._open_profile_zoom.assert_not_called()

    def test_grid_nonleft_clicks_and_empty_cells_leave_selection_unchanged(self):
        for x, y, button, double in (
            (1.5, 2.5, 3, False),
            (1.5, 2.5, 3, True),
            (1.5, 2.5, 2, False),
            (6.5, 0.5, 1, False),
            (6.5, 0.5, 1, True),
            (-0.5, 2.5, 1, True),
        ):
            with self.subTest(x=x, y=y, button=button, double=double):
                frame = self._grid_frame()
                dialog = SimpleNamespace(roi_index=0, close=mock.Mock())
                frame._profile_zoom_dialog = dialog

                self._grid_click(frame, x, y, button=button, double=double)

                self.assertEqual(frame.current_roi_idx, 0)
                self.assertEqual(frame.profile_clicks, [6.0, 30.0])
                self.assertEqual(frame.roi_clicks, {"02": [12.0, 42.0]})
                frame.roi_table.selection_set.assert_not_called()
                frame._render_current_roi.assert_not_called()
                frame._open_profile_zoom.assert_not_called()
                frame._roi_overview_result.assert_not_called()
                dialog.close.assert_not_called()
                self.assertIs(frame._profile_zoom_dialog, dialog)

    def test_grid_hover_hints_follow_visible_previews_and_the_current_axes(self):
        frame = self._grid_frame()
        frame.canvas = mock.Mock()
        frame._flashing_updated_rois = set()
        preview = SimpleNamespace(
            center=50.0,
            normalized_x=np.array([40.0, 50.0, 60.0]),
            normalized_y=np.array([0.0, 100.0, 0.0]),
            baseline_x=np.array([40.0, 60.0]),
            baseline_y=np.array([0.0, 0.0]),
        )
        frame._roi_overview_result.side_effect = (
            lambda index: preview if index == 1 else None
        )
        frame._draw_roi_overview_grid()
        frame.figure.canvas.draw()
        first_axes = frame.ax_roi_grid
        first_hint = first_axes._aidas_hover_hint

        def motion(x, y):
            pixel_x, pixel_y = frame.ax_roi_grid.transData.transform((x, y))
            event = MouseEvent(
                "motion_notify_event", frame.figure.canvas, pixel_x, pixel_y
            )
            frame._on_plot_motion(event)
            return event

        motion(1.5, 2.5)
        self.assertTrue(first_hint.get_visible())
        self.assertEqual(first_hint.get_text(), "Double-click to zoom")
        frame.canvas.get_tk_widget().configure.assert_called_with(cursor="hand2")
        motion(2.5, 2.5)
        self.assertTrue(first_hint.get_visible())
        self.assertEqual(first_hint.get_text(), "Click to select")
        motion(6.5, 0.5)
        self.assertFalse(first_hint.get_visible())
        frame.canvas.get_tk_widget().configure.assert_called_with(cursor="")

        event = motion(1.5, 2.5)
        frame._on_plot_axes_leave(event)
        self.assertFalse(first_hint.get_visible())
        frame.canvas.get_tk_widget().configure.assert_called_with(cursor="")

        frame.figure = Figure(figsize=(6, 3), dpi=100)
        FigureCanvasAgg(frame.figure)
        frame.ax_roi_grid = frame.figure.add_subplot(111)
        frame._roi_overview_result.side_effect = None
        frame._roi_overview_result.return_value = None
        frame._draw_roi_overview_grid()
        frame.figure.canvas.draw()
        current_hint = frame.ax_roi_grid._aidas_hover_hint
        motion(1.5, 2.5)
        self.assertTrue(current_hint.get_visible())
        self.assertEqual(current_hint.get_text(), "Click to select")
        self.assertIsNot(current_hint, first_hint)
        self.assertEqual(first_axes._aidas_preview_cells, {1})
        self.assertEqual(frame.ax_roi_grid._aidas_preview_cells, set())
        frame._on_plot_axes_leave(SimpleNamespace(inaxes=first_axes))
        self.assertTrue(current_hint.get_visible())
        self.assertFalse(first_hint.get_visible())
        frame.canvas.draw_idle.reset_mock()
        frame._hide_roi_grid_hint(SimpleNamespace(inaxes=None))
        self.assertFalse(current_hint.get_visible())
        self.assertFalse(first_hint.get_visible())
        frame.canvas.draw_idle.assert_called_once_with()
        frame.canvas.get_tk_widget().configure.assert_called_with(cursor="")

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

    def test_dragged_boundary_uses_exact_selected_sample_on_release(self):
        profile = np.full(20, 9.0)
        profile[11] = 1.5
        dialog = Step4ProfileZoomDialog.__new__(Step4ProfileZoomDialog)
        dialog.profile = profile
        dialog.ax = object()
        dialog._drag_boundary = "end"
        dialog._start = 4
        dialog._end = 16
        dialog.canvas = mock.Mock()
        dialog._move_boundary = mock.Mock()

        dialog._on_release(
            SimpleNamespace(inaxes=dialog.ax, xdata=14.0)
        )

        dialog._move_boundary.assert_called_once_with(
            "end",
            14,
            refresh_measurements=True,
        )
        self.assertIsNone(dialog._drag_boundary)
        dialog.canvas.draw_idle.assert_called_once_with()

    def test_click_replaces_auto_detected_boundary_without_minimum_snapping(self):
        profile = np.full(30, 9.0)
        profile[5] = 1.5
        dialog = Step4ProfileZoomDialog.__new__(Step4ProfileZoomDialog)
        dialog.profile = profile
        dialog.ax = object()
        dialog._start = 10
        dialog._end = 24
        dialog._drag_boundary = None
        dialog._boundary_at_event = mock.Mock(return_value=None)
        dialog._move_boundary = mock.Mock()
        dialog.canvas = mock.Mock()

        dialog._on_press(
            SimpleNamespace(inaxes=dialog.ax, xdata=7.0, button=1)
        )

        dialog._move_boundary.assert_called_once_with(
            "start",
            7,
            refresh_measurements=True,
        )
        self.assertIsNone(dialog._drag_boundary)
        dialog.canvas.draw_idle.assert_called_once_with()

    def test_zoom_manual_handlers_do_not_call_local_minimum_search(self):
        press_source = inspect.getsource(Step4ProfileZoomDialog._on_press)
        release_source = inspect.getsource(Step4ProfileZoomDialog._on_release)

        self.assertNotIn("_nearest_local_minimum", press_source)
        self.assertNotIn("_nearest_local_minimum", release_source)
        self.assertIn("_nearest_profile_sample", press_source)
        self.assertIn("_nearest_profile_sample", release_source)

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
