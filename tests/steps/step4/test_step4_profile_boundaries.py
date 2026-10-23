import unittest
from types import SimpleNamespace
from unittest import mock

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from aidas.steps.step4_analyze_isez import (
    ISezROI,
    RESULTS_HEADERS,
    Step4ProfileZoomDialog,
    _profile_selection_plot_data,
    analyze_and_save_roi,
)


class Step4ProfileBoundaryAlignmentTests(unittest.TestCase):
    @staticmethod
    def _dialog(*, adjust_bounds=True):
        dialog = Step4ProfileZoomDialog.__new__(Step4ProfileZoomDialog)
        positions = np.arange(140, dtype=float)
        dialog.profile = 15 + 80 * np.exp(-((positions - 80) / 8) ** 2) + positions * 0.12
        dialog.roi_index = 3
        dialog._start, dialog._end = 70, 95
        dialog._saved_bounds = dialog._applied_bounds = (70, 95)
        dialog._adjust_bounds = adjust_bounds
        dialog._saved_adjust_bounds = dialog._applied_adjust_bounds = adjust_bounds
        dialog._actions_busy = False
        dialog._drag_boundary = dialog._drag_view = dialog._context_window = None
        dialog.save_button = mock.Mock()
        dialog.apply_button = mock.Mock()
        dialog._on_apply_callback = mock.Mock(return_value=True)
        dialog._on_save_callback = mock.Mock(return_value=True)
        dialog._on_preview_callback = mock.Mock(
            return_value={name: 1.0 for name in RESULTS_HEADERS[1:]}
        )
        dialog.measurement_vars = {name: mock.Mock() for name in RESULTS_HEADERS[1:]}
        dialog.figure = Figure(figsize=(8, 5), dpi=100)
        FigureCanvasAgg(dialog.figure)
        dialog.ax = dialog.figure.add_subplot(111)
        dialog.canvas = mock.Mock()
        dialog._draw_profile()
        dialog.figure.canvas.draw()
        return dialog

    def _assert_handles_match_baseline(self, dialog):
        start, end = dialog._baseline_line.get_xdata()
        np.testing.assert_array_equal(dialog._start_line.get_xdata(), [start, start])
        np.testing.assert_array_equal(dialog._end_line.get_xdata(), [end, end])
        self.assertEqual(dialog._displayed_profile_bounds(), (start, end))

    def test_automatic_handles_align_with_both_analyzed_endpoints(self):
        dialog = self._dialog()
        self.assertNotEqual(dialog._plot_data["adjusted_start"], dialog._start)
        self.assertNotEqual(dialog._plot_data["adjusted_end"], dialog._end)

        self._assert_handles_match_baseline(dialog)

        self.assertEqual((dialog._start, dialog._end), (70, 95))
        self.assertTrue(dialog._adjust_bounds)

    def test_manual_handles_align_with_exact_selected_samples(self):
        dialog = self._dialog(adjust_bounds=False)

        self._assert_handles_match_baseline(dialog)

        self.assertEqual(dialog._displayed_profile_bounds(), (70, 95))

    def test_hit_testing_uses_visible_handles_instead_of_original_clicks(self):
        dialog = self._dialog()
        displayed = dialog._displayed_profile_bounds()
        for name, visible, original in zip(("start", "end"), displayed, (70, 95)):
            with self.subTest(boundary=name):
                visible_x = dialog.ax.transData.transform((visible, 0))[0]
                visible_event = SimpleNamespace(inaxes=dialog.ax, xdata=float(visible), x=visible_x)
                self.assertEqual(
                    dialog._boundary_at_event(visible_event, tolerance_pixels=0.25), name
                )
                original_x = dialog.ax.transData.transform((original, 0))[0]
                original_event = SimpleNamespace(inaxes=dialog.ax, xdata=float(original), x=original_x)
                self.assertIsNone(
                    dialog._boundary_at_event(original_event, tolerance_pixels=0.25)
                )

    def test_first_manual_edit_preserves_opposite_analyzed_endpoint(self):
        for boundary, target in (("start", 60), ("end", 100)):
            with self.subTest(boundary=boundary):
                dialog = self._dialog()
                effective_start, effective_end = dialog._displayed_profile_bounds()

                dialog._move_boundary(boundary, target, refresh_measurements=True)

                expected = (target, effective_end) if boundary == "start" else (effective_start, target)
                self.assertEqual((dialog._start, dialog._end), expected)
                self.assertFalse(dialog._adjust_bounds)
                self._assert_handles_match_baseline(dialog)
                dialog._on_preview_callback.assert_called_once_with(
                    3, *expected, adjust_bounds=False
                )
                dialog._on_apply_callback.assert_not_called()
                dialog._on_save_callback.assert_not_called()

    def test_subsequent_manual_edits_keep_exact_samples_and_measurement_mode(self):
        dialog = self._dialog()
        dialog._move_boundary("start", 60, refresh_measurements=True)
        dialog._move_boundary("end", 100, refresh_measurements=True)

        self.assertEqual((dialog._start, dialog._end), (60, 100))
        self.assertFalse(dialog._adjust_bounds)
        self._assert_handles_match_baseline(dialog)
        dialog._on_preview_callback.assert_called_with(3, 60, 100, adjust_bounds=False)
        expected = _profile_selection_plot_data(dialog.profile, 60, 100, adjust_bounds=False)
        np.testing.assert_allclose(dialog._boundary_curve.get_xdata(), expected["x"])
        np.testing.assert_allclose(dialog._boundary_curve.get_ydata(), expected["y"])

    def test_selecting_same_visible_handle_does_not_change_automatic_bounds(self):
        dialog = self._dialog()
        visible_start, _visible_end = dialog._displayed_profile_bounds()

        dialog._move_boundary("start", visible_start, refresh_measurements=True)

        self.assertEqual((dialog._start, dialog._end), (70, 95))
        self.assertTrue(dialog._adjust_bounds)
        self.assertFalse(dialog._has_pending_changes())
        self._assert_handles_match_baseline(dialog)

    def test_returning_to_saved_visible_endpoint_restores_original_automatic_selection(self):
        dialog = self._dialog()
        saved_visible = dialog._displayed_profile_bounds()
        dialog._move_boundary("start", 60, refresh_measurements=False)
        self.assertTrue(dialog._has_pending_changes())

        dialog._move_boundary("start", saved_visible[0], refresh_measurements=True)

        self.assertEqual((dialog._start, dialog._end), (70, 95))
        self.assertTrue(dialog._adjust_bounds)
        self.assertFalse(dialog._has_pending_changes())
        self.assertEqual(dialog._displayed_profile_bounds(), saved_visible)
        self._assert_handles_match_baseline(dialog)
        dialog._on_preview_callback.assert_called_once_with(3, 70, 95, adjust_bounds=True)

    def test_redrawing_corrected_handles_preserves_raw_profile_and_saved_analysis(self):
        dialog = self._dialog()
        original = dialog.profile.copy()
        roi = ISezROI("04", left=1, right=4, low=1, high=140)
        saved = analyze_and_save_roi(
            np.empty((140, 4)), roi, start_click=70, end_click=95, profile=original
        )
        saved_baseline_x, saved_baseline_y = saved.baseline_x.copy(), saved.baseline_y.copy()
        saved_normalized_x, saved_normalized_y = saved.normalized_x.copy(), saved.normalized_y.copy()

        dialog._draw_profile()
        dialog._update_closed_boundary()

        self._assert_handles_match_baseline(dialog)
        np.testing.assert_array_equal(dialog.profile, original)
        np.testing.assert_array_equal(saved.baseline_x, saved_baseline_x)
        np.testing.assert_array_equal(saved.baseline_y, saved_baseline_y)
        np.testing.assert_array_equal(saved.normalized_x, saved_normalized_x)
        np.testing.assert_array_equal(saved.normalized_y, saved_normalized_y)
        np.testing.assert_allclose(dialog._baseline_line.get_xdata(), saved.baseline_x)
        np.testing.assert_allclose(dialog._baseline_line.get_ydata(), saved.baseline_y)
        self.assertEqual((saved.start, saved.end), (70, 95))
        self.assertEqual(dialog._saved_bounds, (70, 95))
        self.assertEqual(dialog._applied_bounds, (70, 95))
        dialog._on_apply_callback.assert_not_called()
        dialog._on_save_callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
