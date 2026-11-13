"""Boundary-aware flat minima apply to main-plot placement only."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from aidas.steps import step4_analyze_isez as module
from tests.steps.step4 import test_step4_profile_zoom as fixtures


class Step4PlateauSnappingTests(unittest.TestCase):
    @staticmethod
    def _profile():
        profile = np.full(60, 10.0)
        profile[4:8] = 1.0      # Samples 5..8: Start uses 8.
        profile[33:38] = 0.5   # Samples 34..38: End uses 34.
        return profile

    @staticmethod
    def _frame(*, disabled=False, profile=None):
        return fixtures.Step4ProfileZoomHelpersTests._line_positioning_frame(
            disabled=disabled, profile=profile,
        )

    @staticmethod
    def _click(frame, position):
        fixtures.Step4ProfileZoomHelpersTests._click_main_profile(frame, position)

    def test_start_uses_right_edge_and_end_uses_left_edge_of_same_minimum_plateau(self):
        profile = np.array([9.0, 7.0, 1.0, 1.0, 1.0, 1.0, 8.0, 9.0])
        self.assertEqual(module._nearest_local_minimum(profile, 4.2, boundary="start"), (6, 1.0))
        self.assertEqual(module._nearest_local_minimum(profile, 4.2, boundary="end"), (3, 1.0))

    def test_default_minimum_selection_keeps_legacy_distance_and_lower_index_ties(self):
        plateau = np.array([9.0, 7.0, 1.0, 1.0, 1.0, 1.0, 8.0, 9.0])
        valleys = np.array([8.0, 1.0, 7.0, 1.0, 9.0])
        self.assertEqual(module._nearest_local_minimum(plateau, 4.2), (4, 1.0))
        self.assertEqual(module._nearest_local_minimum(plateau, 4.2, boundary=None), (4, 1.0))
        self.assertEqual(module._nearest_local_minimum(valleys, 3.0, search_radius=2), (2, 1.0))

    def test_plateau_expansion_can_extend_beyond_the_initial_search_radius(self):
        profile = np.full(80, 20.0)
        profile[9:65] = 2.0
        self.assertEqual(module._nearest_local_minimum(profile, 35, search_radius=2, boundary="start"), (65, 2.0))
        self.assertEqual(module._nearest_local_minimum(profile, 35, search_radius=2, boundary="end"), (10, 2.0))

    def test_equal_disconnected_valleys_choose_nearest_valley_before_expanding(self):
        profile = np.array([9.0, 1.0, 1.0, 9.0, 9.0, 1.0, 1.0, 9.0])
        for click, start, end in ((4.5, 3, 2), (5.5, 7, 6)):
            with self.subTest(click=click):
                self.assertEqual(module._nearest_local_minimum(profile, click, boundary="start"), (start, 1.0))
                self.assertEqual(module._nearest_local_minimum(profile, click, boundary="end"), (end, 1.0))

    def test_isolated_minimum_stays_at_the_same_sample_for_either_boundary(self):
        profile = np.array([9.0, 5.0, 1.0, 6.0, 9.0])
        for boundary in (None, "start", "end"):
            with self.subTest(boundary=boundary):
                self.assertEqual(module._nearest_local_minimum(profile, 3.1, boundary=boundary), (3, 1.0))

    def test_single_sample_and_profile_endpoint_plateaus_stay_in_bounds(self):
        for boundary in ("start", "end"):
            with self.subTest(boundary=boundary):
                self.assertEqual(module._nearest_local_minimum(np.array([2.5]), 20, boundary=boundary), (1, 2.5))
        flat = np.ones(20)
        self.assertEqual(module._nearest_local_minimum(flat, 10, boundary="start"), (20, 1.0))
        self.assertEqual(module._nearest_local_minimum(flat, 10, boundary="end"), (1, 1.0))

    def test_nearly_equal_neighbors_do_not_join_an_exact_minimum_plateau(self):
        higher = np.nextafter(1.0, np.inf)
        profile = np.array([9.0, higher, 1.0, 1.0, higher, 9.0])
        self.assertEqual(module._nearest_local_minimum(profile, 3.4, boundary="start"), (4, 1.0))
        self.assertEqual(module._nearest_local_minimum(profile, 3.4, boundary="end"), (3, 1.0))

    def test_nonfinite_gaps_do_not_connect_equal_valleys(self):
        for gap in (np.nan, np.inf, -np.inf):
            with self.subTest(gap=gap):
                profile = np.array([9.0, 1.0, 1.0, gap, 1.0, 1.0, 9.0])
                self.assertEqual(module._nearest_local_minimum(profile, 3, boundary="start"), (3, 1.0))
                self.assertEqual(module._nearest_local_minimum(profile, 3, boundary="end"), (2, 1.0))
                self.assertEqual(module._nearest_local_minimum(profile, 5, boundary="start"), (6, 1.0))
                self.assertEqual(module._nearest_local_minimum(profile, 5, boundary="end"), (5, 1.0))

    def test_snapping_accepts_read_only_profiles_and_never_changes_their_values(self):
        profile = np.array([9.0, 1.0, 1.0, np.nan, 1.0, 1.0, 9.0])
        original = profile.copy()
        profile.flags.writeable = False
        for boundary in (None, "start", "end"):
            module._nearest_local_minimum(profile, 3, boundary=boundary)
        np.testing.assert_array_equal(profile, original)

    def test_unknown_boundary_is_rejected(self):
        for boundary in ("middle", "", "START"):
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                module._nearest_local_minimum(self._profile(), 6, boundary=boundary)

    def test_initial_main_plot_clicks_use_plateau_edges_and_keep_automatic_save(self):
        profile = self._profile()
        original = profile.copy()
        frame = self._frame(profile=profile)

        self._click(frame, 6)
        self.assertEqual(frame.profile_clicks, [8.0])
        self._click(frame, 36)

        self.assertEqual(frame.profile_clicks, [8.0, 34.0])
        self.assertEqual(frame.roi_clicks["01"], [8.0, 34.0])
        self.assertFalse(frame._should_adjust_roi_bounds("01"))
        self.assertEqual(frame._render_current_roi.call_count, 2)
        frame.after_idle.assert_called_once()
        frame.after_idle.call_args.args[0]()
        frame._auto_save_current_roi.assert_called_once_with()
        np.testing.assert_array_equal(profile, original)

    def test_reversed_initial_clicks_assign_edges_to_final_start_and_end_roles(self):
        frame = self._frame(profile=self._profile())

        self._click(frame, 36)
        self._click(frame, 6)

        self.assertEqual(frame.profile_clicks, [8.0, 34.0])
        self.assertEqual(frame.roi_clicks["01"], [8.0, 34.0])
        frame.after_idle.assert_called_once()

    def test_two_clicks_on_one_plateau_keep_an_ordered_selection(self):
        profile = np.full(40, 4.0)
        profile[4:8] = 1.0
        for first, second in ((6, 7), (7, 6)):
            with self.subTest(first=first, second=second):
                frame = self._frame(profile=profile)
                self._click(frame, first)
                self._click(frame, second)
                self.assertEqual(frame.profile_clicks, sorted(frame.profile_clicks))
                self.assertLess(frame.profile_clicks[0], frame.profile_clicks[1])
                self.assertEqual(frame.roi_clicks["01"], frame.profile_clicks)

    def test_unequal_plateau_heights_keep_selected_inner_edges_in_actual_saved_analysis(self):
        for start_height, end_height in ((1.0, 0.0), (0.0, 1.0)):
            with self.subTest(start=start_height, end=end_height):
                profile = np.full(40, 4.0)
                profile[4:8], profile[24:28] = start_height, end_height
                original = profile.copy()
                frame = self._frame(profile=profile)
                image = np.repeat((profile * 3)[:, None], 4, axis=1)
                roi = module.ISezROI("01", left=1, right=4, low=1, high=40)
                frame.image, frame.rois = image, [roi]

                self._click(frame, 6)
                self._click(frame, 27)

                self.assertEqual(frame.profile_clicks, [8.0, 25.0])
                self.assertFalse(frame._should_adjust_roi_bounds("01"))
                saved = module.analyze_and_save_roi(
                    image, roi, start_click=frame.profile_clicks[0], end_click=frame.profile_clicks[1],
                    adjust_bounds=frame._should_adjust_roi_bounds("01"),
                )
                self.assertEqual((saved.start, saved.end), (8, 25))
                self.assertEqual((saved.adjusted_start, saved.adjusted_end), (8, 25))
                self.assertTrue(saved.manual_bounds)
                np.testing.assert_array_equal(saved.baseline_x, [8, 25])
                np.testing.assert_array_equal(profile, original)

    def test_mixed_plateau_and_singleton_keep_original_matlab_adjustment_on_other_endpoint(self):
        for plateau_side in ("start", "end"):
            with self.subTest(plateau_side=plateau_side):
                profile = np.full(40, 20.0)
                if plateau_side == "start":
                    profile[4:8] = 0.0
                    profile[23:27] = [10.1, 10.0, 10.1, 11.0]
                    clicks, selected = (6, 25), (8, 25)
                    expected_legacy = (8, 26)
                else:
                    profile[5:9] = [11.0, 10.1, 10.0, 10.1]
                    profile[24:28] = 0.0
                    clicks, selected = (8, 27), (8, 25)
                    expected_legacy = (7, 25)
                legacy = module.adjust_isez_bounds(profile, *selected)[:2]
                self.assertEqual(legacy, expected_legacy)
                frame = self._frame(profile=profile)

                self._click(frame, clicks[0])
                self._click(frame, clicks[1])

                expected = [float(selected[0]), float(legacy[1])] if plateau_side == "start" else [float(legacy[0]), float(selected[1])]
                self.assertEqual(frame.profile_clicks, expected)
                self.assertFalse(frame._should_adjust_roi_bounds("01"))

    def test_isolated_main_minima_keep_existing_automatic_adjustment_mode(self):
        profile = np.full(40, 4.0)
        profile[4], profile[24] = 1.0, 0.0
        frame = self._frame(profile=profile)

        self._click(frame, 6)
        self._click(frame, 27)

        self.assertEqual(frame.profile_clicks, [5.0, 25.0])
        self.assertTrue(frame._should_adjust_roi_bounds("01"))

    def test_repeated_same_plateau_click_does_not_readjust_the_opposite_exact_boundary(self):
        profile = np.full(100, 20.0)
        profile[9:20] = 0.0
        profile[20] = 5.0
        for sample in range(79, 101):
            profile[sample - 1] = 10.0 + 0.05 * (sample - 80)
        original = profile.copy()
        frame = self._frame(profile=profile)
        image = np.repeat((profile * 3)[:, None], 4, axis=1)
        roi = module.ISezROI("01", left=1, right=4, low=1, high=100)
        frame.image, frame.rois = image, [roi]
        frame.profile_clicks = [20.0, 80.0]
        frame._set_roi_adjustment("01", True)
        saved = []

        def save_current_selection():
            result = module.analyze_and_save_roi(
                image, roi, start_click=frame.profile_clicks[0], end_click=frame.profile_clicks[1],
                adjust_bounds=frame._should_adjust_roi_bounds("01"),
            )
            frame.completed["01"] = result
            saved.append(result)

        frame._auto_save_current_roi.side_effect = save_current_selection

        # The first correction performs the original non-plateau adjustment
        # once. A second adjustment would move End from 81 to 82, then 83.
        self._click(frame, 19)
        self.assertEqual(frame.profile_clicks, [20.0, 81.0])
        self.assertFalse(frame._should_adjust_roi_bounds("01"))
        frame.after_idle.call_args.args[0]()
        for _ in range(3):
            self._click(frame, 19)
            self.assertEqual(frame.profile_clicks, [20.0, 81.0])
            self.assertEqual(frame.roi_clicks["01"], [20.0, 81.0])
            self.assertFalse(frame._should_adjust_roi_bounds("01"))
            frame.after_idle.call_args.args[0]()

        self.assertEqual(len(saved), 4)
        for result in saved:
            self.assertEqual((result.start, result.end), (20, 81))
            self.assertEqual((result.adjusted_start, result.adjusted_end), (20, 81))
            self.assertTrue(result.manual_bounds)
        frame._auto_save_current_roi.assert_has_calls([mock.call()] * 4)
        np.testing.assert_array_equal(profile, original)

    def test_main_plot_corrections_choose_boundary_from_original_click_before_snapping(self):
        profile = np.full(60, 10.0)
        profile[19:23] = 1.0
        # Existing endpoints are isolated minima, so finalization cannot also
        # reinterpret a flat background at the untouched boundary as a plateau.
        profile[9], profile[29] = 0.0, 0.0
        for click, expected in ((19, [23.0, 30.0]), (21, [10.0, 20.0])):
            with self.subTest(click=click):
                frame = self._frame(profile=profile)
                frame.profile_clicks = [10.0, 30.0]

                self._click(frame, click)

                self.assertEqual(frame.profile_clicks, expected)
                self.assertEqual(frame.roi_clicks["01"], expected)
                self.assertFalse(frame._should_adjust_roi_bounds("01"))
                frame.after_idle.assert_called_once()

    def test_correcting_a_minimum_does_not_snap_the_opposite_flat_maximum(self):
        profile = np.full(60, 10.0)
        profile[19:23] = 1.0
        for click, selected in ((19, (23, 30)), (21, (10, 20))):
            with self.subTest(click=click):
                frame = self._frame(profile=profile)
                frame.profile_clicks = [10.0, 30.0]
                adjusted_start, adjusted_end, _slope = module.adjust_isez_bounds(profile, *selected)

                self._click(frame, click)

                expected = (23, adjusted_end) if click == 19 else (adjusted_start, 20)
                self.assertEqual(frame.profile_clicks, list(expected))
                self.assertFalse(frame._should_adjust_roi_bounds("01"))

    def test_disabling_automatic_positioning_keeps_initial_clicked_samples_exact(self):
        frame = self._frame(disabled=True, profile=self._profile())
        with mock.patch.object(module, "_nearest_local_minimum", side_effect=AssertionError("Manual click must stay exact")):
            self._click(frame, 6.1)
            self._click(frame, 36.2)
        self.assertEqual(frame.profile_clicks, [6.0, 36.0])
        self.assertEqual(frame.roi_clicks["01"], [6.0, 36.0])
        self.assertFalse(frame._should_adjust_roi_bounds("01"))
        frame.after_idle.assert_called_once()

    def test_disabling_automatic_positioning_keeps_corrections_exact(self):
        for click, expected in ((19.1, [19.0, 30.0]), (21.1, [10.0, 21.0])):
            with self.subTest(click=click):
                frame = self._frame(disabled=True, profile=self._profile())
                frame.profile_clicks = [10.0, 30.0]
                with mock.patch.object(module, "_nearest_local_minimum", side_effect=AssertionError("Manual correction must stay exact")):
                    self._click(frame, click)
                self.assertEqual(frame.profile_clicks, expected)
                self.assertFalse(frame._should_adjust_roi_bounds("01"))
                frame.after_idle.assert_called_once()

    def test_zoom_editor_keeps_exact_clicked_samples_on_the_same_flat_minima(self):
        for click, boundary in ((6.1, "start"), (36.2, "end")):
            with self.subTest(click=click):
                dialog = module.Step4ProfileZoomDialog.__new__(module.Step4ProfileZoomDialog)
                dialog._actions_busy = False
                dialog._closing = False
                dialog.profile = self._profile()
                dialog.ax = object()
                dialog._start, dialog._end = 10, 40
                dialog._drag_boundary = None
                dialog._boundary_at_event = mock.Mock(return_value=None)
                dialog._move_boundary = mock.Mock()
                dialog.canvas = mock.Mock()
                with mock.patch.object(module, "_nearest_local_minimum", side_effect=AssertionError("Zoom editor must stay exact")):
                    dialog._on_press(SimpleNamespace(inaxes=dialog.ax, xdata=click, button=1))
                dialog._move_boundary.assert_called_once_with(
                    boundary, round(click), refresh_measurements=True,
                )


if __name__ == "__main__":
    unittest.main()
