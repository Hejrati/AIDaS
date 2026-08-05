import unittest

import numpy as np

from aidas.steps.step4_analyze_isez import (
    _focused_profile_limits,
    _nearest_profile_sample,
    _profile_selection_boundary,
    _profile_zoom_window_geometry,
    _updated_profile_bounds,
)


class Step4ProfileZoomHelpersTests(unittest.TestCase):
    def test_focused_limits_add_margin_around_selection(self):
        self.assertEqual(_focused_profile_limits(100, 20, 40), (15, 45))

    def test_focused_limits_clamp_margin_to_profile(self):
        self.assertEqual(_focused_profile_limits(30, 2, 10), (1, 15))
        self.assertEqual(_focused_profile_limits(30, 29, 25), (20, 30))

    def test_nearest_sample_uses_one_based_index_and_exact_value(self):
        profile = np.array([2.5, 4.25, 8.75, 16.0])
        self.assertEqual(_nearest_profile_sample(profile, 2.2), (2, 4.25))
        self.assertEqual(_nearest_profile_sample(profile, 99), (4, 16.0))

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
