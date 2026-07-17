from __future__ import annotations

import unittest

from aidas.core.display import centered_position, fit_size_to_bounds


class DisplayPositionTests(unittest.TestCase):
    def test_centers_on_primary_monitor_work_area(self):
        self.assertEqual(centered_position((0, 0, 1920, 1040), 572, 816), (674, 112))

    def test_centers_on_monitor_to_the_left_of_primary(self):
        self.assertEqual(centered_position((-1920, 0, 0, 1040), 572, 816), (-1246, 112))

    def test_centers_on_monitor_above_primary(self):
        self.assertEqual(centered_position((0, -1200, 1920, 0), 572, 816), (674, -1008))

    def test_keeps_oversized_window_at_work_area_origin(self):
        self.assertEqual(centered_position((1920, 40, 2420, 740), 572, 816), (1920, 40))


class DisplaySizingTests(unittest.TestCase):
    def test_design_size_is_not_enlarged_on_large_display(self):
        self.assertEqual(
            fit_size_to_bounds((0, 0, 3840, 2160), 480, 620),
            (480, 620, 1.0),
        )

    def test_design_size_scales_down_uniformly_on_small_display(self):
        width, height, scale = fit_size_to_bounds(
            (0, 0, 320, 240),
            480,
            620,
            maximum_fraction=0.9,
        )
        self.assertLess(scale, 1.0)
        self.assertLessEqual(width, 288)
        self.assertLessEqual(height, 216)
        self.assertAlmostEqual(width / height, 480 / 620, places=2)


if __name__ == "__main__":
    unittest.main()
