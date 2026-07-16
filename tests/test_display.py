from __future__ import annotations

import unittest

from aidas.core.display import centered_position


class DisplayPositionTests(unittest.TestCase):
    def test_centers_on_primary_monitor_work_area(self):
        self.assertEqual(centered_position((0, 0, 1920, 1040), 572, 816), (674, 112))

    def test_centers_on_monitor_to_the_left_of_primary(self):
        self.assertEqual(centered_position((-1920, 0, 0, 1040), 572, 816), (-1246, 112))

    def test_centers_on_monitor_above_primary(self):
        self.assertEqual(centered_position((0, -1200, 1920, 0), 572, 816), (674, -1008))

    def test_keeps_oversized_window_at_work_area_origin(self):
        self.assertEqual(centered_position((1920, 40, 2420, 740), 572, 816), (1920, 40))


if __name__ == "__main__":
    unittest.main()
