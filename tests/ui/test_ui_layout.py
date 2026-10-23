from __future__ import annotations

import unittest

from aidas.utils.ui_layout import LAYOUT, workspace_sidebar_width


class WorkspaceLayoutTests(unittest.TestCase):
    def test_main_window_uses_three_quarters_of_the_work_area(self):
        self.assertEqual(LAYOUT.screen_fraction, 0.75)

    def test_design_width_keeps_the_compact_fixed_sidebar(self):
        sidebar = workspace_sidebar_width(LAYOUT.design_width)

        self.assertEqual(sidebar, LAYOUT.sidebar_width)

    def test_large_workspace_does_not_enlarge_the_sidebar(self):
        self.assertEqual(workspace_sidebar_width(3000), LAYOUT.sidebar_width)

    def test_minimum_window_keeps_both_panes_usable(self):
        usable = LAYOUT.minimum_width - LAYOUT.divider_width
        sidebar = workspace_sidebar_width(LAYOUT.minimum_width)

        self.assertGreaterEqual(sidebar, LAYOUT.sidebar_minimum)
        self.assertGreaterEqual(usable - sidebar, LAYOUT.content_minimum)

    def test_small_embedded_workspace_never_returns_an_overlapping_split(self):
        width = 500
        sidebar = workspace_sidebar_width(width)

        self.assertGreater(sidebar, 0)
        self.assertLess(sidebar, width - LAYOUT.divider_width)

    def test_requested_width_is_bounded_by_available_space(self):
        width = 1000

        self.assertEqual(
            workspace_sidebar_width(
                width,
                sidebar_width=-3,
                sidebar_minimum=0,
                content_minimum=0,
            ),
            1,
        )
        self.assertEqual(
            workspace_sidebar_width(
                width,
                sidebar_width=2000,
                sidebar_minimum=0,
                content_minimum=0,
            ),
            width - LAYOUT.divider_width,
        )


if __name__ == "__main__":
    unittest.main()
