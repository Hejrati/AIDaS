from __future__ import annotations

import unittest

from aidas.utils.ui_layout import LAYOUT, workspace_sidebar_width


class WorkspaceLayoutTests(unittest.TestCase):
    def test_design_width_keeps_the_standard_thirty_seventy_split(self):
        usable = LAYOUT.design_width - LAYOUT.divider_width
        sidebar = workspace_sidebar_width(LAYOUT.design_width)

        self.assertEqual(sidebar, round(usable * LAYOUT.sidebar_ratio))
        self.assertEqual(usable - sidebar, 892)

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

    def test_invalid_ratio_is_bounded(self):
        width = 1000

        self.assertEqual(
            workspace_sidebar_width(width, ratio=-3, sidebar_minimum=0, content_minimum=0),
            round((width - LAYOUT.divider_width) * 0.10),
        )
        self.assertEqual(
            workspace_sidebar_width(width, ratio=4, sidebar_minimum=0, content_minimum=0),
            round((width - LAYOUT.divider_width) * 0.90),
        )


if __name__ == "__main__":
    unittest.main()
