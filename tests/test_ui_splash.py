from __future__ import annotations

import inspect
import unittest

from aidas.ui.splash import SPLASH_APPEARANCE_MODE, SplashWindow, _splash_color
from aidas.ui.theme import COLOR_PAIRS


class _ValueStub:
    def __init__(self) -> None:
        self.value = None

    def set(self, value) -> None:
        self.value = value


class SplashWindowTests(unittest.TestCase):
    def test_splash_has_one_fixed_light_appearance(self):
        self.assertEqual(SPLASH_APPEARANCE_MODE, "Light")
        self.assertEqual(_splash_color("surface"), COLOR_PAIRS["surface"][0])
        self.assertEqual(_splash_color("text"), COLOR_PAIRS["text"][0])

    def test_splash_outer_edge_is_full_bleed_surface_without_top_accent(self):
        source = inspect.getsource(SplashWindow.__init__)

        self.assertIn('self.configure(fg_color=_splash_color("surface"))', source)
        self.assertIn('panel.pack(fill="both", expand=True)', source)
        self.assertNotIn('fg_color=_splash_color("border_strong")', source)
        self.assertNotIn('panel.pack(fill="both", expand=True, padx=1, pady=1)', source)
        self.assertIn("panel.grid_rowconfigure(0, minsize=", source)

    def test_progress_contract_clamps_and_normalizes_percentage(self):
        splash = object.__new__(SplashWindow)
        splash.percent_var = _ValueStub()
        splash.status_var = _ValueStub()
        splash.progress = _ValueStub()
        splash.update_idletasks = lambda: None

        SplashWindow.set_progress(splash, 125, "Ready")

        self.assertEqual(splash.percent_var.value, "100%")
        self.assertEqual(splash.status_var.value, "Ready")
        self.assertEqual(splash.progress.value, 1.0)


if __name__ == "__main__":
    unittest.main()
