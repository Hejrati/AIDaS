from __future__ import annotations


import inspect
import tkinter as tk
import unittest
from unittest.mock import Mock

import customtkinter as ctk

from aidas.app import AIDaSApp
from aidas.ui.splash import SPLASH_APPEARANCE_MODE, SplashWindow, _splash_color
from aidas.ui.theme import COLOR_PAIRS

from tests.support.paths import PROJECT_ROOT


class _ValueStub:
    def __init__(self) -> None:
        self.value = None

    def set(self, value) -> None:
        self.value = value


class SplashWindowTests(unittest.TestCase):
    def test_appearance_callbacks_do_not_hide_the_visible_splash(self):
        class QuietRoot(ctk.CTk):
            _deactivate_windows_window_header_manipulation = True

        try:
            root = QuietRoot()
        except tk.TclError as error:
            self.skipTest(f"Tk unavailable: {error}")
        root.withdraw()
        self.addCleanup(root.destroy)
        appearance = ctk.get_appearance_mode()
        self.addCleanup(ctk.set_appearance_mode, appearance)
        splash = SplashWindow(
            root,
            logo_path=str(PROJECT_ROOT / "assets" / "aidas.png"),
            title="AIDaS",
            subtitle="OCT Image Processing",
            affiliation="Wayne State University",
            lab_name="MVPRL",
            copyright_notice="AIDaS",
        )
        root.update()
        self.assertTrue(splash.winfo_viewable())
        unmapped = []

        def on_unmap(event):
            if event.widget is splash:
                unmapped.append(splash.percent_var.get())

        splash.bind("<Unmap>", on_unmap, add="+")
        for mode in ("Dark", "Light", "System"):
            with self.subTest(mode=mode):
                splash.set_progress(54, "Applying interface theme...")
                ctk.set_appearance_mode(mode)
                ctk.AppearanceModeTracker.update_callbacks()
                root.update()
                self.assertEqual(unmapped, [])
                self.assertTrue(splash.winfo_viewable())
                self.assertEqual(splash.cget("fg_color"), _splash_color("surface"))
                self.assertEqual(splash.status_var.get(), "Applying interface theme...")

    def test_startup_progress_paints_and_processes_window_events(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app._splash = Mock()
        app._splash.winfo_exists.return_value = True
        app.update_idletasks = Mock()
        app.update = Mock()

        app._set_splash_progress(30, "Loading tools...")

        app._splash.set_progress.assert_called_once_with(30, "Loading tools...")
        app.update_idletasks.assert_not_called()
        app.update.assert_called_once_with()

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

    def test_loading_region_blends_into_the_splash_surface(self):
        source = inspect.getsource(SplashWindow.__init__)
        loading_source = source[source.index("loading_region = ctk.CTkFrame(") :]
        loading_source = loading_source.split("loading_region.grid(", 1)[0]

        self.assertIn('fg_color="transparent"', loading_source)
        self.assertIn("corner_radius=0", loading_source)
        self.assertIn("border_width=0", loading_source)
        self.assertNotIn('surface_subtle', loading_source)
        self.assertNotIn('border_color=', loading_source)

    def test_classic_splash_uses_a_flat_progress_track(self):
        source = inspect.getsource(SplashWindow.__init__)

        self.assertIn(
            'corner_radius=0 if get_interface_mode() == "Classic" else spacing(4)',
            source,
        )

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
