from pathlib import Path
import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.app import AIDaSApp
from aidas.ui.components import AppButton
from aidas.ui.splash import SplashWindow
from aidas.ui.theme import get_interface_mode, refresh_interface_widgets, set_interface_mode


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class ThemeWidgetRefreshTests(unittest.TestCase):
    def setUp(self):
        self.original_interface = get_interface_mode()
        self.original_appearance = ctk.get_appearance_mode()
        self.original_source = ctk.AppearanceModeTracker.appearance_mode_set_by
        set_interface_mode("Modern", redraw=False)
        ctk.set_appearance_mode("Light")
        try:
            self.root = _QuietRoot()
            self.root.withdraw()
            self.root.attributes("-alpha", 0)
        except tk.TclError as error:
            self.skipTest(f"A transparent Tk test window is unavailable: {error}")
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)

    def tearDown(self):
        if getattr(self, "root", None) is not None:
            self.root.destroy()
        set_interface_mode(self.original_interface, redraw=False)
        ctk.set_appearance_mode("System" if self.original_source == "system" else self.original_appearance)

    def _reject_native_corner_queries(self):
        original_cget = tk.Misc.cget

        def guarded_cget(widget, option):
            if option == "corner_radius":
                raise AssertionError(f"Unsupported native corner query: {type(widget).__name__}")
            return original_cget(widget, option)

        return mock.patch.object(tk.Misc, "cget", guarded_cget)

    def test_mixed_tree_avoids_native_queries_and_restores_scrollable_outer_radius(self):
        native = tk.Frame(self.root)
        native.pack()
        for widget_class in (tk.Label, tk.Canvas, ttk.Frame, ttk.Label, ttk.Button, ttk.Entry):
            widget_class(native).pack()
        rounded = ctk.CTkFrame(native, corner_radius=11)
        rounded.pack()
        label = ctk.CTkLabel(rounded, text="Content", corner_radius=5)
        label.pack()
        button = AppButton(rounded, text="Action", corner_radius=13)
        button.pack()
        square = ctk.CTkFrame(rounded, corner_radius=0)
        square.pack()
        bare = ctk.CTkBaseClass(self.root)
        bare.pack()
        scrollable = ctk.CTkScrollableFrame(self.root, corner_radius=17, label_text="Settings")
        scrollable.pack()
        nested = ctk.CTkFrame(scrollable, corner_radius=8)
        nested.pack()
        dialog = ctk.CTkToplevel(self.root)
        dialog.withdraw()
        dialog_content = ctk.CTkFrame(dialog, corner_radius=19)
        dialog_content.pack()
        expected = {
            rounded: 11, label: 5, button: 13, square: 0,
            scrollable._parent_frame: 17, scrollable._label: 17,
            nested: 8, dialog_content: 19,
        }

        with self._reject_native_corner_queries(), mock.patch.object(
            bare, "cget", side_effect=AssertionError("A bare CTk base has no corner option")
        ):
            set_interface_mode("Classic", redraw=False)
            refresh_interface_widgets(self.root)
            for widget in expected:
                self.assertEqual(widget.cget("corner_radius"), 0)
            self.assertNotIn("_aidas_modern_corner_radius", vars(scrollable))
            self.assertNotIn("_aidas_modern_corner_radius", vars(self.root))
            self.assertNotIn("_aidas_modern_corner_radius", vars(dialog))

            set_interface_mode("Modern", redraw=False)
            refresh_interface_widgets(self.root)
            for widget, radius in expected.items():
                self.assertEqual(widget.cget("corner_radius"), radius)
                self.assertNotIn("_aidas_modern_corner_radius", vars(widget))
        self.assertEqual(self.callback_errors, [])

    def test_preserved_buttons_and_nonrecursive_refresh_keep_their_contract(self):
        parent = ctk.CTkFrame(self.root, corner_radius=11)
        child = AppButton(parent, text="Action", corner_radius=7)
        preserved = AppButton(parent, text="Always rounded", corner_radius=9)
        preserved._aidas_preserve_corner_radius = True
        with self._reject_native_corner_queries():
            set_interface_mode("Classic", redraw=False)
            self.assertEqual(refresh_interface_widgets(parent, recursive=False), 1)
            self.assertEqual(parent.cget("corner_radius"), 0)
            self.assertEqual(child.cget("corner_radius"), 7)
            self.assertEqual(refresh_interface_widgets(preserved, recursive=False), 0)
            self.assertEqual(preserved.cget("corner_radius"), 9)
            self.assertEqual(refresh_interface_widgets(self.root, recursive=False), 0)

            set_interface_mode("Modern", redraw=False)
            self.assertEqual(refresh_interface_widgets(parent, recursive=False), 1)
            self.assertEqual(parent.cget("corner_radius"), 11)
            self.assertEqual(child.cget("corner_radius"), 7)
            self.assertEqual(preserved.cget("corner_radius"), 9)

    def test_splash_progress_drains_queued_startup_refresh_without_corner_errors(self):
        with mock.patch.object(SplashWindow, "deiconify"):
            splash = SplashWindow(
                self.root,
                logo_path=str(Path(__file__).resolve().parents[1] / "assets" / "aidas.png"),
                title="AIDaS", subtitle="Retinal Image Processing", affiliation="Research",
                lab_name="AIDaS", copyright_notice="AIDaS",
            )
        splash.attributes("-alpha", 0)
        splash.attributes("-topmost", False)
        splash.withdraw()
        self.root._splash = splash
        self.root._interface_refresh_after_id = None
        set_interface_mode("Classic", redraw=False)

        with self._reject_native_corner_queries():
            AIDaSApp._queue_interface_widget_refresh(self.root, include_splash=True)
            self.assertIsNotNone(self.root._interface_refresh_after_id)
            splash.set_progress(3, "Preparing interface")
            self.root.update_idletasks()

        self.assertIsNone(self.root._interface_refresh_after_id)
        self.assertEqual(splash.percent_var.get(), "3%")
        self.assertEqual(splash.status_var.get(), "Preparing interface")
        self.assertEqual(self.callback_errors, [])


if __name__ == "__main__":
    unittest.main()
