from __future__ import annotations

import sys
import tkinter as tk
import unittest

import customtkinter as ctk

from aidas.ui.title_bar import install_dialog_title_bar
from aidas.ui.theme import COLOR_PAIRS


@unittest.skipUnless(sys.platform.startswith("win"), "Windows caption integration")
class DialogTitleBarTests(unittest.TestCase):
    def setUp(self):
        class AppRoot(ctk.CTk):
            _deactivate_windows_window_header_manipulation = True

        self.root = AppRoot()
        self.root.interface_mode = "Modern"
        self.root.withdraw()

    def tearDown(self):
        self.root.destroy()

    def test_packed_dialog_uses_shared_caption_without_window_controls(self):
        dialog = tk.Toplevel(self.root)
        dialog.resizable(False, False)
        body = tk.Frame(dialog)
        body.pack(fill="both", expand=True)
        bar = install_dialog_title_bar(dialog)
        self.assertIsNotNone(bar)
        self.assertEqual(dialog.pack_slaves(), [bar, body])
        self.assertEqual(bar.cget("fg_color"), COLOR_PAIRS["window_chrome"])
        self.assertFalse(bar.minimize_button.winfo_manager())
        self.assertFalse(bar.maximize_button.winfo_manager())
        self.assertIs(install_dialog_title_bar(dialog), bar)
        self.assertFalse(dialog.resizable()[0])

    def test_grid_dialog_reserves_caption_row_and_retains_body_weight(self):
        dialog = tk.Toplevel(self.root)
        dialog.rowconfigure(0, weight=1)
        body = tk.Frame(dialog)
        body.grid(row=0, column=0, sticky="nsew")
        bar = install_dialog_title_bar(dialog)
        self.assertIsNotNone(bar)
        self.assertEqual(int(body.grid_info()["row"]), 1)
        self.assertEqual(dialog.grid_rowconfigure(1)["weight"], 1)
        self.assertEqual(dialog.grid_rowconfigure(0)["weight"], 0)

    def test_close_dispatches_existing_protocol_handler(self):
        dialog = tk.Toplevel(self.root)
        calls = []
        dialog.protocol("WM_DELETE_WINDOW", lambda: calls.append("ignore"))
        bar = install_dialog_title_bar(dialog)
        bar.close_button.invoke()
        self.root.update()
        self.assertEqual(calls, ["ignore"])
        self.assertTrue(dialog.winfo_exists())

    def test_caption_survives_mapping_and_tracks_light_and_dark_modes(self):
        self.root.deiconify()
        dialog = ctk.CTkToplevel(self.root)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        bar = install_dialog_title_bar(dialog)
        self.root.update()
        from aidas.ui.title_bar import _WS_CAPTION
        self.assertEqual(bar.controller._api.get_style(bar.controller.handle) & _WS_CAPTION, 0)
        previous = ctk.get_appearance_mode()
        try:
            for mode in ("Light", "Dark"):
                ctk.set_appearance_mode(mode)
                self.root.update()
                self.assertEqual(bar._apply_appearance_mode(bar.cget("fg_color")),
                                 bar._apply_appearance_mode(COLOR_PAIRS["window_chrome"]))
        finally:
            ctk.set_appearance_mode(previous)

    def _settle_events(self, milliseconds=80):
        finished = tk.BooleanVar(master=self.root, value=False)
        self.root.after(milliseconds, lambda: finished.set(True))
        self.root.wait_variable(finished)

    def test_pointer_motion_does_not_schedule_native_header_redraws(self):
        from unittest.mock import patch

        self.root.deiconify()
        dialog = ctk.CTkToplevel(self.root)
        dialog.geometry("600x400")
        bar = install_dialog_title_bar(dialog)
        self._settle_events()
        with patch.object(dialog, "_windows_set_titlebar_color") as native_redraw:
            for offset in range(30):
                bar._resize_edge_at(dialog.winfo_rootx() + 40 + offset,
                                    dialog.winfo_rooty() + 100)
            self._settle_events()
            native_redraw.assert_not_called()



    def test_undecorated_overlays_are_excluded(self):
        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        self.assertIsNone(install_dialog_title_bar(overlay))
