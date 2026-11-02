"""The batch progress log uses the active palette and remains read-only."""

from pathlib import Path
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.steps.step3_flatten import RBatchRunPanel, Step3Frame
from aidas.ui.theme import COLORS, apply_appearance_mode, get_interface_mode, set_interface_mode


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class Step3ProgressThemeTests(unittest.TestCase):
    def setUp(self):
        previous_interface = get_interface_mode()
        previous_appearance = (
            "System" if ctk.AppearanceModeTracker.appearance_mode_set_by == "system"
            else ctk.get_appearance_mode()
        )
        self.addCleanup(ctk.set_appearance_mode, previous_appearance)
        self.addCleanup(set_interface_mode, previous_interface, redraw=False)
        set_interface_mode("Modern", redraw=False)
        ctk.set_appearance_mode("Dark")
        try:
            self.root = _QuietRoot()
        except tk.TclError as error:
            self.skipTest(f"Tk is unavailable: {error}")
        self.addCleanup(self.root.destroy)
        self.root.withdraw()
        self.root.attributes("-alpha", 0)
        self.style = ttk.Style(self.root)
        apply_appearance_mode("Dark", root=self.root, style=self.style)
        step = SimpleNamespace(_normalize_r_output_mode=Step3Frame._normalize_r_output_mode)
        self.folder = Path("example")
        self.panel = RBatchRunPanel(step, self.root, [self.folder], 1, "main.R", "output.R")
        self.panel.pack(fill="both", expand=True)

    def _assert_log_palette(self):
        for option, expected in (
            ("background", COLORS.surface_subtle),
            ("foreground", COLORS.text),
            ("insertbackground", COLORS.text),
            ("selectbackground", COLORS.primary),
            ("selectforeground", COLORS.on_primary),
            ("highlightbackground", COLORS.border_strong),
        ):
            self.assertEqual(self.panel.step_text.cget(option).casefold(), expected.casefold(), option)

    def test_initial_dark_log_is_themed_before_any_later_appearance_refresh(self):
        self._assert_log_palette()
        self.assertEqual(self.panel.step_text.cget("state"), "disabled")
        self.assertIn("Queued...", self.panel.step_text.get("1.0", "end-1c"))

    def test_live_appearance_and_interface_changes_preserve_log_content_and_read_only_state(self):
        self.panel._append_step(self.folder, "Flattening slices", "Done")
        text = self.panel.step_text
        content = text.get("1.0", "end-1c")
        for interface in ("Modern", "Classic", "Modern"):
            set_interface_mode(interface, redraw=False)
            for mode, effective in (("Light", 0), ("Dark", 1), ("System", 0), ("System", 1)):
                with self.subTest(interface=interface, mode=mode, effective=effective):
                    with mock.patch.object(ctk.AppearanceModeTracker, "detect_appearance_mode", return_value=effective):
                        apply_appearance_mode(mode, root=self.root, style=self.style)
                    self._assert_log_palette()
                    self.assertEqual(text.cget("state"), "disabled")
                    self.assertEqual(text.get("1.0", "end-1c"), content)
                    text.insert("end", "This must remain read-only")
                    self.assertEqual(text.get("1.0", "end-1c"), content)


if __name__ == "__main__":
    unittest.main()
