import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock

import customtkinter as ctk

from aidas.steps.step3_flatten import RSetupWizard
from aidas.ui.components import AppButton
from aidas.ui.theme import get_interface_mode, refresh_interface_widgets, set_interface_mode
from aidas.utils.ui_utils import action_button, icon_action_button


class ActionButtonInteractionTests(unittest.TestCase):
    def setUp(self):
        self.previous_interface = get_interface_mode()
        set_interface_mode("Modern", redraw=False)
        class AppRoot(ctk.CTk):
            _deactivate_windows_window_header_manipulation = True
        self.root = AppRoot()
        self.root.geometry("640x360")
        self.root.attributes("-alpha", 0)

    def tearDown(self):
        self.root.destroy()
        set_interface_mode(self.previous_interface, redraw=False)

    def test_keyboard_activation_and_focus_skip_disabled_actions(self):
        command = Mock()
        button = action_button(self.root, self.root, "Browse", command, "folder")
        button.pack()
        entry = ttk.Entry(self.root)
        entry.pack()
        self.root.update()
        self.assertEqual(str(self.root.tk.call("tk_focusNext", str(entry))), str(button))
        self.root.tk.call("focus", "-force", str(button))
        button.event_generate("<Return>")
        self.root.update()
        command.assert_called_once_with()
        button.focus_force()
        button._text_label.event_generate("<space>")
        self.root.update()
        self.assertEqual(command.call_count, 2)
        button.state(["disabled"])
        button._text_label.event_generate("<space>")
        self.root.update()
        self.assertEqual(command.call_count, 2)
        self.assertEqual(str(self.root.tk.call("tk_focusNext", str(entry))), str(entry))

    def test_r_setup_busy_state_disables_rounded_and_native_actions(self):
        content = ttk.Frame(self.root)
        content.pack()
        nested = ttk.Frame(content)
        nested.pack()
        modern = action_button(nested, self.root, "Install", lambda: None, "package")
        modern.pack()
        native = ttk.Button(nested, text="Check")
        native.pack()
        wizard = RSetupWizard.__new__(RSetupWizard)
        wizard.content = content
        for state in ("disabled", "normal"):
            wizard._set_content_buttons(state)
            self.assertEqual(modern.cget("state"), state)
            self.assertEqual(str(native.cget("state")), state)

    def test_retained_actions_round_after_switching_from_classic_to_modern(self):
        set_interface_mode("Classic", redraw=False)
        button = action_button(self.root, self.root, "Save", lambda: None, "save",
                               state="disabled")
        button.pack()
        icon = icon_action_button(self.root, self.root, lambda: None, "folder",
                                  tooltip="Select folder")
        icon.pack()
        self.assertIsInstance(button, AppButton)
        self.assertEqual(button.cget("corner_radius"), 0)
        set_interface_mode("Modern", redraw=False)
        refresh_interface_widgets(self.root)
        self.root.update_idletasks()
        self.assertGreater(button.cget("corner_radius"), 0)
        self.assertEqual(button.cget("state"), "disabled")
        self.assertEqual(icon.winfo_width(), icon.winfo_height())
        self.assertIsNone(icon._text_label)
