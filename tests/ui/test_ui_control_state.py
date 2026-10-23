"""Regression coverage for debugger-visible unsupported state probes."""

import sys
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock

import customtkinter as ctk

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.steps.step2_annotate import Step2Frame
from aidas.utils.ui_utils import NativeNumericSpinbox, SidebarStepFrame


class ControlStateTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = ctk.CTk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def assert_no_unsupported_state_exceptions(self, callback):
        exceptions = []

        def trace(frame, event, arg):
            if event == "exception":
                _, error, _ = arg
                if isinstance(error, (ValueError, tk.TclError)) and "state" in str(error):
                    exceptions.append(str(error))
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            callback()
        finally:
            sys.settrace(previous)
        self.assertEqual(exceptions, [])

    def test_presentation_widgets_are_skipped_without_raising_even_caught_errors(self):
        widgets = [
            ctk.CTkFrame(self.root),
            ctk.CTkLabel(self.root),
            ctk.CTkProgressBar(self.root),
            ctk.CTkScrollbar(self.root),
            ctk.CTkScrollableFrame(self.root),
            tk.Frame(self.root),
            ttk.Frame(self.root),
        ]

        def toggle():
            for widget in widgets:
                for frame_class in (SidebarStepFrame, Step1Frame):
                    self.assertIsNone(frame_class._control_state(widget))
                    self.assertFalse(frame_class._set_control_enabled(widget, False))
                    self.assertFalse(frame_class._set_control_enabled(widget, True))

        self.assert_no_unsupported_state_exceptions(toggle)

    def test_native_ctk_and_frame_based_controls_still_toggle(self):
        widgets = [
            ctk.CTkButton(self.root),
            ctk.CTkEntry(self.root),
            ttk.Button(self.root),
            NativeNumericSpinbox(self.root, tk.StringVar(self.root, value="1")),
        ]
        for widget in widgets:
            with self.subTest(widget=type(widget).__name__):
                self.assertTrue(SidebarStepFrame._set_control_enabled(widget, False))
                self.assertEqual(str(widget.cget("state")), "disabled")
                self.assertFalse(SidebarStepFrame._set_control_enabled(widget, False))
                self.assertTrue(SidebarStepFrame._set_control_enabled(widget, True))
                self.assertEqual(str(widget.cget("state")), "normal")

    def test_segmentation_recursion_reaches_controls_and_preserves_batch_button(self):
        frame = Step2Frame.__new__(Step2Frame)
        ctk.CTkFrame.__init__(frame, self.root)
        frame.segmentation_frame = ctk.CTkFrame(frame)
        nested = ctk.CTkFrame(frame.segmentation_frame)
        ctk.CTkLabel(nested, text="Segmentation")
        control = ctk.CTkButton(nested)
        frame.batch_ai_button = ctk.CTkButton(nested)
        frame._update_batch_ai_button_state = Mock()
        frame._update_boundary_action_buttons = Mock()

        def toggle():
            frame._set_segmentation_frame_enabled(False)
            self.assertEqual(control.cget("state"), "disabled")
            self.assertEqual(frame.batch_ai_button.cget("state"), "normal")
            frame._set_segmentation_frame_enabled(True)
            self.assertEqual(control.cget("state"), "normal")

        self.assert_no_unsupported_state_exceptions(toggle)


if __name__ == "__main__":
    unittest.main()
