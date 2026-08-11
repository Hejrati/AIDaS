from __future__ import annotations

import unittest
import tkinter as tk

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.utils.ui_utils import NativeNumericSpinbox


class _ControlStub:
    def __init__(self, state="normal"):
        self.state = state
        self.configure_calls = []

    def cget(self, name):
        if name != "state":
            raise KeyError(name)
        return self.state

    def configure(self, **options):
        self.configure_calls.append(dict(options))
        if "state" in options:
            self.state = options["state"]


class _UnexpectedContainerTraversal:
    def winfo_children(self):
        raise AssertionError("semantic state updates must not traverse widget internals")


class _VariableStub:
    def __init__(self, value=None, *, error=None):
        self.value = value
        self.error = error

    def get(self):
        if self.error is not None:
            raise self.error
        return self.value

    def set(self, value):
        self.error = None
        self.value = value


class Step1RedrawStateTests(unittest.TestCase):
    def test_control_state_update_is_idempotent(self):
        control = _ControlStub("normal")

        self.assertFalse(Step1Frame._set_control_enabled(control, True))
        self.assertTrue(Step1Frame._set_control_enabled(control, False))
        self.assertFalse(Step1Frame._set_control_enabled(control, False))

        self.assertEqual(control.configure_calls, [{"state": "disabled"}])

    def test_sdb_parameter_toggle_uses_only_semantic_controls(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.sdb_params_frame = _UnexpectedContainerTraversal()
        names = (
            "width_stepper",
            "width_reset_btn",
            "height_stepper",
            "height_reset_btn",
            "offset_stepper",
            "offset_reset_btn",
            "endian_checkbox",
        )
        controls = []
        for name in names:
            control = _ControlStub("normal")
            setattr(frame, name, control)
            controls.append(control)

        frame._set_sdb_parameters_enabled(False)
        frame._set_sdb_parameters_enabled(False)

        for control in controls:
            self.assertEqual(control.configure_calls, [{"state": "disabled"}])

    def test_save_state_refresh_only_configures_changed_controls(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.raw_image = object()
        frame.processed_image = None
        frame.save_all_btn = _ControlStub("disabled")
        frame.undo_crop_btn = _ControlStub("disabled")
        frame.crop_btn = _ControlStub("normal")
        frame.crop_options_btn = _ControlStub("normal")
        frame.target_view_radio = _ControlStub("normal")
        frame.source_view_radio = _ControlStub("normal")

        frame._update_save_button_state()
        self.assertFalse(any(
            control.configure_calls
            for control in (
                frame.save_all_btn,
                frame.undo_crop_btn,
                frame.crop_btn,
                frame.crop_options_btn,
                frame.target_view_radio,
                frame.source_view_radio,
            )
        ))

        frame.processed_image = object()
        frame._update_save_button_state()
        frame._update_save_button_state()

        self.assertEqual(frame.save_all_btn.configure_calls, [{"state": "normal"}])
        self.assertEqual(frame.undo_crop_btn.configure_calls, [{"state": "normal"}])
        self.assertEqual(frame.crop_btn.configure_calls, [{"state": "disabled"}])
        self.assertEqual(frame.crop_options_btn.configure_calls, [{"state": "disabled"}])
        self.assertEqual(frame.source_view_radio.configure_calls, [{"state": "disabled"}])
        self.assertEqual(frame.target_view_radio.configure_calls, [])

    def test_numeric_spinbox_does_not_reconfigure_same_state(self):
        spinbox = object.__new__(NativeNumericSpinbox)
        spinbox._button_state = "normal"
        spinbox.entry = _ControlStub("normal")
        spinbox.up_button = _ControlStub("normal")
        spinbox.down_button = _ControlStub("normal")

        spinbox.configure(state="normal")
        spinbox.configure(state="disabled")
        spinbox.configure(state="disabled")

        self.assertEqual(spinbox.cget("state"), "disabled")
        self.assertEqual(spinbox.entry.configure_calls, [{"state": "disabled"}])
        self.assertEqual(spinbox.up_button.configure_calls, [{"state": "disabled"}])
        self.assertEqual(spinbox.down_button.configure_calls, [{"state": "disabled"}])

    def test_numeric_spinbox_dynamic_maximum_clamps_the_current_value(self):
        spinbox = object.__new__(NativeNumericSpinbox)
        spinbox.minimum = 1
        spinbox.maximum = 10
        spinbox.var = _VariableStub("9")

        spinbox.configure(maximum=3)

        self.assertEqual(spinbox.maximum, 3)
        self.assertEqual(spinbox.var.get(), "3")

    def test_numeric_spinbox_arrows_recover_from_invalid_typed_text(self):
        spinbox = object.__new__(NativeNumericSpinbox)
        spinbox.minimum = 1
        spinbox.maximum = 10
        spinbox.var = _VariableStub(error=tk.TclError("invalid number"))

        spinbox._step(1)

        self.assertEqual(spinbox.var.get(), "1")


if __name__ == "__main__":
    unittest.main()
