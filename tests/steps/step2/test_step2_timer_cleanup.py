"""Fovea-repeat ownership and unchanged live nudge behavior."""

import tkinter as tk
import unittest
from unittest import mock

import customtkinter as ctk
import numpy as np

from aidas.steps.step2_annotate import FOVEA_BOUNDARY_NAME, Step2Frame
from tests.support.annotation import make_annotation_frame


class Step2TimerCleanupTests(unittest.TestCase):
    def test_destroy_cancels_a_held_fovea_nudge(self):
        try:
            root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        root.withdraw()
        try:
            with mock.patch.object(Step2Frame, "_start_ai_device_probe"):
                frame = Step2Frame(root, preferences={"show_step2_annotation_tutorial": False})
            frame.image_data = np.zeros((8, 60), dtype=np.uint16)
            frame.image_canvas.set_image(frame.image_data)
            frame._ensure_fovea_line(30)
            frame._start_fovea_repeat(1)
            pending = frame._fovea_repeat_job
            self.assertEqual(frame.fovea_x, 31)
            self.assertEqual(frame.fovea_x_entry_var.get(), "31")
            self.assertIn(pending, root.tk.call("after", "info"))

            frame.destroy()

            self.assertNotIn(pending, root.tk.call("after", "info"))
            self.assertIsNone(frame._fovea_repeat_job)
            self.assertEqual(frame._fovea_repeat_dir, 0)
            self.assertEqual(frame._fovea_repeat_ticks, 0)
        finally:
            for pending in root.tk.call("after", "info"):
                root.after_cancel(pending)
            root.destroy()

    def test_stopping_repeat_is_safe_before_repeat_state_is_initialized(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.after_cancel = mock.Mock()

        frame._stop_fovea_repeat()

        frame.after_cancel.assert_not_called()
        self.assertIsNone(frame._fovea_repeat_job)
        self.assertEqual(frame._fovea_repeat_dir, 0)
        self.assertEqual(frame._fovea_repeat_ticks, 0)

    def test_live_repeat_retains_original_steps_intervals_trace_and_pixels(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                frame = make_annotation_frame()
                frame.image_data = np.arange(1600, dtype=np.uint16).reshape(8, 200)
                original = frame.image_data.tobytes()
                frame.fovea_x = 50
                frame._fovea_repeat_job = None
                frame._fovea_repeat_dir = 0
                frame._fovea_repeat_ticks = 0

                frame._start_fovea_repeat(direction)
                self.assertEqual(frame.fovea_x, 50 + direction)
                for _ in range(14):
                    frame._run_fovea_repeat()

                expected = 50 + direction * 38
                self.assertEqual(frame.fovea_x, expected)
                self.assertEqual(frame.fovea_x_entry_var.get(), str(expected))
                self.assertEqual(
                    frame.boundary_traces[FOVEA_BOUNDARY_NAME],
                    frame._vertical_line_trace(expected, 8),
                )
                self.assertEqual(frame.image_data.tobytes(), original)
                self.assertEqual(
                    [call.args[0] for call in frame.after.call_args_list],
                    [180, *([90] * 4), *([55] * 9), 35],
                )
                frame._stop_fovea_repeat()
                frame.after_cancel.assert_called_once_with("annotation-poll")
                self.assertIsNone(frame._fovea_repeat_job)


if __name__ == "__main__":
    unittest.main()
