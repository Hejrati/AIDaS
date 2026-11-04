"""Closing Step 1 must discard queued cursor-status work."""

import tkinter as tk
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.steps.step1_resize_raw import Step1Frame


class Step1TimerCleanupTests(unittest.TestCase):
    def test_destroy_cancels_pending_mouse_status_and_discards_sample(self):
        try:
            root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        root.withdraw()
        try:
            with mock.patch.object(Step1Frame, "refresh_sdb_list"):
                frame = Step1Frame(root)
            frame._on_mouse_moved(1, 2, 3)
            frame._on_mouse_moved(10, 11, 12)
            pending = frame._mouse_status_after_id
            self.assertIn(pending, root.tk.call("after", "info"))
            self.assertEqual(frame._pending_mouse_status, (10, 11, 12))

            frame.destroy()

            self.assertNotIn(pending, root.tk.call("after", "info"))
            self.assertIsNone(frame._mouse_status_after_id)
            self.assertIsNone(frame._pending_mouse_status)
        finally:
            # This interpreter belongs to the test. Remove construction-time
            # callbacks before disposing it, including on the original defect.
            for pending in root.tk.call("after", "info"):
                root.after_cancel(pending)
            root.destroy()


if __name__ == "__main__":
    unittest.main()
