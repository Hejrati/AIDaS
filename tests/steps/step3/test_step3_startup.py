import tkinter as tk
import unittest
from unittest import mock

from aidas.steps.step3_flatten import Step3Frame


class Step3StartupTests(unittest.TestCase):
    def test_initial_preview_preserves_the_ready_input_checklist_and_status(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk unavailable: {error}")
        root.withdraw()
        self.addCleanup(root.destroy)

        def valid_inputs(frame):
            frame.info_var.set("Validated input file checklist")
            frame.status_var.set("All inputs ready for processing")
            return {}, []

        original = Step3Frame._render_tutorial
        with mock.patch.object(Step3Frame, "_refresh_input_status", valid_inputs), mock.patch.object(
            Step3Frame, "_render_tutorial", autospec=True, side_effect=original,
        ) as render:
            frame = Step3Frame(root)
        render.assert_called_once_with(frame)
        self.assertEqual(len(frame.plot_holder.winfo_children()), 1)
        self.assertEqual(frame.info_var.get(), "Validated input file checklist")
        self.assertEqual(frame.status_var.get(), "All inputs ready for processing")


if __name__ == "__main__":
    unittest.main()
