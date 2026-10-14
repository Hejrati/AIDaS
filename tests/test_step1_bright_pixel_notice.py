from __future__ import annotations

import tkinter as tk
import unittest
from unittest.mock import patch

import customtkinter as ctk
import numpy as np

from aidas.steps.step1_resize_raw import Step1Frame


class Step1BrightPixelNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = None
        cls.callback_errors = []
        try:
            cls.root = ctk.CTk()
            cls.root.withdraw()
            cls.root.attributes("-alpha", 0)
            cls.root.geometry("1280x820+10000+10000")
            cls.root.deiconify()
        except tk.TclError as exc:
            if cls.root is not None:
                cls.root.destroy()
                cls.root = None
            raise unittest.SkipTest(f"An invisible Tk test window is unavailable: {exc}") from exc
        cls.root.report_callback_exception = lambda *error: cls.callback_errors.append(error)

    @classmethod
    def tearDownClass(cls):
        if cls.root is not None:
            cls.root.destroy()
            cls.root = None

    def setUp(self):
        self.root = type(self).root
        self.scheduled = []
        with patch.object(Step1Frame, "refresh_sdb_list"):
            self.frame = Step1Frame(self.root)
        self.assertNotIn("bright_pixel_status_label", self.frame.__dict__)
        self.frame.pack(fill="both", expand=True)
        self.root.update()
        previous = np.full((32, 32), 46000, dtype=np.uint16)
        self.assertTrue(self.frame._load_image(previous, "previous.sdb"))
        self.frame._set_roi_and_entries(3, 4, 10, 12)
        self.frame.image_canvas.set_zoom(1.7)
        self.root.update()

        self.incoming = np.full((32, 32), 45000, dtype=np.uint16)
        self.incoming[12, 14] = 32005
        self.incoming.setflags(write=False)
        self.mask = np.zeros(self.incoming.shape, dtype=bool)
        self.mask[12, 14] = True

    def tearDown(self):
        self._cancel_scheduled()
        if self.root is not None:
            self.frame.destroy()
        self.assertEqual(self.callback_errors, [], "The Tk callback must finish without errors")

    def _cancel_scheduled(self):
        for callback in self.scheduled:
            try:
                if self.root is not None:
                    self.root.after_cancel(callback)
            except tk.TclError:
                pass
        self.scheduled.clear()

    def _load_with_action(self, action):
        answered = []
        timed_out = []

        def answer():
            try:
                notice = self.frame._bright_pixel_notice
                self.assertIsNotNone(notice)
                viewport = self.frame.image_canvas.canvas
                self.assertIs(notice.master, viewport)
                self.assertIs(notice.winfo_toplevel(), self.root)
                self.assertEqual(notice.winfo_manager(), "place")
                self.assertEqual(notice.winfo_rootx(), viewport.winfo_rootx())
                self.assertEqual(notice.winfo_rooty(), viewport.winfo_rooty())
                self.assertEqual(notice.winfo_width(), viewport.winfo_width())
                self.assertIsNone(self.root.grab_current())
                self.assertTrue(self.frame._bright_pixel_notice_pending)
                np.testing.assert_array_equal(self.frame.image_canvas.get_image(), self.incoming)
                action(notice)
                answered.append(True)
            except BaseException:
                self.frame._dismiss_bright_pixel_notice()
                raise

        def timeout():
            timed_out.append(True)
            self.frame._dismiss_bright_pixel_notice()
            decision = self.frame.__dict__.get("_bright_pixel_decision")
            if decision is not None:
                decision.set(2)

        self.scheduled.extend((self.root.after(500, answer), self.root.after(2500, timeout)))
        try:
            with patch(
                "aidas.steps.step1_resize_raw.messagebox.askyesnocancel",
                side_effect=AssertionError("The embedded notice must not open a native dialog"),
            ):
                result = self.frame._load_image(self.incoming, "incoming.sdb")
        finally:
            self._cancel_scheduled()
        self.assertEqual(timed_out, [], "The notice must resolve through the scheduled action")
        self.assertEqual(answered, [True])
        self.assertEqual(self.callback_errors, [])
        self.assertFalse(self.frame._bright_pixel_notice_pending)
        self.assertIsNone(self.frame._bright_pixel_notice)
        return result

    def test_cancel_restores_previous_canvas_and_source_state(self):
        frame = self.frame
        previous_canvas = frame.image_canvas.get_image()
        previous_raw = frame.raw_image
        previous_source = frame._source_raw_image
        previous_working = frame._working_source_raw_image
        previous_mask = frame._source_bright_pixel_mask
        previous_roi = frame._source_roi
        previous_canvas_roi = frame.image_canvas.get_roi()
        previous_zoom = frame.image_canvas.get_zoom()
        previous_crop_tags = frame.crop_btn.bindtags()
        previous_canvas_tags = frame.image_canvas.canvas.bindtags()
        input_before = self.incoming.tobytes()

        def cancel(notice):
            self.assertFalse(frame._set_roi_and_entries(0, 0, 1, 1))
            self.assertEqual(frame._source_roi, previous_roi)
            notice.cancel_button.invoke()

        self.assertFalse(self._load_with_action(cancel))
        self.assertIs(frame.image_canvas.get_image(), previous_canvas)
        self.assertIs(frame.raw_image, previous_raw)
        self.assertIs(frame._source_raw_image, previous_source)
        self.assertIs(frame._working_source_raw_image, previous_working)
        self.assertIs(frame._source_bright_pixel_mask, previous_mask)
        self.assertEqual(frame.current_file, "previous.sdb")
        self.assertEqual(frame._source_roi, previous_roi)
        self.assertEqual(frame.image_canvas.get_roi(), previous_canvas_roi)
        self.assertEqual(frame.image_canvas.get_zoom(), previous_zoom)
        self.assertEqual(frame.crop_btn.bindtags(), previous_crop_tags)
        self.assertEqual(frame.image_canvas.canvas.bindtags(), previous_canvas_tags)
        self.assertEqual(self.incoming.tobytes(), input_before)

    def test_real_fix_button_changes_only_artifacts_and_preserves_original_input(self):
        input_before = self.incoming.tobytes()
        crop_tags = self.frame.crop_btn.bindtags()
        self.assertTrue(self._load_with_action(lambda notice: notice.fix_button.invoke()))
        np.testing.assert_array_equal(self.frame._source_raw_image, self.incoming)
        np.testing.assert_array_equal(self.frame.raw_image != self.incoming, self.mask)
        self.assertEqual(
            self.frame.raw_image[~self.mask].tobytes(), self.incoming[~self.mask].tobytes()
        )
        self.assertEqual(self.frame.raw_image[12, 14], 45000)
        self.assertEqual(self.incoming.tobytes(), input_before)
        self.assertFalse(self.incoming.flags.writeable)
        self.assertEqual(self.frame.crop_btn.bindtags(), crop_tags)
        self.assertIsNone(self.root.grab_current())

    def test_step1_destruction_resolves_cancel_and_allows_clean_root_close(self):
        previous_raw = self.frame.raw_image
        input_before = self.incoming.tobytes()
        self.assertFalse(self._load_with_action(lambda _notice: self.frame.destroy()))
        self.assertEqual(self.frame.current_file, "previous.sdb")
        self.assertIs(self.frame.raw_image, previous_raw)
        self.assertEqual(self.incoming.tobytes(), input_before)
        # Closing the root also deletes registered class-binding callbacks.
        # A destroyed Step 1 must not leave stale callback names behind.
        self.root.destroy()
        self.root = None
        type(self).root = None
        self.assertEqual(self.callback_errors, [])


if __name__ == "__main__":
    unittest.main()
