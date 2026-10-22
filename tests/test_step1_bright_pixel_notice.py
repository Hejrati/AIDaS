from __future__ import annotations

import tkinter as tk
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import customtkinter as ctk
import numpy as np

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.core.config import Config


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
                self.assertIs(notice.winfo_toplevel(), notice)
                self.assertEqual(str(notice.transient()), str(self.root))
                self.assertIs(self.root.grab_current(), notice)
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
                side_effect=AssertionError("Use the custom modal with Fix and Cancel actions"),
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
            notice._cancel_from_keyboard(None)

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

    def test_ignore_continues_with_original_pixels_without_saving_a_preference(self):
        preferences = Mock()
        preferences.get.return_value = False
        self.frame.preferences = preferences
        input_before = self.incoming.tobytes()

        def ignore(notice):
            notice.ignore_button.invoke()

        self.assertTrue(self._load_with_action(ignore))
        self.assertEqual(self.frame.current_file, "incoming.sdb")
        np.testing.assert_array_equal(self.frame.raw_image, self.incoming)
        np.testing.assert_array_equal(self.frame._source_raw_image, self.incoming)
        self.assertEqual(self.incoming.tobytes(), input_before)
        self.assertFalse(self.frame._source_bright_pixels_smoothed)
        self.assertIsNone(self.root.grab_current())
        preferences.set.assert_not_called()

    def test_close_icon_processes_original_image_without_remembering(self):
        preferences = Mock()
        preferences.get.return_value = False
        self.frame.preferences = preferences

        for checked in (False, True):
            with self.subTest(remember_checked=checked):
                def close(notice):
                    notice.auto_fix_var.set(checked)
                    self.root.tk.call(notice.protocol("WM_DELETE_WINDOW"))

                self.assertTrue(self._load_with_action(close))
                self.assertEqual(self.frame.current_file, "incoming.sdb")
                np.testing.assert_array_equal(self.frame.raw_image, self.incoming)
                self.assertFalse(self.frame._source_bright_pixels_smoothed)
                self.assertIsNone(self.root.grab_current())
                preferences.set.assert_not_called()
        self.assertTrue(self._load_with_action(lambda notice: notice.ignore_button.invoke()))
        preferences.set.assert_not_called()

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

    def test_remembered_fix_persists_and_corrects_future_images_without_a_popup(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Config, "CONFIG_DIR", Path(directory)), patch.object(
                Config, "CONFIG_FILE", Path(directory) / "preferences.json"
            ):
                self.frame.preferences = Config()

                def fix_and_remember(notice):
                    self.assertFalse(notice.auto_fix_var.get())
                    notice.auto_fix_checkbox.toggle()
                    notice.fix_button.invoke()

                self.assertTrue(self._load_with_action(fix_and_remember))
                self.frame.preferences = Config()
                self.assertTrue(self.frame.preferences.get("step1_auto_fix_bright_pixels"))
                with patch(
                    "aidas.steps.step1_resize_raw.BrightPixelNotice",
                    side_effect=AssertionError("Automatic correction must not open a popup"),
                ):
                    self.assertTrue(self.frame._load_image(self.incoming, "next.sdb"))
                np.testing.assert_array_equal(self.frame.raw_image != self.incoming, self.mask)
                self.frame.preferences.set("step1_auto_fix_bright_pixels", False)
                self.frame.preferences.set("step1_bright_pixel_choice", "")
                self.assertFalse(self._load_with_action(lambda notice: notice._cancel_from_keyboard(None)))

    def test_remembered_ignore_persists_and_keeps_future_images_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Config, "CONFIG_DIR", Path(directory)), patch.object(
                Config, "CONFIG_FILE", Path(directory) / "preferences.json"
            ):
                self.frame.preferences = Config()

                def ignore_and_remember(notice):
                    notice.auto_fix_checkbox.toggle()
                    notice.ignore_button.invoke()

                self.assertTrue(self._load_with_action(ignore_and_remember))
                self.frame.preferences = Config()
                self.assertEqual(self.frame.preferences.get("step1_bright_pixel_choice"), "ignore")
                self.assertFalse(self.frame.preferences.get("step1_auto_fix_bright_pixels"))
                with patch(
                    "aidas.steps.step1_resize_raw.BrightPixelNotice",
                    side_effect=AssertionError("Remembered Ignore must not open a popup"),
                ):
                    self.assertTrue(self.frame._load_image(self.incoming, "next.sdb"))
                np.testing.assert_array_equal(self.frame.raw_image, self.incoming)
                self.assertFalse(self.frame._source_bright_pixels_smoothed)
                self.frame.preferences.set("step1_bright_pixel_choice", "")
                self.assertFalse(self._load_with_action(lambda notice: notice._cancel_from_keyboard(None)))

    def test_cancel_does_not_remember_automatic_correction(self):
        preferences = Mock()
        preferences.get.return_value = False
        self.frame.preferences = preferences

        def cancel(notice):
            notice.auto_fix_checkbox.toggle()
            notice._cancel_from_keyboard(None)

        self.assertFalse(self._load_with_action(cancel))
        preferences.set.assert_not_called()

    def test_controls_outside_step1_are_blocked_until_popup_closes(self):
        outside = tk.Button(self.root, text="Other application action")
        outside.place(x=0, y=0)
        self.root.update()
        clicked = Mock()
        outside.bind("<ButtonPress-1>", clicked)
        original_tags = outside.bindtags()
        original_root_tags = self.root.bindtags()
        try:
            def cancel(notice):
                self.assertNotEqual(outside.bindtags(), original_tags)
                self.assertNotEqual(self.root.bindtags(), original_root_tags)
                outside.event_generate("<ButtonPress-1>")
                self.root.update()
                clicked.assert_not_called()
                notice._cancel_from_keyboard(None)

            self.assertFalse(self._load_with_action(cancel))
            self.assertEqual(outside.bindtags(), original_tags)
            self.assertEqual(self.root.bindtags(), original_root_tags)
            self.assertIsNone(self.root.grab_current())
            outside.event_generate("<ButtonPress-1>")
            self.root.update()
            clicked.assert_called_once()
        finally:
            outside.destroy()

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
