"""Exercise inline folder-limit failures with the actual workflow widgets."""

from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.steps import step1_resize_raw as step1
from aidas.steps import step2_annotate as step2
from aidas.steps import step3_flatten as step3
from aidas.steps import step4_analyze_isez as step4


class WorkflowScanLimitUITests(unittest.TestCase):
    def setUp(self):
        self.root = ctk.CTk()
        self.root.attributes("-alpha", 0)
        self.root.geometry("1100x800+10000+10000")
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)
        self.root.update()
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.large = self.base / "large"
        self.large.mkdir()
        for index in range(4):
            (self.large / str(index)).mkdir()
        self.small = self.base / "small"
        self.study = self.small / "study"
        self.study.mkdir(parents=True)
        for folder in (self.large, self.study):
            for name in ("image.sdb", "Light.img", "Light.hdr", "Light_MARKED.img", "Light_MARKED.hdr", "_flat_LIGHT.hdr"):
                (folder / name).touch()
        self.enterContext(mock.patch("aidas.utils.filesystem.MAX_SCAN_SUBFOLDERS", 3))
        self.popup_error = self.enterContext(mock.patch("tkinter.messagebox.showerror"))
        self.popup_warning = self.enterContext(mock.patch("tkinter.messagebox.showwarning"))

    def tearDown(self):
        try:
            self.root.update()
            self.assertFalse(self.callback_errors, self.callback_errors)
            self.assertIsNone(self.root.grab_current())
            self.popup_error.assert_not_called()
            self.popup_warning.assert_not_called()
        finally:
            self.root.destroy()

    def _wait(self, condition):
        deadline = time.monotonic() + 10
        while not condition():
            self.root.update()
            self.assertFalse(self.callback_errors, self.callback_errors)
            self.assertLess(time.monotonic(), deadline, "The folder scan did not finish.")
            time.sleep(0.003)
        self.root.update()

    @staticmethod
    def _visible_text(widget):
        texts = []
        pending = [widget]
        while pending:
            current = pending.pop()
            pending.extend(current.winfo_children())
            if not current.winfo_ismapped():
                continue
            try:
                texts.append(str(current.cget("text")))
            except (tk.TclError, ValueError):
                pass
        return "\n".join(texts)

    def _assert_inline_limit(self, panel, run_button, status):
        self._wait(lambda: "exceeds the limit of 3" in status.get())
        text = self._visible_text(panel)
        self.assertIn("Please select another folder.", text)
        self.assertEqual(panel.rows, [])
        self.assertTrue(panel.table is None or not panel.table.rows)
        self.assertIn("disabled", run_button.state())
        self.assertIsNone(self.root.grab_current())

    def _assert_close_cancels(self, panel_factory, replace_scanner):
        started, finished = threading.Event(), threading.Event()

        def blocked_scan(*_args, cancel_event=None, **_kwargs):
            started.set()
            try:
                self.assertIsNotNone(cancel_event)
                cancel_event.wait(5)
                return [], 0, 0, []
            finally:
                finished.set()

        with replace_scanner(blocked_scan):
            panel = panel_factory()
            panel.pack(fill="both", expand=True)
            self._wait(started.is_set)
            cancelled = panel._scan_cancel
            panel.destroy()
            self.assertTrue(cancelled.is_set())
            self._wait(finished.is_set)
            pending = panel.__dict__.get("_scan_after_id", panel.__dict__.get("_scan_poll_after_id"))
            self.assertIsNone(pending)
            self.assertTrue(panel._scan_events.empty())

    def test_step1_rejects_partial_queue_recovers_and_cancels_on_close(self):
        with mock.patch.object(step1.Step1Frame, "refresh_sdb_list"):
            frame = step1.Step1Frame(self.root, on_batch_segment_folders=lambda _folders: None)
        frame.pack(fill="both", expand=True)
        frame.set_sdb_directory(str(self.large))
        frame.refresh_sdb_list(preview_first=False)
        self._wait(lambda: not frame._sdb_scan_pending)
        self.assertEqual(frame._sdb_directory_files, {})
        self.assertEqual(frame.sdb_directory_listbox.size(), 0)
        self.assertIn("Please select another folder.", frame.sdb_scan_more_label.cget("text"))
        self.assertIn("exceeds the limit of 3", frame.status_var.get())
        self.assertIn("disabled", frame.batch_segment_cropped_btn.state())
        frame.set_sdb_directory(str(self.small))
        frame.refresh_sdb_list(preview_first=False)
        self._wait(lambda: not frame._sdb_scan_pending)
        self.assertIsNone(frame._sdb_scan_error)
        self.assertEqual(len(frame._sdb_directory_files), 1)
        self.assertEqual(frame.sdb_directory_listbox.size(), 1)
        self.assertNotIn("exceeds", frame.status_var.get())

        started, finished = threading.Event(), threading.Event()

        def blocked_scan(_root, cancelled):
            started.set()
            try:
                cancelled.wait(5)
                return {}, set(), []
            finally:
                finished.set()

        with mock.patch.object(step1.Step1Frame, "_scan_sdb_directory", side_effect=blocked_scan):
            frame.refresh_sdb_list(preview_first=False)
            self._wait(started.is_set)
            cancelled = frame._sdb_scan_cancel
            frame.destroy()
            self.assertTrue(cancelled.is_set())
            self.assertIsNone(frame._sdb_scan_after_id)
            self._wait(finished.is_set)

    def test_step2_inline_limit_allows_reselection_and_close_cancels_worker(self):
        # Step 2's recovered input must still need segmentation.
        (self.study / "Light_MARKED.img").unlink()
        (self.study / "Light_MARKED.hdr").unlink()
        scanner = object.__new__(step2.Step2Frame)
        step = SimpleNamespace(
            status_var=tk.StringVar(master=self.root),
            _scan_step2_batch_segmentation_folders=scanner._scan_step2_batch_segmentation_folders,
        )
        panel = step2.Step2BatchSegmentationSelectionPanel(step, self.root, self.large)
        panel.pack(fill="both", expand=True)
        self._assert_inline_limit(panel, panel.next_button, step.status_var)
        with mock.patch.object(step2.filedialog, "askdirectory", return_value=str(self.small)):
            panel._select_another_folder()
        self._wait(lambda: panel._scan_ready)
        self.assertEqual(len(panel.rows), 1)
        self.assertNotIn("disabled", panel.next_button.state())
        self.assertNotIn("exceeds", self._visible_text(panel))
        panel.destroy()
        self._assert_close_cancels(
            lambda: step2.Step2BatchSegmentationSelectionPanel(step, self.root, self.small),
            lambda callback: mock.patch.object(step, "_scan_step2_batch_segmentation_folders", side_effect=callback),
        )

    def test_step3_inline_limit_allows_reselection_and_close_cancels_worker(self):
        scanner = object.__new__(step3.Step3Frame)
        step = SimpleNamespace(
            status_var=tk.StringVar(master=self.root),
            _available_r_worker_limit=lambda: 4,
            _step2_core_allocation=lambda: 0,
            R_OUTPUT_MODE_PARALLEL="parallel", R_OUTPUT_MODE_SEQUENTIAL="sequential",
            REQUIRED_INPUTS=scanner.REQUIRED_INPUTS,
            _find_input_paths=scanner._find_input_paths,
            _folder_has_r_data=scanner._folder_has_r_data,
        )
        panel = step3.RBatchSelectionPanel(step, self.root, self.large)
        panel.pack(fill="both", expand=True)
        self._assert_inline_limit(panel, panel.next_button, step.status_var)
        with mock.patch.object(step3.filedialog, "askdirectory", return_value=str(self.small)):
            panel._select_another_folder()
        self._wait(lambda: panel._scan_ready)
        self.assertEqual(len(panel.rows), 1)
        self.assertNotIn("disabled", panel.next_button.state())
        self.assertNotIn("exceeds", self._visible_text(panel))
        panel.destroy()

        def replace_collector(callback):
            def collect(*args, **kwargs):
                callback(*args, **kwargs)
                return [], 0, []
            return mock.patch.object(step3, "find_candidate_directories", side_effect=collect)

        self._assert_close_cancels(
            lambda: step3.RBatchSelectionPanel(step, self.root, self.small), replace_collector,
        )

    def test_step4_inline_limit_recovers_when_reopened_and_close_cancels_worker(self):
        scanner = object.__new__(step4.Step4Frame)
        step = SimpleNamespace(
            status_var=tk.StringVar(master=self.root),
            _scan_batch_roi_folders=scanner._scan_batch_roi_folders,
        )
        panel = step4.Step4BatchROISelectionPanel(step, self.root, self.large)
        panel.pack(fill="both", expand=True)
        self._assert_inline_limit(panel, panel.process_button, step.status_var)
        panel.destroy()
        panel = step4.Step4BatchROISelectionPanel(step, self.root, self.small)
        panel.pack(fill="both", expand=True)
        self._wait(lambda: panel.table is not None)
        self.assertEqual(len(panel.rows), 1)
        self.assertNotIn("disabled", panel.process_button.state())
        self.assertNotIn("exceeds", self._visible_text(panel))
        panel.destroy()
        self._assert_close_cancels(
            lambda: step4.Step4BatchROISelectionPanel(step, self.root, self.small),
            lambda callback: mock.patch.object(step, "_scan_batch_roi_folders", side_effect=callback),
        )


if __name__ == "__main__":
    unittest.main()
