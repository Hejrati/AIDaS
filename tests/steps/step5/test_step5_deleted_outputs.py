"""Deleted and rebuilt workflow results must not reuse stale Step 5 state."""

from pathlib import Path
import shutil
import tempfile
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk
from openpyxl import Workbook, load_workbook

from aidas.steps.step5_compile import DEFAULT_OUTPUT_FILENAME, Step5Frame


class Step5DeletedOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        try:
            self.root = ctk.CTk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.addCleanup(self.root.destroy)
        self.root.attributes("-alpha", 0)
        self.root.geometry("1100x760+10000+10000")
        self.errors = []
        self.root.report_callback_exception = lambda *error: self.errors.append(error)
        self.invalidated = mock.Mock()
        self.root._on_workflow_invalidated = self.invalidated
        self.completed = mock.Mock()
        self.frame = Step5Frame(self.root, on_compilation_complete=self.completed)
        self.frame.pack(fill="both", expand=True)
        self.root.update()
        patch = mock.patch("aidas.steps.step5_compile.messagebox")
        self.popups = patch.start()
        self.addCleanup(patch.stop)

    def _wait(self, condition):
        deadline = time.monotonic() + 12
        while not condition():
            self.assertLess(time.monotonic(), deadline, "Step 5 work did not finish")
            self.root.update()
            time.sleep(.005)
        self.root.update()
        self.assertFalse(self.errors)

    def _ready(self, directory):
        self.frame.set_input_folder(directory)
        self._wait(lambda: self.frame._input_scan_state == "ready")

    def _write_measurement(self, subject, value):
        for side in ("nasal", "temporal"):
            folder = subject / side
            folder.mkdir(parents=True, exist_ok=True)
            workbook = Workbook()
            try:
                workbook.active.cell(1, 7, "Round")
                workbook.active.cell(2, 7, value)
                workbook.save(folder / "rr_MCPAR.xlsx")
            finally:
                workbook.close()

    def test_returning_to_step5_clears_deleted_excel_notice_without_recreating_folders(self):
        source = self.directory / "scan.sdb"
        source.write_bytes(b"original source retained")
        self._write_measurement(self.directory, 11)
        self._ready(self.directory)
        output = self.directory / DEFAULT_OUTPUT_FILENAME
        output.touch()
        self.frame._show_completion_notice(output)
        self.root.update()
        self.assertTrue(self.frame.completion_notice.winfo_ismapped())
        for side in ("nasal", "temporal"):
            shutil.rmtree(self.directory / side)
        output.unlink()

        with mock.patch.object(self.frame, "_start_compiler") as start:
            self.frame.on_show()
            self._wait(lambda: self.frame._input_scan_state == "ready")
            start.assert_not_called()

        self.assertIsNone(self.frame._completed_output_path)
        self.assertFalse(self.frame.completion_notice.winfo_ismapped())
        self.assertIn("no longer available", self.frame.status_var.get())
        self.assertEqual(list(self.directory.iterdir()), [source])
        self.assertEqual(source.read_bytes(), b"original source retained")
        self.invalidated.assert_called_once_with(5)
        self.assertEqual(self.popups.mock_calls, [])

    def test_tree_refresh_also_clears_deleted_completion_notice(self):
        self._ready(self.directory)
        output = self.directory / DEFAULT_OUTPUT_FILENAME
        output.touch()
        self.frame._show_completion_notice(output)
        output.unlink()

        self.frame.directory_view.refresh()
        self._wait(lambda: self.frame._input_scan_state == "ready")

        self.assertIsNone(self.frame._completed_output_path)
        self.assertIn("no longer available", self.frame.status_var.get())
        self.assertFalse(output.exists())
        self.completed.assert_not_called()

    def test_deleted_input_root_invalidates_ready_tree_and_recovers_when_rebuilt(self):
        subject = self.directory / "Subject"
        self._write_measurement(subject, 12)
        self._ready(subject)
        self.assertTrue(self.frame.directory_view._rows)
        shutil.rmtree(subject)

        with mock.patch.object(self.frame, "_start_compiler") as start:
            self.frame.on_show()
            self._wait(lambda: self.frame._input_scan_state == "error")
            self.assertIsNone(self.frame.directory_view._inventory)
            self.assertEqual(self.frame.directory_view._rows, [])
            self.assertIn("disabled", self.frame.run_button.state())
            self.assertFalse(subject.exists())
            self._write_measurement(subject, 27)
            self.frame.on_show()
            self._wait(lambda: self.frame._input_scan_state == "ready")
            self.assertNotIn("disabled", self.frame.run_button.state())
            self.assertEqual(self.frame.directory_view._inventory.workbooks, 2)
            start.assert_not_called()
        self.assertEqual(self.popups.mock_calls, [])

    def test_output_deleted_before_completion_event_never_marks_workflow_complete(self):
        self._ready(self.directory)
        output = self.directory / DEFAULT_OUTPUT_FILENAME
        output.touch()
        output.unlink()
        self.frame._set_running(True)

        self.frame._compilation_succeeded(SimpleNamespace(output_path=output))
        self._wait(lambda: self.frame._input_scan_state == "ready")

        self.assertFalse(self.frame._running)
        self.assertIsNone(self.frame._completed_output_path)
        self.assertIn("no longer available", self.frame.status_var.get())
        self.completed.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(self.popups.mock_calls, [])

    def test_new_completed_run_at_same_path_compiles_rebuilt_measurements(self):
        subject = self.directory / "Group" / "Subject"
        subject.mkdir(parents=True)
        original = subject / "original.sdb"
        original.write_bytes(b"unchanged source")
        source_step = SimpleNamespace(
            _step5_handoff_generation=1,
            _completion_step5_handoff_ready=lambda: True,
        )
        self.frame.source_step = source_step
        output = subject / DEFAULT_OUTPUT_FILENAME

        for generation, value in ((1, 16), (2, 38)):
            self._write_measurement(subject, value)
            source_step._step5_handoff_generation = generation
            self.frame.start_from_step4(subject)
            self._wait(lambda: self.completed.call_count == generation
                       and self.frame._input_scan_state == "ready")
            workbook = load_workbook(output, data_only=True)
            try:
                self.assertEqual(workbook["rrMCP-AR"]["C3"].value, value)
            finally:
                workbook.close()
            if generation == 1:
                for side in ("nasal", "temporal"):
                    shutil.rmtree(subject / side)
                output.unlink()
                self.frame.on_show()
                self._wait(lambda: self.frame._input_scan_state == "ready")
                self.assertEqual(list(subject.iterdir()), [original])

        self.assertEqual(original.read_bytes(), b"unchanged source")
        self.assertEqual(self.completed.call_count, 2)
        self.assertEqual(self.popups.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
