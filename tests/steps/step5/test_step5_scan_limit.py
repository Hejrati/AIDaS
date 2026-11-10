"""Oversized Step 5 selections stop with inline guidance and no exports."""

from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.services.measurement_inventory import scan_measurement_directory
from aidas.services.measurement_scan import MeasurementScanLimitExceeded
from aidas.steps.step5_compile import Step5Frame
from aidas.ui.theme import get_interface_mode, set_interface_mode


class Step5ScanLimitTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_interface_mode, get_interface_mode(), redraw=False)
        set_interface_mode("Modern", redraw=False)
        try:
            self.root = ctk.CTk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.addCleanup(self.root.destroy)
        self.root.attributes("-alpha", 0)
        self.root.geometry("1100x760+10000+10000")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.errors = []
        self.root.report_callback_exception = lambda *error: self.errors.append(error)
        self.frame = Step5Frame(self.root)
        self.frame.pack(fill="both", expand=True)
        self.root.update()
        patch = mock.patch("aidas.services.measurement_scan.MAX_MEASUREMENT_SUBFOLDERS", 3)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch("aidas.steps.step5_compile.messagebox")
        self.popups = patch.start()
        self.addCleanup(patch.stop)

    def _wait(self, condition):
        deadline = time.monotonic() + 8
        while not condition():
            self.assertLess(time.monotonic(), deadline, "Folder scan did not finish")
            self.root.update()
            time.sleep(.005)
        self.root.update()
        self.assertFalse(self.errors)

    def _oversized(self):
        directory = self.directory / "Large"
        directory.mkdir()
        for number in range(4):
            (directory / f"child{number}").mkdir()
        return directory

    def test_oversized_selection_shows_inline_message_and_cannot_compile(self):
        frame = self.frame
        directory = self._oversized()
        with mock.patch.object(frame, "_start_compiler") as start:
            frame.set_input_folder(directory)
            self.assertIn("disabled", frame.run_button.state())
            self._wait(lambda: frame._input_scan_state == "limited")
            frame.run_button.invoke()
            frame._run_clicked()
            start.assert_not_called()
        tree = frame.directory_view
        self.assertIn("too many subfolders", tree.notice_var.get())
        self.assertIn("limit of 3", tree.notice_var.get())
        self.assertIn("select another folder", tree.notice_var.get())
        self.assertTrue(tree.notice.winfo_ismapped())
        self.assertEqual(tree._rows, [])
        self.assertIsNone(tree._inventory)
        self.assertIsNone(tree._poll_id)
        self.assertIn("disabled", frame.run_button.state())
        self.assertIn("limit of 3", frame.status_var.get())
        self.assertFalse(Path(frame.output_var.get()).exists())
        self.popups.assert_not_called()
        self.assertEqual(self.popups.mock_calls, [])

    def test_exact_limit_nested_folder_is_accepted(self):
        selected = self.directory / "Selected"
        (selected / "one" / "two" / "three").mkdir(parents=True)
        self.frame.set_input_folder(selected)
        self._wait(lambda: self.frame._input_scan_state == "ready")
        self.assertEqual(self.frame.directory_view.notice_var.get(), "")
        self.assertNotIn("disabled", self.frame.run_button.state())

    def test_selecting_another_folder_recovers_without_automatic_export(self):
        self.frame.set_input_folder(self._oversized())
        self._wait(lambda: self.frame._input_scan_state == "limited")
        small = self.directory / "Small"
        small.mkdir()
        with mock.patch.object(self.frame, "_start_compiler") as start:
            self.frame.set_input_folder(small)
            self._wait(lambda: self.frame._input_scan_state == "ready")
            start.assert_not_called()
        self.assertEqual(self.frame.directory_view.notice_var.get(), "")
        self.assertIn("Ready to create", self.frame.status_var.get())
        self.assertNotIn("disabled", self.frame.run_button.state())
        self.assertEqual(self.popups.mock_calls, [])

    def test_scan_in_progress_blocks_even_direct_submission(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def slow_scan(directory, **options):
            entered.set()
            release.wait(5)
            return scan_measurement_directory(directory, **options)

        with mock.patch("aidas.ui.measurement_tree.scan_measurement_directory", side_effect=slow_scan), \
                mock.patch.object(self.frame, "_start_compiler") as start:
            self.frame.set_input_folder(self.directory)
            self._wait(entered.is_set)
            self.frame._run_clicked()
            start.assert_not_called()
            self.assertIn("disabled", self.frame.run_button.state())
            release.set()
            self._wait(lambda: self.frame._input_scan_state == "ready")
            self.frame.run_button.invoke()
            start.assert_called_once()

    def test_completed_step4_handoff_waits_for_scan_then_exports_without_a_click(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        frame = self.frame
        frame.source_step = SimpleNamespace(
            _step5_handoff_generation=1,
            _completion_step5_handoff_ready=lambda: True,
        )

        def slow_scan(directory, **options):
            entered.set()
            release.wait(5)
            return scan_measurement_directory(directory, **options)

        with mock.patch("aidas.ui.measurement_tree.scan_measurement_directory", side_effect=slow_scan), \
                mock.patch.object(frame, "_start_compiler") as start:
            frame.start_from_step4(self.directory)
            self._wait(entered.is_set)
            self.assertEqual(frame._input_scan_state, "scanning")
            start.assert_not_called()
            release.set()
            self._wait(lambda: frame._input_scan_state == "ready")
            start.assert_called_once_with(
                self.directory, Path(frame.output_var.get()), True,
                thread_name="aidas-step5-compiler",
            )
            frame.on_show()
            frame.start_from_step4(self.directory)
            self._wait(lambda: frame._input_scan_state == "ready")
            self.assertEqual(start.call_count, 1)
        self.assertEqual(self.popups.mock_calls, [])

    def test_completed_handoff_to_oversized_folder_never_exports(self):
        frame = self.frame
        frame.source_step = SimpleNamespace(
            _step5_handoff_generation=1,
            _completion_step5_handoff_ready=lambda: True,
        )
        with mock.patch.object(frame, "_start_compiler") as start:
            frame.start_from_step4(self._oversized())
            self._wait(lambda: frame._input_scan_state == "limited")
            frame._try_automatic_export()
            start.assert_not_called()
            self.assertIn("limit of 3", frame.status_var.get())
        self.assertEqual(self.popups.mock_calls, [])

    def test_manual_folder_selection_cancels_a_waiting_automatic_handoff(self):
        frame = self.frame
        frame.source_step = SimpleNamespace(
            _step5_handoff_generation=1,
            _completion_step5_handoff_ready=lambda: True,
        )
        other = self.directory / "Other"
        other.mkdir()
        with mock.patch.object(frame, "_start_compiler") as start:
            frame.start_from_step4(self.directory)
            frame.set_input_folder(other)
            self._wait(lambda: frame._input_scan_state == "ready")
            start.assert_not_called()
        self.assertEqual(frame._input_scan_path, other)
        self.assertEqual(self.popups.mock_calls, [])

    def test_typed_folder_cannot_reuse_a_previous_small_folders_validation(self):
        small = self.directory / "Small"
        small.mkdir()
        self.frame.set_input_folder(small)
        self._wait(lambda: self.frame._input_scan_state == "ready")
        large = self._oversized()
        self.frame.input_var.set(str(large))
        self.assertIn("disabled", self.frame.run_button.state())
        with mock.patch.object(self.frame, "_start_compiler") as start:
            self.frame._run_clicked()
            self._wait(lambda: self.frame._input_scan_state == "limited")
            start.assert_not_called()
        self.assertEqual(self.frame.directory_view._directory, large)
        self.assertEqual(self.popups.mock_calls, [])

    def test_stale_large_scan_cannot_reject_a_new_small_selection(self):
        old = self._oversized()
        new = self.directory / "Small"
        new.mkdir()
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def scan(directory, **options):
            if directory == old:
                entered.set()
                release.wait(5)
                try:
                    return scan_measurement_directory(directory)
                finally:
                    finished.set()
            return scan_measurement_directory(directory, **options)

        with mock.patch("aidas.ui.measurement_tree.scan_measurement_directory", side_effect=scan):
            self.frame.set_input_folder(old)
            self._wait(entered.is_set)
            self.frame.set_input_folder(new)
            self._wait(lambda: self.frame._input_scan_state == "ready")
            release.set()
            self._wait(finished.is_set)
            self.frame.directory_view._poll()
        self.assertEqual(self.frame._input_scan_path, new)
        self.assertEqual(self.frame._input_scan_state, "ready")
        self.assertEqual(self.frame.directory_view.notice_var.get(), "")
        self.assertNotIn("disabled", self.frame.run_button.state())

    def test_compiler_detects_growth_after_preview_without_popup_or_output_changes(self):
        selected = self.directory / "Selected"
        selected.mkdir()
        frame = self.frame
        frame.set_input_folder(selected)
        self._wait(lambda: frame._input_scan_state == "ready")
        output = Path(frame.output_var.get())
        output.write_bytes(b"existing workbook must remain unchanged")
        for number in range(4):
            (selected / f"child{number}").mkdir()
        frame.run_button.invoke()
        self._wait(lambda: not frame._running and frame._input_scan_state == "limited")
        self.assertEqual(output.read_bytes(), b"existing workbook must remain unchanged")
        self.assertIsInstance(frame._input_scan_error, MeasurementScanLimitExceeded)
        self.assertIn("select another folder", frame.directory_view.notice_var.get())
        self.assertIn("disabled", frame.run_button.state())
        self.assertEqual(self.popups.mock_calls, [])

    def test_refresh_can_recover_after_excess_subfolders_are_removed(self):
        selected = self._oversized()
        frame = self.frame
        frame.set_input_folder(selected)
        self._wait(lambda: frame._input_scan_state == "limited")
        (selected / "child3").rmdir()
        frame.directory_view.refresh()
        self._wait(lambda: frame._input_scan_state == "ready")
        self.assertIn("Ready to create", frame.status_var.get())
        self.assertEqual(frame.directory_view.notice_var.get(), "")
        self.assertNotIn("disabled", frame.run_button.state())

    def test_failed_refresh_after_rejection_shows_inline_error_then_recovers(self):
        selected = self._oversized()
        frame = self.frame
        frame.set_input_folder(selected)
        self._wait(lambda: frame._input_scan_state == "limited")
        with mock.patch(
            "aidas.ui.measurement_tree.scan_measurement_directory",
            side_effect=OSError("Folder is temporarily unavailable"),
        ):
            frame.directory_view.refresh()
            self._wait(lambda: frame._input_scan_state == "error")
        self.assertIn("Could not read this folder", frame.status_var.get())
        self.assertIn("temporarily unavailable", frame.directory_view.notice_var.get())
        self.assertIn("disabled", frame.run_button.state())
        self.assertEqual(self.popups.mock_calls, [])
        (selected / "child3").rmdir()
        frame.directory_view.refresh()
        self._wait(lambda: frame._input_scan_state == "ready")
        self.assertIn("Ready to create", frame.status_var.get())
        self.assertNotIn("disabled", frame.run_button.state())


if __name__ == "__main__":
    unittest.main()
