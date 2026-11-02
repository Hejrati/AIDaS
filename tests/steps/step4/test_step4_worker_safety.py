"""Folder limits and worker failures must leave Step 4 recoverable."""

from pathlib import Path
import queue
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps import step4_analyze_isez as step4
from aidas.utils.filesystem import DirectoryScanLimitExceeded
from aidas.utils.worker_polling import WorkerCompletionGuard
from tests.steps.step4 import test_step4_auto_excel as export_fixtures
from tests.steps.step4 import test_step4_batch_stack_building as batch_fixtures


class Step4FolderSafetyTests(unittest.TestCase):
    def _panel(self):
        panel = object.__new__(step4.Step4BatchROISelectionPanel)
        panel.root_dir = Path("selected")
        panel.step_frame = SimpleNamespace(status_var=mock.Mock(), _scan_batch_roi_folders=mock.Mock())
        panel._scan_cancel = threading.Event()
        panel._scan_events = queue.Queue()
        panel._scan_after_id = "pending"
        panel._scan_guard = None
        panel.winfo_exists = mock.Mock(return_value=True)
        panel.after = mock.Mock(return_value="poll")
        panel.after_cancel = mock.Mock()
        panel._scan_done = mock.Mock()
        panel._scan_failed = mock.Mock()
        return panel

    def test_oversized_tree_returns_no_partial_rows_or_metadata_reads(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b", "c"):
                child = root / name
                child.mkdir()
                (child / "_flat_LIGHT.hdr").touch()
            frame = object.__new__(step4.Step4Frame)
            with mock.patch("aidas.utils.filesystem.MAX_SCAN_SUBFOLDERS", 2), \
                    mock.patch.object(frame, "_step4_output_count") as read:
                with self.assertRaises(DirectoryScanLimitExceeded):
                    frame._scan_batch_roi_folders(root)
            read.assert_not_called()

    def test_cancellation_during_output_checks_discards_all_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                child = root / name
                child.mkdir()
                (child / "_flat_LIGHT.hdr").touch()
            cancelled = threading.Event()
            frame = object.__new__(step4.Step4Frame)

            def inspect_outputs(_folder, *, cancel_event):
                self.assertIs(cancel_event, cancelled)
                cancelled.set()
                return 0

            with mock.patch.object(frame, "_step4_output_count", side_effect=inspect_outputs) as read:
                rows, scanned, _, errors = frame._scan_batch_roi_folders(root, cancel_event=cancelled)
            self.assertEqual(rows, [])
            self.assertEqual(scanned, 3)
            self.assertEqual(errors, [])
            read.assert_called_once()

    def test_cancel_stops_pending_poll_and_ignores_stale_delivery(self):
        panel = self._panel()
        cancelled, events = panel._scan_cancel, panel._scan_events
        events.put(("done", ([], 1, 0, [])))
        panel._stop_scan()
        panel._poll_scan_events(cancelled, events)
        self.assertTrue(cancelled.is_set())
        panel.after_cancel.assert_called_once_with("pending")
        panel._scan_done.assert_not_called()
        panel.after.assert_not_called()

    def test_live_scan_can_take_more_than_five_polls(self):
        panel = self._panel()
        worker = mock.Mock()
        worker.is_alive.return_value = True
        panel._scan_guard = WorkerCompletionGuard(worker)
        for _ in range(20):
            panel._poll_scan_events()
        panel._scan_failed.assert_not_called()
        self.assertEqual(panel.after.call_count, 20)

    def test_dead_scan_without_terminal_event_recovers_after_five_polls(self):
        panel = self._panel()
        worker = mock.Mock()
        worker.is_alive.return_value = False
        panel._scan_guard = WorkerCompletionGuard(worker)
        for _ in range(5):
            panel._poll_scan_events()
        panel._scan_failed.assert_called_once()
        self.assertIn("stopped without returning", str(panel._scan_failed.call_args.args[0]))
        self.assertEqual(panel.after.call_count, 4)
        self.assertIsNone(panel._scan_after_id)

    def test_thread_start_error_is_delivered_without_waiting_for_worker(self):
        panel = self._panel()
        with mock.patch.object(step4.threading, "Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("No thread available")
            panel._start_scan()
        panel._poll_scan_events()
        panel._scan_failed.assert_called_once()
        self.assertEqual(str(panel._scan_failed.call_args.args[0]), "No thread available")
        self.assertIsNone(panel._scan_guard)

    def test_limit_warning_is_inline_and_run_stays_disabled(self):
        panel = self._panel()
        del panel._scan_failed
        panel.rows = [{"folder": "partial"}]
        panel.table = None
        panel._scan_summary_counts = {"scanned": 1}
        for name in ("summary_var", "scan_label", "more_label", "more_tooltip", "process_button"):
            setattr(panel, name, mock.Mock())
        error = DirectoryScanLimitExceeded()
        with mock.patch.object(step4.messagebox, "showerror") as popup:
            panel._scan_failed(error)
        panel.summary_var.set.assert_called_once_with(str(error))
        panel.scan_label.configure.assert_called_once_with(text=str(error), wraplength=520)
        panel.process_button.state.assert_called_once_with(["disabled"])
        self.assertEqual(panel.rows, [])
        self.assertIsNone(panel._scan_summary_counts)
        popup.assert_not_called()


class Step4WorkerSafetyTests(unittest.TestCase):
    @staticmethod
    def _worker(alive=False):
        worker = mock.Mock()
        worker.is_alive.return_value = alive
        return worker

    def _detection_frame(self):
        frame = object.__new__(step4.Step4Frame)
        frame._auto_detection_events = queue.Queue()
        frame._auto_detection_guard = WorkerCompletionGuard(self._worker())
        frame._auto_detecting = True
        frame.status_var = mock.Mock()
        frame.after = mock.Mock(return_value="poll")
        frame._apply_auto_detection_run = mock.Mock()
        frame._update_auto_detect_button_state = mock.Mock()
        frame._update_build_stack_button_state = mock.Mock()
        return frame

    def test_dead_detection_worker_releases_busy_state(self):
        frame = self._detection_frame()
        events = frame._auto_detection_events
        with mock.patch.object(step4.messagebox, "showerror") as popup:
            for _ in range(5):
                frame._poll_auto_detection_events(events)
            frame._poll_auto_detection_events(events)  # Already queued callback is harmless.
        self.assertFalse(frame._auto_detecting)
        self.assertIsNone(frame._auto_detection_events)
        self.assertIsNone(frame._auto_detection_guard)
        self.assertEqual(frame.after.call_count, 4)
        popup.assert_called_once()
        frame._apply_auto_detection_run.assert_not_called()

    def test_late_completion_is_consumed_before_guard_can_fail(self):
        frame = self._detection_frame()
        events = frame._auto_detection_events
        for _ in range(4):
            frame._poll_auto_detection_events(events)
        result = object()
        events.put(("done", result))
        with mock.patch.object(step4.messagebox, "showerror") as popup:
            frame._poll_auto_detection_events(events)
        frame._apply_auto_detection_run.assert_called_once_with(result)
        self.assertFalse(frame._auto_detecting)
        popup.assert_not_called()

    def test_detection_start_failure_restores_controls(self):
        frame = self._detection_frame()
        frame._auto_detecting = False
        frame.image = object()
        frame.rois = []
        frame.completed = {}
        frame._has_current_roi_input = mock.Mock(return_value=True)
        frame._close_profile_zoom = mock.Mock()
        frame._cancel_roi_update_animations = mock.Mock()
        frame._auto_detection_parameters = mock.Mock(return_value={})
        frame._start_background_worker = mock.Mock(side_effect=RuntimeError("No thread available"))
        with mock.patch.object(step4.messagebox, "showerror") as popup:
            frame._auto_detect_all_rois()
            frame._poll_auto_detection_events(frame._auto_detection_events)
        self.assertFalse(frame._auto_detecting)
        self.assertIsNone(frame._auto_detection_events)
        self.assertIsNone(frame._auto_detection_guard)
        popup.assert_called_once()
        self.assertIn("No thread available", popup.call_args.args[1])

    def test_single_stack_start_failure_restores_buttons(self):
        fixture = export_fixtures.Step4ExcelHandoffTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        frame = fixture._single_frame()
        frame._start_background_worker = mock.Mock(side_effect=RuntimeError("No thread available"))
        frame._build_stack_outputs()
        self.assertTrue(frame._stack_building)
        frame._callbacks.pop(0)()
        self.assertFalse(frame._stack_building)
        self.assertIsNone(frame._stack_build_events)
        frame.on_continue_to_step5.assert_not_called()
        fixture.error.assert_called_once()

    def test_dead_single_stack_worker_does_not_finish_or_advance(self):
        fixture = export_fixtures.Step4ExcelHandoffTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        frame = fixture._single_frame()
        frame._start_background_worker = mock.Mock(return_value=self._worker())
        frame._build_stack_outputs()
        for _ in range(5):
            frame._callbacks.pop(0)()
        self.assertFalse(frame._stack_building)
        self.assertFalse(frame._stack_build_complete)
        self.assertFalse(frame._callbacks)
        frame.on_continue_to_step5.assert_not_called()
        fixture.error.assert_called_once()

    def test_dead_batch_stack_worker_stops_without_completing_unwritten_tabs(self):
        fixture = batch_fixtures.BatchStackBuildingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        frame = fixture.frame
        frame._start_background_worker = mock.Mock(return_value=self._worker())
        frame._auto_continue_to_step5 = mock.Mock()
        frame._build_stack_outputs()
        for _ in range(5):
            fixture.callbacks.pop(0)()
        self.assertFalse(frame._stack_building)
        self.assertIsNone(frame._stack_build_events)
        self.assertFalse(fixture.states["active"]["complete"])
        self.assertFalse(fixture.states["ready"]["complete"])
        self.assertFalse(fixture.callbacks)
        frame._auto_continue_to_step5.assert_not_called()
        fixture.error.assert_called_once()

    def test_batch_stack_handles_finite_event_slices_before_finishing(self):
        fixture = batch_fixtures.BatchStackBuildingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        frame = fixture.frame
        frame.MAX_BACKGROUND_EVENTS_PER_POLL = 1
        frame._build_stack_outputs()
        fixture.workers.pop(0)()
        fixture.callbacks.pop(0)()
        self.assertTrue(frame._stack_building)
        self.assertTrue(fixture.states["active"]["complete"])
        self.assertFalse(fixture.states["ready"]["complete"])
        fixture.callbacks.pop(0)()
        self.assertTrue(frame._stack_building)
        self.assertTrue(fixture.states["ready"]["complete"])
        fixture.callbacks.pop(0)()
        self.assertFalse(frame._stack_building)

    def test_batch_stack_start_failure_recovers_without_marking_tabs_complete(self):
        fixture = batch_fixtures.BatchStackBuildingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        frame = fixture.frame
        frame._start_background_worker = mock.Mock(side_effect=RuntimeError("No thread available"))
        frame._auto_continue_to_step5 = mock.Mock()
        frame._build_stack_outputs()
        fixture.callbacks.pop(0)()
        self.assertFalse(frame._stack_building)
        self.assertFalse(fixture.states["active"]["complete"])
        self.assertFalse(fixture.states["ready"]["complete"])
        self.assertIsNone(frame._stack_build_events)
        self.assertIsNone(frame._stack_build_guard)
        frame._auto_continue_to_step5.assert_not_called()
        fixture.error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
