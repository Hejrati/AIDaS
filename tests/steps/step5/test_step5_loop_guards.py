"""Keep Step 5 handoffs passive and manual export workers bounded by lifecycle."""

from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step5_compile import DEFAULT_OUTPUT_FILENAME, Step5Frame


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Step5LoopGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        thread_patch = mock.patch("aidas.ui.compiler.threading.Thread")
        self.thread_factory = thread_patch.start()
        self.addCleanup(thread_patch.stop)
        self.worker = self.thread_factory.return_value
        self.worker.is_alive.return_value = True
        popup_patch = mock.patch("aidas.steps.step5_compile.messagebox.showerror")
        self.popup = popup_patch.start()
        self.addCleanup(popup_patch.stop)

    def _frame(self):
        frame = object.__new__(Step5Frame)
        frame._events = queue.Queue()
        frame._running = False
        frame._worker = None
        frame._pending_step4_handoffs = []
        frame._output_user_selected = False
        frame._poll_after_id = None
        frame.input_var = _Var()
        frame.output_var = _Var()
        frame.include_fovea_var = _Var(True)
        frame.status_var = _Var()
        frame.source_step = SimpleNamespace(
            batch_roi_root=self.root, _step5_handoff_generation=0,
        )
        frame.on_compilation_complete = None
        frame._refresh_directory_view = mock.Mock()
        frame._set_running = mock.Mock(
            side_effect=lambda running: setattr(frame, "_running", bool(running))
        )
        frame.winfo_exists = mock.Mock(return_value=True)
        frame.after = mock.Mock(return_value="compile-poll")
        frame.after_cancel = mock.Mock()
        return frame

    def _start_manual(self, frame, folder=None):
        frame.start_from_step4(folder or self.root)
        frame._run_clicked()

    def _finish(self, frame):
        frame._compilation_succeeded(SimpleNamespace(output_path=Path(frame.output_var.get())))

    def test_repeated_handoffs_never_start_a_worker_or_an_idle_poll(self):
        frame = self._frame()

        for _ in range(12):
            frame.start_from_step4(self.root)
            frame._poll_events()

        self.thread_factory.assert_not_called()
        frame.after.assert_not_called()
        self.assertFalse(frame._running)
        self.assertEqual(Path(frame.input_var.get()), self.root)
        self.assertEqual(Path(frame.output_var.get()), self.root / DEFAULT_OUTPUT_FILENAME)
        self.assertIn("Ready to create an Excel file", frame.status_var.get())
        self.popup.assert_not_called()

    def test_manual_runs_remain_available_after_more_than_five_completions(self):
        frame = self._frame()
        frame.start_from_step4(self.root)

        for expected in range(1, 9):
            frame._run_clicked()
            self.assertEqual(self.thread_factory.call_count, expected)
            self.assertTrue(frame._running)
            self._finish(frame)
            self.assertFalse(frame._running)
            frame._poll_events()
            self.assertEqual(self.thread_factory.call_count, expected)

        self.popup.assert_not_called()

    def test_new_step4_generations_also_only_prefill_the_folder(self):
        frame = self._frame()

        for generation in range(8):
            frame.source_step._step5_handoff_generation = generation
            frame.start_from_step4(self.root)

        self.thread_factory.assert_not_called()
        frame.after.assert_not_called()
        self.assertFalse(frame._running)
        self.assertEqual(Path(frame.input_var.get()), self.root)

    def test_folder_aliases_and_completion_reentry_never_start_another_worker(self):
        subject = self.root / "Subject_A"
        side = subject / "nasal"
        side.mkdir(parents=True)
        frame = self._frame()
        self._start_manual(frame)
        frame.start_from_step4(str(self.root / "."))
        frame.start_from_step4(side)
        frame.on_compilation_complete = lambda _result: frame.start_from_step4(side)

        self._finish(frame)
        frame._poll_events()
        frame._poll_events()

        self.assertEqual(self.thread_factory.call_count, 1)
        self.assertEqual(Path(frame.input_var.get()), subject)
        self.assertEqual(Path(frame.output_var.get()), subject / DEFAULT_OUTPUT_FILENAME)
        self.assertEqual(frame._pending_step4_handoffs, [])
        self.assertFalse(frame._running)

    def test_busy_handoffs_replace_the_queue_with_only_the_latest_folder(self):
        frame = self._frame()
        self._start_manual(frame)
        destinations = [self.root / f"input_{number}" for number in range(8)]
        for destination in destinations:
            destination.mkdir()
            frame.start_from_step4(destination)
            frame.start_from_step4(destination)
            self.assertEqual(frame._pending_step4_handoffs, [(destination, 0)])
            self.assertEqual(Path(frame.input_var.get()), self.root)
        self.assertEqual(self.thread_factory.call_count, 1)

        self._finish(frame)
        frame._poll_events()
        frame._poll_events()

        self.assertEqual(self.thread_factory.call_count, 1)
        self.assertEqual(Path(frame.input_var.get()), destinations[-1])
        self.assertEqual(Path(frame.output_var.get()), destinations[-1] / DEFAULT_OUTPUT_FILENAME)
        self.assertEqual(frame._pending_step4_handoffs, [])
        self.assertFalse(frame._running)

    def test_explicit_narrower_subject_handoff_replaces_an_older_broad_batch(self):
        frame = self._frame()
        subject = self.root / "sdb - Copy (14)"
        (subject / "nasal").mkdir(parents=True)
        (subject / "temporal").mkdir()
        self._start_manual(frame)

        frame.start_from_step4(subject)

        self.assertEqual(self.thread_factory.call_count, 1)
        self.assertEqual(frame._pending_step4_handoffs, [(subject, 0)])
        self._finish(frame)
        frame._poll_events()
        self.assertEqual(self.thread_factory.call_count, 1)
        self.assertEqual(Path(frame.input_var.get()), subject)
        self.assertEqual(Path(frame.output_var.get()), subject / DEFAULT_OUTPUT_FILENAME)

        frame._run_clicked()

        self.assertEqual(self.thread_factory.call_count, 2)
        self.assertTrue(frame._running)

    def test_new_folder_requested_during_completion_is_prefilled_without_running(self):
        frame = self._frame()
        destination = self.root / "new_input"
        destination.mkdir()
        self._start_manual(frame)

        def request_next(_result):
            frame.start_from_step4(destination)
            frame.start_from_step4(destination)
            frame._run_clicked()
            self.assertEqual(frame._pending_step4_handoffs, [(destination, 0)])
            self.assertEqual(self.thread_factory.call_count, 1)
            self.assertEqual(Path(frame.input_var.get()), self.root)

        frame.on_compilation_complete = request_next
        self._finish(frame)
        frame.on_compilation_complete = None
        frame._poll_events()

        self.assertEqual(self.thread_factory.call_count, 1)
        self.assertEqual(Path(frame.input_var.get()), destination)
        self.assertEqual(frame._pending_step4_handoffs, [])
        self.assertFalse(frame._running)

    def test_latest_queued_handoff_retains_its_captured_generation_without_running(self):
        frame = self._frame()
        self._start_manual(frame)
        frame.source_step._step5_handoff_generation = 1
        frame.start_from_step4(self.root)
        frame.source_step._step5_handoff_generation = 2
        frame.start_from_step4(self.root)
        self.assertEqual(frame._pending_step4_handoffs, [(self.root, 2)])
        frame.source_step._step5_handoff_generation = 3
        self._finish(frame)

        with mock.patch.object(frame, "start_from_step4", wraps=frame.start_from_step4) as handoff:
            frame._poll_events()

        handoff.assert_called_once_with(self.root, _generation=2)
        self.assertEqual(self.thread_factory.call_count, 1)
        self.assertEqual(frame._pending_step4_handoffs, [])
        self.assertFalse(frame._running)

    def test_new_idle_handoff_discards_an_older_queued_folder(self):
        frame = self._frame()
        older = self.root / "older"
        latest = self.root / "latest"
        older.mkdir()
        latest.mkdir()
        self._start_manual(frame)
        frame.start_from_step4(older)
        self._finish(frame)

        frame.start_from_step4(latest)
        frame._poll_events()

        self.assertEqual(Path(frame.input_var.get()), latest)
        self.assertEqual(Path(frame.output_var.get()), latest / DEFAULT_OUTPUT_FILENAME)
        self.assertEqual(frame._pending_step4_handoffs, [])
        self.assertEqual(self.thread_factory.call_count, 1)

    def test_manual_click_while_worker_is_running_does_not_start_a_second_worker(self):
        frame = self._frame()
        self._start_manual(frame)

        for _ in range(8):
            frame._run_clicked()

        self.assertTrue(frame._running)
        self.assertEqual(self.thread_factory.call_count, 1)

    def test_worker_error_can_prefill_a_queued_folder_but_never_restarts(self):
        frame = self._frame()
        destination = self.root / "next_input"
        destination.mkdir()
        self._start_manual(frame)
        frame.start_from_step4(destination)
        frame._events.put(("error", RuntimeError("Cannot save workbook")))

        frame._poll_events()
        frame._poll_events()

        self.popup.assert_called_once()
        self.assertFalse(frame._running)
        self.assertEqual(self.thread_factory.call_count, 1)
        self.assertEqual(Path(frame.input_var.get()), destination)
        self.assertEqual(frame._pending_step4_handoffs, [])

    def test_failed_manual_export_can_be_retried_only_by_another_manual_click(self):
        frame = self._frame()
        self._start_manual(frame)
        frame._events.put(("error", RuntimeError("Cannot save workbook")))

        frame._poll_events()
        frame._poll_events()

        self.assertFalse(frame._running)
        self.assertEqual(self.thread_factory.call_count, 1)
        self.assertIn("Could not create the Excel file", frame.status_var.get())

        frame._run_clicked()

        self.assertTrue(frame._running)
        self.assertEqual(self.thread_factory.call_count, 2)

    def test_long_running_live_worker_is_not_stopped_after_five_polls(self):
        frame = self._frame()
        self._start_manual(frame)

        for _ in range(20):
            frame._poll_events()

        self.assertTrue(frame._running)
        self.assertEqual(self.thread_factory.call_count, 1)
        self.popup.assert_not_called()

    def test_dead_worker_without_a_terminal_event_fails_on_fifth_check(self):
        frame = self._frame()
        self._start_manual(frame)
        self.worker.is_alive.return_value = False

        for _ in range(4):
            frame._poll_events()
            self.assertTrue(frame._running)
            self.popup.assert_not_called()
        frame._poll_events()

        self.assertFalse(frame._running)
        self.assertEqual(self.thread_factory.call_count, 1)
        self.popup.assert_called_once()
        frame.after.reset_mock()
        frame._poll_events()
        frame.after.assert_not_called()

    def test_queued_terminal_event_wins_over_the_dead_worker_guard(self):
        frame = self._frame()
        self._start_manual(frame)
        self.worker.is_alive.return_value = False
        frame._events.put(("success", SimpleNamespace(output_path=Path(frame.output_var.get()))))

        frame._poll_events()

        self.assertFalse(frame._running)
        self.assertIn("Excel file saved", frame.status_var.get())
        self.popup.assert_not_called()

    def test_live_worker_resets_consecutive_missing_completion_checks(self):
        frame = self._frame()
        self._start_manual(frame)
        self.worker.is_alive.return_value = False
        frame._poll_events()
        frame._poll_events()
        self.worker.is_alive.return_value = True
        frame._poll_events()
        self.worker.is_alive.return_value = False

        for _ in range(4):
            frame._poll_events()
            self.assertTrue(frame._running)
            self.popup.assert_not_called()
        frame._poll_events()

        self.assertFalse(frame._running)
        self.popup.assert_called_once()

    def test_idle_poll_does_not_schedule_an_endless_timer(self):
        frame = self._frame()

        frame._poll_events()
        frame._poll_events()

        frame.after.assert_not_called()
        self.thread_factory.assert_not_called()

    def test_event_poll_is_scheduled_only_once_after_a_manual_start(self):
        frame = self._frame()
        frame.start_from_step4(self.root)
        frame.after.assert_not_called()

        frame._run_clicked()
        frame._ensure_event_poll()
        frame._ensure_event_poll()

        frame.after.assert_called_once_with(frame.EVENT_POLL_MS, frame._poll_events)

    def test_reentrant_poll_does_not_drain_twice_or_recurse(self):
        frame = self._frame()
        frame._drain_compile_events = mock.Mock(side_effect=frame._poll_events)

        frame._poll_events()

        frame._drain_compile_events.assert_called_once_with()
        frame.after.assert_not_called()

    def test_destroy_cancels_poll_and_discards_queued_handoffs(self):
        frame = self._frame()
        frame._poll_after_id = "compile-poll"
        frame._pending_step4_handoffs = [(self.root, 0)]

        with mock.patch("aidas.steps.step5_compile.SidebarStepFrame.destroy") as destroy:
            frame.destroy()
        frame.start_from_step4(self.root)
        frame._run_clicked()
        frame._poll_events()

        frame.after_cancel.assert_called_once_with("compile-poll")
        self.assertIsNone(frame._poll_after_id)
        self.assertEqual(frame._pending_step4_handoffs, [])
        frame.after.assert_not_called()
        self.thread_factory.assert_not_called()
        destroy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
