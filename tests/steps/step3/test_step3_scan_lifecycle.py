"""Step 3 scans and setup polls settle without timing out live work."""

from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step3_flatten import RBatchSelectionPanel, RSetupWizard, Step3Frame
from aidas.utils.filesystem import DirectoryScanLimitExceeded
from aidas.utils.worker_polling import WorkerCompletionGuard


class Step3ScanLifecycleTests(unittest.TestCase):
    def _panel(self, *, alive=False):
        panel = RBatchSelectionPanel.__new__(RBatchSelectionPanel)
        panel.root_dir = Path("selected-folder")
        panel.input_folders = ()
        panel._scan_events = queue.Queue()
        panel._scan_cancel = threading.Event()
        panel._scan_poll_after_id = None
        panel._scan_closed = False
        panel.after = mock.Mock(return_value="scan-poll")
        panel.after_cancel = mock.Mock()
        panel.winfo_exists = mock.Mock(return_value=True)
        panel.next_button = mock.Mock()
        panel.step_frame = SimpleNamespace(status_var=mock.Mock(), REQUIRED_INPUTS=Step3Frame.REQUIRED_INPUTS)
        panel._scan_done = mock.Mock()
        panel._scan_failed = mock.Mock()
        panel._scan_completion = WorkerCompletionGuard(mock.Mock(is_alive=mock.Mock(return_value=alive)))
        return panel

    def test_scan_worker_never_calls_tk_and_detached_results_cannot_replace_new_scan(self):
        panel = self._panel()
        old_events = queue.Queue()
        panel._scan_worker(Path("old"), (), threading.Event(), old_events)
        self.assertTrue(panel._scan_events.empty())
        self.assertEqual(old_events.get_nowait()[0], "done")
        panel.after.assert_not_called()
        panel._scan_done.assert_not_called()
        panel._scan_worker()
        panel._poll_scan_events()
        panel._scan_done.assert_called_once_with([], 0, 0, [])
        panel.after.assert_not_called()

    def test_cancelled_worker_does_not_publish_completion(self):
        panel = self._panel()
        panel._scan_cancel.set()
        panel._scan_worker()
        self.assertTrue(panel._scan_events.empty())
        panel.after.assert_not_called()

    def test_live_worker_never_expires_and_dead_worker_stops_on_fifth_empty_poll(self):
        for alive in (True, False):
            with self.subTest(alive=alive):
                panel = self._panel(alive=alive)
                for _ in range(8 if alive else 5):
                    panel._poll_scan_events()
                self.assertEqual(panel._scan_failed.call_count, 0 if alive else 1)
                self.assertEqual(panel.after.call_count, 8 if alive else 4)

    def test_queued_completion_precedes_dead_worker_guard(self):
        panel = self._panel()
        for _ in range(4):
            panel._poll_scan_events()
        panel._scan_events.put(("done", ([], 0, 0, [])))
        panel._poll_scan_events()
        panel._scan_failed.assert_not_called()
        panel._scan_done.assert_called_once_with([], 0, 0, [])

    def test_thread_start_failure_is_delivered_without_idle_poll(self):
        panel = self._panel()
        with mock.patch("aidas.steps.step3_flatten.threading.Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("cannot start")
            panel._start_scan()
        panel._scan_failed.assert_called_once()
        panel.after.assert_not_called()

    def test_destroy_cancels_scan_and_timer(self):
        panel = self._panel()
        panel._scan_poll_after_id = "existing-poll"
        with mock.patch("tkinter.ttk.Frame.destroy"):
            panel.destroy()
        self.assertTrue(panel._scan_cancel.is_set())
        panel.after_cancel.assert_called_once_with("existing-poll")
        panel._poll_scan_events()
        panel.after.assert_not_called()

    def test_oversized_scan_never_validates_or_returns_partial_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number in range(4):
                (root / str(number)).mkdir()
            panel = self._panel()
            panel.root_dir = root
            panel.input_folders = None
            panel.step_frame._find_input_paths = mock.Mock()
            with mock.patch("aidas.utils.filesystem.MAX_SCAN_SUBFOLDERS", 3):
                panel._scan_worker()
            kind, error = panel._scan_events.get_nowait()
            self.assertEqual(kind, "error")
            self.assertIsInstance(error, DirectoryScanLimitExceeded)
            panel.step_frame._find_input_paths.assert_not_called()
            self.assertTrue(panel._scan_events.empty())

    def test_result_load_limit_preserves_existing_results_without_modal(self):
        frame = Step3Frame.__new__(Step3Frame)
        frame.status_var = mock.Mock()
        previous = object()
        frame.results = previous
        frame._find_r_result_folders = mock.Mock(side_effect=DirectoryScanLimitExceeded())
        frame._load_r_results_from_folder = mock.Mock()
        frame._open_batch_r_result_tabs = mock.Mock()
        with mock.patch("aidas.steps.step3_flatten.messagebox") as dialogs:
            self.assertFalse(frame._load_r_results_from_tree("large"))
        self.assertIs(frame.results, previous)
        frame._load_r_results_from_folder.assert_not_called()
        frame._open_batch_r_result_tabs.assert_not_called()
        self.assertEqual(dialogs.mock_calls, [])

    def test_setup_dead_worker_recovers_and_live_work_is_not_limited(self):
        for alive in (True, False):
            with self.subTest(alive=alive):
                wizard = RSetupWizard.__new__(RSetupWizard)
                wizard._closed = False
                wizard._worker_events = queue.Queue()
                wizard._worker_completion = WorkerCompletionGuard(mock.Mock(is_alive=mock.Mock(return_value=alive)))
                wizard.after = mock.Mock()
                wizard._finish_worker = mock.Mock()
                done = mock.Mock()
                for _ in range(8 if alive else 5):
                    wizard._poll_worker_result(done)
                self.assertEqual(wizard._finish_worker.call_count, 0 if alive else 1)
                self.assertEqual(wizard.after.call_count, 8 if alive else 4)

    def test_setup_start_failure_finishes_immediately_without_poll(self):
        wizard = RSetupWizard.__new__(RSetupWizard)
        wizard._closed = False
        wizard._cancel_event = threading.Event()
        wizard._worker_events = queue.Queue()
        wizard._set_busy = mock.Mock()
        wizard._finish_worker = mock.Mock()
        wizard.after = mock.Mock()
        done = mock.Mock()
        with mock.patch("aidas.steps.step3_flatten.threading.Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("cannot start")
            wizard._run_worker("Checking", mock.Mock(), done)
        wizard._finish_worker.assert_called_once()
        self.assertIs(wizard._finish_worker.call_args.args[0], done)
        wizard.after.assert_not_called()

    def test_exited_r_process_cannot_wait_forever_for_a_descendants_open_stdout(self):
        frame = Step3Frame.__new__(Step3Frame)
        frame._r_cancel_event = threading.Event()
        frame._r_process_lock = threading.Lock()
        frame._active_r_processes = set()
        process = mock.Mock()
        process.poll.return_value = 0
        process.wait.return_value = 0
        events = mock.Mock()
        events.get.side_effect = queue.Empty()
        events.qsize.return_value = 0
        with (
            mock.patch("aidas.steps.step3_flatten.subprocess.Popen", return_value=process),
            mock.patch("aidas.steps.step3_flatten.queue.Queue", return_value=events),
            mock.patch("aidas.steps.step3_flatten.threading.Thread") as reader,
        ):
            reader.return_value.is_alive.return_value = True
            code, error, outcome = frame._run_supervised_r_command(["Rscript"], ".", {}, mock.Mock())
        self.assertEqual(code, 1)
        self.assertEqual(outcome, "failed")
        self.assertIn("without reporting completion", error)
        self.assertEqual(events.get.call_count, 5)
        self.assertEqual(frame._active_r_processes, set())

    def test_r_output_reader_start_failure_cleans_up_the_launched_process(self):
        frame = Step3Frame.__new__(Step3Frame)
        frame._register_r_process = mock.Mock()
        frame._unregister_r_process = mock.Mock()
        frame._terminate_r_process = mock.Mock()
        process = mock.Mock()
        with (
            mock.patch("aidas.steps.step3_flatten.subprocess.Popen", return_value=process),
            mock.patch("aidas.steps.step3_flatten.threading.Thread") as reader,
        ):
            reader.return_value.start.side_effect = RuntimeError("cannot start reader")
            code, error, outcome = frame._run_supervised_r_command(["Rscript"], ".", {}, mock.Mock())
        self.assertEqual((code, outcome), (1, "failed"))
        self.assertIn("Could not start", error)
        frame._terminate_r_process.assert_called_once_with(process, force=True)
        frame._unregister_r_process.assert_called_once_with(process)

    def test_windows_taskkill_failure_uses_direct_child_fallback_with_bounded_wait(self):
        frame = Step3Frame.__new__(Step3Frame)
        for force in (False, True):
            with self.subTest(force=force):
                process = mock.Mock(pid=123)
                process.poll.return_value = None
                with (
                    mock.patch("aidas.steps.step3_flatten.os.name", "nt"),
                    mock.patch("aidas.steps.step3_flatten.subprocess.run", return_value=SimpleNamespace(returncode=1)),
                ):
                    frame._terminate_r_process(process, force=force)
                (process.kill if force else process.terminate).assert_called_once_with()
                process.wait.assert_called_once_with(timeout=2)

    def test_direct_child_fallback_escalates_termination_without_unbounded_wait(self):
        process = mock.Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("Rscript", 2), 1]
        Step3Frame._stop_r_process_direct(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=2), mock.call(timeout=2)])


if __name__ == "__main__":
    unittest.main()
