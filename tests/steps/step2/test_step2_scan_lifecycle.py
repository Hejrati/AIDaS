"""Folder limits and missing completions cannot strand Step 2's selector."""

from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step2_annotate import Step2BatchSegmentationSelectionPanel, Step2Frame
from aidas.steps.step3_flatten import Step3Frame
from aidas.utils.filesystem import DirectoryScanLimitExceeded
from aidas.utils.worker_polling import WorkerCompletionGuard


class Step2ScanLifecycleTests(unittest.TestCase):
    def _panel(self, *, alive=False):
        panel = Step2BatchSegmentationSelectionPanel.__new__(Step2BatchSegmentationSelectionPanel)
        panel.root_dir = "selected-folder"
        panel._scan_events = queue.Queue()
        panel._scan_cancel = threading.Event()
        panel._scan_closed = False
        panel._scan_ready = False
        panel._scan_poll_after_id = None
        panel.after = mock.Mock(return_value="scan-poll")
        panel.after_cancel = mock.Mock()
        panel._scan_done = mock.Mock()
        panel._scan_failed = mock.Mock()
        panel.next_button = mock.Mock()
        panel.step_frame = SimpleNamespace(
            status_var=mock.Mock(),
            _scan_step2_batch_segmentation_folders=mock.Mock(return_value=([{"folder": "result"}], 1, 0, [])),
        )
        panel._scan_completion = WorkerCompletionGuard(mock.Mock(is_alive=mock.Mock(return_value=alive)))
        return panel

    def test_scan_worker_queues_results_without_accessing_tk(self):
        panel = self._panel()
        panel._scan_worker()
        panel.after.assert_not_called()
        panel._scan_done.assert_not_called()
        panel._poll_scan_events()
        panel._scan_done.assert_called_once_with([{"folder": "result"}], 1, 0, [])
        panel.after.assert_not_called()

    def test_cancelled_and_superseded_scan_results_are_discarded(self):
        panel = self._panel()
        old_events, old_cancel = queue.Queue(), threading.Event()
        panel._scan_worker("old-folder", old_cancel, old_events)
        self.assertTrue(panel._scan_events.empty())
        self.assertEqual(old_events.get_nowait()[0], "done")
        old_cancel.set()
        panel._scan_worker("old-folder", old_cancel, old_events)
        self.assertTrue(old_events.empty())
        panel.after.assert_not_called()

    def test_live_scan_does_not_expire_after_five_empty_polls(self):
        panel = self._panel(alive=True)
        for _ in range(8):
            panel._poll_scan_events()
        panel._scan_failed.assert_not_called()
        self.assertEqual(panel.after.call_count, 8)

    def test_dead_scan_fails_on_fifth_missing_poll_but_queued_terminal_wins(self):
        panel = self._panel()
        for _ in range(4):
            panel._poll_scan_events()
        panel._scan_failed.assert_not_called()
        panel._scan_events.put(("done", ([], 0, 0, [])))
        panel._poll_scan_events()
        panel._scan_done.assert_called_once_with([], 0, 0, [])
        panel._scan_failed.assert_not_called()
        panel = self._panel()
        for _ in range(5):
            panel._poll_scan_events()
        panel._scan_failed.assert_called_once()
        self.assertIsNone(panel._scan_poll_after_id)
        self.assertEqual(panel.after.call_count, 4)

    def test_thread_start_failure_finishes_inline_without_polling(self):
        panel = self._panel()
        with mock.patch("aidas.steps.step2_annotate.threading.Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("thread unavailable")
            panel._start_scan()
        panel._scan_failed.assert_called_once()
        panel.after.assert_not_called()

    def test_close_cancels_scan_and_no_callback_is_rescheduled(self):
        panel = self._panel()
        panel._scan_poll_after_id = "existing-poll"
        with mock.patch("tkinter.ttk.Frame.destroy"):
            panel.destroy()
        self.assertTrue(panel._scan_cancel.is_set())
        panel.after_cancel.assert_called_once_with("existing-poll")
        panel._poll_scan_events()
        panel.after.assert_not_called()

    def test_oversized_tree_is_rejected_before_row_validation_or_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(4):
                folder = root / str(index)
                folder.mkdir()
                (folder / "Light.img").write_bytes(b"untouched")
            frame = Step2Frame.__new__(Step2Frame)
            frame._step2_batch_segmentation_row = mock.Mock()
            with mock.patch("aidas.utils.filesystem.MAX_SCAN_SUBFOLDERS", 3):
                with self.assertRaises(DirectoryScanLimitExceeded):
                    frame._scan_step2_batch_segmentation_folders(root)
            frame._step2_batch_segmentation_row.assert_not_called()
            self.assertTrue(all(path.read_bytes() == b"untouched" for path in root.glob("*/Light.img")))

    def test_limit_worker_error_never_delivers_partial_rows(self):
        panel = self._panel()
        error = DirectoryScanLimitExceeded()
        panel.step_frame._scan_step2_batch_segmentation_folders.side_effect = error
        panel._scan_worker()
        panel._poll_scan_events()
        panel._scan_failed.assert_called_once_with(error)
        panel._scan_done.assert_not_called()


class StepWorkerDispatchTests(unittest.TestCase):
    def _frame(self, frame_type, *, alive=False, worker=None):
        frame = frame_type.__new__(frame_type)
        frame.after = mock.Mock(return_value="worker-poll")
        frame.after_cancel = mock.Mock()
        frame._closing = False
        missing = mock.Mock()
        if worker is None:
            worker = mock.Mock(is_alive=mock.Mock(return_value=alive))
        frame._start_worker_dispatch(worker, missing)
        return frame, missing

    def test_background_delivery_is_main_thread_only_and_terminal_is_delivered_once(self):
        for frame_type in (Step2Frame, Step3Frame):
            with self.subTest(step=frame_type.__name__):
                delivered = []
                callback = lambda: delivered.append(threading.get_ident())
                def publish():
                    frame._post_worker_ui(callback)
                    frame._post_worker_ui(callback, terminal=True)
                    frame._post_worker_ui(callback, terminal=True)
                worker = threading.Thread(target=publish)
                frame, missing = self._frame(frame_type, worker=worker)
                worker.start()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(delivered, [])
                self.assertEqual(frame.after.call_count, 1)
                frame._poll_worker_dispatch()
                self.assertEqual(delivered, [threading.get_ident()] * 2)
                self.assertIsNone(frame._worker_ui_poll_after_id)
                missing.assert_not_called()

    def test_replaced_worker_cannot_publish_into_the_next_runs_queue(self):
        for frame_type in (Step2Frame, Step3Frame):
            with self.subTest(step=frame_type.__name__):
                release_old = threading.Event()
                callback = mock.Mock()

                def publish_late():
                    release_old.wait(timeout=2)
                    frame._post_worker_ui(callback, terminal=True)

                old_worker = threading.Thread(target=publish_late)
                frame, _missing = self._frame(frame_type, worker=old_worker)
                old_worker.start()
                old_events = frame._worker_ui_events
                frame._start_worker_dispatch(mock.Mock(is_alive=mock.Mock(return_value=True)), mock.Mock())
                release_old.set()
                old_worker.join(timeout=2)
                self.assertFalse(old_worker.is_alive())
                self.assertTrue(old_events.empty())
                self.assertTrue(frame._worker_ui_events.empty())
                callback.assert_not_called()

    def test_poll_already_delivered_when_dispatch_stops_is_harmless(self):
        for frame_type in (Step2Frame, Step3Frame):
            with self.subTest(step=frame_type.__name__):
                frame, missing = self._frame(frame_type)
                frame._stop_worker_dispatch()
                frame._poll_worker_dispatch()
                self.assertIsNone(frame._worker_ui_poll_after_id)
                missing.assert_not_called()
                self.assertEqual(frame.after.call_count, 1)

    def test_r_pool_progress_keeps_its_coordinators_dispatch_context(self):
        frame = Step3Frame.__new__(Step3Frame)
        frame.after = mock.Mock(return_value="worker-poll")
        frame.after_cancel = mock.Mock()
        frame._closing = False
        frame._r_cancel_event = threading.Event()
        frame._r_worker_limit = mock.Mock(return_value=1)
        frame._r_threads_per_process = mock.Mock(return_value=1)
        frame._folder_has_r_data = mock.Mock(return_value=False)
        frame._r_script_config_for_folder = mock.Mock(return_value={})
        frame._batch_panel_update = mock.Mock()
        frame._on_batch_r_done = mock.Mock()
        progress = mock.Mock()

        def run_pipeline(*_args, **kwargs):
            frame._post_worker_ui(progress)
            return {"folder": kwargs["batch_folder"], "returncode": 0}

        frame._run_r_script_for_config = run_pipeline
        worker = threading.Thread(
            target=frame._batch_r_worker,
            args=("Rscript", "main.R", "output.R", [Path("sample")], 1),
        )
        frame._start_worker_dispatch(worker, mock.Mock())
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        progress.assert_not_called()
        frame._poll_worker_dispatch()
        progress.assert_called_once_with()
        frame._on_batch_r_done.assert_called_once()

        # A pool task finishing after replacement must retain its old origin;
        # neither its progress nor the old coordinator's completion can leak.
        entered, release = threading.Event(), threading.Event()
        progress.reset_mock()
        frame._on_batch_r_done.reset_mock()

        def delayed_pipeline(*args, **kwargs):
            entered.set()
            release.wait(timeout=2)
            return run_pipeline(*args, **kwargs)

        frame._run_r_script_for_config = delayed_pipeline
        old_worker = threading.Thread(
            target=frame._batch_r_worker,
            args=("Rscript", "main.R", "output.R", [Path("sample")], 1),
        )
        frame._start_worker_dispatch(old_worker, mock.Mock())
        old_worker.start()
        self.assertTrue(entered.wait(timeout=2))
        frame._start_worker_dispatch(mock.Mock(is_alive=mock.Mock(return_value=True)), mock.Mock())
        release.set()
        old_worker.join(timeout=2)
        self.assertFalse(old_worker.is_alive())
        self.assertTrue(frame._worker_ui_events.empty())
        frame._poll_worker_dispatch()
        progress.assert_not_called()
        frame._on_batch_r_done.assert_not_called()

    def test_callback_bursts_are_bounded_per_poll_and_final_completion_stops_polling(self):
        for frame_type in (Step2Frame, Step3Frame):
            with self.subTest(step=frame_type.__name__):
                frame, missing = self._frame(frame_type)
                progress, done = mock.Mock(), mock.Mock()
                for _ in range(301):
                    frame._worker_ui_events.put((progress, False))
                frame._worker_ui_events.put((done, True))
                frame._poll_worker_dispatch()
                self.assertEqual(progress.call_count, 200)
                done.assert_not_called()
                self.assertIsNotNone(frame._worker_ui_poll_after_id)
                frame._poll_worker_dispatch()
                self.assertEqual(progress.call_count, 301)
                done.assert_called_once_with()
                self.assertIsNone(frame._worker_ui_poll_after_id)
                missing.assert_not_called()

    def test_only_dead_workers_with_missing_results_fail_after_five_empty_polls(self):
        for frame_type in (Step2Frame, Step3Frame):
            for alive in (True, False):
                with self.subTest(step=frame_type.__name__, alive=alive):
                    frame, missing = self._frame(frame_type, alive=alive)
                    for _ in range(8 if alive else 5):
                        frame._poll_worker_dispatch()
                    self.assertEqual(missing.call_count, 0 if alive else 1)
                    self.assertEqual(frame.after.call_count, 9 if alive else 5)

    def test_stop_detaches_the_queue_without_draining_a_live_producer(self):
        for frame_type in (Step2Frame, Step3Frame):
            with self.subTest(step=frame_type.__name__):
                frame, _missing = self._frame(frame_type)
                events = mock.Mock()
                frame._worker_ui_events = events
                frame._stop_worker_dispatch()
                events.empty.assert_not_called()
                events.get_nowait.assert_not_called()
                self.assertIsNone(frame._worker_ui_events)
                frame.after_cancel.assert_called_once_with("worker-poll")

    def test_generation_change_during_completion_keeps_only_the_new_poll(self):
        for frame_type in (Step2Frame, Step3Frame):
            with self.subTest(step=frame_type.__name__):
                frame, _missing = self._frame(frame_type)
                new_worker = mock.Mock(is_alive=mock.Mock(return_value=True))
                frame._worker_ui_events.put((lambda: frame._start_worker_dispatch(new_worker, mock.Mock()), True))
                frame._poll_worker_dispatch()
                self.assertEqual(frame.after.call_count, 2)
                self.assertEqual(frame._worker_ui_poll_after_id, "worker-poll")

    def test_closing_drops_further_worker_updates_and_polls(self):
        for frame_type in (Step2Frame, Step3Frame):
            with self.subTest(step=frame_type.__name__):
                frame, missing = self._frame(frame_type)
                callback = mock.Mock()
                frame._closing = True
                frame._stop_worker_dispatch()
                frame._post_worker_ui(callback, terminal=True)
                frame._poll_worker_dispatch()
                callback.assert_not_called()
                missing.assert_not_called()
                self.assertEqual(frame.after.call_count, 1)


if __name__ == "__main__":
    unittest.main()
