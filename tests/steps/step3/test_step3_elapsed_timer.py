from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from aidas.steps.step3_flatten import Step3Frame


class _Var:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class Step3ElapsedTimerTests(unittest.TestCase):
    def setUp(self):
        self.frame = Step3Frame.__new__(Step3Frame)
        self.frame.elapsed_time_var = _Var()
        self.frame.status_var = _Var()
        self.frame.progress_text_var = _Var()
        self.frame.info_var = _Var()
        self.frame._r_run_started_at = None
        self.frame._r_run_elapsed_seconds = None
        self.frame._r_elapsed_after_id = None
        self.frame._r_elapsed_generation = 0
        self.frame._pending_batch_restart = None
        self.frame._pending_batch_folders = None
        self.frame.r_batch_run_panel = None
        self.frame._set_process_buttons = mock.Mock()
        self.frame._update_continue_to_step4_button_state = mock.Mock()
        self.frame._open_batch_r_result_tabs = mock.Mock()
        self.callbacks = {}
        self.next_callback = 0

        def after(delay, callback):
            self.assertEqual(delay, 1000)
            self.next_callback += 1
            callback_id = f"timer-{self.next_callback}"
            self.callbacks[callback_id] = callback
            return callback_id

        self.frame.after = after
        self.frame.after_cancel = lambda callback_id: self.callbacks.pop(callback_id)
        self.clock = mock.patch("aidas.steps.step3_flatten.time.monotonic", return_value=100.0).start()
        self.addCleanup(mock.patch.stopall)

    def _tick(self, now):
        self.clock.return_value = now
        callback = self.callbacks.pop(self.frame._r_elapsed_after_id)
        callback()

    def test_live_timer_uses_wall_clock_even_when_event_loop_updates_are_delayed(self):
        self.frame._start_r_elapsed_timer()
        self.assertEqual(self.frame.elapsed_time_var.get(), "Elapsed: 00:00:00")
        self._tick(172.8)
        self.assertEqual(self.frame.elapsed_time_var.get(), "Elapsed: 00:01:12")
        self.assertEqual(len(self.callbacks), 1)
        self._tick(3761.2)
        self.assertEqual(self.frame.elapsed_time_var.get(), "Elapsed: 01:01:01")

    def test_final_duration_is_frozen_and_pending_callback_is_cancelled(self):
        self.frame._start_r_elapsed_timer()
        stale_callback = self.callbacks[self.frame._r_elapsed_after_id]
        self.clock.return_value = 165.9
        elapsed = self.frame._finish_r_elapsed_timer()
        self.assertAlmostEqual(elapsed, 65.9)
        self.assertEqual(self.frame.elapsed_time_var.get(), "Total elapsed: 00:01:05")
        self.assertFalse(self.callbacks)
        self.clock.return_value = 300.0
        stale_callback()
        self.assertEqual(self.frame._finish_r_elapsed_timer(), elapsed)
        self.assertEqual(self.frame.elapsed_time_var.get(), "Total elapsed: 00:01:05")
        self.assertFalse(self.callbacks)

    def test_restart_invalidates_a_callback_from_previous_run(self):
        self.frame._start_r_elapsed_timer()
        stale_callback = self.callbacks[self.frame._r_elapsed_after_id]
        self.clock.return_value = 190.0
        self.frame._start_r_elapsed_timer()
        self.assertEqual(len(self.callbacks), 1)
        stale_callback()
        self.assertEqual(len(self.callbacks), 1)
        self._tick(193.0)
        self.assertEqual(self.frame.elapsed_time_var.get(), "Elapsed: 00:00:03")

    def test_all_terminal_outcomes_include_final_duration_in_summary_and_log(self):
        for outcome, code in (("completed", 0), ("failed", 1), ("cancelled", 130)):
            with self.subTest(outcome=outcome):
                panel = mock.Mock(close_when_finished=False)
                self.frame.r_batch_run_panel = panel
                self.clock.return_value = 100.0
                self.frame._start_r_elapsed_timer()
                self.clock.return_value = 225.0
                self.frame._on_batch_r_done([
                    {"folder": Path("unused-test-folder"), "outcome": outcome, "returncode": code},
                ])
                self.assertIn("Total elapsed: 00:02:05.", self.frame.status_var.get())
                panel.set_summary.assert_called_once_with(self.frame.status_var.get())
                panel.log.assert_called_once_with(self.frame.status_var.get())
                self.assertFalse(self.frame._busy)
                self.assertFalse(self.callbacks)
                self.assertEqual(self.frame.elapsed_time_var.get(), "Total elapsed: 00:02:05")

    def test_cancellation_request_keeps_measuring_until_processes_have_exited(self):
        self.frame._busy = True
        self.frame._r_cancel_event = threading.Event()
        self.frame._terminate_active_r_processes = mock.Mock()
        self.frame._start_r_elapsed_timer()
        with mock.patch("aidas.steps.step3_flatten.notify_workflow_cancelled"), mock.patch(
            "aidas.steps.step3_flatten.threading.Thread",
        ):
            self.frame._cancel_batch_r_runs()
        self.assertTrue(self.frame._r_cancel_event.is_set())
        self._tick(145.0)
        self.assertEqual(self.frame.elapsed_time_var.get(), "Elapsed: 00:00:45")
        self.clock.return_value = 148.0
        self.frame._on_batch_r_done([
            {"folder": Path("unused-test-folder"), "outcome": "cancelled", "returncode": 130},
        ])
        self.assertEqual(self.frame.elapsed_time_var.get(), "Total elapsed: 00:00:48")
        self.assertFalse(self.callbacks)

    def test_worker_start_failure_freezes_timer_and_reports_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            main = directory / "main.R"
            output = directory / "output.R"
            main.touch()
            output.touch()
            self.frame._busy = False
            self.frame._r_cancel_event = threading.Event()
            self.frame._ensure_r_ready_with_wizard = mock.Mock(return_value=Path("Rscript.exe"))
            self.frame._available_r_worker_limit = mock.Mock(return_value=1)
            self.frame._clear_plot_holder = mock.Mock()
            self.frame.plot_holder = mock.Mock()
            panel = mock.Mock()

            def cannot_start():
                self.clock.return_value = 102.0
                raise RuntimeError("worker unavailable")

            with mock.patch("aidas.steps.step3_flatten.RBatchRunPanel", return_value=panel), mock.patch(
                "aidas.steps.step3_flatten.threading.Thread",
            ) as worker, mock.patch("aidas.steps.step3_flatten.messagebox.showerror"):
                worker.return_value.start.side_effect = cannot_start
                self.frame._start_batch_r_runs([directory], 1, main, output)
            self.assertFalse(self.frame._busy)
            self.assertFalse(self.callbacks)
            self.assertEqual(self.frame.elapsed_time_var.get(), "Total elapsed: 00:00:02")
            self.assertIn("Total elapsed: 00:00:02.", self.frame.status_var.get())
            panel.finish.assert_called_once_with()

    def test_reset_stops_the_timer_and_clears_previous_duration(self):
        self.frame._start_r_elapsed_timer()
        stale_callback = self.callbacks[self.frame._r_elapsed_after_id]
        self.frame._render = mock.Mock()
        self.frame._reset_to_tutorial_state()
        stale_callback()
        self.assertFalse(self.callbacks)
        self.assertIsNone(self.frame._r_run_started_at)
        self.assertIsNone(self.frame._r_run_elapsed_seconds)
        self.assertEqual(self.frame.elapsed_time_var.get(), "")

    def test_destroy_cancels_timer_before_destroying_tk_widgets(self):
        self.frame._start_r_elapsed_timer()
        with mock.patch("aidas.steps.step3_flatten.SidebarStepFrame.destroy") as destroy:
            self.frame.destroy()
        destroy.assert_called_once_with()
        self.assertFalse(self.callbacks)
        self.assertIsNone(self.frame._r_run_started_at)


if __name__ == "__main__":
    unittest.main()
