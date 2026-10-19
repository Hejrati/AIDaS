from pathlib import Path
import queue
import threading
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step4_compiler_dialog import Step4CompilerDialog
from aidas.steps.step5_compile import Step5Frame


class CompilerPresentationTests(unittest.TestCase):
    VIEWS = (Step4CompilerDialog, Step5Frame)

    def _view(self, view_type):
        view = object.__new__(view_type)
        view._events = queue.Queue()
        view._append_log = mock.Mock()
        view._update_progress = mock.Mock()
        view._compilation_succeeded = mock.Mock()
        view._compilation_failed = mock.Mock()
        return view

    def test_worker_delivers_events_in_order_without_touching_the_view(self):
        for view_type in self.VIEWS:
            with self.subTest(view=view_type.__name__):
                view = self._view(view_type)
                result = SimpleNamespace(output_path=Path("compiled.xlsx"))
                backend_threads = []

                def compile_results(root, output, **options):
                    self.assertEqual((root, output), (Path("root"), result.output_path))
                    self.assertIs(options["include_fovea"], False)
                    backend_threads.append(threading.get_ident())
                    options["log_callback"]("Scanning")
                    options["progress_callback"](2, 5, "Working")
                    return result

                patch_target = (
                    "aidas.steps.step4_compiler_dialog.compile_step4_results"
                    if view_type is Step4CompilerDialog else
                    "aidas.services.step4_compiler.compile_step4_results"
                )
                with mock.patch(patch_target, side_effect=compile_results):
                    worker = threading.Thread(
                        target=view._compile_worker,
                        args=(Path("root"), result.output_path, False),
                    )
                    worker.start()
                    worker.join(timeout=10)
                self.assertFalse(worker.is_alive())
                self.assertNotEqual(backend_threads, [threading.get_ident()])
                view._append_log.assert_not_called()
                view._compilation_succeeded.assert_not_called()
                observer = mock.Mock()
                observer.attach_mock(view._append_log, "log")
                observer.attach_mock(view._update_progress, "progress")
                observer.attach_mock(view._compilation_succeeded, "success")
                view._drain_compile_events()
                self.assertEqual(observer.mock_calls, [
                    mock.call.log("Scanning"),
                    mock.call.progress((2, 5, "Working")),
                    mock.call.success(result),
                ])
                view._compilation_failed.assert_not_called()

    def test_backend_errors_reach_the_existing_failure_handler(self):
        for view_type in self.VIEWS:
            with self.subTest(view=view_type.__name__):
                view = self._view(view_type)
                error = OSError("disk full")
                view._compiler_function = mock.Mock(side_effect=error)
                view._compile_worker(Path("root"), Path("compiled.xlsx"), True)
                view._compilation_failed.assert_not_called()
                view._drain_compile_events()
                view._compilation_failed.assert_called_once_with(error)
                view._compilation_succeeded.assert_not_called()

    def test_progress_keeps_each_views_fallback_and_ignores_invalid_payloads(self):
        for view_type, fallback in (
            (Step4CompilerDialog, "Compiling measurements…"),
            (Step5Frame, "Creating the Excel file…"),
        ):
            with self.subTest(view=view_type.__name__):
                view = object.__new__(view_type)
                view.progress = mock.Mock()
                view.status_var = mock.Mock()
                view._update_progress((9, 5, " "))
                view.progress.configure.assert_called_once_with(maximum=5, value=5)
                view.status_var.set.assert_called_once_with(f"{fallback} (100%)")
                view.progress.reset_mock()
                view.status_var.reset_mock()
                for payload in (None, (), ("bad", 5, "Working")):
                    view._update_progress(payload)
                view.progress.configure.assert_not_called()
                view.status_var.set.assert_not_called()

    def test_dialog_closing_does_not_drain_or_reschedule_events(self):
        view = self._view(Step4CompilerDialog)
        view._closing = True
        view._poll_after_id = "old"
        view.after = mock.Mock()
        view._events.put(("success", object()))
        view._poll_events()
        self.assertIsNone(view._poll_after_id)
        self.assertEqual(view._events.qsize(), 1)
        view.after.assert_not_called()
        view._compilation_succeeded.assert_not_called()

    def test_dialog_success_callback_can_close_without_rescheduling(self):
        view = self._view(Step4CompilerDialog)
        view._closing = False
        view.after = mock.Mock()
        view._compilation_succeeded.side_effect = lambda _result: setattr(view, "_closing", True)
        view._events.put(("success", object()))
        view._poll_events()
        view._compilation_succeeded.assert_called_once()
        view.after.assert_not_called()

    def test_step5_destroyed_during_completion_tolerates_missing_tk(self):
        view = self._view(Step5Frame)
        view.after = mock.Mock(side_effect=tk.TclError("application destroyed"))
        view._events.put(("success", object()))
        view._poll_events()
        view._compilation_succeeded.assert_called_once()

    def test_dialog_close_still_waits_for_work_and_cancels_its_poll(self):
        view = object.__new__(Step4CompilerDialog)
        view._closing = False
        view._running = True
        view._poll_after_id = "poll"
        view.after_cancel = mock.Mock()
        view.grab_release = mock.Mock()
        view.destroy = mock.Mock()
        view._on_close_callback = mock.Mock()
        with mock.patch("aidas.steps.step4_compiler_dialog.messagebox.showwarning") as warning:
            view.close()
        warning.assert_called_once()
        view.destroy.assert_not_called()
        view._running = False
        view.close()
        view.close()
        view.after_cancel.assert_called_once_with("poll")
        view.destroy.assert_called_once_with()
        view._on_close_callback.assert_called_once_with(view)
