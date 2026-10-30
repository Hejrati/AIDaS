"""Worker-start failures must restore both compiler views without Tk."""

from pathlib import Path
import queue
import unittest
from unittest import mock

from aidas.steps.step4_compiler_dialog import Step4CompilerDialog
from aidas.steps.step5_compile import Step5Frame


class CompilerWorkerLifecycleTests(unittest.TestCase):
    VIEWS = (Step4CompilerDialog, Step5Frame)

    def _view(self, view_type):
        view = object.__new__(view_type)
        view._events = queue.Queue()
        view._running = False
        view._worker = object()  # A previous run must not become a live watch.
        view._clear_log = mock.Mock()
        view._append_log = mock.Mock()
        view._compilation_succeeded = mock.Mock()
        view._update_run_button_state = mock.Mock()
        view.status_var = mock.Mock()
        view.after = mock.Mock()
        view.after_cancel = mock.Mock()
        for name in (
            "input_entry", "input_browse_button", "include_fovea_checkbox",
            "output_entry", "output_browse_button", "run_button", "progress",
        ):
            setattr(view, name, mock.Mock())
        view.run_button_disabled_icon = object()
        return view

    def test_thread_construction_failure_restores_controls_without_queued_work(self):
        for view_type in self.VIEWS:
            with self.subTest(view=view_type.__name__):
                view = self._view(view_type)
                error = RuntimeError("cannot allocate a compiler thread")
                with mock.patch("aidas.ui.compiler.threading.Thread", side_effect=error), \
                        mock.patch("aidas.steps.step4_compiler_dialog.messagebox.showerror") as popup:
                    view._start_compiler(
                        Path("root"), Path("output.xlsx"), True,
                        thread_name="compiler-test",
                    )
                self._assert_failed_start(view, error, popup)

    def test_thread_start_failure_restores_controls_without_queued_work(self):
        for view_type in self.VIEWS:
            with self.subTest(view=view_type.__name__):
                view = self._view(view_type)
                error = RuntimeError("cannot start a new thread")
                worker = mock.Mock()
                worker.start.side_effect = error
                with mock.patch("aidas.ui.compiler.threading.Thread", return_value=worker), \
                        mock.patch("aidas.steps.step4_compiler_dialog.messagebox.showerror") as popup:
                    view._start_compiler(
                        Path("root"), Path("output.xlsx"), False,
                        thread_name="compiler-test",
                    )
                worker.start.assert_called_once_with()
                self._assert_failed_start(view, error, popup)

    def _assert_failed_start(self, view, error, popup):
        self.assertFalse(view._running)
        self.assertIsNone(view._worker)
        self.assertTrue(view._events.empty())
        view._compilation_succeeded.assert_not_called()
        view.after.assert_not_called()
        view.after_cancel.assert_not_called()
        view.input_entry.configure.assert_called_with(state="normal")
        view.output_entry.configure.assert_called_with(state="normal")
        status = view.status_var.set.call_args.args[0]
        if isinstance(view, Step4CompilerDialog):
            self.assertEqual(status, "Compilation failed. See the log for details.")
        else:
            self.assertEqual(status, f"Could not create the Excel file: {error}")
        popup.assert_called_once_with(
            "Compilation failed" if isinstance(view, Step4CompilerDialog)
            else "Excel file creation failed",
            str(error), parent=view,
        )

    def test_successful_start_keeps_one_worker_and_its_original_arguments(self):
        for view_type in self.VIEWS:
            with self.subTest(view=view_type.__name__):
                view = self._view(view_type)
                worker = mock.Mock()
                with mock.patch("aidas.ui.compiler.threading.Thread", return_value=worker) as create:
                    view._start_compiler(
                        Path("root"), Path("output.xlsx"), False,
                        thread_name="compiler-test",
                    )
                self.assertTrue(view._running)
                self.assertIs(view._worker, worker)
                create.assert_called_once_with(
                    target=view._compile_worker,
                    args=(Path("root"), Path("output.xlsx"), False),
                    name="compiler-test", daemon=True,
                )
                worker.start.assert_called_once_with()
                self.assertTrue(view._events.empty())
                view._compilation_succeeded.assert_not_called()
                if isinstance(view, Step5Frame):
                    view.after.assert_called_once_with(view.EVENT_POLL_MS, view._poll_events)
                else:
                    view.after.assert_not_called()


if __name__ == "__main__":
    unittest.main()
