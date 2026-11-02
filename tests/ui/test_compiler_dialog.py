from __future__ import annotations

from pathlib import Path
import queue
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step4_compiler_dialog import Step4CompilerDialog
from aidas.services.measurement_scan import MeasurementScanLimitExceeded
from aidas.utils.worker_polling import WorkerCompletionGuard


class _VariableStub:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _ProgressStub:
    def __init__(self, **options):
        self.options = {"maximum": 1, "value": 0, **options}

    def configure(self, **options):
        self.options.update(options)

    def cget(self, option):
        return self.options[option]


class Step4CompilerDialogTests(unittest.TestCase):
    def _polling_dialog(self, *, alive=False):
        dialog = object.__new__(Step4CompilerDialog)
        dialog._closing = False
        dialog._running = True
        dialog._events = queue.Queue()
        dialog._poll_after_id = None
        dialog._completion_guard = WorkerCompletionGuard(SimpleNamespace(is_alive=lambda: alive))
        dialog.after = mock.Mock(return_value="poll")
        dialog._append_log = mock.Mock()
        dialog._compilation_failed = mock.Mock(side_effect=lambda _error: setattr(dialog, "_running", False))
        return dialog

    def test_idle_compiler_does_not_schedule_an_endless_timer(self):
        dialog = self._polling_dialog()
        dialog._running = False
        dialog._poll_events()
        dialog.after.assert_not_called()
        self.assertIsNone(dialog._poll_after_id)

    def test_dead_compiler_without_result_finishes_on_fifth_check(self):
        dialog = self._polling_dialog()
        for _ in range(5):
            dialog._poll_events()
        self.assertEqual(dialog.after.call_count, 4)
        dialog._compilation_failed.assert_called_once()
        self.assertFalse(dialog._running)
        self.assertIsNone(dialog._poll_after_id)

    def test_live_compiler_has_no_five_poll_cutoff(self):
        dialog = self._polling_dialog(alive=True)
        for _ in range(25):
            dialog._poll_events()
        dialog._compilation_failed.assert_not_called()
        self.assertEqual(dialog.after.call_count, 25)

    def test_limit_error_stays_inline_without_popup(self):
        dialog = object.__new__(Step4CompilerDialog)
        dialog.status_var = _VariableStub()
        dialog._set_running = mock.Mock()
        dialog._append_log = mock.Mock()
        error = MeasurementScanLimitExceeded()
        with mock.patch("aidas.steps.step4_compiler_dialog.messagebox.showerror") as popup:
            dialog._compilation_failed(error)
        dialog._set_running.assert_called_once_with(False)
        self.assertEqual(dialog.status_var.get(), str(error))
        popup.assert_not_called()

    def test_progress_uses_real_completed_and_total_units(self):
        dialog = object.__new__(Step4CompilerDialog)
        dialog.progress = _ProgressStub()
        dialog.status_var = _VariableStub()

        dialog._update_progress((3, 5, "Processed ELM-RPE: LE 007."))

        self.assertEqual(dialog.progress.options["maximum"], 5)
        self.assertEqual(dialog.progress.options["value"], 3)
        self.assertEqual(
            dialog.status_var.get(),
            "Processed ELM-RPE: LE 007. (60%)",
        )

    def test_progress_clamps_out_of_range_values(self):
        dialog = object.__new__(Step4CompilerDialog)
        dialog.progress = _ProgressStub()
        dialog.status_var = _VariableStub()

        dialog._update_progress((9, 5, "Finishing"))

        self.assertEqual(dialog.progress.options["value"], 5)
        self.assertEqual(dialog.status_var.get(), "Finishing (100%)")

    def test_success_forces_full_bar_without_showing_a_popup(self):
        dialog = object.__new__(Step4CompilerDialog)
        dialog.progress = _ProgressStub(maximum=5, value=4)
        dialog.output_var = _VariableStub("old.xlsx")
        dialog.status_var = _VariableStub()
        dialog._set_running = mock.Mock()
        dialog._on_success_callback = mock.Mock()
        dialog.update_idletasks = mock.Mock()
        output = Path("compiled.xlsx")

        with mock.patch(
            "aidas.steps.step4_compiler_dialog.messagebox.showinfo"
        ) as showinfo:
            dialog._compilation_succeeded(SimpleNamespace(output_path=output))

        dialog._set_running.assert_called_once_with(False)
        self.assertEqual(dialog.progress.options["value"], 5.0)
        self.assertEqual(dialog.output_var.get(), str(output))
        self.assertEqual(dialog.status_var.get(), f"Saved workbook: {output}")
        dialog.update_idletasks.assert_called_once_with()
        dialog._on_success_callback.assert_called_once_with(output)
        showinfo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
