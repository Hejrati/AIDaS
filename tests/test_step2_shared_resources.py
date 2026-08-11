from __future__ import annotations

import queue
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step2_annotate import (
    Step2BatchSegmentationSelectionPanel,
    Step2Frame,
)


class Step2SharedResourceTests(unittest.TestCase):
    def test_step3_allocation_reduces_step2_fallback_maximum(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.get_step3_core_usage = lambda: 3

        with mock.patch.object(Step2Frame, "_available_core_count", return_value=4):
            self.assertEqual(frame._shared_core_budget(), (4, 3, 1))
            self.assertEqual(frame._normalized_shared_core_limit(4), 1)

    def test_all_busy_cores_leave_one_shared_fallback_core(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.get_step3_core_usage = lambda: 4

        with mock.patch.object(Step2Frame, "_available_core_count", return_value=4):
            self.assertEqual(frame._shared_core_budget(), (4, 4, 0))
            self.assertEqual(frame._normalized_shared_core_limit(4), 1)

    def test_active_gpu_worker_does_not_reserve_fallback_cores(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame._segmenter_running = True
        frame._active_ai_core_limit = 3
        frame._active_ai_using_gpu = False

        with mock.patch.object(Step2Frame, "_available_core_count", return_value=4):
            self.assertEqual(frame.active_core_allocation(), 3)
            frame._active_ai_using_gpu = True
            self.assertEqual(frame.active_core_allocation(), 0)
            frame._segmenter_running = False
            self.assertEqual(frame.active_core_allocation(), 0)

    def test_confirmed_gpu_hides_the_fallback_core_selection(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame._ai_gpu_compatible = True
        frame._ai_runtime_using_gpu = None
        self.assertTrue(frame._gpu_execution_available())

        panel = Step2BatchSegmentationSelectionPanel.__new__(
            Step2BatchSegmentationSelectionPanel
        )
        panel.step_frame = frame
        panel.core_row = mock.Mock()
        panel.core_row.winfo_manager.return_value = "pack"
        panel._core_row_pack_options = {
            "side": "bottom",
            "fill": "x",
            "pady": (0, 8),
        }

        self.assertFalse(panel._sync_device_controls())
        panel.core_row.pack_forget.assert_called_once_with()

    def test_core_selection_returns_after_live_gpu_fallback(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame._ai_gpu_compatible = True
        frame._ai_runtime_using_gpu = False
        self.assertFalse(frame._gpu_execution_available())

        panel = Step2BatchSegmentationSelectionPanel.__new__(
            Step2BatchSegmentationSelectionPanel
        )
        panel.step_frame = frame
        panel.core_row = mock.Mock()
        panel.core_row.winfo_manager.return_value = ""
        panel._core_row_pack_options = {
            "side": "bottom",
            "fill": "x",
            "pady": (0, 8),
        }

        self.assertTrue(panel._sync_device_controls())
        panel.core_row.pack.assert_called_once_with(
            **panel._core_row_pack_options
        )

    def test_user_is_warned_before_sharing_a_fully_busy_core(self):
        frame = Step2Frame.__new__(Step2Frame)

        with mock.patch(
            "aidas.steps.step2_annotate.messagebox.askyesno",
            return_value=False,
        ) as ask:
            self.assertFalse(frame._confirm_shared_core_contention(4, 4))

        self.assertIn("may slow down AIDaS", ask.call_args.args[1])
        with mock.patch(
            "aidas.steps.step2_annotate.messagebox.askyesno"
        ) as ask:
            self.assertTrue(frame._confirm_shared_core_contention(4, 3))
            ask.assert_not_called()

    def test_device_probe_parses_isolated_worker_capability(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame._aidas_worker_command = lambda: ["worker"]
        frame._aidas_worker_env = lambda: {"AIDAS": "1"}
        frame._hidden_subprocess_kwargs = lambda: {}
        completed = SimpleNamespace(
            returncode=0,
            stdout='diagnostic\n{"compatible_gpu": true, "providers": ["DmlExecutionProvider"]}\n',
            stderr="",
        )

        with mock.patch(
            "aidas.steps.step2_annotate.subprocess.run",
            return_value=completed,
        ) as run:
            result = frame._probe_ai_device_capability()

        self.assertTrue(result["compatible_gpu"])
        self.assertEqual(run.call_args.args[0], ["worker", "--probe-device"])

    def test_runtime_status_distinguishes_gpu_from_core_processing(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.ai_device_status_var = mock.Mock()
        frame._ai_gpu_compatible = True

        frame._set_runtime_ai_device_status(
            "DirectML GPU (adapter 0)",
            "DmlExecutionProvider",
        )

        self.assertTrue(frame._active_ai_using_gpu)
        self.assertIn(
            "Using compatible GPU",
            frame.ai_device_status_var.set.call_args.args[0],
        )

        frame._set_runtime_ai_device_status(
            "ONNX Runtime CPU",
            "CPUExecutionProvider",
            "DirectML initialization failed",
        )
        self.assertFalse(frame._active_ai_using_gpu)
        self.assertIn(
            "Using core processing",
            frame.ai_device_status_var.set.call_args.args[0],
        )

    def test_device_probe_result_is_applied_by_the_tk_poll_callback(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.ai_device_status_var = mock.Mock()
        frame._ai_device_probe_results = queue.SimpleQueue()
        frame._ai_device_probe_results.put(
            {
                "compatible_gpu": True,
                "providers": ("DmlExecutionProvider",),
            }
        )

        frame._poll_ai_device_probe()

        self.assertTrue(frame._ai_gpu_compatible)
        self.assertIn(
            "Compatible DirectML GPU detected",
            frame.ai_device_status_var.set.call_args.args[0],
        )


if __name__ == "__main__":
    unittest.main()
