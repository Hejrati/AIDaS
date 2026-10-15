from __future__ import annotations

import queue
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step2_annotate import (
    BOUNDARY_NAMES,
    Step2BatchSegmentationSelectionPanel,
    Step2Frame,
)


class Step2SharedResourceTests(unittest.TestCase):
    def test_idle_fallback_leaves_one_core_for_the_app(self):
        frame = Step2Frame.__new__(Step2Frame)

        with mock.patch.object(Step2Frame, "_available_core_count", return_value=4):
            self.assertEqual(frame._shared_core_budget(), (4, 0, 3))
            self.assertEqual(frame._normalized_shared_core_limit(), 3)

    def test_step3_allocation_reduces_step2_fallback_maximum(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.get_step3_core_usage = lambda: 1

        with mock.patch.object(Step2Frame, "_available_core_count", return_value=4):
            self.assertEqual(frame._shared_core_budget(), (4, 1, 2))
            self.assertEqual(frame._normalized_shared_core_limit(), 2)

    def test_legacy_core_limits_cannot_override_the_automatic_budget(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.get_step3_core_usage = lambda: 1

        with mock.patch.object(Step2Frame, "_available_core_count", return_value=4):
            for stale_limit in (None, 0, 1, 4, 999, "invalid"):
                with self.subTest(stale_limit=stale_limit):
                    self.assertEqual(
                        frame._normalized_shared_core_limit(stale_limit), 2
                    )

    def test_single_or_unknown_cpu_count_still_allows_one_fallback_worker(self):
        frame = Step2Frame.__new__(Step2Frame)

        for count in (None, 1):
            with self.subTest(count=count), mock.patch(
                "aidas.steps.step2_annotate.os.process_cpu_count",
                return_value=None,
                create=True,
            ), mock.patch(
                "aidas.steps.step2_annotate.os.sched_getaffinity",
                side_effect=OSError,
                create=True,
            ), mock.patch(
                "aidas.steps.step2_annotate.os.cpu_count", return_value=count
            ):
                self.assertEqual(frame._available_core_count(), 1)
                self.assertEqual(frame._shared_core_budget(), (1, 0, 0))
                self.assertEqual(frame._normalized_shared_core_limit(), 1)

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

    def test_confirmed_gpu_is_available_without_fallback_core_controls(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame._ai_gpu_compatible = True
        frame._ai_runtime_using_gpu = None
        self.assertTrue(frame._gpu_execution_available())

    def test_live_gpu_fallback_uses_the_automatic_cpu_budget(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame._ai_gpu_compatible = True
        frame._ai_runtime_using_gpu = False
        self.assertFalse(frame._gpu_execution_available())

        with mock.patch.object(Step2Frame, "_available_core_count", return_value=4):
            self.assertEqual(frame._normalized_shared_core_limit(), 3)

    def test_batch_selection_starts_without_a_configurable_core_limit(self):
        panel = Step2BatchSegmentationSelectionPanel.__new__(
            Step2BatchSegmentationSelectionPanel
        )
        panel.step_frame = mock.Mock()
        panel.root_dir = "scan-root"
        rows = [{"image_paths": ["scan-root/Light.img"]}]
        panel.table = mock.Mock()
        panel.table.selected_rows.return_value = rows

        panel._run_selected()

        panel.step_frame._start_step2_batch_segmentation_from_rows.assert_called_once_with(
            rows, "scan-root"
        )

    def test_batch_launch_recalculates_the_budget_instead_of_using_a_legacy_limit(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.get_step3_core_usage = lambda: 1
        frame._image_pair_key = lambda path: path
        frame._close_step2_batch_segmentation_panel = mock.Mock()
        frame.status_var = mock.Mock()
        frame._show_single_image_canvas = mock.Mock()
        frame._collect_folder_fovea_lines = mock.Mock(return_value={})
        frame._run_aidas_batch_segmentation = mock.Mock()
        image_paths = ["scan-root/Light.img"]

        with mock.patch.object(Step2Frame, "_available_core_count", return_value=4):
            frame._start_step2_batch_segmentation_from_rows(
                [{"image_paths": image_paths}], "scan-root", core_limit=1
            )

        frame._run_aidas_batch_segmentation.assert_called_once_with(
            image_paths=image_paths,
            manual_fovea_by_path={},
            image_side_by_path=None,
            core_limit=2,
        )

    def test_single_image_worker_receives_the_automatic_fallback_budget(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.image_data = [[0]]
        frame._segmenter_running = False
        frame._all_required_boundaries_complete = lambda: False
        frame.aidas_model_path = "boundary-model.onnx"
        frame.fovea_x = 0
        frame.current_file = "scan-root/Light.img"
        frame.active_boundary = BOUNDARY_NAMES[0]
        frame.image_canvas = mock.Mock()
        frame._ensure_fovea_line = mock.Mock()
        frame._capture_current_editor_state = lambda: {
            "fovea_x": 0,
            "save_orientation": "temporal",
        }
        frame._image_pair_key = lambda path: path
        frame._set_segmentation_running = mock.Mock()
        frame._run_aidas_batch_segmenter_worker = mock.Mock()

        for step3_used, expected in ((0, 3), (1, 2)):
            frame.get_step3_core_usage = lambda: step3_used
            with self.subTest(step3_used=step3_used), mock.patch.object(
                Step2Frame, "_available_core_count", return_value=4
            ), mock.patch(
                "aidas.steps.step2_annotate.os.path.isfile", return_value=True
            ), mock.patch(
                "aidas.steps.step2_annotate.threading.Thread"
            ) as thread:
                frame._segment_current_image()

                self.assertEqual(thread.call_args.kwargs["args"][4], expected)
                self.assertEqual(frame._active_ai_core_limit, expected)
                thread.return_value.start.assert_called_once_with()

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
