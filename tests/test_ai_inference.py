from __future__ import annotations

from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from aidas.ai.inference import (
    AIForAIDASPredictor,
    CPU_PROVIDER,
    DIRECTML_PROVIDER,
    _normalize_for_model,
    _soft_argmax_y_numpy,
    choose_execution_providers,
    probe_execution_devices,
)


class ProviderSelectionTests(unittest.TestCase):
    def test_device_probe_reports_directml_compatibility(self):
        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: [DIRECTML_PROVIDER, CPU_PROVIDER]
        )
        with mock.patch(
            "aidas.ai.inference._import_onnxruntime",
            return_value=fake_ort,
        ):
            result = probe_execution_devices()

        self.assertTrue(result["compatible_gpu"])
        self.assertEqual(
            result["providers"],
            (DIRECTML_PROVIDER, CPU_PROVIDER),
        )

    def test_auto_prefers_directml_and_keeps_cpu_fallback(self):
        selection = choose_execution_providers(
            [CPU_PROVIDER, DIRECTML_PROVIDER],
            requested="auto",
            device_id=2,
        )
        self.assertTrue(selection.uses_directml)
        self.assertEqual(selection.providers[0], (DIRECTML_PROVIDER, {"device_id": "2"}))
        self.assertEqual(selection.providers[1], CPU_PROVIDER)

    def test_auto_uses_cpu_when_directml_is_unavailable(self):
        selection = choose_execution_providers([CPU_PROVIDER], requested="auto")
        self.assertFalse(selection.uses_directml)
        self.assertEqual(selection.providers, (CPU_PROVIDER,))

    def test_explicit_directml_reports_available_providers(self):
        with self.assertRaisesRegex(RuntimeError, "DmlExecutionProvider is not available"):
            choose_execution_providers([CPU_PROVIDER], requested="dml")

    def test_negative_adapter_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero or greater"):
            choose_execution_providers([CPU_PROVIDER], requested="auto", device_id=-1)


class ArrayProcessingTests(unittest.TestCase):
    def test_normalization_is_float32_contiguous_and_bounded(self):
        image = np.array([[10, 20], [30, 40]], dtype=">i2")
        normalized = _normalize_for_model(image)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(normalized.flags.c_contiguous)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_soft_argmax_uses_the_height_axis(self):
        logits = np.zeros((1, 2, 5, 3), dtype=np.float32)
        result = _soft_argmax_y_numpy(logits)
        np.testing.assert_allclose(result, np.full((1, 2, 3), 2.0, dtype=np.float32))


class _FakeTensorInfo:
    def __init__(self, name, tensor_type="tensor(float)"):
        self.name = name
        self.type = tensor_type


class _FakeSession:
    def __init__(self, providers, *, fail_run=False):
        self._providers = [item[0] if isinstance(item, tuple) else item for item in providers]
        self._fail_run = fail_run

    def get_inputs(self):
        return [_FakeTensorInfo("image")]

    def get_outputs(self):
        return [_FakeTensorInfo("logits")]

    def get_providers(self):
        return list(self._providers)

    def run(self, output_names, feeds):
        if self._fail_run:
            raise RuntimeError("simulated DirectML dispatch failure")
        image = feeds["image"]
        return [np.zeros((1, 6, image.shape[2], image.shape[3]), dtype=np.float32)]


class _FakeOrt:
    class SessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self.enable_mem_pattern = True
            self.execution_mode = None
            self.intra_op_num_threads = None
            self.inter_op_num_threads = None

    GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL="all")
    ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="sequential")

    def __init__(self, *, fail_directml_init=False, fail_directml_run=False):
        self.fail_directml_init = fail_directml_init
        self.fail_directml_run = fail_directml_run
        self.session_calls = []

    def get_available_providers(self):
        return [DIRECTML_PROVIDER, CPU_PROVIDER]

    def InferenceSession(self, model_path, sess_options, providers):
        self.session_calls.append((model_path, sess_options, providers))
        first = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if first == DIRECTML_PROVIDER and self.fail_directml_init:
            raise RuntimeError("simulated DirectML initialization failure")
        return _FakeSession(
            providers,
            fail_run=(first == DIRECTML_PROVIDER and self.fail_directml_run),
        )


class PredictorFallbackTests(unittest.TestCase):
    def _model_path(self, directory):
        path = Path(directory) / "model.onnx"
        path.write_bytes(b"fake model for mocked runtime")
        return path

    def test_initialization_failure_falls_back_to_cpu(self):
        fake_ort = _FakeOrt(fail_directml_init=True)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("aidas.ai.inference._import_onnxruntime", return_value=fake_ort):
                predictor = AIForAIDASPredictor(
                    boundary_model_path=str(self._model_path(directory)),
                    provider_name="auto",
                )
        self.assertEqual(predictor.execution_provider, CPU_PROVIDER)
        self.assertIn("initialization failed", predictor.fallback_reason)
        self.assertEqual(len(fake_ort.session_calls), 2)

    def test_execution_failure_retries_once_on_cpu(self):
        fake_ort = _FakeOrt(fail_directml_run=True)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("aidas.ai.inference._import_onnxruntime", return_value=fake_ort):
                predictor = AIForAIDASPredictor(
                    boundary_model_path=str(self._model_path(directory)),
                    provider_name="auto",
                )
                result = predictor.predict(np.arange(35, dtype=np.uint16).reshape(5, 7))

        self.assertEqual(result.execution_provider, CPU_PROVIDER)
        self.assertEqual(result.device, "ONNX Runtime CPU")
        self.assertIn("execution failed", result.fallback_reason)
        self.assertEqual(result.boundaries.shape, (6, 7))
        self.assertEqual(len(fake_ort.session_calls), 2)

    def test_core_limit_configures_every_fallback_session(self):
        fake_ort = _FakeOrt(fail_directml_init=True)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("aidas.ai.inference._import_onnxruntime", return_value=fake_ort):
                AIForAIDASPredictor(
                    boundary_model_path=str(self._model_path(directory)),
                    provider_name="auto",
                    core_limit=3,
                )

        self.assertEqual(len(fake_ort.session_calls), 2)
        for _model, options, _providers in fake_ort.session_calls:
            self.assertEqual(options.intra_op_num_threads, 3)
            self.assertEqual(options.inter_op_num_threads, 1)
            self.assertEqual(options.execution_mode, "sequential")

    def test_predictor_rejects_nonpositive_core_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "one or greater"):
                AIForAIDASPredictor(
                    boundary_model_path=str(self._model_path(directory)),
                    core_limit=0,
                )


if __name__ == "__main__":
    unittest.main()
