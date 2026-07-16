"""ONNX Runtime inference for the AI_ForAIDAS OCT segmentation model.

On Windows, ``onnxruntime-directml`` supplies both the DirectML and CPU
execution providers.  Automatic mode prefers DirectML and transparently
retries on CPU if the GPU provider cannot initialize or execute this model.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Optional

import numpy as np


NUM_BOUNDARIES = 6
DIRECTML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class PredictionResult:
    boundaries: np.ndarray
    fovea_x: Optional[int]
    device: str
    boundary_model_path: str
    execution_provider: str
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class ProviderSelection:
    requested: str
    providers: tuple
    uses_directml: bool
    device_label: str


class AIForAIDASPredictor:
    """Reusable ONNX predictor that keeps one optimized session in memory."""

    def __init__(
        self,
        *,
        boundary_model_path: str,
        provider_name: str = "auto",
        device_id: int = 0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        if not os.path.isfile(boundary_model_path):
            raise FileNotFoundError(f"AI_ForAIDAS ONNX model not found: {boundary_model_path}")

        self._progress_callback = progress_callback
        self.boundary_model_path = os.path.abspath(boundary_model_path)
        self.requested_provider = _normalize_provider_name(provider_name)
        self.device_id = _validate_device_id(device_id)
        self.fallback_reason = None

        self._report_progress("importing_onnx_runtime", 0.10)
        self._ort = _import_onnxruntime()
        self._report_progress("resolving_execution_provider", 0.16)
        self._selection = choose_execution_providers(
            self._ort.get_available_providers(),
            requested=self.requested_provider,
            device_id=self.device_id,
        )
        self._report_progress("loading_onnx_model", 0.24)
        self._session = self._create_session_with_cpu_fallback(self._selection)
        self._validate_model_contract()
        self._report_progress("model_ready", 0.45)

    @property
    def execution_provider(self) -> str:
        providers = tuple(self._session.get_providers())
        if DIRECTML_PROVIDER in providers and self._selection.uses_directml:
            return DIRECTML_PROVIDER
        return providers[0] if providers else CPU_PROVIDER

    @property
    def device(self) -> str:
        if self.execution_provider == DIRECTML_PROVIDER:
            return f"DirectML GPU (adapter {self.device_id})"
        return "ONNX Runtime CPU"

    def _report_progress(self, stage: str, fraction: float) -> None:
        if self._progress_callback is not None:
            self._progress_callback(stage, fraction)

    def _create_session_with_cpu_fallback(self, selection: ProviderSelection):
        try:
            return _create_onnx_session(
                self._ort,
                self.boundary_model_path,
                selection,
            )
        except Exception as exc:
            if not selection.uses_directml or selection.requested != "auto":
                raise RuntimeError(
                    f"Could not load the AI_ForAIDAS ONNX model with {selection.device_label}: {exc}"
                ) from exc
            self.fallback_reason = f"DirectML initialization failed: {exc}"
            self._report_progress("falling_back_to_cpu", 0.30)
            self._selection = choose_execution_providers(
                self._ort.get_available_providers(),
                requested="cpu",
                device_id=self.device_id,
            )
            return _create_onnx_session(
                self._ort,
                self.boundary_model_path,
                self._selection,
            )

    def _validate_model_contract(self) -> None:
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                "AI_ForAIDAS ONNX model must have exactly one input and one output; "
                f"found {len(inputs)} input(s) and {len(outputs)} output(s)."
            )
        if inputs[0].type != "tensor(float)":
            raise RuntimeError(
                f"AI_ForAIDAS ONNX input must be float32, found {inputs[0].type}."
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def _run_model(self, input_tensor: np.ndarray) -> np.ndarray:
        try:
            return self._session.run(
                [self._output_name],
                {self._input_name: input_tensor},
            )[0]
        except Exception as exc:
            if not self._selection.uses_directml or self.requested_provider != "auto":
                raise RuntimeError(
                    f"AI_ForAIDAS ONNX inference failed on {self.device}: {exc}"
                ) from exc

            self.fallback_reason = f"DirectML execution failed: {exc}"
            self._report_progress("falling_back_to_cpu", 0.72)
            self._selection = choose_execution_providers(
                self._ort.get_available_providers(),
                requested="cpu",
                device_id=self.device_id,
            )
            self._session = _create_onnx_session(
                self._ort,
                self.boundary_model_path,
                self._selection,
            )
            self._validate_model_contract()
            try:
                return self._session.run(
                    [self._output_name],
                    {self._input_name: input_tensor},
                )[0]
            except Exception as cpu_exc:
                raise RuntimeError(
                    "AI_ForAIDAS ONNX inference failed on DirectML and on the CPU fallback.\n\n"
                    f"DirectML error: {exc}\n\nCPU error: {cpu_exc}"
                ) from cpu_exc

    def predict(self, image: np.ndarray) -> PredictionResult:
        self._report_progress("normalizing_image", 0.52)
        image_norm = _normalize_for_model(image)
        height, width = image_norm.shape

        self._report_progress("preparing_onnx_input", 0.60)
        input_tensor = np.ascontiguousarray(image_norm[np.newaxis, np.newaxis], dtype=np.float32)
        self._report_progress("running_onnx_model", 0.68)
        logits = np.asarray(self._run_model(input_tensor), dtype=np.float32)

        expected_prefix = (1,)
        if logits.ndim != 4 or logits.shape[:1] != expected_prefix:
            raise RuntimeError(
                f"AI_ForAIDAS ONNX model returned invalid logits shape {logits.shape}; "
                "expected (1, boundaries, height, width)."
            )
        if logits.shape[2:] != (height, width):
            raise RuntimeError(
                f"AI_ForAIDAS ONNX model returned spatial shape {logits.shape[2:]}; "
                f"expected {(height, width)}."
            )

        self._report_progress("postprocessing", 0.88)
        pred_y = _soft_argmax_y_numpy(logits)[0]
        if pred_y.shape[0] < NUM_BOUNDARIES:
            raise RuntimeError(
                f"AI_ForAIDAS returned {pred_y.shape[0]} boundary rows; expected {NUM_BOUNDARIES}."
            )

        boundaries = np.sort(pred_y[:NUM_BOUNDARIES], axis=0).astype(np.float32, copy=False)
        boundaries = np.clip(boundaries, 0, height - 1)

        return PredictionResult(
            boundaries=boundaries,
            fovea_x=None,
            device=self.device,
            boundary_model_path=self.boundary_model_path,
            execution_provider=self.execution_provider,
            fallback_reason=self.fallback_reason,
        )


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "AI_ForAIDAS requires ONNX Runtime. On Windows, install "
            "onnxruntime-directml from requirements.txt; on other systems, install "
            "the CPU onnxruntime package.\n\n"
            f"Original error: {exc}"
        ) from exc
    return ort


def _normalize_provider_name(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    aliases = {
        "directml": "dml",
        "gpu": "dml",
        "onnx-cpu": "cpu",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "dml", "cpu"}:
        raise ValueError(f"Unsupported ONNX execution provider request: {requested}")
    return value


def _validate_device_id(device_id: int) -> int:
    value = int(device_id)
    if value < 0:
        raise ValueError("DirectML device ID must be zero or greater.")
    return value


def choose_execution_providers(
    available_providers,
    *,
    requested: str = "auto",
    device_id: int = 0,
) -> ProviderSelection:
    """Return a deterministic DirectML/CPU provider configuration."""
    requested = _normalize_provider_name(requested)
    device_id = _validate_device_id(device_id)
    available = tuple(str(provider) for provider in available_providers)

    if requested in {"auto", "dml"} and DIRECTML_PROVIDER in available:
        return ProviderSelection(
            requested=requested,
            providers=(
                (DIRECTML_PROVIDER, {"device_id": str(device_id)}),
                CPU_PROVIDER,
            ),
            uses_directml=True,
            device_label=f"DirectML GPU adapter {device_id}",
        )

    if requested == "dml":
        raise RuntimeError(
            "DirectML was requested, but DmlExecutionProvider is not available. "
            "Install onnxruntime-directml on Windows and update the GPU driver. "
            f"Available providers: {', '.join(available) or '(none)'}"
        )

    if CPU_PROVIDER not in available:
        raise RuntimeError(
            "ONNX Runtime CPUExecutionProvider is unavailable. "
            f"Available providers: {', '.join(available) or '(none)'}"
        )

    return ProviderSelection(
        requested=requested,
        providers=(CPU_PROVIDER,),
        uses_directml=False,
        device_label="ONNX Runtime CPU",
    )


def _create_onnx_session(ort, model_path: str, selection: ProviderSelection):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if selection.uses_directml:
        # Required by the DirectML execution provider.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=list(selection.providers),
    )


def _normalize_for_model(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"AI_ForAIDAS expects a 2-D grayscale image, got shape {arr.shape}.")
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
    arr = arr.astype(np.float32, copy=False)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.ascontiguousarray((arr - lo) / (hi - lo + 1e-8), dtype=np.float32)


def _soft_argmax_y_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 4:
        raise ValueError(f"Expected 4-D ONNX logits, got shape {logits.shape}.")
    height = logits.shape[2]
    stable_logits = logits - np.max(logits, axis=2, keepdims=True)
    probabilities = np.exp(stable_logits)
    probabilities /= np.sum(probabilities, axis=2, keepdims=True)
    y_indices = np.arange(height, dtype=np.float32).reshape(1, 1, height, 1)
    return np.sum(probabilities * y_indices, axis=2, dtype=np.float32)


def predict_boundaries_and_fovea(
    image: np.ndarray,
    *,
    boundary_model_path: str,
    provider_name: str = "auto",
    device_id: int = 0,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> PredictionResult:
    """Run ONNX AI_ForAIDAS inference on a 2-D OCT image."""
    predictor = AIForAIDASPredictor(
        boundary_model_path=boundary_model_path,
        provider_name=provider_name,
        device_id=device_id,
        progress_callback=progress_callback,
    )
    result = predictor.predict(image)
    if progress_callback is not None:
        progress_callback("prediction_ready", 0.95)
    return result
