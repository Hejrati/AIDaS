"""Compare AI array processing with its original float32 implementation.

Run with the app's Python: python release_tools/profile_ai.py
Add --with-model to validate real CPU ONNX outputs, or --image PATH to also
validate an existing grayscale/Analyze image. Study data is never written.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from aidas.ai.inference import (
    AIForAIDASPredictor,
    _normalize_for_model,
    _soft_argmax_y_numpy,
)


def reference_normalize(image):
    """Frozen preprocessing from v3.4.2a2, before buffer reuse."""
    arr = np.asarray(image).astype(np.float32, copy=False)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.ascontiguousarray((arr - lo) / (hi - lo + 1e-8), dtype=np.float32)


def reference_soft_argmax(logits):
    """Frozen postprocessing from v3.4.2a2, including reduction order."""
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=2, keepdims=True)
    indices = np.arange(logits.shape[2], dtype=np.float32).reshape(1, 1, logits.shape[2], 1)
    return np.sum(probabilities * indices, axis=2, dtype=np.float32)


def measure(callback, value, repeats):
    callback(value)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback(value)
        durations.append(time.perf_counter() - started)
    # NumPy reports its data-buffer allocations to tracemalloc. The input is
    # allocated before tracing, so this measures temporary/output memory.
    tracemalloc.start()
    try:
        callback(value)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return {
        "median_ms": round(statistics.median(durations) * 1000, 3),
        "peak_temporary_MiB": round(peak / 1024 ** 2, 3),
    }


def compare(reference, optimized, value, repeats):
    before, after = reference(value), optimized(value)
    if before.dtype != after.dtype or before.shape != after.shape or before.tobytes() != after.tobytes():
        raise AssertionError("AI array processing changed output bits")
    return {
        "bitwise_identical": True,
        "reference": measure(reference, value, repeats),
        "optimized": measure(optimized, value, repeats),
    }


def validate_model(model_path, image_path=None):
    predictor = AIForAIDASPredictor(
        boundary_model_path=str(model_path), provider_name="cpu", core_limit=2,
    )
    images = [np.arange(height * width, dtype=np.uint16).reshape(height, width)
              for height, width in ((65, 127), (177, 257), (257, 513))]
    if image_path:
        from aidas.utils.io_utils import read_analyze, read_tiff
        from PIL import Image

        suffix = image_path.suffix.lower()
        if suffix in {".hdr", ".img"}:
            data = read_analyze(str(image_path))
            data = data[0] if data.ndim == 3 else data
            data = np.flipud(data)  # Same stored-y orientation used by Step 2.
        elif suffix in {".tif", ".tiff"}:
            data = read_tiff(str(image_path))
            data = data[0] if data.ndim == 3 else data
        else:
            with Image.open(image_path) as image:
                data = np.asarray(image.convert("L"))
        images.append(data)
    cases = []
    for image in images:
        normalized = reference_normalize(image)
        if normalized.tobytes() != _normalize_for_model(image).tobytes():
            raise AssertionError("Model normalization changed output bits")
        started = time.perf_counter()
        logits = predictor._run_model(normalized[None, None])
        elapsed = time.perf_counter() - started
        before = np.clip(np.sort(reference_soft_argmax(logits)[0][:6], axis=0), 0, image.shape[0] - 1)
        after = np.clip(np.sort(_soft_argmax_y_numpy(logits)[0][:6], axis=0), 0, image.shape[0] - 1)
        if before.tobytes() != after.tobytes():
            raise AssertionError(f"Model boundaries changed for {image.shape}")
        cases.append({"shape": list(image.shape), "boundaries_bitwise_identical": True,
                      "model_execution_seconds": round(elapsed, 4)})
    return {"provider": predictor.execution_provider, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=461)
    parser.add_argument("--width", type=int, default=2851)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if min(args.height, args.width, args.repeats) < 1:
        parser.error("height, width and repeats must be positive")
    rng = np.random.default_rng(731)
    image = rng.integers(0, 65536, size=(args.height, args.width), dtype=np.uint16)
    logits = rng.normal(size=(1, 6, args.height, args.width)).astype(np.float32)
    result = {
        "shape": [args.height, args.width],
        "normalization": compare(reference_normalize, _normalize_for_model, image, args.repeats),
        "soft_argmax": compare(reference_soft_argmax, _soft_argmax_y_numpy, logits, args.repeats),
    }
    if args.with_model or args.image:
        model = Path(__file__).resolve().parents[1] / "OCT Segmenter/AI_ForAIDAS/model_img.onnx"
        result["real_model"] = validate_model(model, args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
