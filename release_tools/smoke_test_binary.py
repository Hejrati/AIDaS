"""Exercise ONNX inference through the packaged AIDaS executable."""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidas.ai.client import AIWorkerClient


DEFAULT_EXECUTABLE = PROJECT_ROOT / "dist" / "AIDaS" / "AIDaS.exe"
MODEL_PATH = PROJECT_ROOT / "OCT Segmenter" / "AI_ForAIDAS" / "model_img.onnx"


def smoke_test(executable: Path) -> None:
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Packaged AIDaS executable not found: {executable}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

    image = np.arange(65 * 127, dtype=np.uint16).reshape(65, 127)
    with tempfile.TemporaryDirectory(prefix="aidas_binary_smoke_") as temp_root:
        with AIWorkerClient(
            [str(executable), "--aidas-ai-worker"],
            model_path=str(MODEL_PATH),
            provider_name="cpu",
            temp_root=temp_root,
        ) as client:
            result = client.predict(image)

    if result["execution_provider"] != "CPUExecutionProvider":
        raise RuntimeError(
            "Packaged ONNX worker selected an unexpected provider: "
            f"{result['execution_provider']}"
        )
    if result["boundaries"].shape != (6, image.shape[1]):
        raise RuntimeError(
            "Packaged ONNX worker returned an unexpected boundary shape: "
            f"{result['boundaries'].shape}"
        )

    print(
        "Packaged AIDaS ONNX worker smoke test passed: "
        f"provider={result['execution_provider']}, "
        f"boundaries={result['boundaries'].shape}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "executable",
        nargs="?",
        type=Path,
        default=DEFAULT_EXECUTABLE,
        help="Path to the packaged AIDaS executable",
    )
    args = parser.parse_args()
    smoke_test(args.executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
