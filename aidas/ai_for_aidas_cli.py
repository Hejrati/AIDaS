"""Command-line bridge for running AI_ForAIDAS from an external Python env."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from aidas.ai_for_aidas_inference import predict_boundaries_and_fovea


def main():
    parser = argparse.ArgumentParser(description="Run AI_ForAIDAS inference on a NumPy image.")
    parser.add_argument("--image-npy", required=True, help="Input 2-D image saved with numpy.save")
    parser.add_argument("--model", required=True, help="AI_ForAIDAS boundary model .pth")
    parser.add_argument("--vline-model", default="", help="Optional fovea/vline model .pth")
    parser.add_argument("--no-vline", action="store_true", help="Skip foveal center prediction")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-npz", required=True, help="Output .npz path for predicted arrays")
    args = parser.parse_args()

    image = np.load(args.image_npy)
    vline_path = args.vline_model if args.vline_model and not args.no_vline else None
    prediction = predict_boundaries_and_fovea(
        image,
        boundary_model_path=args.model,
        vline_model_path=vline_path,
        predict_fovea=bool(vline_path),
        device_name=args.device,
    )

    fovea_x = -1 if prediction.fovea_x is None else int(prediction.fovea_x)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        boundaries=prediction.boundaries,
        fovea_x=np.array([fovea_x], dtype=np.int64),
        device=np.array([prediction.device]),
    )

    print(json.dumps({
        "boundaries_shape": list(prediction.boundaries.shape),
        "device": prediction.device,
        "fovea_x": None if prediction.fovea_x is None else int(prediction.fovea_x),
        "output_npz": os.path.abspath(args.output_npz),
    }))


if __name__ == "__main__":
    main()
