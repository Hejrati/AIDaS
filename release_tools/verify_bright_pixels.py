"""Verify bright-pixel correction on SDB crops without changing source files.

Example:
    python release_tools/verify_bright_pixels.py SAMPLE_DIRECTORY \
        --roi light:45:548:721:161:2 --roi larger:64:516:700:223:4 \
        --roi default:0:585:768:128:0

Each ROI is NAME:X:Y:WIDTH:HEIGHT, optionally followed by an expected count.
Artifacts, including Analyze roundtrips and histogram previews, go to
debug_outputs/bright_pixels by default. The real Step 1 load/crop/save methods
run with UI controls stubbed and the canvas notice answered Smooth on a workspace copy.
"""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import sys
from types import MethodType, SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.steps.step2_annotate import Step2Frame
from aidas.ai.inference import _normalize_for_model
from aidas.utils.bright_pixels import detect_bright_pixels_in_bands, smooth_bright_pixels
from aidas.utils.io_utils import read_analyze, read_raw_oct, scale_image


def _roi(value):
    fields = value.split(":")
    if len(fields) not in (5, 6):
        raise argparse.ArgumentTypeError("Use NAME:X:Y:WIDTH:HEIGHT[:EXPECTED_COUNT]")
    name = fields[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise argparse.ArgumentTypeError("ROI names may contain letters, digits, _ and -")
    try:
        numbers = tuple(int(item) for item in fields[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI coordinates and counts must be integers") from exc
    x, y, width, height = numbers[:4]
    if min(x, y) < 0 or min(width, height) <= 0 or (len(numbers) == 5 and numbers[4] < 0):
        raise argparse.ArgumentTypeError("Coordinates/counts must be nonnegative, dimensions positive")
    return name, (x, y, width, height), numbers[4] if len(numbers) == 5 else None


def _hashes(folder):
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in {".sdb", ".hdr", ".img"}
    }


def _app_crop(raw, roi, source_copy, output_folder):
    """Execute production SDB load, canvas notice, crop, scaling and Analyze save."""
    noop = lambda *args, **kwargs: None
    state = SimpleNamespace(
        raw_image=raw,
        _source_roi=roi,
        current_file=str(source_copy),
        processed_image=None,
        view_mode_var=SimpleNamespace(set=noop),
        image_canvas=SimpleNamespace(enable_roi=noop, set_image=noop),
        _on_processed_image=None,
        _cropped_sdb_files=set(),
        _saved_output_directories=set(),
        _source_output_directory=lambda: str(output_folder),
        _build_output_name=lambda suffix: "crop",
        _path_key=lambda path: str(Path(path).resolve()),
        _directory_is_complete=lambda path: False,
    )
    for name in (
        "_update_zoom_label", "_set_sdb_parameters_enabled", "_update_image_status",
        "_update_processing_controls", "_set_roi_controls_enabled", "_show_crop_notification",
        "_refresh_sdb_progress_colors", "_update_batch_handoff_button_state", "_update_queue_controls",
        "_set_default_roi",
    ):
        setattr(state, name, noop)
    with mock.patch.object(Step1Frame, "_ask_bright_pixel_action", return_value=True) as prompt:
        for name in (
            "_set_bright_pixel_status", "_set_source_bright_pixel_status", "_offer_bright_pixel_removal",
            "_ask_bright_pixel_action", "_build_processed_target_image", "_save_processed_image",
        ):
            setattr(state, name, MethodType(getattr(Step1Frame, name), state))
        assert Step1Frame._load_image(state, raw, str(source_copy)), "Production SDB load failed"
        source_prompt_count = prompt.call_count
        state._source_roi = roi
        assert Step1Frame._crop_and_scale(state), "Production crop/save failed"
    return state, source_prompt_count, prompt.call_count


def _preview(before, after, threshold, count, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 6.5), gridspec_kw={"height_ratios": [1, 1, 2]})
    for axis, data, title in zip(axes[:2], (before, after), ("Original crop", "Only detected pixels smoothed")):
        axis.imshow(scale_image(data, sx=3), cmap="gray", vmin=float(data.min()),
                    vmax=float(data.max()), aspect="auto", interpolation="nearest")
        axis.set_title(f"{title} | display range {data.min()} to {data.max()}", loc="left", fontsize=10)
        axis.axis("off")
    lo = min(float(before.min()), float(after.min()))
    hi = max(float(before.max()), float(after.max()))
    bins = np.linspace(lo, hi + (hi == lo), 257)
    axes[2].hist(before.ravel(), bins=bins, histtype="step", label="Original", linewidth=1.5)
    axes[2].hist(after.ravel(), bins=bins, histtype="step", label="Corrected", linewidth=1.0)
    if threshold is not None:
        axes[2].axvline(threshold, color="tab:red", linestyle="--", label=f"Threshold {threshold:g}")
    axes[2].set_yscale("symlog", linthresh=1)
    axes[2].set_xlabel("Signed 16-bit pixel value")
    axes[2].set_ylabel("Pixel count")
    axes[2].legend()
    fig.suptitle(f"{count} source pixels changed; every unmasked pixel preserved exactly", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample_directory", type=Path)
    parser.add_argument("--roi", type=_roi, action="append", required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "debug_outputs" / "bright_pixels")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--offset", type=int, default=1050)
    parser.add_argument("--expected-source-count", type=int)
    args = parser.parse_args()
    folder = args.sample_directory.resolve()
    output = args.output.resolve()
    if output == folder or folder in output.parents:
        parser.error("Output must be outside the sample directory")
    sources = sorted(path for path in folder.iterdir() if path.suffix.lower() == ".sdb")
    if not sources:
        parser.error("No SDB files found")
    original_hashes = _hashes(folder)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for source in sources:
        source_folder = output / source.stem
        source_folder.mkdir(exist_ok=True)
        source_copy = source_folder / source.name
        shutil.copy2(source, source_copy)
        raw = read_raw_oct(str(source_copy), width=args.width, height=args.height, offset=args.offset)
        raw_before = raw.copy()
        signed_source = raw.astype(np.int16, copy=True)
        source_detection = detect_bright_pixels_in_bands(signed_source)
        if args.expected_source_count is not None:
            assert source_detection.count == args.expected_source_count, (
                f"Expected {args.expected_source_count} source detections, found {source_detection.count}"
            )
        working_source = smooth_bright_pixels(signed_source, source_detection.mask)
        assert np.array_equal(working_source[~source_detection.mask], signed_source[~source_detection.mask]), (
            "Source correction changed unmasked pixels"
        )
        assert np.count_nonzero(working_source != signed_source) == source_detection.count, (
            "Source correction changed pixels outside the detected mask"
        )
        for name, roi, expected_count in args.roi:
            x, y, width, height = roi
            assert x + width <= raw.shape[1] and y + height <= raw.shape[0], f"ROI {name} exceeds image bounds"
            crop = raw[y:y + height, x:x + width].astype(np.int16, copy=True)
            working_crop = working_source[y:y + height, x:x + width].copy()
            source_roi_mask = source_detection.mask[y:y + height, x:x + width]
            crop_detection = detect_bright_pixels_in_bands(working_crop)
            additional_mask = crop_detection.mask & ~source_roi_mask
            detected_mask = source_roi_mask | additional_mask
            detected_count = int(np.count_nonzero(detected_mask))
            additional_count = int(np.count_nonzero(additional_mask))
            threshold = crop_detection.threshold if additional_count else source_detection.threshold
            if expected_count is not None:
                assert detected_count == expected_count, f"{name}: expected {expected_count}, found {detected_count}"
            corrected = smooth_bright_pixels(working_crop, additional_mask)
            assert corrected.dtype == crop.dtype, "Pixel dtype changed"
            assert np.array_equal(corrected[~detected_mask], crop[~detected_mask]), "Unmasked pixels changed"
            assert np.count_nonzero(corrected != crop) == detected_count, "Changed pixels differ from detected mask"
            roi_folder = source_folder / name
            roi_folder.mkdir(exist_ok=True)
            state, source_prompt_count, prompt_count = _app_crop(raw, roi, source_copy, roi_folder)
            processed = state.processed_image
            expected_scaled = scale_image(corrected, sx=3, sy=1)
            scaled_mask = scale_image(detected_mask, sx=3, sy=1)
            assert np.array_equal(processed, expected_scaled), "App crop differs from utility result"
            assert source_prompt_count == bool(source_detection.count), "App SDB-open prompt differs from detection"
            assert prompt_count == bool(source_detection.count) + bool(additional_count), (
                "App total prompt count differs from source and additional crop detections"
            )
            assert np.array_equal(state._source_raw_image, raw_before), "App changed its original source buffer"
            assert np.array_equal(state.raw_image.astype(np.int16), working_source), "App working source differs from correction"
            assert np.array_equal(state._source_bright_pixel_mask, source_detection.mask), "App source mask differs"
            reread = read_analyze(str(roi_folder / "crop.hdr"))
            assert reread.shape == (2, *expected_scaled.shape), "Analyze shape changed"
            assert np.array_equal(reread, np.stack([expected_scaled, expected_scaled])), "Analyze roundtrip changed values"
            header = (roi_folder / "crop.hdr").read_bytes()
            datatype, bit_depth = struct.unpack_from(">hh", header, 70)
            assert (datatype, bit_depth) == (4, 16), "Crop must use the default signed 16-bit Analyze format"
            assert reread.dtype == np.dtype("int16"), "Saved crop lost its 16-bit pixel values"
            annotation_image, template, source_was_8bit = Step2Frame._read_image_for_annotation(
                str(roi_folder / "crop.img")
            )
            assert not source_was_8bit, "Step 2 treated the crop as an 8-bit image"
            assert template["dtype"] == np.dtype("int16"), "Step 2 lost the Analyze dtype"
            assert annotation_image.dtype == np.dtype("int16"), "Step 2 changed the crop bit depth"
            assert np.array_equal(annotation_image, processed), "Step 2 changed saved pixel values"
            model_input = _normalize_for_model(annotation_image)
            assert model_input.dtype == np.dtype("float32") and model_input.flags.c_contiguous
            assert np.isfinite(model_input).all() and 0 <= model_input.min() <= model_input.max() <= 1
            assert np.unique(model_input).size == np.unique(annotation_image).size, (
                "Model preparation lost distinct 16-bit intensity levels"
            )
            assert np.array_equal(raw, raw_before), "App changed the source image"
            assert hashlib.sha256(source_copy.read_bytes()).hexdigest() == original_hashes[str(source)], "App changed the SDB copy"
            coordinates = np.argwhere(detected_mask)
            result = {
                "source": source.name, "roi": name, "coordinates_xywh": roi,
                "source_detected_pixels": source_detection.count,
                "source_threshold": source_detection.threshold,
                "source_unmasked_pixels_preserved": int(np.count_nonzero(~source_detection.mask)),
                "detected_pixels": detected_count, "threshold": threshold,
                "additional_crop_detected_pixels": additional_count,
                "changed_source_pixels": int(np.count_nonzero(corrected != crop)),
                "changed_scaled_pixels_per_slice": int(np.count_nonzero(scaled_mask)),
                "unmasked_source_pixels_preserved": int(np.count_nonzero(~detected_mask)),
                "source_prompts": source_prompt_count, "prompts": prompt_count,
                "Analyze_shape": list(reread.shape),
                "Analyze_dtype": str(reread.dtype),
                "Analyze_datatype": datatype,
                "Analyze_bit_depth": bit_depth,
                "segmentation_image_dtype": str(annotation_image.dtype),
                "segmentation_model_dtype": str(model_input.dtype),
                "segmentation_intensity_levels": int(np.unique(model_input).size),
                "pixel_changes_y_x_before_after": [
                    [int(yy), int(xx), int(crop[yy, xx]), int(corrected[yy, xx])]
                    for yy, xx in coordinates
                ],
            }
            results.append(result)
            _preview(crop, corrected, threshold, detected_count, roi_folder / "preview.png")
            print(
                f"{source.name} {name}: {source_detection.count} source detections, "
                f"{detected_count} in ROI, {additional_count} additional; "
                f"{result['unmasked_source_pixels_preserved']} unchanged, {prompt_count} prompt(s); "
                f"16-bit crop and Step 2 float32 input passed ({np.unique(model_input).size} intensity levels)"
            )
    final_hashes = _hashes(folder)
    assert final_hashes == original_hashes, "Sample input files changed"
    report = {"original_sample_files_unchanged": True, "sample_sha256": original_hashes, "results": results}
    (output / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Original sample hashes unchanged. Report: {output / 'verification_report.json'}")


if __name__ == "__main__":
    main()
