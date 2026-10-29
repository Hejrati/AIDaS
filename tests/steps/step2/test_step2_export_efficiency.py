"""Coordinate and Analyze-volume parity for Step 2 export optimizations."""

import copy
from decimal import Decimal
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from aidas.steps.step2_annotate import (
    ADDITIONAL_MARK_VALUES,
    COMMON_MARK_VALUES,
    MARKED_BACKGROUND_MAX,
    RESAMPLE_NEAREST,
    Step2Frame,
    _polyline_pixels,
    _resize_to_standard_format,
)


def _legacy_coordinates(points, scale_x, scale_y, width, height, *, coerce_float=False):
    """Frozen scalar rounding from the original export implementation."""
    result = []
    for x, y in points:
        if coerce_float:
            x, y = float(x), float(y)
        sx = int(np.round(x * scale_x))
        sy = int(np.round(y * scale_y))
        result.append((max(0, min(sx, width - 1)), max(0, min(sy, height - 1))))
    return result


def _legacy_scaled_traces(traces, source_shape, target_shape):
    source_h, source_w = map(int, source_shape)
    target_h, target_w = map(int, target_shape)
    if source_h <= 0 or source_w <= 0:
        return {}
    if (source_h, source_w) == (target_h, target_w):
        return {
            name: {
                "points": list(trace.get("points", [])),
                "pixels": list(trace.get("pixels", [])),
                "color": trace.get("color", ""),
            }
            for name, trace in traces.items()
        }
    result = {}
    for name, trace in traces.items():
        points = _legacy_coordinates(
            trace.get("points", []), target_w / source_w, target_h / source_h,
            target_w, target_h, coerce_float=True,
        )
        result[name] = {
            "points": points,
            "pixels": _polyline_pixels(points),
            "color": trace.get("color", ""),
        }
    return result


def _legacy_marked_volume(frame, *, current=False):
    target_slices, target_h, target_w, _dtype = frame._reference_marked_volume_spec()
    base = np.minimum(frame._image_uint8(frame.image_data).copy(), np.uint8(MARKED_BACKGROUND_MAX))
    source_h, source_w = base.shape
    traces = frame._copy_trace_dict(frame.boundary_traces) if current else frame.boundary_traces
    if (source_h, source_w) != (target_h, target_w):
        base = np.array(
            Image.fromarray(base).resize((target_w, target_h), RESAMPLE_NEAREST),
            dtype=np.uint8,
        )
        if current:
            traces = _legacy_scaled_traces(traces, (source_h, source_w), (target_h, target_w))
        else:
            traces = {
                name: {
                    "points": _legacy_coordinates(
                        trace.get("points", []), target_w / source_w, 1.0, target_w, target_h,
                    ),
                    "pixels": _legacy_coordinates(
                        trace.get("pixels", []), target_w / source_w, 1.0, target_w, target_h,
                    ),
                    "color": trace.get("color", ""),
                }
                for name, trace in traces.items()
            }
    marked = np.array(base, copy=True)
    for values in (COMMON_MARK_VALUES, ADDITIONAL_MARK_VALUES):
        for name, value in values.items():
            trace = traces.get(name)
            if trace:
                mask = Step2Frame._render_trace_mask(trace, target_w, target_h, boundary_name=name)
                if mask is not None:
                    marked[mask] = np.uint8(value)
    volume = np.stack([marked] * max(1, int(target_slices)), axis=0).astype(np.uint8, copy=False)
    return _resize_to_standard_format(volume)


class Step2ExportEfficiencyTests(unittest.TestCase):
    def test_coordinate_scaling_matches_scalar_rounding_and_clipping(self):
        rng = np.random.default_rng(427)
        values = rng.uniform(-3000, 4000, size=(256, 2))
        values[:8] = [(-2.5, -1.5), (-0.5, 0.5), (1.5, 2.5), (3.5, 4.5),
                      (14.5, 27.5), (0, 0), (1472, 176), (1e25, -1e25)]
        for dtype in (np.float16, np.float32, np.float64, np.int16, np.int64, np.uint16):
            # Keep the full dtype range finite, including float16.
            points = values.clip(-3000, 4000).astype(dtype)
            points.setflags(write=False)
            before = points.copy()
            for scale_x, scale_y in ((1.0, 1.0), (2133 / 1473, 1.0), (0.5, 1.5)):
                for coerce_float in (False, True):
                    with self.subTest(dtype=dtype, scale=(scale_x, scale_y), coerce=coerce_float):
                        expected = _legacy_coordinates(
                            points, scale_x, scale_y, 2133, 177, coerce_float=coerce_float,
                        )
                        actual = Step2Frame._scale_trace_coordinates(
                            points, scale_x, scale_y, 2133, 177, coerce_float=coerce_float,
                        )
                        self.assertEqual(actual, expected)
                        self.assertTrue(all(type(value) is int for point in actual for value in point))
                        np.testing.assert_array_equal(points, before)

    def test_empty_large_and_unusual_coordinates_keep_existing_behavior(self):
        cases = ([], [(1e25, -1e25)], [(10 ** 100, -10 ** 100)],
                 [(Decimal("1.5"), Decimal("2.5"))])
        for points in cases:
            for coerce_float in (False, True):
                with self.subTest(points=points, coerce=coerce_float):
                    args = (points, 1.5, 0.5, 23, 17)
                    try:
                        expected = _legacy_coordinates(*args, coerce_float=coerce_float)
                    except (TypeError, ValueError, OverflowError) as exc:
                        with self.assertRaises(type(exc)):
                            Step2Frame._scale_trace_coordinates(*args, coerce_float=coerce_float)
                    else:
                        self.assertEqual(
                            Step2Frame._scale_trace_coordinates(*args, coerce_float=coerce_float),
                            expected,
                        )

    def test_nonfinite_coordinates_raise_the_original_conversion_error(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                args = ([(value, 1)], 1.5, 1.0, 23, 17)
                try:
                    _legacy_coordinates(*args)
                except (ValueError, OverflowError) as exc:
                    with self.assertRaises(type(exc)):
                        Step2Frame._scale_trace_coordinates(*args)

    def test_trace_scaling_preserves_pixels_only_and_resampling_semantics(self):
        traces = {
            "RPE": {"points": [(0.5, -0.5), (3.5, 2.5), (9.0, 5.0)],
                    "pixels": [(0, 5), (9, 0)], "color": "#abcdef"},
            "ELM": {"pixels": [(0, 2), (9, 2)]},
            "Fovea-Center": {"points": [(4, 0), (4, 5)], "pixels": [(4, 3)]},
        }
        before = copy.deepcopy(traces)
        for source, target in (((6, 10), (6, 10)), ((6, 10), (9, 23)),
                               ((6, 10), (3, 5)), ((0, 10), (3, 5))):
            with self.subTest(source=source, target=target):
                self.assertEqual(
                    Step2Frame._scale_traces_to_shape(traces, source, target),
                    _legacy_scaled_traces(traces, source, target),
                )
                self.assertEqual(traces, before)

    def test_marked_volumes_match_original_pixels_dtype_and_slice_count(self):
        for width in (11, 2133, 3000):
            for dtype in (np.uint8, np.int16, np.uint16):
                frame = Step2Frame.__new__(Step2Frame)
                frame.image_data = (np.arange(17 * width).reshape(17, width) - 300).astype(dtype)
                frame.image_data.setflags(write=False)
                frame._source_was_8bit = False
                frame.boundary_traces = {
                    "RPE": {"points": [(0, 2.5), (width / 2, 4.5), (width - 1, 2.5)],
                            "pixels": [(0, 1), (width - 1, 1)], "color": "#ff0000"},
                    "ELM": {"points": [(0, 8), (width - 1, 8)]},
                    "ONL-OPL": {"pixels": [(0, 11), (width - 1, 11)]},
                    "Fovea-Center": {"points": [(width / 2, 0), (width / 2, 16)]},
                }
                original_traces = frame.boundary_traces
                before = copy.deepcopy(original_traces)
                for current in (False, True):
                    with self.subTest(width=width, dtype=dtype, current=current):
                        expected = _legacy_marked_volume(frame, current=current)
                        actual = (frame._build_current_marked_volume() if current
                                  else frame._build_marked_volume())
                        self.assertEqual(actual.shape, (2, 17, 2133))
                        self.assertEqual(actual.dtype, np.dtype(np.uint8))
                        np.testing.assert_array_equal(actual, expected)
                        self.assertIs(frame.boundary_traces, original_traces)
                        self.assertEqual(frame.boundary_traces, before)

    def test_failed_mask_render_does_not_replace_editor_traces(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.image_data = np.zeros((10, 20), dtype=np.uint8)
        frame.boundary_traces = {"RPE": {"points": [(0, 1), (19, 1)]}}
        traces = frame.boundary_traces
        with mock.patch.object(Step2Frame, "_render_trace_mask", side_effect=RuntimeError("render error")):
            with self.assertRaisesRegex(RuntimeError, "render error"):
                frame._build_marked_volume()
        self.assertIs(frame.boundary_traces, traces)

    def test_uint8_promotion_is_exact_for_every_value_and_strided_inputs(self):
        values = np.arange(256, dtype=np.uint8).reshape(16, 16)
        for image in (values, values.T, values[::-1, ::2]):
            image.setflags(write=False)
            before = image.copy()
            expected = np.rint(image.astype(np.float64) * (65535.0 / 255.0)).astype(np.uint16)
            actual, source_was_8bit = Step2Frame._coerce_image_for_annotation(image)
            self.assertTrue(source_was_8bit)
            self.assertEqual(actual.dtype, np.dtype(np.uint16))
            self.assertTrue(actual.flags.c_contiguous)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(image, before)


if __name__ == "__main__":
    unittest.main()
