from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas
from aidas.steps.step1_resize_raw import Step1Frame
from aidas.utils.io_utils import read_analyze, read_raw_oct, write_analyze
from aidas.utils.sdb_reader import detect_sdb_image

from tests.support.sdb_fixture import build_cmdb, ir_chunk, metadata_chunk, oct_chunk


STEP1 = "aidas.steps.step1_resize_raw"


class _Variable:
    """Run write callbacks even for identical values, as a Tk variable does."""

    def __init__(self, value, callback=None):
        self.value = value
        self.callback = callback

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        if self.callback is not None:
            self.callback()


class _Canvas:
    def __init__(self):
        self.image = None
        self.roi_enabled = False

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    def enable_roi(self, enabled):
        self.roi_enabled = enabled


class Step1SdbAutoImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.errors = self.enterContext(patch(f"{STEP1}.messagebox.showerror"))
        self.enterContext(patch(f"{STEP1}.messagebox.showwarning"))
        self.enterContext(patch.object(Step1Frame, "_ask_bright_pixel_action", return_value=False))

    def _frame(self, *, width=8, height=6, offset=0):
        frame = Step1Frame.__new__(Step1Frame)
        frame.default_raw_width = width
        frame.default_raw_height = height
        frame.default_raw_offset = offset
        frame.default_little_endian = True
        frame.raw_import_params = None
        frame.raw_image = None
        frame._source_raw_image = None
        frame._working_source_raw_image = None
        frame._source_bright_pixel_mask = None
        frame._raw_bright_pixel_mask = None
        frame._source_bright_pixel_threshold = None
        frame._source_bright_pixels_smoothed = False
        frame._bright_pixel_source_checked = False
        frame._bright_pixel_notice_pending = False
        frame.processed_image = None
        frame._processed_image_saved = False
        frame.current_file = None
        frame._source_roi = None
        frame._sdb_image_info = None
        frame._sdb_manual_import = False
        frame._sdb_showing_saved_crop = False
        frame._setting_detected_import_params = False
        frame._crop_contours = None
        frame._crop_oct_layout = None
        frame._source_crop_contours = None
        frame._source_crop_oct_layout = None
        frame.image_canvas = _Canvas()
        frame.view_mode_var = _Variable("source")
        frame.sdb_params_summary_var = _Variable("")
        frame.bright_pixel_status_var = _Variable("")
        frame.width_var = _Variable(str(width), frame._on_width_changed)
        frame.height_var = _Variable(str(height), frame._on_import_param_changed)
        frame.offset_var = _Variable(str(offset), frame._on_import_param_changed)
        frame.endian_var = _Variable(True, frame._on_import_param_changed)
        frame._sdb_directory_files = {}
        frame._sdb_directories = []
        frame._cropped_sdb_files = set()
        frame._saved_crop_previews = {}
        frame._saved_output_directories = set()
        frame._on_processed_image = None
        frame._on_batch_segment_folders = None
        frame._update_zoom_label = lambda: None
        frame._update_image_status = lambda: None
        frame._update_processing_controls = lambda: None
        frame._update_batch_handoff_button_state = lambda: None
        frame._update_queue_controls = lambda: None
        frame._set_sdb_parameters_enabled = lambda _enabled: None
        frame._set_roi_controls_enabled = lambda _enabled: None
        frame._render_sdb_directories = lambda: None
        frame._refresh_sdb_progress_colors = lambda: None
        frame._show_crop_notification = lambda *_args, **_kwargs: None

        def default_roi():
            height, width = frame.raw_image.shape
            frame._source_roi = (0, 0, width, height)

        frame._set_default_roi = default_roi
        frame._set_initial_roi = default_roi
        return frame

    def _form_values(self, frame):
        return tuple(
            variable.get()
            for variable in (
                frame.width_var, frame.height_var, frame.offset_var, frame.endian_var
            )
        )

    def _full_pixels(self, frame, source):
        return read_raw_oct(source, **frame.raw_import_params)

    def _assert_native_words_in_full_view(self, frame, words):
        info = frame._sdb_image_info
        start = (info.offset - frame.raw_import_params["offset"]) // 2
        self.assertEqual(frame._crop_oct_layout, (start, info.width, info.height))
        np.testing.assert_array_equal(
            frame.raw_image.ravel()[start:start + words.size].reshape(words.shape),
            words,
        )

    def _native_source(
        self, name="native", *, width=8, height=6, frames=1, edb="match", words=None
    ):
        # Preserve every stored word, including the full unsigned range. Vary
        # rows so an incorrect offset cannot accidentally match the source.
        samples = np.array(
            [0, 1, 0x1234, 0x7FFF, 0x8000, 0xBEEF, 0xFFFE, 0xFFFF], dtype=np.uint16
        )
        if words is None:
            words = np.resize(samples, (height, width))
            words ^= np.arange(height, dtype=np.uint16)[:, None]
        else:
            words = np.asarray(words, dtype=np.uint16)
        chunks = [ir_chunk(3, 4)] + [
            oct_chunk(width, height, words, slice_id=index)
            for index in range(frames)
        ]
        source = self.folder / f"{name}.sdb"
        source.write_bytes(build_cmdb(chunks))
        if edb is not None:
            metadata_width = width if edb == "match" else width + 1
            source.with_suffix(".edb").write_bytes(build_cmdb([
                metadata_chunk(metadata_width, height),
                {"kind": 9, "payload": b"private fixture patient"},
            ]))
        return source, words

    def test_native_file_uses_image_header_and_unchanged_raw_pixels(self):
        source, words = self._native_source()
        frame = self._frame(width=768, height=1200, offset=1050)

        self.assertTrue(frame._open_raw(str(source)))

        self.errors.assert_not_called()
        info = frame._sdb_image_info
        self.assertEqual((info.width, info.height), (8, 6))
        self.assertGreater(info.offset, 140)
        self.assertEqual(frame.raw_import_params, {
            "width": 8, "height": 1200, "offset": 0,
            "bit_depth": 16, "little_endian": True,
        })
        self.assertEqual(self._form_values(frame), ("8", "1200", "0", True))
        expected = self._full_pixels(frame, source)
        np.testing.assert_array_equal(frame.raw_image, expected)
        self._assert_native_words_in_full_view(frame, words)
        self.assertEqual(frame.raw_image.dtype, np.dtype("uint16"))
        np.testing.assert_array_equal(frame.image_canvas.image, expected)
        summary = frame.sdb_params_summary_var.get()
        self.assertIn("Auto-detected OCT: 8 × 6", summary)
        self.assertIn("Full SDB view", summary)
        self.assertIn("1 scan", summary)
        self.assertIn("SDB/EDB confirmed", summary)
        self.assertFalse(frame._sdb_manual_import)

    def test_full_view_keeps_configured_leading_offset_and_covers_entire_file(self):
        source, words = self._native_source()
        for configured_height in (6, 80):
            with self.subTest(configured_height=configured_height):
                frame = self._frame(width=768, height=configured_height, offset=2)
                self.assertTrue(frame._open_raw(str(source)))
                rows_to_end = (source.stat().st_size - 2 + 15) // 16
                self.assertEqual(frame.raw_import_params, {
                    "width": 8, "height": max(configured_height, rows_to_end),
                    "offset": 2, "bit_depth": 16, "little_endian": True,
                })
                np.testing.assert_array_equal(frame.raw_image, self._full_pixels(frame, source))
                self._assert_native_words_in_full_view(frame, words)
                self.assertGreaterEqual(frame.raw_image.size * 2, source.stat().st_size - 2)
                self.assertEqual(frame.offset_var.get(), "2")
        self.errors.assert_not_called()

    def test_auto_import_matches_manual_preview_normalization_and_saved_pixels(self):
        source, words = self._native_source()
        original_bytes = source.read_bytes()
        auto = self._frame(width=768, height=1200, offset=1050)
        self.assertTrue(auto._open_raw(str(source)))
        info = auto._sdb_image_info
        expected = self._full_pixels(auto, source)

        manual_folder = self.folder / "manual"
        manual_folder.mkdir()
        manual_source = manual_folder / "native.sdb"
        manual_source.write_bytes(original_bytes)
        manual_pixels = read_raw_oct(manual_source, **auto.raw_import_params)
        manual = self._frame()
        self.assertTrue(manual._load_image(manual_pixels, str(manual_source)))

        for frame in (auto, manual):
            self.assertEqual(frame.raw_image.dtype, np.dtype("uint16"))
            np.testing.assert_array_equal(frame.raw_image, expected)
            self.assertEqual(frame.raw_image.tobytes(), expected.tobytes())
            np.testing.assert_array_equal(frame._source_raw_image, expected)
            self.assertFalse(frame._source_bright_pixels_smoothed)

        # Both imports retain the canvas's existing unsigned min/max display
        # scaling. Its 8-bit preview must not replace the stored 16-bit pixels.
        display = ImageCanvas._to_display(None, manual_pixels)
        np.testing.assert_array_equal(
            ImageCanvas._to_display(None, auto.image_canvas.image), display
        )
        np.testing.assert_array_equal(
            display,
            (expected.astype(np.float64) * (255.0 / 65535)).astype(np.uint8),
        )
        np.testing.assert_array_equal(auto.raw_image, expected)

        self._assert_native_words_in_full_view(auto, words)
        start = auto._crop_oct_layout[0]
        self.assertEqual(start % info.width, 0, "This fixture's OCT block is row aligned.")
        for frame in (auto, manual):
            frame._source_roi = (0, start // info.width, info.width, info.height)
            self.assertTrue(frame._crop_and_scale())
        signed = np.repeat(words.astype(np.int16), 3, axis=1)
        stack = np.stack([signed, signed])
        saved_bytes = np.stack([np.flipud(signed)] * 2).astype(">i2").tobytes()
        for output in (self.folder, manual_folder):
            np.testing.assert_array_equal(read_analyze(output / "light"), stack)
            self.assertEqual((output / "light.img").read_bytes(), saved_bytes)
        np.testing.assert_array_equal(auto.processed_image, manual.processed_image)
        np.testing.assert_array_equal(
            ImageCanvas._to_display(None, auto.image_canvas.image),
            ImageCanvas._to_display(None, manual.image_canvas.image),
        )
        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertEqual(manual_source.read_bytes(), original_bytes)
        self.errors.assert_not_called()

    def test_auto_open_keeps_manual_byte_order_and_reset_uses_configured_default(self):
        source, words = self._native_source()
        frame = self._frame(width=768, height=1200, offset=1050)
        frame.endian_var.set(False)

        self.assertTrue(frame._open_raw(str(source)))

        self.assertEqual(self._form_values(frame), ("8", "1200", "0", False))
        self.assertFalse(frame.raw_import_params["little_endian"])
        manual = self._full_pixels(frame, source)
        np.testing.assert_array_equal(frame.raw_image, manual)
        self._assert_native_words_in_full_view(frame, words.byteswap())
        self.assertTrue(frame._set_default_import_params())
        self.assertTrue(frame.endian_var.get())
        np.testing.assert_array_equal(
            frame.raw_image,
            self._full_pixels(frame, source),
        )
        self._assert_native_words_in_full_view(frame, words)
        self.errors.assert_not_called()

    def test_auto_import_keeps_manual_bright_pixel_decision_and_source_bits(self):
        signed = np.linspace(-12500, -8000, 64 * 64).reshape(64, 64).astype(np.int16)
        signed[-2:] = 0
        signed[20, 31] = 32005
        signed[40, 17] = 32555
        source, words = self._native_source(width=64, height=64, words=signed)
        auto = self._frame()
        manual = self._frame()

        with patch.object(auto, "_ask_bright_pixel_action", return_value=False) as auto_prompt, patch(
            f"{STEP1}.smooth_bright_pixels"
        ) as smooth:
            self.assertTrue(auto._open_raw(str(source)))
            pixels = self._full_pixels(auto, source)
            with patch.object(manual, "_ask_bright_pixel_action", return_value=False) as manual_prompt:
                self.assertTrue(manual._load_image(pixels, str(source)))

        smooth.assert_not_called()
        auto_prompt.assert_called_once()
        manual_prompt.assert_called_once()
        auto_detection, auto_preview = auto_prompt.call_args.args
        manual_detection, manual_preview = manual_prompt.call_args.args
        self.assertGreaterEqual(auto_detection.count, 2)
        start = auto._crop_oct_layout[0]
        native_mask = auto_detection.mask.ravel()[start:start + words.size].reshape(words.shape)
        self.assertTrue(native_mask[20, 31])
        self.assertTrue(native_mask[40, 17])
        self.assertEqual(np.count_nonzero(native_mask), 2)
        self.assertEqual(auto_detection.threshold, manual_detection.threshold)
        np.testing.assert_array_equal(auto_detection.mask, manual_detection.mask)
        self.assertEqual(auto_prompt.call_args.kwargs, manual_prompt.call_args.kwargs)
        np.testing.assert_array_equal(auto_preview, manual_preview)
        for frame in (auto, manual):
            self.assertEqual(frame.raw_image.dtype, np.dtype("uint16"))
            np.testing.assert_array_equal(frame.raw_image, pixels)
            np.testing.assert_array_equal(
                frame.raw_image.ravel()[start:start + words.size].reshape(words.shape).astype(np.int16),
                signed,
            )
            self.assertFalse(frame._source_bright_pixels_smoothed)
            self.assertIn(f"kept {auto_detection.count}", frame.bright_pixel_status_var.get().lower())
        self.errors.assert_not_called()

    def test_short_scan_keeps_legacy_default_roi_as_an_explicit_preset(self):
        frame = self._frame()
        frame.raw_image = np.zeros((496, 8), dtype=np.uint16)
        frame._set_default_roi = Step1Frame._set_default_roi.__get__(frame, Step1Frame)
        selected = []
        frame._set_roi_and_entries = lambda *roi: selected.append(roi)

        frame._set_default_roi()

        self.errors.assert_not_called()
        self.assertEqual(selected[-1], (0, 495, 8, 1))

    def test_navigation_between_dimensions_does_not_reload_previous_source(self):
        first, _expected_first = self._native_source("first")
        second, words_second = self._native_source("second", width=5, height=4, frames=2)
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(first)))

        with patch(f"{STEP1}.detect_sdb_image", wraps=detect_sdb_image) as detect, patch(
            f"{STEP1}.read_raw_oct", wraps=read_raw_oct
        ) as read_raw:
            self.assertTrue(frame._open_raw(str(second)))

        self.errors.assert_not_called()
        detect.assert_called_once_with(str(second))
        read_raw.assert_called_once_with(str(second), **frame.raw_import_params)
        self.assertEqual(frame.current_file, str(second))
        self.assertEqual(self._form_values(frame)[0], "5")
        expected_height = (second.stat().st_size + 9) // 10
        self.assertEqual(self._form_values(frame)[1], str(expected_height))
        np.testing.assert_array_equal(frame.raw_image, self._full_pixels(frame, second))
        self._assert_native_words_in_full_view(frame, words_second)
        self.assertIn("2 scans", frame.sdb_params_summary_var.get())

    def test_native_to_legacy_navigation_uses_configured_raw_defaults(self):
        native, _expected_native = self._native_source()
        legacy = self.folder / "legacy.sdb"
        expected = np.arange(48, dtype=np.uint16).reshape(6, 8)
        legacy.write_bytes(expected.astype("<u2").tobytes())
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(native)))
        self.assertGreater(frame.raw_import_params["height"], 6)

        self.assertTrue(frame._open_raw(str(legacy)))

        self.errors.assert_not_called()
        self.assertIsNone(frame._sdb_image_info)
        self.assertEqual(self._form_values(frame), ("8", "6", "0", True))
        np.testing.assert_array_equal(frame.raw_image, expected)
        self.assertNotIn("Auto-detected", frame.sdb_params_summary_var.get())

    def test_cancelled_native_open_preserves_previous_parameters_and_image(self):
        first, _expected_first = self._native_source("first")
        second, _expected_second = self._native_source("second", width=5, height=4)
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(first)))
        previous = (
            frame.raw_image, frame._source_raw_image, frame._working_source_raw_image,
            frame._sdb_image_info, frame.image_canvas.image,
        )
        previous_params = dict(frame.raw_import_params)
        previous_form = self._form_values(frame)
        previous_summary = frame.sdb_params_summary_var.get()
        previous_roi = frame._source_roi
        previous_layout = frame._crop_oct_layout
        previous_contours = frame._crop_contours

        with patch.object(frame, "_offer_bright_pixel_removal", return_value=None):
            self.assertFalse(frame._open_raw(str(second)))

        self.errors.assert_not_called()
        actual = (
            frame.raw_image, frame._source_raw_image, frame._working_source_raw_image,
            frame._sdb_image_info, frame.image_canvas.image,
        )
        for before, after in zip(previous, actual):
            self.assertIs(before, after)
        self.assertEqual(frame.current_file, str(first))
        self.assertEqual(frame.raw_import_params, previous_params)
        self.assertEqual(self._form_values(frame), previous_form)
        self.assertEqual(frame.sdb_params_summary_var.get(), previous_summary)
        self.assertEqual(frame._source_roi, previous_roi)
        self.assertEqual(frame._crop_oct_layout, previous_layout)
        self.assertIs(frame._crop_contours, previous_contours)

    def test_width_controls_crop_and_pad_raw_image_without_rereading(self):
        source, _words = self._native_source()
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(source)))
        expected = self._full_pixels(frame, source)
        original_layout = frame._crop_oct_layout

        with patch(f"{STEP1}.read_raw_oct") as raw, patch(f"{STEP1}.detect_sdb_image") as detect:
            frame.width_var.set("4")
            np.testing.assert_array_equal(frame.raw_image, expected[:, 2:6])
            self.assertEqual(frame._crop_oct_layout, ())
            frame.width_var.set("12")
            np.testing.assert_array_equal(frame.raw_image, np.pad(expected, ((0, 0), (2, 2))))
            frame.width_var.set("8")
            np.testing.assert_array_equal(frame.raw_image, expected)
            self.assertEqual(frame._crop_oct_layout, original_layout)

        self.errors.assert_not_called()
        raw.assert_not_called()
        detect.assert_not_called()
        np.testing.assert_array_equal(frame._source_raw_image, expected)

    def test_height_offset_and_endian_edits_preserve_manual_pixel_reading(self):
        source, words = self._native_source()
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(source)))
        info = frame._sdb_image_info
        original = self._full_pixels(frame, source)

        with patch(f"{STEP1}.detect_sdb_image", wraps=detect_sdb_image) as detect, patch(
            f"{STEP1}.read_raw_oct", wraps=read_raw_oct
        ) as raw:
            frame.height_var.set("4")
            np.testing.assert_array_equal(frame.raw_image, original[:4])
            frame.offset_var.set(str(info.offset + 8 * 2))
            np.testing.assert_array_equal(frame.raw_image, words[1:5])
            self.assertEqual(frame._crop_oct_layout, ())
            frame.endian_var.set(False)
            self.assertEqual(frame.raw_image.shape, (4, 8))
            self.assertEqual(frame.raw_image.dtype, np.dtype("uint16"))
            np.testing.assert_array_equal(frame.raw_image, words[1:5].byteswap())

        self.errors.assert_not_called()
        self.assertEqual(detect.call_count, 3)
        self.assertEqual(raw.call_count, 3)
        self.assertEqual(frame._sdb_image_info, info)
        self.assertTrue(frame._sdb_manual_import)

    def test_height_reload_after_width_preview_matches_manual_reshape(self):
        source, _words = self._native_source()
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(source)))
        offset = frame.raw_import_params["offset"]
        expected = self._full_pixels(frame, source)
        frame.width_var.set("4")
        np.testing.assert_array_equal(frame.raw_image, expected[:, 2:6])

        frame.height_var.set("4")

        manual = read_raw_oct(source, width=4, height=4, offset=offset)
        np.testing.assert_array_equal(frame.raw_image, manual)
        np.testing.assert_array_equal(frame._source_raw_image, manual)
        self.assertEqual(frame.raw_import_params["width"], 4)
        self.assertEqual(frame.raw_import_params["height"], 4)
        self.assertEqual(frame.raw_image.dtype, np.dtype("uint16"))
        self.errors.assert_not_called()

    def test_cancelled_native_parameter_reload_restores_controls_and_source_geometry(self):
        source, _words = self._native_source()
        for name in ("height", "offset", "endian"):
            with self.subTest(parameter=name):
                frame = self._frame()
                self.assertTrue(frame._open_raw(str(source)))
                if name == "offset":
                    frame.height_var.set("4")
                info = frame._sdb_image_info
                previous_image = frame.raw_image
                previous_source = frame._source_raw_image
                previous_working = frame._working_source_raw_image
                previous_params = dict(frame.raw_import_params)
                expected = self._full_pixels(frame, source)
                previous_form = self._form_values(frame)
                previous_summary = frame.sdb_params_summary_var.get()
                previous_layout = frame._crop_oct_layout
                previous_contours = frame._crop_contours
                original_apply = frame._apply_import_params
                outcomes = []

                def record_reload(*args, **kwargs):
                    result = original_apply(*args, **kwargs)
                    outcomes.append(result)
                    return result

                with patch.object(frame, "_offer_bright_pixel_removal", return_value=None), patch.object(
                    frame, "_apply_import_params", side_effect=record_reload
                ):
                    if name == "height":
                        frame.height_var.set("4")
                    elif name == "offset":
                        frame.offset_var.set(str(info.offset + 8 * 2))
                    else:
                        frame.endian_var.set(False)

                self.assertEqual(outcomes, [False], "Restoring the controls must suppress their traces.")
                self.assertEqual(self._form_values(frame), previous_form)
                self.assertEqual(frame.raw_import_params, previous_params)
                self.assertEqual(frame.sdb_params_summary_var.get(), previous_summary)
                self.assertIs(frame.raw_image, previous_image)
                self.assertIs(frame._source_raw_image, previous_source)
                self.assertIs(frame._working_source_raw_image, previous_working)
                self.assertIs(frame._sdb_image_info, info)
                self.assertEqual(frame._crop_oct_layout, previous_layout)
                self.assertIs(frame._crop_contours, previous_contours)
                self.assertFalse(frame._setting_detected_import_params)
                frame.width_var.set("4")
                height = previous_params["height"]
                np.testing.assert_array_equal(frame.raw_image, expected[:height, 2:6])
                self.assertEqual(frame.raw_import_params["height"], height)
                self.assertEqual(frame.raw_import_params["offset"], previous_params["offset"])
                self.assertTrue(frame.raw_import_params["little_endian"])
        self.errors.assert_not_called()

    def test_manual_offset_edits_can_include_metadata_like_original_importer(self):
        source, _words = self._native_source()
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(source)))
        frame.offset_var.set(str(frame._sdb_image_info.offset - 2))

        self.errors.assert_not_called()
        self.assertEqual(frame.raw_import_params["offset"], frame._sdb_image_info.offset - 2)
        expected = self._full_pixels(frame, source)
        np.testing.assert_array_equal(frame.raw_image, expected)
        self.assertEqual(frame._crop_oct_layout[0], 1)
        frame.width_var.set("4")
        np.testing.assert_array_equal(frame.raw_image, expected[:, 2:6])
        self.assertEqual(frame._crop_oct_layout, ())
        self.assertTrue(frame._sdb_manual_import)

    def test_reset_defaults_restores_full_manual_view_and_detected_width(self):
        source, words = self._native_source()
        frame = self._frame(width=768, height=1200, offset=1050)
        self.assertTrue(frame._open_raw(str(source)))
        previous_params = dict(frame.raw_import_params)
        expected = self._full_pixels(frame, source)
        frame.width_var.set("4")
        frame.height_var.set("4")
        self.assertTrue(frame._sdb_manual_import)

        self.assertTrue(frame._set_default_import_params())

        self.errors.assert_not_called()
        self.assertEqual(self._form_values(frame), ("8", "1200", "0", True))
        self.assertEqual(frame.raw_import_params, previous_params)
        np.testing.assert_array_equal(frame.raw_image, expected)
        self._assert_native_words_in_full_view(frame, words)
        self.assertFalse(frame._sdb_manual_import)

    def test_cancelled_native_reset_retains_manual_controls_and_accepted_source(self):
        source, _words = self._native_source()
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(source)))
        info = frame._sdb_image_info
        frame.height_var.set("4")
        frame.offset_var.set(str(info.offset + 8 * 2))
        expected = self._full_pixels(frame, source)
        frame.width_var.set("4")
        previous_image = frame.raw_image
        previous_source = frame._source_raw_image
        previous_working = frame._working_source_raw_image
        previous_params = dict(frame.raw_import_params)
        previous_form = self._form_values(frame)
        previous_summary = frame.sdb_params_summary_var.get()

        with patch.object(frame, "_offer_bright_pixel_removal", return_value=None):
            self.assertFalse(frame._set_default_import_params())

        self.errors.assert_not_called()
        self.assertEqual(self._form_values(frame), previous_form)
        self.assertEqual(frame.raw_import_params, previous_params)
        self.assertEqual(frame.sdb_params_summary_var.get(), previous_summary)
        self.assertIs(frame.raw_image, previous_image)
        self.assertIs(frame._source_raw_image, previous_source)
        self.assertIs(frame._working_source_raw_image, previous_working)
        self.assertTrue(frame._sdb_manual_import)
        frame.width_var.set("8")
        np.testing.assert_array_equal(frame.raw_image, expected)

    def test_malformed_recognized_container_does_not_fall_back_to_raw_guesses(self):
        source, _expected = self._native_source()
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(source)))
        broken = self.folder / "broken.sdb"
        broken.write_bytes(b"CMDb" + b"\0" * 32)
        previous_image = frame.raw_image
        previous_form = self._form_values(frame)

        with patch(f"{STEP1}.read_raw_oct") as raw:
            self.assertFalse(frame._open_raw(str(broken)))

        self.errors.assert_called_once()
        raw.assert_not_called()
        self.assertIs(frame.raw_image, previous_image)
        self.assertEqual(frame.current_file, str(source))
        self.assertEqual(self._form_values(frame), previous_form)

    def test_companion_disagreement_is_visible_without_patient_text(self):
        source, words = self._native_source(edb="mismatch")
        frame = self._frame()

        self.assertTrue(frame._open_raw(str(source)))

        self.errors.assert_not_called()
        np.testing.assert_array_equal(frame.raw_image, self._full_pixels(frame, source))
        self._assert_native_words_in_full_view(frame, words)
        summary = frame.sdb_params_summary_var.get()
        self.assertIn("8 × 6", summary)
        self.assertIn("EDB mismatch", summary)
        self.assertNotIn("confirmed", summary)
        self.assertNotIn("private fixture patient", summary)

    def test_saved_crop_preview_is_preserved_then_undo_detects_original_before_deletion(self):
        source, words = self._native_source()
        saved = np.full((3, 7), 11135, dtype=np.int16)
        write_analyze(self.folder / "light", np.stack([saved, saved]))
        old_image_bytes = (self.folder / "light.img").read_bytes()
        frame = self._frame()

        with patch(f"{STEP1}.detect_sdb_image", wraps=detect_sdb_image) as detect:
            self.assertTrue(frame._open_raw(str(source)))
            detect.assert_not_called()

        np.testing.assert_array_equal(frame.processed_image, saved)
        self.assertIsNone(frame.raw_image)
        self.assertIsNone(frame._sdb_image_info)
        self.assertFalse(frame.image_canvas.roi_enabled)
        self.assertIn("Saved crop", frame.sdb_params_summary_var.get())
        self.assertEqual((self.folder / "light.img").read_bytes(), old_image_bytes)
        np.testing.assert_array_equal(read_analyze(self.folder / "light")[0], saved)

        original_remove = os.remove
        read_before_delete = []

        def remove_after_detection(path):
            read_before_delete.append(detect.called and read_raw.called)
            original_remove(path)

        with patch(f"{STEP1}.detect_sdb_image", wraps=detect_sdb_image) as detect, patch(
            f"{STEP1}.read_raw_oct", wraps=read_raw_oct
        ) as read_raw, patch(f"{STEP1}.os.remove", side_effect=remove_after_detection):
            self.assertTrue(frame._reset())

        self.errors.assert_not_called()
        self.assertEqual(read_before_delete, [True, True])
        self.assertFalse((self.folder / "light.img").exists())
        self.assertFalse((self.folder / "light.hdr").exists())
        np.testing.assert_array_equal(frame.raw_image, self._full_pixels(frame, source))
        self._assert_native_words_in_full_view(frame, words)
        self.assertTrue(frame.image_canvas.roi_enabled)
        self.assertIsNone(frame.processed_image)
        self.assertIn("Auto-detected OCT: 8 × 6", frame.sdb_params_summary_var.get())
        self.assertIn("Full SDB view", frame.sdb_params_summary_var.get())

    def test_undo_cancellation_preserves_saved_crop_and_parameters(self):
        source, _expected = self._native_source()
        saved = np.full((3, 7), 11135, dtype=np.int16)
        write_analyze(self.folder / "light", np.stack([saved, saved]))
        old_image_bytes = (self.folder / "light.img").read_bytes()
        old_header_bytes = (self.folder / "light.hdr").read_bytes()
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(source)))
        previous_form = self._form_values(frame)
        previous_summary = frame.sdb_params_summary_var.get()
        previous_image = frame.processed_image

        with patch.object(frame, "_offer_bright_pixel_removal", return_value=None), patch(
            f"{STEP1}.os.remove"
        ) as remove:
            self.assertFalse(frame._reset())

        self.errors.assert_not_called()
        remove.assert_not_called()
        self.assertIs(frame.processed_image, previous_image)
        self.assertIsNone(frame.raw_image)
        self.assertIsNone(frame._sdb_image_info)
        self.assertEqual(self._form_values(frame), previous_form)
        self.assertEqual(frame.sdb_params_summary_var.get(), previous_summary)
        self.assertEqual((self.folder / "light.img").read_bytes(), old_image_bytes)
        self.assertEqual((self.folder / "light.hdr").read_bytes(), old_header_bytes)

    def test_undo_saved_legacy_file_after_native_uses_legacy_defaults(self):
        native, _expected_native = self._native_source()
        legacy_folder = self.folder / "legacy"
        legacy_folder.mkdir()
        legacy = legacy_folder / "legacy.sdb"
        expected = np.arange(48, dtype=np.uint16).reshape(6, 8)
        legacy.write_bytes(expected.astype("<u2").tobytes())
        saved = np.full((3, 7), 11135, dtype=np.int16)
        write_analyze(legacy_folder / "light", np.stack([saved, saved]))
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(native)))
        self.assertTrue(frame._open_raw(str(legacy)))
        np.testing.assert_array_equal(frame.processed_image, saved)

        self.assertTrue(frame._reset())

        self.errors.assert_not_called()
        np.testing.assert_array_equal(frame.raw_image, expected)
        self.assertEqual(self._form_values(frame), ("8", "6", "0", True))
        self.assertFalse((legacy_folder / "light.img").exists())
        self.assertFalse((legacy_folder / "light.hdr").exists())

    def test_cancelled_legacy_open_preserves_saved_crop_summary_and_controls(self):
        source = self.folder / "legacy.sdb"
        raw = np.arange(48, dtype=np.uint16).reshape(6, 8)
        source.write_bytes(raw.astype("<u2").tobytes())
        saved = np.full((3, 7), 11135, dtype=np.int16)
        write_analyze(self.folder / "light", np.stack([saved, saved]))
        incoming_folder = self.folder / "incoming"
        incoming_folder.mkdir()
        incoming = incoming_folder / "next.sdb"
        incoming.write_bytes(raw.astype("<u2").tobytes())
        frame = self._frame()
        self.assertTrue(frame._open_raw(str(source)))
        previous_summary = frame.sdb_params_summary_var.get()
        previous_form = self._form_values(frame)
        previous_image = frame.processed_image

        with patch.object(frame, "_offer_bright_pixel_removal", return_value=None):
            self.assertFalse(frame._open_raw(str(incoming)))

        self.errors.assert_not_called()
        self.assertIs(frame.processed_image, previous_image)
        self.assertEqual(frame.current_file, str(source))
        self.assertEqual(frame.sdb_params_summary_var.get(), previous_summary)
        self.assertEqual(self._form_values(frame), previous_form)
        self.assertTrue(frame._sdb_showing_saved_crop)

    def test_legacy_raw_file_keeps_manual_parameters_and_pixels(self):
        source = self.folder / "legacy.sdb"
        expected = np.arange(48, dtype=np.uint16).reshape(6, 8)
        source.write_bytes(expected.astype("<u2").tobytes())
        frame = self._frame()

        self.assertTrue(frame._open_raw(str(source)))

        self.errors.assert_not_called()
        self.assertEqual(frame.current_file, str(source))
        self.assertIsNone(frame._sdb_image_info)
        self.assertEqual(self._form_values(frame), ("8", "6", "0", True))
        np.testing.assert_array_equal(frame.raw_image, expected)
        self.assertTrue(frame.image_canvas.roi_enabled)
        self.assertNotIn("Auto-detected", frame.sdb_params_summary_var.get())


if __name__ == "__main__":
    unittest.main()
