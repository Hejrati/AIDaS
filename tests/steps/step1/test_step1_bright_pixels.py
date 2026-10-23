from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from aidas.ai.inference import AIForAIDASPredictor
from aidas.steps.step1_resize_raw import Step1Frame
from aidas.steps.step2_annotate import Step2Frame
from aidas.utils.bright_pixels import detect_bright_pixels, detect_bright_pixels_in_bands
from aidas.utils.io_utils import read_analyze, read_raw_oct, write_analyze


STEP1 = "aidas.steps.step1_resize_raw"


class _StringVarStub:
    def __init__(self, value="source"):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _CanvasStub:
    def __init__(self):
        self.image = None
        self.roi_enabled = True

    def set_image(self, image):
        self.image = image

    def enable_roi(self, enabled):
        self.roi_enabled = enabled


def _signed_crop():
    # Unsigned SDB samples become negative anatomical values in the existing
    # signed Analyze output. These two positive artifacts then dominate display.
    crop = np.linspace(-12500, -8000, 64 * 64).reshape(64, 64).astype(np.int16)
    crop[-2:] = 0
    crop[20, 31] = 32005
    crop[40, 17] = 32555
    return crop


def _expected_mask():
    mask = np.zeros((64, 64), dtype=bool)
    mask[20, 31] = True
    mask[40, 17] = True
    return mask


def _mixed_signed_source():
    # The full source has a much broader histogram than the affected tissue.
    # The default crop at y=585 misses both artifacts, so opening must check
    # source bands before any ROI is selected.
    source = np.linspace(11000, 29000, 768 * 128).reshape(768, 128).astype(np.int16)
    source[192:320] = np.linspace(-12500, -8000, 128 * 128).reshape(128, 128)
    source[200, 31] = 32005
    source[280, 17] = 32555
    return source


def _mixed_source_mask():
    mask = np.zeros((768, 128), dtype=bool)
    mask[200, 31] = True
    mask[280, 17] = True
    return mask


class Step1BrightPixelTests(unittest.TestCase):
    def _crop_frame(self, folder, *, signed=None, roi=None):
        signed = _signed_crop() if signed is None else signed
        source = folder / "source.sdb"
        source.write_bytes(signed.astype("<u2").tobytes())
        # Read through the real SDB reader: its source array is read-only.
        height, width = signed.shape
        raw = read_raw_oct(source, width=width, height=height, offset=0)
        frame = Step1Frame.__new__(Step1Frame)
        frame.current_file = str(source)
        frame.raw_image = raw
        frame._source_raw_image = raw.copy()
        frame._source_roi = roi or (0, 0, width, height)
        frame.processed_image = None
        frame._processed_image_saved = False
        frame._sdb_directories = [str(folder)]
        frame._sdb_directory_files = {str(folder): [str(source)]}
        frame._cropped_sdb_files = set()
        frame._saved_crop_previews = {}
        frame._saved_output_directories = set()
        frame.image_canvas = _CanvasStub()
        frame.view_mode_var = _StringVarStub()
        frame.bright_pixel_status_var = _StringVarStub("Not checked")
        frame._update_zoom_label = lambda: None
        frame._update_image_status = lambda: None
        frame._update_processing_controls = lambda: None
        frame._set_sdb_parameters_enabled = lambda _enabled: None
        frame._set_roi_controls_enabled = lambda _enabled: None
        frame._refresh_sdb_progress_colors = lambda: None
        frame._update_batch_handoff_button_state = lambda: None
        frame._update_queue_controls = lambda: None
        frame._on_processed_image = Mock()
        frame.notifications = []
        frame._show_crop_notification = lambda message, **_kwargs: frame.notifications.append(message)
        return frame

    def _open_frame(self, folder, *, signed=None):
        frame = self._crop_frame(folder, signed=_mixed_signed_source() if signed is None else signed)
        source = Path(frame.current_file)
        incoming = frame.raw_image
        frame.width_var = _StringVarStub(str(incoming.shape[1]))
        frame.current_file = None
        frame.raw_image = None
        frame._source_raw_image = None
        frame._source_roi = None

        def set_default_roi():
            height, width = frame.raw_image.shape
            y = min(585, height - 1)
            frame._source_roi = (0, y, width, min(128, height - y))

        frame._set_default_roi = set_default_roi
        frame._set_initial_roi = set_default_roi
        return frame, incoming, source

    def _reopened_saved_frame(self, folder):
        frame, incoming, source = self._open_frame(folder)
        saved = np.full((4, 4), -11000, dtype=np.int16)
        write_analyze(folder / "light", np.stack([saved] * 2))
        frame.current_file = str(source)
        frame.processed_image = saved
        frame._processed_image_saved = True
        frame._working_source_raw_image = None
        frame._source_bright_pixel_mask = None
        frame._raw_bright_pixel_mask = None
        frame._source_bright_pixel_threshold = None
        frame._bright_pixel_source_checked = False
        frame._source_bright_pixels_smoothed = False
        frame.raw_import_params = {
            "width": 128,
            "height": 768,
            "offset": 0,
            "bit_depth": 16,
            "little_endian": True,
        }
        frame._apply_import_params = Mock(return_value=True)
        frame._render_sdb_directories = lambda: None
        frame._cropped_sdb_files.add(frame._path_key(str(source)))
        frame._saved_crop_previews[frame._path_key(str(source))] = saved.copy()
        frame._saved_output_directories.add(frame._path_key(str(folder)))
        frame.image_canvas.set_image(saved)
        frame.image_canvas.enable_roi(False)
        frame.view_mode_var.set("target")
        return frame, incoming, source

    def test_accepting_removal_changes_only_detected_pixels(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.bright_pixel_status_var = _StringVarStub()
        crop = _signed_crop()
        before = crop.copy()
        expected_mask = _expected_mask()
        np.testing.assert_array_equal(detect_bright_pixels(crop).mask, expected_mask)

        with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=True) as prompt, patch(
            f"{STEP1}.messagebox.askyesnocancel"
        ) as native_prompt:
            cleaned = frame._offer_bright_pixel_removal(crop)

        np.testing.assert_array_equal(crop, before)
        self.assertEqual(cleaned.dtype, crop.dtype)
        self.assertEqual(cleaned[~expected_mask].tobytes(), before[~expected_mask].tobytes())
        np.testing.assert_array_equal(cleaned != before, expected_mask)
        self.assertTrue(np.all(cleaned[expected_mask] < 0))
        prompt.assert_called_once()
        self.assertEqual(prompt.call_args.args[0].count, 2)
        np.testing.assert_array_equal(prompt.call_args.args[0].mask, expected_mask)
        self.assertIs(prompt.call_args.args[1], crop)
        self.assertEqual(prompt.call_args.kwargs["context"], "crop")
        native_prompt.assert_not_called()
        self.assertIn("smoothed 2", frame.bright_pixel_status_var.get().lower())

    def test_declining_removal_returns_the_original_crop(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.bright_pixel_status_var = _StringVarStub()
        crop = _signed_crop()
        before = crop.tobytes()

        with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=False) as prompt, patch(
            f"{STEP1}.smooth_bright_pixels"
        ) as smooth:
            self.assertIs(frame._offer_bright_pixel_removal(crop), crop)

        self.assertEqual(crop.tobytes(), before)
        prompt.assert_called_once()
        smooth.assert_not_called()
        self.assertIn("kept 2", frame.bright_pixel_status_var.get().lower())

    def test_canceling_removal_returns_no_crop(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.bright_pixel_status_var = _StringVarStub()
        crop = _signed_crop()
        before = crop.tobytes()

        with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=None) as prompt, patch(
            f"{STEP1}.smooth_bright_pixels"
        ) as smooth:
            self.assertIsNone(frame._offer_bright_pixel_removal(crop))

        self.assertEqual(crop.tobytes(), before)
        prompt.assert_called_once()
        smooth.assert_not_called()
        self.assertIn("cancel", frame.bright_pixel_status_var.get().lower())

    def test_normal_anatomical_bands_do_not_prompt_for_removal(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.bright_pixel_status_var = _StringVarStub()
        rows = np.arange(96, dtype=np.float64)[:, None]
        columns = np.linspace(-350, 350, 128)[None, :]
        # A broad continuous anatomical band has many bright pixels, without
        # the separated, sparse histogram tail of the supplied SDB artifacts.
        crop = (-12500 + 5500 * np.exp(-((rows - 48) / 8) ** 2) + columns).astype(np.int16)
        before = crop.tobytes()

        with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action") as prompt, patch(
            f"{STEP1}.smooth_bright_pixels"
        ) as smooth:
            self.assertIs(frame._offer_bright_pixel_removal(crop), crop)

        self.assertEqual(crop.tobytes(), before)
        prompt.assert_not_called()
        smooth.assert_not_called()
        self.assertIn("no bright", frame.bright_pixel_status_var.get().lower())

    def test_opening_checks_all_source_bands_before_the_default_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, incoming, source = self._open_frame(Path(directory))
            signed = _mixed_signed_source()
            expected_mask = _mixed_source_mask()
            self.assertEqual(detect_bright_pixels(signed).count, 0)
            np.testing.assert_array_equal(detect_bright_pixels_in_bands(signed).mask, expected_mask)
            before = source.read_bytes()
            incoming_before = incoming.tobytes()

            with patch(
                f"{STEP1}.detect_bright_pixels_in_bands", wraps=detect_bright_pixels_in_bands
            ) as detect, patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=True) as prompt:
                self.assertTrue(frame._load_image(incoming, str(source)))

            detect.assert_called_once()
            self.assertEqual(detect.call_args.args[0].dtype, np.dtype("int16"))
            np.testing.assert_array_equal(detect.call_args.args[0], signed)
            prompt.assert_called_once()
            self.assertEqual(prompt.call_args.kwargs["context"], "SDB image")
            self.assertEqual(prompt.call_args.args[0].count, 2)
            np.testing.assert_array_equal(prompt.call_args.args[1], incoming)
            self.assertEqual(prompt.call_args.args[1].dtype, incoming.dtype)
            x, y, width, height = frame._source_roi
            self.assertFalse(expected_mask[y:y + height, x:x + width].any())
            cleaned = frame.raw_image.astype(np.int16)
            np.testing.assert_array_equal(cleaned != signed, expected_mask)
            self.assertEqual(cleaned[~expected_mask].tobytes(), signed[~expected_mask].tobytes())
            np.testing.assert_array_equal(frame._source_raw_image, incoming)
            np.testing.assert_array_equal(frame._working_source_raw_image, frame.raw_image)
            np.testing.assert_array_equal(frame._source_bright_pixel_mask, expected_mask)
            self.assertFalse(frame._source_bright_pixel_mask.flags.writeable)
            self.assertTrue(frame._bright_pixel_source_checked)
            self.assertIsNotNone(frame._source_bright_pixel_threshold)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(incoming.tobytes(), incoming_before)
            self.assertFalse(incoming.flags.writeable)
            self.assertIn("smoothed 2", frame.bright_pixel_status_var.get().lower())

    def test_opening_declined_source_keeps_pixels_and_reports_the_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, incoming, source = self._open_frame(Path(directory))
            before = source.read_bytes()
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=False) as prompt, patch(
                f"{STEP1}.smooth_bright_pixels"
            ) as smooth:
                self.assertTrue(frame._load_image(incoming, str(source)))

            np.testing.assert_array_equal(frame.raw_image, incoming)
            np.testing.assert_array_equal(frame._source_raw_image, incoming)
            np.testing.assert_array_equal(frame._source_bright_pixel_mask, _mixed_source_mask())
            self.assertFalse(frame._source_bright_pixel_mask.flags.writeable)
            self.assertTrue(frame._bright_pixel_source_checked)
            self.assertEqual(source.read_bytes(), before)
            self.assertIn("kept 2", frame.bright_pixel_status_var.get().lower())
            prompt.assert_called_once()
            smooth.assert_not_called()

    def test_canceling_open_preserves_previous_image_and_saved_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame, incoming, source = self._open_frame(folder)
            previous_raw = np.full((12, 16), 54000, dtype=np.uint16)
            previous_working = previous_raw.copy()
            previous_original = previous_raw.copy()
            previous_processed = np.full((4, 12), -11100, dtype=np.int16)
            previous_mask = np.zeros(previous_raw.shape, dtype=bool)
            previous_mask.setflags(write=False)
            frame.raw_image = previous_raw
            frame._source_raw_image = previous_original
            frame._working_source_raw_image = previous_working
            frame._source_bright_pixel_mask = previous_mask
            frame._raw_bright_pixel_mask = previous_mask
            frame._source_bright_pixel_threshold = 23000.0
            frame._bright_pixel_source_checked = True
            frame.processed_image = previous_processed
            frame._processed_image_saved = True
            frame.current_file = str(folder / "previous.sdb")
            previous_roi = (1, 2, 4, 4)
            frame._source_roi = previous_roi
            frame.image_canvas.set_image(previous_processed)
            frame.image_canvas.enable_roi(False)
            frame.view_mode_var.set("target")
            frame._cropped_sdb_files.add(frame._path_key(frame.current_file))
            (folder / "light.img").write_bytes(b"existing saved image")
            (folder / "light.hdr").write_bytes(b"existing saved header")
            source_before = source.read_bytes()

            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=None) as prompt, patch(
                f"{STEP1}.write_analyze"
            ) as save, patch(f"{STEP1}.os.remove") as remove:
                self.assertFalse(frame._load_image(incoming, str(source)))

            self.assertIs(frame.raw_image, previous_raw)
            self.assertIs(frame._source_raw_image, previous_original)
            self.assertIs(frame._working_source_raw_image, previous_working)
            self.assertIs(frame._source_bright_pixel_mask, previous_mask)
            self.assertIs(frame._raw_bright_pixel_mask, previous_mask)
            self.assertEqual(frame._source_bright_pixel_threshold, 23000.0)
            self.assertIs(frame.processed_image, previous_processed)
            self.assertTrue(frame._processed_image_saved)
            self.assertEqual(frame.current_file, str(folder / "previous.sdb"))
            self.assertEqual(frame._source_roi, previous_roi)
            self.assertIs(frame.image_canvas.image, previous_processed)
            self.assertFalse(frame.image_canvas.roi_enabled)
            self.assertEqual(frame.view_mode_var.get(), "target")
            self.assertEqual((folder / "light.img").read_bytes(), b"existing saved image")
            self.assertEqual((folder / "light.hdr").read_bytes(), b"existing saved header")
            self.assertEqual(source.read_bytes(), source_before)
            self.assertIn("cancel", frame.bright_pixel_status_var.get().lower())
            prompt.assert_called_once()
            save.assert_not_called()
            remove.assert_not_called()

    def test_source_corrections_survive_width_preview_cropping_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, incoming, source = self._open_frame(Path(directory))
            source_before = source.read_bytes()
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=True) as prompt, patch(
                f"{STEP1}.read_raw_oct"
            ) as read:
                self.assertTrue(frame._load_image(incoming, str(source)))
                corrected = frame._working_source_raw_image.copy()
                frame.width_var.set("116")
                frame._apply_width_preview_adjustment()
                np.testing.assert_array_equal(frame.raw_image, corrected[:, 6:122])
                np.testing.assert_array_equal(frame._raw_bright_pixel_mask, _mixed_source_mask()[:, 6:122])
                frame.width_var.set("144")
                frame._apply_width_preview_adjustment()
                np.testing.assert_array_equal(frame.raw_image, np.pad(corrected, ((0, 0), (8, 8))))
                np.testing.assert_array_equal(
                    frame._raw_bright_pixel_mask, np.pad(_mixed_source_mask(), ((0, 0), (8, 8)))
                )
                frame.width_var.set("128")
                frame._apply_width_preview_adjustment()

            np.testing.assert_array_equal(frame.raw_image, corrected)
            np.testing.assert_array_equal(frame._source_raw_image, incoming)
            np.testing.assert_array_equal(frame._working_source_raw_image, corrected)
            np.testing.assert_array_equal(frame._source_bright_pixel_mask, _mixed_source_mask())
            self.assertEqual(source.read_bytes(), source_before)
            self.assertIn("smoothed 2", frame.bright_pixel_status_var.get().lower())
            prompt.assert_called_once()
            read.assert_not_called()

    def test_source_correction_saves_in_a_custom_crop_without_another_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame, incoming, source = self._open_frame(folder)
            signed = _mixed_signed_source()
            source_before = source.read_bytes()
            roi = (8, 192, 48, 128)
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=True) as prompt:
                self.assertTrue(frame._load_image(incoming, str(source)))
                frame._source_roi = roi
                self.assertTrue(frame._crop_and_scale())

            x, y, width, height = roi
            original_crop = signed[y:y + height, x:x + width]
            expected = np.repeat(original_crop, 3, axis=1)
            mask = np.repeat(_mixed_source_mask()[y:y + height, x:x + width], 3, axis=1)
            saved = read_analyze(folder / "light")
            np.testing.assert_array_equal(saved[0] != expected, mask)
            np.testing.assert_array_equal(saved[0], saved[1])
            self.assertEqual(saved[0][~mask].tobytes(), expected[~mask].tobytes())
            expected_bytes = np.stack([np.flipud(expected)] * 2).astype(">i2").view(np.uint8)
            actual_bytes = np.frombuffer((folder / "light.img").read_bytes(), dtype=np.uint8)
            actual_bytes = actual_bytes.reshape(expected_bytes.shape)
            byte_mask = np.repeat(np.stack([np.flipud(mask)] * 2), 2, axis=2)
            np.testing.assert_array_equal(actual_bytes[~byte_mask], expected_bytes[~byte_mask])
            self.assertEqual(source.read_bytes(), source_before)
            np.testing.assert_array_equal(frame._source_raw_image, incoming)
            self.assertIn("smoothed 2", frame.bright_pixel_status_var.get().lower())
            prompt.assert_called_once()

    def test_declined_source_artifacts_are_saved_without_another_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame, incoming, source = self._open_frame(folder)
            signed = _mixed_signed_source()
            source_before = source.read_bytes()
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=False) as prompt, patch(
                f"{STEP1}.smooth_bright_pixels"
            ) as smooth:
                self.assertTrue(frame._load_image(incoming, str(source)))
                frame._source_roi = (8, 192, 48, 128)
                self.assertTrue(frame._crop_and_scale())

            expected = np.repeat(signed[192:320, 8:56], 3, axis=1)
            np.testing.assert_array_equal(read_analyze(folder / "light"), np.stack([expected] * 2))
            expected_bytes = np.stack([np.flipud(expected)] * 2).astype(">i2").tobytes()
            self.assertEqual((folder / "light.img").read_bytes(), expected_bytes)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertIn("kept 2", frame.bright_pixel_status_var.get().lower())
            prompt.assert_called_once()
            smooth.assert_not_called()

    def test_accepting_saved_crop_removal_changes_only_masked_saved_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame = self._crop_frame(folder)
            saved_input = np.repeat(_signed_crop(), 3, axis=1)
            mask = np.repeat(_expected_mask(), 3, axis=1)
            write_analyze(folder / "light", np.stack([saved_input] * 2))
            original_bytes = (folder / "light.img").read_bytes()
            source_before = Path(frame.current_file).read_bytes()
            input_before = saved_input.tobytes()
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=True) as prompt:
                self.assertTrue(frame._load_saved_crop(saved_input, frame.current_file))

            saved = read_analyze(folder / "light")
            np.testing.assert_array_equal(saved[0] != saved_input, mask)
            np.testing.assert_array_equal(saved[0], saved[1])
            np.testing.assert_array_equal(frame.processed_image, saved[0])
            byte_mask = np.repeat(np.stack([np.flipud(mask)] * 2), 2, axis=2)
            old_bytes = np.frombuffer(original_bytes, dtype=np.uint8).reshape(byte_mask.shape)
            new_bytes = np.frombuffer((folder / "light.img").read_bytes(), dtype=np.uint8)
            new_bytes = new_bytes.reshape(byte_mask.shape)
            np.testing.assert_array_equal(new_bytes[~byte_mask], old_bytes[~byte_mask])
            self.assertEqual(Path(frame.current_file).read_bytes(), source_before)
            self.assertEqual(saved_input.tobytes(), input_before)
            self.assertTrue(frame._processed_image_saved)
            self.assertIn("smoothed 6", frame.bright_pixel_status_var.get().lower())
            self.assertEqual(prompt.call_args.kwargs["context"], "saved crop")
            self.assertEqual(prompt.call_args.args[0].count, 6)
            np.testing.assert_array_equal(prompt.call_args.args[1], saved_input)
            prompt.assert_called_once()

    def test_declining_saved_crop_removal_keeps_existing_output_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame = self._crop_frame(folder)
            saved_input = np.repeat(_signed_crop(), 3, axis=1)
            write_analyze(folder / "light", np.stack([saved_input] * 2))
            original_img = (folder / "light.img").read_bytes()
            original_hdr = (folder / "light.hdr").read_bytes()
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=False) as prompt, patch(
                f"{STEP1}.write_analyze"
            ) as save:
                self.assertTrue(frame._load_saved_crop(saved_input, frame.current_file))

            np.testing.assert_array_equal(frame.processed_image, saved_input)
            self.assertEqual((folder / "light.img").read_bytes(), original_img)
            self.assertEqual((folder / "light.hdr").read_bytes(), original_hdr)
            self.assertTrue(frame._processed_image_saved)
            self.assertIn("kept 6", frame.bright_pixel_status_var.get().lower())
            prompt.assert_called_once()
            save.assert_not_called()

    def test_canceling_saved_crop_open_preserves_current_source_and_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame = self._crop_frame(folder)
            frame.image_canvas.set_image(frame.raw_image)
            original_raw = frame.raw_image
            original_source = frame._source_raw_image
            original_roi = frame._source_roi
            saved_input = np.repeat(_signed_crop(), 3, axis=1)
            write_analyze(folder / "light", np.stack([saved_input] * 2))
            original_img = (folder / "light.img").read_bytes()
            original_hdr = (folder / "light.hdr").read_bytes()
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=None) as prompt, patch(
                f"{STEP1}.write_analyze"
            ) as save, patch(f"{STEP1}.os.remove") as remove:
                self.assertFalse(frame._load_saved_crop(saved_input, frame.current_file))

            self.assertIs(frame.raw_image, original_raw)
            self.assertIs(frame._source_raw_image, original_source)
            self.assertEqual(frame._source_roi, original_roi)
            self.assertIsNone(frame.processed_image)
            self.assertFalse(frame._processed_image_saved)
            self.assertIs(frame.image_canvas.image, original_raw)
            self.assertTrue(frame.image_canvas.roi_enabled)
            self.assertEqual(frame.view_mode_var.get(), "source")
            self.assertEqual((folder / "light.img").read_bytes(), original_img)
            self.assertEqual((folder / "light.hdr").read_bytes(), original_hdr)
            self.assertIn("cancel", frame.bright_pixel_status_var.get().lower())
            prompt.assert_called_once()
            save.assert_not_called()
            remove.assert_not_called()

    def test_canceling_source_check_during_undo_preserves_saved_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame, _incoming, source = self._reopened_saved_frame(folder)
            previous_processed = frame.processed_image
            old_img = (folder / "light.img").read_bytes()
            old_hdr = (folder / "light.hdr").read_bytes()
            source_before = source.read_bytes()
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=None) as prompt, patch(
                f"{STEP1}.os.remove"
            ) as remove, patch(f"{STEP1}.write_analyze") as save, patch(
                f"{STEP1}.detect_bright_pixels_in_bands", wraps=detect_bright_pixels_in_bands
            ) as detect:
                self.assertFalse(frame._reset())

            self.assertIs(frame.processed_image, previous_processed)
            self.assertIs(frame.image_canvas.image, previous_processed)
            self.assertTrue(frame._processed_image_saved)
            self.assertIsNone(frame.raw_image)
            self.assertIsNone(frame._source_raw_image)
            self.assertIsNone(frame._working_source_raw_image)
            self.assertIsNone(frame._source_bright_pixel_mask)
            self.assertIsNone(frame._source_roi)
            self.assertEqual(frame.current_file, str(source))
            self.assertEqual(frame.view_mode_var.get(), "target")
            self.assertFalse(frame.image_canvas.roi_enabled)
            self.assertIn(frame._path_key(str(source)), frame._cropped_sdb_files)
            self.assertIn(frame._path_key(str(folder)), frame._saved_output_directories)
            self.assertEqual((folder / "light.img").read_bytes(), old_img)
            self.assertEqual((folder / "light.hdr").read_bytes(), old_hdr)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(frame.notifications, [])
            self.assertIn("cancel", frame.bright_pixel_status_var.get().lower())
            self.assertEqual(prompt.call_args.kwargs["context"], "SDB image")
            self.assertEqual(prompt.call_args.args[0].count, 2)
            prompt.assert_called_once()
            detect.assert_called_once()
            remove.assert_not_called()
            save.assert_not_called()

    def test_accepting_source_check_during_undo_repairs_even_a_tiny_recrop(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame, incoming, source = self._reopened_saved_frame(folder)
            signed = _mixed_signed_source()
            source_before = source.read_bytes()

            def accept_before_saved_files_are_deleted(*_args, **_kwargs):
                self.assertTrue((folder / "light.img").exists())
                self.assertTrue((folder / "light.hdr").exists())
                return True

            with patch(
                f"{STEP1}.Step1Frame._ask_bright_pixel_action", side_effect=accept_before_saved_files_are_deleted
            ) as prompt, patch(
                f"{STEP1}.detect_bright_pixels_in_bands", wraps=detect_bright_pixels_in_bands
            ) as detect:
                self.assertTrue(frame._reset())
                detect.assert_called_once()
                np.testing.assert_array_equal(detect.call_args.args[0], signed)
                np.testing.assert_array_equal(frame._source_raw_image, incoming)
                cleaned = frame.raw_image.astype(np.int16)
                np.testing.assert_array_equal(cleaned != signed, _mixed_source_mask())
                np.testing.assert_array_equal(frame._working_source_raw_image, frame.raw_image)
                np.testing.assert_array_equal(frame._source_bright_pixel_mask, _mixed_source_mask())
                self.assertFalse(frame._source_bright_pixel_mask.flags.writeable)
                self.assertTrue(frame._bright_pixel_source_checked)
                self.assertTrue(frame._source_bright_pixels_smoothed)
                self.assertIsNone(frame.processed_image)
                self.assertFalse(frame._processed_image_saved)
                self.assertEqual(frame.view_mode_var.get(), "source")
                self.assertFalse((folder / "light.img").exists())
                self.assertFalse((folder / "light.hdr").exists())
                self.assertEqual(source.read_bytes(), source_before)
                self.assertEqual(frame.notifications, ["Saved crop removed"])
                # Nine source pixels are too few for the local histogram. The
                # approved source repair must still survive this arbitrary ROI.
                frame._source_roi = (30, 199, 3, 3)
                original_crop = signed[199:202, 30:33]
                self.assertEqual(detect_bright_pixels_in_bands(original_crop).count, 0)
                self.assertTrue(frame._crop_and_scale())

            expected = np.repeat(original_crop, 3, axis=1)
            mask = np.repeat(_mixed_source_mask()[199:202, 30:33], 3, axis=1)
            saved = read_analyze(folder / "light")
            self.assertEqual(saved.shape, (2, 3, 9))
            np.testing.assert_array_equal(saved[0] != expected, mask)
            np.testing.assert_array_equal(saved[0], saved[1])
            self.assertEqual(saved[0][~mask].tobytes(), expected[~mask].tobytes())
            np.testing.assert_array_equal(saved[0], np.repeat(cleaned[199:202, 30:33], 3, axis=1))
            np.testing.assert_array_equal(frame._source_raw_image, incoming)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(frame.notifications, ["Saved crop removed", "Saved successfully"])
            self.assertIn("smoothed 2", frame.bright_pixel_status_var.get().lower())
            prompt.assert_called_once()

    def test_opening_a_normal_source_skips_the_prompt_and_reports_check(self):
        with tempfile.TemporaryDirectory() as directory:
            signed = np.linspace(-12500, -8000, 768 * 128).reshape(768, 128).astype(np.int16)
            frame, incoming, source = self._open_frame(Path(directory), signed=signed)
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action") as prompt:
                self.assertTrue(frame._load_image(incoming, str(source)))

            np.testing.assert_array_equal(frame.raw_image, incoming)
            self.assertFalse(frame._source_bright_pixel_mask.any())
            self.assertTrue(frame._bright_pixel_source_checked)
            self.assertIn("no bright", frame.bright_pixel_status_var.get().lower())
            prompt.assert_not_called()

    def test_accepting_crop_preserves_unmasked_saved_bytes_and_source(self):
        for repair in (True, False):
            with self.subTest(repair=repair), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                signed = _signed_crop()
                mask = _expected_mask()
                if not repair:
                    signed[mask] = -11000
                    mask[:] = False
                frame = self._crop_frame(folder, signed=signed)
                source_before = Path(frame.current_file).read_bytes()
                raw_before = frame.raw_image.tobytes()
                preserved_source_before = frame._source_raw_image.tobytes()

                with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=True) as prompt:
                    self.assertTrue(frame._crop_and_scale())

                original_target = np.repeat(signed, 3, axis=1)
                target_mask = np.repeat(mask, 3, axis=1)
                saved = read_analyze(folder / "light")
                self.assertEqual(frame.processed_image.dtype, np.dtype("int16"))
                self.assertEqual(saved.dtype, np.dtype("int16"))
                self.assertEqual(saved.shape, (2, 64, 192))
                np.testing.assert_array_equal(saved[0], saved[1])
                np.testing.assert_array_equal(saved[0] != original_target, target_mask)
                self.assertEqual(
                    saved[0][~target_mask].tobytes(), original_target[~target_mask].tobytes()
                )
                header = (folder / "light.hdr").read_bytes()
                endian = "<" if struct.unpack_from("<i", header, 0)[0] == 348 else ">"
                self.assertEqual(struct.unpack_from(f"{endian}hh", header, 70), (4, 16))
                # Analyze writes two vertically flipped, big-endian slices. Compare
                # bytes in the actual IMG file, excluding only the detected pixels.
                expected_bytes = np.stack([np.flipud(original_target)] * 2).astype(">i2").view(np.uint8)
                actual_bytes = np.frombuffer((folder / "light.img").read_bytes(), dtype=np.uint8)
                actual_bytes = actual_bytes.reshape(expected_bytes.shape)
                byte_mask = np.repeat(np.stack([np.flipud(target_mask)] * 2), 2, axis=2)
                np.testing.assert_array_equal(actual_bytes[~byte_mask], expected_bytes[~byte_mask])

                working, template, source_was_8bit = Step2Frame._read_image_for_annotation(
                    str(folder / "light.img")
                )
                self.assertEqual(working.dtype, np.dtype("int16"))
                self.assertEqual(working.tobytes(), frame.processed_image.tobytes())
                self.assertFalse(source_was_8bit)
                self.assertEqual(template, {
                    "slices": 2, "height": 64, "width": 192, "dtype": np.dtype("int16"),
                })
                working_before = working.tobytes()
                predictor = AIForAIDASPredictor.__new__(AIForAIDASPredictor)
                predictor._progress_callback = None
                predictor.boundary_model_path = "test model"
                predictor._session = Mock()
                predictor._session.get_providers.return_value = ["CPUExecutionProvider"]
                predictor.fallback_reason = None
                predictor._run_model = Mock(
                    return_value=np.zeros((1, 6, *working.shape), dtype=np.float32)
                )

                predictor.predict(working)

                predictor._run_model.assert_called_once()
                model_input = predictor._run_model.call_args.args[0]
                self.assertEqual(model_input.shape, (1, 1, *working.shape))
                self.assertEqual(model_input.dtype, np.dtype("float32"))
                self.assertTrue(model_input.flags.c_contiguous)
                expected_normalized = (
                    working.astype(np.float32) - float(working.min())
                ) / float(int(working.max()) - int(working.min()))
                np.testing.assert_array_equal(model_input[0, 0], expected_normalized)
                self.assertGreater(np.unique(model_input).size, 256)
                self.assertEqual(working.tobytes(), working_before)

                self.assertEqual(Path(frame.current_file).read_bytes(), source_before)
                self.assertEqual(frame.raw_image.tobytes(), raw_before)
                self.assertEqual(frame._source_raw_image.tobytes(), preserved_source_before)
                self.assertFalse(frame.raw_image.flags.writeable)
                self.assertIs(frame.image_canvas.image, frame.processed_image)
                self.assertTrue(frame._processed_image_saved)
                self.assertEqual(frame._sdb_queue_progress(), (1, 1))
                frame._on_processed_image.assert_called_once()
                np.testing.assert_array_equal(frame._on_processed_image.call_args.args[0], saved[0])
                self.assertEqual(frame.notifications, ["Saved successfully"])
                if repair:
                    prompt.assert_called_once()
                else:
                    prompt.assert_not_called()

    def test_declining_crop_saves_all_original_signed_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            signed = _signed_crop()
            frame = self._crop_frame(folder, signed=signed)
            source_before = Path(frame.current_file).read_bytes()

            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=False) as prompt, patch(
                f"{STEP1}.smooth_bright_pixels"
            ) as smooth:
                self.assertTrue(frame._crop_and_scale())

            expected = np.repeat(signed, 3, axis=1)
            np.testing.assert_array_equal(read_analyze(folder / "light"), np.stack([expected] * 2))
            expected_bytes = np.stack([np.flipud(expected)] * 2).astype(">i2").tobytes()
            self.assertEqual((folder / "light.img").read_bytes(), expected_bytes)
            self.assertEqual(Path(frame.current_file).read_bytes(), source_before)
            prompt.assert_called_once()
            smooth.assert_not_called()

    def test_canceling_crop_preserves_preview_and_does_not_save(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame = self._crop_frame(folder)
            frame.image_canvas.set_image(frame.raw_image)
            source_before = Path(frame.current_file).read_bytes()

            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=None) as prompt, patch(
                f"{STEP1}.write_analyze"
            ) as save:
                self.assertFalse(frame._crop_and_scale())

            self.assertIsNone(frame.processed_image)
            self.assertIs(frame.image_canvas.image, frame.raw_image)
            self.assertTrue(frame.image_canvas.roi_enabled)
            self.assertEqual(frame.view_mode_var.get(), "source")
            self.assertFalse(frame._processed_image_saved)
            self.assertEqual(frame._sdb_queue_progress(), (0, 1))
            self.assertEqual(frame.notifications, [])
            self.assertFalse((folder / "light.hdr").exists())
            self.assertFalse((folder / "light.img").exists())
            self.assertEqual(Path(frame.current_file).read_bytes(), source_before)
            frame._on_processed_image.assert_not_called()
            prompt.assert_called_once()
            save.assert_not_called()

    def test_detection_uses_signed_native_resolution_before_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = self._crop_frame(Path(directory))
            expected = _signed_crop()
            with patch(
                f"{STEP1}.detect_bright_pixels_in_bands", wraps=detect_bright_pixels_in_bands
            ) as detect, patch(
                f"{STEP1}.Step1Frame._ask_bright_pixel_action", return_value=False
            ):
                self.assertTrue(frame._crop_and_scale())

            detect.assert_called_once()
            detected_crop = detect.call_args.args[0]
            self.assertEqual(detected_crop.dtype, np.dtype("int16"))
            self.assertEqual(detected_crop.shape, expected.shape)
            np.testing.assert_array_equal(detected_crop, expected)
            self.assertFalse(np.shares_memory(detected_crop, frame.raw_image))

    def test_artifacts_outside_the_crop_do_not_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            signed = _signed_crop()
            frame = self._crop_frame(folder, signed=signed, roi=(0, 0, 64, 12))
            with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action") as prompt:
                self.assertTrue(frame._crop_and_scale())

            expected = np.repeat(signed[:12], 3, axis=1)
            np.testing.assert_array_equal(read_analyze(folder / "light"), np.stack([expected] * 2))
            prompt.assert_not_called()

    def test_legacy_target_builder_preserves_original_pixels_without_prompt(self):
        frame = Step1Frame.__new__(Step1Frame)
        signed = _signed_crop()
        frame.raw_image = signed.astype(np.uint16)
        frame._source_roi = (0, 0, 64, 64)
        with patch(f"{STEP1}.Step1Frame._ask_bright_pixel_action") as prompt:
            result = frame._build_processed_target_image()

        np.testing.assert_array_equal(result, np.repeat(signed, 3, axis=1))
        self.assertEqual(result.dtype, np.dtype("int16"))
        self.assertTrue(result.flags.c_contiguous)
        prompt.assert_not_called()

    def test_prepared_target_builder_uses_the_cleaned_crop(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.raw_image = _signed_crop().astype(np.uint16)
        frame._source_roi = (0, 0, 64, 64)
        prepared = _signed_crop()
        prepared[20, 31] = -10900
        prepared[40, 17] = -9650
        before = prepared.tobytes()

        result = frame._build_processed_target_image(cropped=prepared)

        np.testing.assert_array_equal(result, np.repeat(prepared, 3, axis=1))
        self.assertEqual(prepared.tobytes(), before)
        self.assertEqual(result.dtype, np.dtype("int16"))
        self.assertTrue(result.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
