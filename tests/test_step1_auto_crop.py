"""Crop proposals preserve the manual raw view and explicit save workflow."""

from __future__ import annotations

import tempfile
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.utils.io_utils import read_analyze, read_raw_oct
from aidas.utils.oct_crop import CropSuggestion
from sdb_fixture import build_cmdb, ir_chunk, oct_chunk
from test_oct_crop import curved_retina


STEP1 = "aidas.steps.step1_resize_raw"


class _Variable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Canvas:
    def __init__(self, owner):
        self.owner = owner
        self.image = None
        self.roi = None
        self.roi_enabled = False

    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def enable_roi(self, enabled):
        self.roi_enabled = enabled

    def set_roi(self, roi):
        self.roi = tuple(roi)
        self.owner._on_roi_changed(self.roi)


class Step1AutoCropTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / "manual.sdb"
        self.source.write_bytes(b"source remains unchanged")

    def _frame(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.raw_image = None
        frame.processed_image = None
        frame._source_raw_image = None
        frame._working_source_raw_image = None
        frame._source_roi = None
        frame._crop_oct_layout = None
        frame._source_crop_oct_layout = None
        frame._crop_contours = None
        frame._source_crop_contours = None
        frame._bright_pixel_notice_pending = False
        frame._syncing_view_roi = False
        frame._target_size_edit_active = False
        frame._updating_roi_entries = False
        frame._updating_target_size_entries = False
        frame._on_processed_image = None
        frame._sdb_image_info = None
        frame._cropped_sdb_files = set()
        frame._saved_crop_previews = {}
        frame._saved_output_directories = set()
        frame._sdb_directory_files = {}
        frame._sdb_directories = []
        frame.default_raw_width = 32
        frame.default_raw_height = 60
        frame.default_raw_offset = 0
        frame.default_little_endian = True
        frame.raw_import_params = None
        frame.current_file = None
        frame.image_canvas = _Canvas(frame)
        frame.view_mode_var = _Variable("source")
        frame.roi_x_var = _Variable("0")
        frame.roi_y_var = _Variable("0")
        frame.roi_w_var = _Variable("1")
        frame.roi_h_var = _Variable("1")
        frame.target_w_var = _Variable("3")
        frame.target_h_var = _Variable("1")
        frame.width_var = _Variable("32")
        frame.bright_pixel_status_var = _Variable("")
        frame.crop_region_summary_var = _Variable("")
        for name in (
            "_update_zoom_label", "_update_image_status", "_update_processing_controls",
            "_update_batch_handoff_button_state", "_update_queue_controls",
            "_refresh_sdb_progress_colors", "_render_sdb_directories",
            "_update_import_settings_summary",
        ):
            setattr(frame, name, lambda: None)
        frame._set_sdb_parameters_enabled = lambda _enabled: None
        frame._set_roi_controls_enabled = lambda _enabled: None
        frame._show_crop_notification = lambda *_args, **_kwargs: None
        return frame

    def _native_view(self):
        full = np.full((60, 32), 64000, dtype=np.uint16)
        native = (np.arange(24 * 32).reshape(24, 32) * 19).astype(np.uint16)
        start = 13 * 32 + 5
        full.ravel()[start:start + native.size] = native.ravel()
        return full, native, (start, 32, 24)

    def test_real_sdb_without_edb_draws_local_retinal_box_on_unchanged_full_view(self):
        native, top, bottom = curved_retina()
        rows = np.arange(native.shape[0], dtype=float)[:, None]
        # Remote, coherent intensity changes must not enlarge the retinal box.
        native = np.rint(native + 900 * np.exp(-0.5 * ((rows - 28) / 2.0) ** 2)).astype(np.uint16)
        self.source.write_bytes(build_cmdb([
            ir_chunk(8, 8), oct_chunk(native.shape[1], native.shape[0], native),
        ]))
        before = self.source.read_bytes()
        frame = self._frame()
        frame.default_raw_height = 600
        with patch.object(frame, "_ask_bright_pixel_action", return_value=False):
            self.assertTrue(frame._open_raw(str(self.source)))

        info = frame._sdb_image_info
        self.assertEqual(info.edb_status, "SDB header")
        self.assertIsNone(frame._crop_contours)
        self.assertEqual(frame.raw_image.shape, (600, native.shape[1]))
        manual = read_raw_oct(self.source, **frame.raw_import_params)
        np.testing.assert_array_equal(frame.raw_image, manual)
        np.testing.assert_array_equal(frame.image_canvas.get_image(), manual)
        pixel_start, width, _height = frame._crop_oct_layout
        columns = np.arange(width)
        first_rows = (pixel_start + np.floor(top).astype(int) * width + columns) // width
        last_rows = (pixel_start + np.ceil(bottom).astype(int) * width + columns) // width
        x, y, crop_width, crop_height = frame._source_roi
        self.assertEqual((x, crop_width), (0, width))
        self.assertLessEqual(y, first_rows.min())
        self.assertGreater(y + crop_height, last_rows.max())
        self.assertGreaterEqual(y, first_rows.min() - 45)
        self.assertLessEqual(y + crop_height, last_rows.max() + 45)
        self.assertIn("image content", frame.crop_region_summary_var.get())
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse((self.folder / "light.img").exists())
        self.assertFalse((self.folder / "light.hdr").exists())

    def test_native_detection_uses_only_oct_words_and_keeps_entire_canvas(self):
        frame = self._frame()
        full, native, layout = self._native_view()
        full.setflags(write=False)
        before = full.tobytes()
        contours = np.vstack((np.full(32, 5.0), np.full(32, 18.0)))
        suggestion = CropSuggestion((24, 6, 6, 7), "EDB contours", True)

        with patch(f"{STEP1}.detect_oct_crop", return_value=suggestion) as detector:
            self.assertTrue(frame._load_image(
                full, str(self.source), check_bright_pixels=False,
                crop_oct_layout=layout, crop_contours=contours,
            ))

        detector.assert_called_once()
        np.testing.assert_array_equal(detector.call_args.args[0], native)
        np.testing.assert_array_equal(detector.call_args.kwargs["contours"], contours)
        np.testing.assert_array_equal(frame.image_canvas.get_image(), full)
        self.assertEqual(frame.image_canvas.get_image().shape, (60, 32))
        np.testing.assert_array_equal(frame.raw_image, full)
        np.testing.assert_array_equal(frame._source_raw_image, full)
        np.testing.assert_array_equal(frame._working_source_raw_image, full)
        self.assertEqual(full.tobytes(), before)
        # A box crossing the raw row edge must cover both halves of every layer.
        self.assertEqual(frame._source_roi, (0, 19, 32, 8))
        self.assertEqual(frame.image_canvas.roi, frame._source_roi)
        self.assertIn("Proposed OCT box", frame.crop_region_summary_var.get())
        self.assertIn("EDB contours", frame.crop_region_summary_var.get())
        self.assertFalse((self.folder / "light.img").exists())
        self.assertFalse((self.folder / "light.hdr").exists())

    def test_contour_hints_detach_from_caller_and_manual_drag_stays_selected(self):
        frame = self._frame()
        full, _native, layout = self._native_view()
        contours = np.full((2, 32), 11.0)
        with patch(f"{STEP1}.detect_oct_crop", return_value=CropSuggestion((2, 6, 10, 7), "EDB contours", True)) as detector:
            frame._load_image(full, str(self.source), check_bright_pixels=False, crop_oct_layout=layout, crop_contours=contours)
            contours[:] = 0
            frame._on_roi_changed((3, 17, 19, 20))

        detector.assert_called_once()
        self.assertEqual(frame._source_roi, (3, 17, 19, 20))
        self.assertIn("Manual crop", frame.crop_region_summary_var.get())
        self.assertTrue(np.all(frame._source_crop_contours == 11))

    def test_cancelling_source_load_preserves_previous_box_pixels_and_hints(self):
        frame = self._frame()
        full, _native, layout = self._native_view()
        previous = full.copy()
        frame.raw_image = previous
        frame._source_roi = (3, 11, 20, 9)
        frame._crop_oct_layout = layout
        frame._crop_contours = np.ones((2, 32))
        old_contours = frame._crop_contours
        frame.crop_region_summary_var.set("Manual crop")
        with patch.object(frame, "_offer_bright_pixel_removal", return_value=None), patch(f"{STEP1}.detect_oct_crop") as detector:
            self.assertFalse(frame._load_image(np.zeros((40, 32), dtype=np.uint16), "other.sdb"))

        detector.assert_not_called()
        self.assertIs(frame.raw_image, previous)
        self.assertIs(frame._crop_contours, old_contours)
        self.assertEqual(frame._crop_oct_layout, layout)
        self.assertEqual(frame._source_roi, (3, 11, 20, 9))
        self.assertEqual(frame.crop_region_summary_var.get(), "Manual crop")

    def test_uncertain_legacy_image_selects_full_frame_instead_of_last_row(self):
        frame = self._frame()
        frame.raw_image = np.zeros((496, 32), dtype=np.uint16)
        with patch(f"{STEP1}.detect_oct_crop", return_value=CropSuggestion((0, 0, 32, 496), "full image", False)):
            self.assertFalse(frame._set_initial_roi())
        self.assertEqual(frame._source_roi, (0, 0, 32, 496))
        self.assertIn("uncertain", frame.crop_region_summary_var.get())

    def test_uncertain_native_layers_keep_native_extent_inside_full_view(self):
        frame = self._frame()
        full, _native, layout = self._native_view()
        with patch(f"{STEP1}.detect_oct_crop", return_value=CropSuggestion((0, 0, 32, 24), "full image", False)):
            frame._load_image(full, str(self.source), check_bright_pixels=False, crop_oct_layout=layout)
        self.assertEqual(frame._source_roi, (0, 13, 32, 25))
        self.assertIn("OCT region selected", frame.crop_region_summary_var.get())
        np.testing.assert_array_equal(frame.image_canvas.get_image(), full)

    def test_invalid_native_layout_never_uses_infrared_for_detection(self):
        full, _native, layout = self._native_view()
        for invalid in ((), (-1, 32, 24), (layout[0], 31, 24), (layout[0], 32, 100)):
            with self.subTest(layout=invalid):
                frame = self._frame()
                with patch(f"{STEP1}.detect_oct_crop") as detector:
                    frame._load_image(full, str(self.source), check_bright_pixels=False, crop_oct_layout=invalid)
                detector.assert_not_called()
                self.assertEqual(frame._source_roi, (0, 0, 32, 60))

    def test_native_width_edit_falls_back_and_original_width_restores_hints(self):
        frame = self._frame()
        full, native, layout = self._native_view()
        contours = np.full((2, 32), 11.0)
        with patch(f"{STEP1}.detect_oct_crop", return_value=CropSuggestion((2, 6, 10, 7), "EDB contours", True)) as detector:
            frame._load_image(full, str(self.source), check_bright_pixels=False, crop_oct_layout=layout, crop_contours=contours)
            frame.width_var.set("28")
            frame._apply_width_preview_adjustment()
            self.assertEqual(detector.call_count, 1)
            self.assertEqual(frame._source_roi, (0, 0, 28, 60))
            np.testing.assert_array_equal(frame.raw_image, full[:, 2:30])
            self.assertIsNone(frame._crop_contours)
            frame.width_var.set("32")
            frame._apply_width_preview_adjustment()

        self.assertEqual(detector.call_count, 2)
        np.testing.assert_array_equal(detector.call_args.args[0], native)
        np.testing.assert_array_equal(frame.raw_image, full)
        self.assertEqual(frame._crop_oct_layout, layout)
        np.testing.assert_array_equal(frame._crop_contours, contours)

    def test_legacy_width_edit_analyzes_the_adjusted_view(self):
        frame = self._frame()
        full = np.arange(60 * 32, dtype=np.uint16).reshape(60, 32)
        with patch(f"{STEP1}.detect_oct_crop", return_value=CropSuggestion((2, 6, 10, 7), "image content", True)) as detector:
            frame._load_image(full, str(self.source), check_bright_pixels=False)
            frame.width_var.set("28")
            frame._apply_width_preview_adjustment()
        np.testing.assert_array_equal(detector.call_args.args[0], full[:, 2:30])
        self.assertEqual(frame._source_roi, (2, 6, 10, 7))

    def test_saved_preview_does_not_detect_and_clears_source_hints(self):
        frame = self._frame()
        full, _native, layout = self._native_view()
        frame._crop_oct_layout = frame._source_crop_oct_layout = layout
        frame._crop_contours = frame._source_crop_contours = np.ones((2, 32))
        saved = full[:12].astype(np.int16)
        with patch.object(frame, "_offer_bright_pixel_removal", side_effect=lambda image, **_kwargs: image), patch(f"{STEP1}.detect_oct_crop") as detector:
            self.assertTrue(frame._load_saved_crop(saved, str(self.source)))
        detector.assert_not_called()
        self.assertIsNone(frame.raw_image)
        self.assertIsNone(frame._source_roi)
        self.assertIsNone(frame._crop_contours)
        self.assertIsNone(frame._source_crop_contours)
        self.assertIsNone(frame._crop_oct_layout)
        self.assertIsNone(frame._source_crop_oct_layout)
        np.testing.assert_array_equal(frame.image_canvas.get_image(), saved)
        self.assertIn("Saved crop", frame.crop_region_summary_var.get())

    def test_undo_raw_view_reuses_original_oct_layout_without_changing_pixels(self):
        frame = self._frame()
        full, native, layout = self._native_view()
        with patch(f"{STEP1}.detect_oct_crop", return_value=CropSuggestion((2, 6, 10, 7), "image content", True)) as detector:
            frame._load_image(full, str(self.source), check_bright_pixels=False, crop_oct_layout=layout)
            frame.processed_image = np.zeros((5, 20), dtype=np.int16)
            frame._restore_raw_view()
        self.assertEqual(detector.call_count, 2)
        np.testing.assert_array_equal(detector.call_args.args[0], native)
        np.testing.assert_array_equal(frame.raw_image, full)
        np.testing.assert_array_equal(frame.image_canvas.get_image(), full)
        self.assertIsNone(frame.processed_image)
        self.assertTrue(frame.image_canvas.roi_enabled)

    def test_explicit_crop_saves_manual_words_only_after_user_action(self):
        frame = self._frame()
        full = (np.arange(60 * 32).reshape(60, 32) * 29 + 30000).astype(np.uint16)
        source_before = self.source.read_bytes()
        full_before = full.tobytes()
        with patch(f"{STEP1}.detect_oct_crop", return_value=CropSuggestion((4, 12, 11, 9), "image content", True)):
            frame._load_image(full, str(self.source), check_bright_pixels=False)
        self.assertFalse((self.folder / "light.img").exists())
        frame._directory_is_complete = lambda _directory: True
        with patch.object(frame, "_offer_bright_pixel_removal", side_effect=lambda image, **_kwargs: image):
            self.assertTrue(frame._crop_and_scale())
        saved = read_analyze(self.folder / "light")
        expected = np.repeat(full[12:21, 4:15].astype(np.int16), 3, axis=1)
        self.assertEqual(saved.dtype, np.dtype("int16"))
        self.assertEqual(saved.shape, (2, 9, 33))
        np.testing.assert_array_equal(saved[0], expected)
        np.testing.assert_array_equal(saved[1], expected)
        self.assertEqual(full.tobytes(), full_before)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertIn("Saved crop", frame.crop_region_summary_var.get())

    def test_detection_failure_keeps_full_view_available_for_manual_crop(self):
        frame = self._frame()
        full, _native, layout = self._native_view()
        with patch(f"{STEP1}.detect_oct_crop", side_effect=ValueError("unusable contours")):
            self.assertTrue(frame._load_image(full, str(self.source), check_bright_pixels=False, crop_oct_layout=layout))
        self.assertEqual(frame._source_roi, (0, 0, 32, 60))
        np.testing.assert_array_equal(frame.image_canvas.get_image(), full)
        self.assertIn("uncertain", frame.crop_region_summary_var.get())

    def test_unmappable_native_box_falls_back_without_blocking_import(self):
        frame = self._frame()
        full, _native, layout = self._native_view()
        with patch(f"{STEP1}.detect_oct_crop", return_value=CropSuggestion((2, 6, 10, 7), "image content", True)), patch(f"{STEP1}.map_oct_crop_to_raw", return_value=None):
            self.assertTrue(frame._load_image(full, str(self.source), check_bright_pixels=False, crop_oct_layout=layout))
        self.assertEqual(frame._source_roi, (0, 0, 32, 60))
        np.testing.assert_array_equal(frame.image_canvas.get_image(), full)
        self.assertIn("uncertain", frame.crop_region_summary_var.get())

    def test_explicit_legacy_preset_remains_available_after_auto_proposal(self):
        frame = self._frame()
        frame.raw_image = np.zeros((1200, 32), dtype=np.uint16)
        with patch(f"{STEP1}.detect_oct_crop", return_value=CropSuggestion((2, 101, 24, 90), "image content", True)):
            frame._set_initial_roi()
        frame._set_default_roi()
        self.assertEqual(frame._source_roi, (0, 585, 32, 128))
        self.assertIn("Default region", frame.crop_region_summary_var.get())


class Step1AutoCropCanvasTests(unittest.TestCase):
    def test_real_canvas_keeps_full_view_and_editable_proposal_with_visible_summary(self):
        # CTk retains process-wide DPI timers, so isolate this application's
        # real window from other suites that also construct and close roots.
        code = textwrap.dedent("""
            import tkinter as tk
            from unittest.mock import patch
            import customtkinter as ctk
            import numpy as np
            from aidas.steps.step1_resize_raw import Step1Frame
            from aidas.utils.oct_crop import CropSuggestion
            root = None
            try:
                root = ctk.CTk()
                root.withdraw()
                root.attributes('-alpha', 0)
                root.geometry('1280x820+10000+10000')
                root.deiconify()
            except tk.TclError as error:
                if root is not None:
                    root.destroy()
                print(error)
                raise SystemExit(77)
            try:
                callback_errors = []
                root.report_callback_exception = lambda *error: callback_errors.append(error)
                with patch.object(Step1Frame, 'refresh_sdb_list'):
                    frame = Step1Frame(root)
                frame.pack(fill='both', expand=True)
                full = np.arange(1200 * 768, dtype=np.uint16).reshape(1200, 768)
                layout = (480 * 768, 768, 496)
                suggestion = CropSuggestion((0, 125, 768, 150), 'image content', True)
                with patch('aidas.steps.step1_resize_raw.detect_oct_crop', return_value=suggestion) as detector:
                    assert frame._load_image(full, 'manual.sdb', check_bright_pixels=False, crop_oct_layout=layout)
                    root.update()
                    assert frame.image_canvas.get_roi() == (0, 605, 768, 150)
                    np.testing.assert_array_equal(frame.image_canvas.get_image(), full)
                    assert frame.crop_region_summary_label.winfo_ismapped()
                    assert not frame.sdb_params_section.body.winfo_ismapped()
                    assert frame.crop_options_menu.entrycget(0, 'label') == 'Auto OCT Region'
                    assert frame.crop_options_menu.index('end') == 1
                    assert frame.crop_options_menu.entrycget(1, 'label') == 'Whole Image'
                    frame.image_canvas.set_roi((25, 590, 700, 175))
                    assert frame._source_roi == (25, 590, 700, 175)
                    assert 'Manual crop' in frame.crop_region_summary_var.get()
                    frame.crop_options_menu.invoke(0)
                    assert frame._source_roi == (0, 605, 768, 150)
                assert detector.call_count == 2
                assert not callback_errors, callback_errors
            finally:
                root.destroy()
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 77:
            self.skipTest("An invisible Tk window is unavailable: " + result.stdout.strip())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
