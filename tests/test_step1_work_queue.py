from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.steps.step2_annotate import Step2Frame
from aidas.utils.io_utils import read_analyze, write_analyze


class _ListboxStub:
    def __init__(self):
        self.items = []
        self.selection = ()
        self.configured_rows = []

    def delete(self, _first, _last):
        self.items.clear()
        self.selection = ()

    def insert(self, _where, value):
        self.items.append(value)

    def itemconfigure(self, index, **_options):
        self.configured_rows.append((index, _options))

    def selection_set(self, index):
        self.selection = (index,)

    def selection_clear(self, _first, _last):
        self.selection = ()

    def see(self, _index):
        pass

    def curselection(self):
        return self.selection


class _StringVarStub:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class _ButtonStub:
    def __init__(self):
        self.state = None

    def configure(self, **options):
        if "state" in options:
            self.state = options["state"]


class _CropCanvasStub:
    def set_image(self, image):
        self.image = image

    def enable_roi(self, enabled):
        self.roi_enabled = enabled


class Step1WorkQueueTests(unittest.TestCase):
    def _queue_frame(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame._sdb_directories = ["folder-a", "folder-b", "folder-c"]
        frame._sdb_directory_files = {
            "folder-a": ["folder-a/first.sdb", "folder-a/second.sdb"],
            "folder-b": ["folder-b/third.sdb"],
            "folder-c": ["folder-c/fourth.sdb"],
        }
        frame._cropped_sdb_files = set()
        frame._saved_crop_previews = {}
        frame._saved_output_directories = set()
        frame.current_file = "folder-a/first.sdb"
        frame.processed_image = None
        frame._processed_image_saved = False
        return frame

    def _crop_frame(self, folder):
        frame = self._queue_frame()
        source = folder / "source.sdb"
        raw = np.arange(48, dtype=np.uint16).reshape(6, 8)
        source.write_bytes(raw.tobytes())
        frame.current_file = str(source)
        frame.raw_image = raw
        frame._source_raw_image = raw.copy()
        frame._source_roi = (1, 2, 3, 2)
        frame._sdb_directory_files = {str(folder): [str(source)]}
        frame._sdb_directories = [str(folder)]
        frame.raw_import_params = {"width": 8, "height": 6, "offset": 0}
        frame._apply_import_params = lambda **_kwargs: True
        frame.image_canvas = _CropCanvasStub()
        frame.view_mode_var = _StringVarStub()
        frame._update_zoom_label = lambda: None
        frame._update_image_status = lambda: None
        frame._update_processing_controls = lambda: None
        frame._set_sdb_parameters_enabled = lambda _enabled: None
        frame._set_roi_controls_enabled = lambda _enabled: None
        frame._set_default_roi = lambda: None
        frame._refresh_sdb_progress_colors = lambda: None
        frame._render_sdb_directories = lambda: None
        frame._on_processed_image = None
        frame.notifications = []
        frame._show_crop_notification = lambda message, **_kwargs: frame.notifications.append(message)
        return frame

    def test_cropping_automatically_saves_pixels_and_keeps_the_crop_visible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            frame = self._crop_frame(folder)
            source = frame.current_file
            expected = np.repeat(frame.raw_image[2:4, 1:4], 3, axis=1).astype(np.int16)

            self.assertTrue(frame._crop_and_scale())

            saved = read_analyze(folder / "light")
            np.testing.assert_array_equal(saved, np.stack([expected, expected]))
            np.testing.assert_array_equal(frame.image_canvas.image, expected)
            self.assertEqual(frame.current_file, source)
            self.assertTrue(frame._processed_image_saved)
            self.assertEqual(frame._sdb_queue_progress(), (1, 1))
            self.assertTrue(frame._all_sdb_images_saved())
            self.assertEqual(frame.notifications, ["Saved successfully"])

    def test_failed_automatic_save_does_not_report_success_or_complete_the_crop(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            frame = self._crop_frame(Path(temporary_directory))
            with patch(
                "aidas.steps.step1_resize_raw.write_analyze", side_effect=OSError("disk full")
            ), patch("aidas.steps.step1_resize_raw.messagebox.showerror") as error:
                self.assertFalse(frame._crop_and_scale())

            error.assert_called_once()
            self.assertIsNotNone(frame.processed_image)
            self.assertFalse(frame._processed_image_saved)
            self.assertEqual(frame._sdb_queue_progress(), (0, 1))
            self.assertFalse(frame._all_sdb_images_saved())
            self.assertEqual(frame.notifications, [])

    def test_next_from_a_saved_crop_displays_the_next_original_sdb(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            frame = self._crop_frame(folder)
            first = frame.current_file
            second = folder / "second.sdb"
            original_next = np.full((6, 8), 777, dtype=np.uint16)
            second.write_bytes(original_next.tobytes())
            frame._sdb_directory_files[str(folder)].append(str(second))
            self.assertTrue(frame._crop_and_scale())
            saved_crop = frame.processed_image.copy()

            frame._next_sdb()

            self.assertEqual(frame.current_file, str(second))
            self.assertIsNone(frame.processed_image)
            self.assertEqual(frame.view_mode_var.get(), "source")
            self.assertTrue(frame.image_canvas.roi_enabled)
            np.testing.assert_array_equal(frame.image_canvas.image, original_next)
            self.assertEqual(frame._sdb_queue_progress(), (1, 2))
            np.testing.assert_array_equal(read_analyze(folder / "light")[0], saved_crop)
            np.testing.assert_array_equal(frame._saved_crop_for_path(first), saved_crop)

    def test_selecting_an_uncropped_sdb_does_not_reuse_another_images_light_pair(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            frame = self._crop_frame(folder)
            second = folder / "second.sdb"
            original_next = np.full((6, 8), 777, dtype=np.uint16)
            second.write_bytes(original_next.tobytes())
            frame._sdb_directory_files[str(folder)].append(str(second))
            self.assertTrue(frame._crop_and_scale())

            frame._open_raw(path=str(second))

            self.assertEqual(frame.current_file, str(second))
            self.assertIsNone(frame.processed_image)
            np.testing.assert_array_equal(frame.image_canvas.image, original_next)
            self.assertTrue(frame.image_canvas.roi_enabled)

    def test_undo_deletes_only_light_outputs_and_removes_completion(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            frame = self._crop_frame(folder)
            raw = frame.raw_image.copy()
            self.assertTrue(frame._crop_and_scale())
            (folder / "Light_MARKED.img").write_bytes(b"keep marked output")
            (folder / "notes.txt").write_text("keep notes")

            self.assertTrue(frame._reset())

            self.assertFalse((folder / "light.hdr").exists())
            self.assertFalse((folder / "light.img").exists())
            self.assertTrue((folder / "source.sdb").exists())
            self.assertEqual((folder / "Light_MARKED.img").read_bytes(), b"keep marked output")
            self.assertEqual((folder / "notes.txt").read_text(), "keep notes")
            np.testing.assert_array_equal(frame.image_canvas.image, raw)
            self.assertTrue(frame.image_canvas.roi_enabled)
            self.assertIsNone(frame.processed_image)
            self.assertEqual(frame._saved_crop_previews, {})
            self.assertEqual(frame._sdb_queue_progress(), (0, 1))
            self.assertFalse(frame._all_sdb_images_saved())
            self.assertEqual(frame.notifications, ["Saved successfully", "Saved crop removed"])

    def test_undo_a_reopened_saved_crop_reads_the_original_before_deleting(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            frame = self._crop_frame(folder)
            raw = frame.raw_image.copy()
            self.assertTrue(frame._crop_and_scale())
            frame.raw_image = None
            frame._source_raw_image = None

            self.assertTrue(frame._reset())

            np.testing.assert_array_equal(frame.raw_image, raw)
            np.testing.assert_array_equal(frame.image_canvas.image, raw)
            self.assertFalse((folder / "light.hdr").exists())
            self.assertFalse((folder / "light.img").exists())
            self.assertEqual(frame.notifications[-1], "Saved crop removed")

    def test_undo_preserves_saved_outputs_if_the_original_cannot_be_read(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            frame = self._crop_frame(folder)
            self.assertTrue(frame._crop_and_scale())
            crop = frame.processed_image
            frame.raw_image = None
            with patch(
                "aidas.steps.step1_resize_raw.read_raw_oct", side_effect=OSError("unavailable source")
            ), patch("aidas.steps.step1_resize_raw.messagebox.showerror") as error:
                self.assertFalse(frame._reset())

            error.assert_called_once()
            self.assertTrue((folder / "light.hdr").exists())
            self.assertTrue((folder / "light.img").exists())
            self.assertIs(frame.processed_image, crop)
            self.assertTrue(frame._processed_image_saved)
            self.assertEqual(frame.notifications, ["Saved successfully"])

    def test_undo_reports_deletion_failure_without_claiming_the_crop_was_removed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            frame = self._crop_frame(folder)
            self.assertTrue(frame._crop_and_scale())
            crop = frame.processed_image
            with patch(
                "aidas.steps.step1_resize_raw.os.remove", side_effect=PermissionError("locked output")
            ), patch("aidas.steps.step1_resize_raw.messagebox.showerror") as error:
                self.assertFalse(frame._reset())

            error.assert_called_once()
            self.assertIs(frame.processed_image, crop)
            self.assertTrue(frame._processed_image_saved)
            self.assertTrue((folder / "light.hdr").exists())
            self.assertTrue((folder / "light.img").exists())
            self.assertEqual(frame.notifications, ["Saved successfully"])

    def test_partial_undo_deletion_marks_the_crop_incomplete_and_preserves_its_preview(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            frame = self._crop_frame(folder)
            self.assertTrue(frame._crop_and_scale())
            crop = frame.processed_image
            remove_file = os.remove
            removed = []

            def fail_after_first_delete(path):
                if removed:
                    raise PermissionError("second output is locked")
                remove_file(path)
                removed.append(path)

            with patch(
                "aidas.steps.step1_resize_raw.os.remove", side_effect=fail_after_first_delete
            ), patch("aidas.steps.step1_resize_raw.messagebox.showerror") as error:
                self.assertFalse(frame._reset())

            error.assert_called_once()
            self.assertEqual(len(removed), 1)
            self.assertIs(frame.processed_image, crop)
            self.assertFalse(frame._processed_image_saved)
            self.assertEqual(frame._sdb_queue_progress(), (0, 1))
            self.assertFalse(frame._all_sdb_images_saved())
            self.assertEqual(frame.notifications, ["Saved successfully"])

    def test_saved_output_detection_requires_hdr_and_img_case_insensitively(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            (folder / "LIGHT.HDR").touch()

            self.assertFalse(Step1Frame._folder_has_all_saved_outputs(folder))

            (folder / "light.img").touch()
            self.assertTrue(Step1Frame._folder_has_all_saved_outputs(folder))

    def test_populating_a_new_folder_selects_its_first_image(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.sdb_listbox = _ListboxStub()
        frame._cropped_sdb_files = set()
        frame.current_file = None
        frame._sdb_directory_files = {
            "folder": ["folder/first.sdb", "folder/second.sdb"],
        }

        frame._populate_sdb_files("folder", select_first=True)

        self.assertEqual(frame.sdb_listbox.selection, (0,))
        self.assertEqual(frame._sdb_files[0], "folder/first.sdb")

    def test_preview_opens_the_selected_first_image(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.sdb_listbox = _ListboxStub()
        frame.sdb_listbox.selection_set(0)
        frame._sdb_files = ["folder/first.sdb"]
        frame.current_file = None
        opened = []

        def open_raw(path=None):
            opened.append(path)
            frame.current_file = path

        frame._open_raw = open_raw

        self.assertTrue(frame._preview_selected_sdb())
        self.assertEqual(opened, ["folder/first.sdb"])

    def test_next_queue_item_continues_with_next_folder(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame._sdb_directories = ["folder-a", "folder-b"]
        frame._sdb_directory_files = {
            "folder-a": ["folder-a/first.sdb", "folder-a/second.sdb"],
            "folder-b": ["folder-b/third.sdb"],
        }

        self.assertEqual(
            frame._next_sdb_queue_item("folder-a/second.sdb"),
            ("folder-b", 0, "folder-b/third.sdb"),
        )

    def test_next_queue_item_skips_saved_images_and_complete_folders(self):
        frame = self._queue_frame()
        frame._cropped_sdb_files.add(frame._path_key("folder-a/second.sdb"))
        frame._saved_output_directories.add(frame._path_key("folder-b"))

        self.assertEqual(
            frame._next_sdb_queue_item("folder-a/first.sdb"),
            ("folder-c", 0, "folder-c/fourth.sdb"),
        )

    def test_previous_queue_item_skips_saved_images_and_complete_folders(self):
        frame = self._queue_frame()
        frame._cropped_sdb_files.add(frame._path_key("folder-a/second.sdb"))
        frame._saved_output_directories.add(frame._path_key("folder-b"))

        self.assertEqual(
            frame._previous_sdb_queue_item("folder-c/fourth.sdb"),
            ("folder-a", 0, "folder-a/first.sdb"),
        )

    def test_queue_navigation_does_not_wrap_at_either_end(self):
        frame = self._queue_frame()

        self.assertIsNone(frame._previous_sdb_queue_item("folder-a/first.sdb"))
        self.assertIsNone(frame._next_sdb_queue_item("folder-c/fourth.sdb"))

        frame._cropped_sdb_files = {
            frame._path_key(path)
            for files in frame._sdb_directory_files.values()
            for path in files
        }
        self.assertIsNone(frame._previous_sdb_queue_item("folder-c/fourth.sdb"))
        self.assertIsNone(frame._next_sdb_queue_item("folder-a/first.sdb"))

    def test_queue_progress_includes_folders_hidden_by_search(self):
        frame = self._queue_frame()
        frame._sdb_directories = ["folder-a"]
        frame._cropped_sdb_files.add(frame._path_key("folder-a/first.sdb"))
        frame._saved_output_directories.add(frame._path_key("folder-b"))

        self.assertEqual(frame._sdb_queue_progress(), (2, 4))
        self.assertEqual(
            frame._next_sdb_queue_item("folder-a/second.sdb"),
            ("folder-c", 0, "folder-c/fourth.sdb"),
        )

    def test_queue_controls_show_aggregate_progress_and_available_neighbors(self):
        frame = self._queue_frame()
        frame._cropped_sdb_files.add(frame._path_key("folder-a/second.sdb"))
        frame._saved_output_directories.add(frame._path_key("folder-b"))
        frame.queue_progress_var = _StringVarStub()
        frame.back_sdb_btn = _ButtonStub()
        frame.next_sdb_btn = _ButtonStub()

        frame._update_queue_controls()

        self.assertEqual(frame.queue_progress_var.value, "(2/4)")
        self.assertEqual(frame.back_sdb_btn.state, "disabled")
        self.assertEqual(frame.next_sdb_btn.state, "normal")

        frame.current_file = "folder-c/fourth.sdb"
        frame._update_queue_controls()

        self.assertEqual(frame.back_sdb_btn.state, "normal")
        self.assertEqual(frame.next_sdb_btn.state, "disabled")

    def test_empty_queue_disables_navigation_and_step2(self):
        frame = self._queue_frame()
        frame._sdb_directory_files = {}
        frame._sdb_directories = []
        frame.current_file = None
        frame.queue_progress_var = _StringVarStub()
        frame.back_sdb_btn = _ButtonStub()
        frame.next_sdb_btn = _ButtonStub()
        frame.batch_segment_cropped_btn = _ButtonStub()
        frame._on_batch_segment_folders = lambda _folders: None

        frame._update_queue_controls()
        frame._update_batch_handoff_button_state()

        self.assertEqual(frame.queue_progress_var.value, "(0/0)")
        self.assertEqual(frame.back_sdb_btn.state, "disabled")
        self.assertEqual(frame.next_sdb_btn.state, "disabled")
        self.assertEqual(frame.batch_segment_cropped_btn.state, "disabled")

    def test_navigation_preserves_crop_when_discard_is_cancelled(self):
        for command, current_file in (
            ("_next_sdb", "folder-a/first.sdb"),
            ("_prev_sdb", "folder-c/fourth.sdb"),
        ):
            with self.subTest(command=command):
                frame = self._queue_frame()
                crop = object()
                frame.current_file = current_file
                frame.processed_image = crop
                frame.raw_import_params = {}
                confirmed = []

                def decline_discard(_next_path):
                    confirmed.append(frame.processed_image)
                    return False

                frame._confirm_discard_processed_image = decline_discard

                with patch("aidas.steps.step1_resize_raw.read_raw_oct") as read:
                    getattr(frame, command)()

                read.assert_not_called()
                self.assertEqual(confirmed, [crop])
                self.assertIs(frame.processed_image, crop)
                self.assertEqual(frame.current_file, current_file)

    def test_saved_crop_navigation_does_not_prompt_to_discard(self):
        frame = self._queue_frame()
        frame.processed_image = object()
        frame._processed_image_saved = True

        with patch("aidas.steps.step1_resize_raw.messagebox.askyesno") as prompt:
            self.assertTrue(frame._confirm_discard_processed_image("folder-a/second.sdb"))

        prompt.assert_not_called()

    def test_unsaved_crop_navigation_respects_the_discard_prompt(self):
        frame = self._queue_frame()
        frame.processed_image = object()

        for response in (False, True):
            with self.subTest(response=response):
                with patch(
                    "aidas.steps.step1_resize_raw.messagebox.askyesno",
                    return_value=response,
                ) as prompt:
                    self.assertIs(
                        frame._confirm_discard_processed_image("folder-a/second.sdb"),
                        response,
                    )

                prompt.assert_called_once()

    def test_save_writes_analyze_pair_and_keeps_the_current_crop_visible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            first = str(folder / "first.sdb")
            second = str(folder / "second.sdb")
            frame = Step1Frame.__new__(Step1Frame)
            frame.processed_image = np.zeros((4, 5), dtype=np.int16)
            frame.current_file = first
            frame._sdb_directories = [str(folder)]
            frame._sdb_directory_files = {str(folder): [first, second]}
            frame._cropped_sdb_files = set()
            frame._saved_output_directories = set()
            frame.status_var = _StringVarStub()
            frame._source_output_directory = lambda: str(folder)
            frame._refresh_sdb_progress_colors = lambda: None
            frame._update_batch_handoff_button_state = lambda: None
            opened = []
            frame._open_queued_sdb = lambda directory, index, path: opened.append(
                (directory, index, path)
            )

            with patch(
                "aidas.steps.step1_resize_raw.write_analyze",
                return_value=(str(folder / "light.hdr"), str(folder / "light.img")),
            ) as write:
                self.assertTrue(frame._save_processed_image())

            write.assert_called_once()
            self.assertEqual(opened, [])
            self.assertEqual(frame.current_file, first)
            self.assertTrue(frame._processed_image_saved)
            self.assertIn(frame._path_key(first), frame._cropped_sdb_files)

    def test_failed_save_does_not_complete_or_advance_the_image(self):
        frame = self._queue_frame()
        crop = np.zeros((4, 5), dtype=np.int16)
        frame.processed_image = crop
        frame._source_output_directory = lambda: "folder-a"
        frame._open_queued_sdb = lambda *_args: self.fail(
            "Failed saves must not advance the work queue"
        )

        with patch(
            "aidas.steps.step1_resize_raw.write_analyze",
            side_effect=OSError("disk full"),
        ), patch("aidas.steps.step1_resize_raw.messagebox.showerror") as error:
            self.assertFalse(frame._save_processed_image())

        error.assert_called_once()
        self.assertEqual(frame._sdb_queue_progress(), (0, 4))
        self.assertFalse(frame._all_sdb_images_saved())
        self.assertFalse(frame._processed_image_saved)
        self.assertIs(frame.processed_image, crop)
        self.assertEqual(frame.current_file, "folder-a/first.sdb")

    def test_automatic_save_does_not_attempt_to_open_the_next_image(self):
        frame = self._queue_frame()
        crop = np.zeros((4, 5), dtype=np.int16)
        frame.processed_image = crop
        frame.raw_import_params = {}
        frame.sdb_listbox = _ListboxStub()
        frame._source_output_directory = lambda: "folder-a"
        frame._apply_import_params = lambda **_kwargs: True
        frame._refresh_sdb_progress_colors = lambda: None
        frame._render_sdb_directories = lambda: None

        with patch("aidas.steps.step1_resize_raw.write_analyze") as write, patch(
            "aidas.steps.step1_resize_raw.read_raw_oct",
            side_effect=OSError("next source is unavailable"),
        ) as read, patch("aidas.steps.step1_resize_raw.messagebox.showerror") as error:
            self.assertTrue(frame._save_processed_image())

        write.assert_called_once()
        read.assert_not_called()
        error.assert_not_called()
        self.assertEqual(frame._sdb_queue_progress(), (1, 4))
        self.assertIs(frame.processed_image, crop)
        self.assertTrue(frame._processed_image_saved)
        self.assertEqual(frame.current_file, "folder-a/first.sdb")

    def test_step2_requires_every_image_and_its_saved_output_folder(self):
        frame = self._queue_frame()
        frame.batch_segment_cropped_btn = _ButtonStub()
        frame._on_batch_segment_folders = lambda _folders: None
        frame._saved_output_directories.add(frame._path_key("folder-a"))

        frame._update_batch_handoff_button_state()
        self.assertEqual(frame.batch_segment_cropped_btn.state, "disabled")

        frame._cropped_sdb_files.update({
            frame._path_key("folder-b/third.sdb"),
            frame._path_key("folder-c/fourth.sdb"),
        })
        frame._update_batch_handoff_button_state()
        self.assertEqual(frame.batch_segment_cropped_btn.state, "disabled")

        frame._saved_output_directories.update({
            frame._path_key("folder-b"),
            frame._path_key("folder-c"),
        })
        frame._update_batch_handoff_button_state()
        self.assertEqual(frame.batch_segment_cropped_btn.state, "normal")

        frame.processed_image = object()
        frame._processed_image_saved = False
        frame._update_batch_handoff_button_state()
        self.assertEqual(frame.batch_segment_cropped_btn.state, "disabled")

        frame._processed_image_saved = True
        frame._update_batch_handoff_button_state()
        self.assertEqual(frame.batch_segment_cropped_btn.state, "normal")

    def test_step2_handoff_rejects_partial_work_even_when_called_directly(self):
        frame = self._queue_frame()
        frame._saved_output_directories.add(frame._path_key("folder-a"))
        handed_off = []
        frame._on_batch_segment_folders = lambda folders: handed_off.append(folders)

        with patch("aidas.steps.step1_resize_raw.messagebox.showwarning") as warning:
            frame._send_cropped_folders_to_step2()

        warning.assert_called_once()
        self.assertEqual(handed_off, [])

    def test_step2_handoff_includes_all_folders_when_every_image_is_saved(self):
        frame = self._queue_frame()
        frame._sdb_directories = ["folder-a"]
        frame._saved_output_directories = {
            frame._path_key(folder) for folder in frame._sdb_directory_files
        }
        handed_off = []
        frame._on_batch_segment_folders = lambda folders: handed_off.append(folders)

        frame._send_cropped_folders_to_step2()

        self.assertEqual(handed_off, [["folder-a", "folder-b", "folder-c"]])

    def test_completed_image_opens_its_saved_crop_without_reading_the_sdb(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            crop = np.arange(20, dtype=np.int16).reshape(4, 5)
            write_analyze(folder / "light", np.stack([crop, crop]))
            path = str(folder / "source.sdb")
            frame = self._queue_frame()
            opened = []
            frame._load_saved_crop = lambda image, source: opened.append((image, source))
            frame._apply_import_params = lambda **_kwargs: self.fail(
                "Saved previews must not require raw import parameters"
            )

            with patch("aidas.steps.step1_resize_raw.read_raw_oct") as read_raw:
                frame._open_raw(path=path)

            read_raw.assert_not_called()
            self.assertEqual(len(opened), 1)
            np.testing.assert_array_equal(opened[0][0], crop)
            self.assertEqual(opened[0][1], path)

    def test_each_saved_sdb_retains_its_own_crop_preview_in_the_same_folder(self):
        frame = self._queue_frame()
        frame._source_output_directory = lambda: "folder-a"
        frame._refresh_sdb_progress_colors = lambda: None
        frame._update_batch_handoff_button_state = lambda: None
        frame._open_queued_sdb = lambda *_args: None
        first_crop = np.full((3, 4), 100, dtype=np.int16)
        second_crop = np.full((2, 6), 200, dtype=np.int16)

        with patch("aidas.steps.step1_resize_raw.write_analyze"):
            frame.processed_image = first_crop
            self.assertTrue(frame._save_processed_image())
            frame.current_file = "folder-a/second.sdb"
            frame.processed_image = second_crop
            self.assertTrue(frame._save_processed_image())

        with patch("aidas.steps.step1_resize_raw.read_analyze") as read_saved:
            first_preview = frame._saved_crop_for_path("folder-a/first.sdb")
            second_preview = frame._saved_crop_for_path("folder-a/second.sdb")

        read_saved.assert_not_called()
        np.testing.assert_array_equal(first_preview, first_crop)
        np.testing.assert_array_equal(second_preview, second_crop)
        first_preview[:] = 0
        np.testing.assert_array_equal(
            frame._saved_crop_for_path("folder-a/first.sdb"), first_crop
        )

    def test_unreadable_saved_crop_preserves_the_current_preview(self):
        frame = self._queue_frame()
        crop = np.ones((2, 3), dtype=np.int16)
        frame.processed_image = crop
        frame._processed_image_saved = True
        frame._saved_output_directories.add(frame._path_key("folder-b"))

        with patch(
            "aidas.steps.step1_resize_raw.read_analyze",
            side_effect=ValueError("Invalid Analyze header"),
        ), patch("aidas.steps.step1_resize_raw.read_raw_oct") as read_raw, patch(
            "aidas.steps.step1_resize_raw.messagebox.showerror"
        ) as error:
            frame._open_raw(path="folder-b/third.sdb")

        read_raw.assert_not_called()
        error.assert_called_once()
        self.assertIs(frame.processed_image, crop)
        self.assertEqual(frame.current_file, "folder-a/first.sdb")

    def test_default_roi_spans_the_full_image_width(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.raw_image = np.zeros((1200, 768), dtype=np.uint16)
        captured = []
        frame._set_roi_and_entries = lambda *roi: captured.append(roi)

        frame._set_default_roi()

        self.assertEqual(captured, [(0, 585, 768, 128)])

    def test_crop_progress_repaint_does_not_rebuild_or_preview_the_queue(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.sdb_listbox = _ListboxStub()
        frame.sdb_directory_listbox = _ListboxStub()
        frame._sdb_files = ["folder/first.sdb"]
        frame._cropped_sdb_files = {frame._path_key("folder/first.sdb")}
        frame._active_sdb_directory = None
        frame._sdb_directories = []
        frame._render_sdb_directories = lambda: self.fail(
            "Crop progress must not rebuild the directory list"
        )

        frame._refresh_sdb_progress_colors()

        self.assertEqual(frame.sdb_listbox.configured_rows[0][0], 0)

    def test_programmatic_directory_selection_does_not_open_an_image(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame._sdb_directory_selection_locked = True
        frame.sdb_directory_listbox = _ListboxStub()
        frame.sdb_directory_listbox.selection_set(0)
        frame._preview_selected_sdb = lambda: self.fail(
            "Programmatic directory selection must not preview an image"
        )

        frame._on_sdb_directory_select()

    def test_selected_folder_summary_counts_cropped_and_uncropped_images(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame._active_sdb_directory = "folder"
        frame._sdb_files = [
            "folder/first.sdb",
            "folder/second.sdb",
            "folder/third.sdb",
        ]
        frame._cropped_sdb_files = {frame._path_key("folder/first.sdb")}
        frame._saved_output_directories = set()
        frame.cropped_image_count_var = _StringVarStub()
        frame.uncropped_image_count_var = _StringVarStub()

        frame._update_sdb_image_count_summary()

        self.assertEqual(frame.cropped_image_count_var.value, "Cropped 1")
        self.assertEqual(frame.uncropped_image_count_var.value, "Uncropped 2")

    def test_folder_header_summary_counts_cropped_and_uncropped_folders(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame._sdb_directory_files = {
            "complete": ["complete/first.sdb"],
            "pending": ["pending/first.sdb"],
        }
        frame._saved_output_directories = {frame._path_key("complete")}
        frame._cropped_sdb_files = set()
        frame.cropped_folder_count_var = _StringVarStub()
        frame.uncropped_folder_count_var = _StringVarStub()

        frame._update_sdb_folder_count_summary()

        self.assertEqual(frame.cropped_folder_count_var.value, "Cropped 1")
        self.assertEqual(frame.uncropped_folder_count_var.value, "Uncropped 1")

    def test_cropped_result_disables_source_and_keeps_target_enabled(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.raw_image = object()
        frame.processed_image = object()
        frame.undo_crop_btn = _ButtonStub()
        frame.crop_btn = _ButtonStub()
        frame.crop_options_btn = _ButtonStub()
        frame.source_view_radio = _ButtonStub()
        frame.target_view_radio = _ButtonStub()

        frame._update_processing_controls()

        self.assertEqual(frame.source_view_radio.state, "disabled")
        self.assertEqual(frame.target_view_radio.state, "normal")
        self.assertEqual(frame.crop_options_btn.state, "disabled")

    def test_crop_actions_stay_disabled_without_a_source_image(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.raw_image = None
        frame.processed_image = None
        frame.undo_crop_btn = _ButtonStub()
        frame.crop_btn = _ButtonStub()
        frame.crop_options_btn = _ButtonStub()
        frame.source_view_radio = _ButtonStub()
        frame.target_view_radio = _ButtonStub()

        frame._update_processing_controls()

        self.assertEqual(frame.crop_btn.state, "disabled")
        self.assertEqual(frame.crop_options_btn.state, "disabled")

    def test_step2_handoff_builds_one_batch_row_per_cropped_folder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "model.onnx"
            model.touch()
            ready = root / "ready"
            already_marked = root / "done"
            ready.mkdir()
            already_marked.mkdir()
            (ready / "Light.img").touch()
            (ready / "Light.hdr").touch()
            (already_marked / "Light.img").touch()
            (already_marked / "Light_MARKED.img").touch()

            class Step2Stub:
                _segmenter_running = False
                aidas_model_path = str(model)
                status_var = _StringVarStub()
                _preferred_analyze_pair_path = staticmethod(
                    Step2Frame._preferred_analyze_pair_path
                )

                def _open_step2_batch_segmentation_panel(
                    self,
                    root_dir,
                    *,
                    initial_rows=None,
                ):
                    self.opened = (initial_rows, root_dir)

            step2 = Step2Stub()
            Step2Frame.start_batch_segmentation_for_folders(
                step2,
                [ready, already_marked],
            )

            rows, batch_root = step2.opened
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {Path(row["folder"]) for row in rows},
                {ready, already_marked},
            )
            self.assertEqual(Path(rows[0]["image_paths"][0]), ready / "Light.img")
            self.assertEqual(Path(batch_root), root)

    def test_step2_light_result_tab_uses_its_folder_name(self):
        title = Step2Frame._batch_result_input_title(
            Path("study") / "subject-12" / "Light.img"
        )

        self.assertEqual(title, "subject-12")


if __name__ == "__main__":
    unittest.main()
