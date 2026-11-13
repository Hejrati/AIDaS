"""Deleting derived files must allow the original SDB to be processed again."""

import queue
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.utils.io_utils import read_analyze
from tests.steps.step1 import test_step1_sdb_auto_import as auto_import
from tests.steps.step1.test_step1_work_queue import _ListboxStub


class Step1DeletedOutputsTests(unittest.TestCase):
    setUp = auto_import.Step1SdbAutoImportTests.setUp
    _frame = auto_import.Step1SdbAutoImportTests._frame
    _native_source = auto_import.Step1SdbAutoImportTests._native_source

    def _saved_native(self, *, name="native"):
        source, _words = self._native_source(name, edb=None)
        original = source.read_bytes()
        frame = self._frame()
        frame._sdb_directory_files = {str(self.folder): [str(source)]}
        frame._sdb_directories = [str(self.folder)]
        self.assertTrue(frame._open_raw(str(source)))
        self.assertTrue(frame._crop_and_scale())
        return frame, source, original

    def _delete_outputs(self, *names):
        for name in names or ("light.hdr", "light.img"):
            (self.folder / name).unlink()

    def _select_current(self, frame, source):
        frame.sdb_listbox = _ListboxStub()
        frame.sdb_listbox.selection_set(0)
        frame._sdb_files = [str(source)]
        return frame._preview_selected_sdb()

    def test_native_crop_delete_every_output_and_repeat_in_same_and_fresh_sessions(self):
        frame, source, original = self._saved_native()
        expected = {name: (self.folder / name).read_bytes() for name in ("light.hdr", "light.img")}
        for fresh_session in (False, False, True):
            with self.subTest(fresh_session=fresh_session):
                self._delete_outputs()
                self.assertEqual(list(self.folder.iterdir()), [source])
                if fresh_session:
                    frame = self._frame()
                with patch("aidas.steps.step1_resize_raw.messagebox.askyesno") as confirm:
                    self.assertTrue(frame._open_raw(str(source)))
                confirm.assert_not_called()
                self.assertIsNone(frame.processed_image)
                self.assertFalse(frame._processed_image_saved)
                self.assertTrue(frame._crop_and_scale())
                for name, content in expected.items():
                    self.assertEqual((self.folder / name).read_bytes(), content)
                self.assertEqual(source.read_bytes(), original)
        self.errors.assert_not_called()

    def test_reselect_same_current_image_recovers_from_either_missing_analyze_file(self):
        frame, source, original = self._saved_native()
        expected = read_analyze(self.folder / "light")
        for missing in ("light.hdr", "light.img"):
            with self.subTest(missing=missing):
                self._delete_outputs(missing)
                self.assertTrue(self._select_current(frame, source))
                self.assertIsNone(frame.processed_image)
                self.assertIsNotNone(frame.raw_image)
                self.assertTrue(frame.image_canvas.roi_enabled)
                self.assertEqual(frame._sdb_queue_progress(), (0, 1))
                self.assertFalse(frame._all_sdb_images_saved())
                self.assertTrue(frame._crop_and_scale())
                np.testing.assert_array_equal(read_analyze(self.folder / "light"), expected)
                self.assertEqual(source.read_bytes(), original)

    def test_deleted_saved_only_preview_reloads_native_source_before_recropping(self):
        frame, source, original = self._saved_native()
        self.assertTrue(frame._open_raw(str(source)))
        self.assertIsNone(frame.raw_image)
        self.assertTrue(frame._sdb_showing_saved_crop)
        self._delete_outputs()
        self.assertTrue(self._select_current(frame, source))
        self.assertIsNone(frame.processed_image)
        self.assertFalse(frame._sdb_showing_saved_crop)
        self.assertTrue(frame._crop_and_scale())
        self.assertEqual(source.read_bytes(), original)

    def test_replaced_source_after_output_deletion_is_read_again(self):
        frame, source, _original = self._saved_native()
        self._delete_outputs()
        source, words = self._native_source("native", width=5, height=7, edb=None)
        replacement = source.read_bytes()
        self.assertTrue(self._select_current(frame, source))
        self.assertEqual((frame._sdb_image_info.width, frame._sdb_image_info.height), (5, 7))
        start = (frame._sdb_image_info.offset - frame.raw_import_params["offset"]) // 2
        np.testing.assert_array_equal(
            frame.raw_image.ravel()[start:start + words.size].reshape(words.shape), words
        )
        self.assertTrue(frame._crop_and_scale())
        self.assertEqual(source.read_bytes(), replacement)

    def test_refresh_reconciles_cached_completion_and_reopens_current_without_preview_option(self):
        frame, source, _original = self._saved_native()
        self._delete_outputs()
        frame.sdb_dir_var = SimpleNamespace(get=lambda: str(self.folder))
        frame.sdb_scan_more_label = Mock()
        frame.sdb_scan_tooltip = SimpleNamespace(text="")
        frame._active_sdb_directory = str(self.folder)
        events = queue.Queue()
        cancelled = threading.Event()
        events.put(Step1Frame._scan_sdb_directory(str(self.folder), cancelled))
        with patch.object(Step1Frame, "_folder_has_all_saved_outputs") as rescan:
            frame._poll_sdb_scan(events, cancelled, str(self.folder), str(self.folder), False)
        rescan.assert_not_called()
        self.assertEqual(frame._saved_crop_previews, {})
        self.assertEqual(frame._cropped_sdb_files, set())
        self.assertEqual(frame._sdb_queue_progress(), (0, 1))
        self.assertIsNone(frame.processed_image)
        self.assertIsNotNone(frame.raw_image)
        self.assertEqual(list(self.folder.iterdir()), [source])

    def test_handoff_rechecks_all_saved_folders_and_recovers_current_deleted_crop(self):
        frame, source, _original = self._saved_native()
        sent = []
        frame._on_batch_segment_folders = lambda folders: sent.append(folders)
        self.assertTrue(frame._all_sdb_images_saved())
        self._delete_outputs("light.img")
        frame._send_cropped_folders_to_step2()
        self.assertEqual(sent, [])
        self.assertIsNone(frame.processed_image)
        self.assertFalse(frame._all_sdb_images_saved())
        self.assertFalse((self.folder / "light.img").exists())
        self.assertTrue(frame._crop_and_scale())
        frame._send_cropped_folders_to_step2()
        self.assertEqual(sent, [[str(self.folder)]])

    def test_all_cached_sources_sharing_deleted_pair_are_invalidated(self):
        first, _words = self._native_source("first", edb=None)
        second, _words = self._native_source("second", edb=None)
        frame = self._frame()
        frame._sdb_directory_files = {str(self.folder): [str(first), str(second)]}
        frame._sdb_directories = [str(self.folder)]
        self.assertTrue(frame._open_raw(str(first)))
        self.assertTrue(frame._crop_and_scale())
        first_crop = frame.processed_image.copy()
        self.assertTrue(frame._open_raw(str(second)))
        self.assertTrue(frame._crop_and_scale())
        np.testing.assert_array_equal(frame._saved_crop_for_path(first), first_crop)
        unrelated = self.folder / "unrelated" / "image.sdb"
        unrelated_key = frame._path_key(unrelated)
        frame._cropped_sdb_files.add(unrelated_key)
        frame._saved_crop_previews[unrelated_key] = first_crop.copy()
        self._delete_outputs()
        self.assertIsNone(frame._saved_crop_for_path(first))
        self.assertEqual(frame._cropped_sdb_files, {unrelated_key})
        self.assertEqual(set(frame._saved_crop_previews), {unrelated_key})
        self.assertFalse(frame._directory_is_complete(str(self.folder)))
        self.assertTrue(frame._open_raw(str(second)))
        self.assertTrue(frame._crop_and_scale())
        self.assertIsNone(frame._saved_crop_for_path(first))
        self.assertEqual(frame._sdb_queue_progress(), (1, 2))

    def test_unsaved_manual_crop_is_preserved_on_reselection_and_refresh(self):
        frame, source, _original = self._saved_native()
        self._delete_outputs()
        frame._processed_image_saved = False
        unsaved = frame.processed_image
        self.assertTrue(self._select_current(frame, source))
        self.assertIs(frame.processed_image, unsaved)
        self.assertIsNone(frame._revalidate_saved_crop_directories(saved_directories=set()))
        self.assertIs(frame.processed_image, unsaved)

    def test_only_deleted_known_crops_invalidate_workflow_completion_badges(self):
        frame, source, _original = self._saved_native()
        with patch("aidas.steps.step1_resize_raw.notify_workflow_invalidated") as invalidate:
            self.assertIsNotNone(frame._saved_crop_for_path(source))
            invalidate.assert_not_called()
            self._delete_outputs()
            self.assertIsNone(frame._saved_crop_for_path(source))
            invalidate.assert_called_once_with(frame, 1)
        frame = self._frame()
        with patch("aidas.steps.step1_resize_raw.notify_workflow_invalidated") as invalidate:
            self.assertIsNone(frame._saved_crop_for_path(source))
            invalidate.assert_not_called()

    def test_returning_to_step1_reopens_deleted_saved_crop_and_clears_badges(self):
        frame, source, original = self._saved_native()
        self._delete_outputs()
        with patch("aidas.steps.step1_resize_raw.notify_workflow_invalidated") as invalidate:
            frame.on_show()
        invalidate.assert_called_once_with(frame, 1)
        self.assertIsNone(frame.processed_image)
        self.assertIsNotNone(frame.raw_image)
        self.assertTrue(frame.image_canvas.roi_enabled)
        self.assertEqual(frame._sdb_queue_progress(), (0, 1))
        self.assertEqual(list(self.folder.iterdir()), [source])
        self.assertEqual(source.read_bytes(), original)

    def test_returning_also_revalidates_direct_open_outside_folder_queue(self):
        frame, source, _original = self._saved_native()
        frame._sdb_directory_files = {}
        frame._sdb_directories = []
        self._delete_outputs()
        frame.on_show()
        self.assertIsNone(frame.processed_image)
        self.assertIsNotNone(frame.raw_image)
        self.assertEqual(frame._saved_crop_previews, {})
        self.assertEqual(list(self.folder.iterdir()), [source])

    def test_returning_preserves_unsaved_edits_and_pending_bright_pixel_notice(self):
        frame, source, _original = self._saved_native()
        self._delete_outputs()
        frame._bright_pixel_notice_pending = True
        with patch.object(frame, "_revalidate_saved_crop_directories") as revalidate:
            frame.on_show()
        revalidate.assert_not_called()
        frame._bright_pixel_notice_pending = False
        frame._processed_image_saved = False
        unsaved = frame.processed_image
        with patch.object(frame, "_open_raw") as reopen:
            frame.on_show()
        reopen.assert_not_called()
        self.assertIs(frame.processed_image, unsaved)
        self.assertEqual(list(self.folder.iterdir()), [source])

    def test_previous_navigation_rechecks_deleted_outputs_in_other_folder(self):
        frame, source, _original = self._saved_native()
        other_folder = self.folder / "other"
        other_folder.mkdir()
        other_source = other_folder / "other.sdb"
        other_source.write_bytes(source.read_bytes())
        frame._sdb_directory_files[str(other_folder)] = [str(other_source)]
        frame._sdb_directories.append(str(other_folder))
        self.assertTrue(frame._open_raw(str(other_source)))
        self.assertTrue(frame._crop_and_scale())
        self._delete_outputs()
        frame._prev_sdb()
        self.assertEqual(frame.current_file, str(source))
        self.assertIsNone(frame.processed_image)
        self.assertIsNotNone(frame.raw_image)
        self.assertEqual(frame._sdb_queue_progress(), (1, 2))

    def test_real_window_refresh_restores_crop_controls_and_repeat_outputs(self):
        code = textwrap.dedent("""
            from pathlib import Path
            import tempfile
            import time
            from unittest.mock import Mock, patch
            import customtkinter as ctk
            import numpy as np
            from aidas.steps.step1_resize_raw import Step1Frame
            from tests.support.sdb_fixture import build_cmdb, ir_chunk, oct_chunk

            root = ctk.CTk()
            root.withdraw()
            errors = []
            root.report_callback_exception = lambda *error: errors.append(error)
            preferences = Mock()
            settings = {'sdb_raw_width': 8, 'sdb_raw_height': 6, 'sdb_raw_offset': 0}
            preferences.get.side_effect = lambda key, default=None: settings.get(key, default)
            try:
                with tempfile.TemporaryDirectory() as temporary, \
                        patch.object(Step1Frame, '_ask_bright_pixel_action', return_value=False), \
                        patch.object(Step1Frame, '_show_crop_notification'), \
                        patch('aidas.steps.step1_resize_raw.messagebox.showerror') as popup:
                    folder = Path(temporary)
                    source = folder / 'native.sdb'
                    words = np.arange(48, dtype=np.uint16).reshape(6, 8)
                    original = build_cmdb([ir_chunk(3, 4), oct_chunk(8, 6, words)])
                    source.write_bytes(original)
                    with patch.object(Step1Frame, 'refresh_sdb_list'):
                        frame = Step1Frame(root, preferences=preferences)
                    frame.pack(fill='both', expand=True)
                    frame.set_sdb_directory(str(folder))

                    def finish_scan():
                        deadline = time.monotonic() + 10
                        while frame._sdb_scan_pending and time.monotonic() < deadline:
                            root.update()
                            time.sleep(.01)
                        root.update()
                        assert not frame._sdb_scan_pending

                    frame.refresh_sdb_list(preview_first=True)
                    finish_scan()
                    assert frame.current_file == str(source)
                    assert frame.crop_btn.cget('state') == 'normal'
                    frame.crop_btn.invoke()
                    root.update()
                    expected = {name: (folder / name).read_bytes() for name in ('light.hdr', 'light.img')}
                    assert frame.undo_crop_btn.cget('state') == 'normal'
                    for name in expected:
                        (folder / name).unlink()
                    frame.refresh_sdb_list(preview_first=False)
                    finish_scan()
                    assert frame.processed_image is None
                    assert frame.crop_btn.cget('state') == 'normal'
                    assert frame.undo_crop_btn.cget('state') == 'disabled'
                    assert frame.queue_progress_var.get() == '(0/1)'
                    assert list(folder.iterdir()) == [source]
                    frame.crop_btn.invoke()
                    root.update()
                    assert frame.queue_progress_var.get() == '(1/1)'
                    for name, contents in expected.items():
                        assert (folder / name).read_bytes() == contents
                    assert source.read_bytes() == original
                    popup.assert_not_called()
                    assert not errors, errors
                    frame.destroy()
            finally:
                root.destroy()
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=45,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
