from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.steps.step2_annotate import Step2Frame


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


class _ButtonStub:
    def __init__(self):
        self.state = None

    def configure(self, **options):
        if "state" in options:
            self.state = options["state"]


class Step1WorkQueueTests(unittest.TestCase):
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

    def test_save_writes_analyze_pair_and_opens_next_sdb(self):
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
                self.assertTrue(frame._save_analyze_and_advance())

            write.assert_called_once()
            self.assertEqual(opened, [(str(folder), 1, second)])
            self.assertIn(frame._path_key(first), frame._cropped_sdb_files)

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
        frame.save_all_btn = _ButtonStub()
        frame.undo_crop_btn = _ButtonStub()
        frame.crop_btn = _ButtonStub()
        frame.crop_options_btn = _ButtonStub()
        frame.source_view_radio = _ButtonStub()
        frame.target_view_radio = _ButtonStub()

        frame._update_save_button_state()

        self.assertEqual(frame.source_view_radio.state, "disabled")
        self.assertEqual(frame.target_view_radio.state, "normal")
        self.assertEqual(frame.crop_options_btn.state, "disabled")

    def test_crop_actions_stay_disabled_without_a_source_image(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame.raw_image = None
        frame.processed_image = None
        frame.save_all_btn = _ButtonStub()
        frame.undo_crop_btn = _ButtonStub()
        frame.crop_btn = _ButtonStub()
        frame.crop_options_btn = _ButtonStub()
        frame.source_view_radio = _ButtonStub()
        frame.target_view_radio = _ButtonStub()

        frame._update_save_button_state()

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

                def _start_step2_batch_segmentation_from_rows(self, rows, root_dir):
                    self.started = (rows, root_dir)

            step2 = Step2Stub()
            Step2Frame.start_batch_segmentation_for_folders(
                step2,
                [ready, already_marked],
            )

            rows, batch_root = step2.started
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
