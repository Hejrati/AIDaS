from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aidas.steps.step3_flatten import Step3Frame


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _write_files(folder, *names):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"test result")
    return folder.resolve()


def _write_flat_pair(folder):
    return _write_files(folder, "_flat_LIGHT.hdr", "_flat_LIGHT.img")


def _write_png(folder):
    return _write_files(folder, "DARK_MARKED_find_vertex.png")


def _write_array_exports(folder):
    arrays = Path(folder) / Step3Frame.R_ARRAY_EXPORT_DIR
    _write_files(
        arrays,
        *(f"{name}.{extension}" for name in (
            "FLATTENED_DARK_RETINA_RRC",
            "FLATTENED_LIGHT_RETINA_RRC",
            "FLATTENED_MARKERS_RRC",
        ) for extension in ("bin", "shape")),
    )
    return Path(folder).resolve()


def _frame():
    frame = Step3Frame.__new__(Step3Frame)
    frame.current_sdb_dir = "previous input"
    frame.output_sdb_dir = "previous output"
    frame.results = {"previous": object()}
    frame.step4_result_folders = []
    frame.status_var = _Var("previous status")
    frame.info_var = _Var("previous info")
    frame.progress_text_var = _Var("previous progress")
    frame._busy = False
    frame.continue_to_step4_button = None
    frame.on_continue_to_step4 = mock.Mock()
    frame._update_continue_to_step4_button_state = mock.Mock()
    frame._open_batch_r_result_tabs = mock.Mock()
    frame._load_r_results_from_folder = mock.Mock(return_value=False)
    frame.winfo_toplevel = mock.Mock(return_value=None)
    return frame


class Step3ResultFolderRecognitionTests(unittest.TestCase):
    def test_recognizes_each_supported_result_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            folders = [
                _write_png(root / "vertex"),
                _write_files(root / "tissue", "_tissueBorders__LIGHT.png"),
                _write_files(root / "workspace", *Step3Frame.R_WORKSPACE_FILES),
                _write_flat_pair(root / "analyze"),
                _write_array_exports(root / "arrays"),
            ]

            for folder in folders:
                with self.subTest(format=folder.name):
                    self.assertTrue(Step3Frame._folder_has_r_results(folder))

    def test_unrelated_or_incomplete_outputs_do_not_qualify(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            partial_arrays = _write_array_exports(root / "partial arrays")
            (partial_arrays / Step3Frame.R_ARRAY_EXPORT_DIR /
             "FLATTENED_MARKERS_RRC.shape").unlink()
            folders = [
                _write_files(root / "unrelated", "other.RData", "other.png"),
                _write_files(root / "workspace", Step3Frame.R_WORKSPACE_FILES[0]),
                _write_files(root / "analyze", "_flat_LIGHT.hdr"),
                partial_arrays,
            ]
            fake_files = root / "directories"
            for name in Step3Frame.R_WORKSPACE_FILES:
                (fake_files / name).mkdir(parents=True)
            folders.append(fake_files)

            for folder in folders:
                with self.subTest(folder=folder.name):
                    self.assertFalse(Step3Frame._folder_has_r_results(folder))

    def test_array_export_child_is_not_a_second_result_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _write_array_exports(Path(temp_dir) / "subject")

            self.assertFalse(Step3Frame._folder_has_r_results(
                folder / Step3Frame.R_ARRAY_EXPORT_DIR
            ))
            discovered, errors = Step3Frame._find_r_result_folders(folder)

            self.assertEqual(discovered, [folder])
            self.assertEqual(errors, [])


class Step3RecursiveResultDiscoveryTests(unittest.TestCase):
    def test_discovers_nested_subjects_in_stable_recursive_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            second = _write_png(root / "Z subject" / "nasal")
            first = _write_flat_pair(root / "a subject" / "visit 1" / "temporal")
            third = _write_array_exports(root / "Z subject" / "temporal")
            _write_files(root / "not results", "notes.RData")

            discovered, errors = Step3Frame._find_r_result_folders(root)

            self.assertEqual(discovered, [first, second, third])
            self.assertEqual(len(discovered), len(set(discovered)))
            self.assertEqual(errors, [])

    def test_result_bearing_parent_does_not_hide_descendant_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _write_png(Path(temp_dir) / "subject")
            nasal = _write_flat_pair(root / "nasal")
            temporal = _write_png(root / "temporal")
            deeper = _write_png(temporal / "repeat")

            discovered, errors = Step3Frame._find_r_result_folders(root)

            self.assertEqual(discovered, [root, nasal, temporal, deeper])
            self.assertEqual(errors, [])

    def test_denied_branch_is_reported_and_readable_sibling_still_loads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            blocked = root / "a blocked"
            blocked.mkdir()
            readable = _write_png(root / "b readable" / "subject")
            original_iterdir = Path.iterdir

            def guarded_iterdir(path):
                if path == blocked:
                    raise PermissionError("test denied branch")
                return original_iterdir(path)

            with mock.patch.object(Path, "iterdir", guarded_iterdir):
                discovered, errors = Step3Frame._find_r_result_folders(root)

            self.assertEqual(discovered, [readable])
            self.assertTrue(errors)
            self.assertIn("a blocked", str(errors))

    def test_grouping_preserves_parent_output_alongside_its_side_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = _write_png(Path(temp_dir) / "subject")
            temporal = _write_png(parent / "temporal")
            nasal = _write_png(parent / "nasal")

            groups = Step3Frame._group_result_folders([parent, temporal, nasal])

            self.assertEqual(len(groups), 2)
            own_group = next(group for group in groups if not group["sided"])
            sides_group = next(group for group in groups if group["sided"])
            self.assertEqual(own_group["folders"]["result"], parent)
            self.assertEqual(sides_group["folders"]["temporal"], temporal)
            self.assertEqual(sides_group["folders"]["nasal"], nasal)


class Step3RecursiveResultLoadingTests(unittest.TestCase):
    def test_single_result_uses_existing_full_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            folder = _write_flat_pair(root / "subject" / "visit")
            frame = _frame()
            frame._load_r_results_from_folder.return_value = True

            self.assertTrue(frame._load_r_results_from_tree(root))

            frame._load_r_results_from_folder.assert_called_once_with(
                folder, show_errors=False
            )
            frame._open_batch_r_result_tabs.assert_not_called()
            self.assertEqual(frame.step4_result_folders, [folder])
            frame._update_continue_to_step4_button_state.assert_called()

    def test_png_only_result_opens_preview_when_full_loader_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            folder = _write_png(root / "subject" / "temporal")
            frame = _frame()

            with mock.patch("aidas.steps.step3_flatten.messagebox.showerror") as error:
                self.assertTrue(frame._load_r_results_from_tree(root))

            frame._load_r_results_from_folder.assert_called_once_with(
                folder, show_errors=False
            )
            frame._open_batch_r_result_tabs.assert_called_once_with([folder])
            self.assertEqual(frame.step4_result_folders, [])
            error.assert_not_called()

    def test_multiple_results_open_all_and_handoff_every_complete_flat_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            parent = _write_flat_pair(root / "subject A")
            png_only = _write_png(parent / "nasal")
            temporal = _write_flat_pair(parent / "temporal")
            nested = _write_flat_pair(root / "subject B" / "visit" / "nasal")
            _write_files(root / "partial", "_flat_LIGHT.hdr")
            frame = _frame()

            self.assertTrue(frame._load_r_results_from_tree(root))
            frame._continue_to_step4()

            frame._load_r_results_from_folder.assert_not_called()
            frame._open_batch_r_result_tabs.assert_called_once_with(
                [parent, png_only, temporal, nested]
            )
            self.assertEqual(frame.step4_result_folders, [parent, temporal, nested])
            frame.on_continue_to_step4.assert_called_once_with(
                [str(parent), str(temporal), str(nested)]
            )
            frame._update_continue_to_step4_button_state.assert_called()

    def test_no_results_preserves_previous_loaded_state_and_warns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            empty_tree = _write_files(root / "empty" / "nested", "notes.RData")
            previous = _write_flat_pair(root / "previous")
            frame = _frame()
            frame.step4_result_folders = [previous]
            original_results = frame.results
            original_info = frame.info_var.get()
            original_progress = frame.progress_text_var.get()

            with mock.patch("aidas.steps.step3_flatten.messagebox.showwarning") as warning:
                self.assertFalse(frame._load_r_results_from_tree(empty_tree.parent))

            self.assertIs(frame.results, original_results)
            self.assertEqual(frame.current_sdb_dir, "previous input")
            self.assertEqual(frame.output_sdb_dir, "previous output")
            self.assertEqual(frame.step4_result_folders, [previous])
            self.assertIn("no", frame.status_var.get().lower())
            self.assertIn("result", frame.status_var.get().lower())
            self.assertEqual(frame.info_var.get(), original_info)
            self.assertEqual(frame.progress_text_var.get(), original_progress)
            frame._load_r_results_from_folder.assert_not_called()
            frame._open_batch_r_result_tabs.assert_not_called()
            warning.assert_called_once()
            warning_text = " ".join(str(value) for value in warning.call_args.args).lower()
            self.assertIn("result", warning_text)

    def test_suppressed_no_results_warning_does_not_open_dialog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame = _frame()

            with mock.patch("aidas.steps.step3_flatten.messagebox.showwarning") as warning:
                self.assertFalse(frame._load_r_results_from_tree(temp_dir, show_errors=False))

            warning.assert_not_called()

    def test_browse_uses_recursive_loader_and_cancel_is_a_noop(self):
        frame = _frame()
        frame._load_r_results_from_tree = mock.Mock()

        with mock.patch("aidas.steps.step3_flatten.filedialog.askdirectory",
                        return_value="chosen parent"):
            frame._browse_r_results_folder()

        frame._load_r_results_from_tree.assert_called_once_with(
            "chosen parent", show_errors=True
        )
        frame._load_r_results_from_tree.reset_mock()
        with mock.patch("aidas.steps.step3_flatten.filedialog.askdirectory", return_value=""):
            frame._browse_r_results_folder()

        frame._load_r_results_from_tree.assert_not_called()
        frame._load_r_results_from_folder.assert_not_called()
        self.assertEqual(frame.output_sdb_dir, "previous output")


if __name__ == "__main__":
    unittest.main()
