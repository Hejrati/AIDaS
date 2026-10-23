from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from aidas.utils.path_completion import directory_completions


class DirectoryCompletionTests(unittest.TestCase):
    @staticmethod
    def expected(folder, name):
        return name, os.path.join(os.path.abspath(folder), name) + os.sep

    def test_existing_folder_lists_only_immediate_directories_in_stable_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("zebra", "Beta", "alpha"):
                (root / name).mkdir()
            (root / "alpha" / "nested").mkdir()
            (root / "a-file.txt").touch()

            expected = [self.expected(root, name) for name in ("alpha", "Beta", "zebra")]
            self.assertEqual(directory_completions(str(root)), expected)
            self.assertEqual(directory_completions(str(root) + os.sep), expected)

    def test_partial_final_component_matches_directory_names_without_case_sensitivity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("Alpha", "alphabet", "Beta"):
                (root / name).mkdir()
            (root / "al-file.txt").touch()

            self.assertEqual(directory_completions(str(root / "aL")), [
                self.expected(root, "Alpha"), self.expected(root, "alphabet"),
            ])

    def test_quoted_environment_and_home_paths_are_expanded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "folder with spaces").mkdir()
            expected = [self.expected(root, "folder with spaces")]
            with mock.patch.dict(os.environ, {"AIDAS_COMPLETION_TEST_ROOT": temporary}):
                self.assertEqual(directory_completions('  "${AIDAS_COMPLETION_TEST_ROOT}"  '), expected)
                self.assertEqual(directory_completions(f"'{root}'"), expected)
            with mock.patch("aidas.utils.filesystem.os.path.expanduser", return_value=temporary):
                self.assertEqual(directory_completions("~"), expected)

    def test_relative_paths_use_the_working_directory_and_return_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("Alpha", "another", "Beta"):
                (root / name).mkdir()
            previous_directory = os.getcwd()
            try:
                os.chdir(root)
                self.assertEqual(directory_completions("al"), [self.expected(root, "Alpha")])
                self.assertEqual(directory_completions("."), [
                    self.expected(root, name) for name in ("Alpha", "another", "Beta")
                ])
            finally:
                os.chdir(previous_directory)

    def test_empty_missing_or_unreadable_parent_returns_no_completions(self):
        for text in ("", "   ", '""', "''"):
            with self.subTest(text=text), mock.patch("os.scandir") as scan:
                self.assertEqual(directory_completions(text), [])
                scan.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(directory_completions(str(Path(temporary) / "missing" / "child")), [])
            with mock.patch("os.scandir", side_effect=PermissionError("denied")):
                self.assertEqual(directory_completions(temporary), [])

    def test_failed_entry_metadata_does_not_hide_other_matching_folders(self):
        entries = [
            SimpleNamespace(name="available", is_dir=mock.Mock(return_value=True)),
            SimpleNamespace(name="unavailable", is_dir=mock.Mock(side_effect=PermissionError("denied"))),
            SimpleNamespace(name="file.txt", is_dir=mock.Mock(return_value=False)),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("os.scandir", return_value=nullcontext(iter(entries))):
                self.assertEqual(directory_completions(temporary), [self.expected(temporary, "available")])
        for entry in entries:
            entry.is_dir.assert_called_once_with()

    def test_all_matching_folders_are_returned_without_a_result_cap(self):
        names = [f"folder {index:03d}" for index in range(137)]
        entries = [SimpleNamespace(name=name, is_dir=lambda: True) for name in reversed(names)]
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("os.scandir", return_value=nullcontext(iter(entries))):
                self.assertEqual(directory_completions(temporary), [
                    self.expected(temporary, name) for name in names
                ])

    @unittest.skipUnless(os.name == "nt", "Windows drive and UNC path semantics")
    def test_drive_and_unc_roots_preserve_absolute_root_paths(self):
        for root in ("C:\\", "\\\\server\\share\\"):
            with self.subTest(root=root):
                entry = SimpleNamespace(name="Child folder", is_dir=lambda: True)
                with mock.patch("os.path.isdir", return_value=True), mock.patch(
                    "os.scandir", return_value=nullcontext(iter([entry])),
                ) as scan:
                    self.assertEqual(directory_completions(root), [self.expected(root, entry.name)])
                scan.assert_called_once_with(os.path.abspath(root))

    @unittest.skipUnless(os.name == "nt", "Windows UNC path semantics")
    def test_partial_unc_component_scans_only_the_share_parent(self):
        root = "\\\\server\\share\\"
        entries = [SimpleNamespace(name=name, is_dir=lambda: True) for name in ("Alpha", "Beta")]
        with mock.patch("os.path.isdir", return_value=False), mock.patch(
            "os.scandir", return_value=nullcontext(iter(entries)),
        ) as scan:
            self.assertEqual(directory_completions(root + "al"), [self.expected(root, "Alpha")])
        scan.assert_called_once_with(os.path.abspath(root))


if __name__ == "__main__":
    unittest.main()
