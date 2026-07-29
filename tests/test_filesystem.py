from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aidas.utils.filesystem import find_sdb_directories


class FindSdbDirectoriesTests(unittest.TestCase):
    def test_finds_nested_sdb_folders_and_ignores_empty_folders(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "Subject 2"
            second = root / "group" / "Subject 1"
            empty = root / "empty"
            first.mkdir()
            second.mkdir(parents=True)
            empty.mkdir()
            (first / "b.sdb").touch()
            (first / "A.SDB").touch()
            (first / "notes.txt").touch()
            (second / "scan.sdb").touch()

            matches, errors = find_sdb_directories(root)

            self.assertEqual(errors, [])
            self.assertEqual(list(matches), [second, first])
            self.assertEqual(
                matches[first],
                [first / "A.SDB", first / "b.sdb"],
            )
            self.assertNotIn(empty, matches)

    def test_includes_selected_root_when_it_contains_sdb_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "scan.sdb").touch()

            matches, errors = find_sdb_directories(root)

            self.assertEqual(errors, [])
            self.assertEqual(matches, {root: [root / "scan.sdb"]})


if __name__ == "__main__":
    unittest.main()
