from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aidas.utils.r_script_library import discover_r_scripts, import_r_script, user_r_script_dir


class RScriptLibraryTests(unittest.TestCase):
    def test_discovery_puts_embedded_default_first_and_lists_every_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundled = root / "bundled"
            user = root / "user"
            bundled.mkdir()
            user.mkdir()
            default = bundled / "main.R"
            default.write_text("# default", encoding="utf-8")
            (bundled / "older.r").write_text("# older", encoding="utf-8")
            (bundled / "ignore.txt").write_text("not R", encoding="utf-8")
            (user / "custom.R").write_text("# custom", encoding="utf-8")

            choices = discover_r_scripts(default, user)

            self.assertEqual([choice.path.name for choice in choices], ["main.R", "older.r", "custom.R"])
            self.assertTrue(choices[0].is_default)
            self.assertEqual(choices[0].label, "Embedded (default) — main.R")
            self.assertEqual([choice.source for choice in choices], ["bundled", "bundled", "user"])

    def test_discovery_can_keep_main_and_output_script_versions_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundled = root / "bundled"
            main_user = root / "main-user"
            bundled.mkdir()
            main_user.mkdir()
            main = bundled / "RAW_OCT_PROCESSING_current.R"
            main.write_text("# main", encoding="utf-8")
            (bundled / "RAW_OCT_PROCESSING_older.R").write_text("# old main", encoding="utf-8")
            (bundled / "more_outputs_afterRAW_OCT_PROCESSING_current.R").write_text(
                "# output", encoding="utf-8"
            )

            choices = discover_r_scripts(
                main,
                main_user,
                bundled_prefixes=("RAW_OCT_PROCESSING_",),
            )

            self.assertEqual(
                [choice.path.name for choice in choices],
                ["RAW_OCT_PROCESSING_current.R", "RAW_OCT_PROCESSING_older.R"],
            )

    def test_import_preserves_different_versions_with_the_same_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            library = root / "library"
            source_dir.mkdir()
            source = source_dir / "workflow.R"
            source.write_text("version 1", encoding="utf-8")

            first = import_r_script(source, library)
            source.write_text("version 2", encoding="utf-8")
            second = import_r_script(source, library)

            self.assertEqual(first.name, "workflow.R")
            self.assertEqual(second.name, "workflow (2).R")
            self.assertEqual(first.read_text(encoding="utf-8"), "version 1")
            self.assertEqual(second.read_text(encoding="utf-8"), "version 2")

    def test_import_reuses_an_identical_script_and_rejects_non_r_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "workflow.R"
            source.write_text("same", encoding="utf-8")
            library = root / "library"

            first = import_r_script(source, library)
            second = import_r_script(source, library)
            self.assertEqual(first, second)

            text_file = root / "workflow.txt"
            text_file.write_text("not R", encoding="utf-8")
            with self.assertRaises(ValueError):
                import_r_script(text_file, library)

    def test_user_library_uses_local_app_data_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, {"LOCALAPPDATA": temp_dir}):
                self.assertEqual(user_r_script_dir(), Path(temp_dir) / "AIDaS" / "R-scripts")
                self.assertEqual(user_r_script_dir("main"), Path(temp_dir) / "AIDaS" / "R-scripts" / "main")


if __name__ == "__main__":
    unittest.main()
