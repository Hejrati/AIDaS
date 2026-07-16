from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aidas.core.config import Config


class ConfigPersistenceTests(unittest.TestCase):
    def test_existing_preferences_are_preserved_and_new_defaults_are_merged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / ".aidas"
            config_dir.mkdir()
            config_file = config_dir / "preferences.json"
            original = {
                "theme": "vista",
                "rscript_path": r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe",
                "r_package_library_path": r"C:\Users\test\AIDaS-R",
                "custom_future_setting": "keep me",
            }
            config_file.write_text(json.dumps(original), encoding="utf-8")

            with mock.patch.object(Config, "CONFIG_DIR", config_dir), mock.patch.object(
                Config, "CONFIG_FILE", config_file
            ):
                config = Config()
                self.assertEqual(config.get("theme"), "vista")
                self.assertEqual(config.get("custom_future_setting"), "keep me")
                self.assertTrue(config.get("check_for_updates"))
                config.set("last_successful_update_check", 123)

            saved = json.loads(config_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["rscript_path"], original["rscript_path"])
            self.assertEqual(saved["r_package_library_path"], original["r_package_library_path"])
            self.assertEqual(saved["custom_future_setting"], "keep me")
            self.assertEqual(saved["last_successful_update_check"], 123)
            self.assertEqual(list(config_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
