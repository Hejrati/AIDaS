from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aidas.core.config import (
    Config,
    STEP4_AUTO_DETECTION_DEFAULTS,
    validate_step4_auto_detection_preferences,
)


class ConfigPersistenceTests(unittest.TestCase):
    def test_peek_reads_without_creating_the_preferences_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / ".aidas"
            config_file = config_dir / "preferences.json"
            with mock.patch.object(Config, "CONFIG_DIR", config_dir), mock.patch.object(
                Config, "CONFIG_FILE", config_file
            ):
                self.assertEqual(Config.peek("interface_mode"), "Modern")
                self.assertFalse(config_dir.exists())

                config_dir.mkdir()
                config_file.write_text(
                    json.dumps({"interface_mode": "Classic"}),
                    encoding="utf-8",
                )
                self.assertEqual(Config.peek("interface_mode"), "Classic")

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
                self.assertEqual(config.get("appearance_mode"), "System")
                self.assertEqual(config.get("interface_mode"), "Modern")
                self.assertEqual(config.get("sdb_raw_width"), 768)
                self.assertEqual(config.get("sdb_raw_height"), 1200)
                self.assertEqual(config.get("sdb_raw_offset"), 1050)
                self.assertTrue(config.get("sdb_little_endian"))
                self.assertEqual(config.get("r_main_script_path"), "")
                self.assertEqual(config.get("r_output_script_path"), "")
                self.assertEqual(config.get("custom_future_setting"), "keep me")
                self.assertTrue(config.get("check_for_updates"))
                for key, value in STEP4_AUTO_DETECTION_DEFAULTS.items():
                    self.assertEqual(config.get(key), value)
                config.set("last_successful_update_check", 123)

            saved = json.loads(config_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["rscript_path"], original["rscript_path"])
            self.assertEqual(saved["r_package_library_path"], original["r_package_library_path"])
            self.assertEqual(saved["custom_future_setting"], "keep me")
            self.assertEqual(saved["last_successful_update_check"], 123)
            self.assertEqual(list(config_dir.glob("*.tmp")), [])

    def test_step4_auto_detection_preferences_are_parsed_and_validated(self):
        values = {key: str(value) for key, value in STEP4_AUTO_DETECTION_DEFAULTS.items()}
        values["step4_auto_savgol_window"] = "11"
        values["step4_auto_confidence_percent"] = "72"

        parsed = validate_step4_auto_detection_preferences(values)

        self.assertEqual(parsed["step4_auto_savgol_window"], 11)
        self.assertEqual(parsed["step4_auto_confidence_percent"], 72.0)

    def test_step4_auto_detection_rejects_even_smoothing_window(self):
        values = dict(STEP4_AUTO_DETECTION_DEFAULTS)
        values["step4_auto_savgol_window"] = 8

        with self.assertRaisesRegex(ValueError, "odd integer"):
            validate_step4_auto_detection_preferences(values)

    def test_legacy_auto_detection_values_migrate_to_current_units(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / ".aidas"
            config_dir.mkdir()
            config_file = config_dir / "preferences.json"
            config_file.write_text(
                json.dumps(
                    {
                        "step4_auto_savgol_polyorder": 4,
                        "step4_auto_confidence": 0.72,
                        "step4_auto_consistency_tolerance_percent": 8.0,
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(Config, "CONFIG_DIR", config_dir), mock.patch.object(
                Config, "CONFIG_FILE", config_file
            ):
                config = Config()

            self.assertEqual(config.get("step4_auto_confidence_percent"), 72.0)
            self.assertEqual(
                config.get("step4_auto_consistency_tolerance"),
                8.0,
            )
            self.assertNotIn("step4_auto_savgol_polyorder", config.prefs)


if __name__ == "__main__":
    unittest.main()
