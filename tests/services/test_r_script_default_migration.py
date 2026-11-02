from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aidas.core.config import Config
from aidas.steps.step3_flatten import Step3Frame


class RScriptDefaultMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_dir = self.root / "preferences"
        self.config_file = self.config_dir / "preferences.json"
        for name, value in (("CONFIG_DIR", self.config_dir), ("CONFIG_FILE", self.config_file)):
            patcher = mock.patch.object(Config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.originals = {
            "main": self._script("bundled", Step3Frame.R_SCRIPT_NAME),
            "output": self._script("bundled", Step3Frame.R_OUTPUT_SCRIPT_NAME),
        }
        self.optimized = {
            "main": self._script("bundled", Step3Frame.R_OPTIMIZED_SCRIPT_NAME),
            "output": self._script("bundled", Step3Frame.R_OPTIMIZED_OUTPUT_SCRIPT_NAME),
        }
        self.deep = {
            "main": self._script("bundled", Step3Frame.R_DEEP_LAYERS_SCRIPT_NAME),
            "output": self._script("bundled", Step3Frame.R_DEEP_LAYERS_OUTPUT_SCRIPT_NAME),
        }

    def _script(self, folder, name):
        path = self.root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# script fixture\n", encoding="utf-8")
        return path.resolve()

    def _preferences(self, values):
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(values), encoding="utf-8")
        return Config()

    def _frame(self, preferences):
        frame = Step3Frame.__new__(Step3Frame)
        frame.preferences = preferences
        frame._script_path = lambda role="main": self.deep[role]
        frame._user_r_script_dir = lambda role="main": self.root / "user" / role
        return frame

    def _legacy_preferences(self):
        return self._preferences({
            f"r_{role}_script_path": str(path)
            for role, path in self.originals.items()
        })

    def test_default_script_pair_is_the_deeper_layer_pair(self):
        self.assertEqual(Step3Frame._script_path("main").name, Step3Frame.R_DEEP_LAYERS_SCRIPT_NAME)
        self.assertEqual(Step3Frame._script_path("output").name, Step3Frame.R_DEEP_LAYERS_OUTPUT_SCRIPT_NAME)

    def test_older_optimized_defaults_upgrade_to_matching_deep_pair_once(self):
        for version in (0, 1):
            with self.subTest(version=version):
                preferences = self._preferences({
                    **{f"r_{role}_script_path": str(path) for role, path in self.optimized.items()},
                    **{f"r_{role}_script_defaults_version": version for role in self.optimized},
                })
                frame = self._frame(preferences)

                frame._migrate_r_script_defaults()

                for role in self.optimized:
                    self.assertEqual(frame._selected_r_script_path(role), self.deep[role])
                    self.assertEqual(preferences.get(f"r_{role}_script_path"), str(self.deep[role]))
                    self.assertEqual(preferences.get(f"r_{role}_script_defaults_version"), 2)
                reloaded = Config()
                with mock.patch.object(reloaded, "set", wraps=reloaded.set) as persist:
                    self._frame(reloaded)._migrate_r_script_defaults()
                persist.assert_not_called()

    def test_version_one_explicit_original_choices_are_preserved_and_marked_current(self):
        preferences = self._preferences({
            **{f"r_{role}_script_path": str(path) for role, path in self.originals.items()},
            **{f"r_{role}_script_defaults_version": 1 for role in self.originals},
        })
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        for role, path in self.originals.items():
            self.assertEqual(frame._selected_r_script_path(role), path)
            self.assertEqual(preferences.get(f"r_{role}_script_path"), str(path))
            self.assertEqual(preferences.get(f"r_{role}_script_defaults_version"), 2)

    def test_current_and_future_markers_preserve_all_explicit_choices_without_writes(self):
        custom = {role: self._script("external", path.name) for role, path in self.optimized.items()}
        for version in (2, 3, 99):
            for selected in (self.originals, self.optimized, custom):
                with self.subTest(version=version, paths=selected):
                    preferences = self._preferences({
                        **{f"r_{role}_script_path": str(path) for role, path in selected.items()},
                        **{f"r_{role}_script_defaults_version": version for role in selected},
                    })
                    frame = self._frame(preferences)
                    with mock.patch.object(preferences, "set", wraps=preferences.set) as persist:
                        frame._migrate_r_script_defaults()
                    persist.assert_not_called()
                    for role, path in selected.items():
                        self.assertEqual(frame._selected_r_script_path(role), path)
                        self.assertEqual(preferences.get(f"r_{role}_script_defaults_version"), version)

    def test_external_optimized_filenames_are_not_migrated_from_either_older_version(self):
        custom = {role: self._script("external", path.name) for role, path in self.optimized.items()}
        for version in (0, 1):
            with self.subTest(version=version):
                preferences = self._preferences({
                    **{f"r_{role}_script_path": str(path) for role, path in custom.items()},
                    **{f"r_{role}_script_defaults_version": version for role in custom},
                })
                frame = self._frame(preferences)
                frame._migrate_r_script_defaults()
                for role, path in custom.items():
                    self.assertEqual(frame._selected_r_script_path(role), path)
                    self.assertEqual(preferences.get(f"r_{role}_script_path"), str(path))
                    self.assertEqual(preferences.get(f"r_{role}_script_defaults_version"), 2)

    def test_both_legacy_bundled_originals_upgrade_and_remain_active_after_relaunch(self):
        preferences = self._legacy_preferences()
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        saved = json.loads(self.config_file.read_text(encoding="utf-8"))
        for role in self.originals:
            self.assertEqual(frame._selected_r_script_path(role), self.deep[role])
            self.assertEqual(saved[f"r_{role}_script_path"], str(self.deep[role]))
            self.assertEqual(saved[f"r_{role}_script_defaults_version"], 2)

        reloaded = Config()
        relaunched = self._frame(reloaded)
        with mock.patch.object(reloaded, "set", wraps=reloaded.set) as persist:
            relaunched._migrate_r_script_defaults()
            for role in self.originals:
                self.assertEqual(relaunched._selected_r_script_path(role), self.deep[role])
            persist.assert_not_called()

    def test_new_preferences_keep_portable_default_paths_and_unrelated_values(self):
        preferences = self._preferences({"custom_future_setting": "keep me"})
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        self.assertEqual(preferences.get("custom_future_setting"), "keep me")
        for role in self.originals:
            self.assertEqual(preferences.get(f"r_{role}_script_path"), "")
            self.assertEqual(frame._selected_r_script_path(role), self.deep[role])

    def test_original_selected_after_upgrade_survives_future_launches(self):
        preferences = self._legacy_preferences()
        frame = self._frame(preferences)
        frame._migrate_r_script_defaults()

        for role in self.originals:
            frame.select_r_script(role, self.originals[role])

        reloaded = Config()
        relaunched = self._frame(reloaded)
        with mock.patch.object(reloaded, "set", wraps=reloaded.set) as persist:
            relaunched._migrate_r_script_defaults()
            persist.assert_not_called()
        for role in self.originals:
            self.assertEqual(relaunched._selected_r_script_path(role), self.originals[role])
            self.assertEqual(reloaded.get(f"r_{role}_script_defaults_version"), 2)

    def test_explicit_original_selection_before_migration_is_preserved(self):
        preferences = self._preferences({})
        frame = self._frame(preferences)

        for role in self.originals:
            frame.select_r_script(role, self.originals[role])

        reloaded = Config()
        relaunched = self._frame(reloaded)
        relaunched._migrate_r_script_defaults()
        for role in self.originals:
            self.assertEqual(relaunched._selected_r_script_path(role), self.originals[role])

    def test_external_custom_scripts_with_original_filenames_are_preserved(self):
        custom = {
            role: self._script("external", path.name)
            for role, path in self.originals.items()
        }
        preferences = self._preferences({
            f"r_{role}_script_path": str(path) for role, path in custom.items()
        })
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        for role, path in custom.items():
            self.assertEqual(frame._selected_r_script_path(role), path)
            self.assertEqual(preferences.get(f"r_{role}_script_path"), str(path))
        relaunched = self._frame(Config())
        relaunched._migrate_r_script_defaults()
        for role, path in custom.items():
            self.assertEqual(relaunched._selected_r_script_path(role), path)

    def test_other_historical_bundled_scripts_are_preserved(self):
        historical = {
            "main": self._script("bundled", "RAW_OCT_PROCESSING_2020_legacy.R"),
            "output": self._script("bundled", "more_outputs_afterRAW_OCT_PROCESSING_2020_legacy.R"),
        }
        preferences = self._preferences({
            f"r_{role}_script_path": str(path) for role, path in historical.items()
        })
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        for role, path in historical.items():
            self.assertEqual(frame._selected_r_script_path(role), path)
            self.assertEqual(preferences.get(f"r_{role}_script_path"), str(path))

    def test_mixed_custom_and_legacy_original_upgrades_only_original_role(self):
        custom = self._script("external", "my_main_workflow.R")
        preferences = self._preferences({
            "r_main_script_path": str(custom),
            "r_output_script_path": str(self.originals["output"]),
        })
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        self.assertEqual(frame._selected_r_script_path("main"), custom)
        self.assertEqual(frame._selected_r_script_path("output"), self.deep["output"])

    def test_missing_legacy_main_original_alias_upgrades(self):
        original = self.originals["main"]
        alias = original.with_name(original.stem + "_original.R")
        self.assertFalse(alias.exists())
        preferences = self._preferences({"r_main_script_path": str(alias)})
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        self.assertEqual(frame._selected_r_script_path("main"), self.deep["main"])
        self.assertEqual(preferences.get("r_main_script_path"), str(self.deep["main"]))

    def test_each_role_migrates_independently_of_existing_marker(self):
        preferences = self._preferences({
            "r_main_script_path": str(self.originals["main"]),
            "r_main_script_defaults_version": 1,
            "r_output_script_path": str(self.originals["output"]),
            "r_output_script_defaults_version": 0,
        })
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        self.assertEqual(frame._selected_r_script_path("main"), self.originals["main"])
        self.assertEqual(frame._selected_r_script_path("output"), self.deep["output"])
        for role in self.originals:
            self.assertEqual(preferences.get(f"r_{role}_script_defaults_version"), 2)

    def test_version_one_missing_deep_script_retries_optimized_upgrade_later(self):
        self.deep["main"].unlink()
        preferences = self._preferences({
            "r_main_script_path": str(self.optimized["main"]),
            "r_main_script_defaults_version": 1,
        })
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        self.assertEqual(frame._selected_r_script_path("main"), self.optimized["main"])
        self.assertEqual(preferences.get("r_main_script_defaults_version"), 1)
        self.deep["main"].write_text("# now available\n", encoding="utf-8")
        relaunched = self._frame(Config())
        relaunched._migrate_r_script_defaults()
        self.assertEqual(relaunched._selected_r_script_path("main"), self.deep["main"])
        self.assertEqual(relaunched.preferences.get("r_main_script_defaults_version"), 2)

    def test_missing_deep_script_keeps_original_and_defers_migration(self):
        self.deep["main"].unlink()
        preferences = self._legacy_preferences()
        frame = self._frame(preferences)

        frame._migrate_r_script_defaults()

        self.assertEqual(frame._selected_r_script_path("main"), self.originals["main"])
        self.assertEqual(preferences.get("r_main_script_path"), str(self.originals["main"]))
        self.assertFalse(preferences.get("r_main_script_defaults_version"))
        self.assertEqual(frame._selected_r_script_path("output"), self.deep["output"])

        self.deep["main"].write_text("# deep layers is now available\n", encoding="utf-8")
        relaunched = self._frame(Config())
        relaunched._migrate_r_script_defaults()
        self.assertEqual(relaunched._selected_r_script_path("main"), self.deep["main"])
        self.assertEqual(relaunched.preferences.get("r_main_script_defaults_version"), 2)

    def test_disabled_optimization_defers_migration_without_writes(self):
        preferences = self._legacy_preferences()
        frame = self._frame(preferences)

        with mock.patch.object(Step3Frame, "R_OPTIMIZED_ENABLED", False), \
                mock.patch.object(preferences, "set", wraps=preferences.set) as persist:
            frame._migrate_r_script_defaults()
            persist.assert_not_called()
            for role in self.originals:
                self.assertEqual(frame._selected_r_script_path(role), self.originals[role])
                self.assertFalse(preferences.get(f"r_{role}_script_defaults_version"))

        frame._migrate_r_script_defaults()
        for role in self.originals:
            self.assertEqual(frame._selected_r_script_path(role), self.deep[role])

    def test_migration_is_safe_without_preferences(self):
        frame = self._frame(None)

        frame._migrate_r_script_defaults()

        for role in self.originals:
            self.assertEqual(frame._selected_r_script_path(role), self.deep[role])


if __name__ == "__main__":
    unittest.main()
