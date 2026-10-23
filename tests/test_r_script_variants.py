from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest import mock

from aidas.app import SettingsDialog
from aidas.steps.step3_flatten import Step3Frame
from aidas.utils.r_script_library import RScriptChoice, discover_r_scripts


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Preferences:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.writes = []

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value
        self.writes.append((key, value))


def _write_script(path, content="# R script\n"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _settings(frame):
    dialog = SettingsDialog.__new__(SettingsDialog)
    dialog._step3 = frame
    dialog._script_choices = {}
    dialog._script_by_label = {}
    dialog._script_vars = {role: _Var() for role in ("main", "output")}
    dialog._script_status_vars = {role: _Var() for role in ("main", "output")}
    dialog._script_custom_choices = {role: None for role in ("main", "output")}
    dialog._script_popups = {}
    dialog.script_menus = {role: mock.Mock() for role in ("main", "output")}
    dialog._mark_dirty = mock.Mock()
    return dialog


class RScriptVariantLibraryTests(unittest.TestCase):
    def test_version_metadata_and_labels_only_apply_to_known_bundled_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default = _write_script(root / "bundled" / "workflow.R")
            original = _write_script(default.with_name("workflow_original.R"))
            unrelated = _write_script(default.with_name("workflow_historical.R"))
            user = _write_script(root / "user" / original.name)
            choices = discover_r_scripts(
                default, user.parent,
                bundled_versions={default: "optimized", original: "original", user: "original"},
            )
            by_path = {choice.path: choice for choice in choices}

            self.assertEqual(by_path[default].version, "optimized")
            self.assertTrue(by_path[default].is_default)
            self.assertIn("Optimized", by_path[default].label)
            self.assertEqual(by_path[original].version, "original")
            self.assertIn("Original", by_path[original].label)
            self.assertIsNone(by_path[unrelated].version)
            self.assertEqual(by_path[user].source, "user")
            self.assertIsNone(by_path[user].version)
            self.assertTrue(by_path[user].label.startswith("User"))

    def test_version_path_mapping_resolves_aliases_without_duplicate_choices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default = _write_script(root / "bundled" / "workflow.R")
            alias = default.parent / ".." / "bundled" / default.name

            choices = discover_r_scripts(default, root / "user",
                                         bundled_versions={alias: "optimized"})

            self.assertEqual(len(choices), 1)
            self.assertEqual(choices[0].path, default)
            self.assertEqual(choices[0].version, "optimized")

    def test_legacy_discovery_without_version_mapping_keeps_existing_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default = _write_script(root / "bundled" / "main.R")
            other = _write_script(default.with_name("older.R"))
            user = _write_script(root / "user" / "custom.R")

            choices = discover_r_scripts(default, user.parent)

            self.assertEqual([choice.path for choice in choices], [default, other, user])
            self.assertTrue(choices[0].label.startswith("Embedded (default)"))
            self.assertTrue(choices[1].label.startswith("Bundled"))
            self.assertTrue(choices[2].label.startswith("User"))
            self.assertTrue(all(choice.version is None for choice in choices))

    def test_settings_labels_distinguish_both_versions_and_default(self):
        for version, label in (("original", "Original script"),
                               ("optimized", "Optimized version")):
            with self.subTest(version=version):
                choice = RScriptChoice("library label", Path("workflow.R"), "bundled",
                                       is_default=True, version=version)
                display = SettingsDialog._script_display_label(choice)
                status = SettingsDialog._script_status_label(choice)
                self.assertTrue(display.startswith(label + " (default)"))
                self.assertTrue(status.startswith(label))
                self.assertIn("workflow.R", display)
                alternate = RScriptChoice("library label", Path("alternative.R"), "bundled",
                                          version=version)
                self.assertNotIn("(default)", SettingsDialog._script_display_label(alternate))

        imported = RScriptChoice("User script", Path("workflow_optimized.R"), "user")
        self.assertTrue(SettingsDialog._script_display_label(imported).startswith("User"))
        self.assertEqual(SettingsDialog._script_status_label(imported), imported.path.name)


class Step3RScriptVariantSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        bundled = self.root / "bundled"
        self.originals = {
            "main": _write_script(bundled / Step3Frame.R_SCRIPT_NAME),
            "output": _write_script(bundled / Step3Frame.R_OUTPUT_SCRIPT_NAME),
        }
        self.defaults = {
            "main": _write_script(bundled / Step3Frame.R_OPTIMIZED_SCRIPT_NAME),
            "output": _write_script(bundled / Step3Frame.R_OPTIMIZED_OUTPUT_SCRIPT_NAME),
        }
        self.missing_output_variant = bundled / "more_outputs_afterRAW_OCT_PROCESSING_removed.R"
        self.historical = {
            "main": _write_script(bundled / "RAW_OCT_PROCESSING_2020_legacy.R"),
            "output": _write_script(bundled / "more_outputs_afterRAW_OCT_PROCESSING_2020_legacy.R"),
        }
        self.frame = Step3Frame.__new__(Step3Frame)
        self.frame.preferences = _Preferences()
        self.frame._busy = False
        self.frame._script_path = lambda role="main": self.defaults[role]
        self.frame._user_r_script_dir = lambda role="main": self.root / "user" / role

    def test_actual_bundled_script_paths_use_optimized_defaults(self):
        with mock.patch("aidas.steps.step3_flatten.resource_path",
                        side_effect=lambda name: str(self.root / "bundled" / name)):
            self.assertEqual(Step3Frame._script_path("main"), self.defaults["main"])
            self.assertEqual(Step3Frame._script_path("output"), self.defaults["output"])

        self.assertTrue(Step3Frame.R_OPTIMIZED_ENABLED)
        self.assertEqual(self.originals["main"].name, "RAW_OCT_PROCESSING_2023_09SEP-05_WSU.R")
        self.assertEqual(self.defaults["main"].name, "RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized.R")

    def test_script_paths_and_saved_selections_resolve_directory_aliases(self):
        bundled = self.root / "bundled" / ".." / "bundled"
        with mock.patch("aidas.steps.step3_flatten.resource_path",
                        side_effect=lambda name: str(bundled / name)):
            self.assertEqual(Step3Frame._script_path(), self.defaults["main"])
        self.frame.preferences.values["r_main_script_path"] = str(bundled / self.defaults["main"].name)
        self.assertEqual(self.frame._selected_r_script_path(), self.defaults["main"])

    def test_disable_flag_can_restore_original_defaults(self):
        with mock.patch.object(Step3Frame,"R_OPTIMIZED_ENABLED",False), \
                mock.patch("aidas.steps.step3_flatten.resource_path",
                           side_effect=lambda name: str(self.root / "bundled" / name)):
            self.assertEqual(Step3Frame._script_path("main"),self.originals["main"])
            self.assertEqual(Step3Frame._script_path("output"),self.originals["output"])

    def test_legacy_original_alias_preference_selects_the_original_base_file(self):
        legacy_original = self.originals["main"].with_name(
            self.originals["main"].stem + "_original.R")
        self.frame.preferences.values["r_main_script_path"] = str(legacy_original)
        dialog = _settings(self.frame)

        self.assertFalse(legacy_original.exists())
        self.assertEqual(self.frame._selected_r_script_path("main"), self.originals["main"])
        dialog._refresh_r_script_choices("main")

        self.assertEqual(dialog._script_vars["main"].get(), "Original script")
        self.assertEqual(dialog._script_by_label["main"]["Original script"].path,
                         self.originals["main"])
        self.assertEqual(self.frame.preferences.writes, [])

    def test_existing_external_optimized_name_remains_a_supported_custom_choice(self):
        external = _write_script(self.root / "external" / self.defaults["main"].name)
        self.frame.preferences.values["r_main_script_path"] = str(external)
        dialog = _settings(self.frame)

        self.assertEqual(self.frame._selected_r_script_path("main"), external)
        dialog._refresh_r_script_choices("main")
        self.assertEqual(dialog._script_vars["main"].get(), "Custom script")
        self.assertEqual(dialog._script_by_label["main"]["Custom script"].path, external)
        self.assertEqual(self.frame.preferences.writes, [])

    def test_both_roles_list_optimized_defaults_and_original_alternatives(self):
        for role in ("main", "output"):
            with self.subTest(role=role):
                choices = self.frame.available_r_scripts(role)
                by_path = {choice.path: choice for choice in choices}
                self.assertEqual(choices[0].path, self.defaults[role])
                self.assertEqual(choices[0].version, "optimized")
                self.assertTrue(choices[0].is_default)
                self.assertEqual(by_path[self.originals[role]].version, "original")
                self.assertFalse(by_path[self.originals[role]].is_default)
                self.assertIsNone(by_path[self.historical[role]].version)
                self.assertNotIn(self.defaults["output" if role == "main" else "main"], by_path)
                self.assertEqual(self.frame._selected_r_script_path(role), self.defaults[role])
        self.assertEqual(self.frame.preferences.writes, [])

    def test_absent_optimized_scripts_fall_back_to_originals(self):
        for role in ("main", "output"):
            with self.subTest(role=role):
                self.defaults[role].unlink()
                self.historical[role].unlink()
                choices = self.frame.available_r_scripts(role)
                self.assertEqual([choice.path for choice in choices],
                                 [self.originals[role]])
                self.assertEqual(self.frame._selected_r_script_path(role), self.originals[role])

    def test_existing_saved_custom_choices_remain_active_and_are_not_rewritten(self):
        saved = {}
        for role in ("main", "output"):
            saved[role] = _write_script(self.root / "user" / role / f"custom_{role}.R")
            self.frame.preferences.values[f"r_{role}_script_path"] = str(saved[role])

        for role in ("main", "output"):
            self.assertEqual(self.frame._selected_r_script_path(role), saved[role])
            choices = self.frame.available_r_scripts(role)
            custom = next(choice for choice in choices if choice.path == saved[role])
            self.assertEqual(custom.source, "user")
            self.assertIsNone(custom.version)
        self.assertEqual(self.frame.preferences.writes, [])

    def test_saved_bundled_optimized_selection_remains_active(self):
        self.frame.preferences.values["r_main_script_path"] = str(self.defaults["main"])
        dialog = _settings(self.frame)

        self.assertTrue(self.defaults["main"].is_file())
        self.assertEqual(self.frame._selected_r_script_path("main"), self.defaults["main"])
        dialog._refresh_r_script_choices("main")

        self.assertEqual(dialog._script_vars["main"].get(), "Optimized version")
        self.assertEqual(set(dialog._script_by_label["main"]), {"Original script", "Optimized version"})
        self.assertEqual(self.frame.preferences.writes, [])

    def test_bundled_optimized_scripts_can_be_selected_directly(self):
        for role in ("main", "output"):
            with self.subTest(role=role):
                selected = self.defaults[role]
                self.assertEqual(self.frame.select_r_script(role, selected), selected)
                self.assertEqual(self.frame.preferences.get(f"r_{role}_script_path"), str(selected))
                self.assertEqual(self.frame._selected_r_script_path(role), selected)

    def test_selecting_original_and_custom_paths_persists_exact_paths_and_preserves_history(self):
        for role in ("main", "output"):
            paths = [self.defaults[role], self.originals[role], self.historical[role]]
            for path in paths:
                with self.subTest(role=role, path=path.name):
                    self.assertEqual(self.frame.select_r_script(role, path), path)
                    self.assertEqual(self.frame.preferences.get(f"r_{role}_script_path"), str(path))
                    self.assertEqual(self.frame._selected_r_script_path(role), path)
                    listed = [choice.path for choice in self.frame.available_r_scripts(role)]
                    self.assertIn(self.defaults[role], listed)
                    self.assertIn(self.originals[role], listed)
                    self.assertIn(self.historical[role], listed)

    def test_imports_named_like_bundled_variants_remain_user_choices(self):
        for role in ("main", "output"):
            with self.subTest(role=role):
                builtin = self.defaults[role]
                source = _write_script(self.root / "imports" / builtin.name,
                                       "# independently edited user script\n")
                imported = self.frame.import_r_script_for_role(source, role)
                choice = next(choice for choice in self.frame.available_r_scripts(role)
                              if choice.path == imported)
                self.assertEqual(choice.source, "user")
                self.assertIsNone(choice.version)
                self.assertTrue(SettingsDialog._script_display_label(choice).startswith("User"))
                self.assertEqual(imported.read_text(encoding="utf-8"),
                                 "# independently edited user script\n")
                self.assertEqual(self.frame._selected_r_script_path(role), self.defaults[role])

    def test_settings_refresh_shows_active_versions_without_rewriting_preferences(self):
        dialog = _settings(self.frame)

        for role, expected_label in (("main", "Optimized version"),
                                     ("output", "Optimized version")):
            with self.subTest(role=role):
                dialog._refresh_r_script_choices(role)
                self.assertEqual(dialog._script_vars[role].get(), expected_label)
                self.assertTrue(dialog._script_status_vars[role].get().startswith(
                    "Active: " + expected_label
                ))
                self.assertEqual(set(dialog._script_by_label[role]),
                                 {"Original script", "Optimized version"})
        self.assertEqual(self.frame.preferences.writes, [])
        dialog._mark_dirty.assert_not_called()

    def test_saved_missing_output_variant_falls_back_to_optimized_default(self):
        self.frame.preferences.values["r_output_script_path"] = str(self.missing_output_variant)
        dialog = _settings(self.frame)

        self.assertEqual(self.frame._selected_r_script_path("output"), self.defaults["output"])
        dialog._refresh_r_script_choices("output")

        self.assertEqual(dialog._script_vars["output"].get(), "Optimized version")
        self.assertEqual(set(dialog._script_by_label["output"]), {"Original script", "Optimized version"})
        self.assertEqual(self.frame.preferences.writes, [])

    def test_active_custom_script_uses_fixed_label_and_keeps_filename_in_status(self):
        custom = _write_script(self.root / "user" / "main" / "custom_processing.R")
        self.frame.preferences.values["r_main_script_path"] = str(custom)
        dialog = _settings(self.frame)

        dialog._refresh_r_script_choices("main")

        self.assertEqual(dialog._script_vars["main"].get(), "Custom script")
        self.assertEqual(set(dialog._script_by_label["main"]),
                         {"Original script", "Optimized version", "Custom script"})
        self.assertEqual(dialog._script_by_label["main"]["Custom script"].path, custom)
        self.assertIn(custom.name, dialog._script_status_vars["main"].get())
        self.assertEqual(self.frame.preferences.writes, [])

    def test_settings_apply_persists_pending_version_choices_for_both_roles(self):
        dialog = _settings(self.frame)
        for role in ("main", "output"):
            dialog._refresh_r_script_choices(role)
            selected = self.defaults[role]
            alternate_label = next(label for label, choice in dialog._script_by_label[role].items()
                                   if choice.path == selected)
            dialog._script_vars[role].set(alternate_label)
            dialog._select_r_script(role)
            self.assertTrue(dialog._script_status_vars[role].get().startswith("Pending:"))
        self.assertEqual(self.frame.preferences.writes, [])

        dialog._validated_sdb_defaults = mock.Mock(return_value=(10, 20, 0, True))
        dialog._validated_auto_detection_settings = mock.Mock(return_value={})
        dialog._apply_annotation_tutorial_setting = mock.Mock()
        dialog._preferences = self.frame.preferences
        dialog._step1 = SimpleNamespace(set_sdb_parameter_defaults=mock.Mock())
        dialog.update_checks_var = _Var(True)
        dialog.bright_pixel_confirmation_var = _Var(True)
        dialog.interface_menu = mock.Mock()
        dialog.interface_menu.get.return_value = "Classic"
        dialog.appearance_menu = mock.Mock()
        dialog.appearance_menu.get.return_value = "Dark"
        dialog._set_interface_command = mock.Mock(return_value="Classic")
        dialog._set_appearance_command = mock.Mock()
        dialog._sync_interface_controls = mock.Mock()
        dialog.apply_status_var = _Var()
        dialog._presentation_refresh_pending = False

        dialog._apply_changes()

        for role, label in (("main", "Optimized version"), ("output", "Optimized version")):
            selected = self.defaults[role]
            self.assertEqual(self.frame._selected_r_script_path(role), selected)
            self.assertEqual(self.frame.preferences.get(f"r_{role}_script_path"),
                             str(selected))
            self.assertTrue(dialog._script_status_vars[role].get().startswith("Active: " + label))

    def test_optimized_original_and_custom_choices_reach_actual_r_commands(self):
        self.frame._r_cancel_event = threading.Event()
        self.frame.after = mock.Mock()
        self.frame._r_run_env_for_config = mock.Mock(return_value={})
        self.frame._run_supervised_r_command = mock.Mock(return_value=(0, "", "completed"))
        config = {
            "input_dir": str(self.root), "output_dir": str(self.root),
            "reference_dark": "DARK_MARKED", "reference_light": "LIGHT_MARKED",
            "to_process_dark": "DARK", "to_process_light": "LIGHT",
            "image_index_light": "1,2", "image_index_dark": "1,2", "pixel_width": "3.89",
        }

        for role in ("main", "output"):
            selections = [self.defaults[role], self.originals[role]]
            selections.append(_write_script(self.root / "user" / role / f"custom_{role}.R"))
            for selected in selections:
                with self.subTest(role=role, selected=selected.name):
                    self.frame.select_r_script(role, selected)
                    actual_selected = self.frame._selected_r_script_path(role)
                    if role == "main":
                        result = self.frame._run_main_r_script_for_config(
                            Path("Rscript.exe"), actual_selected, config
                        )
                        self.assertEqual(Path(result["cmd"][0][2]), selected)
                    else:
                        result = self.frame._run_output_r_script_for_config(
                            Path("Rscript.exe"), actual_selected, config, {"cmd": [], "stdout": ""}
                        )
                        self.assertIn(f"source({self.frame._r_string(selected)}",
                                      result["cmd"][0][3])


if __name__ == "__main__":
    unittest.main()
