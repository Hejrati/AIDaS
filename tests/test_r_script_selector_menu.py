from __future__ import annotations

import inspect
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from aidas.app import SettingsDialog
from aidas.utils.r_script_library import RScriptChoice


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Popup:
    def __init__(self):
        self.entries = []

    def delete(self, first, last):
        self.entries.clear()

    def add_command(self, **options):
        self.entries.append({"kind": "command", **options})

    def add_separator(self):
        self.entries.append({"kind": "separator"})

    def invoke(self, index):
        entry = self.entries[index]
        if entry.get("state", "normal") != "disabled":
            entry["command"]()


class _Preferences:
    def __init__(self):
        self.values = {}
        self.writes = []

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value
        self.writes.append((key, value))


class RScriptSelectorMenuTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.preferences = _Preferences()
        self.choices = {}
        self.paths = {}
        self.defaults = {}
        for role, default in (("main", "original"), ("output", "original")):
            versions = ("original",)
            self.paths[role] = {
                version: self._script(f"bundled/{role}_{version}.R")
                for version in versions
            }
            self.defaults[role] = self.paths[role][default]
            self.choices[role] = [
                RScriptChoice(version, self.paths[role][version], "bundled",
                              version == default, version)
                for version in [default]
            ]
            self.choices[role].extend([
                RScriptChoice("Historical", self._script(f"bundled/{role}_historical.R"), "bundled"),
                RScriptChoice("User", self._script(f"library/{role}/old_import.R"), "user"),
            ])
        self.frame = SimpleNamespace(
            _busy=False,
            available_r_scripts=mock.Mock(side_effect=lambda role: self.choices[role]),
            _selected_r_script_path=mock.Mock(side_effect=self._selected_path),
            import_r_script_for_role=mock.Mock(side_effect=self._import_script),
            select_r_script=mock.Mock(side_effect=self._persist_script),
        )

    def _script(self, relative, content="# custom R script\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path.resolve()

    def _selected_path(self, role):
        return Path(self.preferences.get(f"r_{role}_script_path", self.defaults[role]))

    def _persist_script(self, role, path):
        path = Path(path).resolve()
        self.preferences.set(f"r_{role}_script_path", str(path))
        return path

    def _import_script(self, source, role):
        target = self.root / "library" / role / Path(source).name
        shutil.copyfile(source, target)
        target = target.resolve()
        self.choices[role].append(RScriptChoice("User", target, "user"))
        return target

    def _dialog(self, *, classic=False):
        dialog = SettingsDialog.__new__(SettingsDialog)
        dialog._step3 = self.frame
        dialog._preferences = self.preferences
        dialog._classic_settings = classic
        dialog._script_choices = {}
        dialog._script_by_label = {}
        dialog._script_vars = {role: _Var() for role in self.choices}
        dialog._script_status_vars = {role: _Var() for role in self.choices}
        dialog._script_custom_choices = {role: None for role in self.choices}
        dialog._script_popups = {role: _Popup() for role in self.choices}
        dialog.script_menus = {role: mock.Mock() for role in self.choices}
        dialog._mark_dirty = mock.Mock()
        for role in self.choices:
            dialog._refresh_r_script_choices(role)
        return dialog

    def _labels(self, dialog, role):
        return [entry["label"].lstrip("\u2713 ")
                for entry in dialog._script_popups[role].entries
                if entry["kind"] == "command"]

    def _checked(self, dialog, role):
        return [entry["label"][2:]
                for entry in dialog._script_popups[role].entries
                if entry["kind"] == "command" and entry["label"].startswith("\u2713 ")]

    def _prepare_apply(self, dialog):
        dialog._validated_sdb_defaults = mock.Mock(return_value=(10, 20, 0, True))
        dialog._validated_auto_detection_settings = mock.Mock(return_value={})
        dialog._apply_annotation_tutorial_setting = mock.Mock()
        dialog._step1 = SimpleNamespace(set_sdb_parameter_defaults=mock.Mock())
        dialog.update_checks_var = _Var(True)
        dialog.interface_menu = mock.Mock()
        dialog.interface_menu.get.return_value = "Classic"
        dialog.appearance_menu = mock.Mock()
        dialog.appearance_menu.get.return_value = "Dark"
        dialog._set_interface_command = mock.Mock(return_value="Classic")
        dialog._set_appearance_command = mock.Mock()
        dialog._sync_interface_controls = mock.Mock()
        dialog.apply_status_var = _Var()
        dialog._presentation_refresh_pending = False

    def test_both_presentations_show_role_specific_actions_and_a_real_separator(self):
        for classic in (False, True):
            dialog = self._dialog(classic=classic)
            for role in self.choices:
                with self.subTest(classic=classic, role=role):
                    entries = dialog._script_popups[role].entries
                    self.assertEqual([entry["kind"] for entry in entries],
                                     ["command", "separator", "command"])
                    self.assertEqual(self._labels(dialog, role),
                                     ["Original script", "Select custom script\u2026"])
                    self.assertTrue(all(entry.get("state", "normal") == "normal"
                                        for entry in entries if entry["kind"] == "command"))
                    self.assertEqual(set(dialog._script_by_label[role]),
                                     {"Original script"})
                    self.assertFalse(any("historical" in label or "User" in label
                                         for label in self._labels(dialog, role)))
        self.assertEqual(self.preferences.writes, [])

    def test_saved_selection_gets_the_checkmark_instead_of_the_bundled_default(self):
        self.preferences.values["r_main_script_path"] = str(self.paths["main"]["original"])
        self.preferences.values["r_output_script_path"] = str(self.paths["output"]["original"])
        dialog = self._dialog()

        self.assertEqual(self._checked(dialog, "main"), ["Original script"])
        self.assertEqual(self._checked(dialog, "output"), ["Original script"])
        for role in self.choices:
            self.assertFalse(any("default" in label for label in self._labels(dialog, role)))
        dialog._mark_dirty.assert_not_called()
        self.assertEqual(self.preferences.writes, [])

    def test_popup_callbacks_stage_each_role_and_move_its_checkmark_before_apply(self):
        dialog = self._dialog()
        dialog._script_popups["main"].invoke(0)
        dialog._script_popups["output"].invoke(0)

        self.assertEqual(self._checked(dialog, "main"), ["Original script"])
        self.assertEqual(self._checked(dialog, "output"), ["Original script"])
        for role in self.choices:
            self.assertTrue(dialog._script_status_vars[role].get().startswith("Pending:"))
        self.assertEqual(self.frame._selected_r_script_path("main"), self.defaults["main"])
        self.assertEqual(self.frame._selected_r_script_path("output"), self.defaults["output"])
        self.frame.select_r_script.assert_not_called()
        self.assertEqual(self.preferences.writes, [])

    def test_external_saved_custom_script_stays_active_without_a_history_entry(self):
        external = self._script("external/independent.R")
        self.preferences.values["r_main_script_path"] = str(external)
        dialog = self._dialog()

        self.assertEqual(dialog._script_vars["main"].get(), "Custom script")
        self.assertEqual(self._checked(dialog, "main"), ["Select custom script\u2026"])
        self.assertEqual(dialog._script_by_label["main"]["Custom script"].path, external)
        self.assertEqual(dialog._script_by_label["main"]["Custom script"].source, "user")
        self.assertIn(external.name, dialog._script_status_vars["main"].get())
        self.assertEqual(len(self._labels(dialog, "main")), 2)
        self.assertEqual(self.preferences.writes, [])

    def test_canceling_the_custom_picker_preserves_the_previous_pending_choice(self):
        dialog = self._dialog()
        dialog._choose_r_script_option("main", "Original script")
        before = dialog._script_status_vars["main"].get()
        dialog._mark_dirty.reset_mock()

        with mock.patch("aidas.app.filedialog.askopenfilename", return_value="") as picker:
            dialog._script_popups["main"].invoke(2)

        picker.assert_called_once()
        self.assertEqual(dialog._script_vars["main"].get(), "Original script")
        self.assertEqual(dialog._script_status_vars["main"].get(), before)
        self.assertEqual(self._checked(dialog, "main"), ["Original script"])
        dialog._mark_dirty.assert_not_called()
        self.frame.import_r_script_for_role.assert_not_called()
        self.assertEqual(self.preferences.writes, [])

    def test_failed_custom_import_preserves_the_checked_choice_and_preferences(self):
        dialog = self._dialog()
        before = dialog._script_status_vars["main"].get()
        self.frame.import_r_script_for_role.side_effect = ValueError("Not an R script")

        with mock.patch("aidas.app.filedialog.askopenfilename", return_value="invalid.txt"), \
                mock.patch("aidas.app.messagebox.showerror") as error:
            dialog._choose_r_script_option("main", "Custom script")

        error.assert_called_once()
        self.assertEqual(dialog._script_status_vars["main"].get(), before)
        self.assertEqual(self._checked(dialog, "main"), ["Original script"])
        dialog._mark_dirty.assert_not_called()
        self.assertEqual(self.preferences.writes, [])

    def test_successful_custom_picker_remains_pending_until_apply(self):
        dialog = self._dialog()
        selected = self._script("outside/main_optimized.R", "# independently edited custom\n")

        with mock.patch("aidas.app.filedialog.askopenfilename", return_value=str(selected)):
            dialog._choose_r_script_option("main", "Custom script")

        imported = (self.root / "library" / "main" / selected.name).resolve()
        self.assertEqual(dialog._script_vars["main"].get(), "Custom script")
        self.assertEqual(self._checked(dialog, "main"), ["Select custom script\u2026"])
        self.assertEqual(dialog._script_by_label["main"]["Custom script"].path, imported)
        self.assertIsNone(dialog._script_by_label["main"]["Custom script"].version)
        self.assertTrue(dialog._script_status_vars["main"].get().startswith("Pending:"))
        self.assertEqual(self.frame._selected_r_script_path("main"), self.defaults["main"])
        self.assertEqual(self.preferences.writes, [])
        self.assertEqual(imported.read_bytes(), selected.read_bytes())

        self._prepare_apply(dialog)
        dialog._apply_changes()

        self.assertEqual(self.frame._selected_r_script_path("main"), imported)
        self.assertTrue(dialog._script_status_vars["main"].get().startswith("Active:"))
        self.assertEqual(self._checked(dialog, "main"), ["Select custom script\u2026"])

    def test_switching_to_a_builtin_keeps_custom_mapping_without_an_extra_menu_item(self):
        custom = self._script("external/remembered.R")
        self.preferences.values["r_main_script_path"] = str(custom)
        dialog = self._dialog()
        dialog._choose_r_script_option("main", "Original script")
        dialog._refresh_r_script_choices("main", select_path=self.paths["main"]["original"])

        self.assertEqual(self._checked(dialog, "main"), ["Original script"])
        self.assertEqual(dialog._script_custom_choices["main"].path, custom)
        self.assertEqual(dialog._script_by_label["main"]["Custom script"].path, custom)
        self.assertEqual(self._labels(dialog, "main"),
                         ["Original script", "Select custom script\u2026"])
        self.assertEqual(self.preferences.writes, [])

    def test_output_custom_action_targets_output_without_changing_main(self):
        dialog = self._dialog()
        selected = self._script("outside/output_custom.R")

        with mock.patch("aidas.app.filedialog.askopenfilename", return_value=str(selected)) as picker:
            dialog._script_popups["output"].invoke(2)

        self.frame.import_r_script_for_role.assert_called_once_with(selected, "output")
        self.assertEqual(picker.call_args.kwargs["filetypes"][0], ("R scripts", "*.R"))
        self.assertEqual(dialog._script_vars["output"].get(), "Custom script")
        self.assertEqual(self._checked(dialog, "output"), ["Select custom script\u2026"])
        self.assertEqual(dialog._script_vars["main"].get(), "Original script")
        self.assertEqual(self._checked(dialog, "main"), ["Original script"])
        self.assertEqual(self.preferences.writes, [])

        self._prepare_apply(dialog)
        dialog._apply_changes()

        imported = (self.root / "library" / "output" / selected.name).resolve()
        self.assertEqual(self.frame._selected_r_script_path("output"), imported)
        self.assertEqual(self.frame._selected_r_script_path("main"), self.defaults["main"])
        self.assertEqual(self._checked(dialog, "output"), ["Select custom script\u2026"])

    def test_neither_popup_offers_optimized_metadata_from_stale_discovery(self):
        for role in ("main","output"):
            removed = self._script(f"bundled/{role}_optimized.R")
            self.choices[role].append(
                RScriptChoice("Disabled optimized",removed,"bundled",version="optimized")
            )
        dialog = self._dialog()

        for role in ("main","output"):
            self.assertEqual(self._labels(dialog,role),
                             ["Original script","Select custom script\u2026"])
            self.assertNotIn("Optimized version",dialog._script_by_label[role])
            self.assertEqual(self._checked(dialog,role),["Original script"])

    def test_busy_batch_blocks_choices_and_picker_without_moving_the_checkmark(self):
        dialog = self._dialog()
        before = dialog._script_status_vars["main"].get()
        self.frame._busy = True

        with mock.patch("aidas.app.messagebox.showwarning") as warning, \
                mock.patch("aidas.app.filedialog.askopenfilename") as picker:
            dialog._choose_r_script_option("main", "Original script")
            dialog._choose_r_script_option("main", "Custom script")

        self.assertEqual(warning.call_count, 2)
        picker.assert_not_called()
        self.assertEqual(dialog._script_status_vars["main"].get(), before)
        self.assertEqual(self._checked(dialog, "main"), ["Original script"])
        dialog._mark_dirty.assert_not_called()
        self.assertEqual(self.preferences.writes, [])

    def test_missing_variant_is_disabled_while_custom_selection_remains_available(self):
        self.choices["main"] = [choice for choice in self.choices["main"]
                                if choice.version != "original"]
        self.paths["main"]["original"].unlink()
        dialog = self._dialog()
        entries = dialog._script_popups["main"].entries

        self.assertEqual(entries[0]["state"], "disabled")
        self.assertEqual(entries[2]["state"], "normal")
        self.assertEqual(self._checked(dialog, "main"), ["Select custom script\u2026"])
        self.assertEqual(len(self._labels(dialog, "main")), 2)

    def test_popup_rebuild_does_not_accumulate_old_history_or_separator_entries(self):
        dialog = self._dialog()
        for _ in range(3):
            dialog._update_r_script_popup("main")
        self.assertEqual(len(dialog._script_popups["main"].entries), 3)
        self.assertEqual(self._checked(dialog, "main"), ["Original script"])

    def test_settings_has_no_separate_add_script_button_in_either_presentation(self):
        for build in (SettingsDialog.__init__, SettingsDialog._build_classic_settings):
            with self.subTest(builder=build.__name__):
                self.assertNotIn('text="Add R script..."', inspect.getsource(build))


if __name__ == "__main__":
    unittest.main()
