from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.app import AIDaSApp, SettingsDialog
from aidas.ui.theme import (
    COLOR_PAIRS,
    apply_appearance_mode,
    get_interface_mode,
    resolve_color,
    set_interface_mode,
)
from aidas.utils.r_script_library import RScriptChoice


class _Preferences:
    def __init__(self, values=None):
        self.saved = []
        self.values = dict(values or {})

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.saved.append((key, value))
        self.values[key] = value


class _Status:
    def __init__(self, text="ready"):
        self.options = {"text": text}

    def configure(self, **options):
        self.options.update(options)

    def cget(self, name):
        return self.options[name]


class _Notebook:
    def __init__(self, selected=2):
        self.selected = selected

    def select(self, value=None):
        if value is None:
            return self.selected
        self.selected = value

    def index(self, value):
        return int(value)


class _StatefulMenu:
    def __init__(self, value):
        self.value = value
        self.options = {}

    def get(self):
        return self.value

    def set(self, value):
        self.value = value

    def configure(self, **options):
        self.options.update(options)


class InterfacePreferenceTests(unittest.TestCase):
    def _app_stub(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.interface_mode = "Modern"
        app.requested_interface_mode = "Modern"
        app.appearance_mode = "Dark"
        app.preferences = _Preferences()
        app.style = object()
        app.status = _Status()
        app.status_bar = None
        app.header = None
        app.window_title_bar = None
        app.menu_bar = None
        app.classic_menu = None
        app.notebook = _Notebook()
        app.step1 = object()
        app.step2 = object()
        app.step3 = SimpleNamespace(refresh_appearance=mock.Mock())
        app.step4 = object()
        app.step5 = object()
        app.update_idletasks = mock.Mock()
        app.winfo_width = mock.Mock(return_value=1000)
        app.winfo_height = mock.Mock(return_value=700)
        app.state = mock.Mock(return_value="normal")
        app.after_idle = mock.Mock()
        app.after = mock.Mock()
        app._restore_native_title_bar = mock.Mock(return_value=True)
        app._destroy_application_menus = mock.Mock()
        app._destroy_workflow_header = mock.Mock()
        app._destroy_progress_strip = mock.Mock()
        app._destroy_status_surface = mock.Mock()
        app._install_modern_title_bar = mock.Mock()
        app._build_menu = mock.Mock()
        app._build_workflow_header = mock.Mock()
        app._build_progress_strip = mock.Mock()
        app._build_status_surface = mock.Mock()
        app._queue_interface_widget_refresh = mock.Mock()
        app._sync_settings_interface_controls = mock.Mock()
        app._set_status_message = mock.Mock()
        return app

    def _settings_stub(self, app, *, selected_appearance):
        dialog = SettingsDialog.__new__(SettingsDialog)
        dialog._validated_sdb_defaults = mock.Mock(return_value=(10, 20, 0, True))
        dialog._validated_auto_detection_settings = mock.Mock(return_value={})
        choices = {
            role: SimpleNamespace(path=Path(f"{role}.R"))
            for role, _label in dialog.SCRIPT_ROLES
        }
        dialog._script_by_label = {role: {role: choice} for role, choice in choices.items()}
        dialog._script_vars = {role: _StatefulMenu(role) for role in choices}
        dialog._script_status_vars = {role: mock.Mock() for role in choices}
        dialog._step3 = SimpleNamespace(_busy=False, select_r_script=mock.Mock())
        dialog._step1 = SimpleNamespace(set_sdb_parameter_defaults=mock.Mock())
        dialog._step2 = SimpleNamespace(
            set_annotation_tutorial_enabled=mock.Mock(),
            set_manual_segmentation_enabled=mock.Mock(),
        )
        dialog._preferences = app.preferences
        dialog.annotation_tutorial_var = _StatefulMenu(True)
        dialog._annotation_tutorial_applied_value = True
        dialog._annotation_tutorial_dirty = False
        dialog.manual_segmentation_var = _StatefulMenu(False)
        dialog._manual_segmentation_applied_value = False
        dialog.bright_pixel_confirmation_var = _StatefulMenu(True)
        dialog.update_checks_var = SimpleNamespace(get=lambda: True)
        dialog.interface_menu = _StatefulMenu("Modern")
        dialog.appearance_menu = _StatefulMenu(selected_appearance)
        dialog.apply_status_var = mock.Mock()
        dialog._parent = app
        dialog._classic_settings = app.interface_mode == "Classic"
        dialog._presentation_mode = app.interface_mode
        dialog._presentation_refresh_pending = False
        dialog._schedule_presentation_refresh = mock.Mock()
        dialog.winfo_exists = mock.Mock(return_value=True)
        dialog._set_interface_command = app._set_interface
        dialog._set_appearance_command = app._set_theme
        app._settings_dialog = dialog
        app._sync_settings_interface_controls = AIDaSApp._sync_settings_interface_controls.__get__(
            app, AIDaSApp,
        )
        return dialog

    def test_manual_segmentation_is_applied_live_and_persisted_once_per_change(self):
        app = self._app_stub()
        dialog = self._settings_stub(app, selected_appearance="Dark")
        dialog._set_appearance_command = mock.Mock()

        dialog._apply_changes()
        dialog._step2.set_manual_segmentation_enabled.assert_not_called()
        self.assertFalse(app.preferences.get("step2_manual_segmentation_enabled"))

        dialog.manual_segmentation_var.set(True)
        self.assertFalse(app.preferences.get("step2_manual_segmentation_enabled"))
        dialog._apply_changes()
        dialog._apply_changes()
        dialog._step2.set_manual_segmentation_enabled.assert_called_once_with(True)
        self.assertTrue(app.preferences.get("step2_manual_segmentation_enabled"))

        dialog.manual_segmentation_var.set(False)
        dialog._apply_changes()
        self.assertFalse(app.preferences.get("step2_manual_segmentation_enabled"))
        self.assertEqual(dialog._step2.set_manual_segmentation_enabled.call_args_list,
                         [mock.call(True), mock.call(False)])

    def test_manual_segmentation_preference_is_applied_without_an_open_step2(self):
        app = self._app_stub()
        dialog = self._settings_stub(app, selected_appearance="Dark")
        dialog._set_appearance_command = mock.Mock()
        dialog._step2 = None
        dialog.manual_segmentation_var.set(True)

        dialog._apply_changes()

        self.assertTrue(app.preferences.get("step2_manual_segmentation_enabled"))

    def test_settings_can_enable_automatic_correction_and_restore_confirmation(self):
        app = self._app_stub()
        dialog = self._settings_stub(app, selected_appearance="Dark")
        dialog._set_appearance_command = mock.Mock()
        dialog.bright_pixel_confirmation_var.set(False)
        dialog._apply_changes()
        self.assertTrue(app.preferences.get("step1_auto_fix_bright_pixels"))
        dialog.bright_pixel_confirmation_var.set(True)
        dialog._apply_changes()
        self.assertFalse(app.preferences.get("step1_auto_fix_bright_pixels"))

    def test_settings_preserves_remembered_ignore_and_can_restore_confirmation(self):
        app = self._app_stub()
        app.preferences.set("step1_bright_pixel_choice", "ignore")
        dialog = self._settings_stub(app, selected_appearance="Dark")
        dialog._set_appearance_command = mock.Mock()
        dialog.bright_pixel_confirmation_var.set(False)
        dialog._apply_changes()
        self.assertEqual(app.preferences.get("step1_bright_pixel_choice"), "ignore")
        self.assertFalse(app.preferences.get("step1_auto_fix_bright_pixels"))
        dialog.bright_pixel_confirmation_var.set(True)
        dialog._apply_changes()
        self.assertEqual(app.preferences.get("step1_bright_pixel_choice"), "")
        self.assertFalse(app.preferences.get("step1_auto_fix_bright_pixels"))

    def test_settings_restores_guide_once_and_applies_disabling_immediately(self):
        app = self._app_stub()
        dialog = self._settings_stub(app, selected_appearance="Dark")
        dialog._annotation_tutorial_applied_value = False
        dialog._set_appearance_command = mock.Mock()

        dialog._apply_changes()
        dialog._apply_changes()

        dialog._step2.set_annotation_tutorial_enabled.assert_called_once_with(True)
        self.assertTrue(app.preferences.get("show_step2_annotation_tutorial"))

        dialog.annotation_tutorial_var.set(False)
        dialog._annotation_tutorial_choice_changed()
        dialog._apply_changes()

        self.assertFalse(app.preferences.get("show_step2_annotation_tutorial"))
        self.assertEqual(dialog._step2.set_annotation_tutorial_enabled.call_args_list,
                         [mock.call(True), mock.call(False)])
        self.assertFalse(dialog._annotation_tutorial_dirty)

    def test_explicit_off_on_toggle_replays_session_dismissed_guide_once(self):
        app = self._app_stub()
        dialog = self._settings_stub(app, selected_appearance="Dark")
        dialog._set_appearance_command = mock.Mock()

        dialog._apply_changes()
        dialog._step2.set_annotation_tutorial_enabled.assert_not_called()
        dialog.annotation_tutorial_var.set(False)
        dialog._annotation_tutorial_choice_changed()
        dialog.annotation_tutorial_var.set(True)
        dialog._annotation_tutorial_choice_changed()
        dialog._apply_changes()
        dialog._apply_changes()

        dialog._step2.set_annotation_tutorial_enabled.assert_called_once_with(True)
        self.assertTrue(app.preferences.get("show_step2_annotation_tutorial"))
        self.assertFalse(dialog._annotation_tutorial_dirty)

    def test_guide_preference_is_saved_without_step2_before_interface_refresh(self):
        app = self._app_stub()
        dialog = self._settings_stub(app, selected_appearance="Dark")
        dialog._step2 = None
        dialog.annotation_tutorial_var.set(False)
        dialog._annotation_tutorial_choice_changed()
        dialog.interface_menu.set("Classic")

        def change_interface(selected):
            self.assertFalse(app.preferences.get("show_step2_annotation_tutorial"))
            dialog._sync_interface_controls(selected, "Light")
            self.assertFalse(dialog.annotation_tutorial_var.get())
            return selected

        dialog._set_interface_command = change_interface
        dialog._apply_changes()

        self.assertFalse(dialog.annotation_tutorial_var.get())
        self.assertFalse(dialog._annotation_tutorial_dirty)
        self.assertEqual(dialog.apply_status_var.set.call_args.args[0], "All settings applied")

    def test_settings_applies_and_persists_selected_appearance_after_interface_resync(self):
        for previous, selected in (("Dark", "Light"), ("System", "Dark"), ("System", "Light")):
            with self.subTest(previous=previous, selected=selected):
                app = self._app_stub()
                app.appearance_mode = previous
                dialog = self._settings_stub(app, selected_appearance=selected)

                with (
                    mock.patch("aidas.app.apply_appearance_mode") as apply,
                    mock.patch("aidas.app.ctk.get_appearance_mode", return_value=selected),
                ):
                    dialog._apply_changes()

                self.assertEqual(app.appearance_mode, selected)
                self.assertEqual(dialog.appearance_menu.get(), selected)
                self.assertEqual(
                    [saved for saved in app.preferences.saved if saved[0] == "appearance_mode"],
                    [("appearance_mode", selected)],
                )
                apply.assert_called_once_with(selected, root=app, style=app.style)

    def test_settings_keeps_selected_appearance_when_switching_classic_to_modern(self):
        app = self._app_stub()
        app.interface_mode = "Classic"
        app.appearance_mode = "System"
        dialog = self._settings_stub(app, selected_appearance="Dark")

        def switch_interface(selected):
            app.interface_mode = selected
            app._sync_settings_interface_controls()
            return selected

        dialog._set_interface_command = switch_interface
        with (
            mock.patch("aidas.app.apply_appearance_mode") as apply,
            mock.patch("aidas.app.ctk.get_appearance_mode", return_value="Dark"),
        ):
            dialog._apply_changes()

        self.assertEqual(app.interface_mode, "Modern")
        self.assertEqual(app.appearance_mode, "Dark")
        self.assertEqual(dialog.appearance_menu.get(), "Dark")
        self.assertIn(("appearance_mode", "Dark"), app.preferences.saved)
        apply.assert_called_once_with("Dark", root=app, style=app.style)

    def test_selecting_classic_switches_live_without_rebuilding_workflows(self):
        app = self._app_stub()
        retained = (app.notebook, app.step1, app.step2, app.step3, app.step4, app.step5)

        with (
            mock.patch("aidas.app.logical_window_size", return_value=(1000, 700)),
            mock.patch("aidas.app.set_interface_mode", return_value="Classic"),
            mock.patch("aidas.app.apply_appearance_mode") as apply,
            mock.patch("aidas.app.refresh_native_widgets"),
            mock.patch("aidas.app.reassert_client_size") as reassert,
            mock.patch("aidas.app.synchronize_window_chrome"),
            mock.patch("aidas.app.ctk.get_appearance_mode", return_value="Light"),
        ):
            app._set_interface(" classic ")

        self.assertEqual(app.interface_mode, "Classic")
        self.assertEqual(app.requested_interface_mode, "Classic")
        self.assertEqual(app.preferences.saved, [("interface_mode", "Classic")])
        apply.assert_called_once_with(
            "Light",
            root=app,
            style=app.style,
            force_ctk_redraw=True,
            defer_ctk_ms=25,
        )
        reassert.assert_called_once_with(app, 1000, 700)
        self.assertEqual(
            retained,
            (app.notebook, app.step1, app.step2, app.step3, app.step4, app.step5),
        )
        self.assertEqual(app.notebook.selected, 2)
        app._install_modern_title_bar.assert_not_called()
        app._build_menu.assert_called_once_with()
        app._build_workflow_header.assert_called_once_with()

    def test_reselecting_current_interface_is_a_noop(self):
        app = self._app_stub()

        app._set_interface("Modern")

        self.assertEqual(app.preferences.saved, [])
        app._build_menu.assert_called_once_with()
        app._destroy_application_menus.assert_not_called()

    def test_failed_native_caption_restore_keeps_modern_active(self):
        app = self._app_stub()
        app._restore_native_title_bar.return_value = False

        with mock.patch(
            "aidas.app.logical_window_size",
            return_value=(1000, 700),
        ):
            app._set_interface("Classic")

        self.assertEqual(app.interface_mode, "Modern")
        self.assertEqual(app.requested_interface_mode, "Modern")
        self.assertEqual(app.preferences.saved, [])
        app._destroy_application_menus.assert_not_called()
        self.assertIn("could not be restored", app._set_status_message.call_args.args[0])

    def test_shell_build_failure_restores_previous_mode_without_persisting(self):
        app = self._app_stub()
        retained = (app.notebook, app.step1, app.step2, app.step3, app.step4, app.step5)
        app._build_menu.side_effect = [RuntimeError("menu failed"), None]

        with (
            mock.patch("aidas.app.logical_window_size", return_value=(1000, 700)),
            mock.patch(
                "aidas.app.set_interface_mode",
                side_effect=lambda value, **_options: value,
            ),
            mock.patch("aidas.app.apply_appearance_mode"),
            mock.patch("aidas.app.refresh_native_widgets"),
            mock.patch("aidas.app.synchronize_window_chrome"),
            mock.patch("aidas.app.ctk.get_appearance_mode", return_value="Light"),
        ):
            applied = app._set_interface("Classic")

        self.assertEqual(applied, "Modern")
        self.assertEqual(app.interface_mode, "Modern")
        self.assertEqual(app.requested_interface_mode, "Modern")
        self.assertEqual(app.preferences.saved, [])
        self.assertEqual(
            retained,
            (app.notebook, app.step1, app.step2, app.step3, app.step4, app.step5),
        )
        app._install_modern_title_bar.assert_called_once_with()
        self.assertIn("was not applied", app._set_status_message.call_args.args[0])

    def test_maximized_switch_defers_only_the_normal_client_size_correction(self):
        app = self._app_stub()
        app.interface_mode = "Classic"
        app.requested_interface_mode = "Classic"
        app._normal_logical_client_size = (900, 620)
        app.state = mock.Mock(return_value="zoomed")

        with (
            mock.patch("aidas.app.logical_window_size", return_value=(1600, 900)),
            mock.patch("aidas.app.set_interface_mode", return_value="Modern"),
            mock.patch("aidas.app.apply_appearance_mode"),
            mock.patch("aidas.app.refresh_native_widgets"),
            mock.patch("aidas.app.synchronize_window_chrome"),
            mock.patch("aidas.app.ctk.get_appearance_mode", return_value="Dark"),
        ):
            applied = app._set_interface("Modern")

        self.assertEqual(applied, "Modern")
        self.assertEqual(app._pending_normal_client_size, (900, 620))
        self.assertNotIn(mock.call("normal"), app.state.call_args_list)

    def test_title_bar_restores_native_caption_before_widget_destruction(self):
        events = []
        controller = SimpleNamespace(
            restore_native_caption=lambda: events.append("restore") or True
        )
        title_bar = SimpleNamespace(
            controller=controller,
            destroy=lambda: events.append("destroy"),
        )
        app = AIDaSApp.__new__(AIDaSApp)
        app.window_title_bar = title_bar

        self.assertTrue(app._restore_native_title_bar())
        self.assertEqual(events, ["restore", "destroy"])
        self.assertIsNone(app.window_title_bar)

    def test_classic_keeps_light_active_and_retains_modern_choice(self):
        app = self._app_stub()
        app.interface_mode = "Classic"
        app.step3 = None

        with mock.patch("aidas.app.apply_appearance_mode", return_value="Light") as apply:
            app._set_theme("Dark")

        apply.assert_called_once_with("Light", root=app, style=app.style)
        self.assertEqual(app.appearance_mode, "Dark")
        self.assertEqual(app.preferences.saved, [("appearance_mode", "Dark")])
        self.assertEqual(app.status.options["text"], "")

    def test_classic_menu_keeps_all_application_callbacks(self):
        app = self._app_stub()
        app.interface_mode = "Classic"
        app.requested_interface_mode = "Classic"
        app.classic_menu = None
        app.update_controller = SimpleNamespace(check_now=mock.Mock())
        app._build_menu = AIDaSApp._build_menu.__get__(app, AIDaSApp)
        native_menu = SimpleNamespace(menubar=object())

        with mock.patch(
            "aidas.app.build_classic_application_menu",
            return_value=native_menu,
        ) as build_menu:
            app._build_menu()

        options = build_menu.call_args.kwargs
        self.assertIs(options["settings_command"].__self__, app)
        self.assertIs(options["browse_sdb_command"].__self__, app)
        self.assertIs(options["check_updates_command"], app.update_controller.check_now)
        self.assertIs(options["set_interface_command"].__self__, app)
        self.assertIs(options["set_appearance_command"].__self__, app)
        self.assertIs(app.menubar, native_menu.menubar)
        self.assertIsNone(app.menu_bar)

    def test_classic_startup_primes_reusable_modern_shell_surfaces(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.interface_mode = "Classic"
        app._modern_menu_bar_cache = None
        app._modern_header_cache = None
        app._modern_status_bar_cache = None
        menu = mock.Mock()
        header = object()
        status = object()
        app._new_modern_menu_bar = mock.Mock(return_value=menu)
        app._new_modern_workflow_header = mock.Mock(return_value=header)
        app._current_status_text = mock.Mock(return_value="ready")

        with mock.patch("aidas.app.AppStatusBar", return_value=status) as status_type:
            app._prime_modern_shell_cache()

        menu.suspend.assert_called_once_with()
        self.assertIs(app._modern_menu_bar_cache, menu)
        self.assertIs(app._modern_header_cache, header)
        self.assertIs(app._modern_status_bar_cache, status)
        status_type.assert_called_once_with(app, text="ready")


class SharedWorkflowConstructionTests(unittest.TestCase):
    def test_mode_is_selected_before_root_and_splash_construction(self):
        source = inspect.getsource(AIDaSApp.__init__)

        self.assertIn('Config.peek("interface_mode", "Modern")', source)
        self.assertNotIn("self.preferences = Config()", source)
        self.assertLess(source.index("set_interface_mode("), source.index("super().__init__()"))
        self.assertLess(source.index("set_interface_mode("), source.index("SplashWindow("))
        self.assertIn('"<Map>"', source)
        self.assertIn("bind_all", source)

    def test_both_shells_use_one_current_set_of_workflow_frames(self):
        source = inspect.getsource(AIDaSApp._build_application)

        self.assertEqual(source.count("self.preferences = Config()"), 1)
        self.assertEqual(source.count('"Loading preferences..."'), 1)
        for step in range(1, 5):
            self.assertEqual(source.count(f"self.step{step} = Step{step}Frame("), 1)
        self.assertIn("self._install_modern_title_bar()", source)
        self.assertIn("self._build_workflow_header()", source)
        self.assertIn('ttk.Notebook(self, style="AIDaS.TNotebook")', source)

    def test_settings_disables_classic_appearance_and_only_applies_it_in_modern(self):
        sync_source = inspect.getsource(SettingsDialog._sync_interface_controls)
        apply_source = inspect.getsource(SettingsDialog._apply_changes)

        self.assertIn('selected == "Modern"', sync_source)
        self.assertIn(
            'if applied_interface == selected_interface == "Modern":',
            apply_source,
        )
        self.assertIn("self._set_interface_command", apply_source)
        self.assertIn("self._set_appearance_command", apply_source)

    def test_settings_reports_actual_mode_when_interface_switch_is_rejected(self):
        dialog = SettingsDialog.__new__(SettingsDialog)
        dialog._validated_sdb_defaults = mock.Mock(return_value=(10, 20, 0, True))
        choices = {
            role: SimpleNamespace(path=Path(f"{role}.R"))
            for role, _label in dialog.SCRIPT_ROLES
        }
        dialog._script_by_label = {
            role: {role: choice} for role, choice in choices.items()
        }
        dialog._script_vars = {
            role: SimpleNamespace(get=lambda value=role: value)
            for role, _label in dialog.SCRIPT_ROLES
        }
        dialog._script_status_vars = {
            role: mock.Mock() for role, _label in dialog.SCRIPT_ROLES
        }
        dialog._step3 = SimpleNamespace(_busy=False, select_r_script=mock.Mock())
        dialog._step1 = SimpleNamespace(set_sdb_parameter_defaults=mock.Mock())
        dialog._preferences = mock.Mock()
        dialog.bright_pixel_confirmation_var = _StatefulMenu(True)
        dialog.update_checks_var = SimpleNamespace(get=lambda: True)
        dialog.interface_menu = mock.Mock()
        dialog.interface_menu.get.return_value = "Classic"
        dialog.appearance_menu = mock.Mock()
        dialog.appearance_menu.get.return_value = "Dark"
        dialog._set_interface_command = mock.Mock(return_value="Modern")
        dialog._set_appearance_command = mock.Mock()
        dialog.apply_status_var = mock.Mock()
        dialog._parent = SimpleNamespace(interface_mode="Modern")

        dialog._apply_changes()

        dialog.interface_menu.set.assert_called_with("Modern")
        dialog.appearance_menu.configure.assert_called_with(state="normal")
        dialog._set_appearance_command.assert_not_called()
        self.assertIn(
            "Modern remains active",
            dialog.apply_status_var.set.call_args.args[0],
        )

    def test_settings_close_is_deferred_until_apply_has_finished(self):
        dialog = SettingsDialog.__new__(SettingsDialog)
        dialog._validated_sdb_defaults = mock.Mock(return_value=(10, 20, 0, True))
        choices = {
            role: SimpleNamespace(path=Path(f"{role}.R"))
            for role, _label in dialog.SCRIPT_ROLES
        }
        dialog._script_by_label = {
            role: {role: choice} for role, choice in choices.items()
        }
        dialog._script_vars = {
            role: SimpleNamespace(get=lambda value=role: value)
            for role, _label in dialog.SCRIPT_ROLES
        }
        dialog._script_status_vars = {
            role: mock.Mock() for role, _label in dialog.SCRIPT_ROLES
        }
        dialog._step3 = SimpleNamespace(_busy=False, select_r_script=mock.Mock())
        dialog._step1 = SimpleNamespace(set_sdb_parameter_defaults=mock.Mock())
        dialog._preferences = mock.Mock()
        dialog.bright_pixel_confirmation_var = _StatefulMenu(True)
        dialog.update_checks_var = SimpleNamespace(get=lambda: True)
        dialog.interface_menu = mock.Mock()
        dialog.interface_menu.get.return_value = "Modern"
        dialog.appearance_menu = mock.Mock()
        dialog.appearance_menu.get.return_value = "Dark"
        dialog.apply_status_var = mock.Mock()
        dialog._presentation_mode = "Classic"
        dialog._classic_settings = True
        dialog._presentation_refresh_after_id = None
        dialog._presentation_refresh_pending = False
        dialog._applying_changes = False
        queued = []
        dialog.after_idle = lambda callback: queued.append(callback) or "after#1"
        dialog._close = mock.Mock()
        parent = SimpleNamespace(interface_mode="Modern", _settings_dialog=dialog)
        parent._show_settings = mock.Mock()
        dialog._parent = parent

        def switch_interface(_mode):
            dialog._schedule_presentation_refresh()
            return "Modern"

        def set_appearance(_mode):
            # Applying appearance can pump Tk idle work.  No destructive
            # presentation callback may exist until the Apply transaction ends.
            self.assertEqual(queued, [])

        dialog._set_interface_command = switch_interface
        dialog._set_appearance_command = set_appearance

        dialog._apply_changes()

        self.assertEqual(len(queued), 1)
        dialog.appearance_menu.get.assert_called()
        dialog.apply_status_var.set.assert_called_with("All settings applied")
        queued[0]()
        dialog._close.assert_called_once_with()
        self.assertIsNone(parent._settings_dialog)
        parent._show_settings.assert_not_called()

    def test_settings_modal_grab_waits_until_replacement_is_viewable(self):
        dialog = SettingsDialog.__new__(SettingsDialog)
        dialog._modal_activation_after_id = None
        dialog.winfo_exists = mock.Mock(return_value=True)
        dialog.winfo_viewable = mock.Mock(side_effect=(False, True))
        dialog.deiconify = mock.Mock()
        dialog.lift = mock.Mock()
        dialog.grab_set = mock.Mock()
        dialog.focus_force = mock.Mock()
        queued = []
        dialog.after = lambda _delay, callback: queued.append(callback) or "after#1"

        dialog._activate_modal_when_visible()

        dialog.grab_set.assert_not_called()
        self.assertEqual(len(queued), 1)
        queued.pop()()
        dialog.grab_set.assert_called_once_with()
        dialog.focus_force.assert_called_once_with()
        dialog.lift.assert_called_once_with()

    def test_show_settings_uses_the_active_interface_and_reuses_dialog(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app._settings_dialog = None
        app.interface_mode = "Classic"
        app.requested_interface_mode = "Modern"
        app.preferences = object()
        app.appearance_mode = "Dark"
        app.step1 = object()
        app.step2 = object()
        app.step3 = object()
        existing = mock.Mock()
        existing.winfo_exists.return_value = True

        with mock.patch("aidas.app.SettingsDialog", return_value=existing) as dialog_type:
            app._show_settings()
            app._show_settings()

        self.assertEqual(dialog_type.call_count, 1)
        self.assertEqual(dialog_type.call_args.kwargs["interface_mode"], "Classic")
        self.assertIs(dialog_type.call_args.kwargs["step2"], app.step2)
        existing.deiconify.assert_called_once_with()
        existing.lift.assert_called_once_with()
        existing.focus_force.assert_called_once_with()
        existing._schedule_modal_activation.assert_called_once_with(delay_ms=1)

    def test_show_about_uses_the_active_interface_and_reuses_dialog(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app._about_dialog = None
        app.interface_mode = "Classic"
        existing = mock.Mock()
        existing.winfo_exists.return_value = True

        with mock.patch("aidas.app.AboutDialog", return_value=existing) as dialog_type:
            app._show_about()
            app._show_about()

        self.assertEqual(dialog_type.call_count, 1)
        self.assertEqual(dialog_type.call_args.kwargs["interface_mode"], "Classic")
        existing.lift.assert_called_once_with()
        existing.focus_force.assert_called_once_with()


class SettingsCategoryNavigationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = ctk.CTk()
            self.root.withdraw()
            self.root.attributes("-alpha", 0)
            self.root.geometry("1024x820+0+0")
            self.root.deiconify()
        except tk.TclError as exc:
            self.skipTest(f"A transparent Tk test window is unavailable: {exc}")
        self.dialogs = []

    def tearDown(self):
        for dialog in self.dialogs:
            if dialog.winfo_exists():
                dialog._close()
        self.root.destroy()

    def _open(self, presentation, *, values=None):
        preferences = _Preferences(values)
        step3 = SimpleNamespace(
            _busy=False,
            R_REQUIRED_VERSION="3.3.1",
            available_r_scripts=mock.Mock(return_value=[]),
            _selected_r_script_path=mock.Mock(return_value=None),
            select_r_script=mock.Mock(),
        )
        with mock.patch.object(SettingsDialog, "_schedule_modal_activation"):
            dialog = SettingsDialog(
                self.root,
                preferences=preferences,
                interface_mode=presentation,
                set_interface_command=lambda value: value,
                appearance_mode="Light",
                set_appearance_command=mock.Mock(),
                step1=SimpleNamespace(set_sdb_parameter_defaults=mock.Mock()),
                step2=SimpleNamespace(
                    set_annotation_tutorial_enabled=mock.Mock(),
                    set_manual_segmentation_enabled=mock.Mock(),
                ),
                step3=step3,
            )
        self.dialogs.append(dialog)
        dialog.attributes("-alpha", 0)
        self.root.update()
        return dialog, preferences, step3

    def _navigate(self, dialog, key):
        navigation = dialog.category_navigation
        navigation.selection_set(key)
        navigation.focus(key)
        navigation.event_generate("<<TreeviewSelect>>")
        self.root.update()
        self.assertEqual(navigation.selection(), (key,))
        self.assertTrue(dialog._category_pages[key].winfo_viewable())

    def test_categories_are_built_on_demand_and_reused_in_both_presentations(self):
        for presentation in ("Modern", "Classic"):
            with self.subTest(presentation=presentation):
                dialog, preferences, step3 = self._open(presentation)
                self.assertIsInstance(dialog.category_navigation, ttk.Treeview)
                self.assertEqual(set(dialog._category_pages), {"general"})
                self.assertEqual(
                    dialog.category_navigation.get_children(),
                    tuple(key for key, _label in SettingsDialog.CATEGORIES),
                )
                first_page = dialog._category_pages["general"]
                step3.available_r_scripts.assert_not_called()
                for key in ("step1", "step2", "step4"):
                    self._navigate(dialog, key)
                step3.available_r_scripts.assert_not_called()
                self._navigate(dialog, "step3")
                self.assertEqual(step3.available_r_scripts.call_args_list,
                                 [mock.call("main"), mock.call("output")])
                script_page = dialog._category_pages["step3"]
                self._navigate(dialog, "general")
                self.assertIs(dialog._category_pages["general"], first_page)
                self._navigate(dialog, "step3")
                self.assertIs(dialog._category_pages["step3"], script_page)
                self.assertEqual(step3.available_r_scripts.call_count, 2)
                self.assertEqual(preferences.saved, [])
                if presentation == "Classic":
                    self.assertIsInstance(dialog.interface_menu, ttk.Combobox)
                    self.assertIsInstance(dialog.annotation_tutorial_switch, ttk.Checkbutton)
                    self.assertIsInstance(dialog.manual_segmentation_switch, ttk.Checkbutton)
                    self.assertIsInstance(dialog.bright_pixel_confirmation_switch, ttk.Checkbutton)
                else:
                    self.assertIsInstance(dialog.interface_menu, ctk.CTkOptionMenu)
                    self.assertIsInstance(dialog.annotation_tutorial_switch, ctk.CTkSwitch)
                    self.assertIsInstance(dialog.manual_segmentation_switch, ctk.CTkSwitch)
                    self.assertIsInstance(dialog.bright_pixel_confirmation_switch, ctk.CTkSwitch)
                dialog._close()

    def test_pending_settings_survive_category_navigation(self):
        for presentation in ("Modern", "Classic"):
            with self.subTest(presentation=presentation):
                dialog, preferences, _step3 = self._open(presentation)
                self._navigate(dialog, "step1")
                dialog.sdb_default_vars["sdb_raw_width"].set("1001")
                dialog.bright_pixel_confirmation_var.set(False)
                self._navigate(dialog, "step2")
                dialog.annotation_tutorial_var.set(False)
                dialog.manual_segmentation_var.set(True)
                self._navigate(dialog, "step4")
                dialog.auto_detection_vars["step4_auto_start_min"].set("71")
                self._navigate(dialog, "general")
                for key in ("step1", "step2", "step4"):
                    self._navigate(dialog, key)
                self.assertEqual(dialog.sdb_default_vars["sdb_raw_width"].get(), "1001")
                self.assertFalse(dialog.bright_pixel_confirmation_var.get())
                self.assertFalse(dialog.annotation_tutorial_var.get())
                self.assertTrue(dialog.manual_segmentation_var.get())
                self.assertEqual(dialog.auto_detection_vars["step4_auto_start_min"].get(), "71")
                self.assertEqual(preferences.saved, [])
                dialog._close()

    def test_apply_without_opening_r_scripts_preserves_saved_scripts(self):
        original = {
            "r_main_script_path": "C:/saved/main.R",
            "r_output_script_path": "C:/saved/output.R",
        }
        for presentation in ("Modern", "Classic"):
            with self.subTest(presentation=presentation):
                dialog, preferences, step3 = self._open(presentation, values=original)

                dialog._apply_changes()

                step3.available_r_scripts.assert_not_called()
                step3._selected_r_script_path.assert_not_called()
                step3.select_r_script.assert_not_called()
                for key, value in original.items():
                    self.assertEqual(preferences.get(key), value)
                    self.assertNotIn(key, [key for key, _value in preferences.saved])
                self.assertEqual(set(dialog._category_pages), {"general"})
                self.assertEqual(dialog.apply_status_var.get(), "All settings applied")
                dialog._close()

    def test_invalid_parameters_show_their_category_before_reporting_error(self):
        for presentation in ("Modern", "Classic"):
            with self.subTest(presentation=presentation):
                dialog, _preferences, _step3 = self._open(presentation)
                dialog.sdb_default_vars["sdb_raw_width"].set("invalid")
                with mock.patch("aidas.app.messagebox.showerror") as error:
                    self.assertIsNone(dialog._validated_sdb_defaults())
                error.assert_called_once()
                self.assertEqual(dialog.category_navigation.selection(), ("step1",))
                dialog.sdb_default_vars["sdb_raw_width"].set("768")
                self._navigate(dialog, "general")
                dialog.auto_detection_vars["step4_auto_savgol_window"].set("8")
                with mock.patch("aidas.app.messagebox.showerror") as error:
                    self.assertIsNone(dialog._validated_auto_detection_settings())
                error.assert_called_once()
                self.assertEqual(dialog.category_navigation.selection(), ("step4",))
                dialog._close()

    def test_ok_keeps_invalid_settings_open_and_applies_valid_settings_before_closing(self):
        for presentation in ("Modern", "Classic"):
            with self.subTest(presentation=presentation):
                dialog, preferences, step3 = self._open(presentation)
                self.assertEqual(dialog.ok_button.cget("text"), "OK")
                self.assertEqual(dialog.close_button.cget("text"), "Cancel")
                self.assertEqual(dialog.apply_button.cget("text"), "Apply")
                dialog.sdb_default_vars["sdb_raw_width"].set("invalid")

                with mock.patch("aidas.app.messagebox.showerror") as error:
                    dialog.ok_button.invoke()

                error.assert_called_once()
                self.assertTrue(dialog.winfo_exists())
                self.assertEqual(preferences.saved, [])
                dialog.sdb_default_vars["sdb_raw_width"].set("999")
                dialog.ok_button.invoke()

                self.assertFalse(dialog.winfo_exists())
                self.assertEqual(preferences.get("sdb_raw_width"), 999)
                dialog._step1.set_sdb_parameter_defaults.assert_called_once_with(
                    width=999, height=1200, offset=1050, little_endian=True,
                )
                step3.available_r_scripts.assert_not_called()

    def test_cancel_closes_without_persisting_draft_settings(self):
        original = {
            "sdb_raw_width": 768,
            "step2_manual_segmentation_enabled": False,
            "show_step2_annotation_tutorial": True,
        }
        for presentation in ("Modern", "Classic"):
            with self.subTest(presentation=presentation):
                dialog, preferences, _step3 = self._open(presentation, values=original)
                self._navigate(dialog, "step1")
                dialog.sdb_default_vars["sdb_raw_width"].set("999")
                self._navigate(dialog, "step2")
                if presentation == "Modern":
                    dialog.manual_segmentation_switch.toggle()
                    dialog.annotation_tutorial_switch.toggle()
                else:
                    dialog.manual_segmentation_switch.invoke()
                    dialog.annotation_tutorial_switch.invoke()
                self.assertEqual(dialog.apply_status_var.get(), "Unsaved changes")

                dialog.close_button.invoke()

                self.assertFalse(dialog.winfo_exists())
                self.assertEqual(preferences.values, original)
                self.assertEqual(preferences.saved, [])
                dialog._step1.set_sdb_parameter_defaults.assert_not_called()
                dialog._step2.set_manual_segmentation_enabled.assert_not_called()
                dialog._step2.set_annotation_tutorial_enabled.assert_not_called()

    def test_r_script_drafts_survive_navigation_and_apply_only_the_changed_role(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            choices = {}
            for role in ("main", "output"):
                choices[role] = []
                for version in ("original", "optimized"):
                    path = Path(temp_dir) / f"{role}_{version}.R"
                    path.write_text("print('test')\n", encoding="utf-8")
                    choices[role].append(RScriptChoice(
                        version, path.resolve(), "bundled",
                        is_default=version == "original", version=version,
                    ))
            for presentation in ("Modern", "Classic"):
                with self.subTest(presentation=presentation):
                    original = {
                        f"r_{role}_script_path": str(role_choices[0].path)
                        for role, role_choices in choices.items()
                    }
                    dialog, preferences, step3 = self._open(presentation, values=original)
                    selected_paths = {role: values[0].path for role, values in choices.items()}
                    step3.available_r_scripts.side_effect = lambda role: choices[role]
                    step3._selected_r_script_path.side_effect = lambda role: selected_paths[role]

                    def persist_script(role, path):
                        selected_paths[role] = path
                        preferences.set(f"r_{role}_script_path", str(path))

                    step3.select_r_script.side_effect = persist_script
                    self._navigate(dialog, "step3")
                    self.assertEqual(step3.available_r_scripts.call_args_list,
                                     [mock.call("main"), mock.call("output")])
                    self.assertEqual(dialog._script_vars["main"].get(), "Original script")
                    dialog._choose_r_script_option("main", "Optimized version")
                    self.assertEqual(preferences.saved, [])
                    self._navigate(dialog, "general")
                    self._navigate(dialog, "step3")
                    self.assertEqual(dialog._script_vars["main"].get(), "Optimized version")
                    self.assertEqual(dialog._script_vars["output"].get(), "Original script")

                    dialog.apply_button.invoke()

                    step3.select_r_script.assert_called_once_with("main", choices["main"][1].path)
                    self.assertEqual(preferences.get("r_main_script_path"), str(choices["main"][1].path))
                    self.assertEqual(preferences.get("r_output_script_path"), original["r_output_script_path"])
                    script_saves = [(key, value) for key, value in preferences.saved
                                    if key in original]
                    self.assertEqual(script_saves,
                                     [("r_main_script_path", str(choices["main"][1].path))])
                    discovery_count = step3.available_r_scripts.call_count
                    selected_path_count = step3._selected_r_script_path.call_count

                    dialog.apply_button.invoke()

                    self.assertEqual(step3.available_r_scripts.call_count, discovery_count)
                    self.assertEqual(step3._selected_r_script_path.call_count, selected_path_count)
                    self.assertEqual(step3.select_r_script.call_count, 1)
                    self.assertTrue(dialog.winfo_exists())
                    dialog._close()

    def test_native_settings_surfaces_and_headers_follow_live_appearance_changes(self):
        original_interface = get_interface_mode()
        original_appearance = ctk.get_appearance_mode()
        original_source = ctk.AppearanceModeTracker.appearance_mode_set_by
        try:
            set_interface_mode("Modern", redraw=False)
            dialog, preferences, _step3 = self._open("Modern")
            for key in ("step1", "step2", "step4"):
                self._navigate(dialog, key)
            dialog.sdb_default_vars["sdb_raw_width"].set("999")
            headings = []
            for key, page in dialog._category_pages.items():
                headings.extend(
                    child for child in page.winfo_children()
                    if isinstance(child, tk.Label)
                    and child.cget("text") == dict(SettingsDialog.CATEGORIES)[key]
                )
            self.assertEqual(len(headings), len(dialog._category_pages))
            navigation = dialog.category_navigation.master
            navigation_heading = next(
                child for child in navigation.winfo_children()
                if isinstance(child, tk.Label) and child.cget("text") == "Options"
            )
            footer_status = next(
                child for child in dialog.apply_button.master.winfo_children()
                if isinstance(child, tk.Label)
            )
            for mode in ("Light", "Dark", "Light"):
                with self.subTest(mode=mode):
                    style = ttk.Style(dialog)
                    apply_appearance_mode(mode, root=dialog, style=style)
                    self.root.update()
                    surface = resolve_color(COLOR_PAIRS["surface"], mode)
                    subtle = resolve_color(COLOR_PAIRS["surface_subtle"], mode)
                    text = resolve_color(COLOR_PAIRS["text"], mode)
                    for frame in (dialog.settings_panel, dialog.category_content):
                        self.assertEqual(frame.cget("background"), surface)
                    for frame in (navigation, dialog.footer, dialog.apply_button.master):
                        self.assertEqual(frame.cget("background"), subtle)
                    for heading in headings:
                        self.assertEqual(heading.cget("background"), surface)
                        self.assertEqual(heading.cget("foreground"), text)
                    self.assertEqual(navigation_heading.cget("background"), subtle)
                    self.assertEqual(navigation_heading.cget("foreground"), text)
                    self.assertEqual(footer_status.cget("background"), subtle)
                    self.assertEqual(style.lookup("AIDaS.SettingsNav.Treeview", "background"), subtle)
                    self.assertEqual(style.lookup("AIDaS.SettingsNav.Treeview", "fieldbackground"), subtle)
                    for button in (dialog.ok_button, dialog.close_button, dialog.apply_button):
                        self.assertEqual(button._apply_appearance_mode(button.cget("bg_color")), subtle)
                    self.assertEqual(dialog.sdb_default_vars["sdb_raw_width"].get(), "999")
                    self.assertEqual(preferences.saved, [])
        finally:
            for dialog in self.dialogs:
                if dialog.winfo_exists():
                    dialog._close()
            set_interface_mode(original_interface, redraw=False)
            ctk.set_appearance_mode("System" if original_source == "system" else original_appearance)


class AnnotationTutorialControlTests(unittest.TestCase):
    def test_both_settings_presentations_restore_saved_guide_choice_and_track_interaction(self):
        root = None
        try:
            try:
                root = ctk.CTk()
                root.withdraw()
                root.attributes("-alpha", 0)
                root.geometry("1024x820+0+0")
                root.deiconify()
            except tk.TclError as exc:
                self.skipTest(f"A transparent Tk test window is unavailable: {exc}")

            for presentation, initial in (("Modern", False), ("Classic", False), ("Modern", True)):
                with self.subTest(presentation=presentation, initial=initial):
                    preferences = _Preferences(
                        {
                            "show_step2_annotation_tutorial": initial,
                            "step2_manual_segmentation_enabled": initial,
                            "step1_auto_fix_bright_pixels": not initial,
                        }
                    )
                    step2 = SimpleNamespace(
                        set_annotation_tutorial_enabled=mock.Mock(),
                        set_manual_segmentation_enabled=mock.Mock(),
                    )
                    with (
                        mock.patch.object(SettingsDialog, "_refresh_r_script_choices"),
                        mock.patch.object(SettingsDialog, "_schedule_modal_activation"),
                    ):
                        dialog = SettingsDialog(
                            root,
                            preferences=preferences,
                            interface_mode=presentation,
                            set_interface_command=lambda value: value,
                            appearance_mode="Light",
                            set_appearance_command=mock.Mock(),
                            step1=SimpleNamespace(),
                            step2=step2,
                            step3=SimpleNamespace(R_REQUIRED_VERSION="3.3.1"),
                        )
                    try:
                        dialog.attributes("-alpha", 0)
                        settled = tk.BooleanVar(master=root, value=False)
                        root.after(100, lambda: settled.set(True))
                        root.wait_variable(settled)
                        root.update()
                        self.assertEqual(set(dialog._category_pages), {"general"})
                        dialog._select_category("step1")
                        root.update()
                        confirmation = dialog.bright_pixel_confirmation_switch
                        self.assertTrue(confirmation.winfo_viewable())
                        self.assertEqual(dialog.bright_pixel_confirmation_var.get(), initial)
                        if presentation == "Modern":
                            confirmation.toggle()
                        else:
                            confirmation.invoke()
                        self.assertEqual(dialog.bright_pixel_confirmation_var.get(), not initial)
                        self.assertEqual(preferences.get("step1_auto_fix_bright_pixels"), not initial)
                        dialog._select_category("step2")
                        root.update()
                        control = dialog.annotation_tutorial_switch
                        self.assertTrue(control.winfo_viewable())
                        self.assertEqual(control.cget("text"), "Enable tutorial")
                        self.assertEqual(dialog.annotation_tutorial_var.get(), initial)
                        self.assertFalse(dialog._annotation_tutorial_dirty)
                        if presentation == "Modern":
                            control.toggle()
                        else:
                            control.invoke()
                        self.assertEqual(dialog.annotation_tutorial_var.get(), not initial)
                        self.assertTrue(dialog._annotation_tutorial_dirty)
                        self.assertEqual(preferences.saved, [])
                        dialog._sync_interface_controls("Classic", "Light")
                        dialog._sync_interface_controls("Modern", "Dark")
                        self.assertEqual(dialog.annotation_tutorial_var.get(), not initial)
                        dialog._apply_annotation_tutorial_setting()
                        step2.set_annotation_tutorial_enabled.assert_called_once_with(not initial)
                        self.assertEqual(preferences.saved,
                                         [("show_step2_annotation_tutorial", not initial)])
                        preferences.saved.clear()
                        manual = dialog.manual_segmentation_switch
                        self.assertEqual(manual.cget("text"), "Enable manual segmentation (Step 2)")
                        self.assertEqual(dialog.manual_segmentation_var.get(), initial)
                        if presentation == "Modern":
                            manual.toggle()
                        else:
                            manual.invoke()
                        self.assertEqual(dialog.manual_segmentation_var.get(), not initial)
                        self.assertEqual(preferences.saved, [])
                        dialog._apply_manual_segmentation_setting()
                        step2.set_manual_segmentation_enabled.assert_called_once_with(not initial)
                        self.assertEqual(preferences.saved,
                                         [("step2_manual_segmentation_enabled", not initial)])
                    finally:
                        dialog._close()
                        root.update()
        finally:
            if root is not None:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
