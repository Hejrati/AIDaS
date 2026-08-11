from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.app import AIDaSApp, SettingsDialog


class _Preferences:
    def __init__(self):
        self.saved = []

    def set(self, key, value):
        self.saved.append((key, value))


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
        app.update_idletasks = mock.Mock()
        app.winfo_width = mock.Mock(return_value=1000)
        app.winfo_height = mock.Mock(return_value=700)
        app.state = mock.Mock(return_value="normal")
        app.after_idle = mock.Mock()
        app.after = mock.Mock()
        app._restore_native_title_bar = mock.Mock(return_value=True)
        app._destroy_application_menus = mock.Mock()
        app._destroy_workflow_header = mock.Mock()
        app._destroy_status_surface = mock.Mock()
        app._install_modern_title_bar = mock.Mock()
        app._build_menu = mock.Mock()
        app._build_workflow_header = mock.Mock()
        app._build_status_surface = mock.Mock()
        app._queue_interface_widget_refresh = mock.Mock()
        app._sync_settings_interface_controls = mock.Mock()
        app._set_status_message = mock.Mock()
        return app

    def test_selecting_classic_switches_live_without_rebuilding_workflows(self):
        app = self._app_stub()
        retained = (app.notebook, app.step1, app.step2, app.step3, app.step4)

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
            (app.notebook, app.step1, app.step2, app.step3, app.step4),
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
        retained = (app.notebook, app.step1, app.step2, app.step3, app.step4)
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
            (app.notebook, app.step1, app.step2, app.step3, app.step4),
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
        self.assertIn("Modern appearance saved", app.status.options["text"])

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

    def test_settings_reopen_is_deferred_until_apply_has_finished(self):
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
        parent._show_settings.assert_called_once_with()

    def test_show_settings_uses_the_active_interface_and_reuses_dialog(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app._settings_dialog = None
        app.interface_mode = "Classic"
        app.requested_interface_mode = "Modern"
        app.preferences = object()
        app.appearance_mode = "Dark"
        app.step1 = object()
        app.step3 = object()
        existing = mock.Mock()
        existing.winfo_exists.return_value = True

        with mock.patch("aidas.app.SettingsDialog", return_value=existing) as dialog_type:
            app._show_settings()
            app._show_settings()

        self.assertEqual(dialog_type.call_count, 1)
        self.assertEqual(dialog_type.call_args.kwargs["interface_mode"], "Classic")
        existing.lift.assert_called_once_with()
        existing.focus_force.assert_called_once_with()

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


if __name__ == "__main__":
    unittest.main()
