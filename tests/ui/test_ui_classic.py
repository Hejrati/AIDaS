from __future__ import annotations

import unittest
from unittest import mock

from aidas.ui import classic


class _FakeVariable:
    def __init__(self, *, master=None, value=""):
        self.master = master
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeMenu:
    def __init__(self, master, **options):
        self.master = master
        self.options = options
        self.items = []
        self.destroy_calls = 0

    def add_command(self, **options):
        self.items.append(("command", options))

    def add_separator(self):
        self.items.append(("separator", {}))

    def add_cascade(self, **options):
        self.items.append(("cascade", options))

    def add_radiobutton(self, **options):
        self.items.append(("radiobutton", options))

    def index(self, index):
        if index == "end":
            return len(self.items) - 1 if self.items else None
        return int(index)

    def entryconfigure(self, index, **options):
        self.items[int(index)][1].update(options)

    def destroy(self):
        self.destroy_calls += 1


class _FakeRoot:
    def __init__(self):
        self.configure_calls = []
        self.destroy_calls = 0

    def configure(self, **options):
        self.configure_calls.append(options)

    def winfo_toplevel(self):
        return self

    def destroy(self):
        self.destroy_calls += 1


class ClassicMenuChoiceTests(unittest.TestCase):
    def test_choice_matching_is_case_insensitive_and_whitespace_tolerant(self):
        self.assertEqual(
            classic.canonical_menu_choice(" classic ", ("Modern", "Classic")),
            "Classic",
        )

    def test_unknown_choice_uses_first_declared_value(self):
        self.assertEqual(
            classic.canonical_menu_choice("obsolete", ("Modern", "Classic")),
            "Modern",
        )

    def test_empty_choice_group_is_rejected(self):
        with self.assertRaises(ValueError):
            classic.canonical_menu_choice("Modern", ())


class ClassicApplicationMenuTests(unittest.TestCase):
    def setUp(self):
        self.root = _FakeRoot()
        self.calls = []
        self.patches = (
            mock.patch.object(classic.tk, "Menu", _FakeMenu),
            mock.patch.object(classic.tk, "StringVar", _FakeVariable),
        )
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _build(self):
        callback = lambda name: lambda *args: self.calls.append((name, *args))
        return classic.build_classic_application_menu(
            self.root,
            interface_modes=("Modern", "Classic"),
            current_interface="classic",
            set_interface_command=callback("interface"),
            appearance_modes=("System", "Light", "Dark"),
            current_appearance="dark",
            set_appearance_command=callback("appearance"),
            browse_sdb_command=callback("browse"),
            settings_command=callback("settings"),
            check_updates_command=callback("updates"),
            about_command=callback("about"),
            tutorial_command=callback("tutorial"),
            exit_command=callback("exit"),
        )

    def test_builds_native_file_view_and_help_menu_model(self):
        menu = self._build()

        self.assertIs(self.root.configure_calls[-1]["menu"], menu.menubar)
        self.assertEqual(
            [item[1]["label"] for item in menu.menubar.items],
            ["File", "View", "Help"],
        )
        self.assertEqual(
            [item[1].get("label") for item in menu.file_menu.items],
            ["Browse SDB Parent Directory", "Settings...", None, "Exit"],
        )
        self.assertEqual(
            [item[1]["label"] for item in menu.view_menu.items],
            ["Interface"],
        )
        self.assertIs(menu.appearance_menu, menu.interface_menu)
        self.assertEqual(
            [item[1].get("label") for item in menu.interface_menu.items],
            ["Modern", "Classic", None, "Appearance", "System", "Light", "Dark"],
        )
        self.assertEqual(
            [item[1].get("label") for item in menu.help_menu.items],
            [
                "Workflow Tutorial...",
                None,
                "Check for Updates...",
                None,
                "About",
            ],
        )
        self.assertEqual(menu.current_interface, "Classic")
        self.assertEqual(menu.current_appearance, "Dark")

    def test_radio_groups_retain_variables_and_dispatch_canonical_values(self):
        menu = self._build()

        interface_modern = menu.interface_menu.items[0][1]
        appearance_light = menu.interface_menu.items[5][1]
        self.assertIs(interface_modern["variable"], menu.interface_var)
        self.assertIs(appearance_light["variable"], menu.appearance_var)

        menu.interface_var.set(interface_modern["value"])
        interface_modern["command"]()
        menu.appearance_var.set(appearance_light["value"])
        appearance_light["command"]()

        self.assertIn(("interface", "Modern"), self.calls)
        self.assertIn(("appearance", "Light"), self.calls)

    def test_classic_disables_all_retained_modern_appearance_entries(self):
        menu = self._build()

        appearance_entries = menu.interface_menu.items[4:]
        self.assertFalse(menu.appearance_enabled)
        self.assertEqual(
            [options["state"] for _kind, options in appearance_entries],
            [classic.tk.DISABLED] * 3,
        )
        self.assertEqual(menu.current_appearance, "Dark")

    def test_set_interface_toggles_appearance_entries_without_callbacks(self):
        menu = self._build()

        self.assertEqual(menu.set_interface("Modern"), "Modern")
        self.assertTrue(menu.appearance_enabled)
        self.assertEqual(
            [options["state"] for _kind, options in menu.interface_menu.items[4:]],
            [classic.tk.NORMAL] * 3,
        )

        self.assertEqual(menu.set_interface("Classic"), "Classic")
        self.assertFalse(menu.appearance_enabled)
        self.assertEqual(
            [options["state"] for _kind, options in menu.interface_menu.items[4:]],
            [classic.tk.DISABLED] * 3,
        )
        self.assertEqual(self.calls, [])

    def test_disabled_appearance_selection_is_defensively_ignored(self):
        menu = self._build()

        menu._select_appearance()

        self.assertNotIn(("appearance", "Dark"), self.calls)

    def test_file_and_help_commands_preserve_application_callbacks(self):
        menu = self._build()

        command_items = (
            menu.file_menu.items[0],
            menu.file_menu.items[1],
            menu.file_menu.items[3],
            menu.help_menu.items[0],
            menu.help_menu.items[2],
            menu.help_menu.items[4],
        )
        for _kind, options in command_items:
            options["command"]()

        self.assertEqual(
            self.calls,
            [
                ("browse",),
                ("settings",),
                ("exit",),
                ("tutorial",),
                ("updates",),
                ("about",),
            ],
        )

    def test_programmatic_updates_do_not_invoke_callbacks(self):
        menu = self._build()

        self.assertEqual(menu.set_interface("MODERN"), "Modern")
        self.assertEqual(menu.set_appearance(" light "), "Light")
        self.assertEqual(menu.current_interface, "Modern")
        self.assertEqual(menu.current_appearance, "Light")
        self.assertEqual(self.calls, [])

    def test_destroy_is_idempotent(self):
        menu = self._build()

        menu.destroy()
        menu.destroy()

        self.assertEqual(menu.menubar.destroy_calls, 1)
        self.assertEqual(self.root.configure_calls[-1], {"menu": ""})


if __name__ == "__main__":
    unittest.main()
