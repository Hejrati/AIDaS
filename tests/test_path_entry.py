from pathlib import Path
import os
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.ui.path_entry import bind_path_entry
from aidas.utils.filesystem import existing_directory, normalize_path_input


class PathInputTests(unittest.TestCase):
    def test_pasted_quotes_environment_and_spaces_resolve_to_the_existing_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "folder with spaces"
            folder.mkdir()
            with mock.patch.dict(os.environ, {"AIDAS_TEST_PATH_ROOT": temporary}):
                self.assertEqual(existing_directory('  "${AIDAS_TEST_PATH_ROOT}/folder with spaces"  '), folder.resolve())
                self.assertEqual(existing_directory(f"'{folder}'"), folder.resolve())
            self.assertEqual(normalize_path_input('"~"'), os.path.expanduser("~"))

    def test_empty_missing_and_file_paths_are_not_accepted_as_folders(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "file.txt"
            file.touch()
            for value in (None, "", "  ", '""', "''", file, file.parent / "missing"):
                with self.subTest(value=value):
                    self.assertIsNone(existing_directory(value))


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class PathEntryInteractionTests(unittest.TestCase):
    def setUp(self):
        self.root = _QuietRoot()
        self.root.attributes("-alpha", 0)
        self.root.geometry("600x180+0+0")
        self.root.update()
        try:
            self.previous_clipboard = self.root.clipboard_get()
        except tk.TclError:
            self.previous_clipboard = None

    def tearDown(self):
        self.root.clipboard_clear()
        if self.previous_clipboard is not None:
            self.root.clipboard_append(self.previous_clipboard)
        self.root.destroy()

    def _entry(self, kind, command=None):
        entry = kind(self.root)
        entry.pack(fill="x", padx=10, pady=10)
        menu = bind_path_entry(entry, command)
        target = getattr(entry, "_entry", entry)
        self.root.update()
        target.focus_force()
        self.root.update()
        return entry, target, menu

    def test_native_and_ctk_entries_support_select_all_paste_copy_cut_and_enter(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                submit = mock.Mock()
                entry, target, menu = self._entry(kind, submit)
                target.insert(0, "previous address")
                target.event_generate("<Control-a>")
                self.root.clipboard_clear()
                self.root.clipboard_append("C:/folder with spaces")
                target.event_generate("<Control-v>")
                self.root.update()
                self.assertEqual(target.get(), "C:/folder with spaces")
                target.event_generate("<Control-a>")
                menu.invoke(1)
                self.assertEqual(self.root.clipboard_get(), "C:/folder with spaces")
                menu.invoke(0)
                self.assertEqual(target.get(), "")
                menu.invoke(2)
                self.assertEqual(target.get(), "C:/folder with spaces")
                target.event_generate("<Return>")
                self.root.update()
                submit.assert_called_once_with()
                entry.configure(state="disabled")
                target.event_generate("<Return>")
                submit.assert_called_once_with()
                entry.destroy()

    def test_context_menu_reflects_selection_clipboard_and_disabled_state(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                entry, target, menu = self._entry(kind)
                target.insert(0, "address")
                target.selection_range(0, "end")
                self.root.clipboard_clear()
                self.root.clipboard_append("new address")
                with mock.patch.object(menu, "tk_popup") as popup:
                    target.event_generate("<Button-3>", x=4, y=4)
                    self.root.update()
                    popup.assert_called_once()
                    for index in (0, 1, 2, 4):
                        self.assertEqual(menu.entrycget(index, "state"), "normal")
                    entry.configure(state="disabled")
                    target.event_generate("<Button-3>", x=4, y=4)
                    for index in (0, 1, 2, 4):
                        self.assertEqual(menu.entrycget(index, "state"), "disabled")
                entry.destroy()
