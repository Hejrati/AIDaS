from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest
from unittest import mock

from aidas.steps.step4_compiler_dialog import Step4CompilerDialog
from aidas.steps.step5_compile import DEFAULT_OUTPUT_FILENAME, Step5Frame


class _Variable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class CompilerPathEntryTests(unittest.TestCase):
    VIEW_CLASSES = (Step5Frame, Step4CompilerDialog)

    def make_view(self, view_class, folder):
        view = object.__new__(view_class)
        view._running = False
        view._accepted_input_folder = str(folder)
        view._output_user_selected = False
        view.input_var = _Variable(str(folder))
        view.output_var = _Variable(str(folder / "previous.xlsx"))
        view.status_var = _Variable("Previous selection is ready.")
        view._start_compiler = mock.Mock()
        return view

    def test_enter_accepts_quoted_or_expanded_paths_without_starting_compilation(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path(temporary).resolve()
            selected = previous / "pasted folder"
            selected.mkdir()
            entries = (f'  "{selected}"  ', f"  '{selected}'  ", '"${AIDAS_COMPILER_TEST_FOLDER}"')
            with mock.patch.dict("os.environ", {"AIDAS_COMPILER_TEST_FOLDER": str(selected)}):
                for view_class in self.VIEW_CLASSES:
                    for entry in entries:
                        with self.subTest(view=view_class.__name__, entry=entry):
                            view = self.make_view(view_class, previous)
                            view.input_var.set(entry)

                            view._open_input_folder()

                            self.assertEqual(view.input_var.get(), str(selected))
                            self.assertEqual(view._accepted_input_folder, str(selected))
                            self.assertEqual(Path(view.output_var.get()).parent, selected)
                            self.assertEqual(Path(view.output_var.get()).suffix, ".xlsx")
                            self.assertIn(str(selected), view.status_var.get())
                            view._start_compiler.assert_not_called()

    def test_invalid_enter_restores_the_previous_folder_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            file_path = folder / "file.txt"
            file_path.write_text("not a directory", encoding="utf-8")
            for view_class in self.VIEW_CLASSES:
                for invalid in ("", str(folder / "missing"), str(file_path)):
                    with self.subTest(view=view_class.__name__, invalid=invalid):
                        view = self.make_view(view_class, folder)
                        previous_output = view.output_var.get()
                        previous_status = view.status_var.get()
                        view.input_var.set(invalid)

                        with mock.patch(f"{view_class.__module__}.messagebox.showerror") as error:
                            view._open_input_folder()

                        self.assertEqual(view.input_var.get(), str(folder))
                        self.assertEqual(view._accepted_input_folder, str(folder))
                        self.assertEqual(view.output_var.get(), previous_output)
                        self.assertEqual(view.status_var.get(), previous_status)
                        error.assert_called_once()
                        view._start_compiler.assert_not_called()

    def test_invalid_enter_without_a_previous_folder_restores_an_empty_selection(self):
        for view_class in self.VIEW_CLASSES:
            with self.subTest(view=view_class.__name__):
                view = self.make_view(view_class, Path("unused"))
                view._accepted_input_folder = ""
                view.input_var.set("a-folder-that-does-not-exist")
                with mock.patch(f"{view_class.__module__}.messagebox.showerror"):
                    view._open_input_folder()
                self.assertEqual(view.input_var.get(), "")

    def test_enter_preserves_an_explicit_output_workbook(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path(temporary).resolve()
            selected = previous / "new input"
            selected.mkdir()
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    view = self.make_view(view_class, previous)
                    view._output_user_selected = True
                    output = view.output_var.get()
                    view.input_var.set(str(selected))

                    view._open_input_folder()

                    self.assertEqual(view.input_var.get(), str(selected))
                    self.assertEqual(view.output_var.get(), output)

    def test_enter_preserves_a_typed_or_pasted_output_workbook(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path(temporary).resolve()
            selected = previous / "new input"
            selected.mkdir()
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    view = self.make_view(view_class, previous)
                    view._default_output_path = view.output_var.get()
                    custom_output = f'"{previous / "my workbook.xlsx"}"'
                    view.output_var.set(custom_output)
                    view.input_var.set(str(selected))

                    view._open_input_folder()

                    self.assertEqual(view.input_var.get(), str(selected))
                    self.assertEqual(view.output_var.get(), custom_output)
                    self.assertFalse(view._output_user_selected)

    def test_default_workbook_follows_consecutive_input_folder_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path(temporary).resolve()
            selected = previous / "new input"
            selected.mkdir()
            final = previous / "another input"
            final.mkdir()
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    view = self.make_view(view_class, previous)
                    view._default_output_path = view.output_var.get()
                    for folder in (selected, final):
                        view.input_var.set(str(folder))
                        view._open_input_folder()
                        self.assertEqual(Path(view.output_var.get()).parent, folder)
                        self.assertEqual(view._default_output_path, view.output_var.get())

    def test_browse_and_enter_use_the_same_folder_selection_behavior(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path(temporary).resolve()
            selected = previous / "new input"
            selected.mkdir()
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    entered = self.make_view(view_class, previous)
                    browsed = self.make_view(view_class, previous)
                    entered.input_var.set(str(selected))
                    entered._open_input_folder()
                    with mock.patch(
                        f"{view_class.__module__}.filedialog.askdirectory", return_value=str(selected),
                    ):
                        browsed._browse_input()

                    self.assertEqual(browsed.input_var.get(), entered.input_var.get())
                    self.assertEqual(browsed.output_var.get(), entered.output_var.get())
                    self.assertEqual(browsed.status_var.get(), entered.status_var.get())

    def test_running_compilation_prevents_folder_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    view = self.make_view(view_class, folder)
                    view._running = True
                    view.set_input_folder = mock.Mock(wraps=view.set_input_folder)
                    with mock.patch(f"{view_class.__module__}.messagebox.showerror") as error:
                        view._open_input_folder()
                    view.set_input_folder.assert_not_called()
                    self.assertFalse(view.set_input_folder(folder))
                    error.assert_not_called()

    def test_compiler_validation_accepts_quoted_folder_and_workbook_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path(temporary).resolve()
            folder = previous / "measurement input"
            folder.mkdir()
            workbook = previous / "combined results.xlsx"
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    view = self.make_view(view_class, previous)
                    view.input_var.set(f'"{folder}"')
                    view.output_var.set(f'"{workbook}"')

                    self.assertEqual(view._validated_paths(), (folder, workbook))
                    self.assertEqual(view.input_var.get(), str(folder))
                    self.assertEqual(view._accepted_input_folder, str(folder))
                    self.assertEqual(view.output_var.get(), str(workbook))

    def test_empty_quoted_workbook_path_reports_a_missing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            for view_class in self.VIEW_CLASSES:
                for empty in ('""', "''", ' "  " '):
                    with self.subTest(view=view_class.__name__, empty=empty):
                        view = self.make_view(view_class, folder)
                        view.output_var.set(empty)
                        with mock.patch(f"{view_class.__module__}.messagebox.showerror") as error:
                            self.assertIsNone(view._validated_paths())
                        error.assert_called_once()

    def test_validation_uses_canonical_paths_for_input_and_new_workbook(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            nested = folder / "nested"
            nested.mkdir()
            alias = nested / ".."
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    view = self.make_view(view_class, folder)
                    view.input_var.set(str(alias))
                    view.output_var.set(str(alias / "combined results"))

                    self.assertEqual(view._validated_paths(), (folder, folder / "combined results.xlsx"))

                    self.assertEqual(view.input_var.get(), str(folder))
                    self.assertEqual(view._accepted_input_folder, str(folder))
                    self.assertEqual(view.output_var.get(), str(folder / "combined results.xlsx"))

    def test_browse_with_an_empty_quoted_output_uses_the_default_workbook_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    view = self.make_view(view_class, folder)
                    view.output_var.set('""')
                    with mock.patch(
                        f"{view_class.__module__}.filedialog.asksaveasfilename", return_value="",
                    ) as browse:
                        view._browse_output()
                    self.assertEqual(browse.call_args.kwargs["initialdir"], str(folder))
                    self.assertEqual(browse.call_args.kwargs["initialfile"], DEFAULT_OUTPUT_FILENAME)

    def test_equivalent_default_output_spelling_still_follows_input_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path(temporary).resolve()
            nested = previous / "nested"
            nested.mkdir()
            selected = previous / "new input"
            selected.mkdir()
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    view = self.make_view(view_class, previous)
                    view._default_output_path = view.output_var.get()
                    view.output_var.set(str(nested / ".." / "previous.xlsx"))

                    self.assertTrue(view.set_input_folder(selected))

                    self.assertEqual(Path(view.output_var.get()).parent, selected)
                    self.assertEqual(view._default_output_path, view.output_var.get())

    @unittest.skipUnless(os.name == "nt", "Windows short paths are platform-specific")
    def test_windows_short_paths_validate_and_track_the_same_default_workbook(self):
        import ctypes
        from ctypes import wintypes

        get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
        get_short_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short_path.restype = wintypes.DWORD
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve() / "Compiler Directory With Long Name"
            folder.mkdir()
            length = get_short_path(str(folder), None, 0)
            if not length:
                self.skipTest("This filesystem does not expose Windows short paths")
            buffer = ctypes.create_unicode_buffer(length)
            if not get_short_path(str(folder), buffer, length):
                self.skipTest("This filesystem does not expose Windows short paths")
            short_folder = Path(buffer.value)
            if short_folder == folder:
                self.skipTest("Short-name generation is disabled for this directory")
            selected = folder / "next input"
            selected.mkdir()
            for view_class in self.VIEW_CLASSES:
                with self.subTest(view=view_class.__name__):
                    view = self.make_view(view_class, folder)
                    view.input_var.set(str(short_folder))
                    view.output_var.set(str(short_folder / "combined.xlsx"))

                    self.assertEqual(view._validated_paths(), (folder, folder / "combined.xlsx"))
                    self.assertEqual(view.input_var.get(), str(folder))
                    self.assertEqual(view.output_var.get(), str(folder / "combined.xlsx"))

                    view._default_output_path = str(folder / "previous.xlsx")
                    view.output_var.set(str(short_folder / "previous.xlsx"))
                    self.assertTrue(view.set_input_folder(selected))
                    self.assertEqual(Path(view.output_var.get()).parent, selected)
                    self.assertEqual(view._default_output_path, view.output_var.get())


if __name__ == "__main__":
    unittest.main()
