from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aidas.steps.step4_compiler_dialog import Step4CompilerDialog
from aidas.steps.step5_compile import Step5Frame


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


if __name__ == "__main__":
    unittest.main()
