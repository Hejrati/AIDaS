from __future__ import annotations

from pathlib import Path
import queue
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from aidas.app import AIDaSApp
from aidas.steps.step5_compile import DEFAULT_OUTPUT_FILENAME, Step5Frame


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Step5CompileTests(unittest.TestCase):
    def test_handoff_folder_prefills_input_and_default_workbook(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame = object.__new__(Step5Frame)
            frame.input_var = _Var()
            frame.output_var = _Var()
            frame.status_var = _Var()
            frame._output_user_selected = False

            accepted = frame.set_input_folder(temp_dir)

            self.assertTrue(accepted)
            self.assertEqual(Path(frame.input_var.get()), Path(temp_dir))
            self.assertEqual(
                Path(frame.output_var.get()),
                Path(temp_dir) / DEFAULT_OUTPUT_FILENAME,
            )
            self.assertIn("Ready to compile", frame.status_var.get())

    def test_validation_adds_xlsx_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame = object.__new__(Step5Frame)
            frame.input_var = _Var(temp_dir)
            frame.output_var = _Var(str(Path(temp_dir) / "combined"))

            paths = frame._validated_paths()

            self.assertEqual(paths, (Path(temp_dir), Path(temp_dir) / "combined.xlsx"))
            self.assertEqual(Path(frame.output_var.get()).suffix, ".xlsx")

    def test_worker_uses_existing_compiler_service(self):
        frame = object.__new__(Step5Frame)
        frame._events = queue.Queue()
        root = Path("root")
        output = Path("compiled.xlsx")
        result = SimpleNamespace(output_path=output)

        with mock.patch(
            "aidas.services.step4_compiler.compile_step4_results",
            return_value=result,
        ) as compile_results:
            frame._compile_worker(root, output, True)

        compile_results.assert_called_once_with(
            root,
            output,
            include_fovea=True,
            log_callback=mock.ANY,
            progress_callback=mock.ANY,
        )
        self.assertEqual(frame._events.get_nowait(), ("success", result))

    def test_app_handoff_selects_step5_and_prefills_folder(self):
        app = AIDaSApp.__new__(AIDaSApp)
        step5 = mock.Mock()
        app.__dict__["step5"] = step5
        app.notebook = mock.Mock()
        app.update_idletasks = mock.Mock()
        folder = Path("measurement-root")

        app._on_step4_continue_to_step5(folder)

        app.notebook.select.assert_called_once_with(step5)
        app.update_idletasks.assert_called_once_with()
        step5.set_input_folder.assert_called_once_with(folder)


if __name__ == "__main__":
    unittest.main()
