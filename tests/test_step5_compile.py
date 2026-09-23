from __future__ import annotations

import inspect
from pathlib import Path
import queue
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from aidas.app import AIDaSApp
from aidas.steps.step5_compile import DEFAULT_OUTPUT_FILENAME, Step5Frame
from aidas.utils.ui_utils import _compile_icon_image


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Step5CompileTests(unittest.TestCase):
    def test_compile_icon_uses_the_vaadin_grid_shape(self):
        icon = _compile_icon_image("#FFFFFF", 64)

        self.assertEqual(icon.size, (64, 64))
        self.assertEqual(icon.getpixel((0, 0))[3], 0)
        self.assertEqual(icon.getpixel((4, 8)), (255, 255, 255, 255))
        self.assertEqual(icon.getpixel((54, 2)), (255, 255, 255, 255))
        self.assertEqual(icon.getpixel((47, 0))[3], 0)

    def test_step5_compile_button_uses_dedicated_compile_icon(self):
        source = inspect.getsource(Step5Frame._build_ui)

        self.assertIn("load_compile_ctk_icon", source)
        self.assertNotIn("flat-color-icons--process.png", source)

    def test_success_notifies_app_after_the_workbook_is_saved(self):
        frame = object.__new__(Step5Frame)
        frame.progress = mock.Mock()
        frame.progress.cget.return_value = 5
        frame.output_var = _Var("requested.xlsx")
        frame.status_var = _Var()
        frame._set_running = mock.Mock()
        frame._append_log = mock.Mock()
        frame.on_compilation_complete = mock.Mock()
        result = SimpleNamespace(output_path=Path("compiled.xlsx"))

        with mock.patch("aidas.steps.step5_compile.messagebox.showinfo"):
            frame._compilation_succeeded(result)

        frame.on_compilation_complete.assert_called_once_with(Path("compiled.xlsx"))

    def test_app_progress_only_advances_and_marks_final_success_complete(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.progress_strip = mock.Mock()
        app._workflow_completed_steps = 3

        app._advance_workflow_progress(1)
        app._on_step5_compilation_complete(Path("compiled.xlsx"))

        self.assertEqual(app._workflow_completed_steps, 5)
        self.assertEqual(
            app.progress_strip.set_completed_steps.call_args_list,
            [mock.call(3), mock.call(5)],
        )

    def test_app_progress_animates_only_when_a_milestone_advances(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.progress_strip = mock.Mock()
        app._workflow_completed_steps = 0
        app._show_workflow_completion_popup = mock.Mock()

        app._advance_workflow_progress(1)
        app._advance_workflow_progress(1)

        app.progress_strip.animate_completion.assert_called_once_with(0)
        app._show_workflow_completion_popup.assert_called_once_with(0)

    def test_workflow_handoff_uses_a_large_workspace_completion_popup(self):
        source = inspect.getsource(AIDaSApp._show_workflow_completion_popup)

        self.assertIn("width=500", source)
        self.assertIn("height=285", source)
        self.assertIn("Step {completed_index + 1} complete", source)
        self.assertNotIn("Progress bar updated above", source)
        self.assertNotIn("Next: Step", source)
        self.assertIn("content.place(relx=0.5, rely=0.5, anchor=\"center\")", source)
        self.assertIn('"certificate-badge-iconify.png"', source)
        self.assertIn("badge_size = 112", source)
        self.assertIn("badge_canvas = tk.Canvas", source)
        self.assertIn("badge_canvas.create_text", source)
        self.assertIn("text=str(completed_index + 1)", source)
        self.assertIn("ProgressCircle.NUMBER_FONT_FAMILY", source)
        self.assertIn("ProgressCircle.default_number_font_size(badge_size)", source)
        self.assertIn("ProgressCircle.NUMBER_FONT_WEIGHT", source)
        self.assertNotIn("ctk.CTkLabel(\n            badge_holder", source)
        self.assertNotIn("number_disk", source)
        self.assertNotIn("frames =", source)
        self.assertIn("self.after(1500, close_popup)", source)

    def test_completion_popup_iconify_badge_asset_is_packaged(self):
        asset = Path(__file__).parents[1] / "assets" / "certificate-badge-iconify.png"

        self.assertTrue(asset.is_file())
        self.assertGreater(asset.stat().st_size, 0)

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
