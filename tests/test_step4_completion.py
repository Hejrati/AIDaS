from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from aidas.steps.step4_analyze_isez import Step4Frame


class _VariableStub:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


def _completion_frame() -> Step4Frame:
    frame = object.__new__(Step4Frame)
    frame.status_var = _VariableStub()
    frame.batch_roi_notebook = None
    frame._active_batch_roi_tab = None
    frame.batch_roi_paths = []
    frame.batch_roi_index = -1
    return frame


class Step4CompletionTests(unittest.TestCase):
    def test_step4_module_import_does_not_require_openpyxl(self):
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import builtins\n"
            "real_import = builtins.__import__\n"
            "def blocked_import(name, *args, **kwargs):\n"
            "    if name == 'openpyxl' or name.startswith('openpyxl.'):\n"
            "        error = ModuleNotFoundError(\"No module named 'openpyxl'\")\n"
            "        error.name = 'openpyxl'\n"
            "        raise error\n"
            "    return real_import(name, *args, **kwargs)\n"
            "builtins.__import__ = blocked_import\n"
            "from aidas.steps.step4_analyze_isez import Step4Frame\n"
            "print(Step4Frame.__name__)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Step4Frame", result.stdout)

    def test_step4_no_longer_owns_the_measurement_compiler(self):
        source = inspect.getsource(Step4Frame._build_ui)

        self.assertNotIn("Compile measurements", source)
        self.assertNotIn("compiler_button", source)
        self.assertIn('text="Go to Step 5"', source)

    def test_step4_handoff_opens_step5_with_the_batch_root(self):
        frame = object.__new__(Step4Frame)
        frame.batch_roi_root = Path("measurement-root")
        frame.output_dir_var = mock.Mock()
        frame.input_dir_var = mock.Mock()
        frame.on_continue_to_step5 = mock.Mock()

        frame._continue_to_step5()

        frame.on_continue_to_step5.assert_called_once_with(Path("measurement-root"))

    def test_roi_table_stays_compact_enough_to_reveal_its_actions(self):
        source = inspect.getsource(Step4Frame._build_ui)

        self.assertEqual(Step4Frame.ROI_TABLE_VISIBLE_ROWS, 3)
        self.assertIn("height=self.ROI_TABLE_VISIBLE_ROWS", source)

    def test_build_stack_action_uses_a_reserved_sidebar_footer(self):
        source = inspect.getsource(Step4Frame._build_ui)
        footer_start = source.index("self.sidebar_footer = ctk.CTkFrame")
        button_start = source.index("self.build_stacks_button = action_button")
        button_end = source.index("self.build_stacks_button.pack", button_start)

        self.assertIn("before=self.sidebar", source[footer_start:button_start])
        self.assertIn("self.sidebar_footer", source[button_start:button_end])
        self.assertNotIn("roi_box", source[button_start:button_end])

    def test_stack_build_routes_all_success_notifications_through_one_finisher(self):
        source = inspect.getsource(Step4Frame._build_stack_outputs)

        self.assertIn("self._finish_stack_build(outdir)", source)
        self.assertNotIn("messagebox.showinfo", source)

    def test_standalone_build_shows_one_clear_completion_popup(self):
        frame = _completion_frame()

        with mock.patch(
            "aidas.steps.step4_analyze_isez.messagebox.showinfo"
        ) as showinfo:
            frame._finish_stack_build(Path("output"))

        showinfo.assert_called_once()
        title, message = showinfo.call_args.args
        self.assertEqual(title, "Processing Complete")
        self.assertIn("All processing is done.", message)
        self.assertIn("MAX_Stack.tif", message)
        self.assertIs(showinfo.call_args.kwargs["parent"], frame)

    def test_batch_advances_without_a_per_folder_popup(self):
        frame = _completion_frame()
        frame.batch_roi_notebook = object()
        frame._active_batch_roi_tab = "tab-one"
        frame._mark_active_batch_roi_complete = mock.Mock()
        frame._select_next_incomplete_batch_roi_tab = mock.Mock(return_value=True)
        frame._show_processing_complete = mock.Mock()

        frame._finish_stack_build(Path("first-output"))

        frame._mark_active_batch_roi_complete.assert_called_once_with()
        frame._select_next_incomplete_batch_roi_tab.assert_called_once_with()
        frame._show_processing_complete.assert_not_called()

    def test_final_batch_tab_shows_only_the_terminal_popup(self):
        frame = _completion_frame()
        frame.batch_roi_notebook = object()
        frame._active_batch_roi_tab = "tab-last"
        frame._mark_active_batch_roi_complete = mock.Mock()
        frame._select_next_incomplete_batch_roi_tab = mock.Mock(return_value=False)
        frame._show_processing_complete = mock.Mock()

        frame._finish_stack_build(Path("last-output"))

        frame._show_processing_complete.assert_called_once_with(
            "Every selected Step 4 folder is complete."
        )
        self.assertIn("Processing complete", frame.status_var.value)

    def test_legacy_batch_keeps_advancing_without_an_intermediate_popup(self):
        frame = _completion_frame()
        frame.batch_roi_paths = [Path("one"), Path("two")]
        frame.batch_roi_index = 0
        frame._load_next_batch_roi = mock.Mock()
        frame._show_processing_complete = mock.Mock()

        frame._finish_stack_build(Path("one"))

        frame._load_next_batch_roi.assert_called_once_with()
        frame._show_processing_complete.assert_not_called()

    def test_legacy_batch_notifies_once_after_its_last_folder(self):
        frame = _completion_frame()
        frame.batch_roi_paths = [Path("only")]
        frame.batch_roi_index = 0
        frame.batch_roi_skipped = 0
        frame._show_processing_complete = mock.Mock()

        frame._load_next_batch_roi()

        frame._show_processing_complete.assert_called_once_with(
            "Every selected Step 4 folder in this batch is complete."
        )
        self.assertIn("Processing complete", frame.status_var.value)


if __name__ == "__main__":
    unittest.main()
