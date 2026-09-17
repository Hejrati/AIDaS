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
    frame._stack_building = False
    frame._stack_build_complete = False
    frame._show_processed_grid_notice = mock.Mock()
    frame._update_build_stack_button_state = mock.Mock()
    frame._update_continue_to_step5_button_state = mock.Mock()
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

    def test_roi_navigation_places_described_erase_action_between_previous_and_next(self):
        source = inspect.getsource(Step4Frame._build_ui)
        previous = source.index("self.previous_roi_button = action_button")
        erase = source.index("self.clear_button = action_button")
        next_button = source.index("self.next_roi_button = action_button")

        self.assertLess(previous, erase)
        self.assertLess(erase, next_button)
        self.assertIn('"Erase"', source[erase:next_button])
        self.assertIn("Erase the current ROI selection and saved result", source[erase:next_button])
        self.assertIn("column=1", source[erase:next_button])
        self.assertIn("column=2", source[next_button:])

    def test_main_roi_controls_do_not_show_start_end_entries_or_confirm_button(self):
        source = inspect.getsource(Step4Frame._build_ui)

        self.assertNotIn("self.start_entry", source)
        self.assertNotIn("self.end_entry", source)
        self.assertNotIn("self.apply_button", source)
        self.assertNotIn("control_row", source)

    def test_build_stack_action_uses_a_reserved_sidebar_footer(self):
        source = inspect.getsource(Step4Frame._build_ui)
        footer_start = source.index("self.sidebar_footer = ctk.CTkFrame")
        button_start = source.index("self.build_stacks_button = action_button")
        button_end = source.index("self.build_stacks_button.pack", button_start)

        self.assertIn("before=self.sidebar", source[footer_start:button_start])
        self.assertIn("self.sidebar_footer", source[button_start:button_end])
        self.assertNotIn("roi_box", source[button_start:button_end])

    def test_auto_detect_uses_the_process_icon_and_explains_roi_21(self):
        source = inspect.getsource(Step4Frame._build_ui)

        self.assertIn('"flat-color-icons--process.png"', source)
        self.assertIn("image=self.auto_detect_button_icon", source)
        self.assertIn("ROI 21 remains manual", source)

    def test_stack_build_routes_all_success_notifications_through_one_finisher(self):
        source = inspect.getsource(Step4Frame._build_stack_outputs)

        self.assertIn("self._finish_stack_build(outdir)", source)
        self.assertNotIn("messagebox.showinfo", source)

    def test_standalone_build_shows_results_actions_in_the_grid(self):
        frame = _completion_frame()
        frame._show_processing_complete = mock.Mock()

        frame._finish_stack_build(Path("output"))

        output_dir = Path("output").resolve()
        frame._show_processing_complete.assert_called_once_with(
            mock.ANY,
            output_dir=output_dir,
        )
        self.assertEqual(frame._last_results_dir, output_dir)
        self.assertTrue(frame._stack_build_complete)

    def test_open_results_action_opens_the_exact_results_directory(self):
        frame = _completion_frame()

        with mock.patch("aidas.steps.step4_analyze_isez._open_directory") as open_directory:
            frame._open_results_directory(Path("subject-results"))

        open_directory.assert_called_once_with(Path("subject-results"))

    def test_batch_results_folder_is_the_output_folder_not_batch_root(self):
        frame = _completion_frame()
        frame.batch_roi_root = Path("batch-root")
        frame.batch_roi_notebook = object()
        frame._active_batch_roi_tab = "tab-one"
        frame._mark_active_batch_roi_complete = mock.Mock()
        frame._select_next_incomplete_batch_roi_tab = mock.Mock(return_value=True)

        frame._finish_stack_build(Path("batch-root") / "subject-results")

        expected = (Path("batch-root") / "subject-results").resolve()
        self.assertEqual(frame._last_results_dir, expected)
        frame._show_processed_grid_notice.assert_called_once_with(expected)

    def test_grid_notice_has_open_folder_and_restart_actions(self):
        source = inspect.getsource(Step4Frame._show_grid_notice)

        self.assertIn('text="Open results folder"', source)
        self.assertIn('text="Restart this file"', source)

    def test_stack_build_displays_wait_notice_before_creating_files(self):
        source = inspect.getsource(Step4Frame._build_stack_outputs)

        notice_position = source.index("self._show_stack_building_notice(outdir)")
        save_position = source.index("outdir.mkdir")
        self.assertLess(notice_position, save_position)
        self.assertIn("Please wait", source)

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
        frame._show_processed_grid_notice.assert_called_once_with(Path("first-output").resolve())

    def test_final_batch_tab_keeps_the_processed_message_in_the_grid(self):
        frame = _completion_frame()
        frame.batch_roi_notebook = object()
        frame._active_batch_roi_tab = "tab-last"
        frame._mark_active_batch_roi_complete = mock.Mock()
        frame._select_next_incomplete_batch_roi_tab = mock.Mock(return_value=False)
        frame._show_processing_complete = mock.Mock()

        frame._finish_stack_build(Path("last-output"))

        frame._show_processing_complete.assert_not_called()
        frame._show_processed_grid_notice.assert_called_once_with(Path("last-output").resolve())
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
            "Every selected Step 4 folder in this batch is complete.",
            output_dir=None,
        )
        self.assertIn("Processing complete", frame.status_var.value)


if __name__ == "__main__":
    unittest.main()
