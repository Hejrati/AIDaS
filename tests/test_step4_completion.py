from __future__ import annotations

import inspect
from pathlib import Path
import queue
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
    def test_build_stack_replaces_auto_detect_as_highlighted_action_when_ready(self):
        frame = object.__new__(Step4Frame)
        frame.rois = [mock.Mock(suffix="01"), mock.Mock(suffix="02")]
        frame.completed = {"01": object(), "02": object()}
        frame._stack_building = False
        frame._stack_build_complete = False
        frame.auto_detect_button = mock.Mock()
        frame.build_stacks_button = mock.Mock()

        frame._update_build_stack_button_state()

        frame.build_stacks_button.set_variant.assert_called_once_with("primary")
        frame.build_stacks_button.state.assert_called_once_with(["!disabled"])
        frame.auto_detect_button.set_variant.assert_called_once_with("secondary")

    def test_auto_detect_remains_highlighted_until_every_roi_is_complete(self):
        frame = object.__new__(Step4Frame)
        frame.rois = [mock.Mock(suffix="01"), mock.Mock(suffix="02")]
        frame.completed = {"01": object()}
        frame._stack_building = False
        frame._stack_build_complete = False
        frame.auto_detect_button = mock.Mock()
        frame.build_stacks_button = mock.Mock()

        frame._update_build_stack_button_state()

        frame.build_stacks_button.set_variant.assert_called_once_with("secondary")
        frame.build_stacks_button.state.assert_called_once_with(["disabled"])
        frame.auto_detect_button.set_variant.assert_called_once_with("primary")

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
        button_start = source.index("self.build_stacks_button = AppButton")
        button_end = source.index("self.build_stacks_button.pack", button_start)

        self.assertIn("before=self.sidebar", source[footer_start:button_start])
        self.assertIn("self.sidebar_footer", source[button_start:button_end])
        self.assertNotIn("roi_box", source[button_start:button_end])

    def test_step4_footer_actions_share_the_standard_rounded_button(self):
        source = inspect.getsource(Step4Frame._build_ui)
        footer_start = source.index("self.auto_detect_button = AppButton")
        footer_end = source.index("self.continue_to_step5_button.pack", footer_start)
        footer_source = source[footer_start:footer_end]

        self.assertEqual(footer_source.count("= AppButton("), 3)
        self.assertIn('"flat-color-icons--stack-of-photos.png"', footer_source)
        self.assertNotIn("self.build_stacks_button = action_button", footer_source)

    def test_auto_detect_uses_the_process_icon_and_explains_roi_21(self):
        source = inspect.getsource(Step4Frame._build_ui)

        self.assertIn('"flat-color-icons--process.png"', source)
        self.assertIn("image=self.auto_detect_button_icon", source)
        self.assertIn("ROI 21 remains manual", source)

    def test_stack_build_routes_all_success_notifications_through_one_finisher(self):
        source = inspect.getsource(Step4Frame._poll_stack_build_events)

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
        self.assertIn("self._completion_notice_button_action()", source)
        self.assertIn("command=button_command", source)
        self.assertNotIn("_step4_completion_label", source)

    def test_completion_notice_goes_to_next_tab_while_batch_is_incomplete(self):
        frame = object.__new__(Step4Frame)
        frame.batch_roi_notebook = object()
        frame.batch_roi_tab_states = {
            "complete": {"complete": True},
            "pending": {"complete": False},
        }

        text, command, enabled = frame._completion_notice_button_action()

        self.assertEqual(text, "Go to next tab (1/2 completed)")
        self.assertEqual(command, frame._go_to_next_batch_roi_tab)
        self.assertTrue(enabled)

    def test_completion_notice_goes_to_step5_when_batch_is_complete(self):
        frame = object.__new__(Step4Frame)
        frame.on_continue_to_step5 = mock.Mock()
        frame.batch_roi_notebook = object()
        frame.batch_roi_tab_states = {
            "first": {"complete": True},
            "last": {"complete": True},
        }

        text, command, enabled = frame._completion_notice_button_action()

        self.assertEqual(text, "Go to Step 5 (2/2 completed)")
        self.assertEqual(command, frame._continue_to_step5)
        self.assertTrue(enabled)

    def test_retained_completion_notice_refreshes_its_label_and_command(self):
        frame = object.__new__(Step4Frame)
        frame.on_continue_to_step5 = mock.Mock()
        frame.batch_roi_notebook = object()
        frame.batch_roi_tab_states = {
            "complete": {"complete": True},
            "pending": {"complete": False},
        }
        button = mock.Mock()
        notice = mock.Mock(_step4_continue_button=button)
        frame.plot_holder = mock.Mock(_step4_grid_notice=notice)

        frame._refresh_completion_notice_progress()

        button.configure.assert_called_once_with(
            text="Go to next tab (1/2 completed)",
            command=frame._go_to_next_batch_roi_tab,
        )
        button.state.assert_called_once_with(["!disabled"])

        button.reset_mock()
        frame.batch_roi_tab_states["pending"]["complete"] = True
        frame._refresh_completion_notice_progress()

        button.configure.assert_called_once_with(
            text="Go to Step 5 (2/2 completed)",
            command=frame._continue_to_step5,
        )
        button.state.assert_called_once_with(["!disabled"])

    def test_completion_step5_action_waits_for_every_batch_tab(self):
        frame = object.__new__(Step4Frame)
        frame.on_continue_to_step5 = mock.Mock()
        frame.batch_roi_notebook = object()
        frame.batch_roi_tab_states = {
            "complete": {"complete": True},
            "pending": {"complete": False},
        }

        self.assertFalse(frame._completion_step5_handoff_ready())
        self.assertEqual(frame._completion_progress_counts(), (1, 2))

        frame.batch_roi_tab_states["pending"]["complete"] = True

        self.assertTrue(frame._completion_step5_handoff_ready())
        self.assertEqual(frame._completion_progress_counts(), (2, 2))

    def test_main_step5_button_shows_the_same_batch_progress(self):
        frame = object.__new__(Step4Frame)
        frame.continue_to_step5_button = mock.Mock()
        frame.batch_roi_notebook = object()
        frame.batch_roi_tab_states = {
            "complete": {"complete": True},
            "pending": {"complete": False},
        }

        frame._update_continue_to_step5_button_state()

        frame.continue_to_step5_button.configure.assert_called_once_with(
            text="Go to Step 5 (1/2 completed)"
        )
        frame.continue_to_step5_button.state.assert_called_once_with(["disabled"])

        frame.continue_to_step5_button.reset_mock()
        frame.batch_roi_tab_states["pending"]["complete"] = True
        frame._update_continue_to_step5_button_state()

        frame.continue_to_step5_button.configure.assert_called_once_with(
            text="Go to Step 5 (2/2 completed)"
        )
        frame.continue_to_step5_button.state.assert_called_once_with(["!disabled"])

    def test_main_step5_button_hides_progress_when_no_file_tabs_are_open(self):
        frame = object.__new__(Step4Frame)
        frame.continue_to_step5_button = mock.Mock()
        frame.batch_roi_notebook = None
        frame.batch_roi_tab_states = {}
        frame.batch_roi_paths = []
        frame.batch_roi_index = -1
        frame.current_path = None
        frame.image = None

        frame._update_continue_to_step5_button_state()

        frame.continue_to_step5_button.configure.assert_called_once_with(
            text="Go to Step 5"
        )
        frame.continue_to_step5_button.state.assert_called_once_with(["disabled"])

    def test_stack_build_displays_wait_notice_before_creating_files(self):
        start_source = inspect.getsource(Step4Frame._build_stack_outputs)
        worker_source = inspect.getsource(Step4Frame._write_stack_output_files)

        notice_position = start_source.index("self._show_stack_building_notice(outdir)")
        worker_position = start_source.index("self._start_background_worker(worker)")
        self.assertLess(notice_position, worker_position)
        self.assertIn("Please wait", start_source)
        self.assertIn("outdir.mkdir", worker_source)

    def test_busy_notice_uses_a_running_back_and_forth_indicator(self):
        source = inspect.getsource(Step4Frame._show_grid_notice)

        self.assertIn('mode="indeterminate"', source)
        self.assertIn("indeterminate_speed=0.9", source)
        self.assertIn("progress.start()", source)
        self.assertIn("notice.grab_set()", source)

    def test_detection_and_file_saving_leave_tk_work_on_polling_callbacks(self):
        detection_source = inspect.getsource(Step4Frame._auto_detect_all_rois)
        stack_source = inspect.getsource(Step4Frame._build_stack_outputs)

        for source in (detection_source, stack_source):
            self.assertIn("self._start_background_worker(worker)", source)
            self.assertIn("self.after(", source)
            self.assertNotIn("self.update_idletasks()", source)

    def test_completed_stack_worker_is_finished_by_the_tk_poll_callback(self):
        frame = _completion_frame()
        events = queue.Queue()
        events.put(("done", None))
        frame._stack_build_events = events
        frame._stack_building = True
        frame._finish_stack_build = mock.Mock()
        frame._update_auto_detect_button_state = mock.Mock()

        frame._poll_stack_build_events(events, Path("output"))

        self.assertFalse(frame._stack_building)
        frame._finish_stack_build.assert_called_once_with(Path("output"))
        frame._update_build_stack_button_state.assert_called_once_with()
        frame._update_auto_detect_button_state.assert_called_once_with()

    def test_batch_stays_on_completed_tab_and_shows_navigation_popup(self):
        frame = _completion_frame()
        frame.batch_roi_notebook = object()
        frame._active_batch_roi_tab = "tab-one"
        frame.on_continue_to_step5 = mock.Mock()
        frame.batch_roi_tab_states = {
            "tab-one": {"complete": True},
            "tab-two": {"complete": False},
        }
        frame._mark_active_batch_roi_complete = mock.Mock()
        frame._select_next_incomplete_batch_roi_tab = mock.Mock(return_value=True)
        frame._show_processing_complete = mock.Mock()

        frame._finish_stack_build(Path("first-output"))

        frame._mark_active_batch_roi_complete.assert_called_once_with()
        frame._select_next_incomplete_batch_roi_tab.assert_not_called()
        frame._show_processing_complete.assert_not_called()
        frame._show_processed_grid_notice.assert_called_once_with(Path("first-output").resolve())
        self.assertIn("Go to next tab", frame.status_var.value)

    def test_completion_popup_next_tab_action_selects_only_after_click(self):
        frame = object.__new__(Step4Frame)
        frame._select_next_incomplete_batch_roi_tab = mock.Mock(return_value=True)
        frame._refresh_completion_notice_progress = mock.Mock()

        frame._go_to_next_batch_roi_tab()

        frame._select_next_incomplete_batch_roi_tab.assert_called_once_with()
        frame._refresh_completion_notice_progress.assert_not_called()

    def test_final_batch_tab_keeps_the_processed_message_in_the_grid(self):
        frame = _completion_frame()
        frame.batch_roi_notebook = object()
        frame._active_batch_roi_tab = "tab-last"
        frame.batch_roi_tab_states = {"tab-last": {"complete": True}}
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
