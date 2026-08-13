import inspect
import unittest
from unittest import mock

from aidas.steps.step2_annotate import Step2Frame


class _CanvasStub:
    def __init__(self, manager=""):
        self.manager = manager
        self.pack_calls = []
        self.pack_forget_calls = 0

    def winfo_manager(self):
        return self.manager

    def pack(self, **options):
        self.manager = "pack"
        self.pack_calls.append(options)

    def pack_forget(self):
        self.manager = ""
        self.pack_forget_calls += 1


class _NotebookStub:
    def __init__(self, tabs=()):
        self._tabs = list(tabs)
        self.destroyed = False

    def tabs(self):
        return tuple(self._tabs)

    def nametowidget(self, name):
        return name

    def forget(self, tab):
        if tab in self._tabs:
            self._tabs.remove(tab)

    def destroy(self):
        self.destroyed = True


class _ControlStub:
    def __init__(self, state="normal"):
        self.state_value = state

    def cget(self, option):
        if option != "state":
            raise KeyError(option)
        return self.state_value

    def configure(self, **options):
        self.state_value = options["state"]


class Step2ResultLifecycleTests(unittest.TestCase):
    def test_sidebar_omits_redundant_swap_button(self):
        source = inspect.getsource(Step2Frame._build_controls)

        self.assertNotIn("flip_sides_button", source)
        self.assertNotIn("Swap left / right sides", source)

    def test_single_preview_stays_hidden_while_batch_folder_panel_is_open(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.single_image_canvas = _CanvasStub(manager="pack")
        batch_canvas = object()
        frame.image_canvas = batch_canvas
        frame.batch_results_notebook = None
        frame.batch_segmentation_panel = object()
        frame._active_batch_result_tab = "existing-tab"

        shown = frame._show_single_image_canvas()

        self.assertFalse(shown)
        self.assertEqual(frame.single_image_canvas.pack_forget_calls, 1)
        self.assertEqual(frame.single_image_canvas.pack_calls, [])
        self.assertIs(frame.image_canvas, batch_canvas)
        self.assertEqual(frame._active_batch_result_tab, "existing-tab")

    def test_single_preview_can_return_after_batch_folder_panel_closes(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.single_image_canvas = _CanvasStub()
        frame.image_canvas = frame.single_image_canvas
        frame.batch_results_notebook = None
        frame.batch_segmentation_panel = None

        shown = frame._show_single_image_canvas()

        self.assertTrue(shown)
        self.assertEqual(len(frame.single_image_canvas.pack_calls), 1)

    def test_pending_step1_crop_does_not_mutate_hidden_result_editor(self):
        frame = Step2Frame.__new__(Step2Frame)
        prepared = (object(), "new-crop.img", True)
        frame._pending_external_image = prepared
        frame.batch_segmentation_panel = object()
        frame.current_file = "active-result.img"
        frame.image_data = object()
        frame._input_analyze_template = {"width": 10}
        frame._source_was_8bit = False
        frame._show_image = mock.Mock()

        rendered = frame.render_pending_external_image()

        self.assertFalse(rendered)
        self.assertIs(frame._pending_external_image, prepared)
        self.assertEqual(frame.current_file, "active-result.img")
        self.assertEqual(frame._input_analyze_template, {"width": 10})
        self.assertFalse(frame._source_was_8bit)
        frame._show_image.assert_not_called()

    def _session_frame(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame._batch_result_canvases = [object()]
        frame._batch_result_tab_canvases = {"tab": object()}
        frame._batch_result_states = {"tab": {}}
        frame._active_batch_result_tab = "tab"
        frame._single_editor_state = {"image": object()}
        frame._clear_image_display = mock.Mock()
        frame._update_batch_ai_button_state = mock.Mock()
        return frame

    def test_finishing_last_result_restores_new_batch_action_only_after_clear(self):
        frame = self._session_frame()
        notebook = _NotebookStub()
        frame.batch_results_notebook = notebook

        frame._finish_batch_results_session(notebook, "Done")

        self.assertTrue(notebook.destroyed)
        self.assertIsNone(frame.batch_results_notebook)
        self.assertEqual(frame._batch_result_canvases, [])
        self.assertEqual(frame._batch_result_tab_canvases, {})
        self.assertEqual(frame._batch_result_states, {})
        self.assertIsNone(frame._active_batch_result_tab)
        self.assertIsNone(frame._single_editor_state)
        frame._clear_image_display.assert_called_once_with("Done")
        frame._update_batch_ai_button_state.assert_called_once_with()

    def test_last_result_keeps_new_batch_clickable_but_image_actions_disabled(self):
        frame = self._session_frame()
        notebook = _NotebookStub()
        frame.batch_results_notebook = notebook
        frame.batch_ai_button = _ControlStub("disabled")
        frame.saved_button = _ControlStub("normal")
        frame.save_all_button = _ControlStub("normal")
        frame.continue_to_step3_button = _ControlStub("normal")
        frame._segmenter_running = False
        frame.batch_segmentation_panel = None

        def clear_empty_editor(_message):
            for control in (
                frame.saved_button,
                frame.save_all_button,
                frame.continue_to_step3_button,
            ):
                control.configure(state="disabled")

        frame._clear_image_display = mock.Mock(side_effect=clear_empty_editor)
        frame._update_batch_ai_button_state = (
            lambda: Step2Frame._update_batch_ai_button_state(frame)
        )

        frame._finish_batch_results_session(notebook, "Done")

        self.assertEqual(frame.batch_ai_button.state_value, "normal")
        self.assertEqual(frame.saved_button.state_value, "disabled")
        self.assertEqual(frame.save_all_button.state_value, "disabled")
        self.assertEqual(frame.continue_to_step3_button.state_value, "disabled")

    def test_manual_close_last_tab_uses_shared_session_teardown(self):
        frame = self._session_frame()
        canvas = object()
        notebook = mock.Mock()
        notebook.tabs.return_value = ()
        frame._batch_result_canvases = [canvas]
        frame._batch_result_tab_canvases = {"tab": canvas}
        frame._batch_result_states = {"tab": {}}
        frame._confirm_close_batch_result_tab = mock.Mock(return_value=True)
        frame._sync_active_batch_result_state = mock.Mock()
        frame._finish_batch_results_session = mock.Mock()

        frame._close_batch_result_tab(notebook, "tab", canvas)

        frame._finish_batch_results_session.assert_called_once_with(
            notebook,
            "All batch result images saved or removed. No image is loaded.",
        )

    def test_save_all_close_last_tab_uses_shared_session_teardown(self):
        frame = self._session_frame()
        notebook = _NotebookStub()
        frame.batch_results_notebook = notebook
        frame._finish_batch_results_session = mock.Mock()

        frame._close_saved_batch_result_tabs(["tab"])

        frame._finish_batch_results_session.assert_called_once_with(
            notebook,
            "All batch result images were saved and closed.",
        )


if __name__ == "__main__":
    unittest.main()
