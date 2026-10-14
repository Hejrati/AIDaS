from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk
import numpy as np

from aidas.canvas.image_canvas import ImageCanvas
from aidas.core.config import Config
from aidas.steps.step2_annotate import BOUNDARY_NAMES, FOVEA_BOUNDARY_NAME, Step2Frame
from aidas.ui.theme import (
    COLOR_PAIRS, apply_appearance_mode, get_interface_mode, resolve_color, set_interface_mode,
)


class _Var:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class ManualAnnotationTests(unittest.TestCase):
    def _frame(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.image_data = np.arange(80, dtype=np.int16).reshape(8, 10)
        frame.image_canvas = mock.Mock()
        frame.image_canvas.get_active_line.return_value = []
        frame._segmenter_running = False
        frame._batch_fovea_picker_active = False
        frame.active_boundary = BOUNDARY_NAMES[0]
        frame.fovea_x = 5
        frame.boundary_traces = {FOVEA_BOUNDARY_NAME: frame._vertical_line_trace(5, 8)}
        frame.boundary_order = [FOVEA_BOUNDARY_NAME]
        frame.boundary_completion_vars = {name: _Var(False) for name in BOUNDARY_NAMES}
        frame.status_var = _Var()
        frame.active_trace_var = _Var()
        frame.fovea_x_entry_var = _Var(5)
        frame.fovea_line_var = _Var()
        frame.boundary_workflow_status_var = _Var()
        for name in ('_refresh_boundary_lists', '_refresh_trace_list', '_select_trace_by_name',
                     '_update_boundary_action_buttons', '_sync_boundary_canvas_state',
                     '_update_boundary_progress_bar', '_set_fovea_controls_enabled',
                     '_sync_active_batch_result_state', '_save_marked_images'):
            setattr(frame, name, mock.Mock())
        return frame

    def test_finishing_a_line_stops_drawing_and_preserves_its_points(self):
        frame = self._frame()
        name = frame.active_boundary
        frame.image_canvas.get_active_line.return_value = [(0, 2), (9, 4)]
        frame._start_boundary_for_name = mock.Mock()

        frame._finish_boundary()

        self.assertEqual(frame.boundary_traces[name]['points'], [(0, 2), (9, 4)])
        self.assertTrue(frame.boundary_completion_vars[name].get())
        self.assertEqual(frame.active_boundary, name)
        frame._start_boundary_for_name.assert_not_called()
        frame._sync_boundary_canvas_state.assert_called()
        frame._refresh_boundary_lists.assert_called_with(select_completed_name=name)

    def test_delete_uses_the_visible_selection_and_preserves_other_lines(self):
        frame = self._frame()
        first, second = BOUNDARY_NAMES[:2]
        traces, _ = frame._boundary_traces_from_rows(np.ones((6, 10)), (8, 10))
        frame.boundary_traces.update(traces)
        frame.boundary_order.extend(BOUNDARY_NAMES)
        frame._set_completion_from_traces()
        other_trace = frame._copy_trace_dict({second: traces[second]})

        frame._delete_selected_boundary()

        self.assertNotIn(first, frame.boundary_traces)
        self.assertFalse(frame.boundary_completion_vars[first].get())
        self.assertEqual(frame.boundary_traces[second], other_trace[second])
        self.assertIn(FOVEA_BOUNDARY_NAME, frame.boundary_traces)
        self.assertEqual(frame.active_boundary, first)

    def test_revert_all_returns_every_line_to_incomplete_and_keeps_fovea(self):
        frame = self._frame()
        traces, order = frame._boundary_traces_from_rows(np.ones((6, 10)), (8, 10))
        frame.boundary_traces.update(traces)
        frame.boundary_order.extend(order)
        frame._set_completion_from_traces()
        fovea = frame._copy_trace_dict({FOVEA_BOUNDARY_NAME: frame.boundary_traces[FOVEA_BOUNDARY_NAME]})

        with mock.patch('aidas.steps.step2_annotate.messagebox.askyesno', return_value=True):
            frame._clear_all_traces()
        self.assertEqual(list(frame.boundary_traces), [FOVEA_BOUNDARY_NAME])
        self.assertEqual(frame.boundary_traces, fovea)
        self.assertEqual(frame._incomplete_boundary_names(), BOUNDARY_NAMES)
        self.assertEqual(frame.fovea_x, 5)

    def test_destructive_controls_do_nothing_while_ai_is_running(self):
        frame = self._frame()
        frame._segmenter_running = True
        original = frame._annotation_snapshot()
        with mock.patch('aidas.steps.step2_annotate.messagebox.askyesno') as confirm:
            frame._delete_selected_boundary()
            frame._clear_all_traces()
            frame._finish_boundary()
        self.assertEqual(frame._annotation_snapshot(), original)
        confirm.assert_not_called()

    def test_single_line_ai_requests_only_the_selected_incomplete_boundary(self):
        frame = self._frame()
        frame._segment_current_image = mock.Mock()
        frame._segment_selected_boundary()
        frame._segment_current_image.assert_called_once_with(boundary_names=[BOUNDARY_NAMES[0]])
        frame.boundary_completion_vars[BOUNDARY_NAMES[0]].set(True)
        frame._segment_current_image.reset_mock()
        frame._segment_selected_boundary()
        frame._segment_current_image.assert_not_called()

    def test_ai_all_button_disables_when_complete_and_reenables_after_reverting(self):
        frame = self._frame()
        del frame._update_boundary_action_buttons
        button = mock.Mock()
        frame.ai_all_lines_btn = button
        frame._set_control_enabled = mock.Mock()
        frame._update_continue_to_step3_button_state = mock.Mock()

        def enabled():
            frame._update_boundary_action_buttons()
            return next(call.args[1] for call in frame._set_control_enabled.call_args_list
                        if call.args[0] is button)

        self.assertTrue(enabled())
        traces, order = frame._boundary_traces_from_rows(np.ones((6, 10)), (8, 10))
        frame.boundary_traces.update(traces)
        frame.boundary_order.extend(order)
        frame._set_completion_from_traces()
        frame._set_control_enabled.reset_mock()
        self.assertFalse(enabled())

        frame._delete_selected_boundary()
        frame._set_control_enabled.reset_mock()
        self.assertTrue(enabled())

        frame._segmenter_running = True
        frame._set_control_enabled.reset_mock()
        self.assertFalse(enabled())

    def test_ai_all_does_not_start_inference_when_every_boundary_exists(self):
        frame = self._frame()
        traces, _ = frame._boundary_traces_from_rows(np.ones((6, 10)), (8, 10))
        frame.boundary_traces.update(traces)
        with mock.patch('aidas.steps.step2_annotate.threading.Thread') as thread:
            frame._segment_current_image()
        thread.assert_not_called()

    def test_current_ai_result_updates_its_tab_without_replacing_other_tabs(self):
        frame = self._frame()
        frame._set_segmentation_running = mock.Mock()
        frame._set_runtime_ai_device_status = mock.Mock()
        frame._active_batch_result_tab = 'other'
        canvas = mock.Mock()
        target = {'canvas': canvas}
        other = {'traces': {'keep': 'unchanged'}}
        frame._batch_result_states = {'target': target, 'other': other}
        frame._batch_result_tab_canvases = {}
        frame._load_editor_state = mock.Mock()
        traces, _ = frame._boundary_traces_from_rows(np.ones((6, 10)), (8, 10))
        source = {'tab_key': 'target', 'fovea_x': 5, 'image': frame.image_data}

        frame._on_current_image_segmentation_done({'results': [{'traces': traces}]}, source)

        self.assertEqual(len(target['traces']), 7)
        self.assertEqual(other, {'traces': {'keep': 'unchanged'}})
        frame._load_editor_state.assert_not_called()
        canvas.set_line_overlays.assert_called_once()

    def test_single_line_ai_preserves_every_other_manual_boundary(self):
        frame = self._frame()
        frame._set_segmentation_running = mock.Mock()
        frame._set_runtime_ai_device_status = mock.Mock()
        frame._active_batch_result_tab = 'other'
        old, _ = frame._boundary_traces_from_rows(np.ones((6, 10)), (8, 10))
        predicted, _ = frame._boundary_traces_from_rows(np.full((6, 10), 3), (8, 10))
        target = {'traces': old, 'canvas': mock.Mock()}
        frame._batch_result_states = {'target': target}
        frame._batch_result_tab_canvases = {}
        selected = BOUNDARY_NAMES[0]
        request = {'tab_key': 'target', 'fovea_x': 5, 'image': frame.image_data, 'boundary_names': [selected]}
        frame._on_current_image_segmentation_done({'results': [{'traces': predicted}]}, request)
        self.assertEqual(target['traces'][selected], predicted[selected])
        for name in BOUNDARY_NAMES[1:]:
            self.assertEqual(target['traces'][name], old[name])

    def test_repeated_single_line_ai_replaces_only_that_boundary_without_duplicates(self):
        frame = self._frame()
        frame._set_segmentation_running = mock.Mock()
        frame._set_runtime_ai_device_status = mock.Mock()
        frame._active_batch_result_tab = 'other'
        old, _ = frame._boundary_traces_from_rows(np.ones((6, 10)), (8, 10))
        selected = BOUNDARY_NAMES[2]
        del old[selected]
        old[BOUNDARY_NAMES[0]]['color'] = '#abcdef'
        old[BOUNDARY_NAMES[0]]['points'] = [(1, 2), (6, 4), (9, 3)]
        old[BOUNDARY_NAMES[0]]['handle_indices'] = [0, 1, 2]
        old[FOVEA_BOUNDARY_NAME] = frame._vertical_line_trace(4, 8)
        original = frame._copy_trace_dict(old)
        target = {'traces': old, 'canvas': mock.Mock(), 'image': frame.image_data,
                  'fovea_x': 4, 'order': list(old) + [BOUNDARY_NAMES[0]]}
        frame._batch_result_states = {'target': target}
        frame._batch_result_tab_canvases = {}
        request = {'tab_key': 'target', 'fovea_x': 5, 'image': frame.image_data.copy(),
                   'source_image': frame.image_data, 'boundary_names': [selected]}
        predicted, _ = frame._boundary_traces_from_rows(np.full((6, 10), 3), (8, 10))

        for _ in range(2):
            frame._on_current_image_segmentation_done({'results': [{'traces': predicted}]}, request)
            self.assertEqual(target['traces'][selected], predicted[selected])
            for name, trace in original.items():
                self.assertEqual(target['traces'][name], trace)
            self.assertEqual(len(target['order']), 7)
            self.assertEqual(len(set(target['order'])), 7)
            overlays = target['canvas'].set_line_overlays.call_args.args[0]
            self.assertEqual(len(overlays), 6)
            self.assertEqual({item['label'] for item in overlays}, set(BOUNDARY_NAMES))

    def test_single_image_ai_keeps_source_image_and_view_without_selecting_edit_handles(self):
        frame = self._frame()
        frame.current_file = 'Light.img'
        frame.single_image_canvas = frame.image_canvas
        frame._active_batch_result_tab = None
        frame._set_segmentation_running = mock.Mock()
        frame._set_runtime_ai_device_status = mock.Mock()
        frame._load_editor_state = mock.Mock()
        frame._set_active_boundary_target = mock.Mock()
        predicted, _ = frame._boundary_traces_from_rows(np.full((6, 10), 3), (8, 10))
        request = {**frame._annotation_snapshot(), 'input': frame.current_file,
                   'source_file': frame.current_file, 'image': frame.image_data.copy(),
                   'source_image': frame.image_data, 'boundary_names': [BOUNDARY_NAMES[0]]}

        frame._on_current_image_segmentation_done({'results': [{'traces': predicted}]}, request)

        loaded_state = frame._load_editor_state.call_args.args[0]
        self.assertIs(loaded_state['image'], frame.image_data)
        self.assertFalse(frame._load_editor_state.call_args.kwargs['fit_view'])
        frame._set_active_boundary_target.assert_not_called()

    def test_overlay_rebuild_draws_each_saved_boundary_once(self):
        frame = self._frame()
        traces, _ = frame._boundary_traces_from_rows(np.ones((6, 10)), (8, 10))
        frame.boundary_traces.update(traces)
        frame.boundary_order += BOUNDARY_NAMES + [BOUNDARY_NAMES[0], FOVEA_BOUNDARY_NAME]
        frame._rebuild_saved_overlays()
        overlays = frame.image_canvas.set_line_overlays.call_args.args[0]
        self.assertEqual([item['label'] for item in overlays], BOUNDARY_NAMES)

    def test_edited_handles_and_line_color_survive_canvas_and_state_refreshes(self):
        frame = self._frame()
        name = BOUNDARY_NAMES[0]
        frame.boundary_traces[name] = {'color': '#abcdef'}
        frame.boundary_order.append(name)
        frame.image_canvas.get_line_handle_indices.return_value = [0, 2, 3]
        points = [(0, 2), (2.5, 3), (4, 4), (9, 2)]
        original_fovea = frame._copy_trace_dict(frame.boundary_traces)[FOVEA_BOUNDARY_NAME]
        frame._on_saved_line_edited(name, points)
        self.assertEqual(frame.boundary_traces[name]['color'], '#abcdef')
        self.assertEqual(frame.boundary_traces[name]['handle_indices'], [0, 2, 3])
        self.assertEqual(frame.boundary_traces[FOVEA_BOUNDARY_NAME], original_fovea)
        overlays = frame.image_canvas.set_line_overlays.call_args.args[0]
        self.assertEqual(overlays[0]['handle_indices'], [0, 2, 3])
        copied = frame._annotation_snapshot()['traces']
        copied[name]['handle_indices'].append(1)
        self.assertEqual(frame.boundary_traces[name]['handle_indices'], [0, 2, 3])

    def test_ai_failure_restores_an_unfinished_manual_line(self):
        frame = self._frame()
        frame._set_segmentation_running = mock.Mock()
        frame._set_active_boundary_target = mock.Mock()
        frame._active_batch_result_tab = 'target'
        request = {'tab_key': 'target', 'source_image': frame.image_data,
                   'active_boundary': BOUNDARY_NAMES[0], 'unfinished_line': [(1, 2), (8, 3)]}
        with mock.patch('aidas.steps.step2_annotate.messagebox.showerror'):
            frame._on_current_image_segmentation_done({'exception': 'model error'}, request)
        frame._set_active_boundary_target.assert_called_once_with(BOUNDARY_NAMES[0])
        frame.image_canvas.set_active_line.assert_called_once_with([(1, 2), (8, 3)])

    def test_ai_failure_keeps_existing_annotations(self):
        frame = self._frame()
        original = frame._annotation_snapshot()
        frame._set_segmentation_running = mock.Mock()
        with mock.patch('aidas.steps.step2_annotate.messagebox.showerror') as error:
            frame._on_current_image_segmentation_done({'exception': 'model error'}, {})
        frame._set_segmentation_running.assert_called_once_with(False)
        error.assert_called_once()
        self.assertEqual(frame._annotation_snapshot(), original)

    def test_current_ai_worker_uses_the_displayed_image_and_existing_orientation_conversion(self):
        frame = self._frame()
        for name in ('_aidas_worker_command', '_aidas_worker_env', '_hidden_subprocess_kwargs',
                     '_set_segmenter_progress_value', '_append_segmenter_log', '_set_runtime_ai_device_status',
                     '_write_segmenter_log_file', '_on_current_image_segmentation_done', '_read_image_for_annotation'):
            setattr(frame, name, mock.Mock())
        frame.after = lambda _delay, callback: callback()
        image = frame.image_data.copy()
        state = {'image': image, 'template': object()}
        path = 'Light.img'
        client = mock.Mock()
        client.startup_result = {}
        client.predict.return_value = {'boundaries': np.ones((6, 10)), 'device': 'CPU'}
        with mock.patch('aidas.steps.step2_annotate.AIWorkerClient') as worker_class:
            worker = worker_class.return_value
            worker.startup_result = client.startup_result
            worker.predict = client.predict
            frame._run_aidas_batch_segmenter_worker(
                [path], 'model.onnx', 'auto', 0, 2,
                {frame._image_pair_key(path): 5}, {}, current_image_state=state,
            )
        np.testing.assert_array_equal(client.predict.call_args.args[0], np.flipud(image))
        frame._read_image_for_annotation.assert_not_called()
        result, supplied_state = frame._on_current_image_segmentation_done.call_args.args
        self.assertIs(supplied_state, state)
        self.assertEqual(len(result['results']), 1)
        self.assertFalse(result['failures'])
        np.testing.assert_array_equal(image, frame.image_data)


class AnnotationTutorialTests(unittest.TestCase):
    def test_canvas_banner_waits_for_visible_image_and_remembers_dismissal(self):
        root = None
        original_widget_scaling = ctk.ScalingTracker.widget_scaling
        original_interface = get_interface_mode()
        original_appearance = ctk.get_appearance_mode()
        original_source = ctk.AppearanceModeTracker.appearance_mode_set_by
        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(
            Config, "CONFIG_DIR", Path(temporary_directory)
        ), mock.patch.object(
            Config, "CONFIG_FILE", Path(temporary_directory) / "preferences.json"
        ), mock.patch.object(Step2Frame, "_start_ai_device_probe"):
            try:
                set_interface_mode("Modern", redraw=False)
                ctk.set_appearance_mode("Light")
                ctk.set_widget_scaling(1)
                try:
                    root = ctk.CTk()
                    root.withdraw()
                    root.attributes("-alpha", 0)
                    root.geometry("1024x680+10000+10000")
                except tk.TclError as exc:
                    self.skipTest(f"A transparent Tk test window is unavailable: {exc}")

                root.deiconify()
                apply_appearance_mode("Light", root=root, style=ttk.Style(root))
                image = np.zeros((32, 96), dtype=np.int16)

                def finish_current_segmentation(frame):
                    frame._ensure_fovea_line(48)
                    state = frame._capture_current_editor_state()
                    state.update(
                        source_image=frame.image_data, source_file=frame.current_file,
                        boundary_names=[BOUNDARY_NAMES[0]],
                    )
                    traces, _order = frame._boundary_traces_from_rows(
                        np.ones((6, 96)), image.shape,
                    )
                    frame._on_current_image_segmentation_done(
                        {"results": [{"traces": traces}]}, state,
                    )

                def show_segmented_image(frame, path):
                    frame._show_image(image, path)
                    finish_current_segmentation(frame)

                def visible_tutorial(frame):
                    banner = frame.annotation_tutorial
                    return banner is not None and banner.winfo_exists() and banner.winfo_ismapped()

                def canvas_geometry(frame):
                    canvas = frame.image_canvas.canvas
                    return (canvas.winfo_rootx(), canvas.winfo_rooty(),
                            canvas.winfo_width(), canvas.winfo_height())

                def assert_guide_layout(frame):
                    banner = frame.annotation_tutorial
                    canvas = frame.image_canvas.canvas
                    scale = frame._get_widget_scaling()
                    expected_width = canvas.winfo_width()
                    self.assertIs(banner.master, canvas)
                    self.assertAlmostEqual(banner.winfo_width(), expected_width, delta=1)
                    self.assertAlmostEqual(banner.winfo_rootx(), canvas.winfo_rootx(), delta=1)
                    self.assertAlmostEqual(banner.winfo_rooty(),
                                           canvas.winfo_rooty(), delta=1)
                    self.assertLessEqual(banner.winfo_rootx() + banner.winfo_width(),
                                         canvas.winfo_rootx() + canvas.winfo_width() + 1)
                    self.assertLessEqual(banner.winfo_rooty() + banner.winfo_height(),
                                         canvas.winfo_rooty() + canvas.winfo_height())
                    logical_width = expected_width / scale
                    columns = 4 if logical_width >= 940 else 2 if logical_width >= 520 else 1
                    headings = list(frame.annotation_tutorial_heading_labels.values())
                    for index, heading in enumerate(headings):
                        row, column = divmod(index, columns)
                        self.assertAlmostEqual(heading.winfo_rooty(),
                                               headings[row * columns].winfo_rooty(), delta=1)
                        if column:
                            self.assertGreater(heading.winfo_rootx(), headings[index - 1].winfo_rootx())
                        elif row:
                            self.assertAlmostEqual(heading.winfo_rootx(), headings[0].winfo_rootx(), delta=1)
                            self.assertGreater(heading.winfo_rooty(), headings[index - columns].winfo_rooty())
                    content = (
                        frame.annotation_tutorial_title,
                        *frame.annotation_tutorial_heading_labels.values(),
                        *frame.annotation_tutorial_labels.values(),
                        frame.annotation_tutorial_info_icon,
                        frame.annotation_tutorial_opt_out_btn,
                        frame.annotation_tutorial_ok_btn,
                    )
                    for widget in content:
                        self.assertTrue(widget.winfo_viewable())
                        self.assertGreaterEqual(widget.winfo_rootx(), banner.winfo_rootx())
                        self.assertGreaterEqual(widget.winfo_rooty(), banner.winfo_rooty())
                        self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(),
                                             banner.winfo_rootx() + banner.winfo_width())
                        self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                             banner.winfo_rooty() + banner.winfo_height())
                    for label in frame.annotation_tutorial_labels.values():
                        self.assertGreaterEqual(label.winfo_height(), label.winfo_reqheight() - 1)
                    instruction_bottom = max(label.winfo_rooty() + label.winfo_height()
                                             for label in frame.annotation_tutorial_labels.values())
                    for button in (frame.annotation_tutorial_opt_out_btn,
                                   frame.annotation_tutorial_ok_btn):
                        self.assertGreater(button.winfo_rooty(), instruction_bottom)
                        self.assertLessEqual(button.winfo_rooty() + button.winfo_height(),
                                             banner.winfo_rooty() + banner.winfo_height())
                    opt_out = frame.annotation_tutorial_opt_out_btn
                    ok = frame.annotation_tutorial_ok_btn
                    self.assertAlmostEqual(opt_out.winfo_rooty(), ok.winfo_rooty(), delta=1)
                    self.assertAlmostEqual(opt_out.winfo_height(), ok.winfo_height(), delta=1)
                    self.assertAlmostEqual(ok.winfo_rootx() - opt_out.winfo_rootx() - opt_out.winfo_width(),
                                           round(8 * scale), delta=1)
                    self.assertEqual(banner.cget("corner_radius"), 0)
                    self.assertFalse(hasattr(banner, "close_button"))
                    self.assertFalse(hasattr(frame, "annotation_tutorial_close_btn"))
                    self.assertIsNone(root.grab_current())

                preferences = Config()
                frame = Step2Frame(root, preferences=preferences)
                frame.pack(fill="both", expand=True)
                root.update()
                self.assertFalse(visible_tutorial(frame))
                empty_geometry = canvas_geometry(frame)
                frame.set_annotation_tutorial_enabled(True)
                root.update()
                self.assertFalse(visible_tutorial(frame))
                frame._show_image(image, "Step 1 output")
                root.update()
                self.assertFalse(visible_tutorial(frame))
                frame._ensure_fovea_line(48)
                frame._queue_annotation_tutorial()
                root.update()
                self.assertEqual(set(frame.boundary_traces), {FOVEA_BOUNDARY_NAME})
                self.assertFalse(visible_tutorial(frame))
                manually_completed, _order = frame._boundary_traces_from_rows(
                    np.ones((6, 96)), image.shape,
                )
                frame.boundary_traces[BOUNDARY_NAMES[0]] = manually_completed[BOUNDARY_NAMES[0]]
                frame._queue_annotation_tutorial()
                root.update()
                self.assertFalse(visible_tutorial(frame))
                frame.boundary_traces.pop(BOUNDARY_NAMES[0])
                frame._set_segmentation_running(True, animate_progress=False)
                root.update()
                self.assertFalse(visible_tutorial(frame))
                finish_current_segmentation(frame)
                root.update()
                self.assertTrue(visible_tutorial(frame))
                self.assertEqual(canvas_geometry(frame), empty_geometry)
                self.assertFalse(hasattr(frame, "annotation_help_panel"))
                self.assertFalse(hasattr(frame, "content_status_bar"))

                # Running AI again hides an already-visible guide immediately;
                # its completed result makes the guide eligible again.
                frame._set_segmentation_running(True, animate_progress=False)
                self.assertFalse(visible_tutorial(frame))
                frame._queue_annotation_tutorial()
                root.update()
                self.assertFalse(visible_tutorial(frame))
                finish_current_segmentation(frame)
                root.update()
                self.assertTrue(visible_tutorial(frame))

                def sidebar_labels(widget):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Label):
                            yield child.cget("text")
                        yield from sidebar_labels(child)

                self.assertNotIn("Draw", list(sidebar_labels(frame.ctrl)))
                banner = frame.annotation_tutorial
                canvas = frame.image_canvas.canvas
                self.assertEqual(set(frame.annotation_tutorial_labels),
                                 {"Draw", "Edit", "Add point", "Remove point"})
                assert_guide_layout(frame)
                self.assertTrue(frame.annotation_tutorial_info_icon.winfo_viewable())
                self.assertEqual(frame.annotation_tutorial_title.cget("text"), "Manual annotation")
                self.assertEqual(frame.annotation_tutorial_opt_out_btn.cget("text"), "Do not show")
                self.assertEqual(frame.annotation_tutorial_ok_btn.cget("text"), "OK")

                # The embedded guide leaves normal canvas drawing available.
                frame._start_boundary_for_name(BOUNDARY_NAMES[-1])
                click_x, click_y = frame.image_canvas._i2c(40, 16)
                self.assertGreater(click_y, banner.winfo_height())
                canvas.event_generate("<Button-1>", x=round(click_x), y=round(click_y))
                root.update()
                self.assertEqual(frame.image_canvas.get_active_line(), [(40, 16)])
                frame.image_canvas.set_active_line([])
                frame.active_boundary = None
                frame._sync_boundary_canvas_state()

                # A visible guide follows theme changes without changing the
                # image, zoom, or space available to the annotation canvas.
                displayed_image = frame.image_data
                zoom = frame.image_canvas.get_zoom()
                for interface, mode in (("Modern", "Dark"), ("Modern", "Light"),
                                        ("Classic", "Light"), ("Classic", "Dark"),
                                        ("Modern", "Dark")):
                    with self.subTest(guide_interface=interface, guide_appearance=mode):
                        set_interface_mode(interface, redraw=False)
                        apply_appearance_mode(mode, root=root, style=ttk.Style(root))
                        root.update()
                        self.assertTrue(visible_tutorial(frame))
                        self.assertIs(frame.image_data, displayed_image)
                        self.assertEqual(frame.image_canvas.get_zoom(), zoom)
                        geometry = canvas_geometry(frame)
                        self.assertEqual(geometry[:2], empty_geometry[:2])
                        # Classic's native scrollbar border changes the
                        # viewport by two pixels; the banner reserves no row.
                        for actual, expected in zip(geometry[2:], empty_geometry[2:]):
                            self.assertAlmostEqual(actual, expected, delta=2)
                        self.assertEqual(banner.title._label.cget("foreground"),
                                         resolve_color(COLOR_PAIRS["text"], mode))
                        self.assertEqual(banner.info_icon.cget("background"),
                                         resolve_color(COLOR_PAIRS["surface"], mode))
                        self.assertEqual(banner.opt_out_button._canvas.itemcget("inner_parts", "fill"),
                                         resolve_color(COLOR_PAIRS["primary"], mode))
                        self.assertEqual(banner.ok_button._canvas.itemcget("inner_parts", "fill"),
                                         resolve_color(COLOR_PAIRS["success"], mode))
                        assert_guide_layout(frame)

                # OK dismisses only this session, and Settings
                # can reopen it without changing the displayed annotation.
                frame.annotation_tutorial_ok_btn.invoke()
                root.update()
                self.assertFalse(visible_tutorial(frame))
                self.assertTrue(Config().get("show_step2_annotation_tutorial"))
                frame._queue_annotation_tutorial()
                root.update()
                self.assertFalse(visible_tutorial(frame))
                frame.set_annotation_tutorial_enabled(True)
                root.update()
                self.assertTrue(visible_tutorial(frame))
                self.assertIs(frame.image_data, displayed_image)
                self.assertEqual(frame.image_canvas.get_zoom(), zoom)
                self.assertEqual(canvas_geometry(frame), empty_geometry)
                assert_guide_layout(frame)

                # Escape belongs to the guide's focused controls rather than
                # intercepting keyboard input from the annotation canvas.
                button_canvas = frame.annotation_tutorial_ok_btn._canvas
                button_canvas.focus_force()
                root.update()
                button_canvas.event_generate("<Escape>")
                root.update()
                self.assertFalse(visible_tutorial(frame))
                self.assertTrue(Config().get("show_step2_annotation_tutorial"))
                frame.set_annotation_tutorial_enabled(True)
                root.update()
                self.assertTrue(visible_tutorial(frame))
                self.assertIs(frame.image_data, displayed_image)
                self.assertEqual(frame.image_canvas.get_zoom(), zoom)
                assert_guide_layout(frame)

                # The banner follows the drawing viewport while preserving
                # its flush edges and keeping the two actions together.
                root.geometry("1280x820+10000+10000")
                root.update()
                assert_guide_layout(frame)
                root.geometry("1600x820+10000+10000")
                root.update()
                assert_guide_layout(frame)
                self.assertAlmostEqual(frame.annotation_tutorial_heading_labels["Draw"].winfo_rooty(),
                                       frame.annotation_tutorial_heading_labels["Remove point"].winfo_rooty(),
                                       delta=1)

                # Selecting another result moves the banner to that tab's
                # actual drawing viewport, below the tab strip and clear of
                # its scrollbars. The previous viewport owns no stale guide.
                with mock.patch.object(frame, "_read_image_for_annotation",
                                       return_value=(image, None, False)):
                    frame._open_aidas_batch_results_viewer([
                        {"input": "First.img", "boundaries": np.ones((6, 96))},
                        {"input": "Second.img", "boundaries": np.ones((6, 96))},
                    ])
                self.assertEqual(
                    [frame.batch_results_notebook.tab(tab, "tooltip")
                     for tab in frame.batch_results_notebook.tabs()],
                    [str(Path(filename).resolve()) for filename in ("First.img", "Second.img")],
                )
                root.update()
                self.assertTrue(visible_tutorial(frame))
                assert_guide_layout(frame)
                first_banner = frame.annotation_tutorial
                first_canvas = frame.image_canvas.canvas
                first_canvas_geometry = canvas_geometry(frame)
                second_tab = frame.batch_results_notebook.tabs()[1]
                frame.batch_results_notebook.select(second_tab)
                root.update()
                self.assertTrue(visible_tutorial(frame))
                self.assertIsNot(frame.image_canvas.canvas, first_canvas)
                self.assertFalse(first_banner.winfo_exists())
                assert_guide_layout(frame)
                self.assertEqual((first_canvas.winfo_rootx(), first_canvas.winfo_rooty(),
                                  first_canvas.winfo_width(), first_canvas.winfo_height()),
                                 first_canvas_geometry)
                show_segmented_image(frame, "Step 1 output")
                root.update()
                self.assertTrue(visible_tutorial(frame))
                assert_guide_layout(frame)
                banner = frame.annotation_tutorial
                for widget in (frame.annotation_tutorial_info_icon,
                               frame.annotation_tutorial_opt_out_btn,
                               frame.annotation_tutorial_ok_btn):
                    self.assertTrue(widget.winfo_viewable())
                    self.assertGreaterEqual(widget.winfo_rootx(), banner.winfo_rootx())
                    self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(),
                                         banner.winfo_rootx() + banner.winfo_width())
                    self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                         banner.winfo_rooty() + banner.winfo_height())
                root.geometry("1024x160+10000+10000")
                root.update()
                if visible_tutorial(frame):
                    assert_guide_layout(frame)
                root.geometry("1024x680+10000+10000")
                root.update()
                self.assertTrue(visible_tutorial(frame))
                assert_guide_layout(frame)

                frame.annotation_tutorial_ok_btn.invoke()
                root.update()
                self.assertFalse(visible_tutorial(frame))
                self.assertTrue(Config().get("show_step2_annotation_tutorial"))
                frame._show_image(image, "Another image")
                frame._on_mouse_moved(5, 8, np.int16(42))
                frame.image_canvas.set_zoom(2)
                root.update()
                self.assertFalse(visible_tutorial(frame))
                self.assertEqual(canvas_geometry(frame), empty_geometry)

                # Enabling the guide on a fresh image still waits for its
                # completed result, preserving the image and current zoom.
                displayed_image = frame.image_data
                zoom = frame.image_canvas.get_zoom()
                frame.set_annotation_tutorial_enabled(True)
                root.update()
                self.assertFalse(visible_tutorial(frame))
                finish_current_segmentation(frame)
                root.update()
                self.assertTrue(visible_tutorial(frame))
                self.assertIs(frame.image_data, displayed_image)
                self.assertEqual(frame.image_canvas.get_zoom(), zoom)
                assert_guide_layout(frame)
                frame.destroy()

                # A new session still shows the guide after OK, and waits for
                # a previously hidden Step 2 canvas to become visible, even
                # with enlarged display scaling and initial text wrapping.
                ctk.set_widget_scaling(1.5)
                frame = Step2Frame(root, preferences=Config())
                show_segmented_image(frame, "Step 1 output")
                root.update()
                self.assertFalse(visible_tutorial(frame))
                frame.pack(fill="both", expand=True)
                root.update()
                self.assertTrue(visible_tutorial(frame))
                assert_guide_layout(frame)
                frame.annotation_tutorial_opt_out_btn.invoke()
                root.update()
                self.assertFalse(visible_tutorial(frame))
                self.assertFalse(Config().get("show_step2_annotation_tutorial"))

                # Settings can reverse persistent opt-out in the same session.
                frame.set_annotation_tutorial_enabled(True)
                root.update()
                self.assertTrue(visible_tutorial(frame))
                self.assertTrue(Config().get("show_step2_annotation_tutorial"))
                assert_guide_layout(frame)
                frame.set_annotation_tutorial_enabled(False)
                root.update()
                self.assertFalse(visible_tutorial(frame))
                self.assertFalse(Config().get("show_step2_annotation_tutorial"))
                frame.destroy()

                frame = Step2Frame(root, preferences=Config())
                frame.pack(fill="both", expand=True)
                show_segmented_image(frame, "Step 1 output")
                root.update()
                self.assertFalse(visible_tutorial(frame))
                frame.pack_forget()
                frame.set_annotation_tutorial_enabled(True)
                root.update()
                self.assertFalse(visible_tutorial(frame))
                frame.pack(fill="both", expand=True)
                root.update()
                self.assertTrue(visible_tutorial(frame))
                assert_guide_layout(frame)
                frame._queue_annotation_tutorial()
                frame.destroy()
                root.update()
            finally:
                ctk.set_widget_scaling(original_widget_scaling)
                if root is not None:
                    root.destroy()
                set_interface_mode(original_interface, redraw=False)
                ctk.set_appearance_mode("System" if original_source == "system" else original_appearance)


class PolylineDoubleClickTests(unittest.TestCase):
    def _canvas(self, points):
        canvas = ImageCanvas.__new__(ImageCanvas)
        canvas._line_on = True
        canvas._cb_line_finish = mock.Mock()
        canvas._data = np.zeros((10, 10))
        canvas._drag_vertical_line = False
        canvas._active_line = list(points)
        canvas.canvas = SimpleNamespace(canvasx=lambda x: x, canvasy=lambda y: y)
        canvas._c2i = lambda x, y: (x, y)
        canvas._clamp_image_point = lambda x, y: (x, y)
        canvas._redraw_overlays = mock.Mock()
        canvas._emit_line_change = mock.Mock()
        return canvas

    def test_double_click_commits_endpoint_once(self):
        for points in ([(0, 1)], [(0, 1), (7, 3)]):
            canvas = self._canvas(points)
            self.assertEqual(canvas._on_double_click(SimpleNamespace(x=7, y=3)), 'break')
            self.assertEqual(canvas.get_active_line(), [(0, 1), (7, 3)])
            canvas._cb_line_finish.assert_called_once_with()

    def test_double_click_does_not_finish_a_single_point_or_a_fovea_drag(self):
        canvas = self._canvas([(1, 1)])
        canvas._on_double_click(SimpleNamespace(x=1, y=1))
        canvas._cb_line_finish.assert_not_called()
        canvas._drag_vertical_line = True
        canvas._on_double_click(SimpleNamespace(x=7, y=3))
        self.assertEqual(canvas.get_active_line(), [(1, 1)])
        canvas._cb_line_finish.assert_not_called()

    def test_other_canvas_modes_keep_their_normal_press_behavior(self):
        canvas = self._canvas([])
        canvas._line_on = False
        canvas._on_press = mock.Mock()
        event = SimpleNamespace(x=7, y=3)
        canvas._on_double_click(event)
        canvas._on_press.assert_called_once_with(event)


class ImageJLineEditingTests(unittest.TestCase):
    def _canvas(self, points):
        canvas = ImageCanvas.__new__(ImageCanvas)
        canvas._data = np.zeros((100, 100))
        canvas._windowing_system = 'win32'
        canvas._zoom = 1.0
        canvas._img_offset_x = canvas._img_offset_y = 0
        canvas._line_on = False
        canvas._roi_on = False
        canvas._vertical_line_on = False
        canvas._drag_vertical_line = False
        canvas._active_line = []
        canvas._editable_line_label = 'test'
        canvas._line_edit_drag = None
        canvas._line_overlays = [{'label': 'test', 'points': list(points)}]
        canvas._cb_line_edit = mock.Mock()
        canvas._cb_line_finish = mock.Mock()
        canvas._cb_line = mock.Mock()
        canvas.canvas = SimpleNamespace(canvasx=lambda x: x, canvasy=lambda y: y, focus_set=mock.Mock())
        canvas._redraw_overlays = mock.Mock()
        canvas._refresh_cursor_for_mode = mock.Mock()
        return canvas

    def test_dragging_a_completed_vertex_edits_only_that_vertex(self):
        points = [(10, 10), (30, 20), (50, 20), (70, 10)]
        canvas = self._canvas(points)
        canvas._on_press(SimpleNamespace(x=30, y=20, state=0))
        canvas._on_drag_motion(SimpleNamespace(x=30, y=35))
        canvas._on_release(None)
        canvas._cb_line_edit.assert_called_once_with('test', [(10, 10), (30, 35), (50, 20), (70, 10)])
        self.assertIsNone(canvas._line_edit_drag)

    def test_shift_click_splits_a_vertex_along_its_adjacent_segments(self):
        canvas = self._canvas([(10, 10), (40, 40), (70, 10)])
        canvas._on_press(SimpleNamespace(x=40, y=40, state=1))
        canvas._on_release(None)
        canvas._cb_line_edit.assert_called_once_with('test', [(10, 10), (30, 30), (50, 30), (70, 10)])

    def test_alt_click_deletes_one_vertex_with_the_imagej_minimum(self):
        canvas = self._canvas([(10, 10), (30, 20), (50, 20), (70, 10)])
        canvas._on_press(SimpleNamespace(x=30, y=20, state=0x20000))
        canvas._cb_line_edit.assert_called_once_with('test', [(10, 10), (50, 20), (70, 10)])
        canvas._cb_line_edit.reset_mock()
        canvas._on_press(SimpleNamespace(x=50, y=20, state=0x20000))
        canvas._cb_line_edit.assert_not_called()

    def test_dragging_a_segment_moves_the_entire_line_without_distorting_it(self):
        canvas = self._canvas([(10, 10), (30, 20), (50, 20), (70, 10)])
        canvas._on_press(SimpleNamespace(x=40, y=20, state=0))
        canvas._on_drag_motion(SimpleNamespace(x=45, y=25))
        canvas._on_release(None)
        canvas._cb_line_edit.assert_called_once_with('test', [(15, 15), (35, 25), (55, 25), (75, 15)])

    def test_clicking_start_handle_finishes_without_closing_the_open_polyline(self):
        canvas = self._canvas([])
        canvas.set_editable_line(None)
        canvas._line_on = True
        canvas._active_line = [(10, 10), (40, 20), (70, 15)]
        canvas._on_press(SimpleNamespace(x=11, y=11, state=0))
        canvas._cb_line_finish.assert_called_once_with()
        self.assertEqual(canvas.get_active_line(), [(10, 10), (40, 20), (70, 15)])

    def test_manual_line_click_takes_precedence_over_the_foveal_marker(self):
        canvas = self._canvas([])
        canvas.set_editable_line(None)
        canvas._line_on = True
        canvas._vertical_line_on = True
        canvas._on_press(SimpleNamespace(x=50, y=20, state=0))
        self.assertEqual(canvas.get_active_line(), [(50, 20)])
        self.assertFalse(canvas._drag_vertical_line)

    def test_saved_line_edits_use_image_coordinates_after_zoom_and_pan(self):
        canvas = self._canvas([(10, 10), (30, 20), (50, 20), (70, 10)])
        canvas._zoom = 2.0
        canvas._img_offset_x, canvas._img_offset_y = -10, 8
        canvas._on_press(SimpleNamespace(x=50, y=48, state=0))
        canvas._on_drag_motion(SimpleNamespace(x=60, y=58))
        canvas._on_release(None)
        canvas._cb_line_edit.assert_called_once_with('test', [(10, 10), (35, 25), (50, 20), (70, 10)])

    def test_dense_ai_line_handles_are_spaced_without_changing_boundary_points(self):
        points = [(x, 30) for x in range(2133)]
        canvas = self._canvas(points)
        canvas.canvas = mock.Mock()
        canvas._line_width = 2
        for zoom in (0.45, 1.0, 4.0):
            with self.subTest(zoom=zoom):
                canvas._zoom = zoom
                indices = canvas._line_handle_indices(points)
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-1], len(points) - 1)
                self.assertTrue(all((b-a)*zoom >= 16 for a, b in zip(indices, indices[1:])))
                canvas.canvas.reset_mock()
                canvas._draw_polyline(points, '#ffd500', show_vertices=True)
                self.assertEqual(canvas.canvas.create_rectangle.call_count, len(indices))
                self.assertEqual(canvas.canvas.create_line.call_count, 1)
                self.assertEqual(canvas._line_overlays[0]['points'], points)

    def test_dense_line_hit_testing_uses_only_visible_handles(self):
        points = [(x, 30) for x in range(100)]
        canvas = self._canvas(points)
        self.assertEqual(canvas._hit_line_vertex(points, 16, 30), 16)
        self.assertIsNone(canvas._hit_line_vertex(points, 8, 30))
        self.assertTrue(canvas._hit_line_segment(points, 8, 30))

    def test_unselected_ai_line_draws_no_vertex_boxes(self):
        canvas = self._canvas([(x, 30) for x in range(100)])
        canvas.canvas = mock.Mock()
        canvas._line_width = 2
        canvas._editable_line_label = None
        canvas._line_overlays[0]['color'] = '#ffd500'
        canvas._draw_text_label = mock.Mock()
        canvas._draw_line_overlays()
        canvas.canvas.create_line.assert_called_once()
        canvas.canvas.create_rectangle.assert_not_called()

    def test_shift_adds_a_visible_ai_handle_and_preserves_samples_outside_its_neighbors(self):
        points = [(x, 30) for x in range(100)]
        canvas = self._canvas(points)
        before_handles = canvas._line_handle_indices(points)
        canvas._on_press(SimpleNamespace(x=32, y=30, state=1))
        canvas._on_release(None)
        item = canvas._line_overlays[0]
        self.assertEqual(item['points'][:17], points[:17])
        self.assertEqual(item['points'][19:], points[48:])
        self.assertEqual(len(canvas._line_handle_indices(item['points'], item['handle_indices'])), len(before_handles) + 1)
        self.assertEqual(len(item['points']), len(set(item['points'])))
        canvas.set_line_overlays([{**item, 'color': '#ffd500'}])
        reloaded = canvas._line_overlays[0]
        self.assertEqual(reloaded['points'], item['points'])
        self.assertEqual(reloaded['handle_indices'], item['handle_indices'])
        self.assertEqual(canvas.get_line_handle_indices('test'), item['handle_indices'])

    def test_alt_removes_a_visible_ai_handle_without_a_neighbor_reappearing(self):
        points = [(x, 30) for x in range(100)]
        for system, modifier in (('win32', 0x20000), ('x11', 0x0008),
                                 ('aqua', 0x0010), ('win32', 0x0004)):
            with self.subTest(system=system, modifier=modifier):
                canvas = self._canvas(points)
                canvas._windowing_system = system
                before = [points[index] for index in canvas._line_handle_indices(points)]
                canvas._on_press(SimpleNamespace(x=32, y=30, state=modifier))
                item = canvas._line_overlays[0]
                after = [item['points'][index] for index in canvas._line_handle_indices(item['points'], item['handle_indices'])]
                self.assertEqual(after, [point for point in before if point != (32, 30)])
                self.assertEqual(item['points'], points[:17] + points[48:])
                canvas.set_line_overlays([{**item, 'color': '#ffd500'}])
                reloaded = canvas._line_overlays[0]
                self.assertIsNone(canvas._hit_line_vertex(reloaded['points'], 32, 30, reloaded['handle_indices']))
                canvas._cb_line_edit.assert_called_once_with('test', item['points'])

    def test_dense_line_endpoint_removal_keeps_the_other_handles_and_samples(self):
        points = [(x, 30) for x in range(100)]
        for vertex in (0, 99):
            with self.subTest(vertex=vertex):
                canvas = self._canvas(points)
                before = [points[index] for index in canvas._line_handle_indices(points)]
                canvas._on_press(SimpleNamespace(x=vertex, y=30, state=0x20000))
                item = canvas._line_overlays[0]
                after = [item['points'][index] for index in canvas._line_handle_indices(item['points'], item['handle_indices'])]
                self.assertEqual(after, [point for point in before if point != (vertex, 30)])
                self.assertEqual(item['points'], points[16:] if vertex == 0 else points[:81])

    def test_shift_can_split_adjacent_image_pixels_after_zooming_in(self):
        canvas = self._canvas([(0, 30), (1, 30), (2, 30), (3, 30)])
        canvas._zoom = 32
        canvas._on_press(SimpleNamespace(x=32, y=960, state=1))
        canvas._on_release(None)
        item = canvas._line_overlays[0]
        self.assertEqual(len(item['points']), 5)
        self.assertEqual(len(set(item['points'])), 5)
        canvas.set_line_overlays([{**item, 'color': '#ffd500'}])
        self.assertEqual(len(canvas._line_overlays[0]['points']), 5)
        self.assertEqual(canvas.get_line_handle_indices('test'), [0, 1, 2, 3, 4])

    def test_edited_handles_stay_spaced_after_zooming_out(self):
        canvas = self._canvas([(x, 30) for x in range(100)])
        canvas._on_press(SimpleNamespace(x=32, y=30, state=1))
        canvas._on_release(None)
        canvas._zoom = 0.3
        item = canvas._line_overlays[0]
        visible = canvas._line_handle_indices(item['points'], item['handle_indices'])
        self.assertTrue(all((item['points'][b][0]-item['points'][a][0])*canvas._zoom >= 8
                            for a, b in zip(visible, visible[1:])))

    def test_windows_num_lock_and_other_non_editing_flags_never_delete_points(self):
        points = [(10, 10), (30, 20), (50, 20), (70, 10), (90, 20)]
        for state in (0, 0x8, 0xA, 0x28, 0x108, 0x12A):
            with self.subTest(state=state):
                canvas = self._canvas(points)
                canvas._on_press(SimpleNamespace(x=30, y=20, state=state))
                canvas._on_release(None)
                self.assertEqual(canvas._line_overlays[0]['points'], points)
                canvas._cb_line_edit.assert_not_called()

    def test_shift_with_windows_num_lock_always_adds_a_handle(self):
        points = [(10, 10), (30, 20), (50, 20), (70, 10), (90, 20)]
        for state in (0x1, 0x9, 0xB, 0x29, 0x109, 0x12B):
            with self.subTest(state=state):
                canvas = self._canvas(points)
                canvas._on_press(SimpleNamespace(x=30, y=20, state=state))
                canvas._on_release(None)
                item = canvas._line_overlays[0]
                self.assertEqual(len(item['points']), len(points) + 1)
                self.assertEqual(len(item['handle_indices']), len(points) + 1)
                canvas._cb_line_edit.assert_called_once()

    def test_alt_and_control_remove_exactly_one_handle_with_num_lock_enabled(self):
        points = [(10, 10), (30, 20), (50, 20), (70, 10), (90, 20)]
        for state in (0x20008, 0x2000A, 0x20108, 0xC, 0xE, 0x10C):
            with self.subTest(state=state):
                canvas = self._canvas(points)
                canvas._on_press(SimpleNamespace(x=30, y=20, state=state))
                canvas._on_release(None)
                self.assertEqual(canvas._line_overlays[0]['points'], [points[0], *points[2:]])
                canvas._cb_line_edit.assert_called_once()

    def test_plain_drag_with_num_lock_moves_the_vertex_without_deleting_it(self):
        points = [(10, 10), (30, 20), (50, 20), (70, 10), (90, 20)]
        canvas = self._canvas(points)
        canvas._on_press(SimpleNamespace(x=30, y=20, state=0x8))
        canvas._on_drag_motion(SimpleNamespace(x=30, y=35, state=0x108))
        canvas._on_release(None)
        canvas._cb_line_edit.assert_called_once_with('test', [points[0], (30, 35), *points[2:]])
