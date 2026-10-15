import unittest
from unittest import mock

import numpy as np

import test_step2_manual_annotation as fixtures
from aidas.steps.step2_annotate import BOUNDARY_NAMES


class BoundaryRestoreCacheTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixtures.ManualAnnotationTests()._frame()
        traces, order = self.frame._boundary_traces_from_rows(np.ones((6, 10)), (8, 10))
        self.frame.boundary_traces.update(traces)
        self.frame.boundary_order.extend(order)
        self.original = self.frame._copy_trace_dict(traces)
        self.frame._set_completion_from_traces()

    def test_single_arrow_restores_exact_line_without_loading_model(self):
        frame = self.frame
        name = BOUNDARY_NAMES[0]
        frame._delete_selected_boundary()
        with mock.patch('aidas.steps.step2_annotate.os.path.isfile') as model_check:
            frame._segment_current_image(boundary_names=[name])
        model_check.assert_not_called()
        self.assertEqual(frame.boundary_traces[name], self.original[name])
        self.assertTrue(frame.boundary_completion_vars[name].get())

    def test_all_arrows_and_repeated_cycles_restore_every_line(self):
        frame = self.frame
        for _ in range(3):
            with mock.patch('aidas.steps.step2_annotate.messagebox.askyesno', return_value=True):
                frame._clear_all_traces()
            with mock.patch('aidas.steps.step2_annotate.os.path.isfile') as model_check:
                frame._segment_current_image()
            model_check.assert_not_called()
            self.assertEqual(frame._incomplete_boundary_names(), [])
            self.assertEqual({name: frame.boundary_traces[name] for name in BOUNDARY_NAMES}, self.original)

    def test_edited_removed_line_requires_inference_but_other_lines_restore(self):
        frame = self.frame
        edited = BOUNDARY_NAMES[0]
        frame.image_canvas.get_line_handle_indices.return_value = None
        frame._on_saved_line_edited(edited, [(0, 3), (9, 4)])
        with mock.patch('aidas.steps.step2_annotate.messagebox.askyesno', return_value=True):
            frame._clear_all_traces()
        remaining = frame._restore_cached_boundaries(list(BOUNDARY_NAMES))
        self.assertEqual(remaining, [edited])
        self.assertNotIn(edited, frame.boundary_traces)
        for name in BOUNDARY_NAMES[1:]:
            self.assertEqual(frame.boundary_traces[name], self.original[name])

    def test_cache_isolated_between_canvases_and_replaced_images(self):
        frame = self.frame
        name = BOUNDARY_NAMES[0]
        first_canvas = frame.image_canvas
        frame._delete_selected_boundary()
        frame.image_canvas = mock.Mock()
        frame.image_canvas.get_active_line.return_value = []
        self.assertEqual(frame._restore_cached_boundaries([name]), [name])
        frame.image_canvas = first_canvas
        self.assertEqual(frame._restore_cached_boundaries([name]), [])
        frame._delete_selected_boundary()
        frame.image_data = frame.image_data.copy()
        self.assertEqual(frame._restore_cached_boundaries([name]), [name])

    def test_mixed_restore_worker_receives_only_the_modified_layer(self):
        frame = self.frame
        edited = BOUNDARY_NAMES[0]
        frame.boundary_traces[edited]['user_modified'] = True
        with mock.patch('aidas.steps.step2_annotate.messagebox.askyesno', return_value=True):
            frame._clear_all_traces()
        frame.aidas_model_path = 'model.onnx'
        frame.current_file = 'image.hdr'
        frame._shared_core_budget = mock.Mock(return_value=(4, 0, 3))
        frame._confirm_shared_core_contention = mock.Mock(return_value=True)
        frame._normalized_shared_core_limit = mock.Mock(return_value=3)
        frame._capture_current_editor_state = mock.Mock(return_value={
            'fovea_x': 5, 'save_orientation': 'temporal'})
        frame._set_segmentation_running = mock.Mock()
        with mock.patch('aidas.steps.step2_annotate.os.path.isfile', return_value=True), mock.patch(
                'aidas.steps.step2_annotate.threading.Thread') as worker:
            frame._segment_current_image()
        request = worker.call_args.kwargs['kwargs']['current_image_state']
        self.assertEqual(request['boundary_names'], [edited])
        worker.return_value.start.assert_called_once()

    def test_manual_redrawing_invalidates_hidden_cached_line(self):
        frame = self.frame
        name = BOUNDARY_NAMES[0]
        frame._delete_selected_boundary()
        frame._on_active_line_changed([(0, 4), (9, 4)])
        self.assertEqual(frame._restore_cached_boundaries([name]), [name])

    def test_edit_flag_survives_tab_state_copy(self):
        frame = self.frame
        frame.image_canvas.get_line_handle_indices.return_value = None
        frame._on_saved_line_edited(BOUNDARY_NAMES[0], [(0, 3), (9, 4)])
        copied = frame._copy_trace_dict(frame.boundary_traces)
        self.assertTrue(copied[BOUNDARY_NAMES[0]]['user_modified'])

    def test_reverting_one_or_all_immediately_disables_next_step_until_restored(self):
        frame = self.frame
        frame._active_batch_result_tab = 'active'
        frame._batch_result_states = {'active': {'traces': frame._copy_trace_dict(frame.boundary_traces)}}
        frame.continue_to_step3_button = mock.Mock()
        frame._set_control_enabled = mock.Mock()
        frame._update_boundary_action_buttons.side_effect = frame._update_continue_to_step3_button_state
        frame._update_continue_to_step3_button_state()
        self.assertTrue(frame._set_control_enabled.call_args.args[1])
        frame._delete_selected_boundary()
        self.assertFalse(frame._set_control_enabled.call_args.args[1])
        # The saved state still has every trace; the live editor must take precedence.
        self.assertTrue(frame._batch_state_complete(frame._batch_result_states['active']))
        frame._restore_cached_boundaries([BOUNDARY_NAMES[0]])
        self.assertTrue(frame._set_control_enabled.call_args.args[1])
        with mock.patch('aidas.steps.step2_annotate.messagebox.askyesno', return_value=True):
            frame._clear_all_traces()
        self.assertFalse(frame._set_control_enabled.call_args.args[1])
        frame._restore_cached_boundaries(list(BOUNDARY_NAMES))
        self.assertTrue(frame._set_control_enabled.call_args.args[1])

    def test_missing_boundary_in_inactive_tab_blocks_button_and_handoff(self):
        frame = self.frame
        frame._active_batch_result_tab = 'active'
        frame._batch_result_states = {
            'active': {'traces': frame.boundary_traces},
            'inactive': {'traces': {name: self.original[name] for name in BOUNDARY_NAMES[1:]}},
        }
        frame.continue_to_step3_button = mock.Mock()
        frame._set_control_enabled = mock.Mock()
        frame._save_all_batch_result_tabs = mock.Mock()
        frame.on_continue_to_step3 = mock.Mock()
        frame._update_continue_to_step3_button_state()
        self.assertFalse(frame._set_control_enabled.call_args.args[1])
        with mock.patch('aidas.steps.step2_annotate.messagebox.showwarning') as warning:
            frame._save_all_and_continue_to_step3_button()
        warning.assert_called_once()
        frame._save_all_batch_result_tabs.assert_not_called()
        frame.on_continue_to_step3.assert_not_called()


if __name__ == '__main__':
    unittest.main()
