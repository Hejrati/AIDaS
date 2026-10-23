"""Headless annotation fixtures shared by Step 2 regression suites."""

from unittest import mock

import numpy as np

from aidas.steps.step2_annotate import BOUNDARY_NAMES, FOVEA_BOUNDARY_NAME, Step2Frame


class StubVariable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def make_annotation_frame():
    frame = Step2Frame.__new__(Step2Frame)
    frame._manual_segmentation_enabled = True
    frame.image_data = np.arange(80, dtype=np.int16).reshape(8, 10)
    frame.image_canvas = mock.Mock()
    frame.image_canvas.get_active_line.return_value = []
    frame._segmenter_running = False
    frame._batch_fovea_picker_active = False
    frame.active_boundary = BOUNDARY_NAMES[0]
    frame.fovea_x = 5
    frame.boundary_traces = {FOVEA_BOUNDARY_NAME: frame._vertical_line_trace(5, 8)}
    frame.boundary_order = [FOVEA_BOUNDARY_NAME]
    frame.boundary_completion_vars = {name: StubVariable(False) for name in BOUNDARY_NAMES}
    frame.status_var = StubVariable()
    frame.active_trace_var = StubVariable()
    frame.fovea_x_entry_var = StubVariable(5)
    frame.fovea_line_var = StubVariable()
    frame.boundary_workflow_status_var = StubVariable()
    for name in ('_refresh_boundary_lists', '_refresh_trace_list', '_select_trace_by_name',
                 '_update_boundary_action_buttons', '_sync_boundary_canvas_state',
                 '_update_boundary_progress_bar', '_set_fovea_controls_enabled',
                 '_sync_active_batch_result_state', '_save_marked_images'):
        setattr(frame, name, mock.Mock())
    return frame
