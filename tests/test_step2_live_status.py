import unittest

import numpy as np

from aidas.steps.step2_annotate import Step2Frame


class _VariableStub:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = str(value)


class _CanvasStub:
    def __init__(self, zoom):
        self.zoom = zoom

    def get_zoom(self):
        return self.zoom


class Step2LiveStatusTests(unittest.TestCase):
    def _frame(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame.image_data = np.zeros((128, 1473), dtype=np.int16)
        frame.image_canvas = _CanvasStub(0.49)
        frame.status_var = _VariableStub()
        frame.fovea_x = 666
        frame._last_status_mouse_sample = None
        return frame

    def test_zoom_callback_refreshes_status_without_mouse_motion(self):
        frame = self._frame()
        frame._last_status_mouse_sample = (4, 125, np.int16(-14674))

        frame._on_canvas_zoom_changed(0.75)

        self.assertIn("(4, 125)  val=-14674", frame.status_var.value)
        self.assertIn("Zoom: 75%", frame.status_var.value)
        self.assertIn("Fovea x: 666", frame.status_var.value)

    def test_fovea_callback_refreshes_status_without_mouse_motion(self):
        frame = self._frame()
        frame.boundary_traces = {}
        frame.fovea_line_var = _VariableStub()
        frame.fovea_x_entry_var = _VariableStub()
        frame._updating_fovea_entry = False

        frame._on_vertical_line_changed(700)

        self.assertEqual(frame.fovea_x, 700)
        self.assertEqual(frame.fovea_x_entry_var.value, "700")
        self.assertIn("Zoom: 49%", frame.status_var.value)
        self.assertIn("Fovea x: 700", frame.status_var.value)

    def test_mouse_motion_updates_cached_sample_and_all_live_fields(self):
        frame = self._frame()

        frame._on_mouse_moved(10, 20, np.int16(42))

        self.assertEqual(frame._last_status_mouse_sample, (10, 20, np.int16(42)))
        self.assertIn("(10, 20)  val=42", frame.status_var.value)
        self.assertIn("Zoom: 49%", frame.status_var.value)
        self.assertIn("Fovea x: 666", frame.status_var.value)


if __name__ == "__main__":
    unittest.main()
