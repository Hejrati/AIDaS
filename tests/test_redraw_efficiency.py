import unittest

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas
from aidas.steps.step2_annotate import Step2Frame


class _ControlStub:
    def __init__(self, state="normal"):
        self.current_state = state
        self.configure_calls = []

    def cget(self, option):
        if option != "state":
            raise KeyError(option)
        return self.current_state

    def configure(self, **options):
        self.configure_calls.append(options)
        self.current_state = options["state"]


class RedrawEfficiencyTests(unittest.TestCase):
    def test_step2_control_updates_are_idempotent(self):
        control = _ControlStub("normal")

        self.assertFalse(Step2Frame._set_control_enabled(control, True))
        self.assertTrue(Step2Frame._set_control_enabled(control, False))
        self.assertFalse(Step2Frame._set_control_enabled(control, False))
        self.assertEqual(control.configure_calls, [{"state": "disabled"}])

    def test_reapplying_disabled_line_mode_does_not_emit_empty_change(self):
        canvas = ImageCanvas.__new__(ImageCanvas)
        canvas._line_on = False
        canvas._active_line = []
        canvas._line_preview = None
        redraws = []
        emissions = []
        canvas._redraw_overlays = lambda: redraws.append(True)
        canvas._emit_line_change = lambda: emissions.append(True)
        canvas._refresh_cursor_for_mode = lambda: None

        canvas.enable_line(False)

        self.assertEqual(redraws, [])
        self.assertEqual(emissions, [])

    def test_disabling_line_mode_still_clears_real_active_points(self):
        canvas = ImageCanvas.__new__(ImageCanvas)
        canvas._line_on = False
        canvas._active_line = [(1, 2)]
        canvas._line_preview = None
        redraws = []
        emissions = []
        canvas._redraw_overlays = lambda: redraws.append(True)
        canvas._emit_line_change = lambda: emissions.append(True)
        canvas._refresh_cursor_for_mode = lambda: None

        canvas.enable_line(False)

        self.assertEqual(canvas._active_line, [])
        self.assertEqual(redraws, [True])
        self.assertEqual(emissions, [True])

    def test_integer_preview_matches_existing_min_max_scaling(self):
        canvas = ImageCanvas.__new__(ImageCanvas)
        data = np.array([[-32768, -1024, 0], [1, 12000, 32767]], dtype=np.int16)
        reference = data.astype(np.float64)
        reference = np.clip(
            (reference - reference.min()) / (reference.max() - reference.min()) * 255.0,
            0,
            255,
        ).astype(np.uint8)

        np.testing.assert_array_equal(canvas._to_display(data), reference)


if __name__ == "__main__":
    unittest.main()
