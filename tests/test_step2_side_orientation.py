import inspect
import unittest

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas
from aidas.steps.step2_annotate import (
    DEFAULT_SAVE_ORIENTATION,
    IMAGE_SIDE_NASAL,
    IMAGE_SIDE_TEMPORAL,
    SAVE_ORIENTATION_NASAL_TO_TEMPORAL,
    SAVE_ORIENTATION_TEMPORAL_TO_NASAL,
    Step2Frame,
)


class Step2SideOrientationTests(unittest.TestCase):
    def _frame(self, orientation=DEFAULT_SAVE_ORIENTATION):
        frame = Step2Frame.__new__(Step2Frame)
        frame._current_save_orientation = orientation
        frame._batch_result_states = {}
        frame._active_batch_result_tab = None
        return frame

    def test_explicit_image_side_maps_to_internal_save_geometry(self):
        self.assertEqual(
            Step2Frame._save_orientation_for_image_side(IMAGE_SIDE_TEMPORAL),
            SAVE_ORIENTATION_NASAL_TO_TEMPORAL,
        )
        self.assertEqual(
            Step2Frame._save_orientation_for_image_side(IMAGE_SIDE_NASAL),
            SAVE_ORIENTATION_TEMPORAL_TO_NASAL,
        )

    def test_pair_save_uses_explicit_per_image_orientation(self):
        frame = self._frame(SAVE_ORIENTATION_TEMPORAL_TO_NASAL)
        volume = np.arange(12).reshape(1, 3, 4)
        source = volume + 100

        nasal, temporal, nasal_source, temporal_source = frame._orient_volumes_for_pair_save(
            volume,
            source,
            SAVE_ORIENTATION_NASAL_TO_TEMPORAL,
        )

        np.testing.assert_array_equal(nasal, np.flip(volume, axis=-1))
        np.testing.assert_array_equal(temporal, volume)
        np.testing.assert_array_equal(nasal_source, np.flip(source, axis=-1))
        np.testing.assert_array_equal(temporal_source, source)

    def test_inactive_batch_save_passes_that_tabs_orientation(self):
        frame = self._frame(SAVE_ORIENTATION_TEMPORAL_TO_NASAL)
        frame.current_file = "active.img"
        frame.image_data = np.zeros((2, 2))
        frame.boundary_traces = {}
        frame.boundary_order = []
        frame.fovea_x = None
        frame._input_analyze_template = None
        frame._source_was_8bit = False
        frame.boundary_completion_vars = {}
        frame._active_batch_result_tab = "active"
        frame._batch_result_states["inactive"] = {
            "input": "inactive.img",
            "image": np.ones((2, 2)),
            "traces": {"RPE": {"points": [(0, 0), (1, 0)]}},
            "order": ["RPE"],
            "fovea_x": None,
            "template": None,
            "source_was_8bit": False,
            "save_orientation": SAVE_ORIENTATION_NASAL_TO_TEMPORAL,
        }
        captured = []
        frame._set_completion_from_traces = lambda: None
        frame._save_current_marked_orientation_pair = (
            lambda orientation=None: captured.append(orientation) or ("nasal", "temporal")
        )

        result = frame._save_batch_result_state("inactive", save_orientation_pair=True)

        self.assertEqual(result, ("nasal", "temporal"))
        self.assertEqual(captured, [SAVE_ORIENTATION_NASAL_TO_TEMPORAL])
        self.assertEqual(frame.current_file, "active.img")

    def test_canvas_has_no_legacy_swap_side_control(self):
        source = inspect.getsource(ImageCanvas)

        self.assertNotIn("Swap side", source)
        self.assertNotIn("_side_flip_button", source)
        self.assertNotIn("set_side_labels", source)


if __name__ == "__main__":
    unittest.main()
