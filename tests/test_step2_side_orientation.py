import unittest

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas
from aidas.steps.step2_annotate import (
    DEFAULT_SAVE_ORIENTATION,
    SAVE_ORIENTATION_NASAL_TO_TEMPORAL,
    SAVE_ORIENTATION_TEMPORAL_TO_NASAL,
    Step2Frame,
)


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _SideLabelCanvas:
    def __init__(self):
        self.labels = []

    def set_side_labels(self, left, right, *, on_flip=None):
        self.labels.append((left, right, on_flip))


class _TkCanvasStub:
    def __init__(self):
        self.texts = []

    def delete(self, _tag):
        return None

    def winfo_width(self):
        return 800

    def canvasx(self, value):
        return value + 100

    def canvasy(self, value):
        return value + 50

    def create_text(self, x, y, **kwargs):
        self.texts.append((x, y, kwargs))
        return len(self.texts)

    def bbox(self, _item_id):
        return (10, 10, 80, 30)

    def create_rectangle(self, *_args, **_kwargs):
        return 100 + len(self.texts)

    def create_polygon(self, *_args, **_kwargs):
        return 150 + len(self.texts)

    def create_line(self, *_args, **_kwargs):
        return 200 + len(self.texts)

    def tag_lower(self, *_args):
        return None

    def tag_bind(self, *_args):
        return None

    def itemconfigure(self, *_args, **_kwargs):
        return None


class _FlipButtonStub:
    def __init__(self):
        self.hidden = False

    def place_forget(self):
        self.hidden = True


class Step2SideOrientationTests(unittest.TestCase):
    def _frame(self, orientation=DEFAULT_SAVE_ORIENTATION):
        frame = Step2Frame.__new__(Step2Frame)
        frame.save_orientation_var = _Var(orientation)
        frame._batch_result_states = {}
        frame._active_batch_result_tab = None
        frame.image_canvas = _SideLabelCanvas()
        return frame

    def test_flip_updates_canvas_labels_and_active_image_state(self):
        frame = self._frame(SAVE_ORIENTATION_TEMPORAL_TO_NASAL)
        frame._active_batch_result_tab = "tab-1"
        frame._batch_result_states["tab-1"] = {}

        frame._flip_image_sides()

        self.assertEqual(frame.save_orientation_var.get(), SAVE_ORIENTATION_NASAL_TO_TEMPORAL)
        left, right, callback = frame.image_canvas.labels[-1]
        self.assertEqual((left, right), ("Nasal", "Temporal"))
        self.assertEqual(callback, frame._flip_image_sides)
        self.assertEqual(
            frame._batch_result_states["tab-1"]["save_orientation"],
            SAVE_ORIENTATION_NASAL_TO_TEMPORAL,
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

    def test_canvas_side_badges_are_pinned_to_viewport_edges(self):
        image_canvas = ImageCanvas.__new__(ImageCanvas)
        image_canvas.canvas = _TkCanvasStub()
        image_canvas._data = np.zeros((2, 2))
        image_canvas._side_labels = ("Temporal", "Nasal")

        image_canvas._draw_side_labels()

        left, right = image_canvas.canvas.texts
        self.assertEqual(left[2]["text"], "TEMPORAL")
        self.assertEqual(left[2]["anchor"], "center")
        self.assertEqual(right[2]["text"], "NASAL")
        self.assertEqual(right[2]["anchor"], "center")
        self.assertEqual(right[0], 781)

    def test_swap_button_is_hidden_when_canvas_image_is_cleared(self):
        image_canvas = ImageCanvas.__new__(ImageCanvas)
        image_canvas.canvas = _TkCanvasStub()
        image_canvas._data = None
        image_canvas._side_labels = ("Temporal", "Nasal")
        image_canvas._side_flip_button = _FlipButtonStub()

        image_canvas._draw_side_labels()

        self.assertTrue(image_canvas._side_flip_button.hidden)


if __name__ == "__main__":
    unittest.main()
