import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from aidas.canvas.image_canvas import ImageCanvas
from aidas.steps.step2_annotate import (
    DEFAULT_SAVE_ORIENTATION,
    IMAGE_SIDE_NASAL,
    IMAGE_SIDE_TEMPORAL,
    SAVE_ORIENTATION_NASAL_TO_TEMPORAL,
    SAVE_ORIENTATION_TEMPORAL_TO_NASAL,
    FOVEA_BOUNDARY_NAME,
    Step2Frame,
)
from aidas.utils.io_utils import read_analyze, write_analyze


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

        np.testing.assert_array_equal(nasal, volume)
        np.testing.assert_array_equal(temporal, np.flip(volume, axis=-1))
        np.testing.assert_array_equal(nasal_source, source)
        np.testing.assert_array_equal(temporal_source, np.flip(source, axis=-1))

    def test_orientation_labels_still_report_the_selected_left_side(self):
        frame = self._frame()
        for side in (IMAGE_SIDE_NASAL, IMAGE_SIDE_TEMPORAL):
            with self.subTest(left_side=side):
                orientation = frame._save_orientation_for_image_side(side)
                self.assertEqual(frame._image_side_for_save_orientation(orientation), side)
                self.assertEqual(frame._save_orientation_label(orientation), side.title())

    def test_single_save_matches_the_nasal_member_of_the_pair(self):
        frame = self._frame()
        volume = np.arange(12).reshape(1, 3, 4)
        for orientation in (SAVE_ORIENTATION_TEMPORAL_TO_NASAL,
                            SAVE_ORIENTATION_NASAL_TO_TEMPORAL):
            with self.subTest(orientation=orientation):
                nasal, _temporal, _nasal_source, _temporal_source = (
                    frame._orient_volumes_for_pair_save(volume, volume + 100, orientation)
                )
                np.testing.assert_array_equal(
                    frame._orient_volume_for_single_save(volume, orientation), nasal
                )

    def _disk_frame(self, folder, left_side, *, fovea_x=337):
        folder.mkdir(parents=True, exist_ok=True)
        frame = self._frame(Step2Frame._save_orientation_for_image_side(left_side))
        height, width = 24, 2133
        rows, columns = np.indices((height, width))
        frame.image_data = (rows * 31 + columns * 3 - 2000).astype(np.int16)
        frame.current_file = str(folder / "Light.img")
        write_analyze(str(folder / "Light"), np.stack([frame.image_data] * 2))
        frame.boundary_traces, frame.boundary_order = frame._boundary_traces_from_rows(
            np.repeat(np.array([2, 5, 8, 11, 15, 21])[:, None], width, axis=1),
            frame.image_data.shape,
        )
        frame.boundary_traces[FOVEA_BOUNDARY_NAME] = frame._vertical_line_trace(fovea_x, height)
        frame.boundary_order.append(FOVEA_BOUNDARY_NAME)
        frame.fovea_x = fovea_x
        frame._source_was_8bit = False
        frame._input_analyze_template = None
        frame.boundary_completion_vars = {}
        frame.is_step3_folder_active = None
        frame._sync_active_batch_result_state = mock.Mock()
        frame.status_var = mock.Mock()
        return frame

    def _assert_saved_pair(self, folder, left_side, original_marked, original_source, fovea_x):
        for output_side in (IMAGE_SIDE_NASAL, IMAGE_SIDE_TEMPORAL):
            with self.subTest(left_side=left_side, output_side=output_side):
                # Step 3 analyzes the right side of its input. The unchanged
                # image therefore belongs to the side opposite the LEFT answer.
                mirrored = output_side == left_side
                expected_marked = np.flip(original_marked, axis=-1) if mirrored else original_marked
                expected_source = np.flip(original_source, axis=-1) if mirrored else original_source
                marked = read_analyze(str(folder / output_side / "Light_MARKED"))
                source = read_analyze(str(folder / output_side / "Light"))
                self.assertEqual((marked.dtype.kind, marked.dtype.itemsize), ("u", 1))
                self.assertEqual((source.dtype.kind, source.dtype.itemsize), ("i", 2))
                np.testing.assert_array_equal(marked, expected_marked)
                np.testing.assert_array_equal(source, expected_source)
                expected_fovea = original_marked.shape[-1] - 1 - fovea_x if mirrored else fovea_x
                np.testing.assert_array_equal(np.flatnonzero(marked[0, 0] == 243), [expected_fovea])
                self.assertGreater(np.count_nonzero(marked == 255), 0)
                self.assertGreater(np.count_nonzero(marked == 254), 0)

    def test_both_left_side_answers_save_correct_analyze_folders_and_aligned_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            for left_side in (IMAGE_SIDE_NASAL, IMAGE_SIDE_TEMPORAL):
                with self.subTest(left_side=left_side):
                    folder = Path(temporary) / left_side
                    frame = self._disk_frame(folder, left_side)
                    original_image = frame.image_data.copy()
                    original_annotations = frame._annotation_snapshot()
                    marked = frame._build_current_marked_volume()
                    source = np.stack([frame.image_data] * 2)

                    pair = frame._save_current_marked_orientation_pair()

                    self.assertEqual(pair, (str(folder / "nasal" / "Light_MARKED"),
                                            str(folder / "temporal" / "Light_MARKED")))
                    self._assert_saved_pair(folder, left_side, marked, source, frame.fovea_x)
                    np.testing.assert_array_equal(frame.image_data, original_image)
                    np.testing.assert_array_equal(read_analyze(str(folder / "Light")), source)
                    self.assertEqual(frame._annotation_snapshot(), original_annotations)

    def test_inactive_disk_save_uses_its_own_side_and_restores_active_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            for inactive_side in (IMAGE_SIDE_NASAL, IMAGE_SIDE_TEMPORAL):
                with self.subTest(inactive_left_side=inactive_side):
                    active_side = IMAGE_SIDE_TEMPORAL if inactive_side == IMAGE_SIDE_NASAL else IMAGE_SIDE_NASAL
                    root = Path(temporary) / inactive_side
                    active = self._disk_frame(root / "active", active_side, fovea_x=531)
                    inactive = self._disk_frame(root / "inactive", inactive_side)
                    original_file = active.current_file
                    original_image = active.image_data
                    original_annotations = active._annotation_snapshot()
                    original_orientation = active._current_save_orientation
                    inactive_marked = inactive._build_current_marked_volume()
                    inactive_source = np.stack([inactive.image_data] * 2)
                    active._active_batch_result_tab = "active"
                    active._batch_result_states = {"inactive": {
                        "input": inactive.current_file,
                        "image": inactive.image_data,
                        "traces": inactive.boundary_traces,
                        "order": inactive.boundary_order,
                        "fovea_x": inactive.fovea_x,
                        "save_orientation": inactive._current_save_orientation,
                    }}

                    active._save_batch_result_state("inactive", save_orientation_pair=True)

                    self._assert_saved_pair(root / "inactive", inactive_side,
                                            inactive_marked, inactive_source, inactive.fovea_x)
                    self.assertEqual(active.current_file, original_file)
                    self.assertIs(active.image_data, original_image)
                    self.assertEqual(active._annotation_snapshot(), original_annotations)
                    self.assertEqual(active._current_save_orientation, original_orientation)
                    self.assertFalse((root / "active" / "nasal").exists())
                    self.assertFalse((root / "active" / "temporal").exists())

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
