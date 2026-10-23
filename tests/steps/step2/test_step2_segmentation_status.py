from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aidas.steps.step2_annotate import Step2Frame


class Step2SegmentationStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame = Step2Frame.__new__(Step2Frame)

    def _source(self, name="source"):
        folder = self.root / name
        folder.mkdir()
        for filename in ("Light.img", "Light.hdr"):
            (folder / filename).write_bytes(b"source")
        return folder

    @staticmethod
    def _segmented(folder):
        folder.mkdir(exist_ok=True)
        for filename in ("Light.img", "Light.hdr", "Light_MARKED.img", "Light_MARKED.hdr"):
            (folder / filename).write_bytes(b"saved")

    def test_both_saved_sides_disable_parent_source_in_scanner(self):
        folder = self._source()
        for side in ("nasal", "temporal"):
            self._segmented(folder / side)

        rows, _scanned, _skipped, errors = self.frame._scan_step2_batch_segmentation_folders(self.root)

        self.assertEqual(errors, [])
        parent = next(row for row in rows if Path(row["folder"]) == folder)
        self.assertTrue(parent["locked"])
        self.assertFalse(parent["include"])
        self.assertEqual(parent["image_paths"], [])
        self.assertEqual(parent["status"], "Already segmented: nasal and temporal")
        self.assertEqual(parent["segmented_sides"], ["nasal", "temporal"])
        self.assertEqual(parent["completed"], ["Light"])
        self.assertTrue(all(row["locked"] for row in rows))

    def test_one_saved_side_stays_selectable_and_identifies_the_side(self):
        for side in ("nasal", "temporal"):
            with self.subTest(side=side):
                folder = self._source(side)
                self._segmented(folder / side)

                row = self.frame._step2_batch_segmentation_row(folder)

                self.assertFalse(row["locked"])
                self.assertTrue(row["include"])
                self.assertEqual(row["status"], f"Ready: Light; segmented: {side} only")
                self.assertEqual(row["segmented_sides"], [side])
                self.assertEqual(row["pending"], ["Light"])
                self.assertEqual(row["image_paths"], [str(folder / "Light.img")])

    def test_side_folders_without_segmented_images_remain_ready(self):
        folder = self._source()
        for side in ("nasal", "temporal"):
            side_folder = folder / side
            side_folder.mkdir()
            (side_folder / "Light_MARKED.hdr").write_bytes(b"incomplete save")
            (side_folder / "DARK_MARKED_vertex.png").write_bytes(b"preview")

        row = self.frame._step2_batch_segmentation_row(folder)

        self.assertFalse(row["locked"])
        self.assertEqual(row["status"], "Ready: Light")
        self.assertEqual(row["segmented_sides"], [])

    def test_saved_sides_are_detected_with_mixed_case_names(self):
        folder = self._source()
        for side in ("NaSaL", "TEMPORAL"):
            side_folder = folder / side
            side_folder.mkdir()
            (side_folder / "LIGHT_MARKED.IMG").write_bytes(b"saved")

        row = self.frame._step2_batch_segmentation_row(folder)

        self.assertTrue(row["locked"])
        self.assertEqual(row["status"], "Already segmented: nasal and temporal")

    def test_existing_marked_images_next_to_source_remain_processed(self):
        folder = self._source()
        self._segmented(folder)

        row = self.frame._step2_batch_segmentation_row(folder)

        self.assertTrue(row["locked"])
        self.assertEqual(row["status"], "Already segmented: Light")

    def test_step1_handoff_uses_same_side_detection_for_explicit_folders(self):
        both = self._source("both")
        nasal = self._source("nasal-only")
        for side in ("nasal", "temporal"):
            self._segmented(both / side)
        self._segmented(nasal / "nasal")
        self.frame._segmenter_running = False
        self.frame.aidas_model_path = str(self.root / "model.onnx")
        Path(self.frame.aidas_model_path).write_bytes(b"test model")
        self.frame.status_var = mock.Mock()
        self.frame._open_step2_batch_segmentation_panel = mock.Mock(return_value=True)

        self.frame.start_batch_segmentation_for_folders([both, nasal])

        rows = self.frame._open_step2_batch_segmentation_panel.call_args.kwargs["initial_rows"]
        self.assertTrue(rows[0]["locked"])
        self.assertFalse(rows[1]["locked"])
        self.assertEqual(rows[0]["status"], "Already segmented: nasal and temporal")
        self.assertEqual(rows[1]["status"], "Ready: Light; segmented: nasal only")


if __name__ == "__main__":
    unittest.main()
