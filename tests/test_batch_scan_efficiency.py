from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import Mock, patch

from aidas.steps.step2_annotate import Step2Frame, Step2BatchSegmentationTable
from aidas.steps.step3_flatten import Step3Frame, RBatchSelectionPanel, RBatchSelectionTable
from aidas.steps.step4_analyze_isez import Step4Frame, Step4BatchROITable
from aidas.utils.filesystem import find_candidate_directories


class BatchScanEfficiencyTests(unittest.TestCase):
    def test_repeated_status_values_are_measured_once_per_font_and_padding(self):
        for table_type in (Step2BatchSegmentationTable, RBatchSelectionTable, Step4BatchROITable):
            with self.subTest(table=table_type.__name__):
                table = table_type.__new__(table_type)
                table._tree_font = Mock()
                table._tree_font.measure.return_value = 45
                table._heading_font = Mock()
                table._heading_font.measure.return_value = 50
                for _ in range(3000):
                    self.assertEqual(table._measure_text("Ready"), 63)
                table._tree_font.measure.assert_called_once_with("Ready")
                self.assertEqual(table._measure_text("Ready", heading=True), 68)
                self.assertEqual(table._measure_text("Ready", padding=20), 65)

    def test_sparse_tree_preserves_counts_and_validates_only_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(30):
                (root / str(index)).mkdir()
            study = root / "10" / "study"
            study.mkdir()
            for name in ("Light.img", "Light.hdr", "Light_MARKED.img", "Light_MARKED.hdr", "_flat_LIGHT.hdr"):
                (study / name).touch()
            # A directory with a matching filename must not count as input.
            (root / "20" / "Light.img").mkdir()
            step2 = Step2Frame.__new__(Step2Frame)
            with patch.object(step2, "_step2_batch_segmentation_row", wraps=step2._step2_batch_segmentation_row) as check:
                rows, scanned, skipped, errors = step2._scan_step2_batch_segmentation_folders(root)
            self.assertEqual(check.call_count, 1)
            self.assertEqual((scanned, skipped, errors), (33, 32, []))
            self.assertEqual([Path(row["folder"]) for row in rows], [study])

            step4 = Step4Frame.__new__(Step4Frame)
            with patch.object(step4, "_flat_light_path_for_folder", wraps=step4._flat_light_path_for_folder) as check:
                rows, scanned, skipped, errors = step4._scan_batch_roi_folders(root)
            self.assertEqual(check.call_count, 1)
            self.assertEqual((scanned, skipped, errors), (33, 32, []))
            self.assertEqual([row["folder"] for row in rows], [study])

            panel = RBatchSelectionPanel.__new__(RBatchSelectionPanel)
            panel.step_frame = Step3Frame.__new__(Step3Frame)
            panel.input_folders = None
            panel.root_dir = root
            panel._scan_events = queue.Queue()
            with patch.object(panel.step_frame, "_find_input_paths", wraps=panel.step_frame._find_input_paths) as check:
                panel._scan_worker()
            self.assertEqual(check.call_count, 1)
            kind, (rows, scanned, skipped, errors) = panel._scan_events.get_nowait()
            self.assertEqual(kind, "done")
            self.assertEqual((scanned, skipped, errors), (33, 32, []))
            self.assertEqual([row["folder"] for row in rows], [study])

    def test_prefilter_is_case_insensitive_and_does_not_cache_rescans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "LIGHT.IMG"
            self.assertEqual(find_candidate_directories(root, (("light.img",),))[0], [])
            path.touch()
            self.assertEqual(find_candidate_directories(root, (("light.img",),))[0], [root])
            path.unlink()
            self.assertEqual(find_candidate_directories(root, (("light.img",),))[0], [])

    def test_prefilter_records_unreadable_branches_and_continues(self):
        def walk(root, *, topdown, onerror):
            onerror(PermissionError(13, "denied", str(root / "denied")))
            yield str(root), ["study"], []
            yield str(root / "study"), [], ["Light.img"]

        with patch("aidas.utils.filesystem.os.walk", side_effect=walk):
            candidates, count, errors = find_candidate_directories(Path("root"), (("light.img",),))
        self.assertEqual(candidates, [Path("root/study")])
        self.assertEqual(count, 2)
        self.assertEqual(errors[0][0], Path("root/denied"))


if __name__ == "__main__":
    unittest.main()
