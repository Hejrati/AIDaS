from __future__ import annotations

import unittest

from aidas.steps.step2_annotate import Step2Frame
from aidas.steps.step3_flatten import Step3Frame
from aidas.steps.step4_analyze_isez import Step4Frame


class BatchTabLabelTests(unittest.TestCase):
    def test_step2_title_does_not_embed_a_close_character(self):
        self.assertEqual(Step2Frame._batch_result_tab_text("test1"), "test1")

    def test_step3_groups_temporal_and_nasal_results_by_subject(self):
        groups = Step3Frame._group_result_folders(
            ["test1/temporal", "test1/nasal"],
        )

        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["sided"])
        self.assertEqual(groups[0]["label"], "test1")
        self.assertEqual(groups[0]["folders"]["temporal"].name, "temporal")
        self.assertEqual(groups[0]["folders"]["nasal"].name, "nasal")

    def test_step4_folder_title_stays_fixed_as_roi_and_stack_work_finishes(self):
        step = object.__new__(Step4Frame)
        step.batch_roi_notebook = None
        step.rois = ("A", "B", "C")

        state = {"folder": None, "base_label": "1. test1", "completed": {}}
        self.assertEqual(step._batch_roi_tab_text(state), "1. test1")
        state["completed"] = dict.fromkeys(step.rois)
        self.assertEqual(step._batch_roi_tab_text(state), "1. test1")
        state["complete"] = True
        self.assertEqual(step._batch_roi_tab_text(state), "1. test1")


if __name__ == "__main__":
    unittest.main()
