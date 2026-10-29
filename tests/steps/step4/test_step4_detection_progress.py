from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import threading
import unittest
from unittest import mock

import numpy as np

from aidas.core.config import STEP4_AUTO_DETECTION_DEFAULTS
from aidas.steps import step4_analyze_isez as step4


class Step4DetectionProgressTests(unittest.TestCase):
    def _run_mixed_rois(self, progress_callback=None):
        rois = step4.default_isez_rois()
        accepted = step4.ProfileBoundaryDetection(
            start=78, peak=88, end=100, confidence=0.95,
            accepted=True, prominence=10.0, quadratic_r2=0.99,
            reason="accepted",
        )

        def profile(_image, roi):
            if roi.suffix == "01":
                raise ValueError("Profile unavailable")
            return np.full(151, int(roi.suffix), dtype=float)

        def detect(values, **_parameters):
            if values[0] == 2:
                raise ValueError("Detection unavailable")
            if values[0] == 4:
                return replace(accepted, accepted=False, reason="Review needed")
            return accepted

        def analyze(_image, roi, **_parameters):
            if roi.suffix == "05":
                raise ValueError("Measurement unavailable")
            return SimpleNamespace(start=78, end=100)

        with (
            mock.patch.object(step4, "intensity_profile", side_effect=profile) as extract,
            mock.patch.object(step4, "detect_profile_boundaries", side_effect=detect),
            mock.patch.object(
                step4, "apply_profile_detection_consistency",
                side_effect=lambda values, **_parameters: values,
            ),
            mock.patch.object(step4, "analyze_and_save_roi", side_effect=analyze) as measure,
        ):
            result = step4.Step4Frame._compute_auto_detection_run(
                object(), rois, {"03"}, dict(STEP4_AUTO_DETECTION_DEFAULTS),
                progress_callback,
            )
        return result, extract, measure

    def test_progress_finishes_with_identical_results_for_success_skip_and_error_paths(self):
        expected, _, _ = self._run_mixed_rois()
        progress = []
        actual, extract, measure = self._run_mixed_rois(
            lambda done, total: progress.append((done, total))
        )

        self.assertEqual(actual, expected)
        self.assertEqual(progress, [(done, 42) for done in range(43)])
        self.assertEqual(extract.call_count, 20)
        self.assertEqual(measure.call_count, 16)
        self.assertEqual(actual.reviews["01"], "Profile unavailable")
        self.assertEqual(actual.reviews["02"], "Detection unavailable")
        self.assertEqual(actual.reviews["04"].reason, "Review needed")
        self.assertEqual(actual.reviews["05"].reason, "measurement failed: Measurement unavailable")
        self.assertNotIn("03", actual.completed)
        self.assertNotIn("03", actual.roi_clicks)
        self.assertIn("must be selected manually", actual.reviews["fovea"])
        self.assertNotIn("fovea", actual.completed)
        self.assertNotIn("fovea", actual.roi_clicks)
        self.assertEqual(actual.current_roi_idx, 0)

    def test_progress_observer_failures_cannot_change_detection_results(self):
        expected, _, _ = self._run_mixed_rois()
        calls = []

        def unavailable_observer(done, total):
            calls.append((done, total))
            raise RuntimeError("Progress display was closed")

        actual, _, _ = self._run_mixed_rois(unavailable_observer)
        self.assertEqual(actual, expected)
        self.assertEqual(calls[-1], (42, 42))

    def test_progress_runs_on_the_calling_worker_without_tk(self):
        owner = threading.get_ident()
        observed_threads = []
        with ThreadPoolExecutor(max_workers=1) as workers:
            result, _, _ = workers.submit(
                self._run_mixed_rois,
                lambda _done, _total: observed_threads.append(threading.get_ident()),
            ).result(timeout=10)
        self.assertTrue(result.completed)
        self.assertEqual(len(observed_threads), 43)
        self.assertEqual(len(set(observed_threads)), 1)
        self.assertNotEqual(observed_threads[0], owner)

    def test_empty_roi_list_reports_zero_work(self):
        progress = []
        result = step4.Step4Frame._compute_auto_detection_run(
            object(), [], set(), dict(STEP4_AUTO_DETECTION_DEFAULTS),
            lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(progress, [(0, 0)])
        self.assertEqual(result.completed, {})
        self.assertEqual(result.reviews, {})
        self.assertEqual(result.review_count, 0)
        self.assertEqual(result.current_roi_idx, 0)


if __name__ == "__main__":
    unittest.main()
