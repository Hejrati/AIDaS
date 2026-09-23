from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from aidas.steps.step4_analyze_isez import (
    AUTO_CONFIDENCE_THRESHOLD,
    AUTO_MANUAL_ONLY_ROI_INDEX,
    ISezROI,
    ProfileBoundaryDetection,
    Step4Frame,
    apply_profile_detection_consistency,
    detect_profile_boundaries,
)
from aidas.core.config import STEP4_AUTO_DETECTION_DEFAULTS


def _bell_profile() -> np.ndarray:
    samples = np.arange(1, 151, dtype=np.float64)
    profile = np.full(samples.size, 12.0)
    bell = (samples >= 76) & (samples <= 100)
    profile[bell] = 25.0 - 0.12 * (samples[bell] - 88.0) ** 2
    return profile


def _detection(start: int, end: int, *, accepted: bool = True) -> ProfileBoundaryDetection:
    return ProfileBoundaryDetection(
        start=start,
        peak=(start + end) // 2,
        end=end,
        confidence=0.95,
        accepted=accepted,
        prominence=8.0,
        quadratic_r2=0.9,
        reason="accepted" if accepted else "review",
    )


class _VariableStub:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


class Step4AutomaticBoundaryDetectionTests(unittest.TestCase):
    def test_detects_two_minima_around_a_quadratic_bell(self):
        result = detect_profile_boundaries(_bell_profile())

        self.assertEqual((result.start, result.peak, result.end), (76, 88, 100))
        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.confidence, AUTO_CONFIDENCE_THRESHOLD)
        self.assertGreaterEqual(result.quadratic_r2, 0.9)

    def test_interpolates_non_finite_samples_without_moving_the_bell(self):
        profile = _bell_profile()
        profile[[82, 87, 93]] = np.nan

        result = detect_profile_boundaries(profile)

        self.assertTrue(result.accepted)
        self.assertLessEqual(abs(result.start - 76), 1)
        self.assertLessEqual(abs(result.end - 100), 1)

    def test_monotonic_profile_is_returned_for_manual_review(self):
        result = detect_profile_boundaries(np.linspace(1.0, 20.0, 150))

        self.assertFalse(result.accepted)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("no valid", result.reason)

    def test_cross_roi_consistency_rejects_a_boundary_outlier(self):
        detections = [
            _detection(77 + index % 3, 99 + index % 3)
            for index in range(20)
        ]
        detections.append(_detection(90, 110))

        results = apply_profile_detection_consistency(detections)

        self.assertTrue(all(item.accepted for item in results[:-1]))
        self.assertFalse(results[-1].accepted)
        self.assertIn("consensus", results[-1].reason)

    def test_cross_roi_consistency_tolerance_is_measured_in_samples(self):
        detections = [
            _detection(100, 120),
            _detection(100, 120),
            _detection(100, 120),
            _detection(108, 120),
        ]

        permissive = apply_profile_detection_consistency(
            detections,
            minimum_tolerance=10.0,
        )
        strict = apply_profile_detection_consistency(
            detections,
            minimum_tolerance=5.0,
        )

        self.assertTrue(permissive[-1].accepted)
        self.assertFalse(strict[-1].accepted)

    def test_auto_detect_accepts_confident_roi_and_marks_uncertain_roi(self):
        frame = object.__new__(Step4Frame)
        frame.image = object()
        frame.rois = [ISezROI("01", 1, 2), ISezROI("02", 3, 4)]
        frame.completed = {}
        frame.auto_detection_reviews = {}
        frame.roi_clicks = {}
        frame.current_roi_idx = 0
        frame._auto_detecting = False
        frame._stack_build_complete = False
        frame.batch_roi_notebook = None
        frame._active_batch_roi_tab = None
        frame.status_var = _VariableStub()
        frame._close_profile_zoom = mock.Mock()
        frame._cancel_roi_update_animations = mock.Mock()
        frame._update_auto_detect_button_state = mock.Mock()
        frame._update_build_stack_button_state = mock.Mock()
        frame.update_idletasks = mock.Mock()
        frame._load_current_roi_clicks = mock.Mock()
        frame._refresh_roi_list = mock.Mock()
        frame._select_roi_in_list = mock.Mock()
        frame._render_current_roi = mock.Mock()
        custom_parameters = dict(STEP4_AUTO_DETECTION_DEFAULTS)
        custom_parameters.update(
            step4_auto_start_min=72,
            step4_auto_start_max=91,
            step4_auto_end_min=92,
            step4_auto_end_max=112,
            step4_auto_savgol_window=11,
            step4_auto_confidence_percent=70.0,
            step4_auto_min_quadratic_r2=0.6,
            step4_auto_consistency_tolerance=8.0,
        )
        frame.preferences = SimpleNamespace(
            get=lambda key, default=None: custom_parameters.get(key, default)
        )

        accepted = _detection(78, 100)
        uncertain = _detection(79, 101, accepted=False)
        saved = SimpleNamespace(start=78, end=100)
        with mock.patch(
            "aidas.steps.step4_analyze_isez.intensity_profile",
            return_value=_bell_profile(),
        ), mock.patch(
            "aidas.steps.step4_analyze_isez.detect_profile_boundaries",
            side_effect=[accepted, uncertain],
        ) as detect, mock.patch(
            "aidas.steps.step4_analyze_isez.apply_profile_detection_consistency",
            side_effect=lambda values, **_kwargs: values,
        ) as consistency, mock.patch(
            "aidas.steps.step4_analyze_isez.analyze_and_save_roi",
            return_value=saved,
        ):
            frame._auto_detect_all_rois()

        self.assertIs(frame.completed["01"], saved)
        self.assertNotIn("02", frame.completed)
        self.assertIs(frame.auto_detection_reviews["02"], uncertain)
        self.assertEqual(frame.roi_clicks["02"], [79.0, 101.0])
        self.assertEqual(frame.current_roi_idx, 1)
        self.assertIn("amber", frame.status_var.value)
        self.assertFalse(frame._auto_detecting)
        self.assertEqual(
            detect.call_args.kwargs,
            {
                "start_range": (72, 91),
                "end_range": (92, 112),
                "smoothing_window": 11,
                "confidence_threshold": 0.7,
                "minimum_quadratic_r2": 0.6,
            },
        )
        self.assertEqual(consistency.call_args.kwargs["minimum_tolerance"], 8.0)

    def test_auto_detect_never_processes_roi_21(self):
        frame = object.__new__(Step4Frame)
        frame.image = object()
        frame.rois = [ISezROI(f"{index + 1:02d}", index * 2 + 1, index * 2 + 2) for index in range(21)]
        frame.completed = {}
        frame.auto_detection_reviews = {}
        frame.roi_clicks = {}
        frame.current_roi_idx = 0
        frame._auto_detecting = False
        frame._stack_build_complete = False
        frame.batch_roi_notebook = None
        frame._active_batch_roi_tab = None
        frame.status_var = _VariableStub()
        frame._close_profile_zoom = mock.Mock()
        frame._cancel_roi_update_animations = mock.Mock()
        frame._update_auto_detect_button_state = mock.Mock()
        frame._update_build_stack_button_state = mock.Mock()
        frame.update_idletasks = mock.Mock()
        frame._load_current_roi_clicks = mock.Mock()
        frame._refresh_roi_list = mock.Mock()
        frame._select_roi_in_list = mock.Mock()
        frame._render_current_roi = mock.Mock()

        detection = _detection(78, 100)
        saved = SimpleNamespace(start=78, end=100)
        with mock.patch(
            "aidas.steps.step4_analyze_isez.intensity_profile",
            return_value=_bell_profile(),
        ) as profile, mock.patch(
            "aidas.steps.step4_analyze_isez.detect_profile_boundaries",
            return_value=detection,
        ) as detect, mock.patch(
            "aidas.steps.step4_analyze_isez.analyze_and_save_roi",
            return_value=saved,
        ) as analyze:
            frame._auto_detect_all_rois()

        manual_roi = frame.rois[AUTO_MANUAL_ONLY_ROI_INDEX]
        self.assertEqual(profile.call_count, 20)
        self.assertEqual(detect.call_count, 20)
        self.assertEqual(analyze.call_count, 20)
        self.assertNotIn(manual_roi.suffix, frame.completed)
        self.assertNotIn(manual_roi.suffix, frame.roi_clicks)
        self.assertIn(manual_roi.suffix, frame.auto_detection_reviews)
        self.assertIn("must be selected manually", frame.auto_detection_reviews[manual_roi.suffix])
        self.assertEqual(frame.current_roi_idx, AUTO_MANUAL_ONLY_ROI_INDEX)

    def test_readme_documents_ranges_scoring_and_manual_review(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

        for text in (
            "Auto-detect ROIs 1-20",
            "ROI 21 is deliberately excluded",
            "70-90",
            "90-110",
            "Savitzky-Golay",
            "66%",
            "amber",
        ):
            with self.subTest(text=text):
                self.assertIn(text, readme)

    def test_savitzky_golay_polynomial_order_is_fixed_at_degree_two(self):
        with mock.patch(
            "scipy.signal.savgol_filter",
            side_effect=lambda values, **_kwargs: values,
        ) as smoother:
            detect_profile_boundaries(_bell_profile(), smoothing_window=11)

        self.assertEqual(smoother.call_args.kwargs["polyorder"], 2)


if __name__ == "__main__":
    unittest.main()
