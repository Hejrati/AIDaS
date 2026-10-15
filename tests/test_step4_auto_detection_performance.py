from dataclasses import fields
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from aidas.core.config import STEP4_AUTO_DETECTION_DEFAULTS
from aidas.steps import step4_analyze_isez as step4


def original_profile(image, roi):
    values = np.asarray(image, dtype=np.float64) / 3.0
    left, right, low, high = step4.roi_bounds_for_image(roi, values.shape)
    return np.nanmean(values[low:high, left:right], axis=1)


class AutomaticROIPerformanceTests(unittest.TestCase):
    def test_profiles_are_identical_for_integer_float_and_strided_images(self):
        rng = np.random.default_rng(38)
        roi = step4.ISezROI("test", 7, 29, low=3, high=80)
        for dtype in (np.int16, np.uint16, np.float32, np.float64):
            image = rng.integers(0, 30000, size=(100, 160)).astype(dtype)
            if np.issubdtype(dtype, np.floating):
                image[20, 10] = np.nan
            for source in (image, image[:, ::2], np.asfortranarray(image)):
                original = source.copy()
                np.testing.assert_array_equal(step4.intensity_profile(source, roi), original_profile(source, roi))
                np.testing.assert_array_equal(source, original)

    def test_float_conversion_is_limited_to_the_roi(self):
        image = np.ones((768, 3000), dtype=np.uint16)
        roi = step4.default_isez_rois()[0]
        with patch.object(step4.np, "asarray", wraps=np.asarray) as convert:
            step4.intensity_profile(image, roi)
        conversions = [call.args[0] for call in convert.call_args_list
                       if call.kwargs.get("dtype") is np.float64]
        self.assertEqual(len(conversions), 1)
        self.assertEqual(conversions[0].shape, (151, 120))

    def test_full_run_matches_original_profiles_and_repeated_measurement(self):
        samples = np.arange(1, 152, dtype=np.float64)
        profile = np.full(samples.size, 12.0)
        bell = (samples >= 76) & (samples <= 100)
        profile[bell] = 25.0 - 0.12 * (samples[bell] - 88.0) ** 2
        rng = np.random.default_rng(74)
        image = profile[:, None] * 3 + rng.normal(0, 0.7, size=(151, 600))
        rois = [step4.ISezROI(str(i), i * 20 + 1, i * 20 + 20, low=1, high=151) for i in range(21)]
        analyze = step4.analyze_and_save_roi

        def original_analysis(image, roi, **kwargs):
            # Reproduce the original second profile extraction on acceptance.
            kwargs.pop("profile", None)
            return analyze(image, roi, profile=original_profile(image, roi), **kwargs)

        for completed in (set(), {"0", "4"}):
            parameters = dict(STEP4_AUTO_DETECTION_DEFAULTS)
            with patch.object(step4, "intensity_profile", side_effect=original_profile), \
                    patch.object(step4, "analyze_and_save_roi", side_effect=original_analysis):
                expected = step4.Step4Frame._compute_auto_detection_run(image, rois, completed, parameters)
            with patch.object(step4, "intensity_profile", wraps=step4.intensity_profile) as extraction:
                actual = step4.Step4Frame._compute_auto_detection_run(image, rois, completed, parameters)
            self.assertEqual(extraction.call_count, 20)
            self.assertEqual(actual.reviews, expected.reviews)
            self.assertEqual(actual.roi_clicks, expected.roi_clicks)
            self.assertEqual(actual.removed_clicks, expected.removed_clicks)
            self.assertEqual(actual.current_roi_idx, expected.current_roi_idx)
            self.assertEqual(actual.review_count, expected.review_count)
            self.assertEqual(actual.completed.keys(), expected.completed.keys())
            self.assertGreater(len(actual.completed), 0)
            self.assertNotIn(rois[20].suffix, actual.completed)
            for suffix, result in actual.completed.items():
                reference = expected.completed[suffix]
                for field in fields(result):
                    observed, original = getattr(result, field.name), getattr(reference, field.name)
                    if isinstance(observed, np.ndarray):
                        np.testing.assert_array_equal(observed, original)
                    else:
                        self.assertEqual(observed, original)

    def test_smoother_preparation_queues_only_one_worker_and_never_needs_tk(self):
        frame = step4.Step4Frame.__new__(step4.Step4Frame)
        workers = []
        frame._start_background_worker = workers.append
        with patch.dict(step4.sys.modules, {"scipy.signal": SimpleNamespace()}):
            frame._warm_auto_detection_tools()
            frame._warm_auto_detection_tools()
        self.assertEqual(len(workers), 1)
        # The queued function only imports; it does not read frame state or Tk.
        workers[0]()


if __name__ == "__main__":
    unittest.main()
