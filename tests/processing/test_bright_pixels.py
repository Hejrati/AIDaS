from __future__ import annotations

import unittest

import numpy as np

from aidas.utils.bright_pixels import (
    detect_bright_pixels,
    detect_bright_pixels_in_bands,
    smooth_bright_pixels,
)


class BrightPixelDetectionTests(unittest.TestCase):
    def test_signed_oct_histogram_detects_isolated_positive_artifacts(self):
        image = np.linspace(-15000, -8000, 128 * 128).reshape(128, 128).astype(np.int16)
        image[:7] = 0
        image[12, 19] = 32005
        image[28, 94] = 32555
        original = image.copy()

        detection = detect_bright_pixels(image)

        self.assertEqual(detection.count, 2)
        self.assertTrue(detection.mask[12, 19])
        self.assertTrue(detection.mask[28, 94])
        self.assertGreater(detection.threshold, 0)
        self.assertFalse(np.any(detection.mask[:7]))
        np.testing.assert_array_equal(image, original)

    def test_wide_oct_intensity_spread_still_detects_all_separated_artifacts(self):
        # Reproduce the wider supplied sample's quartiles, where requiring
        # six IQRs above the median would miss its least-bright defect.
        values = (-19000, -14998, -11185, -8508, -6000)
        image = np.interp(
            np.linspace(0, 1, 200 * 200), (0, 0.25, 0.5, 0.75, 1), values
        ).reshape(200, 200).astype(np.int16)
        image[:17] = 0
        positions = ((25, 36), (42, 84), (67, 98), (90, 143))
        for position, value in zip(positions, (26777, 28287, 32005, 32555)):
            image[position] = value

        detection = detect_bright_pixels(image)

        self.assertEqual(detection.count, 4)
        self.assertTrue(all(detection.mask[position] for position in positions))

    def test_broad_full_width_oct_distribution_detects_all_isolated_artifacts(self):
        # Nonzero quartiles follow the supplied full-width crop at y=500.
        # Its tissue distribution is wider than the first, tightly cropped
        # example, and zero padding occupies about one sixth of the image.
        image = np.interp(
            np.linspace(0, 1, 108 * 128),
            (0, 0.25, 0.5, 0.75, 1),
            (-32737, -18324, -16962, -9929, -1),
        ).reshape(108, 128).astype(np.int16)
        image = np.pad(image, ((20, 0), (0, 0)))
        positions = ((25, 36), (42, 84), (67, 98), (90, 113))
        for position, value in zip(positions, (26777, 28287, 32005, 32555)):
            image[position] = value
        original = image.copy()
        expected = np.zeros(image.shape, dtype=bool)
        for position in positions:
            expected[position] = True

        detection = detect_bright_pixels(image)

        np.testing.assert_array_equal(detection.mask, expected)
        np.testing.assert_array_equal(image, original)
        repaired = smooth_bright_pixels(image, detection.mask)
        np.testing.assert_array_equal(repaired[~expected], original[~expected])

    def test_bimodal_signed_tissue_retains_both_modes_and_detects_bright_artifacts(self):
        image = np.concatenate((
            np.linspace(-19000, -16000, 128 * 64),
            np.linspace(-10500, -5000, 128 * 64),
        )).reshape(128, 128).astype(np.int16)

        # Two ordinary tissue populations must not trigger a repair prompt.
        self.assertEqual(detect_bright_pixels(image).count, 0)
        positions = ((12, 19), (28, 94), (73, 45), (99, 68))
        for position, value in zip(positions, (26777, 28287, 32005, 32555)):
            image[position] = value
        expected = np.zeros(image.shape, dtype=bool)
        for position in positions:
            expected[position] = True

        detection = detect_bright_pixels(image)

        np.testing.assert_array_equal(detection.mask, expected)

    def test_padding_dominated_crop_retains_moderate_highlights_and_detects_artifacts(self):
        image = np.zeros((128, 128), dtype=np.int16)
        image[-24:] = np.linspace(-19000, -5000, 24 * 128).reshape(24, 128)
        image[108:110, 60:63] = 1000
        image[113, 39] = 32005
        image[121, 92] = 32555
        original = image.copy()
        expected = np.zeros(image.shape, dtype=bool)
        expected[113, 39] = True
        expected[121, 92] = True

        detection = detect_bright_pixels(image)

        np.testing.assert_array_equal(detection.mask, expected)
        repaired = smooth_bright_pixels(image, detection.mask)
        np.testing.assert_array_equal(repaired[~expected], original[~expected])
        np.testing.assert_array_equal(repaired[108:110, 60:63], original[108:110, 60:63])

    def test_signed_bright_band_is_retained_while_isolated_artifacts_are_detected(self):
        image = np.linspace(-20000, -5000, 256 * 256).reshape(256, 256).astype(np.int16)
        image[:8] = 0
        image[100, 30:226] = 25000
        image[12, 19] = 32005
        image[28, 94] = 32555
        expected = np.zeros(image.shape, dtype=bool)
        expected[12, 19] = True
        expected[28, 94] = True

        detection = detect_bright_pixels(image)

        np.testing.assert_array_equal(detection.mask, expected)
        repaired = smooth_bright_pixels(image, detection.mask)
        np.testing.assert_array_equal(repaired[100, 30:226], image[100, 30:226])

    def test_continuous_histogram_does_not_remove_ordinary_bright_tissue(self):
        image = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)

        detection = detect_bright_pixels(image)

        self.assertEqual(detection.count, 0)
        self.assertIsNone(detection.threshold)

    def test_thin_bright_anatomical_band_is_protected(self):
        image = np.full((256, 256), 1000, dtype=np.uint16)
        image[100, 30:226] = 30000

        self.assertEqual(detect_bright_pixels(image).count, 0)

    def test_diagonal_bright_anatomical_band_is_protected(self):
        image = np.full((256, 256), 1000, dtype=np.uint16)
        coordinates = np.arange(30, 226)
        image[coordinates, coordinates] = 30000

        self.assertEqual(detect_bright_pixels(image).count, 0)

    def test_small_adjacent_blob_is_detected(self):
        image = np.full((128, 128), 1100, dtype=np.uint16)
        image[45:49, 60:64] = 65000

        detection = detect_bright_pixels(image)

        self.assertEqual(detection.count, 16)
        self.assertTrue(np.all(detection.mask[45:49, 60:64]))

    def test_zero_padding_above_negative_tissue_is_retained(self):
        image = np.full((64, 64), -10000, dtype=np.int16)
        image[0, :2] = 0

        self.assertEqual(detect_bright_pixels(image).count, 0)

    def test_empty_flat_and_nonfinite_images_have_no_detection(self):
        for image in (
            np.empty((0, 0), dtype=np.uint16),
            np.full((64, 64), 65535, dtype=np.uint16),
            np.full((16, 16), np.nan),
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                detection = detect_bright_pixels(image)
                self.assertEqual(detection.mask.shape, image.shape)
                self.assertEqual(detection.count, 0)

    def test_extreme_finite_floats_are_detected_without_nonfinite_samples(self):
        image = np.full((64, 64), -1e308)
        image[10, 10] = 1e308
        image[20, 20] = np.nan
        image[30, 30] = np.inf

        with np.errstate(over="raise", invalid="raise"):
            detection = detect_bright_pixels(image)

        self.assertEqual(detection.count, 1)
        self.assertTrue(detection.mask[10, 10])
        self.assertTrue(np.isfinite(detection.threshold))


class BrightPixelBandDetectionTests(unittest.TestCase):
    def test_source_bands_detect_body_defects_hidden_by_positive_background(self):
        image = np.empty((512, 128), dtype=np.int16)
        image[:128] = np.linspace(1, 32767, 128 * 128).reshape(128, 128)
        image[128:] = np.linspace(-20000, -7000, 384 * 128).reshape(384, 128)
        positions = ((260, 31), (280, 17), (310, 93), (340, 61))
        for position, value in zip(positions, (26777, 28287, 32005, 32555)):
            image[position] = value
        original = image.copy()
        expected = np.zeros(image.shape, dtype=bool)
        for position in positions:
            expected[position] = True
        self.assertEqual(detect_bright_pixels(image).count, 0)

        detection = detect_bright_pixels_in_bands(image)

        np.testing.assert_array_equal(detection.mask, expected)
        self.assertIsInstance(detection.threshold, float)
        self.assertFalse(detection.mask.flags.writeable)
        np.testing.assert_array_equal(image, original)
        repaired = smooth_bright_pixels(image, detection.mask)
        np.testing.assert_array_equal(repaired[~expected], original[~expected])

    def test_overlapping_bands_cover_defects_at_a_background_boundary(self):
        image = np.empty((384, 128), dtype=np.int16)
        image[:64] = np.linspace(1, 32767, 64 * 128).reshape(64, 128)
        image[64:] = np.linspace(-17000, -8000, 320 * 128).reshape(320, 128)
        image[127, 20] = 32005
        image[128, 30] = 32555
        expected = np.zeros(image.shape, dtype=bool)
        expected[127, 20] = True
        expected[128, 30] = True
        # A nonoverlapping first band includes the distracting background.
        self.assertEqual(detect_bright_pixels(image[:128]).count, 0)
        self.assertEqual(detect_bright_pixels(image).count, 0)

        detection = detect_bright_pixels_in_bands(image)

        np.testing.assert_array_equal(detection.mask, expected)

    def test_final_band_covers_source_rows_after_the_last_regular_stride(self):
        image = np.empty((300, 128), dtype=np.int16)
        image[:160] = np.linspace(1, 32767, 160 * 128).reshape(160, 128)
        image[160:] = np.linspace(-17000, -8000, 140 * 128).reshape(140, 128)
        image[-1, 127] = 32555
        self.assertEqual(detect_bright_pixels(image).count, 0)

        detection = detect_bright_pixels_in_bands(image)

        self.assertEqual(detection.count, 1)
        self.assertTrue(detection.mask[-1, 127])

    def test_full_width_bright_layer_is_preserved_while_body_defects_are_detected(self):
        image = np.empty((512, 256), dtype=np.int16)
        image[:64] = np.linspace(1, 32767, 64 * 256).reshape(64, 256)
        image[64:] = np.linspace(-18000, -8000, 448 * 256).reshape(448, 256)
        image[250, :] = 30000
        image[260, 31] = 32005
        image[280, 17] = 32555
        expected = np.zeros(image.shape, dtype=bool)
        expected[260, 31] = True
        expected[280, 17] = True

        detection = detect_bright_pixels_in_bands(image)

        np.testing.assert_array_equal(detection.mask, expected)
        self.assertFalse(np.any(detection.mask[250]))

    def test_global_detector_retains_defects_too_dense_for_individual_bands(self):
        image = np.full((512, 128), 1000, dtype=np.uint16)
        image[230:250, 50:70] = 65000
        expected = image == 65000
        # The compact defect is rare globally, but exceeds 1% of each band.
        self.assertEqual(detect_bright_pixels(image[192:320]).count, 0)

        detection = detect_bright_pixels_in_bands(image)

        np.testing.assert_array_equal(detection.mask, expected)

    def test_small_readonly_noncontiguous_image_matches_whole_image_detection(self):
        base = np.full((64, 128), -10000, dtype=np.int16)
        image = base[:, ::2]
        image[30, 31] = 32005
        image.setflags(write=False)
        original = image.copy()

        detection = detect_bright_pixels_in_bands(image)

        np.testing.assert_array_equal(detection.mask, detect_bright_pixels(image).mask)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(detection.count, 1)
        self.assertFalse(detection.mask.flags.writeable)

    def test_empty_and_tiny_sources_have_no_detection(self):
        for shape in ((0, 0), (0, 5), (150, 0), (1, 8)):
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.int16)
                detection = detect_bright_pixels_in_bands(image)
                self.assertEqual(detection.mask.shape, shape)
                self.assertEqual(detection.count, 0)
                self.assertIsNone(detection.threshold)
                self.assertFalse(detection.mask.flags.writeable)

    def test_band_height_requires_a_positive_integer(self):
        image = np.zeros((32, 32), dtype=np.int16)
        for value in (0, -1, True, False, 1.5, "128", None):
            with self.subTest(band_height=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    detect_bright_pixels_in_bands(image, band_height=value)
        self.assertEqual(
            detect_bright_pixels_in_bands(image, band_height=np.int64(128)).count,
            0,
        )


class BrightPixelSmoothingTests(unittest.TestCase):
    def test_only_masked_values_change_and_all_original_hot_values_are_excluded(self):
        image = np.array(
            [[65535, 10, 20], [30, 65535, 40], [50, 60, 70]], dtype=np.uint16
        )
        original = image.copy()
        mask = image == 65535

        result = smooth_bright_pixels(image, mask)

        self.assertEqual(result.dtype, image.dtype)
        self.assertEqual(result[0, 0], 20)
        self.assertEqual(result[1, 1], 40)
        np.testing.assert_array_equal(result[~mask], original[~mask])
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.shares_memory(result, image))

    def test_dense_blob_is_repaired_from_good_neighbors_without_uint_overflow(self):
        image = np.full((64, 64), 50000, dtype=np.uint16)
        image[10:54, 10:54] = 65535
        mask = image == 65535

        result = smooth_bright_pixels(image, mask)

        np.testing.assert_array_equal(result, np.full_like(image, 50000))

    def test_integer_mean_rounding_uses_exact_neighbor_sum(self):
        image = np.array([[9000, 2], [3, 9000]], dtype=np.int16)
        mask = image == 9000

        result = smooth_bright_pixels(image, mask)

        self.assertEqual(result[0, 0], 2)
        self.assertEqual(result[1, 1], 2)

    def test_signed_intensities_and_readonly_noncontiguous_input_are_preserved(self):
        base = np.arange(80, dtype=np.int16).reshape(8, 10) - 200
        image = base[:, ::2]
        image[4, 2] = 32555
        image.setflags(write=False)
        original = image.copy()
        mask = image == 32555

        result = smooth_bright_pixels(image, mask)

        np.testing.assert_array_equal(result[~mask], original[~mask])
        np.testing.assert_array_equal(image, original)
        self.assertLess(result[4, 2], 0)
        self.assertEqual(result.dtype, np.dtype("int16"))

    def test_unmasked_nonfinite_values_are_preserved(self):
        image = np.array([[100, np.nan, 100], [np.nan, 9000, np.nan], [100, np.nan, 100]])
        mask = np.zeros(image.shape, dtype=bool)
        mask[1, 1] = True

        result = smooth_bright_pixels(image, mask)

        self.assertEqual(result[1, 1], 100)
        np.testing.assert_array_equal(result[~mask], image[~mask])

    def test_averaging_extreme_finite_float_neighbors_does_not_overflow(self):
        image = np.full((3, 3), 1e308)
        image[1, 1] = -1e308
        mask = np.zeros(image.shape, dtype=bool)
        mask[1, 1] = True

        with np.errstate(over="raise", invalid="raise"):
            result = smooth_bright_pixels(image, mask)

        self.assertTrue(np.isfinite(result[1, 1]))
        self.assertAlmostEqual(result[1, 1] / 1e308, 1.0)
        np.testing.assert_array_equal(result[~mask], image[~mask])

    def test_disconnected_mask_uses_good_samples_without_modifying_nan_barrier(self):
        image = np.full((9, 9), 20.0)
        image[1:8, 1:8] = np.nan
        image[4, 4] = 9000
        mask = np.zeros(image.shape, dtype=bool)
        mask[4, 4] = True

        result = smooth_bright_pixels(image, mask)

        self.assertEqual(result[4, 4], 20)
        np.testing.assert_array_equal(result[~mask], image[~mask])

    def test_empty_mask_returns_independent_unchanged_copy(self):
        image = np.arange(20, dtype=np.uint16).reshape(4, 5)

        result = smooth_bright_pixels(image, np.zeros(image.shape, dtype=bool))

        np.testing.assert_array_equal(result, image)
        self.assertFalse(np.shares_memory(result, image))

    def test_all_masked_input_requires_usable_neighbors(self):
        image = np.full((8, 8), 65535, dtype=np.uint16)

        with self.assertRaisesRegex(ValueError, "unmasked finite neighbors"):
            smooth_bright_pixels(image, np.ones(image.shape, dtype=bool))

    def test_shape_validation(self):
        with self.assertRaisesRegex(ValueError, "2-D grayscale"):
            detect_bright_pixels(np.zeros((2, 2, 2), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "match the image shape"):
            smooth_bright_pixels(np.zeros((2, 2)), np.zeros((3, 3), dtype=bool))


if __name__ == "__main__":
    unittest.main()
