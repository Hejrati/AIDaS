"""Crop proposals must preserve retinal coverage and original pixel values."""

import unittest

import numpy as np

from aidas.utils.oct_crop import detect_oct_crop, map_oct_crop_to_raw


def curved_retina(*, height=420, width=320, blank_sides=0, wrapped=False):
    """Layered curved retina above a long, gently fading choroid signal."""
    rng = np.random.default_rng(913)
    xx = np.arange(width, dtype=float)
    yy = np.arange(height, dtype=float)[:, None]
    top = 95 + 30 * ((xx - width / 2) / (width / 2)) ** 2
    top += 14 * np.exp(-((xx - width / 2) / (width / 10)) ** 2)
    bottom = top + 77 + 5 * np.sin(xx / 47)
    data = 120 + rng.normal(0, 4, (height, width))
    inside = (yy >= top) & (yy <= bottom)
    data += inside * 115
    for position, amplitude, sigma in (
        (top, 470, 1.8),
        (top + 18, 120, 2.0),
        (top + 39, 95, 2.0),
        (bottom - 8, 480, 2.2),
        (bottom, 145, 1.5),
        # A faint outside layer must survive a proposal driven by bright layers.
        (bottom + 12, 45, 1.5),
    ):
        data += amplitude * np.exp(-0.5 * ((yy - position) / sigma) ** 2)
    distance = yy - bottom - 16
    data += np.where(distance > 0, 130 * np.exp(-np.maximum(distance, 0) / 70), 0)
    data = np.clip(np.rint(data), 0, 65535).astype(np.uint16)
    if wrapped:
        # Native words above the signed range with zero-valued background
        # speckles resemble the supplied SDB without decoding its encoding.
        data = (47000 + data.astype(np.uint32) * 17).astype(np.uint16)
        background = (yy < top - 10) | (yy > bottom + 65)
        data[background & (rng.random(data.shape) < 0.38)] = 0
    if blank_sides:
        data[:, :blank_sides] = 0
        data[:, width - blank_sides:] = 0
    return data, top, bottom + 12


class OctCropTests(unittest.TestCase):
    def assert_contains(self, roi, top, bottom, *, first=0, last=None):
        x, y, width, height = roi
        last = len(top) if last is None else last
        self.assertLessEqual(x, first)
        self.assertGreaterEqual(x + width, last)
        self.assertLessEqual(y, int(np.floor(np.min(top[first:last]))))
        self.assertGreaterEqual(y + height, int(np.ceil(np.max(bottom[first:last]))) + 1)

    def test_curved_retina_keeps_faint_layers_and_removes_unused_vertical_space(self):
        image, top, bottom = curved_retina()
        original = image.copy()
        result = detect_oct_crop(image)
        self.assertTrue(result.confident)
        self.assertEqual(result.method, "image content")
        self.assert_contains(result.roi, top, bottom)
        _, y, _, height = result.roi
        self.assertGreater(y, 0)
        self.assertLess(y + height, image.shape[0] - 30)
        np.testing.assert_array_equal(image, original)

    def test_zero_speckles_in_native_words_do_not_drive_crop_to_image_edges(self):
        image, top, bottom = curved_retina(wrapped=True)
        original = image.copy()
        result = detect_oct_crop(image)
        self.assertTrue(result.confident)
        self.assert_contains(result.roi, top, bottom)
        _, y, _, height = result.roi
        self.assertGreater(y, 0)
        self.assertLess(y + height, image.shape[0])
        np.testing.assert_array_equal(image, original)

    def test_fragmented_outer_band_is_kept_with_continuous_inner_surface(self):
        height, width = 420, 320
        xx = np.arange(width)
        yy = np.arange(height)[:, None]
        top = 110 + 12 * np.cos(xx / 61)
        bottom = top + 110
        rng = np.random.default_rng(426)
        data = 100 + rng.normal(0, 2, (height, width))
        data += ((yy >= top) & (yy <= bottom)) * 60
        data += 650 * np.exp(-0.5 * ((yy - top) / 1.8) ** 2)
        # Outer retinal signal is coherent in smaller lateral segments,
        # separated by wide shadows. It must not be discarded just because
        # the inner surface spans a greater fraction of the scan width.
        visible = ((xx >= 32) & (xx < 112)) | ((xx >= 180) & (xx < 270))
        amplitude = np.where(visible, 430, 18)
        data += amplitude * np.exp(-0.5 * ((yy - bottom) / 2.0) ** 2)
        image = np.clip(np.rint(data), 0, 65535).astype(np.uint16)
        result = detect_oct_crop(image)
        self.assertTrue(result.confident)
        self.assert_contains(result.roi, top, bottom)

    def test_short_visible_outer_segment_is_included_or_crop_is_declined(self):
        height, width = 420, 320
        yy = np.arange(height)[:, None]
        inner = 650 * np.exp(-0.5 * ((yy - 100) / 1.8) ** 2)
        data = np.broadcast_to(100 + inner, (height, width)).copy()
        outer = 430 * np.exp(-0.5 * ((yy - 210) / 2.0) ** 2)
        data[:, 180:210] += outer
        image = np.rint(data).astype(np.uint16)
        original = image.copy()
        result = detect_oct_crop(image)
        x, y, crop_width, crop_height = result.roi
        self.assertLessEqual(x, 180)
        self.assertGreaterEqual(x + crop_width, 210)
        self.assertLessEqual(y, 98)
        self.assertGreaterEqual(y + crop_height, 213)
        np.testing.assert_array_equal(image, original)

    def test_single_thin_bright_surface_cannot_establish_all_retinal_layers(self):
        yy = np.arange(420)[:, None]
        data = np.broadcast_to(
            100 + 650 * np.exp(-0.5 * ((yy - 100) / 1.8) ** 2), (420, 320)
        )
        image = np.rint(data).astype(np.uint16)
        result = detect_oct_crop(image)
        self.assertEqual(result.roi, (0, 0, 320, 420))
        self.assertEqual(result.method, "full image")
        self.assertFalse(result.confident)

    def test_lateral_reflection_bundle_does_not_replace_good_middle_seed(self):
        image, top, bottom = curved_retina()
        rows = np.arange(image.shape[0], dtype=float)[:, None]
        data = image.astype(float)
        for position in (24, 44, 64):
            data[:, :112] += 1500 * np.exp(-0.5 * ((rows - position) / 2.0) ** 2)
        image = np.rint(data).astype(np.uint16)
        original = image.copy()
        result = detect_oct_crop(image)
        self.assertTrue(result.confident)
        self.assert_contains(result.roi, top, bottom)
        _, y, _, crop_height = result.roi
        self.assertGreaterEqual(y, int(np.floor(top.min())) - 40)
        self.assertLessEqual(y + crop_height, int(np.ceil(bottom.max())) + 45)
        np.testing.assert_array_equal(image, original)

    def test_broad_retinal_plateau_is_kept_below_brighter_upper_reflection(self):
        rows = np.arange(420, dtype=float)[:, None]
        cases = [(background, contrast, position, amplitude)
                 for background in (100, 200)
                 for contrast in (120, -120)
                 if background + contrast >= 0
                 for position in (25, 55, 70)
                 for amplitude in (0, 300, 500, 650, 900, 1400)]
        for background, contrast, position, amplitude in cases:
            with self.subTest(background=background, contrast=contrast,
                              position=position, amplitude=amplitude):
                data = np.full((420, 320), float(background))
                data[100:211] += contrast
                data += amplitude * np.exp(-0.5 * ((rows - position) / 2.0) ** 2)
                image = np.rint(data).astype(np.uint16)
                original = image.copy()
                result = detect_oct_crop(image)
                self.assertTrue(result.confident)
                self.assertEqual(result.method, "image content")
                x, y, width, height = result.roi
                self.assertEqual((x, width), (0, 320))
                self.assertGreaterEqual(y, 55)
                self.assertLessEqual(y, 100)
                self.assertGreaterEqual(y + height, 211)
                self.assertLessEqual(y + height, 255)
                np.testing.assert_array_equal(image, original)

    def test_only_confirmed_blank_side_margins_are_removed(self):
        image, top, bottom = curved_retina(blank_sides=36)
        result = detect_oct_crop(image)
        self.assert_contains(result.roi, top, bottom, first=36, last=image.shape[1] - 36)
        x, _, width, _ = result.roi
        self.assertGreater(x, 0)
        self.assertLess(x + width, image.shape[1])

    def test_real_retinal_signal_at_image_sides_keeps_full_width(self):
        image, top, bottom = curved_retina()
        result = detect_oct_crop(image)
        self.assertEqual(result.roi[0], 0)
        self.assertEqual(result.roi[2], image.shape[1])
        self.assert_contains(result.roi, top, bottom)

    def test_matching_contours_take_priority_without_altering_pixels(self):
        image, top, bottom = curved_retina()
        original = image.copy()
        result = detect_oct_crop(image, contours=np.stack([top, bottom]))
        self.assertEqual(result.method, "EDB contours")
        self.assertTrue(result.confident)
        self.assert_contains(result.roi, top, bottom)
        np.testing.assert_array_equal(image, original)

    def test_missing_contour_tails_are_not_evidence_for_trimming_sides(self):
        image, top, bottom = curved_retina()
        contours = np.stack([top, bottom])
        contours[:, :14] = np.nan
        contours[:, -19:] = np.finfo(np.float32).max
        contours[:, 145:150] = np.nan
        result = detect_oct_crop(image, contours=contours)
        self.assertEqual(result.method, "EDB contours")
        self.assertEqual(result.roi[0], 0)
        self.assertEqual(result.roi[2], image.shape[1])
        self.assert_contains(result.roi, top, bottom)

    def test_invalid_contours_do_not_override_image_content(self):
        image, top, bottom = curved_retina()
        contours = np.full((2, image.shape[1]), np.finfo(np.float32).max)
        result = detect_oct_crop(image, contours=contours)
        self.assertEqual(result.method, "image content")
        self.assert_contains(result.roi, top, bottom)

    def test_unrelated_contour_width_does_not_override_image_content(self):
        image, top, bottom = curved_retina()
        result = detect_oct_crop(image, contours=np.ones((2, 64)) * 12)
        self.assertEqual(result.method, "image content")
        self.assert_contains(result.roi, top, bottom)

    def test_flat_and_tiny_inputs_keep_full_image(self):
        for image in (
            np.zeros((180, 240), dtype=np.uint16),
            np.full((180, 240), 47000, dtype=np.uint16),
            np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint16),
        ):
            with self.subTest(shape=image.shape, value=int(image[0, 0])):
                result = detect_oct_crop(image)
                self.assertEqual(result.roi, (0, 0, image.shape[1], image.shape[0]))
                self.assertEqual(result.method, "full image")
                self.assertFalse(result.confident)

    def test_noise_without_coherent_retinal_layers_keeps_full_image(self):
        rng = np.random.default_rng(725)
        image = np.clip(rng.normal(18000, 3000, (320, 400)), 0, 65535).astype(np.uint16)
        original = image.copy()
        result = detect_oct_crop(image)
        self.assertEqual(result.roi, (0, 0, 400, 320))
        self.assertEqual(result.method, "full image")
        self.assertFalse(result.confident)
        np.testing.assert_array_equal(image, original)


class OctCropMappingTests(unittest.TestCase):
    def test_aligned_native_payload_maps_inside_full_manual_import(self):
        self.assertEqual(
            map_oct_crop_to_raw(
                (10, 15, 60, 40), pixel_start=2000,
                oct_width=100, view_width=100, view_height=200,
            ),
            (10, 35, 60, 40),
        )

    def test_nonzero_start_column_without_wrap_preserves_exact_box(self):
        self.assertEqual(
            map_oct_crop_to_raw(
                (10, 15, 60, 40), pixel_start=2010,
                oct_width=100, view_width=100, view_height=200,
            ),
            (20, 35, 60, 40),
        )

    def test_wrapped_rows_need_full_width_and_both_endpoint_rows(self):
        self.assertEqual(
            map_oct_crop_to_raw(
                (10, 15, 60, 40), pixel_start=2050,
                oct_width=100, view_width=100, view_height=200,
            ),
            (0, 35, 100, 41),
        )

    def test_width_mismatch_cannot_make_a_trusted_native_crop(self):
        self.assertIsNone(map_oct_crop_to_raw(
            (0, 15, 100, 40), pixel_start=2000,
            oct_width=100, view_width=120, view_height=200,
        ))

    def test_payload_outside_view_cannot_make_a_trusted_native_crop(self):
        self.assertIsNone(map_oct_crop_to_raw(
            (10, 15, 60, 40), pixel_start=19000,
            oct_width=100, view_width=100, view_height=200,
        ))

    def test_negative_or_out_of_bounds_coordinates_are_rejected(self):
        for roi, start in (((-1, 15, 60, 40), 0), ((70, 15, 60, 40), 0), ((10, 15, 60, 40), -1)):
            with self.subTest(roi=roi, start=start):
                self.assertIsNone(map_oct_crop_to_raw(
                    roi, pixel_start=start,
                    oct_width=100, view_width=100, view_height=200,
                ))


if __name__ == "__main__":
    unittest.main()
