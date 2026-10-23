"""Image-only proposals follow the retinal band, not distant bright artifacts."""

import unittest

import numpy as np

from aidas.utils.oct_crop import detect_oct_crop

from tests.support.retina import curved_retina


class OctCropCenterTests(unittest.TestCase):
    def assert_retinal_box(self, result, top, bottom, image, *, margin=45):
        """Keep curved ends and faint layers within a reasonably local box."""
        self.assertTrue(result.confident)
        self.assertEqual(result.method, "image content")
        x, y, width, height = result.roi
        self.assertEqual((x, width), (0, image.shape[1]))
        self.assertLessEqual(y, int(np.floor(top.min())))
        self.assertGreaterEqual(y + height, int(np.ceil(bottom.max())) + 1)
        self.assertGreaterEqual(y, max(0, int(np.floor(top.min())) - margin))
        self.assertLessEqual(y + height, min(image.shape[0], int(np.ceil(bottom.max())) + margin))

    def test_distant_upper_bands_do_not_pull_box_into_vitreous_space(self):
        # A coherent reflection can span a short region or the full width.
        # Its brightness alone does not make the empty gap retinal tissue.
        for band_width in (20, 28, 320):
            with self.subTest(band_width=band_width):
                image, top, bottom = curved_retina()
                rows = np.arange(image.shape[0], dtype=float)[:, None]
                data = image.astype(float)
                left = (image.shape[1] - band_width) // 2
                data[:, left:left + band_width] += 900 * np.exp(
                    -0.5 * ((rows - 28) / 2.0) ** 2
                )
                image = np.rint(data).astype(np.uint16)
                original = image.copy()
                self.assert_retinal_box(detect_oct_crop(image), top, bottom, image)
                np.testing.assert_array_equal(image, original)

    def test_distant_lower_band_does_not_include_empty_space_below_layers(self):
        image, top, bottom = curved_retina()
        rows = np.arange(image.shape[0], dtype=float)[:, None]
        data = image.astype(float)
        data[:, 100:138] += 1000 * np.exp(-0.5 * ((rows - 370) / 2.0) ** 2)
        image = np.rint(data).astype(np.uint16)
        original = image.copy()
        self.assert_retinal_box(detect_oct_crop(image), top, bottom, image)
        np.testing.assert_array_equal(image, original)

    def test_center_tracks_retina_when_it_is_far_from_geometric_center(self):
        # Identical tissue near the upper and lower portions of a large view
        # must be found from the data, rather than a fixed middle rectangle.
        for shift in (0, 360):
            with self.subTest(shift=shift):
                base, top, bottom = curved_retina(height=800)
                if shift:
                    rng = np.random.default_rng(510)
                    image = np.clip(
                        np.rint(120 + rng.normal(0, 4, base.shape)), 0, 65535
                    ).astype(np.uint16)
                    image[shift:] = base[:-shift]
                else:
                    image = base
                original = image.copy()
                self.assert_retinal_box(
                    detect_oct_crop(image), top + shift, bottom + shift,
                    image, margin=55,
                )
                np.testing.assert_array_equal(image, original)

    def test_isolated_bright_pixels_do_not_move_or_enlarge_retinal_box(self):
        image, top, bottom = curved_retina()
        clean = detect_oct_crop(image)
        image[20, 155] = 65535
        image[390, 80] = 65535
        image[345, 240] = 65535
        original = image.copy()
        result = detect_oct_crop(image)
        self.assert_retinal_box(result, top, bottom, image)
        self.assertEqual(result.roi, clean.roi)
        np.testing.assert_array_equal(image, original)

    def test_bright_thin_band_at_image_middle_does_not_replace_offcenter_retina(self):
        image, top, bottom = curved_retina(height=700)
        rows = np.arange(image.shape[0], dtype=float)[:, None]
        data = image.astype(float)
        data += 1400 * np.exp(-0.5 * ((rows - 350) / 2.0) ** 2)
        image = np.rint(data).astype(np.uint16)
        original = image.copy()
        self.assert_retinal_box(detect_oct_crop(image), top, bottom, image, margin=55)
        np.testing.assert_array_equal(image, original)

    def test_central_shadow_uses_neighboring_columns_to_locate_retinal_band(self):
        image, top, bottom = curved_retina()
        # A central shadow hides the layers over a quarter of the scan. The
        # adjoining tissue still supplies enough information for a local box.
        image[:, 120:200] = 120
        original = image.copy()
        self.assert_retinal_box(detect_oct_crop(image), top, bottom, image)
        np.testing.assert_array_equal(image, original)

    def test_reversed_contrast_finds_dark_retinal_layers_from_intensity_changes(self):
        image, top, bottom = curved_retina()
        # Every value stays positive and inside uint16; the geometry remains
        # identical while the brightest tissue becomes the darkest tissue.
        image = (2000 - image.astype(np.int32)).astype(np.uint16)
        original = image.copy()
        self.assert_retinal_box(detect_oct_crop(image), top, bottom, image)
        np.testing.assert_array_equal(image, original)


if __name__ == "__main__":
    unittest.main()
