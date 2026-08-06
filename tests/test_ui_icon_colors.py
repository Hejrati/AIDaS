import unittest

from PIL import Image

from aidas.utils.ui_utils import _dark_mode_icon, _tint_icon


class UIIconColorTests(unittest.TestCase):
    def test_dark_mode_preserves_saturated_red_and_blue_pixels(self):
        source = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
        source.putpixel((0, 0), (255, 0, 0, 255))
        source.putpixel((1, 0), (63, 81, 181, 255))
        source.putpixel((2, 0), (5, 5, 5, 255))

        converted = _dark_mode_icon(source)

        self.assertEqual(converted.getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(converted.getpixel((1, 0)), (63, 81, 181, 255))
        self.assertEqual(converted.getpixel((2, 0)), (221, 231, 240, 255))

    def test_tint_uses_requested_color_and_preserves_alpha(self):
        source = Image.new("RGBA", (2, 1), (10, 20, 30, 0))
        source.putpixel((1, 0), (10, 20, 30, 128))

        tinted = _tint_icon(source, "#07131C")

        self.assertEqual(tinted.getpixel((0, 0)), (7, 19, 28, 0))
        self.assertEqual(tinted.getpixel((1, 0)), (7, 19, 28, 128))


if __name__ == "__main__":
    unittest.main()
