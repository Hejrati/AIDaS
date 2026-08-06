from __future__ import annotations

import unittest

from aidas.ui.windowing import (
    _hex_to_colorref,
    _is_dark_color,
    _native_border_colorref,
    centered_logical_geometry,
    centered_physical_geometry,
    logical_window_size,
    physical_window_size,
)


class _ScaledWindow:
    def __init__(self, scale: float):
        self.scale = scale

    def _apply_window_scaling(self, value: int) -> int:
        return int(value * self.scale)

    def _reverse_window_scaling(self, value: int) -> int:
        return int(value / self.scale)


class UIWindowingTests(unittest.TestCase):
    def test_windows_colorref_uses_bgr_integer_layout(self):
        self.assertEqual(_hex_to_colorref("#123456"), 0x563412)

    def test_custom_frame_requests_no_native_windows_11_border(self):
        window = type("Window", (), {"_aidas_suppress_native_border": True})()
        self.assertEqual(_native_border_colorref(window, "#123456"), 0xFFFFFFFE)

    def test_native_frame_keeps_themed_border_color(self):
        self.assertEqual(_native_border_colorref(object(), "#123456"), 0x563412)

    def test_caption_palette_classifies_light_and_dark_colors(self):
        self.assertTrue(_is_dark_color("#141D27"))
        self.assertFalse(_is_dark_color("#FFFFFF"))

    def test_size_conversion_uses_ctk_window_scaling(self):
        window = _ScaledWindow(1.5)
        self.assertEqual(logical_window_size(window, 1440, 900), (960, 600))
        self.assertEqual(physical_window_size(window, 960, 600), (1440, 900))

    def test_logical_geometry_centers_using_rendered_physical_size(self):
        window = _ScaledWindow(1.5)
        self.assertEqual(
            centered_logical_geometry(
                window,
                900,
                600,
                bounds=(0, 0, 1920, 1080),
            ),
            "900x600+285+90",
        )

    def test_physical_geometry_is_not_scaled_twice(self):
        window = _ScaledWindow(1.5)
        self.assertEqual(
            centered_physical_geometry(
                window,
                1440,
                780,
                bounds=(0, 0, 1920, 1080),
            ),
            "960x520+240+150",
        )

    def test_native_tk_window_falls_back_to_one_to_one_dimensions(self):
        window = object()
        self.assertEqual(logical_window_size(window, 800, 600), (800, 600))
        self.assertEqual(physical_window_size(window, 800, 600), (800, 600))


if __name__ == "__main__":
    unittest.main()
