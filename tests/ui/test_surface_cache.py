"""Keep repeated decorative paints cheap without changing their coverage."""

import gc
import math
import unittest
from unittest import mock
import weakref

from aidas.ui import rendering


class _Geometry:
    def __init__(self, kind="polygon", coords=(8, 8, 172, 8, 172, 32, 8, 32), width=16):
        self.kind = kind
        self.points = coords
        self.width = width
        self.join = "round"
        self.fill = "#123456"

    def type(self, _item):
        return self.kind

    def coords(self, _item):
        return self.points

    def itemcget(self, _item, option):
        return {"width": self.width, "joinstyle": self.join, "fill": self.fill}[option]


class SurfaceMaskCacheTests(unittest.TestCase):
    def setUp(self):
        rendering._cached_item_mask.cache_clear()
        self.addCleanup(rendering._cached_item_mask.cache_clear)

    def test_recolors_and_equivalent_controls_reuse_antialiasing_geometry(self):
        first = _Geometry()
        second = _Geometry()
        with mock.patch.object(rendering, "_rounded_mask", wraps=rendering._rounded_mask) as rasterize:
            original = rendering._item_mask(first, 1, (180, 40)).tobytes()
            for color in ("#ABCDEF", "#101820", "#123456"):
                first.fill = color
                self.assertEqual(rendering._item_mask(first, 1, (180, 40)).tobytes(), original)
                self.assertEqual(rendering._item_mask(second, 99, (180, 40)).tobytes(), original)
            rasterize.assert_called_once()

    def test_geometry_and_dpi_changes_keep_exact_uncached_pixels(self):
        for scale in (1, 1.25, 1.5, 2):
            size = math.ceil(180 * scale), math.ceil(40 * scale)
            for kind, points, width in (
                ("rectangle", (0, 0, 180, 40), 0),
                ("polygon", (8, 8, 172, 8, 172, 32, 8, 32), 16),
                ("line", (4, 4, 11, 13, 22, 6), 2),
                ("oval", (5, 5, 20, 20), 1),
            ):
                with self.subTest(scale=scale, kind=kind):
                    canvas = _Geometry(kind, tuple(point * scale for point in points), width * scale)
                    for join in ("round", "miter"):
                        canvas.join = join
                        actual = rendering._item_mask(canvas, 1, size)
                        reference = rendering._rasterize_item_mask(
                            size, kind, canvas.points, canvas.width,
                            kind == "line" and join == "round",
                        )
                        self.assertEqual(actual.tobytes(), reference.tobytes())
                    # A moved primitive must not keep the prior coverage mask.
                    canvas.points = tuple(point + 1 for point in canvas.points)
                    moved = rendering._item_mask(canvas, 1, size)
                    reference = rendering._rasterize_item_mask(
                        size, kind, canvas.points, canvas.width, False,
                    )
                    self.assertEqual(moved.tobytes(), reference.tobytes())

    def test_large_panels_are_not_retained_and_small_cache_is_bounded(self):
        canvas = _Geometry()
        with mock.patch.object(rendering, "_rounded_mask", wraps=rendering._rounded_mask) as rasterize:
            for _ in range(2):
                rendering._item_mask(canvas, 1, (512, 512))
            self.assertEqual(rasterize.call_count, 2)
            self.assertEqual(rendering._cached_item_mask.cache_info().currsize, 0)
        for width in range(32, 200):
            rendering._item_mask(canvas, 1, (width, 40))
        self.assertLessEqual(rendering._cached_item_mask.cache_info().currsize, 128)

    def test_cached_masks_do_not_keep_widgets_or_tk_interpreters_alive(self):
        canvas = _Geometry()
        reference = weakref.ref(canvas)
        rendering._item_mask(canvas, 1, (180, 40))
        del canvas
        gc.collect()
        self.assertIsNone(reference())


if __name__ == "__main__":
    unittest.main()
