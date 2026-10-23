from __future__ import annotations

from collections import deque
import math
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import patch

from customtkinter.windows.widgets.core_rendering import CTkCanvas, DrawEngine
import numpy as np
from PIL import Image, ImageChops, ImageOps, ImageTk

from aidas.ui import rendering


def _border_mask(width, height, radius, border_width):
    outer, inner = rendering.rounded_rectangle_masks(
        width, height, radius, border_width,
    )
    return outer, inner, np.asarray(ImageChops.subtract(outer, inner))


def _reachable(pixels, starts, *, diagonals=False):
    """Flood a binary image independently of the renderer's shape construction."""
    height, width = pixels.shape
    seen = set(starts)
    pending = deque(starts)
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonals:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while pending:
        x, y = pending.popleft()
        for dx, dy in offsets:
            neighbor = nx, ny = x + dx, y + dy
            if (
                0 <= nx < width and 0 <= ny < height
                and pixels[ny, nx] and neighbor not in seen
            ):
                seen.add(neighbor)
                pending.append(neighbor)
    return seen


class _CanvasStub:
    """Record physical canvas geometry without opening a Tk window."""

    def __init__(self):
        self.items = {}
        self.next_id = 1

    def find_withtag(self, tag):
        return tuple(
            item_id for item_id, item in self.items.items()
            if tag == item_id or tag in item["tags"]
        )

    def create_polygon(self, coordinates, **options):
        item_id = self.next_id
        self.next_id += 1
        self.items[item_id] = {
            "kind": "polygon",
            "tags": options.pop("tags", ()),
            "coords": tuple(coordinates),
            "options": options,
        }
        return item_id

    def create_rectangle(self, *coordinates, **options):
        item_id = self.create_polygon(coordinates, **options)
        self.items[item_id]["kind"] = "rectangle"
        return item_id

    def coords(self, tag, *coordinates):
        matches = self.find_withtag(tag)
        if not coordinates:
            return self.items[matches[0]]["coords"] if matches else ()
        if len(coordinates) == 1 and isinstance(coordinates[0], (tuple, list)):
            coordinates = tuple(coordinates[0])
        for item_id in matches:
            self.items[item_id]["coords"] = tuple(coordinates)

    def itemconfig(self, tag, **options):
        for item_id in self.find_withtag(tag):
            self.items[item_id]["options"].update(options)

    def delete(self, *tags):
        for tag in tags:
            for item_id in self.find_withtag(tag):
                del self.items[item_id]

    def tag_lower(self, *_args):
        pass

    def type(self, tag):
        matches = self.find_withtag(tag)
        return self.items[matches[0]]["kind"] if matches else ""

    def itemcget(self, tag, option):
        matches = self.find_withtag(tag)
        return str(self.items[matches[0]]["options"].get(option, ""))


class RoundedBorderMaskTests(unittest.TestCase):
    def test_thin_borders_form_a_closed_connected_perimeter_at_corner_joins(self):
        # Sidebar cards, small icon buttons, and workflow navigation all use
        # thin strokes; a disconnected corner or a missing join must fail.
        for width, height, radius, border_width in (
            (355, 39, 10, 1),
            (49, 39, 6, 1),
            (61, 51, 14, 1.25),
            (71, 53, 20, 1.5),
            (280, 35, 10, 1),
            (357, 69, 13, 2),
            (29, 29, 99, 1),
        ):
            with self.subTest(size=(width, height), radius=radius, border=border_width):
                _, _, ring = _border_mask(width, height, radius, border_width)
                visible = ring >= 32
                ys, xs = np.nonzero(visible)
                self.assertGreater(len(xs), 0)
                connected = _reachable(
                    visible, [(int(xs[0]), int(ys[0]))], diagonals=True,
                )
                self.assertEqual(len(connected), len(xs), "The border has detached segments")

                # A ring with a single missing segment can remain connected.
                # Flooding the exterior also proves that it is closed.
                background = ~visible
                exterior_starts = {
                    (x, y)
                    for x, y in (
                        *((x, 0) for x in range(width)),
                        *((x, height - 1) for x in range(width)),
                        *((0, y) for y in range(height)),
                        *((width - 1, y) for y in range(height)),
                    )
                    if background[y, x]
                }
                exterior = _reachable(background, exterior_starts)
                self.assertNotIn(
                    (width // 2, height // 2), exterior,
                    "A gap in the border connects its interior to the exterior",
                )

    def test_odd_dimensions_and_fractional_strokes_keep_opposite_sides_symmetric(self):
        for width, height, radius, border_width in (
            (355, 39, 10, 1),
            (43, 37, 6.25, 1.25),
            (71, 53, 14.5, 1.5),
            (37, 37, 40, 0.75),
            (73, 55, 0, 1.75),
        ):
            with self.subTest(size=(width, height), radius=radius, border=border_width):
                outer, inner, _ = _border_mask(width, height, radius, border_width)
                for mask in (outer, inner):
                    self.assertEqual(mask.mode, "L")
                    self.assertEqual(mask.size, (width, height))
                    for flipped in (ImageOps.mirror(mask), ImageOps.flip(mask)):
                        difference = ImageChops.difference(mask, flipped)
                        self.assertLessEqual(difference.getextrema()[1], 1)

    def test_border_thickness_tracks_display_scale_without_integer_rounding(self):
        for scale in (1, 1.1, 1.25, 1.5, 1.75, 2):
            with self.subTest(scale=scale):
                width, height = 355 * scale, 39 * scale
                outer, inner, ring = _border_mask(width, height, 10 * scale, scale)
                expected_size = (math.ceil(width), math.ceil(height))
                self.assertEqual(outer.size, expected_size)
                self.assertEqual(inner.size, expected_size)
                column = ring[:, expected_size[0] // 2].astype(float) / 255
                row = ring[expected_size[1] // 2, :].astype(float) / 255
                thicknesses = (
                    column[:len(column) // 2].sum(),
                    column[len(column) // 2:].sum(),
                    row[:len(row) // 2].sum(),
                    row[len(row) // 2:].sum(),
                )
                for thickness in thicknesses:
                    self.assertAlmostEqual(thickness, scale, delta=0.2)
                self.assertLess(max(thicknesses) - min(thicknesses), 0.2)

    def test_antialiased_curve_has_partial_coverage_without_filling_the_corners(self):
        outer, inner, ring = _border_mask(67, 45, 10, 1)
        for mask in (outer, inner):
            pixels = np.asarray(mask)
            self.assertEqual(int(pixels[0, 0]), 0)
            self.assertEqual(int(pixels[22, 33]), 255)
            self.assertTrue(np.any((pixels[:10, :10] > 0) & (pixels[:10, :10] < 255)))
        self.assertEqual(int(ring[22, 33]), 0)
        self.assertTrue(np.all(np.asarray(inner) <= np.asarray(outer)))

    def test_square_border_keeps_full_odd_sized_canvas_and_one_pixel_inset(self):
        outer, inner, ring = _border_mask(35, 27, 0, 1)
        self.assertTrue(np.all(np.asarray(outer) == 255))
        self.assertGreaterEqual(int(ring[0, 17]), 220)
        self.assertGreaterEqual(int(ring[-1, 17]), 220)
        self.assertGreaterEqual(int(ring[13, 0]), 220)
        self.assertGreaterEqual(int(ring[13, -1]), 220)
        self.assertAlmostEqual(ring[:, 17].sum() / 255, 2, delta=0.2)
        self.assertAlmostEqual(ring[13, :].sum() / 255, 2, delta=0.2)
        self.assertLessEqual(int(ring[2:-2, 2:-2].max()), 5)
        self.assertEqual(inner.getpixel((17, 13)), 255)

    def test_zero_width_border_has_no_visible_ring(self):
        for radius in (0, 6, 14.5):
            with self.subTest(radius=radius):
                outer, inner, ring = _border_mask(67, 45, radius, 0)
                self.assertEqual(outer.tobytes(), inner.tobytes())
                self.assertFalse(np.any(ring))

    def test_subpixel_strokes_remain_visible_and_proportional(self):
        for border_width in (0.25, 0.5, 0.75):
            with self.subTest(border=border_width):
                _, _, ring = _border_mask(67, 45, 10, border_width)
                self.assertGreater(int(ring[0, 33]), 0)
                self.assertAlmostEqual(
                    ring[:22, 33].sum() / 255, border_width, delta=0.15,
                )

    def test_curve_cache_reuses_geometry_and_stays_bounded_during_resizing(self):
        cache = rendering._corner_patch
        cache.cache_clear()
        try:
            rendering.rounded_rectangle_masks(355, 39, 10, 1.25)
            before = cache.cache_info()
            rendering.rounded_rectangle_masks(555, 39, 10, 1.25)
            after = cache.cache_info()
            self.assertGreater(after.hits, before.hits)
            self.assertEqual(after.misses, before.misses)
            self.assertIsNotNone(after.maxsize)

            for index in range(after.maxsize + 32):
                rendering.rounded_rectangle_masks(81, 51, 6 + index / 100, 1.25)
            self.assertLessEqual(cache.cache_info().currsize, after.maxsize)
        finally:
            cache.cache_clear()

    def test_complementary_border_and_fill_do_not_add_a_dark_seam(self):
        width, height = 67, 45
        outer, inner = rendering.rounded_rectangle_masks(width, height, 10, 1.25)
        background = Image.new("RGB", (width, height), (255, 255, 255))
        surface_color = (43, 97, 149)
        painted = rendering._paint_pair(background, outer, surface_color, inner, surface_color)
        actual = np.asarray(painted).astype(int)
        fully_covered = np.asarray(outer) == 255
        # One value of rounding error is allowed from 8-bit multiplication;
        # blending the two masks independently would leave a visible seam.
        self.assertLessEqual(
            np.abs(actual[fully_covered] - np.asarray(surface_color)).max(), 1,
        )


class SmoothBorderInstallationTests(unittest.TestCase):
    def test_repeated_installation_keeps_the_same_renderer_methods(self):
        rendering.install_smooth_borders()
        draw = DrawEngine.draw_rounded_rect_with_border
        itemconfig = CTkCanvas.itemconfig

        rendering.install_smooth_borders()
        rendering.install_smooth_borders()

        self.assertIs(DrawEngine.draw_rounded_rect_with_border, draw)
        self.assertIs(CTkCanvas.itemconfig, itemconfig)

    def test_draw_engine_preserves_physical_extents_and_fractional_border_insets(self):
        rendering.install_smooth_borders()
        for scale in (1, 1.1, 1.25, 1.5, 1.75, 2):
            with self.subTest(scale=scale):
                canvas = _CanvasStub()
                engine = DrawEngine(canvas)
                width, height = 355 * scale, 39 * scale
                engine.draw_rounded_rect_with_border(width, height, 10 * scale, scale)
                bounds = {}
                for tag in ("border_parts", "inner_parts"):
                    item = canvas.items[canvas.find_withtag(tag)[0]]
                    coordinates = item["coords"]
                    half_stroke = item["options"].get("width", 0) / 2
                    bounds[tag] = (
                        min(coordinates[0::2]) - half_stroke,
                        min(coordinates[1::2]) - half_stroke,
                        max(coordinates[0::2]) + half_stroke,
                        max(coordinates[1::2]) + half_stroke,
                    )
                for actual, expected in zip(bounds["border_parts"], (0, 0, width, height)):
                    self.assertAlmostEqual(actual, expected, delta=0.01)
                for actual, expected in zip(
                    bounds["inner_parts"],
                    (scale, scale, width - scale, height - scale),
                ):
                    self.assertAlmostEqual(actual, expected, delta=0.02)

                # Verify the raster used for native tagged geometry agrees
                # with the pure shape API, including fractional dimensions.
                size = math.ceil(width), math.ceil(height)
                outer = rendering._item_mask(canvas, canvas.find_withtag("border_parts")[0], size)
                inner = rendering._item_mask(canvas, canvas.find_withtag("inner_parts")[0], size)
                expected_outer, expected_inner = rendering.rounded_rectangle_masks(
                    width, height, 10 * scale, scale,
                )
                self.assertEqual(outer.tobytes(), expected_outer.tobytes())
                # Geometry keeps subpixel precision, so arbitrary display
                # scales allow only a small coverage difference.
                difference = np.abs(np.asarray(inner).astype(int) - np.asarray(expected_inner).astype(int))
                self.assertLess(difference.mean(), 0.25)

    def test_classic_and_modern_roundtrip_restores_rounded_corner_cutouts(self):
        rendering.install_smooth_borders()
        for scale in (1, 1.25):
            with self.subTest(scale=scale):
                canvas = _CanvasStub()
                engine = DrawEngine(canvas)
                width, height = 67 * scale, 45 * scale
                size = math.ceil(width), math.ceil(height)

                def coverage(tag):
                    combined = Image.new("L", size, 0)
                    for item in canvas.find_withtag(tag):
                        combined = ImageChops.lighter(
                            combined, rendering._item_mask(canvas, item, size),
                        )
                    return combined

                self.assertTrue(engine.draw_rounded_rect_with_border(
                    width, height, 10 * scale, scale,
                ))
                initial = {tag: coverage(tag) for tag in ("border_parts", "inner_parts")}
                initial_counts = {tag: len(canvas.find_withtag(tag)) for tag in initial}
                self.assertEqual(initial["border_parts"].getpixel((0, 0)), 0)

                self.assertTrue(engine.draw_rounded_rect_with_border(width, height, 0, scale))
                self.assertEqual(coverage("border_parts").getpixel((0, 0)), 255)

                self.assertTrue(engine.draw_rounded_rect_with_border(
                    width, height, 10 * scale, scale,
                ))
                for tag, mask in initial.items():
                    self.assertEqual(len(canvas.find_withtag(tag)), initial_counts[tag])
                    self.assertEqual(coverage(tag).tobytes(), mask.tobytes())


class SmoothSurfacePerformanceTests(unittest.TestCase):
    def test_deferred_changes_coalesce_and_paint_the_final_geometry_and_color(self):
        self.root._aidas_defer_canvas_updates = True
        canvas = CTkCanvas(self.root, width=120, height=40, bg="#112233")
        engine = DrawEngine(canvas)
        with patch.object(canvas, "winfo_width", return_value=120), patch.object(
            canvas, "winfo_height", return_value=40,
        ), patch.object(rendering, "_repaint", wraps=rendering._repaint) as repaint:
            for width, color in ((80, "#334455"), (100, "#556677"), (120, "#4488AA")):
                engine.draw_rounded_rect_with_border(width, 40, 8, 0)
                canvas.itemconfigure("inner_parts", fill=color, outline=color)
                self.root.update_idletasks()
                self.assertEqual(repaint.call_count, 1)
            self.assertIsNone(getattr(canvas, "_aidas_surface_photo", None))
            rendering.finish_deferred_rendering(self.root)
            self.root.update_idletasks()
            self.assertEqual(repaint.call_count, 2)
        pixels = ImageTk.getimage(canvas._aidas_surface_photo)
        self.assertEqual(pixels.size, (120, 40))
        self.assertEqual(pixels.getpixel((110, 20))[:3], (68, 136, 170))
        self.assertFalse(self.root._aidas_deferred_surfaces)

    def test_destroyed_deferred_surface_is_skipped_at_final_paint(self):
        self.root._aidas_defer_canvas_updates = True
        canvas = CTkCanvas(self.root, width=120, height=40, bg="#112233")
        engine = DrawEngine(canvas)
        engine.draw_rounded_rect_with_border(120, 40, 8, 0)
        self.root.update_idletasks()
        canvas.destroy()
        rendering.finish_deferred_rendering(self.root)
        self.root.update_idletasks()
        self.assertFalse(self.root._aidas_deferred_surfaces)

    def test_canvas_moves_do_not_repaint_but_size_changes_do(self):
        canvas = SimpleNamespace()
        with patch.object(rendering, "_schedule") as schedule:
            rendering._canvas_configured(canvas, SimpleNamespace(width=180, height=40, x=0, y=0))
            rendering._canvas_configured(canvas, SimpleNamespace(width=180, height=40, x=0, y=-20))
            rendering._canvas_configured(canvas, SimpleNamespace(width=225, height=50, x=0, y=-20))
            rendering._canvas_configured(canvas, SimpleNamespace(width=225, height=50, x=10, y=-20))
        self.assertEqual(schedule.call_count, 2)

    def setUp(self):
        rendering.install_smooth_borders()
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    @staticmethod
    def _paint(canvas):
        pending = getattr(canvas, "_aidas_paint_pending", None)
        if pending is not None:
            canvas.after_cancel(pending)
        rendering._repaint(canvas)

    def test_scrollbar_hit_regions_reuse_masks_preserve_pixels_and_survive_hover_and_motion(self):
        for scale in (1, 1.25, 2):
            with self.subTest(scale=scale):
                width, height = math.ceil(14 * scale), math.ceil(120 * scale)
                canvas = CTkCanvas(self.root, width=width, height=height, bg="#112233")
                engine = DrawEngine(canvas)
                canvas.tag_bind("border_parts", "<Button-1>", lambda _event: None)
                canvas.tag_bind("scrollbar_parts", "<Enter>", lambda _event: None)
                with patch.object(canvas, "winfo_width", return_value=width), patch.object(
                    canvas, "winfo_height", return_value=height,
                ):
                    prior_ids = None
                    for start, end, fill in ((0.1, 0.4, "#4488AA"), (0.1, 0.4, "#66AACC"), (0.3, 0.6, "#4488AA")):
                        engine.draw_rounded_scrollbar(
                            14 * scale, 120 * scale, 6 * scale, 2 * scale, start, end, "vertical",
                        )
                        canvas.itemconfigure("border_parts", fill="#112233", outline="#112233")
                        canvas.itemconfigure("scrollbar_parts", fill=fill, outline=fill)
                        with patch.object(rendering, "_item_mask", wraps=rendering._item_mask) as masks:
                            self._paint(canvas)
                        visible = [item for item, state in canvas._aidas_vectors.items() if state != "hidden"]
                        self.assertEqual(masks.call_count, len(visible), "Each primitive must be rasterized only once")
                        surface = ImageTk.getimage(canvas._aidas_surface_photo).convert("RGB")
                        ids = {}
                        for tag in ("border_parts", "scrollbar_parts"):
                            # Independently combine coverage to check the exact
                            # extent of the antialiased interaction region.
                            combined = Image.new("L", (width, height), 0)
                            for item in canvas.find_withtag(tag):
                                combined = ImageChops.lighter(
                                    combined, rendering._item_mask(canvas, item, (width, height)),
                                )
                            expected_bounds = combined.getbbox()
                            item, photo = canvas._aidas_hit_regions[tag]
                            ids[tag] = item
                            self.assertEqual(tk.Canvas.bbox(canvas, item), expected_bounds)
                            actual = ImageTk.getimage(photo).convert("RGB")
                            self.assertEqual(actual.tobytes(), surface.crop(expected_bounds).tobytes())
                        if prior_ids is not None:
                            self.assertEqual(ids, prior_ids, "Hover and movement must retain native binding targets")
                        prior_ids = ids
                canvas.destroy()

    def test_canvas_idle_deferral_is_scoped_and_normal_idle_updates_resume(self):
        flagged = tk.Toplevel(self.root)
        flagged.withdraw()
        other = tk.Toplevel(self.root)
        other.withdraw()
        canvas = CTkCanvas(flagged)
        other_canvas = CTkCanvas(other)
        self.root.update_idletasks()
        calls = []
        flagged._aidas_defer_canvas_updates = True
        canvas.after_idle(lambda: calls.append("pending"))

        canvas.update_idletasks()
        self.assertEqual(calls, [])
        other_canvas.update_idletasks()
        self.assertEqual(calls, ["pending"])

        canvas.after_idle(lambda: calls.append("resumed"))
        flagged._aidas_defer_canvas_updates = False
        canvas.update_idletasks()
        self.assertEqual(calls, ["pending", "resumed"])


if __name__ == "__main__":
    unittest.main()
