"""Smooth, DPI-aware rendering for CustomTkinter's canvas controls.

Tk's polygon geometry remains the source of truth for widget state and layout.
Its tagged items are retained, while curved surfaces are painted together into
a raster image. Only small corner/path tiles are supersampled; large flat
surfaces never allocate an enlarged bitmap.
"""

from __future__ import annotations

from functools import lru_cache, wraps
import math
import tkinter as tk

from PIL import Image, ImageChops, ImageDraw, ImageTk
from customtkinter.windows.widgets.core_rendering import CTkCanvas, DrawEngine
from customtkinter.windows.widgets.core_widget_classes import CTkBaseClass


_SCALE = 4
_GEOMETRY_SCALE = 64
_SURFACE_TAG = "aidas_smooth_surface"
_HIT_TAG = "aidas_smooth_hit_region"
_INSTALLED = False
_PART_TAGS = (
    "background_corner_top_left", "background_corner_top_right",
    "background_corner_bottom_right", "background_corner_bottom_left",
    "border_parts_left", "border_parts_right", "inner_parts_left",
    "inner_parts_right", "border_parts", "inner_parts", "progress_parts",
    "slider_parts", "scrollbar_parts", "checkmark", "dropdown_arrow",
)


@lru_cache(maxsize=128)
def _corner_patch(width: int, height: int, radius: float,
                  phase_x: float, phase_y: float) -> Image.Image:
    """Return one antialiased corner, independent of the full widget size."""
    padding = 3
    patch = Image.new("L", ((width + padding) * _SCALE, (height + padding) * _SCALE), 0)
    draw = ImageDraw.Draw(patch)
    # The opposite corners sit outside this small tile. Pillow clips the path
    # without allocating a bitmap for the complete rectangle.
    draw.rounded_rectangle(
        (phase_x * _SCALE, phase_y * _SCALE,
         (width + radius * 2 + 3) * _SCALE - 1,
         (height + radius * 2 + 3) * _SCALE - 1),
        radius=radius * _SCALE, fill=255,
    )
    return patch.resize((width + padding, height + padding), Image.Resampling.LANCZOS).crop((0, 0, width, height))


def _rectangle_mask(size: tuple[int, int], bounds: tuple[float, ...]) -> Image.Image:
    """Rasterize an exclusive rectangle with exact fractional edge coverage."""
    mask = Image.new("L", size, 0)
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        return mask
    x0, y0 = math.floor(left), math.floor(top)
    x1, y1 = math.ceil(right), math.ceil(bottom)
    width, height = x1 - x0, y1 - y0
    block = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(block)
    x_first = min(1.0, right - x0) - (left - x0)
    x_last = right - max(left, x1 - 1)
    y_first = min(1.0, bottom - y0) - (top - y0)
    y_last = bottom - max(top, y1 - 1)
    if x_first < 1:
        draw.line((0, 0, 0, height - 1), fill=round(255 * x_first))
    if x_last < 1:
        draw.line((width - 1, 0, width - 1, height - 1), fill=round(255 * x_last))
    for row, coverage in ((0, y_first), (height - 1, y_last)):
        if coverage < 1:
            strip = block.crop((0, row, width, row + 1))
            strip = ImageChops.multiply(strip, Image.new("L", strip.size, round(255 * coverage)))
            block.paste(strip, (0, row))
    mask.paste(block, (x0, y0))
    return mask


def _rounded_mask(size: tuple[int, int], bounds: tuple[float, ...], radius: float) -> Image.Image:
    left, top, right, bottom = bounds
    radius = max(0.0, min(float(radius), (right - left) / 2, (bottom - top) / 2))
    mask = _rectangle_mask(size, bounds)
    if not radius:
        return mask
    x0, y0, x1, y1 = math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)
    # Tiles stop at the midpoint, so opposite corners never overwrite one
    # another in circles and short pill-shaped controls.
    extent = math.ceil(radius) + 2
    left_width, right_width = min(extent, (x1 - x0 + 1) // 2), min(extent, (x1 - x0) // 2)
    top_height, bottom_height = min(extent, (y1 - y0 + 1) // 2), min(extent, (y1 - y0) // 2)
    for width, height, phase_x, phase_y, flip_x, flip_y, x, y in (
        (left_width, top_height, left - x0, top - y0, False, False, x0, y0),
        (right_width, top_height, x1 - right, top - y0, True, False, x1 - right_width, y0),
        (right_width, bottom_height, x1 - right, y1 - bottom, True, True, x1 - right_width, y1 - bottom_height),
        (left_width, bottom_height, left - x0, y1 - bottom, False, True, x0, y1 - bottom_height),
    ):
        if not width or not height:
            continue
        patch = _corner_patch(width, height, radius, round(phase_x, 6), round(phase_y, 6))
        if flip_x:
            patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if flip_y:
            patch = patch.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask.paste(patch, (x, y))
    return mask


def rounded_rectangle_masks(width: float, height: float, corner_radius: float,
                            border_width: float) -> tuple[Image.Image, Image.Image]:
    """Return physical-size outer/inner coverage masks for a continuous border.

    Subtract the inner mask from the outer mask to obtain border coverage.
    Their complementary coverage prevents gaps or dark double blending where
    a thin outline meets its fill.
    """
    size = max(1, math.ceil(width)), max(1, math.ceil(height))
    radius = max(0.0, min(corner_radius, width / 2, height / 2))
    border = max(0.0, border_width)
    outer = _rounded_mask(size, (0, 0, width, height), radius)
    inner = _rounded_mask(size, (border, border, width - border, height - border), max(0, radius - border))
    return outer, ImageChops.darker(outer, inner)


class _ScaledCanvas:
    """Keep CTk's rounding below a display pixel, retaining native coordinates."""

    def __init__(self, canvas: CTkCanvas):
        self.canvas = canvas

    def __getattr__(self, name):
        return getattr(self.canvas, name)

    @staticmethod
    def _coords(args):
        return tuple(tuple(value / _GEOMETRY_SCALE for value in arg) if isinstance(arg, (tuple, list))
                     else arg / _GEOMETRY_SCALE for arg in args)

    def coords(self, tag_or_id, *args):
        if not args:
            return [value * _GEOMETRY_SCALE for value in self.canvas.coords(tag_or_id)]
        return self.canvas.coords(tag_or_id, *self._coords(args))

    def _create(self, name, args, options):
        options = dict(options)
        for option in ("width", "activewidth", "disabledwidth"):
            if option in options:
                options[option] /= _GEOMETRY_SCALE
        return getattr(self.canvas, name)(*self._coords(args), **options)

    def create_rectangle(self, *args, **kwargs):
        return self._create("create_rectangle", args, kwargs)

    def create_polygon(self, *args, **kwargs):
        return self._create("create_polygon", args, kwargs)

    def create_oval(self, *args, **kwargs):
        return self._create("create_oval", args, kwargs)

    def create_line(self, *args, **kwargs):
        return self._create("create_line", args, kwargs)

    def itemconfig(self, tag_or_id, *args, **kwargs):
        options = dict(kwargs)
        for option in ("width", "activewidth", "disabledwidth"):
            if option in options:
                options[option] /= _GEOMETRY_SCALE
        return self.canvas.itemconfig(tag_or_id, *args, **options)

    itemconfigure = itemconfig


def _schedule(canvas: CTkCanvas) -> None:
    if getattr(canvas, "_aidas_paint_pending", None) is None and not getattr(canvas, "_aidas_painting", False):
        try:
            canvas._aidas_paint_pending = canvas.after_idle(lambda: _repaint(canvas))
        except tk.TclError:
            pass


def _rgb(canvas: CTkCanvas, color: str) -> tuple[int, int, int]:
    values = canvas.winfo_rgb(color)
    return tuple(round(value / 257) for value in values)


def _part(canvas: CTkCanvas, item: int) -> str:
    tags = canvas.gettags(item)
    return next((tag for tag in _PART_TAGS if tag in tags), str(item))


def _path_mask(size, coords, width, kind, smooth=False):
    """Rasterize uncommon paths in bounded tiles, never a 4x-window bitmap."""
    mask = Image.new("L", size, 0)
    margin = width / 2 + 2
    xs, ys = coords[::2], coords[1::2]
    x0 = max(0, math.floor(min(xs) - margin))
    y0 = max(0, math.floor(min(ys) - margin))
    x1 = min(size[0], math.ceil(max(xs) + margin))
    y1 = min(size[1], math.ceil(max(ys) + margin))
    # Most paths are tiny checkmarks and dropdown arrows. Tiling also bounds
    # memory for any future large custom path using a CTkCanvas.
    for top in range(y0, y1, 128):
        for left in range(x0, x1, 128):
            right, bottom = min(x1, left + 128), min(y1, top + 128)
            tile = Image.new("L", ((right - left + 4) * _SCALE, (bottom - top + 4) * _SCALE), 0)
            points = [((x - left + 2) * _SCALE, (y - top + 2) * _SCALE) for x, y in zip(xs, ys)]
            draw = ImageDraw.Draw(tile)
            stroke = max(1, round(width * _SCALE))
            if kind == "oval":
                draw.ellipse((*points[0], *points[1]), fill=255)
            elif kind == "polygon":
                draw.polygon(points, fill=255)
                if width:
                    draw.line(points + points[:1], fill=255, width=stroke, joint="curve")
            else:
                draw.line(points, fill=255, width=stroke, joint="curve")
                if smooth and width:
                    radius = stroke / 2
                    for x, y in points:
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
            tile = tile.resize((right - left + 4, bottom - top + 4), Image.Resampling.LANCZOS)
            mask.paste(tile.crop((2, 2, right - left + 2, bottom - top + 2)), (left, top))
    return mask


def _item_mask(canvas: CTkCanvas, item: int, size) -> Image.Image:
    kind = canvas.type(item)
    coords = canvas.coords(item)
    width = float(canvas.itemcget(item, "width") or 0)
    if kind == "rectangle":
        return _rounded_mask(size, tuple(coords), 0)
    if kind == "polygon":
        xs, ys = coords[::2], coords[1::2]
        if len(coords) >= 8 and len(set(xs)) <= 2 and len(set(ys)) <= 2:
            radius = width / 2
            return _rounded_mask(size, (min(xs) - radius, min(ys) - radius,
                                       max(xs) + radius, max(ys) + radius), radius)
    return _path_mask(size, coords, width, kind,
                      canvas.itemcget(item, "joinstyle") == "round" if kind == "line" else False)


def _paint_pair(base: Image.Image, outer: Image.Image, border_color,
                inner: Image.Image, fill_color) -> Image.Image:
    inner = ImageChops.darker(outer, inner)
    border = ImageChops.subtract(outer, inner)
    background = ImageChops.multiply(base, ImageChops.invert(outer).convert("RGB"))
    border_pixels = ImageChops.multiply(Image.new("RGB", base.size, border_color), border.convert("RGB"))
    fill_pixels = ImageChops.multiply(Image.new("RGB", base.size, fill_color), inner.convert("RGB"))
    return ImageChops.add(ImageChops.add(background, border_pixels), fill_pixels)


def _clear_hit_regions(canvas: CTkCanvas) -> None:
    for item, photo in getattr(canvas, "_aidas_hit_regions", {}).values():
        tk.Canvas.delete(canvas, item)
    canvas._aidas_hit_regions = {}
    canvas._aidas_hit_ids = set()


def _sync_hit_regions(canvas: CTkCanvas, base: Image.Image, items: list[int]) -> None:
    """Retain native tag bindings on dropdown segments and scrollbar thumbs.

    Canvas images hit-test their complete bounds. Crops of the final surface
    therefore preserve these rectangular interaction regions while leaving
    native text and custom icon images above them. Their IDs survive repaint,
    so a hover recolor never generates a spurious mouse leave/enter cycle.
    """
    regions = getattr(canvas, "_aidas_hit_regions", {})
    canvas._aidas_hit_regions = regions
    anchor = canvas.find_withtag(_SURFACE_TAG)[0]
    for tag in getattr(canvas, "_aidas_bound_tags", {}):
        members = [item for item in items if tag in canvas.gettags(item)]
        mask = Image.new("L", base.size, 0)
        for item in members:
            mask = ImageChops.lighter(mask, _item_mask(canvas, item, base.size))
        bounds = mask.getbbox()
        existing = regions.get(tag)
        if bounds is None:
            if existing and canvas.type(existing[0]):
                tk.Canvas.itemconfigure(canvas, existing[0], state="hidden")
            continue
        photo = canvas._aidas_surface_photo if bounds == (0, 0, *base.size) else ImageTk.PhotoImage(base.crop(bounds), master=canvas)
        if existing and canvas.type(existing[0]):
            item = existing[0]
            tk.Canvas.coords(canvas, item, bounds[0], bounds[1])
            tk.Canvas.itemconfigure(canvas, item, image=photo, state="normal")
        else:
            item = tk.Canvas.create_image(canvas, bounds[0], bounds[1], anchor="nw",
                                          image=photo, tags=(_HIT_TAG, tag))
        regions[tag] = item, photo
        tk.Canvas.tag_raise(canvas, item, anchor)
        anchor = item
    canvas._aidas_hit_ids = {item for item, photo in regions.values()}


def _repaint(canvas: CTkCanvas) -> None:
    canvas._aidas_paint_pending = None
    try:
        if not canvas.winfo_exists():
            return
        vectors = getattr(canvas, "_aidas_vectors", {})
        items = [item for item in canvas.find_all() if item in vectors and vectors[item] != "hidden"]
        smoothing = any(
            canvas.type(item) in ("line", "oval")
            or (canvas.type(item) == "polygon" and float(canvas.itemcget(item, "width") or 0) > 0)
            or ("border_parts" in canvas.gettags(item) and "inner_parts" in tuple(tag for other in items for tag in canvas.gettags(other)))
            for item in items
        )
        canvas._aidas_painting = True
        canvas._aidas_smoothing = smoothing
        for item, state in vectors.items():
            if canvas.type(item):
                tk.Canvas.itemconfigure(canvas, item, state="hidden" if smoothing else state)
        if not smoothing:
            tk.Canvas.delete(canvas, _SURFACE_TAG)
            _clear_hit_regions(canvas)
            canvas._aidas_surface_photo = None
            return
        size = max(1, int(canvas.winfo_width())), max(1, int(canvas.winfo_height()))
        base = Image.new("RGB", size, _rgb(canvas, canvas.cget("background")))
        groups = {}
        for item in items:
            color = canvas.itemcget(item, "fill")
            if not color:
                continue
            key = _part(canvas, item), color
            mask = _item_mask(canvas, item, size)
            groups[key] = ImageChops.lighter(groups[key], mask) if key in groups else mask
        consumed = set()
        for key, mask in groups.items():
            if key in consumed:
                continue
            part, color = key
            inner_part = part.replace("border_parts", "inner_parts") if part.startswith("border_parts") else None
            partner = next((candidate for candidate in groups if candidate[0] == inner_part), None)
            if partner is not None:
                base = _paint_pair(base, mask, _rgb(canvas, color), groups[partner], _rgb(canvas, partner[1]))
                consumed.add(partner)
            else:
                base.paste(_rgb(canvas, color), mask=mask)
        canvas._aidas_surface_photo = ImageTk.PhotoImage(base, master=canvas)
        surface = canvas.find_withtag(_SURFACE_TAG)
        if surface:
            tk.Canvas.itemconfigure(canvas, surface[0], image=canvas._aidas_surface_photo)
        else:
            tk.Canvas.create_image(canvas, 0, 0, anchor="nw", image=canvas._aidas_surface_photo, tags=(_SURFACE_TAG,))
        canvas.tag_lower(_SURFACE_TAG)
        _sync_hit_regions(canvas, base, items)
    except tk.TclError:
        pass
    finally:
        canvas._aidas_painting = False


def install_smooth_borders() -> None:
    """Install the shared renderer once before constructing application widgets."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    DrawEngine.preferred_drawing_method = "polygon_shapes"
    original_dimensions = CTkBaseClass._update_dimensions_event

    @wraps(original_dimensions)
    def update_dimensions(self, event):
        scaling = self._get_widget_scaling()
        width, height = event.width / scaling, event.height / scaling
        if self._current_width != width or self._current_height != height:
            self._current_width, self._current_height = width, height
            self._draw(no_color_updates=True)

    CTkBaseClass._update_dimensions_event = update_dimensions
    original_init = DrawEngine.__init__

    @wraps(original_init)
    def engine_init(self, canvas):
        original_init(self, _ScaledCanvas(canvas))
        self.set_round_to_even_numbers(False, False)

    DrawEngine.__init__ = engine_init
    original_canvas_init = CTkCanvas.__init__

    @wraps(original_canvas_init)
    def canvas_init(self, *args, **kwargs):
        original_canvas_init(self, *args, **kwargs)
        # Canvas allocation can change even when its owning CTk widget keeps
        # exactly its requested dimensions. Repaint after the canvas itself
        # receives its final size instead of keeping an initial 1-pixel image.
        self.bind("<Configure>", lambda event: _schedule(self), add="+")

    CTkCanvas.__init__ = canvas_init
    original_tag_bind = CTkCanvas.tag_bind

    @wraps(original_tag_bind)
    def tag_bind(self, tag_or_id, sequence=None, func=None, add=None):
        if func is not None:
            if not hasattr(self, "_aidas_bound_tags"):
                self._aidas_bound_tags = {}
            self._aidas_bound_tags[tag_or_id] = None
            _schedule(self)
        return original_tag_bind(self, tag_or_id, sequence, func, add)

    CTkCanvas.tag_bind = tag_bind
    original_find = CTkCanvas.find_withtag

    @wraps(original_find)
    def find_withtag(self, tag_or_id):
        result = original_find(self, tag_or_id)
        if tag_or_id == "current":
            return result
        hit_ids = getattr(self, "_aidas_hit_ids", ())
        return tuple(item for item in result if item not in hit_ids)

    CTkCanvas.find_withtag = find_withtag
    dimensions = {
        "draw_background_corners": ("width", "height"),
        "draw_rounded_rect_with_border": ("width", "height", "corner_radius", "border_width"),
        "draw_rounded_rect_with_border_vertical_split": ("width", "height", "corner_radius", "border_width", "left_section_width"),
        "draw_rounded_progress_bar_with_border": ("width", "height", "corner_radius", "border_width"),
        "draw_rounded_slider_with_border_and_button": ("width", "height", "corner_radius", "border_width", "button_length", "button_corner_radius"),
        "draw_rounded_scrollbar": ("width", "height", "corner_radius", "border_spacing"),
        "draw_checkmark": ("width", "height", "size"),
        "draw_dropdown_arrow": ("x_position", "y_position", "size"),
    }
    import inspect
    for name, names in dimensions.items():
        original = getattr(DrawEngine, name)
        signature = inspect.signature(original)

        def make_draw(original, signature, names, name):
            @wraps(original)
            def draw(self, *args, **kwargs):
                bound = signature.bind(self, *args, **kwargs)
                for field in names:
                    if field in bound.arguments:
                        bound.arguments[field] *= _GEOMETRY_SCALE
                if name == "draw_rounded_rect_with_border":
                    radius = bound.arguments.get("corner_radius", 0)
                    # Keep CTk's established tags and zero-cost native painting
                    # for the many ordinary square application fills.
                    method = "font_shapes" if not radius else "polygon_shapes"
                    previous = getattr(self, "_aidas_rect_backend", None)
                    if previous is not None and previous != method:
                        # A Classic/Modern switch changes primitive families.
                        # Old square rectangles must not remain underneath a
                        # new rounded polygon and fill its corner cutouts.
                        self._canvas.delete("inner_parts", "border_parts")
                    self._aidas_rect_backend = method
                    bound.arguments["overwrite_preferred_drawing_method"] = method
                return original(*bound.args, **bound.kwargs)
            return draw

        setattr(DrawEngine, name, make_draw(original, signature, names, name))

    # CTk's polygon backend subtracts one extra pixel on square inner shapes
    # as compensation for Tk's native stroke rasterizer. Our coverage masks
    # have no such inaccuracy, so keep the two inset edges concentric instead.
    def polygon_rectangle(self, width, height, radius, border, inner_radius):
        canvas = self._canvas
        changed = False
        if border > 0:
            if not canvas.find_withtag("border_parts"):
                canvas.create_polygon((0, 0, 0, 0), tags=("border_line_1", "border_parts"))
                changed = True
            canvas.coords("border_line_1", radius, radius, width - radius, radius,
                          width - radius, height - radius, radius, height - radius)
            canvas.itemconfig("border_line_1", joinstyle=tk.ROUND, width=radius * 2)
        else:
            canvas.delete("border_parts")
        if not canvas.find_withtag("inner_parts"):
            canvas.create_polygon((0, 0, 0, 0), tags=("inner_line_1", "inner_parts"), joinstyle=tk.ROUND)
            changed = True
        inset = border + inner_radius
        canvas.coords("inner_line_1", inset, inset, width - inset, inset,
                      width - inset, height - inset, inset, height - inset)
        canvas.itemconfig("inner_line_1", width=inner_radius * 2)
        if changed:
            canvas.tag_lower("inner_parts")
            canvas.tag_lower("border_parts")
            canvas.tag_lower("background_parts")
        return changed

    DrawEngine._DrawEngine__draw_rounded_rect_with_border_polygon_shapes = polygon_rectangle

    for name in ("create_rectangle", "create_polygon", "create_line", "create_oval"):
        original = getattr(CTkCanvas, name)

        def make_create(original):
            @wraps(original)
            def create(self, *args, **kwargs):
                item = original(self, *args, **kwargs)
                if not hasattr(self, "_aidas_vectors"):
                    self._aidas_vectors = {}
                self._aidas_vectors[item] = kwargs.get("state", "normal")
                if getattr(self, "_aidas_smoothing", False):
                    tk.Canvas.itemconfigure(self, item, state="hidden")
                _schedule(self)
                return item
            return create

        setattr(CTkCanvas, name, make_create(original))

    original_coords = CTkCanvas.coords

    @wraps(original_coords)
    def coords(self, tag_or_id, *args):
        target = tag_or_id
        if isinstance(tag_or_id, str):
            matches = self.find_withtag(tag_or_id)
            if matches:
                target = matches[0]
        result = original_coords(self, target, *args)
        if args:
            _schedule(self)
        return result

    CTkCanvas.coords = coords
    original_itemconfig = CTkCanvas.itemconfig

    def itemconfig(self, tag_or_id, cnf=None, **kwargs):
        matches = self.find_withtag(tag_or_id)
        target = matches[0] if matches else tag_or_id
        if cnf is not None and not isinstance(cnf, dict):
            return tk.Canvas.itemconfigure(self, target, cnf, **kwargs)
        options = dict(cnf or {})
        options.update(kwargs)
        if not options:
            return tk.Canvas.itemconfigure(self, target)
        vectors = getattr(self, "_aidas_vectors", {})
        for item in matches:
            values = dict(options)
            if item in vectors:
                if "state" in values:
                    vectors[item] = values["state"]
                if getattr(self, "_aidas_smoothing", False):
                    values["state"] = "hidden"
            original_itemconfig(self, item, **values)
        _schedule(self)

    CTkCanvas.itemconfig = itemconfig
    CTkCanvas.itemconfigure = itemconfig
    original_itemcget = CTkCanvas.itemcget

    @wraps(original_itemcget)
    def itemcget(self, tag_or_id, option):
        items = self.find_withtag(tag_or_id)
        vectors = getattr(self, "_aidas_vectors", {})
        if option == "state" and items and items[0] in vectors:
            return vectors[items[0]]
        return original_itemcget(self, items[0] if items else tag_or_id, option)

    CTkCanvas.itemcget = itemcget
    for name in ("type", "gettags"):
        original = getattr(CTkCanvas, name)

        def make_query(original):
            @wraps(original)
            def query(self, tag_or_id):
                items = self.find_withtag(tag_or_id) if isinstance(tag_or_id, str) else ()
                return original(self, items[0] if items else tag_or_id)
            return query

        setattr(CTkCanvas, name, make_query(original))
    original_delete = CTkCanvas.delete

    @wraps(original_delete)
    def delete(self, *args):
        vectors = getattr(self, "_aidas_vectors", {})
        for tag in args:
            for item in self.find_withtag(tag):
                vectors.pop(item, None)
        result = original_delete(self, *args)
        _schedule(self)
        return result

    CTkCanvas.delete = delete
    original_configure = CTkCanvas.configure

    @wraps(original_configure)
    def configure(self, cnf=None, **kwargs):
        result = original_configure(self, cnf, **kwargs)
        if kwargs or isinstance(cnf, dict):
            _schedule(self)
        return result

    CTkCanvas.configure = configure
    CTkCanvas.config = configure
    original_destroy = CTkCanvas.destroy

    @wraps(original_destroy)
    def destroy(self):
        pending = getattr(self, "_aidas_paint_pending", None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except tk.TclError:
                pass
        self._aidas_paint_pending = None
        self._aidas_surface_photo = None
        _clear_hit_regions(self)
        getattr(self, "_aidas_vectors", {}).clear()
        return original_destroy(self)

    CTkCanvas.destroy = destroy


__all__ = ["install_smooth_borders", "rounded_rectangle_masks"]
