"""ImageCanvas widget — displays numpy images with zoom, pan, ROI selection, and line tracing.

Features:
    - Display 8/16-bit grayscale images (auto-normalised for display)
    - Zoom via mouse-wheel or touchpad scroll/pinch gesture
    - Pan via right-click drag (or scrollbars)
    - Interactive rectangle ROI via left-click drag
    - Interactive polyline tracing with saved overlays
    - Pixel coordinate + value tracking callback
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from aidas.canvas.annotation_labels import LabelSpec, PolylineAnchorIndex, layout_annotation_labels

# Backwards-compatible resampling constant: newer Pillow exposes
# `Image.Resampling`, older versions use module-level constants like
# `Image.NEAREST`.
try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except Exception:
    RESAMPLE_NEAREST = Image.NEAREST


class ImageCanvas(ttk.Frame):
    """Zoomable image canvas with rectangle ROI, line tracing, and vertical marker."""

    RESIZE_REDRAW_DEBOUNCE_MS = 100
    ZOOM_REDRAW_DEBOUNCE_MS = 16
    ZOOM_STEP_FACTOR = 1.25
    WINDOWS_WHEEL_DELTA = 120.0
    MAX_WHEEL_ZOOM_STEPS = 4.0

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        parent,
        *,
        on_roi_change=None,
        on_mouse_move=None,
        on_line_change=None,
        on_line_finish=None,
        on_line_edit=None,
        on_vertical_line_change=None,
        on_zoom_change=None,
        auto_fit_on_resize=False,
    ):
        super().__init__(parent)

        # Callbacks
        self._cb_roi = on_roi_change      # (x, y, w, h)
        self._cb_mouse = on_mouse_move    # (ix, iy, value)
        self._cb_line = on_line_change    # (points)
        self._cb_line_finish = on_line_finish  # () on double-click
        self._cb_line_edit = on_line_edit  # (label, points) after editing a saved line
        self._cb_vertical_line = on_vertical_line_change  # (x or None)
        self._cb_zoom = on_zoom_change    # (zoom)
        self._auto_fit_on_resize = bool(auto_fit_on_resize)

        # Image state
        self._data = None          # numpy (H, W) original
        self._display_data = None  # uint8 display-normalized cache
        self._display_pil = None   # cached PIL wrapper for viewport rendering
        self._photo = None         # current PhotoImage
        self._img_id = None        # canvas item id
        self._img_offset_x = 0.0   # displayed image left on canvas coordinates
        self._img_offset_y = 0.0   # displayed image top on canvas coordinates
        self._base_size = None     # (data_id, zoom, canvas_w, canvas_h)
        self._draw_region = None
        self._resize_redraw_after_id = None

        # Zoom
        self._zoom = 1.0
        self._pending_zoom = None
        self._pending_zoom_focus = None
        self._zoom_redraw_after_id = None
        try:
            self._windowing_system = str(
                self.tk.call("tk", "windowingsystem")
            ).lower()
        except tk.TclError:
            self._windowing_system = ""

        # ROI — image-coordinate ints (x, y, w, h) or None
        self._roi = None
        self._roi_on = False
        self._roi_items = []       # canvas ids for rect + handles

        # Line tracing — one active polyline plus saved overlays.
        self._line_on = False
        self._line_overlays = []   # list[dict(points, color, label)]
        self._active_line = []     # list[(x, y)] image-coordinate ints
        self._line_preview = None  # canvas-coordinate preview point
        self._line_color = "#ffff00"
        self._editable_line_label = None
        self._line_edit_drag = None
        self._line_width = 2
        self._label_font_family = "TkDefaultFont"
        self._label_font_size = 4
        self._label_fill = "#ffffff"
        self._label_background = "#111827"
        self._annotation_label_items = {}
        self._annotation_label_sources = []

        # Vertical line marker state
        self._vertical_line_on = False
        self._vertical_line_x = None
        self._vertical_line_color = "#ffd500"
        self._drag_vertical_line = False
        self._vertical_line_items = []
        self._fovea_label_offset = 0.0
        self._fovea_label_drawn_offset = 0.0
        self._fovea_label_base_bounds = None
        self._fovea_guidance_target = None
        self._fovea_guidance_renderer = None

        # Drag state
        self._drag = None          # 'tl','tr','bl','br','move' or None
        self._drag_anchor = None   # (canvas_x, canvas_y)
        self._drag_roi0 = None    # ROI at start of drag
        self._is_panning = False
        self._last_cursor = None
        self._last_mouse_sample = None

        self._build_widgets()
        self._bind_events()

    # ------------------------------------------------------------- widgets
    def _build_widgets(self):
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        self._vsb = ttk.Scrollbar(self, orient="vertical", command=self._scroll_y)
        self._hsb = ttk.Scrollbar(self, orient="horizontal", command=self._scroll_x)
        self.canvas.configure(xscrollcommand=self._hsb.set,
                              yscrollcommand=self._vsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._vsb.grid(row=0, column=1, sticky="ns")
        self._hsb.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _bind_events(self):
        self.canvas.bind("<MouseWheel>",      self._on_wheel)
        self.canvas.bind("<Button-4>",        self._on_wheel)
        self.canvas.bind("<Button-5>",        self._on_wheel)
        # Windows touchpad pinch/stretch is exposed as Ctrl+wheel.  Explicit
        # modifier bindings make that gesture win over generic class bindings
        # while retaining ordinary wheel and two-finger-scroll zoom.
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel)
        self.canvas.bind("<Control-Button-4>",   self._on_wheel)
        self.canvas.bind("<Control-Button-5>",   self._on_wheel)
        self.canvas.bind("<Configure>",       self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Escape>", self._on_escape)
        self.canvas.bind("<B1-Motion>",       self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>",   self._on_pan_start)
        self.canvas.bind("<B3-Motion>",       self._on_pan_motion)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<Motion>",          self._on_hover)

    def _scroll_x(self, *args):
        """Scroll horizontally and refresh only the visible image tile."""
        self.canvas.xview(*args)
        self._render_visible_image()

    def _scroll_y(self, *args):
        """Scroll vertically and refresh only the visible image tile."""
        self.canvas.yview(*args)
        self._render_visible_image()

    # ---------------------------------------------------------- public API
    def set_image(self, data: np.ndarray | None):
        """Set an image (H, W) numpy array (any dtype)."""
        self._cancel_pending_zoom()
        self._last_mouse_sample = None
        if data is None:
            self._data = None
            self._display_data = None
            self._display_pil = None
        else:
            arr = np.asarray(data)
            if arr.ndim != 2:
                raise ValueError("ImageCanvas expects a 2-D grayscale image.")
            # Analyze data may be big-endian; normalize to native-endian once.
            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
            self._data = np.ascontiguousarray(arr)
            self._display_data = np.ascontiguousarray(self._to_display(self._data))
            self._display_pil = Image.fromarray(self._display_data, "L")
        self._active_line.clear()
        self._line_overlays.clear()
        self._line_preview = None
        self._editable_line_label = None
        self._line_edit_drag = None
        self._vertical_line_x = None
        self._drag_vertical_line = False
        if data is not None:
            self._auto_zoom()
        self._redraw()
        self._refresh_cursor_for_mode()
        if data is not None:
            self._emit_zoom_change()

    def get_image(self):
        return self._data

    # ROI
    def enable_roi(self, enabled=True):
        enabled = bool(enabled)
        if self._roi_on == enabled:
            return
        self._roi_on = enabled
        if not enabled:
            self._clear_roi()
            self._update_annotation_labels()
        self._refresh_cursor_for_mode()

    # Line tracing
    def enable_line(self, enabled=True):
        enabled = bool(enabled)
        if self._line_on == enabled:
            if not enabled and (self._active_line or self._line_preview is not None):
                self.clear_active_line()
            return
        self._line_on = enabled
        if not self._line_on:
            self.clear_active_line()
        self._refresh_cursor_for_mode()

    def clear_line_overlays(self):
        self._line_overlays.clear()
        self._line_edit_drag = None
        self._redraw_overlays()

    def set_editable_line(self, label):
        """Select a saved line for ImageJ-style vertex and whole-line editing."""
        if self._editable_line_label == label:
            return
        self._editable_line_label = label
        self._line_edit_drag = None
        self._redraw_overlays()

    def _editable_line(self):
        label = getattr(self, "_editable_line_label", None)
        if label is None:
            return None
        return next((item for item in self._line_overlays if item.get("label") == label), None)

    def get_line_handle_indices(self, label):
        """Return the stable editing handles of a line after a user edit."""
        item = next((item for item in self._line_overlays if item.get("label") == label), None)
        if item is None or item.get("handle_indices") is None:
            return None
        return list(item["handle_indices"])

    def _line_handle_indices(self, points, handle_indices=None):
        """Space editing handles in screen pixels without resampling the line."""
        if not points:
            return []
        candidates = (range(len(points)) if handle_indices is None else
                      sorted({0, len(points) - 1, *handle_indices}))
        candidates = [index for index in candidates if 0 <= index < len(points)]
        indices = [candidates[0]]
        # User-added handles can be closer together than the initial AI handles.
        minimum_distance_squared = (16 if handle_indices is None else 8) ** 2
        for index in candidates[1:-1]:
            px, py = self._i2c(*points[indices[-1]])
            cx, cy = self._i2c(*points[index])
            if (cx - px) ** 2 + (cy - py) ** 2 >= minimum_distance_squared:
                indices.append(index)
        if len(points) > 1:
            cx, cy = self._i2c(*points[-1])
            while len(indices) > 1:
                px, py = self._i2c(*points[indices[-1]])
                if (cx - px) ** 2 + (cy - py) ** 2 >= minimum_distance_squared:
                    break
                indices.pop()
            indices.append(len(points) - 1)
        return indices

    def _hit_line_vertex(self, points, cx, cy, handle_indices=None):
        closest = None
        distance = 36.0
        for index in self._line_handle_indices(points, handle_indices):
            px, py = self._i2c(*points[index])
            candidate = (px - cx) ** 2 + (py - cy) ** 2
            if candidate <= distance:
                closest, distance = index, candidate
        return closest

    def _hit_line_segment(self, points, cx, cy):
        for start, end in zip(points, points[1:]):
            ax, ay = self._i2c(*start)
            bx, by = self._i2c(*end)
            dx, dy = bx - ax, by - ay
            length_squared = dx * dx + dy * dy
            fraction = 0.0 if length_squared == 0 else max(0.0, min(1.0, ((cx-ax)*dx + (cy-ay)*dy)/length_squared))
            if (cx - ax - fraction*dx) ** 2 + (cy - ay - fraction*dy) ** 2 <= 25:
                return True
        return False

    @staticmethod
    def _replace_line_points(item, start, stop, replacement):
        """Replace one handle's local span and keep all other handles stable."""
        delta = len(replacement) - (stop - start)
        handles = [index if index < start else index + delta
                   for index in item["handle_indices"] if index < start or index >= stop]
        handles.extend(range(start, start + len(replacement)))
        item["points"][start:stop] = replacement
        item["handle_indices"] = sorted(set(handles))

    def _begin_line_edit(self, event, cx, cy):
        item = self._editable_line()
        if item is None:
            return False
        points = item["points"]
        handles = self._line_handle_indices(points, item.get("handle_indices"))
        vertex = self._hit_line_vertex(points, cx, cy, item.get("handle_indices"))
        if vertex is None and not self._hit_line_segment(points, cx, cy):
            return False
        self.canvas.focus_set()
        before = list(points)
        modifiers = int(getattr(event, "state", 0))
        # Modifier bits differ across Tk platforms. On Windows, Mod1 (0x8)
        # is Num Lock; treating it as Alt would delete on every plain click.
        windowing_system = getattr(self, "_windowing_system", "")
        alt_mask = {"win32": 0x20000, "aqua": 0x0010}.get(windowing_system, 0x0008)
        if item.get("handle_indices") is None:
            item["handle_indices"] = list(handles)
        if vertex is not None:
            handle = handles.index(vertex)
            previous_index = handles[max(0, handle - 1)]
            following_index = handles[min(len(handles) - 1, handle + 1)]
            start = previous_index + 1 if vertex > 0 else 0
            stop = following_index if vertex < len(points) - 1 else len(points)
        if vertex is not None and modifiers & (alt_mask | 0x0004):
            # Remove the visible handle and its dense AI samples; otherwise a
            # neighboring sample would immediately appear in its place.
            if len(handles) > 3:
                self._replace_line_points(item, start, stop, [])
                self._redraw_overlays()
                if self._cb_line_edit is not None:
                    self._cb_line_edit(item["label"], list(points))
            return True
        if vertex is not None and modifiers & 0x0001:
            # ImageJ splits a vertex into two points along its adjacent segments.
            previous = points[previous_index]
            point = points[vertex]
            following = points[following_index]
            first = self._clamp_line_point(*[a + 2*(b-a)/3 for a, b in zip(previous, point)])
            second = self._clamp_line_point(*[a + (b-a)/3 for a, b in zip(point, following)])
            self._replace_line_points(item, start, stop, [first, second])
            vertex = start + 1
        self._line_edit_drag = {
            "item": item, "vertex": vertex, "before": before,
            "origin": self._c2i(cx, cy), "points": list(points),
        }
        self._redraw_overlays()
        return True

    def set_line_overlays(self, overlays):
        # Restoring a tab often supplies the exact lines already on its canvas.
        # Compare before cleaning thousands of vertices or recreating Tk items.
        if isinstance(overlays, (list, tuple)) and len(overlays) == len(self._line_overlays):
            unchanged = True
            for incoming, current in zip(overlays, self._line_overlays):
                if (not isinstance(incoming, dict)
                        or not isinstance(incoming.get("points"), list)
                        or not all(isinstance(point, tuple) for point in incoming["points"])
                        or incoming["points"] != current["points"]
                        or (incoming.get("color") or self._line_color) != current.get("color")
                        or incoming.get("label") != current.get("label")
                        or incoming.get("handle_indices") != current.get("handle_indices")):
                    unchanged = False
                    break
            if unchanged:
                return
        self._line_overlays.clear()
        for overlay in overlays:
            if isinstance(overlay, dict):
                points = overlay.get("points", [])
                color = overlay.get("color") or self._line_color
                label = overlay.get("label")
                handles = overlay.get("handle_indices")
            else:
                points = overlay
                color = self._line_color
                label = None
                handles = None
            cleaned = self._clean_line_points(points)
            if not cleaned:
                continue
            item = {
                "points": cleaned,
                "color": color,
                "label": label,
            }
            if handles is not None:
                # Cleaning can merge coincident points after an edit. Map the
                # original handle indices onto the cleaned polyline.
                index_map = {}
                cleaned_index = -1
                previous = None
                for index, point in enumerate(points):
                    if point is None:
                        continue
                    point = self._clamp_line_point(*point)
                    if point != previous:
                        cleaned_index += 1
                    index_map[index] = cleaned_index
                    previous = point
                item["handle_indices"] = sorted({index_map[index] for index in handles if index in index_map})
            self._line_overlays.append(item)
        self._redraw_overlays()

    def clear_active_line(self):
        changed = bool(self._active_line) or self._line_preview is not None
        self._active_line.clear()
        self._line_preview = None
        if changed:
            self._redraw_overlays()
        self._emit_line_change()

    def set_active_line(self, points):
        self._active_line = self._clean_line_points(points)
        self._line_preview = None
        self._redraw_overlays()
        self._emit_line_change()

    # Vertical line marker
    def enable_vertical_line(self, enabled=True):
        enabled = bool(enabled)
        if self._vertical_line_on == enabled:
            if self._drag_vertical_line:
                self._drag_vertical_line = False
                self._refresh_cursor_for_mode()
            return
        self._vertical_line_on = enabled
        self._drag_vertical_line = False
        self._update_vertical_line_overlay()
        self._refresh_cursor_for_mode()

    def set_vertical_line_x(self, x):
        if x is None:
            self._vertical_line_x = None
        else:
            self._vertical_line_x = self._clamp_image_point(x, 0)[0]
        self._update_vertical_line_overlay()
        self._emit_vertical_line_change()

    def get_vertical_line_x(self):
        return self._vertical_line_x

    def set_fovea_label_offset(self, offset):
        """Move only the decorative marker label by canvas pixels.

        The requested offset survives marker and zoom redraws.  It never moves
        the marker itself or emits an image-coordinate change callback.
        """
        self._fovea_label_offset = float(offset)
        self._position_fovea_label()

    def fovea_label_bounds(self):
        """Return the current label bounds in canvas coordinates, if shown."""
        if not self._vertical_line_on or self._vertical_line_x is None:
            return None
        return self.canvas.bbox("fovea-label")

    def set_fovea_guidance_target(self, x, *, renderer=None):
        """Point a temporary visual cue toward a viewport-relative x position.

        Passing ``None`` removes the arrow. A renderer can draw beyond the
        canvas; it receives the marker's exact screen position, or ``None``
        when hidden. Timing belongs to the owning view.
        """
        previous = self._fovea_guidance_renderer
        if previous is not None and (previous is not renderer or x is None):
            previous(None)
        self._fovea_guidance_target = None if x is None else float(x)
        self._fovea_guidance_renderer = renderer if x is not None else None
        self._draw_fovea_guidance()

    @property
    def vertical_line_x(self):
        return self.get_vertical_line_x()

    @vertical_line_x.setter
    def vertical_line_x(self, x):
        self.set_vertical_line_x(x)

    def clear_vertical_line(self):
        self._vertical_line_x = None
        self._drag_vertical_line = False
        self._update_vertical_line_overlay()
        self._refresh_cursor_for_mode()
        self._emit_vertical_line_change()

    def undo_active_line_vertex(self):
        if self._active_line:
            self._active_line.pop()
            self._line_preview = None
            self._redraw_overlays()
            self._emit_line_change()

    def get_active_line(self):
        return list(self._active_line)

    def add_line_overlay(self, points, *, color=None, label=None):
        cleaned = self._clean_line_points(points)
        if not cleaned:
            return
        self._line_overlays.append({
            "points": cleaned,
            "color": color or self._line_color,
            "label": label,
        })
        self._redraw_overlays()

    def commit_active_line(self, *, color=None, label=None):
        if len(self._active_line) < 2:
            return None
        committed = list(self._active_line)
        self._line_overlays.append({
            "points": committed,
            "color": color or self._line_color,
            "label": label,
        })
        self._active_line = []
        self._line_preview = None
        self._redraw_overlays()
        self._emit_line_change()
        return committed

    def get_line_overlays(self):
        return [
            {
                "points": list(item["points"]),
                "color": item["color"],
                "label": item["label"],
            }
            for item in self._line_overlays
        ]

    def set_roi(self, roi):
        """roi = (x, y, w, h) in image coords, or None."""
        if roi is not None:
            self._roi = self._clamp_roi(*roi)
        else:
            self._roi = None
        self._draw_roi()
        if self._cb_roi and self._roi:
            self._cb_roi(self._roi)

    def get_roi(self):
        return self._roi

    # Zoom
    def get_zoom(self):
        return self._zoom

    def set_zoom(self, z):
        self._cancel_pending_zoom()
        self._zoom_around_visible_center(z)

    def fit_to_window(self):
        self._cancel_pending_zoom()
        self._auto_zoom()
        size = (id(self._data), self._zoom, self.canvas.winfo_width(), self.canvas.winfo_height())
        if self._img_id is None or self._base_size != size:
            self._redraw()
        self._emit_zoom_change()

    def _zoom_around_visible_center(self, zoom):
        """Apply zoom around the image point currently at viewport center."""
        if self._data is None:
            return
        viewport_center_x = self.canvas.canvasx(self.canvas.winfo_width() / 2)
        viewport_center_y = self.canvas.canvasy(self.canvas.winfo_height() / 2)
        focus_x, focus_y = self._c2i(viewport_center_x, viewport_center_y)
        image_height, image_width = self._data.shape[:2]
        focus_x = max(0.0, min(float(image_width), focus_x))
        focus_y = max(0.0, min(float(image_height), focus_y))

        self._zoom = max(0.02, min(30.0, float(zoom)))
        self._redraw()
        self._center_view_on_image_point(focus_x, focus_y)
        self._emit_zoom_change()

    def _queue_zoom(self, zoom):
        """Coalesce rapid wheel events into one viewport-sized redraw."""
        if self._data is None:
            return
        if self._pending_zoom_focus is None:
            viewport_center_x = self.canvas.canvasx(self.canvas.winfo_width() / 2)
            viewport_center_y = self.canvas.canvasy(self.canvas.winfo_height() / 2)
            self._pending_zoom_focus = self._c2i(
                viewport_center_x,
                viewport_center_y,
            )
        self._pending_zoom = max(0.02, min(30.0, float(zoom)))
        if self._zoom_redraw_after_id is None:
            self._zoom_redraw_after_id = self.after(
                self.ZOOM_REDRAW_DEBOUNCE_MS,
                self._apply_pending_zoom,
            )

    def _apply_pending_zoom(self):
        """Render the newest queued zoom level and restore its view focus."""
        self._zoom_redraw_after_id = None
        zoom = self._pending_zoom
        focus = self._pending_zoom_focus
        self._pending_zoom = None
        self._pending_zoom_focus = None
        if self._data is None or zoom is None or focus is None:
            return
        image_height, image_width = self._data.shape[:2]
        focus_x = max(0.0, min(float(image_width), focus[0]))
        focus_y = max(0.0, min(float(image_height), focus[1]))
        self._zoom = zoom
        self._redraw()
        self._center_view_on_image_point(focus_x, focus_y)
        self._emit_zoom_change()

    def _emit_zoom_change(self):
        """Notify the owning view after a zoom operation has rendered."""
        if self._cb_zoom is not None:
            self._cb_zoom(self._zoom)

    def _cancel_pending_zoom(self):
        """Cancel a queued wheel redraw when image/zoom state changes directly."""
        if self._zoom_redraw_after_id is not None:
            try:
                self.after_cancel(self._zoom_redraw_after_id)
            except tk.TclError:
                pass
        self._zoom_redraw_after_id = None
        self._pending_zoom = None
        self._pending_zoom_focus = None

    def _center_view_on_image_point(self, image_x, image_y):
        """Scroll so one image-space point is centered whenever possible."""
        if self._draw_region is None:
            return
        _left, _top, draw_width, draw_height = self._draw_region
        viewport_width = max(1, self.canvas.winfo_width())
        viewport_height = max(1, self.canvas.winfo_height())
        canvas_x, canvas_y = self._i2c(image_x, image_y)

        max_left = max(0.0, draw_width - viewport_width)
        max_top = max(0.0, draw_height - viewport_height)
        desired_left = max(0.0, min(max_left, canvas_x - viewport_width / 2))
        desired_top = max(0.0, min(max_top, canvas_y - viewport_height / 2))
        self.canvas.xview_moveto(desired_left / max(1.0, draw_width))
        self.canvas.yview_moveto(desired_top / max(1.0, draw_height))
        self._render_visible_image()

    # ---------------------------------------------------------- internals
    def _auto_zoom(self):
        self.update_idletasks()
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        if self._data is None:
            return
        ih, iw = self._data.shape[:2]
        self._zoom = min(cw / iw, ch / ih) * 0.95

    def _to_display(self, data):
        """Normalise to uint8 for display using ImageJ-like min/max scaling."""
        if data.dtype == np.uint8:
            return data
        if np.issubdtype(data.dtype, np.integer):
            # OCT inputs are normally signed/unsigned 16-bit. Integer arrays
            # cannot contain NaN/Inf, so avoid allocating and scanning a full
            # finite-value mask for every image shown in the Step 2 picker.
            lo = float(np.min(data))
            hi = float(np.max(data))
            d = np.asarray(data, dtype=np.float64)
            if hi > lo:
                d -= lo
                d *= 255.0 / (hi - lo)
            np.clip(d, 0, 255, out=d)
            return d.astype(np.uint8)
        # Always detach display normalization from caller-owned floating-point
        # arrays.  The in-place operations below must never alter source data
        # that later workflow steps may save or process.
        d = np.array(data, dtype=np.float64, copy=True)
        finite = np.isfinite(d)
        if not np.any(finite):
            return np.zeros(data.shape, dtype=np.uint8)
        if not np.all(finite):
            d[~finite] = np.nan
        lo = float(np.nanmin(d))
        hi = float(np.nanmax(d))
        if hi > lo:
            d -= lo
            d *= 255.0 / (hi - lo)
        else:
            d[finite] = np.clip(d[finite], 0, 255)
        np.clip(d, 0, 255, out=d)
        d[~finite] = 0.0
        return d.astype(np.uint8)

    def _redraw(self):
        self._cancel_resize_redraw()
        self.canvas.delete("all")
        self._annotation_label_items = {}
        self._annotation_label_sources = []
        self._fovea_label_base_bounds = None
        self._fovea_label_drawn_offset = 0.0
        self._roi_items.clear()
        self._img_id = None
        if self._data is None:
            self._draw_fovea_guidance()
            return
        disp = self._display_data
        if disp is None:
            disp = np.ascontiguousarray(self._to_display(self._data))
            self._display_data = disp
            self._display_pil = Image.fromarray(disp, "L")
        zw, zh, draw_w, draw_h, offset_x, offset_y = self._display_geometry(disp)
        self._img_offset_x = offset_x
        self._img_offset_y = offset_y
        self.canvas.configure(scrollregion=(0, 0, draw_w, draw_h))
        self._draw_region = (0, 0, draw_w, draw_h)
        self._base_size = (id(self._data), self._zoom, self.canvas.winfo_width(), self.canvas.winfo_height())
        self._render_visible_image()
        self._redraw_overlays()

    def _render_visible_image(self):
        """Render only source pixels intersecting the current canvas viewport."""
        if self._data is None or self._display_pil is None:
            return

        image_height, image_width = self._data.shape[:2]
        viewport_width = max(1, self.canvas.winfo_width())
        viewport_height = max(1, self.canvas.winfo_height())
        view_left = self.canvas.canvasx(0)
        view_top = self.canvas.canvasy(0)
        view_right = view_left + viewport_width
        view_bottom = view_top + viewport_height

        image_left = self._img_offset_x
        image_top = self._img_offset_y
        image_right = image_left + image_width * self._zoom
        image_bottom = image_top + image_height * self._zoom
        if (
            view_right <= image_left
            or view_left >= image_right
            or view_bottom <= image_top
            or view_top >= image_bottom
        ):
            if self._img_id is not None:
                self.canvas.delete(self._img_id)
                self._img_id = None
            self._photo = None
            self._draw_fovea_guidance()
            self._hide_annotation_labels()
            return

        # Floor/ceil cover the intersecting source pixels without allocating
        # off-screen margins, keeping the Tk image close to viewport size.
        source_x0 = max(
            0,
            int(np.floor((max(view_left, image_left) - image_left) / self._zoom)),
        )
        source_y0 = max(
            0,
            int(np.floor((max(view_top, image_top) - image_top) / self._zoom)),
        )
        source_x1 = min(
            image_width,
            int(np.ceil((min(view_right, image_right) - image_left) / self._zoom)),
        )
        source_y1 = min(
            image_height,
            int(np.ceil((min(view_bottom, image_bottom) - image_top) / self._zoom)),
        )
        if source_x1 <= source_x0 or source_y1 <= source_y0:
            return

        tile_width = max(1, int(round((source_x1 - source_x0) * self._zoom)))
        tile_height = max(1, int(round((source_y1 - source_y0) * self._zoom)))
        tile = self._display_pil.crop(
            (source_x0, source_y0, source_x1, source_y1)
        ).resize((tile_width, tile_height), RESAMPLE_NEAREST)
        self._photo = ImageTk.PhotoImage(tile)
        tile_x = image_left + source_x0 * self._zoom
        tile_y = image_top + source_y0 * self._zoom
        if self._img_id is None:
            self._img_id = self.canvas.create_image(
                tile_x,
                tile_y,
                anchor="nw",
                image=self._photo,
            )
        else:
            self.canvas.coords(self._img_id, tile_x, tile_y)
            self.canvas.itemconfigure(self._img_id, image=self._photo)
        self.canvas.tag_lower(self._img_id)
        if self._vertical_line_on:
            self._position_fovea_label()
        self._update_annotation_labels()

    def _display_geometry(self, disp=None):
        if disp is None:
            disp = self._display_data
        if disp is None:
            disp = self._to_display(self._data)
        ih, iw = disp.shape[:2]
        zw = max(1, int(iw * self._zoom))
        zh = max(1, int(ih * self._zoom))
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        draw_w = max(zw, cw)
        draw_h = max(zh, ch)
        offset_x = (draw_w - zw) / 2.0
        offset_y = (draw_h - zh) / 2.0
        return zw, zh, draw_w, draw_h, offset_x, offset_y

    def _refresh_viewport_geometry(self):
        if self._data is None or self._img_id is None:
            self._redraw()
            return
        _zw, _zh, draw_w, draw_h, offset_x, offset_y = self._display_geometry()
        draw_region = (0, 0, draw_w, draw_h)
        changed = (
            self._draw_region != draw_region
            or self._img_offset_x != offset_x
            or self._img_offset_y != offset_y
        )
        self._img_offset_x = offset_x
        self._img_offset_y = offset_y
        self.canvas.configure(scrollregion=draw_region)
        self._draw_region = draw_region
        self._base_size = (id(self._data), self._zoom, self.canvas.winfo_width(), self.canvas.winfo_height())
        self._render_visible_image()
        if changed:
            self._redraw_overlays()

    def _redraw_overlays(self):
        if self._data is None or self._img_id is None:
            self._hide_annotation_labels()
            return
        self.canvas.delete("overlay")
        self._annotation_label_items = {}
        self._annotation_label_sources = []
        self._clear_vertical_line_overlay()
        self._roi_items.clear()
        self._draw_roi()
        self._draw_line_overlays()
        self._draw_active_line()
        self._draw_vertical_line()
        self._annotation_label_sources = [
            (f"saved-{index}", str(item["label"]), item["color"], PolylineAnchorIndex(item["points"]))
            for index, item in enumerate(self._line_overlays)
            if item.get("label") and item.get("points")
        ]
        if self._active_line:
            self._annotation_label_sources.append(
                ("active", "active", self._line_color, PolylineAnchorIndex(self._active_line))
            )
        self._update_annotation_labels()

    def _clear_vertical_line_overlay(self):
        for item_id in self._vertical_line_items:
            self.canvas.delete(item_id)
        self._vertical_line_items = []
        self._fovea_label_drawn_offset = 0.0
        self._fovea_label_base_bounds = None
        self.canvas.delete("fovea-guidance")

    def _update_vertical_line_overlay(self):
        if self._data is None or self._img_id is None:
            return
        self._clear_vertical_line_overlay()
        self._draw_vertical_line()
        self._update_annotation_labels()

    # ---------------------------------------------------------- ROI drawing
    def _clear_roi(self):
        for i in self._roi_items:
            self.canvas.delete(i)
        self._roi_items.clear()

    def _draw_roi(self):
        self._clear_roi()
        if self._roi is None or not self._roi_on:
            return
        x, y, w, h = self._roi
        cx1, cy1 = self._i2c(x, y)
        cx2, cy2 = self._i2c(x + w, y + h)
        # Dashed rectangle
        r = self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                         outline="#FFD700", width=2, dash=(6, 3), tags=("overlay",))
        self._roi_items.append(r)

        def add_roi_label(
            position,
            *,
            text,
            anchor="center",
            angle=0,
            fill="#FFD700",
            tags=(),
        ):
            """Create a fixed ROI label with a soft contrast backing."""
            label = self.canvas.create_text(
                *position,
                anchor=anchor,
                text=text,
                angle=angle,
                fill=fill,
                font=("Segoe UI", 9, "bold"),
                tags=("overlay", "roi-measurement", *tags),
            )
            bbox = self.canvas.bbox(label)
            if bbox is not None:
                pad_x, pad_y = 4, 3
                outer_shadow = self.canvas.create_rectangle(
                    bbox[0] - pad_x - 2,
                    bbox[1] - pad_y - 2,
                    bbox[2] + pad_x + 2,
                    bbox[3] + pad_y + 2,
                    fill="#303030",
                    outline="",
                    tags=("overlay", "roi-label-shadow"),
                )
                inner_shadow = self.canvas.create_rectangle(
                    bbox[0] - pad_x - 1,
                    bbox[1] - pad_y - 1,
                    bbox[2] + pad_x + 1,
                    bbox[3] + pad_y + 1,
                    fill="#181818",
                    outline="",
                    tags=("overlay", "roi-label-shadow"),
                )
                background = self.canvas.create_rectangle(
                    bbox[0] - pad_x,
                    bbox[1] - pad_y,
                    bbox[2] + pad_x,
                    bbox[3] + pad_y,
                    fill="#000000",
                    outline="",
                    tags=("overlay", "roi-label-background"),
                )
                self.canvas.tag_lower(outer_shadow, label)
                self.canvas.tag_lower(inner_shadow, label)
                self.canvas.tag_lower(background, label)
                self._roi_items.extend((outer_shadow, inner_shadow))
                self._roi_items.append(background)
            self._roi_items.append(label)

        add_roi_label(
            (cx1 - 7, cy1 - 7),
            anchor="se",
            text=f"({x}, {y})",
            fill="#ffffff",
            tags=("roi-origin",),
        )
        add_roi_label(
            ((cx1 + cx2) / 2, cy1 - 7),
            anchor="s",
            text=f"W: {w}",
            tags=("roi-dimensions", "roi-width"),
        )
        add_roi_label(
            (cx2 + 9, (cy1 + cy2) / 2),
            text=f"H: {h}",
            angle=90,
            tags=("roi-dimensions", "roi-height"),
        )
        # Dim outside ROI
        # (skipped for performance — could add later)
        # Corner handles (all same default color)
        hs = 5
        for hx, hy, color in [
            (cx1, cy1, "#DA0404"),
            (cx2, cy1, "#FFD700"),
            (cx1, cy2, "#FFD700"),
            (cx2, cy2, "#FFD700"),
        ]:
            sq = self.canvas.create_rectangle(hx - hs, hy - hs, hx + hs, hy + hs,
                                              fill=color, outline="black", tags=("overlay",))
            self._roi_items.append(sq)
        self._update_annotation_labels()

    # ---------------------------------------------------------- line drawing
    def _clean_line_points(self, points):
        cleaned = []
        for point in points:
            if point is None:
                continue
            x, y = point
            cleaned_point = self._clamp_line_point(x, y)
            if not cleaned or cleaned[-1] != cleaned_point:
                cleaned.append(cleaned_point)
        return cleaned

    def _clamp_line_point(self, x, y):
        """Keep subpixel vertices so Shift-click can split closely spaced points."""
        x, y = float(x), float(y)
        if self._data is None:
            return x, y
        height, width = self._data.shape[:2]
        return max(0.0, min(x, width - 1)), max(0.0, min(y, height - 1))

    def _clamp_image_point(self, x, y):
        if self._data is None:
            return (int(round(x)), int(round(y)))
        ih, iw = self._data.shape[:2]
        ix = max(0, min(int(round(x)), iw - 1))
        iy = max(0, min(int(round(y)), ih - 1))
        return (ix, iy)

    def _emit_line_change(self):
        if self._cb_line is not None:
            self._cb_line(list(self._active_line))

    def _emit_vertical_line_change(self):
        if self._cb_vertical_line is not None:
            self._cb_vertical_line(self._vertical_line_x)

    def _hide_annotation_labels(self):
        for item in getattr(self, "_annotation_label_items", {}).values():
            for name in ("text", "background", "link"):
                self.canvas.itemconfigure(item[name], state="hidden")

    def _update_annotation_labels(self):
        """Lay out only the label layer, leaving trace geometry untouched."""
        sources = getattr(self, "_annotation_label_sources", ())
        if not sources or self._data is None or self._img_id is None:
            self._hide_annotation_labels()
            return
        left, top = self.canvas.canvasx(0), self.canvas.canvasy(0)
        right = left + self.canvas.winfo_width()
        bottom = top + self.canvas.winfo_height()
        if right - left <= 8 or bottom - top <= 8:
            self._hide_annotation_labels()
            return
        image_view = (*self._c2i(left + 4, top + 4), *self._c2i(right - 4, bottom - 4))
        specs = []
        visible = {}
        for key, label, color, geometry in sources:
            anchor = geometry.anchor(image_view)
            item = self._annotation_label_items.get(key)
            if anchor is None:
                if item is not None:
                    for name in ("text", "background", "link"):
                        self.canvas.itemconfigure(item[name], state="hidden")
                continue
            anchor = self._i2c(*anchor)
            if item is None:
                identity = f"annotation-label-{key}"
                link = self.canvas.create_line(
                    0, 0, 0, 0, fill=color, width=1,
                    tags=("overlay", "annotation-label-link", identity),
                )
                background, text = self._draw_text_label(
                    0, 0, text=label, accent=color,
                    tags=("overlay", "annotation-label", identity),
                    font=self._scaled_annotation_label_font(),
                )
                self.canvas.addtag_withtag("annotation-label-text", text)
                self.canvas.addtag_withtag("annotation-label-background", background)
                item = self._annotation_label_items[key] = {
                    "text": text, "background": background, "link": link,
                    "label": label, "color": color,
                }
            for name in ("text", "background", "link"):
                self.canvas.itemconfigure(item[name], state="normal")
            bounds = self.canvas.bbox(item["background"])
            item["anchor"] = anchor
            item["size"] = (bounds[2] - bounds[0], bounds[3] - bounds[1])
            specs.append(LabelSpec(key, anchor, *item["size"]))
            visible[key] = item

        obstacles = []
        fovea_bounds = self.fovea_label_bounds()
        if fovea_bounds is not None:
            # Reserve a small envelope so a brief fovea label cue cannot bump
            # the packed boundary labels or require relayout on every frame.
            obstacles.append((fovea_bounds[0] - 8, fovea_bounds[1] - 2,
                              fovea_bounds[2] + 8, fovea_bounds[3] + 2))
        for roi_label in self.canvas.find_withtag("roi-label-background"):
            bounds = self.canvas.bbox(roi_label)
            if bounds is not None:
                obstacles.append(bounds)
        boxes = layout_annotation_labels(specs, (left, top, right, bottom), obstacles=obstacles)
        for key, item in visible.items():
            box = boxes.get(key)
            if box is None:
                for name in ("text", "background", "link"):
                    self.canvas.itemconfigure(item[name], state="hidden")
                continue
            current = self.canvas.bbox(item["background"])
            dx, dy = box[0] - current[0], box[1] - current[1]
            self.canvas.move(item["background"], dx, dy)
            self.canvas.move(item["text"], dx, dy)
            x1, y1, x2, y2 = self.canvas.bbox(item["background"])
            ax, ay = item["anchor"]
            if ax <= x1:
                endpoint = (x1, (y1 + y2) / 2)
            elif ax >= x2:
                endpoint = (x2, (y1 + y2) / 2)
            elif ay < y1:
                endpoint = (ax, y1)
            else:
                endpoint = (ax, y2)
            self.canvas.coords(item["link"], ax, ay, *endpoint)
        self.canvas.tag_raise("annotation-label-link")
        self.canvas.tag_raise("annotation-label")

    def _draw_line_overlays(self):
        for item in self._line_overlays:
            selected = self._editable_line_label is not None and item.get("label") == self._editable_line_label
            self._draw_polyline(
                item["points"], item["color"],
                show_vertices=selected, handle_indices=item.get("handle_indices"),
            )

    def _draw_active_line(self):
        if not self._active_line:
            return
        self._draw_polyline(
            self._active_line,
            self._line_color,
            preview=self._line_preview,
            show_vertices=True,
        )

    def _draw_vertical_line(self):
        if self._data is None or self._vertical_line_x is None or not self._vertical_line_on:
            self._draw_fovea_guidance()
            return
        ih, _iw = self._data.shape[:2]
        x_top, y_top = self._i2c(self._vertical_line_x, 0)
        x_bottom, y_bottom = self._i2c(self._vertical_line_x, ih - 1)
        width = self._scaled_line_width(4)
        line_id = self.canvas.create_line(
            x_top,
            y_top,
            x_bottom,
            y_bottom,
            fill=self._vertical_line_color,
            width=width,
            dash=(5, 3),
            tags=("vertical_overlay", "fovea-marker"),
        )
        label_ids = self._draw_text_label(
            x_top + 8,
            y_top + 6,
            text=f"Fovea x={self._vertical_line_x}",
            accent=self._vertical_line_color,
            tags=("vertical_overlay", "fovea-label"),
        )
        self._vertical_line_items = [line_id, *label_ids]
        self._fovea_label_drawn_offset = 0.0
        self._fovea_label_base_bounds = self.fovea_label_bounds()
        self._position_fovea_label()

    def _position_fovea_label(self):
        bounds = getattr(self, "_fovea_label_base_bounds", None)
        if bounds is None:
            self._draw_fovea_guidance()
            return
        drawn_offset = getattr(self, "_fovea_label_drawn_offset", 0.0)
        base_left, base_right = bounds[0], bounds[2]
        view_left = self.canvas.canvasx(0) + 2
        view_right = self.canvas.canvasx(self.canvas.winfo_width()) - 2
        offset = getattr(self, "_fovea_label_offset", 0.0)
        marker_x, _marker_y = self._i2c(self._vertical_line_x, 0)
        near_viewport = (
            base_right >= view_left and base_left <= view_right
        ) or view_left - 8 <= marker_x <= view_right + 8
        if near_viewport:
            if base_right - base_left > view_right - view_left:
                # Keep text readable even if a narrow viewport cannot fit it.
                offset = view_left - base_left
            else:
                offset = max(view_left - base_left, min(offset, view_right - base_right))
        self.canvas.move("fovea-label", offset - drawn_offset, 0)
        self._fovea_label_drawn_offset = offset
        self._draw_fovea_guidance()

    def _draw_fovea_guidance(self):
        self.canvas.delete("fovea-guidance")
        target = getattr(self, "_fovea_guidance_target", None)
        renderer = getattr(self, "_fovea_guidance_renderer", None)
        if target is None:
            return
        if self._data is None or not self._vertical_line_on or self._vertical_line_x is None:
            if renderer is not None:
                renderer(None)
            return
        # Anchor to the marker itself. Its label sits to the right and can
        # animate or clamp against the viewport without moving this origin.
        start_x, start_y = self._i2c(self._vertical_line_x, 0)
        view_left = self.canvas.canvasx(0)
        view_top = self.canvas.canvasy(0)
        view_width = self.canvas.winfo_width()
        view_bottom = view_top + self.canvas.winfo_height()
        if (
            start_x < view_left or start_x > view_left + view_width
            or start_y < view_top or start_y > view_bottom
        ):
            if renderer is not None:
                renderer(None)
            return
        if renderer is not None:
            renderer((
                self.canvas.winfo_rootx() + start_x - view_left,
                self.canvas.winfo_rooty() + start_y - view_top,
            ))
            return
        end_x = view_left + max(12, min(target, view_width - 12))
        end_y = view_top + 6
        if start_y <= end_y + 8:
            # There is no room for an upward cue above the marker.
            return
        bend_y = end_y + (start_y - end_y) * 0.35
        self.canvas.create_line(
            start_x, start_y, start_x, bend_y, end_x, end_y,
            smooth=True, splinesteps=16, arrow=tk.LAST,
            arrowshape=(10, 12, 5), width=2,
            fill=self._vertical_line_color,
            tags=("fovea-guidance",),
        )

    def _overlay_zoom_scale(self):
        return max(0.75, min(float(self._zoom), 4.0))

    def _scaled_line_width(self, base_width=None):
        base_width = self._line_width if base_width is None else base_width
        return max(1, min(24, int(round(base_width * self._overlay_zoom_scale()))))

    def _scaled_vertex_radius(self):
        return max(2, min(10, int(round(2 * self._overlay_zoom_scale()))))

    def _scaled_label_font(self):
        size = max(8, min(32, int(round(self._label_font_size * self._overlay_zoom_scale()))))
        return (self._label_font_family, size, "bold")

    def _scaled_annotation_label_font(self):
        # Grow text more gently than image pixels so zoom makes it readable
        # while leaving room to separate labels on closely spaced boundaries.
        size = max(8, min(24, round(11 * max(0.01, self._zoom) ** 0.5)))
        return (self._label_font_family, size, "bold")

    def _scaled_label_padding(self):
        return max(0.25, min(1, int(round(10 * self._overlay_zoom_scale()))))

    def _draw_text_label(self, x, y, *, text, accent, tags, font=None):
        text_id = self.canvas.create_text(
            x,
            y,
            anchor="nw",
            text=str(text),
            fill=self._label_fill,
            font=font or self._scaled_label_font(),
            tags=tags,
        )
        bbox = self.canvas.bbox(text_id)
        if bbox is None:
            return [text_id]
        pad = self._scaled_label_padding()
        rect_id = self.canvas.create_rectangle(
            bbox[0] - pad,
            bbox[1] - pad,
            bbox[2] + pad,
            bbox[3] + pad,
            fill=self._label_background,
            outline=accent,
            width=max(1, min(4, self._scaled_line_width(1))),
            tags=tags,
        )
        self.canvas.tag_lower(rect_id, text_id)
        return [rect_id, text_id]

    def _draw_polyline(self, points, color, preview=None, show_vertices=True, handle_indices=None):
        if not points:
            return

        coords = []
        for ix, iy in points:
            cx, cy = self._i2c(ix, iy)
            coords.extend([cx, cy])

        if len(coords) == 2:
            x, y = coords
            radius = self._scaled_vertex_radius() + 1
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline=color,
                tags=("overlay",),
            )
        else:
            self.canvas.create_line(
                *coords,
                fill=color,
                width=self._scaled_line_width(),
                capstyle="round",
                joinstyle="round",
                tags=("overlay",),
            )

        if show_vertices:
            radius = 3
            for index in self._line_handle_indices(points, handle_indices):
                cx, cy = self._i2c(*points[index])
                self.canvas.create_rectangle(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill=color if index == 0 else "white",
                    outline="black",
                    tags=("overlay",),
                )

        if preview is not None and len(points) >= 1:
            last_x, last_y = self._i2c(*points[-1])
            self.canvas.create_line(
                last_x,
                last_y,
                preview[0],
                preview[1],
                fill=color,
                width=self._scaled_line_width(),
                tags=("overlay",),
            )

    def _clamp_roi(self, x, y, w, h):
        if self._data is None:
            return (int(x), int(y), int(w), int(h))
        ih, iw = self._data.shape[:2]
        x = max(0, min(int(x), iw - 1))
        y = max(0, min(int(y), ih - 1))
        w = max(1, min(int(w), iw - x))
        h = max(1, min(int(h), ih - y))
        return (x, y, w, h)

    # --------------------------------------------------- canvas ↔ image coords
    def _c2i(self, cx, cy):
        return (
            (cx - self._img_offset_x) / self._zoom,
            (cy - self._img_offset_y) / self._zoom,
        )

    def _i2c(self, ix, iy):
        return (
            ix * self._zoom + self._img_offset_x,
            iy * self._zoom + self._img_offset_y,
        )

    # ---------------------------------------------------------- hit-testing
    def _hit(self, cx, cy):
        """Return 'tl','tr','bl','br','move', or None."""
        if self._roi is None:
            return None
        x, y, w, h = self._roi
        rx1, ry1 = self._i2c(x, y)
        rx2, ry2 = self._i2c(x + w, y + h)
        thr = 8
        for tag, hx, hy in [("tl", rx1, ry1), ("tr", rx2, ry1),
                             ("bl", rx1, ry2), ("br", rx2, ry2)]:
            if abs(cx - hx) < thr and abs(cy - hy) < thr:
                return tag
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return "move"
        return None

    def _hit_vertical_line(self, cx):
        if self._data is None or self._vertical_line_x is None or not self._vertical_line_on:
            return False
        line_x, _line_y = self._i2c(self._vertical_line_x, 0)
        return abs(cx - line_x) <= max(8, self._scaled_line_width(4) * 2)

    # ---------------------------------------------------------- mouse events
    @classmethod
    def _wheel_zoom_steps(cls, event, windowing_system=""):
        """Translate wheel/touchpad motion into bounded fractional zoom steps."""

        button = getattr(event, "num", None)
        if button == 4:
            return 1.0
        if button == 5:
            return -1.0

        try:
            delta = float(getattr(event, "delta", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
        if delta == 0:
            return 0.0

        platform = str(windowing_system or "").lower()
        if platform == "win32":
            # A mouse-wheel notch is 120 on Windows, while precision
            # touchpads emit smaller deltas.  Keeping the fraction makes a
            # pinch or two-finger gesture smooth instead of jumping 25% for
            # every high-resolution event.
            steps = delta / cls.WINDOWS_WHEEL_DELTA
        elif platform == "aqua":
            # Tk reports small, already-normalized deltas on macOS.
            steps = delta
        elif abs(delta) >= cls.WINDOWS_WHEEL_DELTA:
            steps = delta / cls.WINDOWS_WHEEL_DELTA
        else:
            steps = 1.0 if delta > 0 else -1.0

        return max(-cls.MAX_WHEEL_ZOOM_STEPS, min(cls.MAX_WHEEL_ZOOM_STEPS, steps))

    def _on_wheel(self, event):
        steps = self._wheel_zoom_steps(
            event,
            getattr(self, "_windowing_system", ""),
        )
        if steps == 0:
            return "break"
        factor = self.ZOOM_STEP_FACTOR ** steps
        base_zoom = self._pending_zoom if self._pending_zoom is not None else self._zoom
        self._queue_zoom(base_zoom * factor)
        return "break"

    def _on_canvas_resize(self, event):
        # Keep image centered whenever the viewport size changes.
        if self._data is None:
            return
        width = max(1, int(getattr(event, "width", self.canvas.winfo_width())))
        height = max(1, int(getattr(event, "height", self.canvas.winfo_height())))
        if not self._canvas_size_changed(width, height):
            self._cancel_resize_redraw()
            return
        self._schedule_resize_redraw()

    def _canvas_size_changed(self, width=None, height=None):
        if self._base_size is None:
            return True
        if width is None:
            width = max(1, int(self.canvas.winfo_width()))
        if height is None:
            height = max(1, int(self.canvas.winfo_height()))
        _data_id, _zoom, last_width, last_height = self._base_size
        return int(last_width) != int(width) or int(last_height) != int(height)

    def _schedule_resize_redraw(self):
        self._cancel_resize_redraw()
        self._resize_redraw_after_id = self.after(
            self.RESIZE_REDRAW_DEBOUNCE_MS,
            self._run_resize_redraw,
        )

    def _cancel_resize_redraw(self):
        if self._resize_redraw_after_id is None:
            return
        try:
            self.after_cancel(self._resize_redraw_after_id)
        except tk.TclError:
            pass
        self._resize_redraw_after_id = None

    def _run_resize_redraw(self):
        self._resize_redraw_after_id = None
        if self._data is None or not self._canvas_size_changed():
            return
        if self._auto_fit_on_resize:
            # Recompute the fit after the resize settles so responsive image
            # previews use the newly available viewport space.
            self._auto_zoom()
            self._redraw()
            self._emit_zoom_change()
        else:
            self._refresh_viewport_geometry()

    def _on_press(self, event):
        if self._data is not None:
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            if self._begin_line_edit(event, cx, cy):
                return
        if self._vertical_line_on and not self._line_on:
            if self._data is None:
                return
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            if self._hit_vertical_line(cx):
                ix, _iy = self._c2i(cx, cy)
                self._vertical_line_x = self._clamp_image_point(ix, 0)[0]
                self._drag_vertical_line = True
                self._update_vertical_line_overlay()
                self._emit_vertical_line_change()
                return

        if self._line_on:
            if self._data is None:
                return
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            self.canvas.focus_set()
            if len(self._active_line) >= 2 and self._cb_line_finish is not None:
                sx, sy = self._i2c(*self._active_line[0])
                if (abs(cx - sx) <= 5 and abs(cy - sy) <= 5) or int(getattr(event, "state", 0)) & 0x0004:
                    self._line_preview = None
                    self._cb_line_finish()
                    return
            point = self._clamp_image_point(*self._c2i(cx, cy))
            if not self._active_line:
                self._active_line = [point]
            elif self._active_line[-1] != point:
                self._active_line.append(point)
            self._line_preview = None
            self._redraw_overlays()
            self._emit_line_change()
            return

        if not self._roi_on:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        hit = self._hit(cx, cy)
        if hit:
            self._drag = hit
            self._drag_anchor = (cx, cy)
            self._drag_roi0 = self._roi
        else:
            # Start a new ROI
            ix, iy = self._c2i(cx, cy)
            self._roi = (int(ix), int(iy), 1, 1)
            self._drag = "br"
            self._drag_anchor = (cx, cy)
            self._drag_roi0 = self._roi
            self._draw_roi()

    def _on_double_click(self, event):
        """Finish a polyline at the double-clicked point, without repeating it."""
        if not self._line_on or self._cb_line_finish is None:
            return self._on_press(event)
        if self._data is None or self._drag_vertical_line or not self._active_line:
            return "break"
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        point = self._clamp_image_point(*self._c2i(cx, cy))
        if self._active_line[-1] != point:
            self._active_line.append(point)
        self._line_preview = None
        self._redraw_overlays()
        self._emit_line_change()
        if len(self._active_line) >= 2:
            self._cb_line_finish()
        return "break"

    def _on_escape(self, _event):
        if self._line_on and self._active_line:
            self.clear_active_line()
            return "break"

    def _on_drag_motion(self, event):
        edit = getattr(self, "_line_edit_drag", None)
        if edit is not None:
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            ix, iy = self._c2i(cx, cy)
            points = edit["item"]["points"]
            if edit["vertex"] is not None:
                points[edit["vertex"]] = self._clamp_line_point(ix, iy)
            else:
                dx, dy = ix - edit["origin"][0], iy - edit["origin"][1]
                old = edit["points"]
                height, width = self._data.shape[:2]
                dx = max(-min(x for x, _ in old), min(dx, width-1-max(x for x, _ in old)))
                dy = max(-min(y for _, y in old), min(dy, height-1-max(y for _, y in old)))
                points[:] = [self._clamp_line_point(x + dx, y + dy) for x, y in old]
            self._redraw_overlays()
            return
        if self._vertical_line_on and self._drag_vertical_line:
            cx = self.canvas.canvasx(event.x)
            ix, _iy = self._c2i(cx, 0)
            new_x = self._clamp_image_point(ix, 0)[0]
            if new_x != self._vertical_line_x:
                self._vertical_line_x = new_x
                self._update_vertical_line_overlay()
                self._emit_vertical_line_change()
            return

        if self._drag is None or self._drag_roi0 is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        dx = (cx - self._drag_anchor[0]) / self._zoom
        dy = (cy - self._drag_anchor[1]) / self._zoom
        ox, oy, ow, oh = self._drag_roi0

        if self._drag == "move":
            nx, ny = ox + dx, oy + dy
            self._roi = self._clamp_roi(nx, ny, ow, oh)
        elif self._drag == "tl":
            nw, nh = ow - dx, oh - dy
            if nw >= 1 and nh >= 1:
                self._roi = self._clamp_roi(ox + dx, oy + dy, nw, nh)
        elif self._drag == "tr":
            nw, nh = ow + dx, oh - dy
            if nw >= 1 and nh >= 1:
                self._roi = self._clamp_roi(ox, oy + dy, nw, nh)
        elif self._drag == "bl":
            nw, nh = ow - dx, oh + dy
            if nw >= 1 and nh >= 1:
                self._roi = self._clamp_roi(ox + dx, oy, nw, nh)
        elif self._drag == "br":
            nw, nh = ow + dx, oh + dy
            if nw >= 1 and nh >= 1:
                self._roi = self._clamp_roi(ox, oy, nw, nh)

        self._draw_roi()
        if self._cb_roi and self._roi:
            self._cb_roi(self._roi)

    def _on_release(self, _event):
        edit = getattr(self, "_line_edit_drag", None)
        self._line_edit_drag = None
        if edit is not None and edit["before"] != edit["item"]["points"] and self._cb_line_edit is not None:
            self._cb_line_edit(edit["item"]["label"], list(edit["item"]["points"]))
        self._drag_vertical_line = False
        self._drag = None
        self._drag_anchor = None
        self._drag_roi0 = None
        self._refresh_cursor_for_mode()

    def _on_pan_start(self, event):
        if self._data is None:
            return
        self._is_panning = True
        self.canvas.scan_mark(event.x, event.y)
        self._set_canvas_cursor("fleur")

    def _on_pan_motion(self, event):
        if not self._is_panning:
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self._render_visible_image()

    def _on_pan_end(self, _event):
        self._is_panning = False
        self._refresh_cursor_for_mode()

    def _set_canvas_cursor(self, cursor):
        if self._last_cursor == cursor:
            return
        self.canvas.configure(cursor=cursor)
        self._last_cursor = cursor

    def _refresh_cursor_for_mode(self):
        if self._data is None:
            self._set_canvas_cursor("")
            return
        if self._is_panning:
            self._set_canvas_cursor("fleur")
        elif self._drag_vertical_line:
            self._set_canvas_cursor("sb_h_double_arrow")
        elif self._line_on:
            self._set_canvas_cursor("crosshair")
        elif self._roi_on:
            self._set_canvas_cursor("crosshair")
        else:
            self._set_canvas_cursor("")

    def _on_hover(self, event):
        if self._data is None:
            self._set_canvas_cursor("")
            return
        if self._drag_vertical_line:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        if self._line_on and self._active_line:
            preview_point = self._clamp_image_point(*self._c2i(cx, cy))
            preview_canvas = self._i2c(*preview_point)
            if self._line_preview != preview_canvas:
                self._line_preview = preview_canvas
                self._redraw_overlays()
        ix, iy = self._c2i(cx, cy)
        ih, iw = self._data.shape[:2]
        if 0 <= int(ix) < iw and 0 <= int(iy) < ih:
            val = self._data[int(iy), int(ix)]
            sample = (int(ix), int(iy), val)
            if self._cb_mouse and sample != self._last_mouse_sample:
                self._last_mouse_sample = sample
                self._cb_mouse(int(ix), int(iy), val)
        # Cursor shape
        if self._is_panning:
            self._set_canvas_cursor("fleur")
            return
        item = self._editable_line()
        if item is not None:
            if self._hit_line_vertex(item["points"], cx, cy, item.get("handle_indices")) is not None:
                self._set_canvas_cursor("crosshair")
                return
            if self._hit_line_segment(item["points"], cx, cy):
                self._set_canvas_cursor("fleur")
                return
        if not self._line_on and self._hit_vertical_line(cx):
            self._set_canvas_cursor("sb_h_double_arrow")
            return
        if self._line_on:
            self._set_canvas_cursor("crosshair")
            return
        if self._roi_on:
            h = self._hit(cx, cy)
            cursors = {"tl": "top_left_corner", "tr": "top_right_corner",
                       "bl": "bottom_left_corner", "br": "bottom_right_corner",
                       "move": "fleur"}
            self._set_canvas_cursor(cursors.get(h, "crosshair"))
            return
        self._set_canvas_cursor("")
