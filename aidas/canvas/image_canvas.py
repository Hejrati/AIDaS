"""ImageCanvas widget — displays numpy images with zoom, pan, ROI selection, and line tracing.

Features:
    - Display 8/16-bit grayscale images (auto-normalised for display)
    - Zoom via mouse-wheel
    - Pan via right-click drag (or scrollbars)
    - Interactive rectangle ROI via left-click drag
    - Interactive polyline tracing with saved overlays
    - Pixel coordinate + value tracking callback
"""

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from aidas.ui.components import AppButton

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

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        parent,
        *,
        on_roi_change=None,
        on_mouse_move=None,
        on_line_change=None,
        on_vertical_line_change=None,
        on_zoom_change=None,
        auto_fit_on_resize=False,
    ):
        super().__init__(parent)

        # Callbacks
        self._cb_roi = on_roi_change      # (x, y, w, h)
        self._cb_mouse = on_mouse_move    # (ix, iy, value)
        self._cb_line = on_line_change    # (points)
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

        # ROI — image-coordinate ints (x, y, w, h) or None
        self._roi = None
        self._roi_on = False
        self._roi_items = []       # canvas ids for rect + handles

        # Line tracing — one active polyline plus saved overlays.
        self._line_on = False
        self._line_overlays = []   # list[dict(points, color, label)]
        self._active_line = []     # list[(x, y)] image-coordinate ints
        self._line_preview = None  # canvas-coordinate preview point
        self._line_color = "#00E5FF"
        self._line_width = 2
        self._label_font_family = "TkDefaultFont"
        self._label_font_size = 4
        self._label_fill = "#ffffff"
        self._label_background = "#111827"

        # Fixed viewport labels used by Step 2 to identify anatomical sides.
        # They live in canvas coordinates (rather than image coordinates) so
        # they stay readable while the image is zoomed or panned.
        self._side_labels = None
        self._side_flip_callback = None
        self._side_flip_button_width = None

        # Vertical line marker state
        self._vertical_line_on = False
        self._vertical_line_x = None
        self._vertical_line_color = "#ffd500"
        self._drag_vertical_line = False
        self._vertical_line_items = []

        # Drag state
        self._drag = None          # 'tl','tr','bl','br','move' or None
        self._drag_anchor = None   # (canvas_x, canvas_y)
        self._drag_roi0 = None    # ROI at start of drag
        self._is_panning = False
        self._last_cursor = None
        self._last_mouse_sample = None

        self._build_widgets()
        self._build_side_flip_button()
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

    def _build_side_flip_button(self):
        """Build a native, anti-aliased action layered over the image canvas."""
        icon_size = 72
        swap_icon = Image.new("RGBA", (icon_size, icon_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(swap_icon)
        line_width = 7
        arrow_color = (255, 255, 255, 255)
        draw.line((13, 23, 56, 23), fill=arrow_color, width=line_width)
        draw.polygon(((56, 12), (70, 23), (56, 34)), fill=arrow_color)
        draw.line((59, 49, 16, 49), fill=arrow_color, width=line_width)
        draw.polygon(((16, 38), (2, 49), (16, 60)), fill=arrow_color)
        self._side_flip_icon = ctk.CTkImage(
            light_image=swap_icon,
            dark_image=swap_icon,
            size=(22, 22),
        )
        self._side_flip_button = AppButton(
            self.canvas,
            text="Swap side",
            variant="success",
            command=self._on_side_flip_clicked,
            image=self._side_flip_icon,
            compound="left",
            width=168,
            height=40,
            corner_radius=20,
            border_width=1,
            fg_color="#16865f",
            hover_color="#1ca675",
            border_color="#58ddb0",
            text_color="#ffffff",
            bg_color="#1e1e1e",
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
        )

    def _bind_events(self):
        self.canvas.bind("<MouseWheel>",      self._on_wheel)
        self.canvas.bind("<Button-4>",        self._on_wheel)
        self.canvas.bind("<Button-5>",        self._on_wheel)
        self.canvas.bind("<Configure>",       self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
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

    def set_side_labels(self, left=None, right=None, *, on_flip=None):
        """Show anatomical sides and an optional viewport-pinned flip action."""
        if left and right:
            self._side_labels = (str(left), str(right))
        else:
            self._side_labels = None
        self._side_flip_callback = on_flip
        self._draw_side_labels()

    # ROI
    def enable_roi(self, enabled=True):
        enabled = bool(enabled)
        if self._roi_on == enabled:
            return
        self._roi_on = enabled
        if not enabled:
            self._clear_roi()
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
        self._redraw_overlays()

    def set_line_overlays(self, overlays):
        self._line_overlays.clear()
        for overlay in overlays:
            if isinstance(overlay, dict):
                points = overlay.get("points", [])
                color = overlay.get("color") or self._line_color
                label = overlay.get("label")
            else:
                points = overlay
                color = self._line_color
                label = None
            cleaned = self._clean_line_points(points)
            if not cleaned:
                continue
            self._line_overlays.append({
                "points": cleaned,
                "color": color,
                "label": label,
            })
        self._redraw_overlays()

    def clear_active_line(self):
        self._active_line.clear()
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
        d = np.asarray(data, dtype=np.float64)
        finite = np.isfinite(d)
        if not np.any(finite):
            return np.zeros(data.shape, dtype=np.uint8)
        if not np.all(finite):
            d = np.where(finite, d, np.nan)
        lo = float(np.nanmin(d))
        hi = float(np.nanmax(d))
        if hi > lo:
            d = np.clip((d - lo) / (hi - lo) * 255.0, 0, 255)
        else:
            d = np.clip(d, 0, 255)
        d = np.nan_to_num(d, nan=0.0, posinf=255.0, neginf=0.0)
        return d.astype(np.uint8)

    def _redraw(self):
        self._cancel_resize_redraw()
        self.canvas.delete("all")
        self._roi_items.clear()
        self._img_id = None
        if self._data is None:
            # The swap action is a real child widget rather than a canvas
            # item, so clearing canvas items alone does not remove it.
            self._draw_side_labels()
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

        self._draw_side_labels()

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

    def _create_rounded_canvas_rect(
        self,
        x0,
        y0,
        x1,
        y1,
        *,
        radius=12,
        fill,
        outline="",
        width=1,
        tags=(),
    ):
        """Create a smooth rounded rectangle without a raster icon asset."""
        radius = max(0, min(float(radius), (x1 - x0) / 2, (y1 - y0) / 2))
        # Repeated tangent points keep Tk's smoothing spline straight along
        # each edge and eliminate the visible seam at the closing corner.
        points = (
            x0 + radius, y0, x0 + radius, y0,
            x1 - radius, y0, x1 - radius, y0,
            x1, y0, x1, y0 + radius, x1, y0 + radius,
            x1, y1 - radius, x1, y1 - radius,
            x1, y1, x1 - radius, y1, x1 - radius, y1,
            x0 + radius, y1, x0 + radius, y1,
            x0, y1, x0, y1 - radius, x0, y1 - radius,
            x0, y0 + radius, x0, y0 + radius,
            x0, y0,
        )
        return self.canvas.create_polygon(
            points,
            smooth=True,
            splinesteps=24,
            fill=fill,
            outline=outline,
            width=width,
            tags=tags,
        )

    def _draw_side_labels(self):
        """Draw modern anatomical-side cards and a centered swap control."""
        self.canvas.delete("side_overlay")
        flip_button = getattr(self, "_side_flip_button", None)
        if self._data is None or not self._side_labels:
            if flip_button is not None:
                flip_button.place_forget()
            return

        viewport_width = max(1, self.canvas.winfo_width())
        view_left = self.canvas.canvasx(0)
        view_top = self.canvas.canvasy(0)
        top = view_top + 14
        height = 46
        margin = 12
        badge_width = max(124, min(184, (viewport_width - 250) / 2))
        left_box = (view_left + margin, top, view_left + margin + badge_width, top + height)
        right_box = (
            view_left + viewport_width - margin - badge_width,
            top,
            view_left + viewport_width - margin,
            top + height,
        )

        side_panels = (
            (left_box, self._side_labels[0], "#f4b942", "#3a2e16", "left"),
            (right_box, self._side_labels[1], "#42c8f5", "#123342", "right"),
        )
        for (x0, y0, x1, y1), label, accent, icon_fill, direction in side_panels:
            self._create_rounded_canvas_rect(
                x0 + 1,
                y0 + 3,
                x1 + 1,
                y1 + 3,
                radius=12,
                fill="#080c13",
                tags=("side_overlay",),
            )
            self._create_rounded_canvas_rect(
                x0,
                y0,
                x1,
                y1,
                radius=12,
                fill="#344158",
                tags=("side_overlay",),
            )
            self._create_rounded_canvas_rect(
                x0 + 1,
                y0 + 1,
                x1 - 1,
                y1 - 1,
                radius=11,
                fill="#141c29",
                tags=("side_overlay",),
            )
            if direction == "left":
                icon_box = (x0 + 8, y0 + 8, x0 + 38, y1 - 8)
                arrow_start, arrow_end = x0 + 31, x0 + 16
                text_x = (x0 + x1 + 30) / 2
            else:
                icon_box = (x1 - 38, y0 + 8, x1 - 8, y1 - 8)
                arrow_start, arrow_end = x1 - 31, x1 - 16
                text_x = (x0 + x1 - 30) / 2
            self._create_rounded_canvas_rect(
                *icon_box,
                radius=9,
                fill=icon_fill,
                tags=("side_overlay",),
            )
            self.canvas.create_line(
                arrow_start,
                (y0 + y1) / 2,
                arrow_end,
                (y0 + y1) / 2,
                fill=accent,
                width=2,
                arrow="last",
                arrowshape=(7, 8, 3),
                tags=("side_overlay",),
            )
            self.canvas.create_text(
                text_x,
                (y0 + y1) / 2,
                anchor="center",
                text=label.upper(),
                fill="#f7f9fc",
                font=("Segoe UI Semibold", 10),
                tags=("side_overlay",),
            )

        if getattr(self, "_side_flip_callback", None) is None or flip_button is None:
            if flip_button is not None:
                flip_button.place_forget()
            return

        button_width = int(min(176, max(156, viewport_width - 2 * (badge_width + margin + 18))))
        if self._side_flip_button_width != button_width:
            flip_button.configure(width=button_width, height=40)
            self._side_flip_button_width = button_width
        if flip_button.winfo_manager() != "place":
            flip_button.place(relx=0.5, y=17, anchor="n")
        flip_button.lift()

    def _on_side_flip_clicked(self, _event=None):
        if self._side_flip_callback is not None:
            self._side_flip_callback()
        return "break"

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
            return
        self.canvas.delete("overlay")
        self._clear_vertical_line_overlay()
        self._roi_items.clear()
        self._draw_roi()
        self._draw_line_overlays()
        self._draw_active_line()
        self._draw_vertical_line()

    def _clear_vertical_line_overlay(self):
        for item_id in self._vertical_line_items:
            self.canvas.delete(item_id)
        self._vertical_line_items = []

    def _update_vertical_line_overlay(self):
        if self._data is None or self._img_id is None:
            return
        self._clear_vertical_line_overlay()
        self._draw_vertical_line()

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

    # ---------------------------------------------------------- line drawing
    def _clean_line_points(self, points):
        cleaned = []
        for point in points:
            if point is None:
                continue
            x, y = point
            cleaned_point = self._clamp_image_point(x, y)
            if not cleaned or cleaned[-1] != cleaned_point:
                cleaned.append(cleaned_point)
        return cleaned

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

    def _draw_line_overlays(self):
        for item in self._line_overlays:
            # Saved overlays can contain thousands of points; skip vertex dots for speed.
            self._draw_polyline(item["points"], item["color"], item.get("label"), show_vertices=False)

    def _draw_active_line(self):
        if not self._active_line:
            return
        self._draw_polyline(
            self._active_line,
            self._line_color,
            "active",
            preview=self._line_preview,
            show_vertices=True,
        )

    def _draw_vertical_line(self):
        if self._data is None or self._vertical_line_x is None or not self._vertical_line_on:
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
            tags=("vertical_overlay",),
        )
        label_ids = self._draw_text_label(
            x_top + 8,
            y_top + 6,
            text=f"Fovea x={self._vertical_line_x}",
            accent=self._vertical_line_color,
            tags=("vertical_overlay",),
        )
        self._vertical_line_items = [line_id, *label_ids]

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

    def _scaled_label_padding(self):
        return max(0.25, min(1, int(round(10 * self._overlay_zoom_scale()))))

    def _draw_text_label(self, x, y, *, text, accent, tags):
        text_id = self.canvas.create_text(
            x,
            y,
            anchor="nw",
            text=str(text),
            fill=self._label_fill,
            font=self._scaled_label_font(),
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

    def _draw_polyline(self, points, color, label=None, preview=None, show_vertices=True):
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
            radius = self._scaled_vertex_radius()
            for ix, iy in points:
                cx, cy = self._i2c(ix, iy)
                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill=color,
                    outline=color,
                    tags=("overlay",),
                )

        if label:
            lx, ly = self._i2c(*points[0])
            gap = max(6, min(18, int(round(8 * self._overlay_zoom_scale()))))
            label_y = ly - gap
            if label_y < self._img_offset_y:
                label_y = ly + gap
            self._draw_text_label(lx + gap, label_y, text=label, accent=color, tags=("overlay",))

        if preview is not None and len(points) >= 1:
            last_x, last_y = self._i2c(*points[-1])
            self.canvas.create_line(
                last_x,
                last_y,
                preview[0],
                preview[1],
                fill=color,
                width=self._scaled_line_width(),
                dash=(4, 3),
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
    def _on_wheel(self, event):
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            factor = 1.25
        else:
            factor = 1 / 1.25
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
        if self._vertical_line_on:
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

    def _on_drag_motion(self, event):
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
        current_items = self.canvas.find_withtag("current")
        if current_items and "side_flip" in self.canvas.gettags(current_items[0]):
            self._set_canvas_cursor("hand2")
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
        if self._hit_vertical_line(cx):
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
