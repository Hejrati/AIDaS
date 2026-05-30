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

import numpy as np
from PIL import Image, ImageTk

# Backwards-compatible resampling constant: newer Pillow exposes
# `Image.Resampling`, older versions use module-level constants like
# `Image.NEAREST`.
try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except Exception:
    RESAMPLE_NEAREST = Image.NEAREST


class ImageCanvas(ttk.Frame):
    """Zoomable image canvas with rectangle ROI, line tracing, and vertical marker."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        parent,
        *,
        on_roi_change=None,
        on_mouse_move=None,
        on_line_change=None,
        on_vertical_line_change=None,
    ):
        super().__init__(parent)

        # Callbacks
        self._cb_roi = on_roi_change      # (x, y, w, h)
        self._cb_mouse = on_mouse_move    # (ix, iy, value)
        self._cb_line = on_line_change    # (points)
        self._cb_vertical_line = on_vertical_line_change  # (x or None)

        # Image state
        self._data = None          # numpy (H, W) original
        self._photo = None         # current PhotoImage
        self._img_id = None        # canvas item id
        self._img_offset_x = 0.0   # displayed image left on canvas coordinates
        self._img_offset_y = 0.0   # displayed image top on canvas coordinates
        self._base_size = None     # (data_id, zoom, canvas_w, canvas_h)

        # Zoom
        self._zoom = 1.0

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

        self._build_widgets()
        self._bind_events()

    # ------------------------------------------------------------- widgets
    def _build_widgets(self):
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        self._vsb = ttk.Scrollbar(self, orient="vertical",   command=self.canvas.yview)
        self._hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
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
        self.canvas.bind("<Configure>",       self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>",   self._on_pan_start)
        self.canvas.bind("<B3-Motion>",       self._on_pan_motion)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<Motion>",          self._on_hover)

    # ---------------------------------------------------------- public API
    def set_image(self, data: np.ndarray | None):
        """Set an image (H, W) numpy array (any dtype)."""
        if data is None:
            self._data = None
        else:
            arr = np.asarray(data)
            if arr.ndim != 2:
                raise ValueError("ImageCanvas expects a 2-D grayscale image.")
            # Analyze data may be big-endian; normalize to native-endian once.
            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
            self._data = np.ascontiguousarray(arr)
        self._active_line.clear()
        self._line_overlays.clear()
        self._line_preview = None
        self._vertical_line_x = None
        self._drag_vertical_line = False
        if data is not None:
            self._auto_zoom()
        self._redraw()

    def get_image(self):
        return self._data

    # ROI
    def enable_roi(self, enabled=True):
        self._roi_on = enabled
        if not enabled:
            self._clear_roi()

    # Line tracing
    def enable_line(self, enabled=True):
        self._line_on = enabled
        if not enabled:
            self.clear_active_line()

    def clear_line_overlays(self):
        self._line_overlays.clear()
        self._redraw_overlays()

    def clear_active_line(self):
        self._active_line.clear()
        self._line_preview = None
        self._redraw_overlays()
        self._emit_line_change()

    # Vertical line marker
    def enable_vertical_line(self, enabled=True):
        self._vertical_line_on = bool(enabled)
        self._drag_vertical_line = False
        self._update_vertical_line_overlay()

    def set_vertical_line_x(self, x):
        if x is None:
            self._vertical_line_x = None
        else:
            self._vertical_line_x = self._clamp_image_point(x, 0)[0]
        self._update_vertical_line_overlay()
        self._emit_vertical_line_change()

    def get_vertical_line_x(self):
        return self._vertical_line_x

    def clear_vertical_line(self):
        self._vertical_line_x = None
        self._drag_vertical_line = False
        self._update_vertical_line_overlay()
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
        self._zoom = max(0.02, min(30.0, z))
        self._redraw()

    def fit_to_window(self):
        self._auto_zoom()
        self._redraw()

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
        self.canvas.delete("all")
        self._roi_items.clear()
        self._img_id = None
        if self._data is None:
            return
        disp = self._to_display(self._data)
        ih, iw = disp.shape[:2]
        zw, zh = max(1, int(iw * self._zoom)), max(1, int(ih * self._zoom))
        self.update_idletasks()
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        draw_w = max(zw, cw)
        draw_h = max(zh, ch)
        self._img_offset_x = (draw_w - zw) / 2.0
        self._img_offset_y = (draw_h - zh) / 2.0
        pil = Image.fromarray(disp, "L").resize((zw, zh), RESAMPLE_NEAREST)
        self._photo = ImageTk.PhotoImage(pil)
        self._img_id = self.canvas.create_image(
            self._img_offset_x,
            self._img_offset_y,
            anchor="nw",
            image=self._photo,
        )
        self.canvas.configure(scrollregion=(0, 0, draw_w, draw_h))
        self._base_size = (id(self._data), self._zoom, self.canvas.winfo_width(), self.canvas.winfo_height())
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
        origin_lbl = self.canvas.create_text(
            cx1 + 8,
            cy1 + 8,
            anchor="nw",
            text=f"({x}, {y})",
            fill="#DA0404",
            tags=("overlay",),
        )
        self._roi_items.append(origin_lbl)
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

    # ---------------------------------------------------------- mouse events
    def _on_wheel(self, event):
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            factor = 1.25
        else:
            factor = 1 / 1.25
        self._zoom = max(0.02, min(30.0, self._zoom * factor))
        self._redraw()

    def _on_canvas_resize(self, _event):
        # Keep image centered whenever the viewport size changes.
        if self._data is not None:
            self._redraw()

    def _on_press(self, event):
        if self._vertical_line_on:
            if self._data is None:
                return
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
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

    def _on_pan_start(self, event):
        if self._data is None:
            return
        self._is_panning = True
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _on_pan_motion(self, event):
        if not self._is_panning:
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_pan_end(self, _event):
        self._is_panning = False

    def _on_hover(self, event):
        if self._data is None:
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
            if self._cb_mouse:
                self._cb_mouse(int(ix), int(iy), val)
        # Cursor shape
        if self._is_panning:
            self.canvas.configure(cursor="fleur")
            return
        if self._vertical_line_on:
            self.canvas.configure(cursor="sb_h_double_arrow")
            return
        if self._line_on:
            self.canvas.configure(cursor="crosshair")
            return
        if self._roi_on:
            h = self._hit(cx, cy)
            cursors = {"tl": "top_left_corner", "tr": "top_right_corner",
                       "bl": "bottom_left_corner", "br": "bottom_right_corner",
                       "move": "fleur"}
            self.canvas.configure(cursor=cursors.get(h, "crosshair"))
