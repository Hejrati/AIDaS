"""ImageCanvas widget — displays numpy images with zoom, pan, and interactive ROI selection.

Features:
    - Display 8/16-bit grayscale images (auto-normalised for display)
    - Zoom via mouse-wheel
    - Pan via right-click drag (or scrollbars)
    - Interactive rectangle ROI via left-click drag
    - Pixel coordinate + value tracking callback
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


class ImageCanvas(ttk.Frame):
    """Zoomable image canvas with rectangle ROI overlay."""

    # ------------------------------------------------------------------ init
    def __init__(self, parent, *, on_roi_change=None, on_mouse_move=None):
        super().__init__(parent)

        # Callbacks
        self._cb_roi = on_roi_change      # (x, y, w, h)
        self._cb_mouse = on_mouse_move    # (ix, iy, value)

        # Image state
        self._data = None          # numpy (H, W) original
        self._photo = None         # current PhotoImage
        self._img_id = None        # canvas item id
        self._img_offset_x = 0.0   # displayed image left on canvas coordinates
        self._img_offset_y = 0.0   # displayed image top on canvas coordinates

        # Zoom
        self._zoom = 1.0

        # ROI — image-coordinate ints (x, y, w, h) or None
        self._roi = None
        self._roi_on = False
        self._roi_items = []       # canvas ids for rect + handles

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
        self._data = data
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
        """Normalise to uint8 for display using 0.5-99.5 percentile stretch."""
        if data.dtype == np.uint8:
            return data
        d = data.astype(np.float64)
        lo, hi = np.percentile(d, [0.5, 99.5])
        if hi > lo:
            d = np.clip((d - lo) / (hi - lo) * 255.0, 0, 255)
        else:
            d = np.clip(d, 0, 255)
        return d.astype(np.uint8)

    def _redraw(self):
        self.canvas.delete("all")
        self._roi_items.clear()
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
        pil = Image.fromarray(disp, "L").resize((zw, zh), Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(pil)
        self._img_id = self.canvas.create_image(
            self._img_offset_x,
            self._img_offset_y,
            anchor="nw",
            image=self._photo,
        )
        self.canvas.configure(scrollregion=(0, 0, draw_w, draw_h))
        self._draw_roi()

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
                                         outline="#FFD700", width=2, dash=(6, 3))
        self._roi_items.append(r)
        origin_lbl = self.canvas.create_text(
            cx1 + 8,
            cy1 + 8,
            anchor="nw",
            text=f"({x}, {y})",
            fill="#DA0404",
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
                                              fill=color, outline="black")
            self._roi_items.append(sq)

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
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
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
        if self._roi_on:
            h = self._hit(cx, cy)
            cursors = {"tl": "top_left_corner", "tr": "top_right_corner",
                       "bl": "bottom_left_corner", "br": "bottom_right_corner",
                       "move": "fleur"}
            self.canvas.configure(cursor=cursors.get(h, "crosshair"))
