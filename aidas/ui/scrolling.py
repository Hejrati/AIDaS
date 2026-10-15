"""Scrolling adapters that let Tk finish painting once per event-loop turn."""

from __future__ import annotations

import tkinter as tk
import sys

import customtkinter as ctk


class DeferredScrollbar(ctk.CTkScrollbar):
    """Keep the CTk thumb without flushing Tk's entire idle queue on each move."""

    def __init__(self, *args, **kwargs):
        self._redraw_after_id = None
        super().__init__(*args, **kwargs)

    def _draw(self, no_color_updates=False):
        # CTk's draw ends with update_idletasks(), which recursively performs
        # layout and raster painting. Our renderer already paints after_idle.
        # Suppress only that internal flush, retaining the upstream drawing,
        # hover, scaling and color behavior and ordinary explicit idle calls.
        canvas = self._canvas
        update_idletasks = canvas.update_idletasks
        canvas.update_idletasks = lambda: None
        try:
            return super()._draw(no_color_updates)
        finally:
            canvas.update_idletasks = update_idletasks

    def set(self, start_value, end_value):
        values = (float(start_value), float(end_value))
        if values == (self._start_value, self._end_value):
            return
        self._start_value, self._end_value = values
        if self._redraw_after_id is None:
            self._redraw_after_id = self.after_idle(self._flush_redraw)

    def _flush_redraw(self):
        self._redraw_after_id = None
        self._draw()

    def destroy(self):
        if self._redraw_after_id is not None:
            self.after_cancel(self._redraw_after_id)
            self._redraw_after_id = None
        super().destroy()


class SettingsScrollableFrame(ctk.CTkScrollableFrame):
    """A CTk settings panel with scoped input and dimension-aware layout."""

    def __init__(self, master, **kwargs):
        self._scoped_bindings = []
        self._content_dimensions = None
        self._viewport_dimension = None
        self._wheel_after_id = None
        self._wheel_targets = {}
        super().__init__(master, **kwargs)

        original = self._scrollbar
        self._scrollbar = DeferredScrollbar(
            self._parent_frame,
            orientation=self._orientation,
            command=(
                self._parent_canvas.xview
                if self._orientation == "horizontal"
                else self._parent_canvas.yview
            ),
            width=original.cget("width"),
            height=original.cget("height"),
            corner_radius=original.cget("corner_radius"),
            border_spacing=original.cget("border_spacing"),
            minimum_pixel_length=original.cget("minimum_pixel_length"),
            fg_color=original.cget("fg_color"),
            button_color=original.cget("button_color"),
            button_hover_color=original.cget("button_hover_color"),
        )
        original.destroy()
        axis = "xscrollcommand" if self._orientation == "horizontal" else "yscrollcommand"
        self._parent_canvas.configure(**{axis: self._scrollbar.set})
        self._create_grid()
        self.bind("<Configure>", self._on_content_configure)

    def bind_all(self, sequence=None, func=None, add=None):
        # CTkScrollableFrame registers wheel/Shift handlers through bind_all.
        # Binding to this window instead keeps other windows independent and
        # lets us remove exactly our callbacks when the panel is destroyed.
        window = self.winfo_toplevel()
        binding_id = tk.Misc.bind(window, sequence, func, add)
        if binding_id is not None:
            self._scoped_bindings.append((window, sequence, binding_id))
        return binding_id

    def _on_content_configure(self, event):
        dimensions = (event.width, event.height)
        if dimensions == self._content_dimensions:
            return
        self._content_dimensions = dimensions
        self._parent_canvas.configure(scrollregion=self._parent_canvas.bbox("all"))

    def _fit_frame_dimensions_to_canvas(self, event):
        dimension = event.height if self._orientation == "horizontal" else event.width
        if dimension == self._viewport_dimension:
            return
        self._viewport_dimension = dimension
        super()._fit_frame_dimensions_to_canvas(event)

    def _mouse_wheel_all(self, event):
        """Combine a burst of wheel input into one viewport move per idle turn."""
        if not self._check_if_valid_scroll(event.widget):
            return
        axis = "x" if self._shift_pressed else "y"
        canvas = self._parent_canvas
        view = canvas.xview() if axis == "x" else canvas.yview()
        if view == (0.0, 1.0):
            return
        if sys.platform.startswith("win"):
            units = -int(event.delta / 6)
        elif sys.platform == "darwin":
            units = -event.delta
        else:
            units = -1 if event.num == 4 else 1
        if not units:
            return
        bounds = tuple(float(value) for value in canvas.cget("scrollregion").split())
        extent = bounds[2] - bounds[0] if axis == "x" else bounds[3] - bounds[1]
        increment = float(canvas.cget(f"{axis}scrollincrement"))
        if extent <= 0 or increment <= 0:
            return
        target = self._wheel_targets.get(axis, view[0]) + units * increment / extent
        self._wheel_targets[axis] = max(0.0, min(1.0 - (view[1] - view[0]), target))
        if self._wheel_after_id is None:
            self._wheel_after_id = self.after_idle(self._flush_wheel)

    def _flush_wheel(self):
        self._wheel_after_id = None
        targets, self._wheel_targets = self._wheel_targets, {}
        for axis, target in targets.items():
            if axis == "x":
                self._parent_canvas.xview_moveto(target)
            else:
                self._parent_canvas.yview_moveto(target)

    def destroy(self):
        if self._wheel_after_id is not None:
            self.after_cancel(self._wheel_after_id)
            self._wheel_after_id = None
        self._wheel_targets.clear()
        for window, sequence, binding_id in self._scoped_bindings:
            try:
                tk.Misc.unbind(window, sequence, binding_id)
            except tk.TclError:
                pass
        self._scoped_bindings.clear()
        super().destroy()
