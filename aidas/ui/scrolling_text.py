"""Clipped single-line text that reveals long addresses without growing its row."""

from __future__ import annotations

import time

import customtkinter as ctk

from aidas.ui.theme import COLOR_PAIRS, TYPOGRAPHY


class ScrollingText(ctk.CTkFrame):
    """Scroll overflowing text between its ends; stop while hidden or fitting."""

    FRAME_MS = 33
    PAUSE_MS = 1000
    SPEED = 50  # logical pixels per second

    def __init__(self, master, *, text="", textvariable=None, font=None,
                 text_color=None, padding=8, height=28, **kwargs):
        self._text_id = None
        self._text = str(textvariable.get() if textvariable is not None else text)
        self._textvariable = textvariable
        self._trace_id = None
        self._font = font or ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size)
        self._text_color = text_color or COLOR_PAIRS["text"]
        self._padding = padding
        self._offset = 0.0
        self._overflow = 0.0
        self._direction = 1
        self._scroll_job = None
        self._last_tick = None
        kwargs.setdefault("fg_color", COLOR_PAIRS["surface_subtle"])
        super().__init__(master, width=1, height=height, corner_radius=0, border_width=0, **kwargs)
        self._text_id = self._canvas.create_text(0, 0, anchor="w", text=self._text)
        self._font.add_size_configure_callback(self._restart)
        if textvariable is not None:
            self._trace_id = textvariable.trace_add("write", self._variable_changed)
        self.bind("<Map>", self._restart, add="+")
        self.bind("<Unmap>", self._stop, add="+")
        self._draw()

    def configure(self, **kwargs):
        if "text" in kwargs:
            self.set_text(kwargs.pop("text"))
        super().configure(**kwargs)

    def cget(self, name):
        if name == "text":
            return self._text
        return super().cget(name)

    def set_text(self, text):
        text = str(text or "")
        if text != self._text:
            self._text = text
            self._canvas.itemconfigure(self._text_id, text=text)
            self._restart()

    def _variable_changed(self, *_args):
        self.set_text(self._textvariable.get())

    def _stop(self, _event=None):
        if self._scroll_job is not None:
            self.after_cancel(self._scroll_job)
            self._scroll_job = None
        self._last_tick = None

    def _restart(self, _event=None):
        self._stop()
        self._offset = 0.0
        self._direction = 1
        self._draw()

    def _draw(self, no_color_updates=False):
        super()._draw(no_color_updates)
        if self._text_id is None:
            return
        self._canvas.itemconfigure(
            self._text_id, font=self._apply_font_scaling(self._font),
            fill=self._apply_appearance_mode(self._text_color),
        )
        bounds = self._canvas.bbox(self._text_id)
        text_width = bounds[2] - bounds[0] if bounds else 0
        scale = self._get_widget_scaling()
        available = max(0, (self._current_width - 2 * self._padding) * scale)
        overflow = max(0.0, text_width - available)
        if abs(overflow - self._overflow) > 0.5:
            self._stop()
            self._offset = 0.0
            self._direction = 1
        self._overflow = overflow
        self._position_text()
        if not overflow:
            self._stop()
        elif self.winfo_ismapped() and self._scroll_job is None:
            self._scroll_job = self.after(self.PAUSE_MS, self._tick)

    def _position_text(self):
        scale = self._get_widget_scaling()
        self._canvas.coords(self._text_id, self._padding * scale - self._offset,
                            self._current_height * scale / 2)

    def _tick(self):
        self._scroll_job = None
        if not self.winfo_ismapped() or self._overflow <= 0:
            self._last_tick = None
            return
        now = time.monotonic()
        elapsed = min(0.1, now - self._last_tick) if self._last_tick is not None else self.FRAME_MS / 1000
        self._last_tick = now
        self._offset = max(0.0, min(self._overflow,
            self._offset + self._direction * self.SPEED * self._get_widget_scaling() * elapsed))
        self._position_text()
        if self._offset in (0.0, self._overflow):
            self._direction *= -1
            self._last_tick = None
            delay = self.PAUSE_MS
        else:
            delay = self.FRAME_MS
        self._scroll_job = self.after(delay, self._tick)

    def destroy(self):
        self._stop()
        self._font.remove_size_configure_callback(self._restart)
        if self._trace_id is not None:
            self._textvariable.trace_remove("write", self._trace_id)
        super().destroy()
