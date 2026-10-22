"""A flying badge linking a workflow handoff button to its header marker."""

import math
import time
import tkinter as tk

from aidas.ui.theme import COLOR_PAIRS, resolve_color


class StepTransition(tk.Toplevel):
    DURATION_MS = 360
    FRAME_MS = 16
    START_SIZE = 44

    def __init__(self, master):
        super().__init__(master)
        self.withdraw()
        self.overrideredirect(True)
        self.transient(master.winfo_toplevel())
        self.attributes("-topmost", True)
        try:
            self.attributes("-disabled", True)
        except tk.TclError:
            pass
        self._job = None
        self._started = None
        self._on_complete = None
        self._bindings = []
        self._target_dirty = True
        self._target_gone = False
        self._last_size = None
        self._last_font_size = None
        self._last_position = None
        # This undecorated sprite has no dialog content, focus or grab. On
        # Windows its background is transparent, leaving just the round badge.
        transparent = "#010203"
        background = resolve_color(COLOR_PAIRS["surface"])
        try:
            self.attributes("-transparentcolor", transparent)
            background = transparent
        except tk.TclError:
            pass
        self.configure(background=background)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                background=background, takefocus=False)
        self.canvas.pack(fill="both", expand=True)
        self._badge = self.canvas.create_oval(1, 1, 43, 43, outline="",
                                             fill=resolve_color(COLOR_PAIRS["success"]))
        self._number = self.canvas.create_text(22, 22, text="",
            fill=resolve_color(COLOR_PAIRS["on_primary"]), font=("Segoe UI", -16, "bold"))

    def start(self, source, target, number, *, on_complete=None):
        if self._job is not None:
            self.after_cancel(self._job)
        self._clear_bindings()
        self._source = source
        self._target = target
        self._target_dirty = True
        self._target_gone = False
        self._last_size = None
        self._last_font_size = None
        self._last_position = None
        self.geometry(f"{self.START_SIZE}x{self.START_SIZE}")
        owner = self.master.winfo_toplevel()
        self._geometry_watchers = set()
        widget = target
        while widget is not None:
            self._geometry_watchers.add(widget)
            binding = widget.bind("<Configure>", self._invalidate_target, add="+")
            self._bindings.append((widget, "<Configure>", binding))
            if widget is owner:
                break
            widget = widget.master
        binding = target.bind("<Destroy>", self._target_destroyed, add="+")
        self._bindings.append((target, "<Destroy>", binding))
        self._on_complete = on_complete
        self.canvas.itemconfigure(self._number, text=str(number))
        self._started = time.monotonic()
        self._last_frame = self._started
        self._elapsed_ms = 0
        self._render()
        self.deiconify()
        self.lift()

    def _invalidate_target(self, event):
        if event.widget in self._geometry_watchers:
            self._target_dirty = True

    def _target_destroyed(self, event):
        if event.widget is self._target:
            self._target_gone = True

    def _read_target(self):
        if self._target_gone or not self._target.winfo_exists():
            self.destroy()
            return False
        width = self._target.winfo_width()
        height = self._target.winfo_height()
        self._target_bounds = (
            self._target.winfo_rootx() + width / 2,
            self._target.winfo_rooty() + height / 2,
            max(16, width),
        )
        self._target_dirty = False
        return True

    def _render(self):
        self._job = None
        now = time.monotonic()
        # Preserve visible frames if page loading temporarily occupies Tk.
        self._elapsed_ms += min((now - self._last_frame) * 1000, self.FRAME_MS * 2)
        self._last_frame = now
        if self._target_gone:
            self.destroy()
            return
        if self._target_dirty and not self._read_target():
            return
        progress = min(1, self._elapsed_ms / self.DURATION_MS)
        eased = 1 - (1 - progress) ** 3
        end_x, end_y, end_size = self._target_bounds
        start_x, start_y = self._source
        x = start_x + (end_x - start_x) * eased
        y = start_y + (end_y - start_y) * eased - math.sin(math.pi * progress) * 40
        position = (round(x - self.START_SIZE / 2), round(y - self.START_SIZE / 2))
        if position != self._last_position:
            # Move a fixed-size transparent window; resizing a native window
            # each frame adds unnecessary layout and compositor work.
            self.geometry(f"{position[0]:+d}{position[1]:+d}")
            self._last_position = position
        size = round(self.START_SIZE + (end_size - self.START_SIZE) * eased)
        if size != self._last_size:
            inset = (self.START_SIZE - size) / 2 + 1
            self.canvas.coords(self._badge, inset, inset, self.START_SIZE - inset, self.START_SIZE - inset)
            self._last_size = size
        font_size = max(9, size // 3)
        if font_size != self._last_font_size:
            self.canvas.itemconfigure(self._number, font=("Segoe UI", -font_size, "bold"))
            self._last_font_size = font_size
        if progress >= 1:
            callback = self._on_complete
            self.destroy()
            if callback is not None:
                callback()
            return
        # Compensate for rendering time instead of adding it to every frame.
        render_ms = (time.monotonic() - now) * 1000
        self._job = self.after(max(1, self.FRAME_MS - round(render_ms)), self._render)

    def _clear_bindings(self):
        for widget, sequence, binding in self._bindings:
            try:
                widget.unbind(sequence, binding)
            except tk.TclError:
                pass
        self._bindings.clear()

    def destroy(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self._clear_bindings()
        super().destroy()
