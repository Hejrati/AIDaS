"""Brief visual movement for required batch prompts."""

from __future__ import annotations

import math
import tkinter as tk


class PromptCue:
    """Move a prompt twice, then leave it still and release its timer jobs.

    ``move_offset`` receives a horizontal offset in logical pixels. The caller
    owns layout and clipping.
    """

    DURATION_MS = 600
    FRAME_MS = 30
    AMPLITUDE = 4

    def __init__(self, owner):
        self._owner = owner
        self._motion_job = None
        self._move_offset = None
        self._frame = 0
        self._generation = 0
        self._destroyed = False
        # CTkFrame.bind redirects to its canvas. Bind to the actual frame so
        # destruction of a child cannot prematurely end the current cue.
        self._destroy_binding = tk.Misc.bind(
            owner, "<Destroy>", self._on_owner_destroy, add="+",
        )

    def start(self, move_offset):
        """Replace the current visual cue."""
        self.stop()
        if self._destroyed:
            return
        self._move_offset = move_offset
        self._frame = 0
        generation = self._generation
        if not self._move_to(0):
            self.stop()
            return
        try:
            self._motion_job = self._owner.after(
                self.FRAME_MS, lambda: self._tick(generation),
            )
        except tk.TclError:
            self.stop()

    def stop(self):
        """Cancel pending motion and restore the prompt's resting position."""
        self._generation += 1
        job, self._motion_job = self._motion_job, None
        if job is not None:
            try:
                self._owner.after_cancel(job)
            except tk.TclError:
                pass
        move_offset, self._move_offset = self._move_offset, None
        if move_offset is not None:
            try:
                move_offset(0)
            except tk.TclError:
                pass

    def destroy(self):
        """Release the cue while leaving its owner available to other callers."""
        self.stop()
        if self._destroy_binding is not None:
            try:
                tk.Misc.unbind(self._owner, "<Destroy>", self._destroy_binding)
            except tk.TclError:
                pass
            self._destroy_binding = None
        self._destroyed = True

    def _move_to(self, offset):
        try:
            self._move_offset(offset)
        except tk.TclError:
            return False
        return True

    def _tick(self, generation):
        if generation != self._generation or self._destroyed:
            return
        self._motion_job = None
        self._frame += 1
        frame_count = self.DURATION_MS // self.FRAME_MS
        offset = round(self.AMPLITUDE * math.sin(4 * math.pi * self._frame / frame_count))
        if not self._move_to(offset):
            self.stop()
            return
        if self._frame >= frame_count:
            self._move_offset = None
            return
        try:
            self._motion_job = self._owner.after(
                self.FRAME_MS, lambda: self._tick(generation),
            )
        except tk.TclError:
            self.stop()

    def _on_owner_destroy(self, event):
        if event.widget is self._owner:
            self.stop()
            self._destroy_binding = None
            self._destroyed = True
