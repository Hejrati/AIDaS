"""Folder suggestions that leave typing and the owning dialog in control."""

from __future__ import annotations

import queue
import threading
import tkinter as tk

from aidas.core.display import monitor_work_area_for_window
from aidas.ui.scrolling import DeferredScrollbar
from aidas.ui.theme import COLORS
from aidas.utils.path_completion import directory_completions


class DirectoryCompletion:
    DEBOUNCE_MS = 120
    POLL_MS = 25
    VISIBLE_ROWS = 9

    def __init__(self, entry):
        self.entry = entry
        self.target = getattr(entry, "_entry", entry)
        self.owner = self.target.winfo_toplevel()
        self._anchor_widgets = set()
        ancestor = self.target
        while ancestor is not None:
            self._anchor_widgets.add(ancestor)
            if ancestor is self.owner:
                break
            ancestor = ancestor.master
        self._anchor_geometry = None
        self.popup = None
        self.listbox = None
        self.scrollbar = None
        self.suggestions = []
        self._results = queue.Queue()
        self._generation = 0
        self._pending = None
        self._pending_action = None
        self._query_text = None
        self._observed_text = self.target.get()
        self._mutating = False
        self._disposed = False
        self._dirty = False
        self._delay_id = None
        self._poll_id = None
        self._focus_id = None
        self._hover_id = None
        self._hovered = False
        self._scrollable = False
        self._outside_id = None
        self._bindings = []
        self._variable = str(self.target.cget("textvariable"))
        self._trace_command = None
        if self._variable:
            self._trace_command = self.target.register(self._text_changed)
            self.target.tk.call("trace", "add", "variable", self._variable, "write", self._trace_command)
        self._bind(self.target, "<KeyRelease>", self._key_released)
        for event in ("<<Paste>>", "<<Cut>>", "<<Clear>>", "<<Undo>>", "<<Redo>>"):
            self._bind(self.target, event, self._after_edit)
        self._bind(self.target, "<Down>", lambda _event: self._arrow(1))
        self._bind(self.target, "<Up>", lambda _event: self._arrow(-1))
        self._bind(self.target, "<Tab>", lambda _event: self._tab(1))
        self._bind(self.target, "<Shift-Tab>", lambda _event: self._tab(-1))
        self._bind(self.target, "<ISO_Left_Tab>", lambda _event: self._tab(-1))
        self._bind(self.target, "<Escape>", self._escape)
        self._bind(self.target, "<FocusOut>", self._focus_out)
        self._bind(self.target, "<Destroy>", self._target_destroyed)
        self._bind(self.owner, "<Configure>", self._owner_changed)
        self._bind(self.owner, "<Unmap>", self._owner_changed)

    def _bind(self, widget, event, callback):
        binding = widget.bind(event, callback, add="+")
        self._bindings.append((widget, event, binding))

    def _cancel(self, name):
        job = getattr(self, name)
        setattr(self, name, None)
        if job is not None:
            try:
                self.owner.after_cancel(job)
            except tk.TclError:
                pass

    def _editable(self):
        return not self._disposed and str(self.target.cget("state")) == "normal"

    def _has_focus(self):
        return self.target.focus_get() is self.target

    def _text_changed(self, *trace_args):
        if self._mutating or self._disposed:
            return
        # A write trace can run before the Entry updates, inside a Tk procedure.
        # Entry textvariables live in global scope, not that procedure's locals.
        text = str(self.target.tk.globalgetvar(self._variable)) if trace_args and self._variable else self.target.get()
        if text == self._observed_text:
            return
        self._observed_text = text
        self.hide()
        self._dirty = self._editable() and self._has_focus()
        if self._dirty and text.strip():
            self._delay_id = self.owner.after(self.DEBOUNCE_MS, self.refresh_now)

    def _key_released(self, event):
        if event.keysym not in ("Tab", "ISO_Left_Tab", "Return", "KP_Enter", "Up", "Down", "Escape"):
            self._text_changed()

    def _after_edit(self, _event):
        # Tk's Entry class performs the actual edit after widget bindings.
        self._cancel("_delay_id")
        self._delay_id = self.owner.after_idle(self._after_edit_idle)

    def _after_edit_idle(self):
        self._delay_id = None
        self._text_changed()

    def refresh_now(self):
        """Request suggestions without waiting for the typing debounce."""
        self.hide()
        if not self._editable() or not self._has_focus():
            return
        text = self.target.get()
        if not text.strip():
            return
        self._observed_text = text
        self._dirty = True
        self._query_text = text
        generation = self._generation
        self._pending = generation
        results = self._results

        def lookup():
            try:
                matches = directory_completions(text)
            except Exception:
                matches = []
            results.put((generation, text, matches))

        threading.Thread(target=lookup, daemon=True, name="aidas-folder-completion").start()
        self._poll_id = self.owner.after(self.POLL_MS, self._poll_results)

    def _poll_results(self):
        self._poll_id = None
        if self._disposed:
            return
        while True:
            try:
                generation, text, matches = self._results.get_nowait()
            except queue.Empty:
                break
            if generation != self._generation or text != self.target.get():
                continue
            self._pending = None
            if not self._editable() or not self._has_focus():
                self.hide()
                return
            self.suggestions = matches
            action = self._pending_action
            self._pending_action = None
            if matches:
                self._show()
                if action is not None:
                    kind, direction = action
                    if kind == "complete":
                        self._complete_next(direction)
                    else:
                        self._move_selection(direction)
            elif action is not None and action[0] == "complete":
                self._move_focus(action[1])
        if self._pending is not None:
            self._poll_id = self.owner.after(self.POLL_MS, self._poll_results)

    def _ensure_popup(self):
        if self.popup is not None:
            return
        popup = self.popup = tk.Toplevel(self.owner)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.transient(self.owner)
        popup.grid_rowconfigure(0, weight=1)
        popup.grid_columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(
            popup, activestyle="none", selectmode="browse", exportselection=False,
            takefocus=False, relief="flat", borderwidth=0, highlightthickness=0,
            selectborderwidth=0, font=self.target.cget("font"), cursor="hand2",
        )
        self.listbox.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=5)
        self.scrollbar = DeferredScrollbar(
            popup, orientation="vertical", command=self.listbox.yview,
            width=8, corner_radius=4, border_spacing=0, minimum_pixel_length=20,
        )
        self.scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 2), pady=5)
        self.scrollbar.grid_remove()
        self.listbox.configure(yscrollcommand=self._set_scrollbar)
        self.listbox.bind("<ButtonPress-1>", self._clicked)
        popup.bind("<Enter>", self._hover_enter)
        popup.bind("<Leave>", self._hover_leave)
        # Widget/class bindings scroll the list or thumb first. Stop at this
        # toplevel before app-wide wheel handlers also move the sidebar below.
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            popup.bind(event, lambda _event: "break")
        self.listbox._apply_aidas_theme = self._apply_theme
        self._apply_theme()

    def _apply_theme(self):
        if self.popup is None or self._disposed:
            return
        self.popup.configure(
            background=COLORS.menu_surface, highlightthickness=1,
            highlightbackground=COLORS.menu_border,
        )
        self.listbox.configure(
            background=COLORS.menu_surface, foreground=COLORS.menu_text,
            selectbackground=COLORS.menu_hover, selectforeground=COLORS.menu_hover_text,
        )
        self.scrollbar.configure(
            fg_color=COLORS.menu_surface, button_color=COLORS.border_strong,
            button_hover_color=COLORS.muted_text,
        )
        self.popup.grid_columnconfigure(1, minsize=self.scrollbar.winfo_reqwidth() + 2)

    def _set_scrollbar(self, first, last):
        self.scrollbar.set(first, last)
        self._scrollable = float(first) > 0 or float(last) < 1
        self._update_scrollbar_visibility()

    def _update_scrollbar_visibility(self):
        if self._hovered and self._scrollable and self._visible():
            self.scrollbar.grid()
        else:
            self.scrollbar.grid_remove()

    def _hover_enter(self, _event):
        self._cancel("_hover_id")
        self._hovered = self._visible()
        self._update_scrollbar_visibility()

    def _hover_leave(self, _event):
        self._cancel("_hover_id")
        self._hover_id = self.owner.after(40, self._check_hover)

    def _check_hover(self):
        self._hover_id = None
        x, y = self.popup.winfo_pointerxy()
        left, top = self.popup.winfo_rootx(), self.popup.winfo_rooty()
        self._hovered = (self._visible() and left <= x < left + self.popup.winfo_width()
                         and top <= y < top + self.popup.winfo_height())
        self._update_scrollbar_visibility()

    def _show(self):
        self._ensure_popup()
        self._apply_theme()
        self.listbox.delete(0, "end")
        self.listbox.insert("end", *(name for name, _path in self.suggestions))
        rows = min(len(self.suggestions), self.VISIBLE_ROWS)
        self.listbox.configure(height=rows)
        self.listbox.selection_clear(0, "end")
        x, anchor_y, width, anchor_height = self._anchor_geometry = self._entry_geometry()
        height = self.listbox.winfo_reqheight() + 12
        below = anchor_y + anchor_height
        bounds = monitor_work_area_for_window(self.owner)
        if bounds is None:
            left, top = self.owner.winfo_vrootx(), self.owner.winfo_vrooty()
            bounds = (left, top, left + self.owner.winfo_vrootwidth(), top + self.owner.winfo_vrootheight())
        left, top, right, bottom = bounds
        width = min(width, right - left)
        height = min(height, bottom - top)
        x = max(left, min(x, right - width))
        y = below if below + height <= bottom else anchor_y - height
        y = max(top, min(y, bottom - height))
        # The explicit '+' preserves absolute negative coordinates on left/up monitors.
        self.popup.geometry(f"{width}x{height}+{x}+{y}")
        self.popup.deiconify()
        self.popup.lift()
        if self._outside_id is None:
            self._outside_id = self.owner.bind("<ButtonPress-1>", self._outside_clicked, add="+")

    def _visible(self):
        return self.popup is not None and bool(self.popup.winfo_ismapped())

    def _move_selection(self, direction):
        selected = self.listbox.curselection()
        index = ((selected[0] + direction) % len(self.suggestions) if selected
                 else 0 if direction > 0 else len(self.suggestions) - 1)
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.listbox.see(index)
        return index

    def _fill(self, index):
        self._mutating = True
        try:
            self.target.delete(0, "end")
            self.target.insert(0, self.suggestions[index][1])
            self.target.selection_clear()
            self.target.icursor("end")
            self._observed_text = self.target.get()
        finally:
            self._mutating = False

    def _complete_next(self, direction):
        selected = self.listbox.curselection()
        # A highlighted arrow-key choice is completed before cycling further.
        index = (selected[0] if selected and self.target.get() != self.suggestions[selected[0]][1]
                 else self._move_selection(direction))
        self._fill(index)

    def _arrow(self, direction):
        if not self._editable():
            self.hide()
            return
        if self._visible() and self.suggestions:
            self._move_selection(direction)
            return "break"
        if self.target.get().strip():
            self.refresh_now()
            self._pending_action = ("highlight", direction)
            return "break"

    def _tab(self, direction):
        if not self._editable():
            self.hide()
            return
        if self._visible() and self.suggestions:
            self._complete_next(direction)
            return "break"
        if self._dirty and (self._pending is not None or self._delay_id is not None):
            self.refresh_now()
            self._pending_action = ("complete", direction)
            return "break"

    def _move_focus(self, direction):
        target = self.target.tk_focusNext() if direction > 0 else self.target.tk_focusPrev()
        if target is not None:
            target.focus_set()

    def accept_for_submit(self):
        """Confirm an open dropdown; allow loading only when it was closed."""
        choosing = self._visible()
        if choosing and self.suggestions:
            selected = self.listbox.curselection()
            if selected:
                self._fill(selected[0])
        self.hide()
        return not choosing

    def _clicked(self, event):
        if not self._editable():
            self.hide()
        elif self.suggestions:
            self._fill(self.listbox.nearest(event.y))
            self.hide()
            self.target.focus_set()
        return "break"

    def _escape(self, _event):
        active = self._visible() or self._pending is not None or self._delay_id is not None
        self.hide()
        return "break" if active else None

    def _outside_clicked(self, event):
        if event.widget not in (self.entry, self.target):
            self.hide()

    def _focus_out(self, _event):
        self._cancel("_focus_id")
        self._focus_id = self.owner.after(30, self._check_focus)

    def _check_focus(self):
        self._focus_id = None
        focus = self.target.focus_get()
        if focus is self.target:
            return
        if self.popup is not None and focus is not None and focus.winfo_toplevel() is self.popup:
            return
        self.hide()

    def _owner_changed(self, event):
        if event.widget is self.owner:
            self.hide()
        elif event.widget in self._anchor_widgets and self._visible():
            if not self.entry.winfo_viewable() or self._entry_geometry() != self._anchor_geometry:
                self.hide()

    def _entry_geometry(self):
        return (self.entry.winfo_rootx(), self.entry.winfo_rooty(),
                self.entry.winfo_width(), self.entry.winfo_height())

    def hide(self):
        self._generation += 1
        self._pending = None
        self._pending_action = None
        self._cancel("_delay_id")
        self._cancel("_poll_id")
        self._cancel("_hover_id")
        self._hovered = False
        if self.popup is not None:
            self.popup.withdraw()
            self.scrollbar.grid_remove()
        self.suggestions = []
        if self._outside_id is not None:
            self.owner.unbind("<ButtonPress-1>", self._outside_id)
            self._outside_id = None

    def _target_destroyed(self, event):
        if event.widget is self.target:
            self.destroy()

    def destroy(self):
        if self._disposed:
            return
        self.hide()
        self._cancel("_focus_id")
        self._disposed = True
        if self._trace_command is not None:
            self.target.tk.call("trace", "remove", "variable", self._variable, "write", self._trace_command)
            self.target.deletecommand(self._trace_command)
        for widget, event, binding in self._bindings:
            try:
                widget.unbind(event, binding)
            except tk.TclError:
                pass
        self._bindings.clear()
        if self.popup is not None:
            self.popup.destroy()
