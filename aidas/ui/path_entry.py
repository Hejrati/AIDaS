"""Standard editing and submission for native and CustomTkinter path fields."""

from __future__ import annotations

import tkinter as tk

from aidas.ui.directory_completion import DirectoryCompletion
from aidas.ui.theme import COLORS


def bind_path_entry(entry, submit_command=None):
    """Add clipboard editing and optional folder completion/Enter-to-open."""
    target = getattr(entry, "_entry", entry)
    menu = tk.Menu(target, tearoff=False)
    entry._path_context_menu = menu

    def select_all(_event=None):
        if str(target.cget("state")) != "disabled":
            target.selection_range(0, "end")
            target.icursor("end")
        return "break"

    def edit(action):
        target.focus_set()
        target.event_generate(action)

    menu.add_command(label="Cut", accelerator="Ctrl+X", command=lambda: edit("<<Cut>>"))
    menu.add_command(label="Copy", accelerator="Ctrl+C", command=lambda: edit("<<Copy>>"))
    menu.add_command(label="Paste", accelerator="Ctrl+V", command=lambda: edit("<<Paste>>"))
    menu.add_separator()
    menu.add_command(label="Select All", accelerator="Ctrl+A", command=select_all)

    def show_menu(event):
        enabled = str(target.cget("state")) != "disabled"
        editable = str(target.cget("state")) == "normal"
        selected = enabled and target.selection_present()
        try:
            pasteable = bool(target.clipboard_get())
        except tk.TclError:
            pasteable = False
        menu.entryconfigure(0, state="normal" if editable and selected else "disabled")
        menu.entryconfigure(1, state="normal" if selected else "disabled")
        menu.entryconfigure(2, state="normal" if editable and pasteable else "disabled")
        menu.entryconfigure(4, state="normal" if enabled else "disabled")
        menu.configure(
            background=COLORS.menu_surface, foreground=COLORS.menu_text,
            activebackground=COLORS.menu_hover, activeforeground=COLORS.menu_hover_text,
            disabledforeground=COLORS.menu_disabled_text,
        )
        if event.type == tk.EventType.KeyPress:
            x, y = target.winfo_rootx(), target.winfo_rooty() + target.winfo_height()
        else:
            x, y = event.x_root, event.y_root
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()
        return "break"

    target.bind("<Control-a>", select_all, add="+")
    target.bind("<Control-A>", select_all, add="+")
    target.bind("<Button-3>", show_menu, add="+")
    target.bind("<Shift-F10>", show_menu, add="+")
    if submit_command is not None:
        completion = entry._path_completion = DirectoryCompletion(entry)

        def submit(_event=None):
            if str(target.cget("state")) != "disabled" and completion.accept_for_submit():
                submit_command()
            return "break"

        target.bind("<Return>", submit, add="+")
        target.bind("<KP_Enter>", submit, add="+")
    return menu
