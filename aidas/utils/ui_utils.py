"""UI helpers shared across AIDaS Tkinter views."""

import tkinter as tk
from tkinter import ttk


def apply_app_icon_to(window):
    """Apply the application's icon to `window` when available.

    The function looks up the top-level/root window and, if it has either
    a stored PhotoImage reference (`_icon_image_ref`) or an ICO path
    (`_icon_ico_path`), applies that icon to the provided `window`.

    This is intentionally forgiving: it ignores any exceptions so callers
    needn't wrap calls in try/except.
    """
    try:
        root = window
        while getattr(root, "master", None):
            root = root.master
    except Exception:
        try:
            root = window.winfo_toplevel()
        except Exception:
            return

    img = getattr(root, "_icon_image_ref", None)
    if img:
        try:
            window.iconphoto(True, img)
            return
        except Exception:
            pass

    ico = getattr(root, "_icon_ico_path", None)
    if ico:
        try:
            window.iconbitmap(ico)
        except Exception:
            pass


class HoverToolTip:
    """Small hover tooltip for Tk and ttk widgets."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.widget.bind("<Enter>", self._show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self.tipwindow is not None:
            return

        x = self.widget.winfo_pointerx() + 12
        y = self.widget.winfo_pointery() + 12
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        # Apply shared helper so tooltip windows use the app icon when shown
        try:
            apply_app_icon_to(tw)
        except Exception:
            pass
        ttk.Label(
            tw,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            padding=(6, 3),
        ).pack()
        self.tipwindow = tw

    def _hide(self, _event=None):
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None
