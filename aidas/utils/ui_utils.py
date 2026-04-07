"""UI helpers shared across AIDaS Tkinter views."""

import tkinter as tk
from tkinter import ttk


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
