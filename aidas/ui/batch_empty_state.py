"""Actionable empty-folder notice for workflow batch selectors."""

from tkinter import ttk

from aidas.ui.components import AppButton
from aidas.ui.theme import TYPOGRAPHY


class NoValidFilesNotice(ttk.Frame):
    def __init__(self, master, *, description, on_select_folder):
        super().__init__(master, padding=24)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        content = ttk.Frame(self)
        content.grid(row=1, column=0)
        ttk.Label(
            content, text="No valid files found",
            font=(TYPOGRAPHY.family, TYPOGRAPHY.subtitle_size, "bold"),
        ).pack(pady=(0, 10))
        message = ttk.Label(content, text=description, justify="center", anchor="center", wraplength=500)
        message.pack(fill="x")
        wraplength = 500

        def resize_message(event):
            nonlocal wraplength
            width = max(180, min(500, event.width - 64))
            if width != wraplength:
                message.configure(wraplength=width)
                wraplength = width

        self.bind("<Configure>", resize_message, add="+")
        self.select_folder_button = AppButton(
            content, text="Select another folder", variant="primary", width=180,
            command=on_select_folder,
        )
        self.select_folder_button.pack(pady=(18, 0))
