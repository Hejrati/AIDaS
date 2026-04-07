"""Main application window for AIDaS — OCT Image Processing.

Houses a ttk.Notebook with tabs for each processing step.
Currently implements Step 1; remaining steps are placeholders.
"""

# Copyright (c) 2026 Behzad Hejrati
# https://github.com/Hejrati

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys

from aidas import __version__
from aidas.steps.step1_resize_raw import Step1Frame
from aidas.config import Config


class AIDaSApp(tk.Tk):
    """Root application window."""

    def __init__(self):
        super().__init__()

        self.title("AIDaS — Retinal Image Processing")
        self.geometry("1280x820")
        self.minsize(900, 600)

        # Center window inside the usable desktop area, not under the taskbar.
        self.after(0, self._center_window)

        # ── Configuration ──
        self.preferences = Config()

        # ── Theme ──
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        current_theme = self.preferences.get("theme", available_themes[0])
        
        # Apply theme if it exists, otherwise use first available
        if current_theme in available_themes:
            self.style.theme_use(current_theme)
        else:
            self.style.theme_use(available_themes[0])
            self.preferences.set("theme", available_themes[0])

        # ── Menu bar ──
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Browse SDB Directory",
                      command=self._menu_browse_sdb)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy,
                      accelerator="Alt+F4")
        menubar.add_cascade(label="File", menu=file_menu)

        # Theme submenu under Help
        theme_menu = tk.Menu(menubar, tearoff=0)
        for theme in self.style.theme_names():
            theme_menu.add_command(
                label=theme.capitalize(),
                command=lambda t=theme: self._set_theme(t)
            )

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_cascade(label="Theme", menu=theme_menu)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

        self.bind_all("<Alt-F4>", lambda _: self.destroy())

        # ── Notebook (tabs for each Step) ──
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=(5, 1))

        # Step 1
        self.step1 = Step1Frame(self.notebook, preferences=self.preferences)
        self.notebook.add(self.step1, text="  Step 1 — Load, Resize & Crop  ")

        # Placeholder tabs for Steps 3-4
        for i, title in [
            (3, "Rename & Flatten Image"),
            (4, "Analyse ISez"),
        ]:
            f = ttk.Frame(self.notebook)
            ttk.Label(f, text=f"Step {i}: {title}\n\n(Coming soon — under development)",
                      font=("", 12), justify="center").pack(expand=True)
            self.notebook.add(f, text=f"  Step {i}  ")

        # ── Status bar ──
        self.status = ttk.Label(self, text=f"AIDaS v{__version__} — ready",
                                relief="sunken", anchor="w", padding=2)
        self.status.pack(side="bottom", fill="x")

    # ── Menu actions ──
    def _center_window(self):
        """Center the window horizontally at the top of the screen."""
        self.update_idletasks()
        width = self.winfo_width()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        self.geometry(f"+{x}+0")

    def _menu_browse_sdb(self):
        self.notebook.select(0)
        file_path = filedialog.askopenfilename(
            title="Select SDB file (directory will be used)",
            initialdir=self.step1.sdb_dir_var.get() or None,
            filetypes=[("SDB files", "*.sdb"), ("All files", "*.*")],
        )
        if file_path:
            import os
            directory = os.path.dirname(file_path)
            self.step1.set_sdb_directory(directory)
            self.step1.refresh_sdb_list()

    def _set_theme(self, theme_name):
        """Change the application theme and save preference."""
        self.style.theme_use(theme_name)
        self.preferences.set("theme", theme_name)
        self.status.config(text=f"AIDaS v{__version__} — theme changed to '{theme_name}'")


    # ── About dialog ──
    @staticmethod
    def _show_about():
        messagebox.showinfo(
            "About AIDaS",
            "AIDaS — OCT Image Processing\n"
            f"Version {__version__}\n\n"
            "Converts ImageJ / R / MATLAB OCT workflows\n"
            "into a cross-platform Python application.\n\n"
            "Copyright (c) 2026 Behzad Hejrati\n"
            "https://github.com/Hejrati\n\n"
            f"Python {sys.version.split()[0]}",
        )
