"""Main application window for AIDaS — OCT Image Processing.

Houses a ttk.Notebook with tabs for each processing step.
Currently implements Step 1; remaining steps are placeholders.
"""

# Copyright (c) 2026 Behzad Hejrati
# https://github.com/Hejrati

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os

from aidas import __version__
from aidas.steps.step1_resize_raw import Step1Frame
from aidas.steps.step2_annotate import Step2Frame
from aidas.steps.step3_flatten import Step3Frame
from aidas.steps.step4_analyze_isez import Step4Frame

from aidas.config import Config


class AIDaSApp(tk.Tk):
    """Root application window."""

    def __init__(self):
        super().__init__()

        self._set_app_icon()

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
        help_menu.add_command(label="About", command=self._show_about)

        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

        self.bind_all("<Alt-F4>", lambda _: self.destroy())

        # ── Notebook (tabs for each Step) ──
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=(2, 5), pady=(5, 1))

        # Step 1
        self.step1 = Step1Frame(
            self.notebook,
            preferences=self.preferences,
            on_processed_image=self._on_step1_processed_image,
        )
        self.notebook.add(self.step1, text="  Step 1 — Load, Resize & Crop  ")

        # Step 2
        self.step2 = Step2Frame(
            self.notebook,
            preferences=self.preferences,
            source_step=self.step1,
            on_output_folder_changed=self._on_step2_output_folder_changed,
        )
        self.notebook.add(self.step2, text="  Step 2 — Annotate and Segment  ")

        # Step 3
        self.step3 = Step3Frame(self.notebook, preferences=self.preferences)
        self.notebook.add(self.step3, text="  Step 3 — Flatten Retina  ")
        self.step4 = Step4Frame(
            self.notebook,
            preferences=self.preferences,
            source_step=self.step3,
        )
        self.notebook.add(self.step4, text="  Step 4 - Analyze ISez  ")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)


        # ── Status bar ──
        self.status = ttk.Label(self, text=f"AIDaS v{__version__} — ready",
                                relief="sunken", anchor="w", padding=2)
        self.status.pack(side="bottom", fill="x")

    def _on_notebook_tab_changed(self, _event=None):
        selected = self.notebook.nametowidget(self.notebook.select())
        if hasattr(selected, "on_show"):
            selected.on_show()

    @staticmethod
    def _resource_path(relative_path):
        """Resolve resource path for source runs and PyInstaller bundles."""
        base_dir = getattr(sys, "_MEIPASS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        return os.path.join(base_dir, relative_path)

    def _set_app_icon(self):
        """Set window/taskbar icon if available; never fail startup if missing."""
        ico_path = self._resource_path(os.path.join("assets", "aidas.ico"))
        png_path = self._resource_path(os.path.join("assets", "aidas.png"))

        # On Windows, iconbitmap works best with .ico files.
        if os.path.isfile(ico_path):
            # remember the ico path so dialogs/Toplevels can reuse it
            self._icon_ico_path = ico_path
            try:
                self.iconbitmap(ico_path)
                return
            except tk.TclError:
                pass

        # PNG fallback for source runs and bundled apps.
        if os.path.isfile(png_path):
            try:
                image = tk.PhotoImage(file=png_path)
                self.iconphoto(True, image)
                # keep a reference so the image isn't garbage-collected
                self._icon_image_ref = image
            except tk.TclError:
                pass

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

    def _on_step1_processed_image(self, image, source_path):
        """Receive newly cropped Step 1 image and load it into Step 2."""
        if getattr(self, "step2", None) is None:
            return
        self.step2.load_external_image(image, source_path=source_path)

    def _on_step2_output_folder_changed(self, folder):
        """Keep Step 3 pointed at the folder Step 2 will save MARKED files into."""
        if getattr(self, "step3", None) is None:
            return
        self.step3.set_input_folder(folder)

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


def main():
    app = AIDaSApp()
    app.mainloop()


if __name__ == "__main__":
    main()
