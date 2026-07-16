"""Main Tkinter application and branded information windows for AIDaS."""

# Copyright (c) 2026 Behzad Hejrati
# https://github.com/Hejrati

from __future__ import annotations

import ctypes
import os
import sys
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, ttk

from aidas import __version__
from aidas.config import Config
from aidas.single_instance import SingleInstanceGuard
from aidas.utils.ui_utils import apply_app_icon_to, build_app_menu, resource_path


APP_TITLE = "AIDaS"
APP_SUBTITLE = "OCT image processing"
LAB_ACRONYM = "MVPRL"
LAB_NAME = "Machine Vision and Pattern Recognition Lab"
LAB_URL = "https://mvprl.cs.wayne.edu"
LAB_URL_TEXT = "mvprl.cs.wayne.edu"
UNIVERSITY_NAME = "Wayne State University"
COPYRIGHT_NOTICE = (
    "Copyright (c) 2026 Machine Vision and Pattern Recognition Lab, "
    "Wayne State University. All rights reserved."
)
LAB_DESCRIPTION = (
    "Established in 2002, Machine Vision and Pattern Recognition Lab aims at\n"
    "performing research in Deep Learning, Data Mining and Multimedia Content\n"
    "Analysis."
)

WINDOW_BG = "#f7f8fa"
BRAND_NAVY = "#103b64"
BRAND_RED = "#c0002b"
BODY_TEXT = "#07111c"
SPLASH_MINIMUM_MS = 2200


def _bootloader_splash_is_alive() -> bool:
    """Return whether this frozen process owns a live PyInstaller splash."""
    try:
        import pyi_splash

        return bool(pyi_splash.is_alive())
    except (ImportError, OSError, RuntimeError):
        return False


def _close_bootloader_splash() -> None:
    """Close PyInstaller's early splash without affecting source runs."""
    try:
        import pyi_splash

        if pyi_splash.is_alive():
            pyi_splash.close()
    except (ImportError, OSError, RuntimeError):
        # Source runs have no pyi_splash module, and a splash failure must not
        # prevent the application from starting or showing a startup notice.
        pass


def _center_geometry(window: tk.Misc, width: int, height: int, *, parent=None) -> str:
    """Return geometry that centers a window on its parent or the screen."""
    window.update_idletasks()
    if parent is not None and parent.winfo_viewable():
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - width) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 2)
    else:
        x = max(0, (window.winfo_screenwidth() - width) // 2)
        y = max(0, (window.winfo_screenheight() - height) // 2)
    return f"{width}x{height}+{x}+{y}"


class SplashWindow(tk.Toplevel):
    """Borderless startup window styled after the original AIDaS splash."""

    WIDTH = 572
    HEIGHT = 816

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.overrideredirect(True)
        self.configure(bg=BRAND_NAVY)
        self.geometry(_center_geometry(self, self.WIDTH, self.HEIGHT))

        panel = tk.Frame(self, bg=WINDOW_BG, bd=0, highlightthickness=0)
        panel.pack(fill="both", expand=True, padx=1, pady=1)

        logo_path = resource_path(os.path.join("assets", "aidas.png"))
        self.logo_image = tk.PhotoImage(file=logo_path)
        tk.Label(
            panel,
            image=self.logo_image,
            bg=WINDOW_BG,
            bd=0,
            highlightthickness=0,
        ).pack(pady=(24, 0))

        tk.Label(
            panel,
            text=APP_TITLE,
            font=("Segoe UI", 28, "bold"),
            fg=BRAND_NAVY,
            bg=WINDOW_BG,
        ).pack(pady=(14, 0))
        tk.Label(
            panel,
            text=APP_SUBTITLE,
            font=("Segoe UI", 11),
            fg=BRAND_NAVY,
            bg=WINDOW_BG,
        ).pack(pady=(1, 0))
        tk.Label(
            panel,
            text=LAB_ACRONYM,
            font=("Segoe UI", 19, "bold"),
            fg=BRAND_RED,
            bg=WINDOW_BG,
        ).pack(pady=(20, 0))
        tk.Label(
            panel,
            text=LAB_NAME,
            font=("Segoe UI", 9, "bold"),
            fg=BODY_TEXT,
            bg=WINDOW_BG,
        ).pack(pady=(1, 0))
        tk.Label(
            panel,
            text=COPYRIGHT_NOTICE,
            font=("Segoe UI", 8),
            fg=BODY_TEXT,
            bg=WINDOW_BG,
            justify="center",
            wraplength=530,
        ).pack(pady=(20, 0))
        tk.Label(
            panel,
            text=UNIVERSITY_NAME,
            font=("Segoe UI", 8),
            fg=BODY_TEXT,
            bg=WINDOW_BG,
        ).pack(pady=(16, 0))

        self.lift()


class AboutDialog(tk.Toplevel):
    """Branded, modal About window opened from the Help menu."""

    WIDTH = 452
    HEIGHT = 380

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.withdraw()
        self.title("About AIDaS")
        self.configure(bg="#f2f2f2")
        self.resizable(False, False)
        self.transient(parent)
        apply_app_icon_to(self)

        tk.Label(
            self,
            text=APP_TITLE,
            font=("Segoe UI", 18, "bold"),
            fg="#000000",
            bg="#f2f2f2",
        ).pack(pady=(22, 0))
        tk.Label(
            self,
            text=f"{APP_SUBTITLE} - Version {__version__}",
            font=("Segoe UI", 9),
            fg="#000000",
            bg="#f2f2f2",
        ).pack(pady=(2, 0))
        tk.Label(
            self,
            text=LAB_ACRONYM,
            font=("Segoe UI", 11, "bold"),
            fg=BRAND_RED,
            bg="#f2f2f2",
        ).pack(pady=(16, 0))
        tk.Label(
            self,
            text=LAB_NAME,
            font=("Segoe UI", 9),
            fg="#000000",
            bg="#f2f2f2",
        ).pack(pady=(5, 0))

        link = tk.Label(
            self,
            text=LAB_URL_TEXT,
            font=("Segoe UI", 9, "underline"),
            fg="#0066cc",
            bg="#f2f2f2",
            cursor="hand2",
        )
        link.pack(pady=(11, 0))
        link.bind("<Button-1>", lambda _event: webbrowser.open_new_tab(LAB_URL))

        tk.Label(
            self,
            text=LAB_DESCRIPTION,
            font=("Segoe UI", 8),
            fg="#000000",
            bg="#f2f2f2",
            justify="center",
        ).pack(pady=(13, 0))
        tk.Label(
            self,
            text=UNIVERSITY_NAME,
            font=("Segoe UI", 8),
            fg="#000000",
            bg="#f2f2f2",
        ).pack(pady=(10, 0))
        tk.Label(
            self,
            text=COPYRIGHT_NOTICE,
            font=("Segoe UI", 8),
            fg="#000000",
            bg="#f2f2f2",
            justify="center",
            wraplength=410,
        ).pack(pady=(10, 0))
        tk.Label(
            self,
            text=f"Python {sys.version.split()[0]}",
            font=("Segoe UI", 8),
            fg="#000000",
            bg="#f2f2f2",
        ).pack(pady=(0, 0))

        ttk.Button(self, text="OK", width=10, command=self._close).pack(pady=(9, 0))

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self.bind("<Return>", lambda _event: self._close())
        self.geometry(_center_geometry(self, self.WIDTH, self.HEIGHT, parent=parent))
        self.deiconify()
        self.grab_set()
        self.focus_force()

    def _close(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()


class AIDaSApp(tk.Tk):
    """Root application window."""

    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.title("AIDaS — Retinal Image Processing")
        self.geometry("1280x820")
        self.minsize(900, 600)
        self._about_dialog = None
        self._set_app_icon()

        self._splash_started_at = time.monotonic()
        self._bootloader_splash_active = _bootloader_splash_is_alive()
        self._splash = None
        if not self._bootloader_splash_active:
            self._splash = SplashWindow(self)
            # Source runs have no bootloader splash, so paint the Tk fallback
            # before importing the processing views and models.
            self.update_idletasks()
            self.update()

        try:
            self._build_application()
        except Exception:
            try:
                splash = self._splash
                if splash is not None and splash.winfo_exists():
                    splash.destroy()
                _close_bootloader_splash()
            finally:
                self.destroy()
            raise

        elapsed_ms = int((time.monotonic() - self._splash_started_at) * 1000)
        delay_ms = 0 if self._bootloader_splash_active else max(0, SPLASH_MINIMUM_MS - elapsed_ms)
        self.after(delay_ms, self._finish_startup)

    def _build_application(self) -> None:
        """Build the main UI while the splash remains visible."""
        # Keep these imports behind the splash: the processing modules load
        # scientific and imaging libraries that can take noticeable time.
        from aidas.steps.step1_resize_raw import Step1Frame
        from aidas.steps.step2_annotate import Step2Frame
        from aidas.steps.step3_flatten import Step3Frame
        from aidas.steps.step4_analyze_isez import Step4Frame

        self.preferences = Config()
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        default_theme = "xpnative" if "xpnative" in available_themes else available_themes[0]
        current_theme = self.preferences.get("theme", default_theme)

        if current_theme in available_themes:
            self.style.theme_use(current_theme)
        else:
            self.style.theme_use(available_themes[0])
            self.preferences.set("theme", available_themes[0])

        self._build_menu()
        self.bind_all("<Alt-F4>", lambda _event: self.destroy())

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=(2, 5), pady=(5, 1))

        self.step1 = Step1Frame(
            self.notebook,
            preferences=self.preferences,
            on_processed_image=self._on_step1_processed_image,
        )
        self.notebook.add(self.step1, text="  Step 1 — Load, Resize & Crop  ")

        self.step2 = Step2Frame(
            self.notebook,
            preferences=self.preferences,
            source_step=self.step1,
            on_output_folder_changed=self._on_step2_output_folder_changed,
        )
        self.notebook.add(self.step2, text="  Step 2 — Annotate and Segment  ")

        self.step3 = Step3Frame(self.notebook, preferences=self.preferences)
        self.notebook.add(self.step3, text="  Step 3 — Flatten Retina  ")

        self.step4 = Step4Frame(self.notebook, preferences=self.preferences)
        self.notebook.add(self.step4, text="  Step 4 — Analyze ISEZ  ")

        self.status = ttk.Label(
            self,
            text=f"AIDaS v{__version__} — ready",
            relief="sunken",
            anchor="w",
            padding=2,
        )
        self.status.pack(side="bottom", fill="x")

    def _finish_startup(self) -> None:
        """Close the splash and reveal the fully initialized main window."""
        splash = getattr(self, "_splash", None)
        if splash is not None and splash.winfo_exists():
            splash.destroy()
        self._splash = None
        self._center_window()
        self.deiconify()
        self.lift()
        # Paint the ready main window underneath the always-on-top bootloader
        # splash, then close that splash for a direct, gap-free handoff.
        self.update_idletasks()
        self.update()
        _close_bootloader_splash()
        self.focus_force()

    @staticmethod
    def _resource_path(relative_path: str) -> str:
        """Resolve a resource path for source runs and PyInstaller bundles."""
        return resource_path(relative_path)

    def _build_menu(self) -> None:
        self.menubar = build_app_menu(
            self,
            themes=self.style.theme_names(),
            current_theme=self.style.theme_use(),
            set_theme_command=self._set_theme,
            browse_sdb_command=self._menu_browse_sdb,
            about_command=self._show_about,
        )

    def _set_app_icon(self) -> None:
        """Set the taskbar icon if available; never fail startup if missing."""
        ico_path = self._resource_path(os.path.join("assets", "aidas.ico"))
        png_path = self._resource_path(os.path.join("assets", "aidas.png"))

        if os.path.isfile(ico_path):
            self._icon_ico_path = ico_path
            try:
                self.iconbitmap(ico_path)
                return
            except tk.TclError:
                pass

        if os.path.isfile(png_path):
            try:
                image = tk.PhotoImage(file=png_path)
                self.iconphoto(True, image)
                self._icon_image_ref = image
            except tk.TclError:
                pass

    def _center_window(self) -> None:
        """Center the main window horizontally at the top of the screen."""
        self.update_idletasks()
        width = self.winfo_width()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        self.geometry(f"+{x}+0")

    def _menu_browse_sdb(self) -> None:
        self.notebook.select(0)
        file_path = filedialog.askopenfilename(
            title="Select SDB file (directory will be used)",
            initialdir=self.step1.sdb_dir_var.get() or None,
            filetypes=[("SDB files", "*.sdb"), ("All files", "*.*")],
        )
        if file_path:
            directory = os.path.dirname(file_path)
            self.step1.set_sdb_directory(directory)
            self.step1.refresh_sdb_list()

    def _set_theme(self, theme_name: str) -> None:
        """Change the application theme and save the preference."""
        self.style.theme_use(theme_name)
        self.preferences.set("theme", theme_name)
        self._build_menu()
        self.status.config(text=f"AIDaS v{__version__} — theme changed to '{theme_name}'")

    def _on_step1_processed_image(self, image, source_path) -> None:
        """Receive a cropped Step 1 image and load it into Step 2."""
        if getattr(self, "step2", None) is not None:
            self.step2.load_external_image(image, source_path=source_path)

    def _on_step2_output_folder_changed(self, folder) -> None:
        """Keep Step 3 pointed at Step 2's MARKED output folder."""
        if getattr(self, "step3", None) is not None:
            self.step3.set_input_folder(folder)

    def _show_about(self) -> None:
        """Open one modal About window, or focus the existing one."""
        dialog = self._about_dialog
        try:
            if dialog is not None and dialog.winfo_exists():
                dialog.lift()
                dialog.focus_force()
                return
        except tk.TclError:
            pass
        self._about_dialog = AboutDialog(self)


def _show_native_notice(title: str, message: str, *, error: bool = False) -> None:
    """Show a startup notice without constructing a second Tk application."""
    if os.name == "nt":
        icon_flag = 0x10 if error else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
        ctypes.windll.user32.MessageBoxW(None, message, title, icon_flag | 0x00010000)
    else:
        print(f"{title}: {message}", file=sys.stderr)


def main() -> int:
    """Run AIDaS if no other desktop instance owns the process guard."""
    guard = SingleInstanceGuard()
    try:
        if not guard.acquire():
            _close_bootloader_splash()
            _show_native_notice(APP_TITLE, "AIDaS is already running.")
            return 0
    except OSError as exc:
        _close_bootloader_splash()
        _show_native_notice(
            APP_TITLE,
            f"AIDaS could not verify that only one instance is running.\n\n{exc}",
            error=True,
        )
        return 1

    try:
        app = AIDaSApp()
        app.mainloop()
    finally:
        guard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
