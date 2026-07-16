"""Main Tkinter application and branded information windows for AIDaS."""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import ctypes
import os
import sys
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from aidas import __version__
from aidas.core.config import Config
from aidas.core.display import centered_geometry, enable_per_monitor_dpi_awareness
from aidas.core.single_instance import SingleInstanceGuard
from aidas.services.update_service import launch_installer
from aidas.services.update_ui import UpdateController
from aidas.utils.ui_utils import apply_app_icon_to, build_app_menu, resource_path


APP_TITLE = "AIDaS"
APP_SUBTITLE = "OCT Image Processing"
LAB_ACRONYM = "MVPRL"
LAB_NAME = "Machine Vision and Pattern Recognition Lab"
LAB_URL = "https://mvprl.cs.wayne.edu"
LAB_URL_TEXT = "mvprl.cs.wayne.edu"
UNIVERSITY_NAME = "Wayne State University"
COPYRIGHT_NOTICE = (
    "Copyright (c) 2026 Machine Vision and Pattern Recognition Lab, "
    "Wayne State University. Licensed under GNU AGPL v3 or later."
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
SPLASH_MINIMUM_MS = 700


def _center_geometry(window: tk.Misc, width: int, height: int, *, parent=None) -> str:
    """Return geometry that centers a window on its parent or the screen."""
    return centered_geometry(window, width, height, parent=parent)


class SplashWindow(tk.Toplevel):
    """Dynamic startup window that reports initialization progress."""

    GOLDEN_RATIO = (1 + 5**0.5) / 2
    HEIGHT = 816
    WIDTH = round(HEIGHT / GOLDEN_RATIO)
    REFERENCE_SCREEN_WIDTH = 1920
    REFERENCE_SCREEN_HEIGHT = 1080

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.withdraw()
        self.overrideredirect(True)
        self.configure(bg=BRAND_NAVY)

        screen_width = max(1, self.winfo_screenwidth())
        screen_height = max(1, self.winfo_screenheight())
        self.scale = min(
            screen_width / self.REFERENCE_SCREEN_WIDTH,
            screen_height / self.REFERENCE_SCREEN_HEIGHT,
        )
        # The same factor drives both dimensions, preserving a portrait
        # golden-ratio frame (height / width = phi) on every display.
        splash_height = max(1, round(self.HEIGHT * self.scale))
        splash_width = max(1, round(splash_height / self.GOLDEN_RATIO))

        def px(value: int) -> int:
            return max(1, round(value * self.scale))

        def font_size(value: int) -> int:
            return max(6, round(value * self.scale))

        panel = tk.Frame(self, bg=WINDOW_BG, bd=0, highlightthickness=0)
        panel.pack(fill="both", expand=True, padx=1, pady=1)

        logo_path = resource_path(os.path.join("assets", "aidas.png"))
        logo_size = px(450)
        with Image.open(logo_path) as logo:
            resized_logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(resized_logo)
        tk.Label(
            panel,
            image=self.logo_image,
            bg=WINDOW_BG,
            bd=0,
            highlightthickness=0,
        ).pack(pady=(px(24), 0))

        tk.Label(
            panel,
            text=APP_TITLE,
            font=("Segoe UI", font_size(28), "bold"),
            fg=BRAND_NAVY,
            bg=WINDOW_BG,
        ).pack(pady=(px(14), 0))
        tk.Label(
            panel,
            text=APP_SUBTITLE,
            font=("Segoe UI", font_size(11)),
            fg=BRAND_NAVY,
            bg=WINDOW_BG,
        ).pack(pady=(px(1), 0))
        tk.Label(
            panel,
            text=LAB_ACRONYM,
            font=("Segoe UI", font_size(19), "bold"),
            fg=BRAND_RED,
            bg=WINDOW_BG,
        ).pack(pady=(px(20), 0))
        tk.Label(
            panel,
            text=LAB_NAME,
            font=("Segoe UI", font_size(9), "bold"),
            fg=BODY_TEXT,
            bg=WINDOW_BG,
        ).pack(pady=(px(1), 0))
        copyright_panel = tk.Frame(panel, bg=WINDOW_BG, bd=0, highlightthickness=0)
        # Reserve the copyright footer before the branding widgets consume
        # the available height.
        copyright_panel.pack(
            side="bottom",
            fill="x",
            padx=px(42),
            pady=(0, px(34)),
            before=panel.winfo_children()[0],
        )

        tk.Label(
            copyright_panel,
            text=COPYRIGHT_NOTICE,
            font=("Segoe UI", font_size(8)),
            fg=BODY_TEXT,
            bg=WINDOW_BG,
            justify="center",
            wraplength=px(420),
        ).pack(fill="x")

        # This region receives all space remaining between the MVPR branding
        # and copyright footer. Expanding it centers the loading line exactly
        # between those two sections.
        loading_region = tk.Frame(panel, bg=WINDOW_BG, bd=0, highlightthickness=0)
        loading_region.pack(fill="both", expand=True, padx=px(42))
        progress_header = tk.Frame(loading_region, bg=WINDOW_BG, bd=0, highlightthickness=0)
        progress_header.pack(fill="x", expand=True)
        self.status_var = tk.StringVar(value="Starting AIDaS...")
        self.percent_var = tk.StringVar(value="0%")
        tk.Label(
            progress_header,
            textvariable=self.status_var,
            font=("Segoe UI", font_size(9)),
            fg=BODY_TEXT,
            bg=WINDOW_BG,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            progress_header,
            textvariable=self.percent_var,
            font=("Segoe UI", font_size(9), "bold"),
            fg=BRAND_NAVY,
            bg=WINDOW_BG,
            anchor="e",
        ).pack(
            side="right",
            # Reserve room for the percentage before the variable-length
            # loading message is laid out.
            before=progress_header.winfo_children()[0],
        )

        self.attributes("-topmost", True)
        self.geometry(_center_geometry(self, splash_width, splash_height))
        self.deiconify()
        self.lift()

    def set_progress(self, value: float, message: str) -> None:
        """Update the visible startup stage and percentage immediately."""
        percent = max(0.0, min(float(value), 100.0))
        self.percent_var.set(f"{percent:.0f}%")
        self.status_var.set(str(message))
        self.update_idletasks()


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
        enable_per_monitor_dpi_awareness()
        super().__init__()
        self.withdraw()
        self.title("AIDaS — Retinal Image Processing")
        self.geometry("1280x820")
        self.minsize(900, 600)
        self._about_dialog = None
        self._set_app_icon()

        self._splash_started_at = time.monotonic()
        self._splash = SplashWindow(self)
        self._set_splash_progress(3, "Starting AIDaS...")

        try:
            self._build_application()
        except Exception:
            try:
                splash = self._splash
                if splash is not None and splash.winfo_exists():
                    splash.destroy()
            finally:
                self.destroy()
            raise

        self._set_splash_progress(100, "Ready")
        elapsed_ms = int((time.monotonic() - self._splash_started_at) * 1000)
        delay_ms = max(0, SPLASH_MINIMUM_MS - elapsed_ms)
        self.after(delay_ms, self._finish_startup)

    def _set_splash_progress(self, value: float, message: str) -> None:
        """Paint one startup stage while the main window is still hidden."""
        splash = getattr(self, "_splash", None)
        if splash is None or not splash.winfo_exists():
            return
        splash.set_progress(value, message)
        self.update_idletasks()
        self.update()

    def _build_application(self) -> None:
        """Build the main UI while the splash remains visible."""
        # Keep scientific and imaging imports behind the splash and expose each
        # expensive stage instead of leaving the startup window motionless.
        self._set_splash_progress(8, "Loading Step 1 image tools...")
        from aidas.steps.step1_resize_raw import Step1Frame

        self._set_splash_progress(18, "Loading Step 2 canvas and AI tools...")
        from aidas.steps.step2_annotate import Step2Frame

        self._set_splash_progress(30, "Loading Step 3 flattening tools...")
        from aidas.steps.step3_flatten import Step3Frame

        self._set_splash_progress(42, "Loading Step 4 analysis tools...")
        from aidas.steps.step4_analyze_isez import Step4Frame

        self._set_splash_progress(50, "Loading preferences...")
        self.preferences = Config()
        self._set_splash_progress(54, "Applying the interface theme...")
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        default_theme = "xpnative" if "xpnative" in available_themes else available_themes[0]
        current_theme = self.preferences.get("theme", default_theme)

        if current_theme in available_themes:
            self.style.theme_use(current_theme)
        else:
            self.style.theme_use(available_themes[0])
            self.preferences.set("theme", available_themes[0])

        self._set_splash_progress(58, "Starting application services...")
        self.update_controller = UpdateController(
            self,
            preferences=self.preferences,
            current_version=__version__,
            status_callback=self._set_status_message,
            restart_blocker_callback=self._update_restart_blocker,
            install_callback=self._queue_update_install,
        )
        self._build_menu()
        self.bind_all("<Alt-F4>", lambda _event: self.destroy())

        self._set_splash_progress(62, "Creating the application workspace...")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=(2, 5), pady=(5, 1))

        self._set_splash_progress(66, "Preparing Step 1 - Load, Resize & Crop...")
        self.step1 = Step1Frame(
            self.notebook,
            preferences=self.preferences,
            on_processed_image=self._on_step1_processed_image,
        )
        self.notebook.add(self.step1, text="  Step 1 — Load, Resize & Crop  ")

        self._set_splash_progress(74, "Preparing Step 2 - Annotate and Segment...")
        self.step2 = Step2Frame(
            self.notebook,
            preferences=self.preferences,
            source_step=self.step1,
            on_output_folder_changed=self._on_step2_output_folder_changed,
        )
        self.notebook.add(self.step2, text="  Step 2 — Annotate and Segment  ")

        self._set_splash_progress(83, "Preparing Step 3 - Flatten Retina...")
        self.step3 = Step3Frame(self.notebook, preferences=self.preferences)
        self.notebook.add(self.step3, text="  Step 3 — Flatten Retina  ")

        self._set_splash_progress(91, "Preparing Step 4 - Analyze ISEZ...")
        self.step4 = Step4Frame(self.notebook, preferences=self.preferences)
        self.notebook.add(self.step4, text="  Step 4 — Analyze ISEZ  ")

        self._set_splash_progress(97, "Finalizing the main window...")
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
        self.update_idletasks()
        self.update()
        self.focus_force()
        self.after(1500, self.update_controller.check_automatically)

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
            check_updates_command=self.update_controller.check_now,
            about_command=self._show_about,
        )

    def _set_status_message(self, message: str) -> None:
        """Show a transient application-level status without assuming startup is complete."""
        status = getattr(self, "status", None)
        if status is not None:
            status.config(text=f"AIDaS v{__version__} — {message}")

    def _update_restart_blocker(self) -> str | None:
        """Describe work that must finish before replacing the application."""
        step2 = getattr(self, "step2", None)
        if step2 is not None and getattr(step2, "_segmenter_running", False):
            return "Step 2 AI segmentation is still running."

        step3 = getattr(self, "step3", None)
        if step3 is not None:
            if getattr(step3, "_busy", False):
                return "Step 3 R batch processing is still running."
            setup_panel = getattr(step3, "r_setup_panel", None)
            if setup_panel is not None and getattr(setup_panel, "busy", False):
                return "Step 3 R or package setup is still running."
        return None

    def _queue_update_install(self, installer_path) -> None:
        """Close the UI; main() starts Setup after releasing the app mutex."""
        self._pending_update_installer = installer_path
        self.destroy()

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
            _show_native_notice(APP_TITLE, "AIDaS is already running.")
            return 0
    except OSError as exc:
        _show_native_notice(
            APP_TITLE,
            f"AIDaS could not verify that only one instance is running.\n\n{exc}",
            error=True,
        )
        return 1

    app = None
    pending_update = None
    interrupted = False
    try:
        app = AIDaSApp()
        try:
            app.mainloop()
        except KeyboardInterrupt:
            # Stopping a VS Code debug session (or pressing Ctrl+C in a
            # terminal) interrupts Tk's blocking event loop.  Treat that as a
            # normal user-requested shutdown instead of emitting a traceback.
            interrupted = True
            try:
                app.destroy()
            except tk.TclError:
                pass
        if not interrupted:
            pending_update = getattr(app, "_pending_update_installer", None)
    finally:
        guard.close()

    if interrupted:
        return 0

    if pending_update is not None:
        try:
            launch_installer(pending_update)
        except Exception as exc:
            _show_native_notice(
                "AIDaS Update",
                f"The verified update was downloaded, but Windows could not start the installer.\n\n{exc}",
                error=True,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
