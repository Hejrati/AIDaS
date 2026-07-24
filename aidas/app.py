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
from aidas.core.display import (
    centered_decorated_position,
    centered_geometry,
    enable_per_monitor_dpi_awareness,
    fit_size_to_bounds,
    fractional_size_of_bounds,
    work_area_bounds,
)
from aidas.core.single_instance import SingleInstanceGuard
from aidas.services.update_service import launch_installer
from aidas.services.update_ui import UpdateController
from aidas.utils.ui_layout import COLORS, LAYOUT
from aidas.utils.ui_utils import (
    apply_app_icon_to,
    build_app_menu,
    configure_aidas_styles,
    resource_path,
)


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

    WIDTH = 480
    HEIGHT = 620
    MAX_SCREEN_FRACTION = 0.88

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.withdraw()
        apply_app_icon_to(self)
        self.overrideredirect(True)
        self.configure(bg=BRAND_NAVY)

        bounds = work_area_bounds(self)
        dpi_scale = max(0.75, float(self.winfo_fpixels("1i")) / 96.0)
        splash_width, splash_height, fit_scale = fit_size_to_bounds(
            bounds,
            round(self.WIDTH * dpi_scale),
            round(self.HEIGHT * dpi_scale),
            maximum_fraction=self.MAX_SCREEN_FRACTION,
        )
        # Preserve the same apparent size at higher DPI when room permits,
        # then uniformly shrink everything when the monitor is too small.
        self.scale = dpi_scale * fit_scale

        def px(value: int) -> int:
            return max(1, round(value * self.scale))

        def font(value: int, *styles: str) -> tuple:
            # Negative Tk font sizes are pixels. This prevents the operating
            # system's DPI scaling from making text larger than its layout.
            return ("Segoe UI", -max(7, round(value * self.scale)), *styles)

        panel = tk.Frame(self, bg=WINDOW_BG, bd=0, highlightthickness=0)
        panel.pack(fill="both", expand=True, padx=1, pady=1)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(5, weight=1)

        logo_path = resource_path(os.path.join("assets", "aidas.png"))
        logo_size = px(240)
        with Image.open(logo_path) as logo:
            resized_logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(resized_logo)
        logo_label = tk.Label(
            panel,
            image=self.logo_image,
            bg=WINDOW_BG,
            bd=0,
            highlightthickness=0,
        )
        logo_label.grid(row=0, column=0, pady=(px(22), 0))

        tk.Label(
            panel,
            text=APP_TITLE,
            font=font(34, "bold"),
            fg=BRAND_NAVY,
            bg=WINDOW_BG,
        ).grid(row=1, column=0, pady=(px(12), 0))
        tk.Label(
            panel,
            text=APP_SUBTITLE,
            font=font(15),
            fg=BRAND_NAVY,
            bg=WINDOW_BG,
        ).grid(row=2, column=0, pady=(px(1), 0))
        tk.Label(
            panel,
            text=LAB_ACRONYM,
            font=font(24, "bold"),
            fg=BRAND_RED,
            bg=WINDOW_BG,
        ).grid(row=3, column=0, pady=(px(16), 0))
        tk.Label(
            panel,
            text=LAB_NAME,
            font=font(13, "bold"),
            fg=BODY_TEXT,
            bg=WINDOW_BG,
            justify="center",
            wraplength=max(1, splash_width - px(36)),
        ).grid(row=4, column=0, sticky="ew", padx=px(18), pady=(px(2), 0))

        loading_region = tk.Frame(panel, bg=WINDOW_BG, bd=0, highlightthickness=0)
        loading_region.grid(row=5, column=0, sticky="nsew", padx=px(32), pady=px(12))
        loading_region.grid_rowconfigure(0, weight=1)
        loading_region.grid_columnconfigure(0, weight=1)
        progress_header = tk.Frame(loading_region, bg=WINDOW_BG, bd=0, highlightthickness=0)
        progress_header.grid(row=0, column=0, sticky="ew")
        progress_header.grid_columnconfigure(0, weight=1)
        progress_header.grid_columnconfigure(1, weight=0)
        self.status_var = tk.StringVar(value="Starting AIDaS...")
        self.percent_var = tk.StringVar(value="0%")
        self.status_label = tk.Label(
            progress_header,
            textvariable=self.status_var,
            font=font(13),
            fg=BODY_TEXT,
            bg=WINDOW_BG,
            anchor="w",
            justify="left",
            # Keep the status text inside its column even if a future startup
            # stage has a substantially longer description.
            wraplength=max(px(120), splash_width - px(140)),
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=(0, px(12)))
        tk.Label(
            progress_header,
            textvariable=self.percent_var,
            font=font(13, "bold"),
            fg=BRAND_NAVY,
            bg=WINDOW_BG,
            anchor="e",
            width=4,
        ).grid(row=0, column=1, sticky="ne")

        tk.Label(
            panel,
            text=COPYRIGHT_NOTICE,
            font=font(11),
            fg=BODY_TEXT,
            bg=WINDOW_BG,
            justify="center",
            wraplength=max(1, splash_width - px(48)),
        ).grid(row=6, column=0, sticky="ew", padx=px(24), pady=(0, px(22)))

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

    PREFERRED_WIDTH = 520
    MAX_SCREEN_FRACTION = 0.9

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.withdraw()
        self.title("About AIDaS")
        self.configure(bg="#f2f2f2")
        self.resizable(True, True)
        self.transient(parent)
        apply_app_icon_to(self)

        bounds = work_area_bounds(self, parent=parent)
        available_width = max(1, bounds[2] - bounds[0])
        available_height = max(1, bounds[3] - bounds[1])
        dpi_scale = max(0.75, float(self.winfo_fpixels("1i")) / 96.0)
        maximum_width = max(1, round(available_width * self.MAX_SCREEN_FRACTION))
        maximum_height = max(1, round(available_height * self.MAX_SCREEN_FRACTION))
        dialog_width = max(1, min(round(self.PREFERRED_WIDTH * dpi_scale), maximum_width))
        content_wrap = max(40, dialog_width - 64)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        canvas = tk.Canvas(self, bg="#f2f2f2", bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        content = tk.Frame(canvas, bg="#f2f2f2", bd=0, highlightthickness=0)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        self._about_canvas = canvas
        self._about_content = content
        self._about_content_window = content_window
        self._wrapped_about_labels: list[tk.Label] = []

        def add_label(*, text: str, font, **options) -> tk.Label:
            label = tk.Label(
                content,
                text=text,
                font=font,
                fg=options.pop("fg", "#000000"),
                bg="#f2f2f2",
                justify=options.pop("justify", "center"),
                wraplength=content_wrap,
                **options,
            )
            self._wrapped_about_labels.append(label)
            return label

        add_label(
            text=APP_TITLE,
            font=("Segoe UI", 18, "bold"),
        ).pack(pady=(22, 0))
        add_label(
            text=f"{APP_SUBTITLE} - Version {__version__}",
            font=("Segoe UI", 9),
        ).pack(pady=(2, 0))
        add_label(
            text=LAB_ACRONYM,
            font=("Segoe UI", 11, "bold"),
            fg=BRAND_RED,
        ).pack(pady=(16, 0))
        add_label(
            text=LAB_NAME,
            font=("Segoe UI", 9),
        ).pack(pady=(5, 0))

        link = add_label(
            text=LAB_URL_TEXT,
            font=("Segoe UI", 9, "underline"),
            fg="#0066cc",
            cursor="hand2",
        )
        link.pack(pady=(11, 0))
        link.bind("<Button-1>", lambda _event: webbrowser.open_new_tab(LAB_URL))

        add_label(
            text=" ".join(LAB_DESCRIPTION.splitlines()),
            font=("Segoe UI", 8),
        ).pack(pady=(13, 0))
        add_label(
            text=UNIVERSITY_NAME,
            font=("Segoe UI", 8),
        ).pack(pady=(10, 0))
        add_label(
            text=COPYRIGHT_NOTICE,
            font=("Segoe UI", 8),
        ).pack(pady=(10, 0))
        add_label(
            text=f"Python {sys.version.split()[0]}",
            font=("Segoe UI", 8),
        ).pack(pady=(2, 16))

        button_panel = ttk.Frame(self, padding=(12, 8, 12, 12))
        button_panel.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(button_panel, text="OK", width=10, command=self._close).pack()

        content.bind("<Configure>", self._sync_about_scroll_region)
        canvas.bind("<Configure>", self._resize_about_content)
        self.bind("<MouseWheel>", self._scroll_about)

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self.bind("<Return>", lambda _event: self._close())
        self.update_idletasks()
        desired_height = content.winfo_reqheight() + button_panel.winfo_reqheight()
        preferred_minimum_height = min(round(240 * dpi_scale), maximum_height)
        dialog_height = max(1, min(maximum_height, max(preferred_minimum_height, desired_height)))
        self.minsize(
            min(round(320 * dpi_scale), dialog_width),
            min(round(240 * dpi_scale), dialog_height),
        )
        self.geometry(_center_geometry(self, dialog_width, dialog_height, parent=parent))
        self.deiconify()
        self.grab_set()
        self.focus_force()

    def _sync_about_scroll_region(self, _event=None) -> None:
        self._about_canvas.configure(scrollregion=self._about_canvas.bbox("all"))

    def _resize_about_content(self, event) -> None:
        width = max(1, int(event.width))
        self._about_canvas.itemconfigure(self._about_content_window, width=width)
        wraplength = max(40, width - 40)
        for label in self._wrapped_about_labels:
            label.configure(wraplength=wraplength)
        self._sync_about_scroll_region()

    def _scroll_about(self, event) -> str:
        if event.delta:
            self._about_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

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
        self.configure(background=COLORS.application)
        bounds = work_area_bounds(self)
        app_width, app_height = fractional_size_of_bounds(
            bounds, LAYOUT.screen_fraction
        )
        self._startup_window_size = (app_width, app_height)
        self.geometry(f"{app_width}x{app_height}")
        self.minsize(
            min(LAYOUT.minimum_width, app_width),
            min(LAYOUT.minimum_height, app_height),
        )
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
        self._startup_finished = False
        self._finish_startup_after_id = self.after(delay_ms, self._finish_startup)

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
        configure_aidas_styles(self.style)

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
        self.notebook = ttk.Notebook(self, style="AIDaS.TNotebook")
        self.notebook.pack(fill="both", expand=True)

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
            style="AIDaS.Status.TLabel",
            anchor="w",
        )
        self.status.pack(side="bottom", fill="x")

    def _finish_startup(self) -> None:
        """Close the splash and reveal the fully initialized main window."""
        if self._startup_finished:
            return
        self._startup_finished = True
        self._finish_startup_after_id = None
        splash = getattr(self, "_splash", None)
        if splash is not None and splash.winfo_exists():
            splash.destroy()
        self._splash = None
        width, height = self._startup_window_size
        self.geometry(_center_geometry(self, width, height))
        self.deiconify()
        self.update_idletasks()
        self._center_window(account_for_decorations=True)
        self.lift()
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
                try:
                    self.iconbitmap(default=ico_path)
                except tk.TclError:
                    pass
            except tk.TclError:
                pass

        if os.path.isfile(png_path):
            try:
                image = tk.PhotoImage(file=png_path)
                self.iconphoto(True, image)
                self._icon_image_ref = image
            except tk.TclError:
                pass

    def _center_window(self, *, account_for_decorations: bool = False) -> None:
        """Center the standard application window in the active work area."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        if not account_for_decorations:
            self.geometry(_center_geometry(self, width, height))
            return

        frame_left = max(0, self.winfo_rootx() - self.winfo_x())
        frame_top = max(0, self.winfo_rooty() - self.winfo_y())
        x, y = centered_decorated_position(
            work_area_bounds(self),
            width,
            height,
            frame_left=frame_left,
            frame_top=frame_top,
        )
        self.geometry(f"{width}x{height}{x:+d}{y:+d}")

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
        configure_aidas_styles(self.style)
        self.configure(background=COLORS.application)
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
