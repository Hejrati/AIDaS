"""Step 3 - batch OCT flattening with the original R workflow."""

from __future__ import annotations

import shutil
import subprocess
import re
import queue
import signal
import time
import urllib.request
from datetime import datetime
import concurrent.futures
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import threading
from pathlib import Path
import os
import sys
import zipfile

from PIL import Image, ImageColor, ImageOps, ImageTk

try:
    import pyreadr
except Exception:
    pyreadr = None

from aidas.utils.io_utils import read_analyze
from aidas.utils.filesystem import skipped_directories_warning, walk_accessible_directories
from aidas.utils.step3_image_utils import (
    placeholder_image as _placeholder_image,
)
from aidas.utils.log_paths import app_log_dir
from aidas.utils.r_script_library import discover_r_scripts, import_r_script, user_r_script_dir
from aidas.ui.theme import COLOR_PAIRS, COLORS, resolve_color
from aidas.ui.tabs import ClosableTabView
from aidas.utils.ui_utils import (
    HoverToolTip,
    SidebarStepFrame,
    action_button,
    load_color_close_icon,
    load_action_icon,
    resource_path,
)


def _normalize_analyze_path(base_path):
    """Return the Analyze header path for a base path or .hdr path."""
    path = str(base_path)
    if path.lower().endswith(".hdr"):
        return Path(path)
    return Path(f"{path}.hdr")


def _load_analyze_volume_r_layout(path):
    """Load an Analyze volume using the R/script display layout."""
    volume = np.asarray(read_analyze(_normalize_analyze_path(path)))
    if volume.ndim == 3:
        volume = np.transpose(volume, (2, 1, 0))[:, ::-1, :]
    if volume.ndim == 4:
        volume = volume[:, :, :, 0]
    return volume


def _grand_profile_and_vertex(final_grand_mean):
    profile_y = np.nanmean(np.asarray(final_grand_mean, dtype=np.float64), axis=0)
    profile_x = np.arange(1.0, profile_y.size + 1.0, 1.0)
    valid = np.where(np.isfinite(profile_y))[0]
    vertex = int(valid[np.nanargmin(profile_y[valid])] + 1) if valid.size else 431
    return np.column_stack((profile_x, profile_y)), vertex


class RSetupWizard(ttk.Frame):
    """Guided R and R-package setup for the original Step 3 R script."""

    STEPS = ("Setup",)

    def __init__(self, step_frame, parent, on_finish=None, close_command=None):
        super().__init__(parent)
        self.step_frame = step_frame
        self.on_finish = on_finish
        self.close_command = close_command
        self.result = None
        self.cancelled = True
        self.current_step = 0
        self.busy = False
        self.rscript_path = step_frame._resolve_rscript_executable()
        self.installer_name = ""
        self.installer_url = ""
        self.installer_path = None
        self.setup_package_names = tuple(
            getattr(step_frame, "R_LOCAL_PACKAGE_ORDER", None)
            or step_frame.R_REQUIRED_PACKAGES
        )
        self.package_status = {name: "pending" for name in self.setup_package_names}
        self.package_library_path = Path(self._default_package_library())
        self.log_path = self._package_log_path()

        self._build_styles()
        self._build_shell()
        self._render_step()
        self.focus_set()

    def _dismiss(self, *, render_previous=True):
        if self.close_command is not None:
            self.close_command()
            return
        close_panel = getattr(self.step_frame, "_close_r_setup_panel", None)
        if callable(close_panel):
            close_panel(render_previous=render_previous)
        else:
            self.destroy()

    def _build_styles(self):
        self.style = ttk.Style(self)
        self.style.configure("WizardTitle.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("WizardSubtitle.TLabel", foreground=COLORS.muted_text)
        self.style.configure("WizardStep.TLabel", padding=(10, 7))
        self.style.configure("WizardStepActive.TLabel", padding=(10, 7), font=("Segoe UI", 9, "bold"))
        self.style.configure("WizardStepDone.TLabel", padding=(10, 7), foreground=COLORS.success)
        self.style.configure("WizardAccent.TButton", padding=(10, 5))

    def _build_shell(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="R environment setup", style="WizardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text=f"Set up R {self.step_frame.R_REQUIRED_VERSION} and the two packages required by Step 3.",
            style="WizardSubtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        middle = ttk.Frame(root)
        middle.pack(fill="both", expand=True)

        self.step_rail = ttk.Frame(middle, width=180)
        self.step_rail.pack(side="left", fill="y", padx=(0, 12))
        self.step_rail.pack_propagate(False)
        self.step_labels = []
        for label in self.STEPS:
            step_label = ttk.Label(self.step_rail, text=label, style="WizardStep.TLabel", anchor="w")
            step_label.pack(fill="x", pady=1)
            self.step_labels.append(step_label)

        right = ttk.Frame(middle)
        right.pack(side="left", fill="both", expand=True)

        self.content = ttk.Frame(right)
        self.content.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(right, text="Setup log")
        log_frame.pack(fill="both", expand=False, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(10, 0))
        self.next_button = action_button(
            footer,
            self,
            "Next",
            self._next,
            "next",
            style="AIDaS.PrimaryAction.TButton",
        )
        self._wizard_finish_icon = load_action_icon(self, "confirm")
        self.next_button.pack(side="right", padx=(4, 0))
        self.back_button = action_button(footer, self, "Back", self._back, "previous")
        self.back_button.pack(side="right", padx=(4, 0))

        self.cancel_button = action_button(footer, self, "Cancel", self._cancel, "cancel")
        self.cancel_button.pack(side="right")

        self._log("R setup wizard opened.")

    def _set_busy(self, busy, text=None):
        self.busy = bool(busy)
        for button in (self.back_button, self.next_button, self.cancel_button):
            button.configure(state="disabled" if busy else "normal")
        self._set_content_buttons("disabled" if busy else "normal")
        if text:
            self._log(text)
        if not busy:
            self._update_nav()

    def _set_content_buttons(self, state):
        def visit(parent):
            for child in parent.winfo_children():
                if isinstance(child, ttk.Button):
                    child.configure(state=state)
                visit(child)

        visit(self.content)

    def _update_nav(self):
        for idx, label in enumerate(self.step_labels):
            prefix = "[x] " if idx < self.current_step else ("[>] " if idx == self.current_step else "[ ] ")
            label.configure(text=prefix + self.STEPS[idx])
            if idx < self.current_step:
                label.configure(style="WizardStepDone.TLabel")
            elif idx == self.current_step:
                label.configure(style="WizardStepActive.TLabel")
            else:
                label.configure(style="WizardStep.TLabel")

        self.back_button.configure(state="disabled" if self.current_step == 0 else "normal")
        is_finish = self.current_step == len(self.STEPS) - 1
        self.next_button.configure(
            text="Finish" if is_finish else "Next",
            image=self._wizard_finish_icon if is_finish else self.next_button._aidas_action_icon,
        )
        if self.current_step == 0 and self.rscript_path is None:
            self.next_button.configure(state="disabled")
        elif self.current_step == 1 and (
            self.rscript_path is None or not self._all_packages_ready()
        ):
            self.next_button.configure(state="disabled")
        else:
            self.next_button.configure(state="normal")

    def _render_step(self):
        self._clear_content()
        renderers = (
            self._render_r_program,
            self._render_packages,
            self._render_finish,
        )
        renderers[self.current_step]()
        self._update_nav()

    def _clear_content(self):
        for child in self.content.winfo_children():
            child.destroy()

    def _section_title(self, title, subtitle):
        ttk.Label(self.content, text=title, style="WizardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            self.content,
            text=subtitle,
            style="WizardSubtitle.TLabel",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(4, 14))

    def _render_r_program(self):
        self._section_title(
            f"R {self.step_frame.R_REQUIRED_VERSION}",
            "AIDaS checks the installed program before using it. Other R versions are ignored.",
        )
        status = (
            f"Ready\n{self.rscript_path}"
            if self.rscript_path is not None
            else f"R {self.step_frame.R_REQUIRED_VERSION} was not found. Search for it or install it below."
        )
        self.r_status_var = tk.StringVar(value=status)

        form = ttk.LabelFrame(self.content, text="R program")
        form.pack(fill="x")
        ttk.Label(form, textvariable=self.r_status_var, wraplength=620, justify="left").pack(
            anchor="w", padx=10, pady=10
        )

        actions = ttk.Frame(self.content)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Check Again", command=self._detect_rscript).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Search Locally...", command=self._locate_rscript).pack(side="left", padx=(0, 6))
        if self.rscript_path is None:
            self.r_install_button = ttk.Button(
                actions,
                text=f"Download and Install R {self.step_frame.R_REQUIRED_VERSION}",
                command=self._download_and_install_r,
            )
            self.r_install_button.pack(side="left", padx=(0, 6))

    def _render_packages(self):
        self._section_title(
            "Packages",
            "AIDaS manages the package location automatically and installs the bundled local packages and their dependencies.",
        )
        ttk.Label(
            self.content,
            text=f"Managed package library:\n{self.package_library_path}",
            style="AIDaS.Muted.TLabel",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        table = ttk.LabelFrame(self.content, text="Package status")
        table.pack(fill="x")
        self.package_status_vars = {}
        for package_name in self.step_frame.R_REQUIRED_PACKAGES:
            row = ttk.Frame(table)
            row.pack(fill="x", padx=10, pady=5)
            ttk.Label(row, text=package_name, width=18).pack(side="left")
            var = tk.StringVar(value=self.package_status.get(package_name, "pending"))
            self.package_status_vars[package_name] = var
            ttk.Label(row, textvariable=var).pack(side="left", fill="x", expand=True)

        actions = ttk.Frame(self.content)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Check Packages", command=self._check_packages).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Install Missing", command=self._install_missing_packages).pack(side="left")

        ttk.Label(
            self.content,
            text="All package installs use AIDaS's local resource bundle; the setup does not contact CRAN.",
            style="AIDaS.Muted.TLabel",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _render_finish(self):
        self._section_title(
            "Ready",
            "R and the Step 3 packages are ready. You can now run batch Step 3 R processing.",
        )
        summary = (
            f"Rscript:\n{self.rscript_path}\n\n"
            f"Package library:\n{self.package_library_path}\n\n"
            f"Log:\n{self.log_path}"
        )
        ttk.Label(self.content, text=summary, wraplength=620, justify="left").pack(anchor="w")

    def _next(self):
        if self.busy:
            return
        if self.current_step == len(self.STEPS) - 1:
            self._finish()
            return
        self.current_step = min(len(self.STEPS) - 1, self.current_step + 1)
        self._render_step()
        if self.current_step == 1 and self.rscript_path is not None:
            self.after(100, self._check_packages)

    def _back(self):
        if self.busy:
            return
        self.current_step = max(0, self.current_step - 1)
        self._render_step()

    def _cancel(self):
        if self.busy:
            return
        self.cancelled = True
        self.result = None
        self._dismiss(render_previous=True)

    def _finish(self):
        self.cancelled = False
        self.result = Path(self.rscript_path) if self.rscript_path is not None else None
        if self.result is not None:
            self.step_frame.r_package_library_path = str(self.package_library_path)
            if self.step_frame.preferences is not None:
                self.step_frame.preferences.set("rscript_path", str(self.result))
                self.step_frame.preferences.set("r_package_library_path", str(self.package_library_path))
        callback = self.on_finish
        result = self.result
        self._dismiss(render_previous=callback is None)
        if callback is not None:
            self.step_frame.after(0, lambda: callback(result))

    def _detect_rscript(self, schedule_package_check=True):
        self.rscript_path = self.step_frame._resolve_rscript_executable()
        self.package_status = {name: "pending" for name in self.setup_package_names}
        self._log(f"Rscript detection: {self.rscript_path or 'not found'}")
        self._refresh_status_display()
        if self.rscript_path is not None and schedule_package_check:
            self.after(100, self._check_packages)

    def _locate_rscript(self):
        selected = filedialog.askopenfilename(
            title=f"Select R {self.step_frame.R_REQUIRED_VERSION} program",
            initialdir=r"C:\Program Files\R" if os.name == "nt" else (self.step_frame.current_sdb_dir or None),
            filetypes=[
                ("R program", "*.exe"),
                ("Executable files", "*.exe"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        rscript = self.step_frame._normalize_r_executable(Path(selected))
        if rscript is None:
            messagebox.showerror(
                "Select R program",
                f"Please select Rscript.exe, R.exe, or Rterm.exe from the R "
                f"{self.step_frame.R_REQUIRED_VERSION} installation.",
                parent=self,
            )
            return
        version = self.step_frame._r_version_for_executable(rscript)
        if version != self.step_frame.R_REQUIRED_VERSION:
            detected = version or "unknown"
            messagebox.showerror(
                "Unsupported R version",
                f"This Step 3 setup only accepts R {self.step_frame.R_REQUIRED_VERSION}. "
                f"The selected program reported R {detected}.",
                parent=self,
            )
            self._log(f"Rejected manually selected R program {rscript} (version {detected}).")
            return
        self.rscript_path = rscript
        self.package_status = {name: "pending" for name in self.setup_package_names}
        if self.step_frame.preferences is not None:
            self.step_frame.preferences.set("rscript_path", str(rscript))
        self._log(f"Selected R {self.step_frame.R_REQUIRED_VERSION}: {rscript}")
        self._refresh_status_display()
        self.after(100, self._check_packages)

    def _download_and_install_r(self, continue_to_packages=True):
        if not self.installer_url:
            self.installer_name = self.step_frame.R_INSTALLER_NAME
            self.installer_url = self.step_frame.R_DOWNLOAD_PAGE + self.installer_name
            self.installer_path = self._r_installer_cache_path()
            self._log(f"R {self.step_frame.R_REQUIRED_VERSION} installer selected: {self.installer_url}")
        self.installer_path = self.installer_path or self._r_installer_cache_path()
        if self.installer_path.exists():
            install_existing = messagebox.askyesno(
                f"Install R {self.step_frame.R_REQUIRED_VERSION}",
                "The R installer is already downloaded in the app's local files.\n\nInstall it now?",
                parent=self,
            )
            if install_existing:
                self._run_downloaded_installer(continue_to_packages=continue_to_packages)
            return

        def worker():
            self.installer_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.installer_url, self.installer_path)
            return self.installer_path

        def done(value, error):
            if error:
                self._log(f"R installer download failed: {error}")
                messagebox.showerror("R Setup", f"Could not download R.\n{error}", parent=self)
                return
            self._log(f"Downloaded R installer: {value}")
            install_now = messagebox.askyesno(
                f"Install R {self.step_frame.R_REQUIRED_VERSION}",
                f"R {self.step_frame.R_REQUIRED_VERSION} was downloaded successfully.\n\n"
                "Do you want to install it now?",
                parent=self,
            )
            if install_now:
                self._run_downloaded_installer(continue_to_packages=continue_to_packages)
            else:
                messagebox.showinfo(
                    "R Setup",
                    "The installer was downloaded. You can install it later from this screen.",
                    parent=self,
                )

        self._run_worker("Downloading R installer...", worker, done)

    def _run_downloaded_installer(self, continue_to_packages=True):
        installer_path = self.installer_path
        if not installer_path or not installer_path.is_file():
            messagebox.showwarning("R Setup", "Download the R installer first.", parent=self)
            return

        def worker():
            return subprocess.run([str(installer_path)], check=False).returncode

        def done(value, error):
            if error:
                self._log(f"R installer could not be started: {error}")
                messagebox.showerror("R Setup", f"Could not run the R installer.\n{error}", parent=self)
                return
            self._log(f"R installer closed with return code {value}.")
            self._detect_rscript(schedule_package_check=False)
            if self.rscript_path is not None:
                messagebox.showinfo(
                    "R Setup",
                    f"R {self.step_frame.R_REQUIRED_VERSION} is ready. Missing packages will now be installed.",
                    parent=self,
                )
                if continue_to_packages:
                    self.after(200, self._install_missing_packages)
            else:
                messagebox.showwarning(
                    "R Setup",
                    f"AIDaS still cannot find R {self.step_frame.R_REQUIRED_VERSION}. "
                    "Use Check Again or Search Locally.",
                    parent=self,
                )

        self._run_worker("Running R installer. Complete the installer window to continue.", worker, done)

    def _cancel(self):
        if self.busy:
            return
        self.cancelled = True
        self.result = None
        self._dismiss(render_previous=True)

    def _finish(self):
        self.cancelled = False
        self.result = Path(self.rscript_path) if self.rscript_path is not None else None
        if self.result is not None:
            self.step_frame.r_package_library_path = str(self.package_library_path)
            if self.step_frame.preferences is not None:
                self.step_frame.preferences.set("rscript_path", str(self.result))
                self.step_frame.preferences.set("r_package_library_path", str(self.package_library_path))
        callback = self.on_finish
        result = self.result
        self._dismiss(render_previous=callback is None)
        if callback is not None:
            self.step_frame.after(0, lambda: callback(result))

    def _default_package_library(self):
        configured = getattr(self.step_frame, "r_package_library_path", None)
        if configured:
            configured_path = Path(configured)
            local_app_data = os.environ.get("LOCALAPPDATA")
            legacy_path = (
                Path(local_app_data) / "AIDaS" / "R-packages"
                if local_app_data
                else None
            )
            try:
                if legacy_path is None or configured_path.resolve() != legacy_path.resolve():
                    return configured_path
            except OSError:
                return configured_path

        documents = Path.home() / "Documents"
        if documents.is_dir():
            return documents / "AIDaS" / "R-packages"
        return Path.home() / "AIDaS_R_packages"

    def _package_log_path(self):
        return app_log_dir() / "step3_r_package_setup.log"

    def _log(self, message):
        text = f"{datetime.now().strftime('%H:%M:%S')}  {message}"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def _log_process_result(self, title, cmd, result):
        self._log(f"{title}: return code {result.returncode}")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write("Command:\n" + " ".join(str(part) for part in cmd) + "\n")
            handle.write("STDOUT:\n" + (result.stdout or "").rstrip() + "\n")
            handle.write("STDERR:\n" + (result.stderr or "").rstrip() + "\n\n")

    @staticmethod
    def _r_string(value):
        return "'" + str(value).replace("\\", "/").replace("'", "\\'") + "'"

    def _r_eval_command(self, expression):
        return self.step_frame._build_r_eval_command(self.rscript_path, expression)

    def _run_worker(self, title, worker, done):
        self._set_busy(True, title)

        def wrapped():
            try:
                value = worker()
                error = None
            except Exception as exc:
                value = None
                error = exc
            self.after(0, lambda: self._finish_worker(done, value, error))

        threading.Thread(target=wrapped, daemon=True).start()

    def _finish_worker(self, done, value, error):
        self._set_busy(False)
        done(value, error)

    def _detect_rscript(self):
        self.rscript_path = self.step_frame._resolve_rscript_executable()
        if hasattr(self, "r_status_var"):
            self.r_status_var.set(
                str(self.rscript_path)
                if self.rscript_path
                else f"R {self.step_frame.R_REQUIRED_VERSION} not found"
            )
        self._log(f"Rscript detection: {self.rscript_path or 'not found'}")
        if self.current_step == 0:
            self._render_step()
        else:
            self._update_nav()

    def _locate_rscript(self):
        selected = filedialog.askopenfilename(
            title=f"Select R {self.step_frame.R_REQUIRED_VERSION} program",
            initialdir=r"C:\Program Files\R" if os.name == "nt" else (self.step_frame.current_sdb_dir or None),
            filetypes=[
                ("R program", "*.exe"),
                ("Executable files", "*.exe"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        rscript = self.step_frame._normalize_r_executable(Path(selected))
        if rscript is None:
            messagebox.showerror(
                "Select R program",
                f"Please select Rscript.exe, R.exe, or Rterm.exe from the R "
                f"{self.step_frame.R_REQUIRED_VERSION} installation.",
                parent=self,
            )
            return
        version = self.step_frame._r_version_for_executable(rscript)
        if version != self.step_frame.R_REQUIRED_VERSION:
            detected = version or "unknown"
            messagebox.showerror(
                "Unsupported R version",
                f"This Step 3 setup only accepts R {self.step_frame.R_REQUIRED_VERSION}. "
                f"The selected program reported R {detected}.",
                parent=self,
            )
            self._log(f"Rejected manually selected R program {rscript} (version {detected}).")
            return
        self.rscript_path = rscript
        if self.step_frame.preferences is not None:
            self.step_frame.preferences.set("rscript_path", str(rscript))
        if hasattr(self, "r_status_var"):
            self.r_status_var.set(str(rscript))
        self._log(f"Selected R {self.step_frame.R_REQUIRED_VERSION}: {rscript}")
        self._render_step()

    def _r_installer_cache_path(self):
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = (Path(local_app_data) / "AIDaS") if local_app_data else (Path.home() / ".aidas")
        return base / "R" / self.step_frame.R_INSTALLER_NAME

    def _download_and_install_r(self):
        if not self.installer_url:
            self.installer_name = self.step_frame.R_INSTALLER_NAME
            self.installer_url = self.step_frame.R_DOWNLOAD_PAGE + self.installer_name
            self.installer_path = self._r_installer_cache_path()
            self._log(f"R {self.step_frame.R_REQUIRED_VERSION} installer selected: {self.installer_url}")
        self.installer_path = self.installer_path or self._r_installer_cache_path()
        if self.installer_path.exists():
            install_existing = messagebox.askyesno(
                f"Install R {self.step_frame.R_REQUIRED_VERSION}",
                "The R installer is already downloaded in the app's local files.\n\nInstall it now?",
                parent=self,
            )
            if install_existing:
                self._run_downloaded_installer()
            return

        def worker():
            self.installer_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.installer_url, self.installer_path)
            return self.installer_path

        def done(value, error):
            if error:
                self._log(f"R installer download failed: {error}")
                messagebox.showerror("R Setup", f"Could not download R.\n{error}", parent=self)
                return
            self._log(f"Downloaded R installer: {value}")
            install_now = messagebox.askyesno(
                f"Install R {self.step_frame.R_REQUIRED_VERSION}",
                f"R {self.step_frame.R_REQUIRED_VERSION} was downloaded successfully.\n\n"
                "Do you want to install it now?",
                parent=self,
            )
            if install_now:
                self._run_downloaded_installer()
            else:
                if hasattr(self, "r_install_button"):
                    self.r_install_button.configure(
                        text=f"Install downloaded R {self.step_frame.R_REQUIRED_VERSION}"
                    )
                messagebox.showinfo(
                    "R Setup",
                    "The installer was downloaded. You can install it later from this screen.",
                    parent=self,
                )

        self._run_worker("Downloading R installer...", worker, done)

    def _run_downloaded_installer(self):
        installer_path = self.installer_path
        if not installer_path or not installer_path.is_file():
            messagebox.showwarning("R Setup", "Download the R installer first.", parent=self)
            return

        def worker():
            result = subprocess.run([str(installer_path)], check=False)
            return result.returncode

        def done(value, error):
            if error:
                self._log(f"R installer could not be started: {error}")
                messagebox.showerror("R Setup", f"Could not run the R installer.\n{error}", parent=self)
                return
            self._log(f"R installer closed with return code {value}.")
            self._detect_rscript()
            if self.rscript_path is not None:
                messagebox.showinfo("R Setup", "R 3.3.1 was found. Click Next to set up packages.", parent=self)
            else:
                messagebox.showwarning(
                    "R Setup",
                    "AIDaS still cannot find R 3.3.1. Finish the installer if it is still open, then click Check Again or Search Locally.",
                    parent=self,
                )

        self._run_worker("Running R installer. Complete the installer window to continue.", worker, done)

    def _package_check_expression(self, package_name):
        lib = self._r_string(self.package_library_path.resolve())
        return (
            f".libPaths(c({lib}, .libPaths())); "
            f"if (requireNamespace({self._r_string(package_name)}, quietly=TRUE)) "
            "quit(status=0) else quit(status=1)"
        )

    def _local_package_path(self, package_name):
        package_file = self.step_frame.R_LOCAL_PACKAGE_FILES.get(package_name)
        if not package_file:
            raise RuntimeError(f"No bundled local package is configured for {package_name}.")
        package_path = Path(resource_path(os.path.join("assets", "r_packages", package_file)))
        if not package_path.is_file():
            raise RuntimeError(f"Bundled R package is missing: {package_path}")
        try:
            with zipfile.ZipFile(package_path) as archive:
                has_description = any(Path(name).name == "DESCRIPTION" for name in archive.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"Bundled R package is not a valid ZIP archive: {package_path}") from exc
        if not has_description:
            raise RuntimeError(f"Bundled R package has no DESCRIPTION file: {package_path}")
        return package_path

    def _package_install_expression(self, package_name):
        lib = self._r_string(self.package_library_path.resolve())
        package_path = self._r_string(self._local_package_path(package_name).resolve())
        return "".join(
            (
                f".libPaths(c({lib}, .libPaths())); ",
                f"install.packages({package_path}, repos=NULL, dependencies=FALSE, ",
                f"type='win.binary', lib={lib})",
            )
        )

    def _check_package_status_worker(self):
        if self.rscript_path is None:
            raise RuntimeError("Rscript is not selected.")
        self.package_library_path.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["R_LIBS_USER"] = str(self.package_library_path.resolve())
        statuses = {}
        for package_name in self.step_frame.R_REQUIRED_PACKAGES:
            expression = self._package_check_expression(package_name)
            cmd = self._r_eval_command(expression)
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._log_process_result(f"Check package {package_name}", cmd, result)
            statuses[package_name] = "installed" if result.returncode == 0 else "missing"
        return statuses

    def _check_packages(self):
        if self.current_step != 1:
            return

        def done(value, error):
            if error:
                self._log(f"Package check failed: {error}")
                messagebox.showerror("R Packages", f"Could not check packages.\n{error}", parent=self)
                return
            self.package_status.update(value)
            for name, status in value.items():
                if hasattr(self, "package_status_vars"):
                    self.package_status_vars[name].set(status)
            self._update_nav()
            self._log("Package check completed.")

        self._run_worker("Checking R packages...", self._check_package_status_worker, done)

    def _install_missing_packages_worker(self):
        statuses = self._check_package_status_worker()
        env = os.environ.copy()
        env["R_LIBS_USER"] = str(self.package_library_path.resolve())
        env["R_INSTALL_STAGED"] = "false"

        package_order = getattr(self.step_frame, "R_LOCAL_PACKAGE_ORDER", None)
        if not package_order:
            package_order = self.step_frame.R_REQUIRED_PACKAGES
        for package_name in package_order:
            check_cmd = self._r_eval_command(self._package_check_expression(package_name))
            check_result = subprocess.run(
                check_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if check_result.returncode == 0:
                self._log(f"Package {package_name} is already installed.")
                if package_name in statuses:
                    statuses[package_name] = "installed"
                continue

            try:
                expression = self._package_install_expression(package_name)
            except Exception as exc:
                self._log(f"Package {package_name} cannot be installed from the local bundle: {exc}")
                if package_name in statuses:
                    statuses[package_name] = "failed"
                continue

            cmd = self._r_eval_command(expression)
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                input="n\n",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._log_process_result(f"Install package {package_name}", cmd, result)
            installed_check = subprocess.run(
                self._r_eval_command(self._package_check_expression(package_name)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            package_status = "installed" if result.returncode == 0 and installed_check.returncode == 0 else "failed"
            if package_name in statuses:
                statuses[package_name] = package_status
            if package_status == "failed":
                self._log(
                    f"Package {package_name} failed its post-installation check; "
                    "dependent packages may also fail."
                )
        return statuses

    def _install_missing_packages(self):
        if self.current_step != 1:
            return

        def done(value, error):
            if error:
                self._log(f"Package installation failed: {error}")
                messagebox.showerror("R Packages", f"Could not install packages.\n{error}", parent=self)
                return
            self.package_status.update(value)
            for name, status in value.items():
                if hasattr(self, "package_status_vars"):
                    self.package_status_vars[name].set(status)
            self._update_nav()
            if self._all_packages_ready():
                self._log("All required R packages are installed.")
                messagebox.showinfo("R Packages", "All required packages are installed.", parent=self)
            else:
                self._log("Some R packages failed to install. See the setup log for details.")
                messagebox.showerror(
                    "R Packages",
                    f"Some packages failed to install.\n\nFull log:\n{self.log_path}",
                    parent=self,
                )

        self._run_worker("Installing missing R packages...", self._install_missing_packages_worker, done)

    def _all_packages_ready(self):
        return all(self.package_status.get(name) == "installed" for name in self.step_frame.R_REQUIRED_PACKAGES)

    # Unified setup page. These methods intentionally keep the setup workflow
    # in one frame: detect everything, show every requirement, then install
    # the missing components with one action.
    def _build_styles(self):
        self.style = ttk.Style(self)
        self.style.configure("WizardTitle.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("WizardSubtitle.TLabel", foreground=COLORS.muted_text)
        self.style.configure("WizardStatus.TLabel", font=("Segoe UI", 11, "bold"))
        self.style.configure("WizardSuccess.TLabel", foreground=COLORS.success, font=("Segoe UI", 10, "bold"))
        self.style.configure("WizardMissing.TLabel", foreground=COLORS.danger, font=("Segoe UI", 10, "bold"))
        self.style.configure("WizardNeutral.TLabel", foreground=COLORS.muted_text, font=("Segoe UI", 10, "bold"))
        self.style.configure("WizardPrimary.TButton", padding=(12, 6))
        self.style.configure("WizardClose.TButton", padding=(8, 3))
        self.style.configure("Wizard.Treeview", rowheight=24, font=("Segoe UI", 9))
        self.style.configure("Wizard.Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build_shell(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(3, minsize=34)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)
        title_area = ttk.Frame(header)
        title_area.grid(row=0, column=0, sticky="ew")
        ttk.Label(title_area, text="R environment setup", style="WizardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title_area,
            text=f"Check R {self.step_frame.R_REQUIRED_VERSION} and install all missing requirements automatically.",
            style="WizardSubtitle.TLabel",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        self.content = ttk.Frame(root)
        self.content.grid(row=1, column=0, sticky="nsew")

        log_frame = ttk.LabelFrame(root, text="Setup log")
        log_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.log_text = tk.Text(log_frame, height=4, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        footer = ttk.Frame(root)
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)
        footer_actions = ttk.Frame(footer)
        footer_actions.grid(row=0, column=0, sticky="w")
        self.install_components_button = action_button(
            footer_actions,
            self,
            "Install missing R and packages",
            self._install_missing_components,
            "package",
            style="AIDaS.PrimaryAction.TButton",
        )
        self.install_components_button.pack(side="left", padx=(0, 6))
        self.check_packages_button = action_button(
            footer_actions,
            self,
            "Check packages",
            self._check_packages,
            "refresh",
        )
        self.check_packages_button.pack(side="left")
        self.setup_complete_label = ttk.Label(
            footer_actions,
            text="Setup complete",
            style="WizardSuccess.TLabel",
        )
        self.close_button = action_button(
            footer,
            self,
            "Close",
            self._close,
            "cancel",
        )
        self.close_button.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self._log("R setup opened.")

    def _set_busy(self, busy, text=None):
        self.busy = bool(busy)
        self._set_content_buttons("disabled" if busy else "normal")
        if text:
            self._log(text)
        self._update_nav()

    def _update_nav(self):
        self._refresh_status_display()

    def _render_step(self):
        self._render_page()

    def _render_page(self):
        self._clear_content()

        ttk.Label(self.content, text="Installation status", style="WizardStatus.TLabel").pack(anchor="w")
        self.overall_status_var = tk.StringVar()
        self.overall_status_label = ttk.Label(
            self.content,
            textvariable=self.overall_status_var,
            style="WizardStatus.TLabel",
            wraplength=760,
            justify="left",
        )
        self.overall_status_label.pack(anchor="w", pady=(4, 10))

        r_frame = ttk.LabelFrame(self.content, text=f"R {self.step_frame.R_REQUIRED_VERSION}")
        r_frame.pack(fill="x", pady=(0, 8))
        r_frame.columnconfigure(0, weight=1)
        self.r_status_var = tk.StringVar()
        self.r_status_label = ttk.Label(r_frame, textvariable=self.r_status_var, wraplength=760, justify="left")
        self.r_status_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))
        r_actions = ttk.Frame(r_frame)
        r_actions.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))
        action_button(
            r_actions,
            self,
            "Check again",
            self._check_all,
            "refresh",
        ).pack(side="left", padx=(0, 6))
        action_button(
            r_actions,
            self,
            "Find R locally…",
            self._locate_rscript,
            "opened_folder",
        ).pack(side="left")
        self.r_auto_install_button = action_button(
            r_actions,
            self,
            f"Download and install R {self.step_frame.R_REQUIRED_VERSION}",
            lambda: self._download_and_install_r(continue_to_packages=True),
            "package",
            style="AIDaS.PrimaryAction.TButton",
        )
        self.r_auto_install_button.pack(side="left", padx=(6, 0))

        package_frame = ttk.LabelFrame(self.content, text="Required R packages")
        package_frame.pack(fill="x", expand=False)
        package_header = ttk.Frame(package_frame)
        package_header.pack(fill="x", padx=10, pady=(8, 6))
        package_header.columnconfigure(0, weight=1)
        self.package_summary_var = tk.StringVar()
        ttk.Label(
            package_header,
            textvariable=self.package_summary_var,
            style="WizardStatus.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            package_header,
            text="Managed by AIDaS",
            style="AIDaS.Muted.TLabel",
        ).grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Label(
            package_frame,
            text=f"Local library: {self.package_library_path}",
            style="AIDaS.Muted.TLabel",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 6))

        self.package_status_holder = ttk.Frame(package_frame)
        self.package_status_holder.pack(fill="x", expand=False, padx=10, pady=(0, 8))
        self.package_status_holder.columnconfigure(0, weight=1)
        self.package_tree = ttk.Treeview(
            self.package_status_holder,
            columns=("package", "role", "status"),
            show="headings",
            height=5,
            selectmode="none",
            style="Wizard.Treeview",
        )
        package_scroll = ttk.Scrollbar(
            self.package_status_holder,
            orient="vertical",
            command=self.package_tree.yview,
        )
        self.package_tree.configure(yscrollcommand=package_scroll.set)
        self.package_tree.heading("package", text="Package")
        self.package_tree.heading("role", text="Role")
        self.package_tree.heading("status", text="Status")
        self.package_tree.column("package", width=220, minwidth=130, stretch=True, anchor="w")
        self.package_tree.column("role", width=110, minwidth=90, stretch=False, anchor="w")
        self.package_tree.column("status", width=110, minwidth=95, stretch=False, anchor="w")
        self.package_tree.tag_configure("installed", foreground=COLORS.success)
        self.package_tree.tag_configure("missing", foreground=COLORS.danger)
        self.package_tree.tag_configure("neutral", foreground=COLORS.muted_text)
        self.package_tree.grid(row=0, column=0, sticky="ew")
        package_scroll.grid(row=0, column=1, sticky="ns")
        self.package_tree_items = {}
        required_packages = set(self.step_frame.R_REQUIRED_PACKAGES)
        display_order = [name for name in self.setup_package_names if name in required_packages]
        display_order.extend(name for name in self.setup_package_names if name not in required_packages)
        for package_name in display_order:
            role = "Required" if package_name in required_packages else "Dependency"
            item = self.package_tree.insert("", "end", values=(package_name, role, "Checking..."))
            self.package_tree_items[package_name] = item
        self._refresh_status_display()
        if self.rscript_path is not None:
            self.after(100, self._check_packages)

    @staticmethod
    def _status_text(status, r_available=True):
        if not r_available:
            return "Waiting for R", "WizardNeutral.TLabel"
        if status == "installed":
            return "Installed", "WizardSuccess.TLabel"
        if status == "missing":
            return "Missing", "WizardMissing.TLabel"
        if status == "failed":
            return "Failed", "WizardMissing.TLabel"
        return "Checking...", "WizardNeutral.TLabel"

    def _refresh_status_display(self):
        if not hasattr(self, "overall_status_var"):
            return

        if self.rscript_path is None:
            self.overall_status_var.set(f"Missing: R {self.step_frame.R_REQUIRED_VERSION}")
            self.overall_status_label.configure(style="WizardMissing.TLabel")
            self.r_status_var.set(f"Missing — R {self.step_frame.R_REQUIRED_VERSION} was not found.")
            self.r_status_label.configure(style="WizardMissing.TLabel")
        else:
            self.r_status_var.set(f"Detected R {self.step_frame.R_REQUIRED_VERSION}\n{self.rscript_path}")
            self.r_status_label.configure(style="WizardSuccess.TLabel")
            pending = any(self.package_status.get(name) == "pending" for name in self.setup_package_names)
            missing = [
                name
                for name in self.setup_package_names
                if self.package_status.get(name) in {"missing", "failed"}
            ]
            if pending:
                self.overall_status_var.set("Checking R packages...")
                self.overall_status_label.configure(style="WizardNeutral.TLabel")
            elif missing:
                package_word = "package" if len(missing) == 1 else "packages"
                self.overall_status_var.set(f"Action needed — {len(missing)} {package_word} missing or failed.")
                self.overall_status_label.configure(style="WizardMissing.TLabel")
            else:
                self.overall_status_var.set(
                    f"Pass — R {self.step_frame.R_REQUIRED_VERSION} and all required packages are ready."
                )
                self.overall_status_label.configure(style="WizardSuccess.TLabel")

        installed_count = sum(
            self.package_status.get(name) == "installed" for name in self.setup_package_names
        )
        problem_count = sum(
            self.package_status.get(name) in {"missing", "failed"} for name in self.setup_package_names
        )
        if self.rscript_path is None:
            self.package_summary_var.set(f"{len(self.setup_package_names)} packages — waiting for R")
        elif problem_count:
            self.package_summary_var.set(
                f"{installed_count} installed · {problem_count} need attention"
            )
        elif installed_count == len(self.setup_package_names):
            self.package_summary_var.set(f"All {installed_count} packages installed")
        else:
            self.package_summary_var.set(f"Checking {len(self.setup_package_names)} packages...")

        for package_name in self.setup_package_names:
            text, _style = self._status_text(
                self.package_status.get(package_name, "pending"),
                r_available=self.rscript_path is not None,
            )
            status = self.package_status.get(package_name, "pending")
            if self.rscript_path is None or status == "pending":
                tag = "neutral"
            elif status == "installed":
                tag = "installed"
            else:
                tag = "missing"
            item = self.package_tree_items[package_name]
            values = self.package_tree.item(item, "values")
            role = values[1] if len(values) > 1 else "Dependency"
            self.package_tree.item(item, values=(package_name, role, text), tags=(tag,))

        ready = self._all_packages_ready()
        if ready:
            if self.install_components_button.winfo_manager():
                self.install_components_button.pack_forget()
            if not self.setup_complete_label.winfo_manager():
                self.setup_complete_label.pack(
                    side="left",
                    padx=(0, 12),
                    before=self.check_packages_button,
                )
        elif self.rscript_path is None:
            if self.setup_complete_label.winfo_manager():
                self.setup_complete_label.pack_forget()
            if self.install_components_button.winfo_manager():
                self.install_components_button.pack_forget()
        else:
            install_text = "Install missing packages"
        if not ready and self.rscript_path is not None:
            if self.setup_complete_label.winfo_manager():
                self.setup_complete_label.pack_forget()
            if not self.install_components_button.winfo_manager():
                self.install_components_button.pack(
                    side="left",
                    padx=(0, 6),
                    before=self.check_packages_button,
                )
            self.install_components_button.configure(
                text=install_text,
                state="disabled" if self.busy else "normal",
            )
        self.check_packages_button.configure(
            state="disabled" if self.busy or self.rscript_path is None else "normal"
        )
        self.r_auto_install_button.configure(
            state="disabled" if self.busy or self.rscript_path is not None else "normal"
        )
        self.close_button.configure(text="Close", state="disabled" if self.busy else "normal")

    def _check_all(self):
        if self.busy:
            return
        self._detect_rscript()

    def _install_missing_components(self):
        if self.busy:
            return
        if self.rscript_path is None:
            self._download_and_install_r(continue_to_packages=True)
        else:
            self._install_missing_packages()

    def _check_packages(self):
        if self.rscript_path is None or self.busy:
            return

        def done(value, error):
            if error:
                self._log(f"Package check failed: {error}")
                messagebox.showerror("R Packages", f"Could not check packages.\n{error}", parent=self)
                return
            self.package_status.update(value)
            self._update_nav()
            self._log("Package check completed.")

        self._run_worker("Checking R packages...", self._check_package_status_worker, done)

    def _install_missing_packages(self):
        if self.rscript_path is None or self.busy:
            return

        def done(value, error):
            if error:
                self._log(f"Package installation failed: {error}")
                messagebox.showerror("R Packages", f"Could not install packages.\n{error}", parent=self)
                return
            self.package_status.update(value)
            self._update_nav()
            if self._all_packages_ready():
                self._log("All bundled R packages are installed.")
                messagebox.showinfo("R Packages", "All packages are installed.", parent=self)
            else:
                self._log("Some R packages failed to install. See the setup log for details.")
                messagebox.showerror(
                    "R Packages",
                    f"Some packages failed to install.\n\nFull log:\n{self.log_path}",
                    parent=self,
                )

        self._run_worker("Installing missing R packages...", self._install_missing_packages_worker, done)

    def _check_package_status_worker(self):
        if self.rscript_path is None:
            raise RuntimeError("Rscript is not selected.")
        self.package_library_path.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["R_LIBS_USER"] = str(self.package_library_path.resolve())
        statuses = {}
        for package_name in self.setup_package_names:
            expression = self._package_check_expression(package_name)
            cmd = self._r_eval_command(expression)
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._log_process_result(f"Check package {package_name}", cmd, result)
            statuses[package_name] = "installed" if result.returncode == 0 else "missing"
        return statuses

    def _all_packages_ready(self):
        return self.rscript_path is not None and all(
            self.package_status.get(name) == "installed" for name in self.setup_package_names
        )

    def _close(self):
        if self.busy:
            return
        if self._all_packages_ready():
            self._finish()
        else:
            self._cancel()

    def _cancel(self):
        if self.busy:
            return
        self.cancelled = True
        self.result = None
        self._dismiss(render_previous=True)

    def _finish(self):
        self.cancelled = False
        self.result = Path(self.rscript_path) if self.rscript_path is not None else None
        if self.result is not None:
            self.step_frame.r_package_library_path = str(self.package_library_path)
            if self.step_frame.preferences is not None:
                self.step_frame.preferences.set("rscript_path", str(self.result))
                self.step_frame.preferences.set("r_package_library_path", str(self.package_library_path))
        callback = self.on_finish
        result = self.result
        self._dismiss(render_previous=callback is None)
        if callback is not None:
            self.step_frame.after(0, lambda: callback(result))

    def _detect_rscript(self, schedule_package_check=True):
        self.rscript_path = self.step_frame._resolve_rscript_executable()
        self.package_status = {name: "pending" for name in self.setup_package_names}
        self._log(f"Rscript detection: {self.rscript_path or 'not found'}")
        self._refresh_status_display()
        if self.rscript_path is not None and schedule_package_check:
            self.after(100, self._check_packages)

    def _locate_rscript(self):
        selected = filedialog.askopenfilename(
            title=f"Select R {self.step_frame.R_REQUIRED_VERSION} program",
            initialdir=r"C:\Program Files\R" if os.name == "nt" else (self.step_frame.current_sdb_dir or None),
            filetypes=[
                ("R program", "*.exe"),
                ("Executable files", "*.exe"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        rscript = self.step_frame._normalize_r_executable(Path(selected))
        if rscript is None:
            messagebox.showerror(
                "Select R program",
                f"Please select Rscript.exe, R.exe, or Rterm.exe from the R "
                f"{self.step_frame.R_REQUIRED_VERSION} installation.",
                parent=self,
            )
            return
        version = self.step_frame._r_version_for_executable(rscript)
        if version != self.step_frame.R_REQUIRED_VERSION:
            detected = version or "unknown"
            messagebox.showerror(
                "Unsupported R version",
                f"This Step 3 setup only accepts R {self.step_frame.R_REQUIRED_VERSION}. "
                f"The selected program reported R {detected}.",
                parent=self,
            )
            self._log(f"Rejected manually selected R program {rscript} (version {detected}).")
            return
        self.rscript_path = rscript
        self.package_status = {name: "pending" for name in self.setup_package_names}
        if self.step_frame.preferences is not None:
            self.step_frame.preferences.set("rscript_path", str(rscript))
        self._log(f"Selected R {self.step_frame.R_REQUIRED_VERSION}: {rscript}")
        self._refresh_status_display()
        self.after(100, self._check_packages)

    def _download_and_install_r(self, continue_to_packages=True):
        if self.busy:
            return
        if not self.installer_url:
            self.installer_name = self.step_frame.R_INSTALLER_NAME
            self.installer_url = self.step_frame.R_DOWNLOAD_PAGE + self.installer_name
            self.installer_path = self._r_installer_cache_path()
            self._log(f"R {self.step_frame.R_REQUIRED_VERSION} installer selected: {self.installer_url}")
        self.installer_path = self.installer_path or self._r_installer_cache_path()
        if self.installer_path.exists():
            install_existing = messagebox.askyesno(
                f"Install R {self.step_frame.R_REQUIRED_VERSION}",
                "The R installer is already downloaded in the app's local files.\n\nInstall it now?",
                parent=self,
            )
            if install_existing:
                self._run_downloaded_installer(continue_to_packages=continue_to_packages)
            return

        def worker():
            self.installer_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.installer_url, self.installer_path)
            return self.installer_path

        def done(value, error):
            if error:
                self._log(f"R installer download failed: {error}")
                messagebox.showerror("R Setup", f"Could not download R.\n{error}", parent=self)
                return
            self._log(f"Downloaded R installer: {value}")
            install_now = messagebox.askyesno(
                f"Install R {self.step_frame.R_REQUIRED_VERSION}",
                f"R {self.step_frame.R_REQUIRED_VERSION} was downloaded successfully.\n\n"
                "Do you want to install it now?",
                parent=self,
            )
            if install_now:
                self._run_downloaded_installer(continue_to_packages=continue_to_packages)
            else:
                messagebox.showinfo(
                    "R Setup",
                    "The installer was downloaded. You can install it later from this screen.",
                    parent=self,
                )

        self._run_worker("Downloading R installer...", worker, done)

    def _run_downloaded_installer(self, continue_to_packages=True):
        installer_path = self.installer_path
        if not installer_path or not installer_path.is_file():
            messagebox.showwarning("R Setup", "Download the R installer first.", parent=self)
            return

        def worker():
            return subprocess.run([str(installer_path)], check=False).returncode

        def done(value, error):
            if error:
                self._log(f"R installer could not be started: {error}")
                messagebox.showerror("R Setup", f"Could not run the R installer.\n{error}", parent=self)
                return
            self._log(f"R installer closed with return code {value}.")
            self._detect_rscript(schedule_package_check=False)
            if self.rscript_path is not None:
                messagebox.showinfo(
                    "R Setup",
                    f"R {self.step_frame.R_REQUIRED_VERSION} is ready. Missing packages will now be installed.",
                    parent=self,
                )
                if continue_to_packages:
                    self.after(200, self._install_missing_packages)
            else:
                messagebox.showwarning(
                    "R Setup",
                    f"AIDaS still cannot find R {self.step_frame.R_REQUIRED_VERSION}. "
                    "Use Check Again or Search Locally.",
                    parent=self,
                )

        self._run_worker("Running R installer. Complete the installer window to continue.", worker, done)


class RBatchSelectionTable(ttk.Frame):
    """Fast folder table for Step 3 batch R script selection."""

    COLUMNS = ("folder", "status", "inputs")
    FOLDER_MIN_WIDTH = 120
    SELECTION_STATUS_WIDTH_VALUES = (
        "Ready",
        "Skipped: RData exists",
    )
    RUN_STATUS_WIDTH_VALUES = (
        "Queued",
        "Validating",
        "Running main R script",
        "Starting R script",
        "Reading R input configuration",
        "Loading Analyze volumes in R",
        "Calculating vertex",
        "Reading RPE line",
        "Fitting RPE spline",
        "Computing apparent angles",
        "Building perpendicular sampling lines",
        "Flattening marker image",
        "Flattening DARK slices",
        "Flattening LIGHT slices",
        "Converting flattened data to raw scale",
        "Building grand mean image",
        "Aligning retina profiles",
        "Exporting R arrays",
        "Drawing borders",
        "Spatially normalizing main retina",
        "Spatially normalizing fovea",
        "Drawing borders and writing outputs",
        "R processing complete",
        "Waiting for output R script",
        "Running output R script",
        "Completed",
        "Cancelled",
        "Timed out",
        "Failed",
    )
    # Kept as the complete public set for callers/tests that need to verify that
    # every possible status has a sizing sample. Each table instance receives
    # the narrower context-specific subset below.
    STATUS_WIDTH_VALUES = SELECTION_STATUS_WIDTH_VALUES + RUN_STATUS_WIDTH_VALUES
    MAX_PROGRESS_VALUE = "100%"
    # The clam Treeview heading layout reserves substantially more horizontal
    # space than its text alone: 16 px of configured padding plus the border and
    # separator hit region at the cell edge. Keep heading sizing independent
    # from body-cell padding so the trailing Progress glyph is never clipped.
    HEADING_WIDTH_PADDING = 30

    def __init__(self, parent, *, status_width_values=None):
        super().__init__(parent)
        self.rows = []
        self._row_by_iid = {}
        self._xscroll_visible = False
        self._xscroll_after_id = None
        self._column_resize_active = False
        self._resize_start_widths = None
        self._manual_column_widths = None
        self._horizontal_chrome_width = None
        self._status_width_values = tuple(
            status_width_values or self.SELECTION_STATUS_WIDTH_VALUES
        )
        self._checkbox_images = self._make_checkbox_images()
        self._tree_font = tkfont.nametofont("TkDefaultFont")
        self._heading_font = self._tree_font.copy()
        self._heading_font.configure(weight="bold")

        self._tree_style = "Step3Batch.Treeview"
        self._style = ttk.Style(self)
        try:
            self._style.configure(self._tree_style, indent=0)
        except tk.TclError:
            pass

        self.tree = ttk.Treeview(
            self,
            columns=self.COLUMNS,
            show=("tree", "headings"),
            selectmode="none",
            style=self._tree_style,
        )
        self.yscroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.yscroll.set, xscrollcommand=self._on_xscroll)

        self.tree.heading(
            "#0",
            text="",
            image=self._checkbox_images["unchecked"],
            anchor="center",
            command=self._toggle_all_ready,
        )
        self.tree.heading("folder", text="Folder")
        self.tree.heading("status", text="Status")
        self.tree.heading("inputs", text="Inputs")

        self.tree.column("#0", width=40, minwidth=40, stretch=False, anchor="center")
        self.tree.column(
            "folder",
            width=520,
            minwidth=self.FOLDER_MIN_WIDTH,
            stretch=False,
            anchor="w",
        )
        self.tree.column("status", width=360, minwidth=120, stretch=False, anchor="w")
        self.tree.column("inputs", width=72, minwidth=60, stretch=False, anchor="center")

        self.tree.tag_configure("locked", foreground="#6b7280")
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.yscroll.grid(row=0, column=1, sticky="ns")
        self.xscroll.grid(row=1, column=0, sticky="ew")
        self.xscroll.grid_remove()
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.bind("<Button-1>", self._on_click, add="+")
        self.tree.bind("<Configure>", self._on_tree_configure, add="+")
        self.tree.bind("<ButtonPress-1>", self._on_tree_button_press, add="+")
        self.tree.bind("<ButtonRelease-1>", self._on_tree_button_release, add="+")

    def _make_checkbox_images(self):
        images = {
            "checked": tk.PhotoImage(width=16, height=16),
            "unchecked": tk.PhotoImage(width=16, height=16),
            "locked": tk.PhotoImage(width=16, height=16),
        }
        for image in images.values():
            image.put("#ffffff", to=(0, 0, 16, 16))
            image.put("#6b7280", to=(2, 2, 14, 3))
            image.put("#6b7280", to=(2, 13, 14, 14))
            image.put("#6b7280", to=(2, 2, 3, 14))
            image.put("#6b7280", to=(13, 2, 14, 14))

        checked = images["checked"]
        for x, y in ((4, 8), (5, 9), (6, 10), (7, 9), (8, 8), (9, 7), (10, 6), (11, 5)):
            checked.put("#111827", to=(x, y, x + 1, y + 1))
            checked.put("#111827", to=(x, y + 1, x + 1, y + 2))

        locked = images["locked"]
        locked.put("#e5e7eb", to=(3, 3, 13, 13))
        locked.put("#9ca3af", to=(5, 7, 11, 9))
        return images

    def set_rows(self, rows):
        self.rows = list(rows or [])
        self._row_by_iid = {}
        self._column_resize_active = False
        self._resize_start_widths = None
        self._manual_column_widths = None
        self._set_xscroll_visible(False)
        self.tree.delete(*self.tree.get_children(""))

        if not self.rows:
            self.tree.insert(
                "",
                "end",
                text="",
                values=("No folders with complete Step 3 inputs were found.", "", ""),
                tags=("locked",),
            )
            self._fit_columns_to_content()
            self._refresh_header_checkbox()
            return

        for index, row in enumerate(self.rows):
            iid = str(index)
            self._row_by_iid[iid] = row
            self.tree.insert(
                "",
                "end",
                iid=iid,
                text="",
                image=self._image_for_row(row),
                values=self._values_for_row(row),
                tags=("locked",) if row.get("locked") else (),
            )
        self._fit_columns_to_content()
        self._refresh_header_checkbox()

    def _image_for_row(self, row):
        if row.get("locked"):
            return self._checkbox_images["locked"]
        if row.get("include"):
            return self._checkbox_images["checked"]
        return self._checkbox_images["unchecked"]

    def _values_for_row(self, row):
        values = row.get("values") or {}
        return (
            values.get("folder", ""),
            values.get("status", ""),
            values.get("inputs", ""),
        )

    def _measure_text(self, text, *, heading=False, padding=18):
        font = self._heading_font if heading else self._tree_font
        return int(font.measure(str(text or ""))) + int(padding)

    def _fit_columns_to_content(self):
        if self._manual_column_widths is not None:
            self._apply_manual_columns_to_view()
            self._schedule_xscroll_visibility_update()
            return
        try:
            status_heading = self.tree.heading("status", "text")
            inputs_heading = self.tree.heading("inputs", "text")
        except tk.TclError:
            return
        widths = {
            "status": max(
                self._measure_text(
                    status_heading,
                    heading=True,
                    padding=self.HEADING_WIDTH_PADDING,
                ),
                *(self._measure_text(value) for value in self._status_width_values),
            ),
            "inputs": max(
                self._measure_text(
                    inputs_heading,
                    heading=True,
                    padding=self.HEADING_WIDTH_PADDING,
                ),
                self._measure_text(self.MAX_PROGRESS_VALUE),
            ),
        }
        # Status and Inputs/Progress fit their widest valid or displayed values.
        # Folder paths are deliberately not measured: Folder owns the remaining
        # view width by default, and a user drag may create horizontal overflow.
        for row in self.rows:
            _folder, status, inputs = self._values_for_row(row)
            widths["status"] = max(widths["status"], self._measure_text(status))
            widths["inputs"] = max(widths["inputs"], self._measure_text(inputs))

        status_width = max(120, widths["status"])
        inputs_width = max(60, widths["inputs"])
        self.tree.column("status", width=status_width, minwidth=status_width)
        self.tree.column("inputs", width=inputs_width, minwidth=inputs_width)
        self._expand_folder_to_view()
        self._schedule_xscroll_visibility_update()

    def _on_tree_configure(self, _event=None):
        if self._manual_column_widths is None:
            self._fit_columns_to_content()
        else:
            self._apply_manual_columns_to_view()
        self._schedule_xscroll_visibility_update()

    def _on_tree_button_press(self, event):
        try:
            self._column_resize_active = self.tree.identify_region(event.x, event.y) == "separator"
            self._resize_start_widths = (
                self._current_column_widths()
                if self._column_resize_active
                else None
            )
        except tk.TclError:
            self._column_resize_active = False
            self._resize_start_widths = None

    def _on_tree_button_release(self, _event=None):
        was_resizing = self._column_resize_active
        self._column_resize_active = False
        if not was_resizing:
            self._schedule_xscroll_visibility_update()
            return
        try:
            self.after_idle(self._finish_column_resize)
        except tk.TclError:
            self._resize_start_widths = None

    def _current_column_widths(self):
        try:
            return {
                column: int(self.tree.column(column, "width"))
                for column in ("#0", *self.COLUMNS)
            }
        except tk.TclError:
            return None

    def _finish_column_resize(self):
        start_widths = self._resize_start_widths
        self._resize_start_widths = None
        current_widths = self._current_column_widths()
        if start_widths is not None and current_widths != start_widths:
            self._manual_column_widths = current_widths
            self._apply_manual_columns_to_view()
        self._schedule_xscroll_visibility_update()

    def _apply_manual_columns_to_view(self):
        widths = self._manual_column_widths
        if widths is None:
            return
        try:
            view_width = max(1, int(self.tree.winfo_width()))
            for column in ("#0", "status", "inputs"):
                self.tree.column(column, width=widths[column])
            chrome_width = self._tree_horizontal_chrome_width()
            available_folder_width = (
                view_width
                - widths["#0"]
                - widths["status"]
                - widths["inputs"]
                - chrome_width
            )
            folder_width = max(
                self.FOLDER_MIN_WIDTH,
                widths["folder"],
                available_folder_width,
            )
            self.tree.column("folder", width=folder_width)
        except (KeyError, tk.TclError):
            return

    def _tree_horizontal_chrome_width(self):
        """Return cached theme border/padding outside the displayed columns."""
        cached_width = getattr(self, "_horizontal_chrome_width", None)
        if cached_width is not None:
            return cached_width
        try:
            column_width = sum(
                int(self.tree.column(column, "width"))
                for column in ("#0", *self.COLUMNS)
            )
            measured_width = max(0, int(self.tree.winfo_reqwidth()) - column_width)
        except tk.TclError:
            return 0
        # winfo_reqwidth() can be clamped once the columns overflow. Calibrate
        # only while the table is in its known auto-fit state, then retain the
        # theme/DPI-specific value for all later manual layouts.
        if self._manual_column_widths is None:
            self._horizontal_chrome_width = measured_width
        return measured_width

    def _on_xscroll(self, first, last):
        """Update only the thumb; geometry is decided from integer widths."""
        self.xscroll.set(first, last)

    def _manual_columns_overflow_view(self):
        """Return whether a user-sized column layout exceeds the usable view."""
        if self._manual_column_widths is None:
            return False
        try:
            displayed_width = sum(
                int(self.tree.column(column, "width"))
                for column in ("#0", *self.COLUMNS)
            )
            usable_width = max(
                1,
                int(self.tree.winfo_width()) - self._tree_horizontal_chrome_width(),
            )
        except tk.TclError:
            return False
        return displayed_width > usable_width

    def _set_xscroll_visible(self, visible):
        visible = bool(visible)
        if visible == self._xscroll_visible:
            return
        self._xscroll_visible = visible
        try:
            if visible:
                self.xscroll.grid(row=1, column=0, sticky="ew")
            else:
                self.xscroll.grid_remove()
                self.tree.xview_moveto(0.0)
        except tk.TclError:
            pass

    def _schedule_xscroll_visibility_update(self, _event=None):
        if self._xscroll_after_id is not None:
            return
        try:
            self._xscroll_after_id = self.after_idle(self._sync_xscroll_visibility)
        except tk.TclError:
            self._xscroll_after_id = None

    def _sync_xscroll_visibility(self):
        self._xscroll_after_id = None
        # Long path text never controls the automatic geometry: it is clipped
        # inside Folder and the bar stays hidden. A real separator drag stores
        # a manual baseline; only that deterministic integer-width overflow may
        # reveal the scrollbar.
        self._set_xscroll_visible(self._manual_columns_overflow_view())

    def _expand_folder_to_view(self):
        if self._manual_column_widths is not None:
            return
        try:
            view_width = max(1, int(self.tree.winfo_width()))
            checkbox_width = int(self.tree.column("#0", "width"))
            status_width = int(self.tree.column("status", "width"))
            inputs_width = int(self.tree.column("inputs", "width"))
        except tk.TclError:
            return

        non_folder_width = checkbox_width + status_width + inputs_width
        chrome_width = self._tree_horizontal_chrome_width()
        desired_folder_width = max(
            self.FOLDER_MIN_WIDTH,
            view_width - non_folder_width - chrome_width,
        )
        try:
            self.tree.column("folder", width=desired_folder_width)
        except tk.TclError:
            pass

    def _refresh_row(self, iid, row):
        try:
            self.tree.item(iid, image=self._image_for_row(row), values=self._values_for_row(row))
        except tk.TclError:
            pass

    def _refresh_header_checkbox(self):
        ready_rows = [row for row in self.rows if not row.get("locked")]
        image = self._checkbox_images["unchecked"]
        if ready_rows and all(bool(row.get("include")) for row in ready_rows):
            image = self._checkbox_images["checked"]
        try:
            self.tree.heading("#0", image=image)
        except tk.TclError:
            pass

    def _on_click(self, event):
        if self.tree.identify_region(event.x, event.y) not in {"cell", "tree"}:
            return None
        if self.tree.identify_column(event.x) != "#0":
            return None
        iid = self.tree.identify_row(event.y)
        row = self._row_by_iid.get(iid)
        if not row or row.get("locked"):
            return "break"
        row["include"] = not bool(row.get("include"))
        self._refresh_row(iid, row)
        self._refresh_header_checkbox()
        return "break"

    def _toggle_all_ready(self):
        ready_rows = [row for row in self.rows if not row.get("locked")]
        if not ready_rows:
            return
        include = not all(bool(row.get("include")) for row in ready_rows)
        for iid, row in self._row_by_iid.items():
            if row.get("locked"):
                continue
            row["include"] = include
            self._refresh_row(iid, row)
        self._refresh_header_checkbox()

    def selected_rows(self):
        return [row for row in self.rows if row.get("include") and not row.get("locked")]


class RBatchSelectionPanel(ttk.Frame):
    """Embedded panel for selecting subfolders to run through the Step 3 R script."""

    def __init__(self, step_frame, parent, root_dir, folders=None):
        super().__init__(parent)
        self.step_frame = step_frame
        self.root_dir = Path(root_dir)
        self.input_folders = None if folders is None else tuple(Path(folder) for folder in folders)
        self.rows = []
        self.table = None

        self._build_ui()
        self._start_scan()

    def _build_ui(self):
        wrapper = ttk.Frame(self, padding=12)
        wrapper.pack(fill="both", expand=True)

        ttk.Label(wrapper, text="Batch R Script Processing", font=("", 12, "bold")).pack(anchor="w")
        if self.input_folders is None:
            instructions = (
                "AIDaS will search the selected folder and subfolders for Light.img and Light_MARKED.img. "
                "Folders containing existing RData are shown as skipped and will not be processed."
            )
        else:
            instructions = (
                "Review the nasal and temporal folders saved in Step 2, select the folders to process, "
                "and press Start to run the selected R scripts. Folders containing existing RData are skipped."
            )
        ttk.Label(
            wrapper,
            text=instructions,
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        top = ttk.Frame(wrapper)
        top.pack(fill="x", pady=(0, 8))
        self.summary_var = tk.StringVar(value=f"Scanning: {self.root_dir}")
        ttk.Label(top, textvariable=self.summary_var, wraplength=760, justify="left").pack(
            side="left",
            fill="x",
            expand=True,
        )
        self.more_label = ttk.Label(top, text="", style="AIDaS.Link.TLabel", cursor="hand2")
        self.more_label.pack(side="right", padx=(8, 0))
        self.more_tooltip = HoverToolTip(self.more_label, "")

        run_box = ttk.Frame(wrapper)
        run_box.pack(side="bottom", fill="x", pady=(10, 0))
        self.action_footer = run_box

        action_row = ttk.Frame(run_box)
        action_row.pack(side="bottom", fill="x")
        action_button(action_row, self, "Cancel", self._cancel, "cancel").pack(side="left")
        self.next_button = action_button(
            action_row,
            self,
            "Run selected folders",
            self._run_selected,
            "process",
            tooltip="Start the configured R scripts for the selected folders.",
            style="AIDaS.PrimaryAction.TButton",
        )
        self.next_button.pack(side="right")
        self.next_button.state(["disabled"])

        settings_row = ttk.Frame(run_box)
        settings_row.pack(side="bottom", fill="x", pady=(0, 8))
        ttk.Label(settings_row, text="Batch Size:").pack(side="left")
        max_workers = self._max_worker_count()
        self.workers_var = tk.IntVar(value=min(4, max_workers))
        self.workers_spin = ttk.Spinbox(
            settings_row,
            from_=1,
            to=max_workers,
            textvariable=self.workers_var,
            width=5,
        )
        self.workers_spin.pack(side="left", padx=(6, 12))
        self.worker_limit_var = tk.StringVar(value=self._worker_limit_text(max_workers))
        ttk.Label(settings_row, textvariable=self.worker_limit_var, style="AIDaS.Muted.TLabel").pack(side="left")
        ttk.Label(settings_row, text="Timeout/script (min):").pack(side="left", padx=(12, 0))
        self.timeout_var = tk.IntVar(value=self.step_frame.DEFAULT_R_SCRIPT_TIMEOUT_MINUTES)
        self.timeout_spin = ttk.Spinbox(
            settings_row,
            from_=1,
            to=10080,
            textvariable=self.timeout_var,
            width=7,
        )
        self.timeout_spin.pack(side="left", padx=(6, 12))
        self.workers_spin.configure(state="disabled")

        mode_row = ttk.Frame(run_box)
        mode_row.pack(side="bottom", fill="x", pady=(0, 8))
        self.execution_mode_footer = mode_row
        ttk.Label(mode_row, text="Second script across folders:").pack(side="left")
        self.output_mode_var = tk.StringVar(value=self.step_frame.R_OUTPUT_MODE_PARALLEL)
        parallel_radio = ttk.Radiobutton(
            mode_row,
            text="Parallel (default)",
            value=self.step_frame.R_OUTPUT_MODE_PARALLEL,
            variable=self.output_mode_var,
        )
        parallel_radio.pack(side="left", padx=(8, 0))
        sequential_radio = ttk.Radiobutton(
            mode_row,
            text="Sequential",
            value=self.step_frame.R_OUTPUT_MODE_SEQUENTIAL,
            variable=self.output_mode_var,
        )
        sequential_radio.pack(side="left", padx=(8, 0))
        self.output_mode_tooltips = (
            HoverToolTip(
                parallel_radio,
                "Each worker runs the first script and then the second script for its folder. "
                "Multiple folders run in parallel using Batch Size.",
            ),
            HoverToolTip(
                sequential_radio,
                "The first script runs in parallel using Batch Size. After it finishes for all folders, "
                "the second script runs for one folder at a time.",
            ),
        )

        # The table is the only vertically flexible region. Packing it after
        # the footer guarantees that resize pressure cannot cover the actions.
        self.table_host = ttk.Frame(wrapper)
        self.table_host.pack(side="top", fill="both", expand=True)
        self.scan_label = ttk.Label(
            self.table_host,
            text="Scanning folders...",
            anchor="center",
            justify="center",
        )
        self.scan_label.pack(fill="both", expand=True)

    def _max_worker_count(self, ready_count=None):
        cpu_limit = self.step_frame._r_worker_limit()
        if ready_count is None:
            return cpu_limit
        return max(1, min(int(ready_count) or 1, cpu_limit))

    def _worker_limit_text(self, max_workers):
        return f"Max: {max_workers} (1 CPU reserved)"

    def _start_scan(self):
        self.step_frame.status_var.set(f"Scanning subfolders under {self.root_dir}...")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        rows = []
        scanned = 0
        missing = 0
        try:
            if self.input_folders is None:
                folders, access_errors = walk_accessible_directories(self.root_dir)
            else:
                folders = list(self.input_folders)
                access_errors = []
            for folder in folders:
                try:
                    scanned += 1
                    input_paths = self.step_frame._find_input_paths(folder)
                    if any(input_paths.get(label) is None for label, *_rest in self.step_frame.REQUIRED_INPUTS):
                        missing += 1
                        continue
                    has_rdata = self.step_frame._folder_has_r_data(folder)
                except OSError as exc:
                    access_errors.append((folder, str(exc)))
                    continue
                status = "Skipped: RData exists" if has_rdata else "Ready"
                try:
                    folder_text = str(folder.relative_to(self.root_dir))
                    if folder_text == ".":
                        folder_text = str(self.root_dir)
                except ValueError:
                    folder_text = str(folder)
                rows.append(
                    {
                        "folder": folder,
                        "include": not has_rdata,
                        "locked": has_rdata,
                        "status": status,
                        "values": {
                            "folder": folder_text,
                            "status": status,
                            "inputs": str(len(self.step_frame.REQUIRED_INPUTS)),
                        },
                    }
                )
        except Exception as exc:
            self.after(0, lambda exc=exc: self._scan_failed(exc))
            return
        self.after(0, lambda: self._scan_done(rows, scanned, missing, access_errors))

    def _scan_failed(self, exc):
        if not self.winfo_exists():
            return
        self.summary_var.set("Scan failed. Move the mouse over More for details.")
        self.more_label.configure(text="More")
        self.more_tooltip.text = f"Could not scan folders.\n{exc}"
        self.step_frame.status_var.set("Batch scan failed.")
        try:
            self.next_button.state(["disabled"])
            self.workers_spin.configure(state="disabled")
        except tk.TclError:
            pass

    def _show_results_table(self, rows):
        for child in self.table_host.winfo_children():
            try:
                child.destroy()
            except tk.TclError:
                pass

        table = RBatchSelectionTable(self.table_host)
        table.set_rows(rows)
        table.pack(fill="both", expand=True)
        self.table = table

    def _scan_done(self, rows, scanned, missing, access_errors):
        if not self.winfo_exists():
            return
        self.rows = rows
        self._show_results_table(rows)
        ready = sum(1 for row in rows if not row["locked"])
        skipped = sum(1 for row in rows if row["locked"])
        summary = (
            f"Scanned {scanned} folders. Found {ready} ready folder(s), {skipped} skipped folder(s) with RData. "
            f"{missing} folder(s) did not contain both required Light inputs. "
            f"{len(access_errors)} inaccessible folder(s) skipped."
        )
        self.summary_var.set(summary)
        self.more_label.configure(text="More" if access_errors else "")
        self.more_tooltip.text = skipped_directories_warning(access_errors) if access_errors else ""
        max_workers = self._max_worker_count(ready)
        self.workers_spin.configure(to=max_workers)
        self.workers_var.set(min(4, max_workers))
        self.worker_limit_var.set(self._worker_limit_text(max_workers))
        self.step_frame.status_var.set("Batch scan complete. Select folders to process.")
        try:
            if ready:
                self.next_button.state(["!disabled"])
                self.workers_spin.configure(state="normal")
            else:
                self.next_button.state(["disabled"])
                self.workers_spin.configure(state="disabled")
        except tk.TclError:
            pass
    def _run_selected(self):
        if self.table is None:
            return
        folders = [row["folder"] for row in self.table.selected_rows()]
        if not folders:
            messagebox.showwarning("Batch Step 3", "Select at least one ready folder.", parent=self)
            return
        main_script_path = self.step_frame._selected_r_script_path("main")
        output_script_path = self.step_frame._selected_r_script_path("output")
        if main_script_path is None or not main_script_path.is_file():
            messagebox.showerror(
                "Batch Step 3",
                "Select an available main processing R script in Settings.",
                parent=self,
            )
            return
        if output_script_path is None or not output_script_path.is_file():
            messagebox.showerror(
                "Batch Step 3",
                "Select an available output R script in Settings.",
                parent=self,
            )
            return
        try:
            workers = max(1, int(self.workers_var.get()))
        except (TypeError, ValueError):
            workers = 1
        try:
            timeout_minutes = max(1, min(10080, int(self.timeout_var.get())))
        except (TypeError, ValueError):
            timeout_minutes = self.step_frame.DEFAULT_R_SCRIPT_TIMEOUT_MINUTES
        self.timeout_var.set(timeout_minutes)
        max_workers = self._max_worker_count(len(folders))
        workers = min(workers, max_workers)
        self.workers_var.set(workers)
        output_mode = self.step_frame._normalize_r_output_mode(self.output_mode_var.get())
        self.step_frame._start_batch_r_runs(
            folders,
            workers,
            main_script_path,
            output_script_path,
            timeout_minutes * 60,
            output_mode=output_mode,
        )

    def _cancel(self):
        self.step_frame._close_r_batch_panel(render_previous=True)


class RBatchRunPanel(ttk.Frame):
    """Embedded progress panel for concurrent folder-level R script runs."""

    def __init__(
        self,
        step_frame,
        parent,
        folders,
        workers,
        main_script_path,
        output_script_path,
        timeout_seconds,
        output_mode="parallel",
    ):
        super().__init__(parent)
        self.step_frame = step_frame
        self.folders = [Path(folder) for folder in folders]
        self.workers = workers
        self.main_script_path = Path(main_script_path)
        self.output_script_path = Path(output_script_path)
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.output_mode = self.step_frame._normalize_r_output_mode(output_mode)
        self.row_by_folder = {}
        self.step_states_by_folder = {}
        self.current_step_by_folder = {}
        self.stop_requested = False
        self.close_when_finished = False
        self._build_ui()

    def _build_ui(self):
        wrapper = ttk.Frame(self, padding=12)
        wrapper.pack(fill="both", expand=True)

        # Reserve the live summary and actions before adding either flexible
        # region.  Tk's packer allocates geometry in packing order; keeping the
        # footer first means the table shrinks and scrolls instead of clipping
        # Restart, Stop, or Close when the window height is constrained.
        output_parallelism = (
            f"up to {self.workers} in parallel as folders finish the first script"
            if self.output_mode == self.step_frame.R_OUTPUT_MODE_PARALLEL
            else "sequentially after all first scripts finish"
        )
        self.summary_var = tk.StringVar(
            value=(
                f"Running {len(self.folders)} folder(s). First script: up to {self.workers} in parallel; "
                f"second script: {output_parallelism}."
            )
        )
        summary_row = ttk.Frame(wrapper)
        summary_row.pack(side="bottom", fill="x", pady=(8, 0))
        self.action_footer = summary_row

        action_box = ttk.Frame(summary_row)
        action_box.pack(side="right", padx=(8, 0))
        self.action_box = action_box
        self.restart_button = action_button(
            action_box,
            self,
            "Restart",
            self._restart_batch,
            "refresh",
        )
        self.restart_button.pack(side="left")
        self.stop_button = ttk.Button(
            action_box,
            text="\u25a0  Stop",
            command=self._cancel_batch,
            style="AIDaS.DangerAction.TButton",
            padding=(11, 6),
        )
        self.stop_button.pack(side="left", padx=(6, 0))
        self.close_icon = load_color_close_icon(self)
        self.close_button = ttk.Button(
            action_box,
            text="Close",
            command=self._close,
            image=self.close_icon,
            compound="left",
            style="AIDaS.Action.TButton",
        )
        self.close_button.pack(side="left", padx=(6, 0))
        self.cancel_button = self.stop_button

        self.summary_label = ttk.Label(
            summary_row,
            textvariable=self.summary_var,
            wraplength=480,
            justify="left",
        )
        self.summary_label.pack(side="left", fill="x", expand=True)
        summary_row.bind("<Configure>", self._resize_summary_footer, add="+")

        ttk.Label(wrapper, text="Running Batch Step 3", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            wrapper,
            text=(
                f"Main script: {self.main_script_path.name}\n"
                f"Output script: {self.output_script_path.name}\n"
                f"Output scheduling: {output_parallelism}"
            ),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        # Step details have their own scrollbar and a bounded requested height.
        # Pack them above the already-reserved footer and before the table so
        # neither fixed region can be covered by the flexible Treeview.
        step_frame = ttk.LabelFrame(wrapper, text="Step progress")
        step_frame.pack(side="bottom", fill="x", pady=(8, 0))
        self.step_progress_frame = step_frame
        visible_step_rows = max(3, min(6, len(self.folders) + 2))
        self.step_text = tk.Text(
            step_frame,
            height=visible_step_rows,
            wrap="word",
            state="disabled",
        )
        step_scroll = ttk.Scrollbar(step_frame, orient="vertical", command=self.step_text.yview)
        self.step_text.configure(yscrollcommand=step_scroll.set)
        self.step_text.pack(side="left", fill="both", expand=True)
        step_scroll.pack(side="right", fill="y")

        rows = []
        for folder in self.folders:
            row = {
                "folder": folder,
                "include": False,
                "locked": True,
                "values": {
                    "folder": str(folder),
                    "status": "Queued",
                    "inputs": "0%",
                },
            }
            rows.append(row)
            self.row_by_folder[str(folder)] = row
            self.step_states_by_folder[str(folder)] = []
            self.current_step_by_folder[str(folder)] = None

        self.table_host = ttk.Frame(wrapper)
        self.table_host.pack(side="top", fill="both", expand=True)
        self.table = RBatchSelectionTable(
            self.table_host,
            status_width_values=RBatchSelectionTable.RUN_STATUS_WIDTH_VALUES,
        )
        self.table.tree.heading("inputs", text="Progress")
        self.table.pack(fill="both", expand=True)
        self.table.set_rows(rows)
        self._render_step_progress()

    def _resize_summary_footer(self, event=None):
        """Wrap summary text inside the width left after reserving action buttons."""
        try:
            total_width = int(event.width) if event is not None else int(self.action_footer.winfo_width())
            action_width = int(self.action_box.winfo_reqwidth())
            wraplength = max(120, total_width - action_width - 16)
            current = int(float(self.summary_label.cget("wraplength")))
            if current != wraplength:
                self.summary_label.configure(wraplength=wraplength)
        except (AttributeError, tk.TclError, TypeError, ValueError):
            return

    def update_folder(self, folder, status=None, progress=None):
        row = self.row_by_folder.get(str(folder))
        if row is None:
            return
        values = dict(row.get("values") or {})
        if status is not None:
            values["status"] = status
        if progress is not None:
            values["inputs"] = f"{int(max(0, min(100, float(progress))))}%"
        row["values"] = values
        for iid, candidate in self.table._row_by_iid.items():
            if candidate is row:
                self.table._refresh_row(iid, row)
                self.table._schedule_xscroll_visibility_update()
                break
        if status is not None:
            if status == "Completed":
                self._finish_current_step(folder, "Done")
                self._append_step(folder, "Completed", "Done")
            elif status in {"Failed", "Cancelled", "Timed out"}:
                self._finish_current_step(folder, status)
            else:
                self._start_step(folder, status)

    def set_summary(self, text):
        self.summary_var.set(text)

    def finish(self):
        self.stop_button.configure(state="disabled")
        self.restart_button.configure(state="normal")
        self.close_button.configure(state="normal")

    def _cancel_batch(self):
        if self.stop_requested:
            return
        confirmed = messagebox.askyesno(
            "Stop Batch Step 3",
            "Stop the batch run?\n\nActive R processes will be terminated and queued folders will be cancelled.",
            parent=self,
        )
        if not confirmed:
            return
        self._request_stop("Stopping active R processes and cancelling queued folders...")

    def _request_stop(self, summary):
        self.stop_requested = True
        self.stop_button.configure(state="disabled")
        self.restart_button.configure(state="disabled")
        self.summary_var.set(summary)
        self.step_frame._cancel_batch_r_runs()

    def _restart_batch(self):
        running = bool(self.step_frame._busy)
        message = (
            "Restart the batch run?\n\nThe active R processes will be stopped first. "
            "A clean run will start after they exit."
            if running
            else "Restart the batch run from the beginning?"
        )
        if not messagebox.askyesno("Restart Batch Step 3", message, parent=self):
            return
        self.restart_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.close_button.configure(state="disabled")
        self.summary_var.set("Preparing a clean batch restart...")
        self.step_frame._restart_batch_r_runs(
            self.folders,
            self.workers,
            self.main_script_path,
            self.output_script_path,
            self.timeout_seconds,
            self.output_mode,
        )

    def _close(self):
        if self.step_frame._busy:
            confirmed = messagebox.askyesno(
                "Close Batch Step 3",
                "The batch is still running. Stop it and close this panel after all R processes exit?",
                parent=self,
            )
            if not confirmed:
                return
            self.close_when_finished = True
            self.close_button.configure(state="disabled")
            if not self.stop_requested:
                self._request_stop("Stopping the batch before closing...")
            return
        self.step_frame._close_r_batch_run_panel(render_previous=True)

    def _append_step(self, folder, label, state):
        key = str(folder)
        entries = self.step_states_by_folder.setdefault(key, [])
        if entries and entries[-1]["label"] == label:
            entries[-1]["state"] = state
        else:
            entries.append({"label": label, "state": state})
        self._render_step_progress()

    def _finish_current_step(self, folder, state):
        key = str(folder)
        current = self.current_step_by_folder.get(key)
        if not current:
            return
        entries = self.step_states_by_folder.setdefault(key, [])
        for entry in reversed(entries):
            if entry["label"] == current:
                entry["state"] = state
                break
        self.current_step_by_folder[key] = None
        self._render_step_progress()

    def _start_step(self, folder, label):
        if label in {"Validating", "Running R script", "Starting R script"}:
            return
        key = str(folder)
        current = self.current_step_by_folder.get(key)
        if current == label:
            return
        if current:
            self._finish_current_step(folder, "Done")
        self.current_step_by_folder[key] = label
        self._append_step(folder, label, "Running")

    def _render_step_progress(self):
        if not hasattr(self, "step_text"):
            return
        lines = []
        for folder in self.folders:
            entries = self.step_states_by_folder.get(str(folder), [])
            if not entries:
                lines.append("Queued...")
                continue
            for entry in entries:
                suffix = "..." if entry["state"] == "Running" else f"... {entry['state']}"
                lines.append(f"{entry['label']}{suffix}")
        self.step_text.configure(state="normal")
        self.step_text.delete("1.0", "end")
        self.step_text.insert("end", "\n".join(lines))
        if lines:
            self.step_text.insert("end", "\n")
        self.step_text.see("end")
        self.step_text.configure(state="disabled")

    def log(self, text):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {text}"
        try:
            with (app_log_dir() / "step3_batch_activity.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


class Step3Frame(SidebarStepFrame):
    """Step 3 tab UI for R setup and batch OCT flattening."""
    TUTORIAL_IMAGE_NAME = "step3_tutorial.png"
    PIXEL_WIDTH_UM = 3.89
    MIN_NEGATIVE_UM = 200.0
    MIN_POSITIVE_UM = 3000.0
    MIN_DEPTH_OUTWARD_UM = 50.0
    MIN_DEPTH_INWARD_UM = 450.0
    CENTERED_FOVEA_GUARD_PX = 100
    REQUIRED_INPUTS = (
        ("Light_MARKED", ("Light_MARKED", "LIGHT_MARKED"), "Light_MARKED.hdr/.img", 8),
        ("LIGHT", ("LIGHT", "Light"), "LIGHT.hdr/.img", 16),
    )
    R_SCRIPT_NAME = "RAW_OCT_PROCESSING_2023_09SEP-05_WSU.R"
    R_OUTPUT_SCRIPT_NAME = "more_outputs_afterRAW_OCT_PROCESSING_2022_11NOV_27_WSU_noHypoDenseBand_EA edited.R"
    DEFAULT_R_SCRIPT_TIMEOUT_MINUTES = 240
    R_OUTPUT_MODE_PARALLEL = "parallel"
    R_OUTPUT_MODE_SEQUENTIAL = "sequential"
    R_REQUIRED_VERSION = "3.3.1"
    R_DOWNLOAD_PAGE = "https://cran-archive.r-project.org/bin/windows/base/old/3.3.1/"
    R_INSTALLER_NAME = f"R-{R_REQUIRED_VERSION}-win.exe"
    R_REQUIRED_PACKAGES = ("AnalyzeFMRI", "RNiftyReg")
    R_LOCAL_PACKAGE_ORDER = (
        "R.methodsS3",
        "R.oo",
        "R.utils",
        "R.matlab",
        "fastICA",
        "Rcpp",
        "RcppEigen",
        "RNifti",
        "ore",
        "xtable",
        "AnalyzeFMRI",
        "RNiftyReg",
    )
    R_LOCAL_PACKAGE_FILES = {
        "AnalyzeFMRI": "AnalyzeFMRI.zip",
        "R.matlab": "R.matlab.zip",
        "R.methodsS3": "R.methodsS3.zip",
        "fastICA": "fastICA.zip",
        "ore": "ore.zip",
        "Rcpp": "Rcpp.zip",
        "RcppEigen": "RcppEigen.zip",
        "RNifti": "RNifti.zip",
        "RNiftyReg": "RNiftyReg.zip",
        "xtable": "xtable.zip",
        "R.oo": "R.oo.zip",
        "R.utils": "R.utils.zip",
    }
    R_WORKSPACE_FILES = (
        "DARK__and__LIGHT__flat.RData",
        "_done_DARK__and__LIGHT.RData",
    )
    R_ARRAY_EXPORT_DIR = "step3_r_arrays"
    R_PROGRESS_BY_STEP = {
        "startup": (1, "Starting R script"),
        "input-config": (2, "Reading R input configuration"),
        "load-images": (5, "Loading Analyze volumes in R"),
        "fovea-center": (8, "Calculating vertex"),
        "rpe-line": (11, "Reading RPE line"),
        "rpe-spline": (14, "Fitting RPE spline"),
        "apparent-angle": (17, "Computing apparent angles"),
        "perpendiculars": (21, "Building perpendicular sampling lines"),
        "flattened-markers": (25, "Flattening marker image"),
        "dark-loop": (36, "Flattening DARK slices"),
        "light-loop": (47, "Flattening LIGHT slices"),
        "post-log-convert": (54, "Converting flattened data to raw scale"),
        "grand-mean": (59, "Building grand mean image"),
        "rough-vit-loop": (63, "Aligning retina profiles"),
        "python-export": (72, "Exporting R arrays"),
        "layer-borders": (78, "Drawing borders"),
        "main-normalization": (86, "Spatially normalizing main retina"),
        "fovea-normalization": (92, "Spatially normalizing fovea"),
        "final-export": (97, "Drawing borders and writing outputs"),
        "done": (100, "R processing complete"),
    }

    def __init__(self, parent, preferences=None):
        super().__init__(parent)
        self.preferences = preferences

        self.current_sdb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.output_sdb_dir = self.current_sdb_dir
        self.results = None
        self.original_light_volume = None
        self.figure = None
        self.canvas = None
        self._preview_photo = None
        self.r_setup_panel = None
        self.r_batch_panel = None
        self.r_batch_run_panel = None
        self.batch_results_notebook = None
        self.batch_result_folders = []
        self.batch_result_tab_states = {}
        self._active_batch_result_tab = None
        self.r_setup_button = None
        self.r_batch_button = None
        self.load_r_results_button = None
        self._busy = False
        self._r_cancel_event = threading.Event()
        self._r_process_lock = threading.Lock()
        self._active_r_processes = set()
        self._active_r_folder_keys = set()
        self._pending_batch_restart = None
        self._pending_input_folder = None
        self._pending_batch_folders = None
        self.r_package_library_path = None if self.preferences is None else self.preferences.get("r_package_library_path")

        self.view_var = tk.StringVar(value="DARK_MARKED_find_vertex")
        self.status_var = tk.StringVar(value="Ready - use batch Step 3 R processing.")
        self.info_var = tk.StringVar(value="")
        self.progress_text_var = tk.StringVar(value="Idle")

        self._build_ui()
        self._refresh_input_status()

    def _build_ui(self):
        self.build_standard_layout(
            sidebar_width=self.SIDEBAR_WIDTH,
            status_var=self.status_var,
            status_bar_content_margin=True,
        )
        process_section = self.add_sidebar_section("Process", pady=(0, 5))
        process = process_section.body

        self.r_setup_button = action_button(
            process,
            self,
            "Set up R and packages…",
            self._open_r_setup_wizard,
            "settings",
            tooltip="Install or verify R and manage the packages required by Step 3.",
        )
        self.r_setup_button.pack(fill="x", pady=2)

        self.r_batch_button = action_button(
            process,
            self,
            "Select folders to flatten…",
            self._open_r_batch_scanner,
            "folder",
            tooltip="Choose a parent folder. AIDaS will find eligible annotated folders inside it.",
        )
        self.r_batch_button.pack(fill="x", pady=2)

        self.load_r_results_button = action_button(
            process,
            self,
            "Load R results…",
            self._browse_r_results_folder,
            "results",
            tooltip="Open a folder containing completed Step 3 results.",
        )
        self.load_r_results_button.pack(fill="x", pady=2)

        ttk.Separator(process, orient="horizontal").pack(fill="x", pady=(6, 4))

        ttk.Label(process, text="View").pack(anchor="w", pady=(6, 2))
        view_combo = ttk.Combobox(
            process,
            textvariable=self.view_var,
            values=["DARK_MARKED_find_vertex", "_tissueBorders__DARK"],
            state="readonly",
        )
        view_combo.pack(fill="x", pady=2)
        view_combo.bind("<<ComboboxSelected>>", lambda _: self._on_view_selected())

        ttk.Separator(process, orient="horizontal").pack(fill="x", pady=(8, 4))
        progress_text_frame = ttk.Frame(process, height=44)
        progress_text_frame.pack(fill="x", pady=(0, 4))
        progress_text_frame.pack_propagate(False)
        ttk.Label(
            progress_text_frame,
            textvariable=self.progress_text_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            foreground="gray",
            justify="left",
        ).pack(fill="both", expand=True)

        self.plot_holder = ttk.Frame(self.content)
        self.plot_holder.pack(fill="both", expand=True)
        self._render()

    def _set_process_buttons(self, state):
        if self.r_setup_button is not None:
            self.r_setup_button.configure(state=state)
        if self.r_batch_button is not None:
            self.r_batch_button.configure(state=state)
        if self.load_r_results_button is not None:
            self.load_r_results_button.configure(state=state)

    def _open_r_setup_wizard(self, on_finish=None):
        if self._busy:
            return None
        self._clear_plot_holder()
        self.r_setup_panel = RSetupWizard(self, self.plot_holder, on_finish=on_finish)
        self.r_setup_panel.pack(fill="both", expand=True)
        self.status_var.set("Step 3 R setup is open in the preview area.")
        self.progress_text_var.set("R setup")
        return None

    def _close_r_setup_panel(self, *, render_previous):
        panel = self.r_setup_panel
        self.r_setup_panel = None
        if panel is not None:
            try:
                panel.destroy()
            except Exception:
                pass
        if render_previous:
            self._render()

    @staticmethod
    def _script_path(role="main"):
        """Return a Step 3 script from source or PyInstaller's bundle."""
        script_name = Step3Frame.R_SCRIPT_NAME if role == "main" else Step3Frame.R_OUTPUT_SCRIPT_NAME
        return Path(resource_path(script_name))

    @staticmethod
    def _user_r_script_dir(role="main"):
        return user_r_script_dir(role)

    def _available_r_scripts(self, role="main"):
        prefixes = ("RAW_OCT_PROCESSING_",) if role == "main" else ("more_outputs_afterRAW_OCT_PROCESSING_",)
        return discover_r_scripts(
            self._script_path(role),
            self._user_r_script_dir(role),
            bundled_prefixes=prefixes,
        )

    def _import_user_r_script(self, source_path, role="main"):
        return import_r_script(Path(source_path), self._user_r_script_dir(role))

    def available_r_scripts(self, role="main"):
        """Expose script choices to the centralized Settings window."""

        return self._available_r_scripts(role)

    def import_r_script_for_role(self, source_path, role="main"):
        """Import a user script selected from centralized Settings."""

        return self._import_user_r_script(source_path, role)

    def _selected_r_script_path(self, role="main"):
        preference_key = "r_main_script_path" if role == "main" else "r_output_script_path"
        configured = None if self.preferences is None else self.preferences.get(preference_key)
        if configured:
            configured_path = Path(configured)
            if configured_path.is_file():
                return configured_path
        choices = self._available_r_scripts(role)
        return choices[0].path if choices else None

    def select_r_script(self, role, path):
        """Persist one Settings-selected R script for future Step 3 runs."""

        selected = Path(path).resolve()
        if not selected.is_file():
            raise FileNotFoundError(selected)
        preference_key = "r_main_script_path" if role == "main" else "r_output_script_path"
        if self.preferences is not None:
            self.preferences.set(preference_key, str(selected))
        return selected

    def _resolve_rscript_executable(self):
        """Find an installed R executable, accepting only the Step 3 R version."""
        candidates = []
        configured = None if self.preferences is None else self.preferences.get("rscript_path")
        if configured:
            candidates.append(Path(configured))

        env_override = os.environ.get("RSCRIPT_PATH") or os.environ.get("R_SCRIPT_PATH")
        if env_override:
            candidates.append(Path(env_override))

        for name in ("Rscript", "Rscript.exe", "R", "R.exe"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))

        candidates.extend(self._installed_r_executable_candidates())

        seen = set()
        for candidate in candidates:
            candidate = self._normalize_r_executable(candidate)
            if candidate is None or not candidate.is_file():
                continue
            try:
                identity = os.path.normcase(str(candidate.resolve()))
            except OSError:
                identity = os.path.normcase(str(candidate))
            if identity in seen:
                continue
            seen.add(identity)
            if self._r_version_for_executable(candidate) == self.R_REQUIRED_VERSION:
                return candidate
        return None

    def _installed_r_executable_candidates(self):
        """Return R executable paths from common system, user, and registry installs."""
        roots = []
        for env_name in ("R_HOME", "R_HOME_DIR", "LOCALAPPDATA", "APPDATA", "USERPROFILE"):
            value = os.environ.get(env_name)
            if value:
                roots.append(Path(value))

        if os.name == "nt":
            roots.extend(
                [
                    Path(r"C:\Program Files\R"),
                    Path(r"C:\Program Files (x86)\R"),
                    Path.home() / "AppData" / "Local" / "Programs" / "R",
                    Path.home() / "AppData" / "Local" / "R",
                    Path.home() / "AppData" / "Roaming" / "R",
                    Path.home() / "R",
                    Path.home() / "scoop" / "apps" / "r",
                ]
            )
            roots.extend(self._registry_r_install_paths())

        # A registry entry may point at the newest side-by-side R install.
        # Also inspect its parent so an older R 3.3.1 install is not missed.
        for root in list(roots):
            root = Path(root)
            if root.name.lower().startswith("r-"):
                roots.append(root.parent)

        candidates = []
        executable_names = ("Rscript.exe", "Rscript", "R.exe", "Rterm.exe", "R", "Rterm")
        seen_dirs = set()
        for root in roots:
            root = Path(root)
            install_dirs = [root]
            try:
                if root.is_dir():
                    install_dirs.extend(
                        child
                        for child in root.iterdir()
                        if child.is_dir() and child.name.lower().startswith("r-")
                    )
            except OSError:
                continue

            for install_dir in install_dirs:
                try:
                    dir_key = os.path.normcase(str(install_dir.resolve()))
                except OSError:
                    dir_key = os.path.normcase(str(install_dir))
                if dir_key in seen_dirs:
                    continue
                seen_dirs.add(dir_key)
                for relative_dir in (Path("bin") / "x64", Path("bin")):
                    for executable_name in executable_names:
                        candidates.append(install_dir / relative_dir / executable_name)
        return candidates

    @staticmethod
    def _registry_r_install_paths():
        if os.name != "nt":
            return []
        try:
            import winreg
        except ImportError:
            return []

        paths = []
        registry_keys = (
            (winreg.HKEY_CURRENT_USER, r"Software\R-core\R"),
            (winreg.HKEY_CURRENT_USER, r"Software\R-core\R32"),
            (winreg.HKEY_CURRENT_USER, r"Software\R-core\R64"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\R-core\R"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\R-core\R32"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\R-core\R64"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\R-core\R"),
        )
        for hive, key_path in registry_keys:
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    for value_name in ("", "InstallPath", "InstallPath32", "InstallPath64", "R_HOME"):
                        try:
                            value, _ = winreg.QueryValueEx(key, value_name)
                        except OSError:
                            continue
                        if value:
                            paths.append(Path(str(value)))
            except OSError:
                continue
        return paths

    def _r_version_for_executable(self, executable):
        """Read the version reported by an R executable without accepting a guess."""
        executable = Path(executable)
        if not executable.is_file():
            return None
        try:
            result = subprocess.run(
                [str(executable), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = result.stdout or ""
        match = re.search(
            r"\b(?:R|R scripting front-end)\s+version\s+(\d+\.\d+\.\d+)(?!\.\d)",
            output,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def _normalize_r_executable(path):
        """Return a non-interactive R executable, preferring Rscript.exe.

        Users may select R.exe or Rterm.exe from the file dialog. If possible,
        convert those selections to the adjacent Rscript.exe. Rgui/RStudio is
        intentionally rejected because it is interactive.
        """
        if not path:
            return None
        path = Path(path)
        if not path.is_file():
            return None

        name = path.name.lower()
        if name in {"rscript.exe", "rscript"}:
            return path

        if name in {"r.exe", "rterm.exe", "r", "rterm"}:
            sibling_name = "Rscript.exe" if name.endswith(".exe") else "Rscript"
            sibling = path.with_name(sibling_name)
            if sibling.is_file():
                return sibling
            return path

        return None

    @staticmethod
    def _build_r_run_command(r_executable, script_path, script_args):
        name = Path(r_executable).name.lower()
        if name in {"rscript.exe", "rscript"}:
            return [str(r_executable), "--vanilla", str(script_path), *script_args]
        if name in {"r.exe", "rterm.exe", "r", "rterm"}:
            return [
                str(r_executable),
                "--vanilla",
                "--slave",
                f"--file={script_path}",
                "--args",
                *script_args,
            ]
        raise RuntimeError(
            "Select Rscript.exe, not the interactive R/RStudio program. "
            "Typical path: C:\\Program Files\\R\\R-x.x.x\\bin\\x64\\Rscript.exe"
        )

    @staticmethod
    def _build_r_eval_command(r_executable, expression):
        name = Path(r_executable).name.lower()
        if name in {"rscript.exe", "rscript"}:
            return [str(r_executable), "--vanilla", "-e", expression]
        if name in {"r.exe", "rterm.exe", "r", "rterm"}:
            return [str(r_executable), "--vanilla", "--slave", "-e", expression]
        raise RuntimeError("R package setup needs Rscript.exe or Rterm.exe.")

    @staticmethod
    def _r_string(value):
        return "'" + str(value).replace("\\", "/").replace("'", "\\'") + "'"

    @classmethod
    def _normalize_r_output_mode(cls, output_mode):
        """Return a supported second-script scheduling mode."""
        if str(output_mode).strip().lower() == cls.R_OUTPUT_MODE_SEQUENTIAL:
            return cls.R_OUTPUT_MODE_SEQUENTIAL
        return cls.R_OUTPUT_MODE_PARALLEL

    @staticmethod
    def _cpu_worker_limit():
        process_cpu_count = getattr(os, "process_cpu_count", None)
        if callable(process_cpu_count):
            try:
                count = process_cpu_count()
                if count:
                    return max(1, int(count))
            except (OSError, TypeError, ValueError):
                pass
        if hasattr(os, "sched_getaffinity"):
            try:
                count = len(os.sched_getaffinity(0))
                if count:
                    return max(1, int(count))
            except (OSError, TypeError, ValueError):
                pass
        return max(1, int(os.cpu_count() or 1))

    @classmethod
    def _r_worker_limit(cls):
        """Leave one logical processor available for the UI and Step 2 AI work."""
        return max(1, cls._cpu_worker_limit() - 1)

    @classmethod
    def _r_threads_per_process(cls, workers):
        """Prevent nested R/BLAS threads from oversubscribing the shared host."""
        workers = max(1, int(workers))
        return max(1, cls._r_worker_limit() // workers)

    def _default_r_package_library(self):
        if self.r_package_library_path:
            configured_path = Path(self.r_package_library_path)
            local_app_data = os.environ.get("LOCALAPPDATA")
            legacy_path = (
                Path(local_app_data) / "AIDaS" / "R-packages"
                if local_app_data
                else None
            )
            try:
                if legacy_path is None or configured_path.resolve() != legacy_path.resolve():
                    return configured_path
            except OSError:
                return configured_path

        documents = Path.home() / "Documents"
        if documents.is_dir():
            return documents / "AIDaS" / "R-packages"
        return Path.home() / "AIDaS_R_packages"

    def _r_env(self, thread_limit=None):
        env = os.environ.copy()
        library_path = self._default_r_package_library()
        if library_path:
            env["R_LIBS_USER"] = str(library_path.resolve())
        if thread_limit is not None:
            thread_limit = str(max(1, int(thread_limit)))
            for name in (
                "OMP_NUM_THREADS",
                "OMP_THREAD_LIMIT",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[name] = thread_limit
        return env

    def _r_package_check_expression(self, package_name):
        library_path = self._default_r_package_library()
        lib = self._r_string(library_path.resolve())
        return (
            f".libPaths(c({lib}, .libPaths())); "
            f"if (requireNamespace({self._r_string(package_name)}, quietly=TRUE)) "
            "quit(status=0) else quit(status=1)"
        )

    def _r_packages_ready(self, rscript):
        library_path = self._default_r_package_library()
        try:
            library_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            return False
        for package_name in self.R_REQUIRED_PACKAGES:
            result = subprocess.run(
                self._build_r_eval_command(rscript, self._r_package_check_expression(package_name)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._r_env(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0:
                return False
        return True

    def _ensure_r_ready_with_wizard(self):
        rscript = self._resolve_rscript_executable()
        if rscript is None:
            self.status_var.set("Rscript was not found. Open Settings > R environment to continue.")
            return None
        if self.preferences is not None:
            self.preferences.set("rscript_path", str(rscript))
        if self._r_packages_ready(rscript):
            self.status_var.set("R and required Step 3 packages are ready.")
            return rscript
        self.status_var.set("Step 3 R packages are missing. Open Settings > R environment to install them.")
        return None

    @staticmethod
    def _analyze_base_name(base_path):
        return Path(str(base_path)).name

    @staticmethod
    def _r_index_string(slice_count):
        return ",".join(str(idx) for idx in range(1, int(slice_count) + 1))

    @staticmethod
    def _short_process_text(text, max_lines=24):
        lines = [line for line in str(text or "").splitlines() if line.strip()]
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[-max_lines:])

    def _progress_from_r_line(self, line):
        match = re.search(r"DEBUG \[([^\]]+)\]\s*(.*)", str(line))
        if not match:
            return None
        step = match.group(1).strip()
        detail = match.group(2).strip()
        progress = self.R_PROGRESS_BY_STEP.get(step)
        if progress is None:
            return None
        percent, label = progress
        if detail and step in {"dark-loop", "light-loop"}:
            match_slice = re.search(r"Processing z=\s*(\d+)\s*of\s*(\d+)", detail)
            if match_slice:
                current = int(match_slice.group(1))
                total = max(1, int(match_slice.group(2)))
                span = 10.0
                percent = min(99.0, float(percent) + (span * (current - 1) / total))
                label = f"{label}: slice {current}/{total}"
        return percent, label

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "values"):
            value = value.values
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        return np.asarray(value)

    def _load_r_workspace_results(self, output_dir):
        if pyreadr is None:
            raise RuntimeError("pyreadr is not installed, so R workspace files cannot be loaded.")

        output_dir = Path(output_dir)
        flat_rdata = output_dir / self.R_WORKSPACE_FILES[0]
        done_rdata = output_dir / self.R_WORKSPACE_FILES[1]
        if not flat_rdata.is_file() or not done_rdata.is_file():
            missing = [name for name in self.R_WORKSPACE_FILES if not (output_dir / name).is_file()]
            raise FileNotFoundError("Missing R workspace file(s): " + ", ".join(missing))

        flat_data = pyreadr.read_r(str(flat_rdata))
        done_data = pyreadr.read_r(str(done_rdata))

        def require(dataset, key):
            if key not in dataset:
                raise KeyError(f"R workspace file is missing required object: {key}")
            return self._to_numpy(dataset[key])

        flattened_dark = require(flat_data, "FLATTENED.DARK.RETINA.RRC")
        flattened_light = require(flat_data, "FLATTENED.LIGHT.RETINA.RRC")
        markers = require(flat_data, "FLATTENED.MARKERS.RRC")

        first_grand_mean = require(done_data, "FIRST.GRAND.MEAN") if "FIRST.GRAND.MEAN" in done_data else None
        second_grand_mean = require(done_data, "SECOND.GRAND.MEAN") if "SECOND.GRAND.MEAN" in done_data else None

        final_grand_mean = np.array(flattened_dark[:, :, 0], copy=True)
        for z in range(1, flattened_dark.shape[2]):
            final_grand_mean = final_grand_mean + flattened_dark[:, :, z]
        for z in range(1, flattened_light.shape[2]):
            final_grand_mean = final_grand_mean + flattened_light[:, :, z]
        final_grand_mean = final_grand_mean / (flattened_dark.shape[2] + flattened_light.shape[2])

        if "vertex" in done_data:
            vertex = int(np.ravel(self._to_numpy(done_data["vertex"]))[0])
            grand_profile, _fallback_vertex = _grand_profile_and_vertex(final_grand_mean)
        elif "vertex" in flat_data:
            vertex = int(np.ravel(self._to_numpy(flat_data["vertex"]))[0])
            grand_profile, _fallback_vertex = _grand_profile_and_vertex(final_grand_mean)
        else:
            grand_profile, vertex = _grand_profile_and_vertex(final_grand_mean)

        def optional_or_empty(dataset, key):
            return require(dataset, key) if key in dataset else np.empty((0, 0), dtype=np.float64)

        dark_indices = np.arange(1, flattened_dark.shape[2] + 1, dtype=np.float64)
        light_indices = np.arange(1, flattened_light.shape[2] + 1, dtype=np.float64)

        return {
            "flattened_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "flattened_light": np.transpose(flattened_light, (2, 0, 1)),
            "final_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "final_light": np.transpose(flattened_light, (2, 0, 1)),
            "markers": markers,
            "first_grand_mean": first_grand_mean,
            "second_grand_mean": second_grand_mean,
            "final_grand_mean": final_grand_mean,
            "grand_profile": grand_profile,
            "vertex": vertex,
            "shift_dark": optional_or_empty(done_data, "SHIFT.POSITION.DARK"),
            "shift_light": optional_or_empty(done_data, "SHIFT.POSITION.LIGHT"),
            "shift_dark_refined": optional_or_empty(done_data, "SHIFT.POSITION.DARK.REFINED"),
            "shift_light_refined": optional_or_empty(done_data, "SHIFT.POSITION.LIGHT.REFINED"),
            "best_lateral_dark": optional_or_empty(done_data, "BEST.LAT.MOVE.DARK"),
            "best_lateral_light": optional_or_empty(done_data, "BEST.LAT.MOVE.LIGHT"),
            "apparent_angles_for_dark": (
                require(flat_data, "APPARENT.ANGLES.FOR.DARK")
                if "APPARENT.ANGLES.FOR.DARK" in flat_data
                else np.column_stack((dark_indices, dark_indices, dark_indices))
            ),
            "apparent_angles_for_light": (
                require(flat_data, "APPARENT.ANGLES.FOR.LIGHT")
                if "APPARENT.ANGLES.FOR.LIGHT" in flat_data
                else np.column_stack((light_indices, light_indices, light_indices))
            ),
            "dark_rrc": flattened_dark,
            "light_rrc": flattened_light,
            "markers_rrc": markers,
        }

    def _load_r_array_export(self, output_dir):
        export_dir = Path(output_dir) / self.R_ARRAY_EXPORT_DIR
        if not export_dir.is_dir():
            raise FileNotFoundError(f"Missing R array export folder: {export_dir}")

        def load_array(name, required=True):
            bin_path = export_dir / f"{name}.bin"
            shape_path = export_dir / f"{name}.shape"
            if not bin_path.is_file() or not shape_path.is_file():
                if required:
                    raise FileNotFoundError(f"Missing R array export: {name}")
                return None
            shape_text = shape_path.read_text(encoding="utf-8").strip()
            shape = tuple(int(part) for part in shape_text.split(",") if part.strip())
            data = np.fromfile(bin_path, dtype="<f8")
            expected = int(np.prod(shape)) if shape else 1
            if data.size != expected:
                raise ValueError(f"R array export {name} has {data.size} values; expected {expected}.")
            if not shape:
                return data
            return data.reshape(shape, order="F")

        flattened_dark = np.asarray(load_array("FLATTENED_DARK_RETINA_RRC"), dtype=np.float64)
        flattened_light = np.asarray(load_array("FLATTENED_LIGHT_RETINA_RRC"), dtype=np.float64)
        markers = np.asarray(load_array("FLATTENED_MARKERS_RRC"), dtype=np.float64)

        first_grand_mean = load_array("FIRST_GRAND_MEAN", required=False)
        second_grand_mean = load_array("SECOND_GRAND_MEAN", required=False)
        final_grand_mean = load_array("FINAL_GRAND_MEAN", required=False)
        if final_grand_mean is None:
            final_grand_mean = np.nanmean(np.concatenate((flattened_dark, flattened_light), axis=2), axis=2)
        if first_grand_mean is None:
            first_grand_mean = np.array(final_grand_mean, copy=True)
        if second_grand_mean is None:
            second_grand_mean = np.array(final_grand_mean, copy=True)

        grand_profile = load_array("GRAND_PROFILE", required=False)
        vertex_data = load_array("VERTEX", required=False)
        if grand_profile is None or grand_profile.shape[0] != final_grand_mean.shape[1]:
            grand_profile, fallback_vertex = _grand_profile_and_vertex(final_grand_mean)
        else:
            _profile, fallback_vertex = _grand_profile_and_vertex(final_grand_mean)
        if vertex_data is not None and np.ravel(vertex_data).size:
            vertex = int(np.ravel(vertex_data)[0])
        else:
            vertex = fallback_vertex

        dark_indices = np.arange(1, flattened_dark.shape[2] + 1, dtype=np.float64)
        light_indices = np.arange(1, flattened_light.shape[2] + 1, dtype=np.float64)

        def optional_or_empty(name):
            value = load_array(name, required=False)
            return np.empty((0, 0), dtype=np.float64) if value is None else value

        return {
            "flattened_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "flattened_light": np.transpose(flattened_light, (2, 0, 1)),
            "final_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "final_light": np.transpose(flattened_light, (2, 0, 1)),
            "markers": markers,
            "first_grand_mean": first_grand_mean,
            "second_grand_mean": second_grand_mean,
            "final_grand_mean": final_grand_mean,
            "grand_profile": grand_profile,
            "vertex": vertex,
            "shift_dark": optional_or_empty("SHIFT_POSITION_DARK"),
            "shift_light": optional_or_empty("SHIFT_POSITION_LIGHT"),
            "shift_dark_refined": optional_or_empty("SHIFT_POSITION_DARK_REFINED"),
            "shift_light_refined": optional_or_empty("SHIFT_POSITION_LIGHT_REFINED"),
            "best_lateral_dark": optional_or_empty("BEST_LAT_MOVE_DARK"),
            "best_lateral_light": optional_or_empty("BEST_LAT_MOVE_LIGHT"),
            "apparent_angles_for_dark": optional_or_empty("APPARENT_ANGLES_FOR_DARK")
            if (export_dir / "APPARENT_ANGLES_FOR_DARK.bin").is_file()
            else np.column_stack((dark_indices, dark_indices, dark_indices)),
            "apparent_angles_for_light": optional_or_empty("APPARENT_ANGLES_FOR_LIGHT")
            if (export_dir / "APPARENT_ANGLES_FOR_LIGHT.bin").is_file()
            else np.column_stack((light_indices, light_indices, light_indices)),
            "dark_rrc": flattened_dark,
            "light_rrc": flattened_light,
            "markers_rrc": markers,
        }

    def _load_r_analyze_results(self, output_dir):
        output_dir = Path(output_dir)
        dark_base = output_dir / "_flat_DARK"
        light_base = output_dir / "_flat_LIGHT"
        if not (dark_base.with_suffix(".hdr").is_file() and light_base.with_suffix(".hdr").is_file()):
            raise FileNotFoundError("Missing R Analyze outputs _flat_DARK.hdr and _flat_LIGHT.hdr.")

        flattened_dark = _load_analyze_volume_r_layout(dark_base)
        flattened_light = _load_analyze_volume_r_layout(light_base)
        final_grand_mean = np.nanmean(np.concatenate((flattened_dark, flattened_light), axis=2), axis=2)
        grand_profile, vertex = _grand_profile_and_vertex(final_grand_mean)
        dark_indices = np.arange(1, flattened_dark.shape[2] + 1, dtype=np.float64)
        light_indices = np.arange(1, flattened_light.shape[2] + 1, dtype=np.float64)

        return {
            "flattened_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "flattened_light": np.transpose(flattened_light, (2, 0, 1)),
            "final_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "final_light": np.transpose(flattened_light, (2, 0, 1)),
            "markers": None,
            "first_grand_mean": final_grand_mean,
            "second_grand_mean": final_grand_mean,
            "final_grand_mean": final_grand_mean,
            "grand_profile": grand_profile,
            "vertex": vertex,
            "shift_dark": np.empty((0, 0), dtype=np.float64),
            "shift_light": np.empty((0, 0), dtype=np.float64),
            "shift_dark_refined": np.empty((0, 0), dtype=np.float64),
            "shift_light_refined": np.empty((0, 0), dtype=np.float64),
            "best_lateral_dark": np.empty((0, 0), dtype=np.float64),
            "best_lateral_light": np.empty((0, 0), dtype=np.float64),
            "apparent_angles_for_dark": np.column_stack((dark_indices, dark_indices, dark_indices)),
            "apparent_angles_for_light": np.column_stack((light_indices, light_indices, light_indices)),
            "dark_rrc": flattened_dark,
            "light_rrc": flattened_light,
            "markers_rrc": None,
        }

    def _load_r_results_with_fallbacks(self, output_dir):
        errors = []
        for loader in (self._load_r_workspace_results, self._load_r_array_export, self._load_r_analyze_results):
            try:
                results = loader(output_dir)
                return results, loader.__name__, errors
            except Exception as exc:
                errors.append(f"{loader.__name__}: {exc}")
        raise RuntimeError("Could not load R outputs using any supported method:\n" + "\n".join(errors))

    def _load_original_light_for_preview(self, folder):
        input_paths = self._find_input_paths(folder)
        light_path = input_paths.get("LIGHT")
        if not light_path:
            self.original_light_volume = None
            return
        try:
            self.original_light_volume = _load_analyze_volume_r_layout(light_path)
        except Exception:
            self.original_light_volume = None

    def _load_r_results_from_folder(self, folder, show_errors=True):
        folder = Path(folder)
        try:
            results, loader_name, loader_errors = self._load_r_results_with_fallbacks(folder)
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Load R Results", f"Could not load Step 3 R results.\n{exc}")
            self.status_var.set("Could not load Step 3 R results.")
            return False

        self.current_sdb_dir = str(folder)
        self.output_sdb_dir = str(folder)
        self.results = results
        self._load_original_light_for_preview(folder)
        self.progress_text_var.set("Loaded R results")
        self.view_var.set("DARK_MARKED_find_vertex")
        self.info_var.set(
            f"flattened_dark: {results['flattened_dark'].shape}\n"
            f"flattened_light: {results['flattened_light'].shape}\n"
            f"final_grand_mean: {results['final_grand_mean'].shape}\n"
            f"vertex: {results['vertex']}\n"
            f"loaded via: {loader_name}"
        )
        if loader_errors:
            self.info_var.set(self.info_var.get() + "\n\nLoader fallbacks:\n" + "\n".join(loader_errors))
        self.status_var.set(f"Loaded Step 3 R results from {folder}.")
        self._render()
        return True

    def _load_result_png(self, filename):
        return self._load_result_png_from_folder(self.output_sdb_dir or self.current_sdb_dir, filename)

    def _load_result_png_from_folder(self, folder, filename):
        png_path = Path(folder) / filename
        if not png_path.is_file() and filename.startswith("_tissueBorders__"):
            matches = sorted(Path(folder).glob("_tissueBorders__*.png"))
            if matches:
                png_path = matches[0]
        if not png_path.is_file():
            raise FileNotFoundError(f"{filename} not found in {png_path.parent}")
        with Image.open(png_path) as img:
            return img.copy()

    def _browse_r_results_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder containing Step 3 R results",
            initialdir=self.output_sdb_dir or self.current_sdb_dir or None,
        )
        if folder:
            self._load_r_results_from_folder(folder, show_errors=True)

    @staticmethod
    def _existing_basepath(folder, names):
        for name in names:
            base = os.path.join(folder, name)
            if os.path.isfile(base + ".hdr") and os.path.isfile(base + ".img"):
                return base
        return None

    @staticmethod
    def _analyze_stack_info(base_path):
        data = np.asarray(read_analyze(_normalize_analyze_path(base_path)))
        if data.ndim == 2:
            shape = (1, int(data.shape[0]), int(data.shape[1]))
        elif data.ndim == 3:
            shape = tuple(int(v) for v in data.shape)
        else:
            raise ValueError(f"Analyze file must be 2-D or 3-D, got shape {data.shape}.")
        return {
            "shape": shape,
            "dtype": str(data.dtype),
            "bits": int(data.dtype.itemsize * 8),
        }

    @classmethod
    def _read_input_stack_info(cls, paths):
        return {label: cls._analyze_stack_info(path) for label, path in paths.items()}

    @classmethod
    def _validate_input_stack_shapes(cls, stack_info):
        shapes = {label: info["shape"] for label, info in stack_info.items()}
        expected = shapes["Light_MARKED"]
        mismatched = {label: shape for label, shape in shapes.items() if shape != expected}
        if mismatched:
            lines = [f"Light_MARKED: {expected}"]
            lines.extend(f"{label}: {shape}" for label, shape in mismatched.items())
            raise ValueError(
                "Step 3 inputs must all have the same Analyze stack shape "
                "(slices, height, width).\n" + "\n".join(lines)
            )
        return shapes

    def _find_input_paths(self, folder):
        return {
            label: self._existing_basepath(folder, names)
            for label, names, _display_name, _required_bits in self.REQUIRED_INPUTS
        }

    def _missing_input_names(self, input_paths):
        return [
            display_name
            for label, _names, display_name, _required_bits in self.REQUIRED_INPUTS
            if input_paths.get(label) is None
        ]

    def _input_requirement_issues(self, input_paths, input_info):
        issues = []
        for label, _names, display_name, required_bits in self.REQUIRED_INPUTS:
            if input_paths.get(label) is None:
                issues.append(display_name)
                continue
            info = input_info.get(label)
            if info is None:
                issues.append(f"{display_name} cannot be read")
                continue
            if info["bits"] != required_bits:
                issues.append(f"{display_name} must be {required_bits}-bit, found {info['bits']}-bit")
        return issues

    def _read_available_input_info(self, input_paths):
        input_info = {}
        read_errors = {}
        for label, path in input_paths.items():
            if path is None:
                continue
            try:
                input_info[label] = self._analyze_stack_info(path)
            except Exception as exc:
                read_errors[label] = str(exc)
        return input_info, read_errors

    def _format_input_checklist(self, input_paths, input_info=None, read_errors=None):
        input_info = {} if input_info is None else input_info
        read_errors = {} if read_errors is None else read_errors
        lines = []
        for label, _names, display_name, required_bits in self.REQUIRED_INPUTS:
            path = input_paths.get(label)
            if path is None:
                lines.append(f"Missing: {display_name}")
            elif label in read_errors:
                lines.append(f"Missing: {display_name} (cannot read)")
            else:
                info = input_info.get(label)
                if info is not None and info["bits"] == required_bits:
                    lines.append(f"OK: {display_name} ({required_bits}-bit)")
                elif info is not None:
                    lines.append(f"Missing: {display_name} ({info['bits']}-bit, needs {required_bits}-bit)")
                else:
                    lines.append(f"Missing: {display_name} (cannot read)")
        return "\n".join(lines)

    def _reset_to_tutorial_state(self):
        self.results = None
        self.original_light_volume = None
        self.view_var.set("DARK_MARKED_find_vertex")
        self.progress_text_var.set("Idle")
        self._render()

    def _refresh_input_status(self):
        # A Step 2 folder notification may arrive while this tab is hidden.
        # Never replace the live R progress/Stop panel or its status text.
        if getattr(self, "_busy", False):
            return None, []
        if not self.current_sdb_dir:
            self._reset_to_tutorial_state()
            input_paths = {label: None for label, _names, _display_name, _required_bits in self.REQUIRED_INPUTS}
            self.info_var.set(
                "Step 3 input files:\n"
                + self._format_input_checklist(input_paths)
                + "\n\nRun batch Step 3 from a folder containing MARKED and RAW Analyze files."
            )
            self.status_var.set("Missing Step 3 input folder.")
            return None, ["Step 3 input folder"]

        input_paths = self._find_input_paths(self.current_sdb_dir)
        input_info, read_errors = self._read_available_input_info(input_paths)
        issues = self._missing_input_names(input_paths)
        issues.extend(self._input_requirement_issues(input_paths, input_info))
        issues = list(dict.fromkeys(issues))

        if issues:
            self._reset_to_tutorial_state()
            self.info_var.set(
                "Step 3 input files:\n"
                + self._format_input_checklist(input_paths, input_info, read_errors)
            )
            self.status_var.set("Step 3 files are missing or do not meet bit-depth requirements.")
        else:
            self.info_var.set(
                "Step 3 is using these files:\n"
                + self._format_input_checklist(input_paths, input_info, read_errors)
            )
            self.status_var.set("All required Step 3 files found with correct bit depth. Ready for batch processing.")

        return input_paths, issues

    def on_show(self):
        self._refresh_input_status()

    def set_input_folder(self, folder):
        if not folder:
            return False
        folder = os.path.abspath(os.fspath(folder))
        if self._busy:
            self._pending_input_folder = folder
            panel = getattr(self, "r_batch_run_panel", None)
            if panel is not None:
                panel.log(f"Step 2 prepared another folder for a later Step 3 run: {folder}")
            return False
        self._pending_input_folder = None
        self.current_sdb_dir = folder
        self.output_sdb_dir = folder
        self.results = None
        self.original_light_volume = None
        self._refresh_input_status()
        return True

    @staticmethod
    def _folder_key(folder):
        return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(folder))))

    def is_folder_active(self, folder):
        """Return True only for folders read/written by the current R batch."""
        if not folder or not self._busy:
            return False
        return self._folder_key(folder) in getattr(self, "_active_r_folder_keys", set())

    def _clear_plot_holder(self):
        if self.canvas is not None:
            try:
                widget = self.canvas.get_tk_widget() if hasattr(self.canvas, "get_tk_widget") else self.canvas
                widget.destroy()
            except Exception:
                pass
            self.canvas = None
        for child in self.plot_holder.winfo_children():
            child.destroy()
        self.figure = None
        self._preview_photo = None
        self.r_batch_panel = None
        self.r_batch_run_panel = None
        self.batch_results_notebook = None
        self.batch_result_folders = []
        self.batch_result_tab_states = {}
        self._active_batch_result_tab = None

    def _open_r_batch_scanner(self):
        if self._busy:
            return
        initial_dir = self._pending_input_folder or self.current_sdb_dir
        root_dir = filedialog.askdirectory(
            title="Choose a parent folder for flattening",
            initialdir=initial_dir or None,
        )
        if not root_dir:
            return
        self._pending_input_folder = None
        self._show_r_batch_scanner(root_dir)

    @staticmethod
    def _normalize_batch_input_folders(folders):
        """Return unique existing folders in stable handoff order."""
        normalized = []
        seen = set()
        for folder in folders or ():
            path = Path(folder).resolve()
            key = os.path.normcase(str(path))
            if key in seen or not path.is_dir():
                continue
            seen.add(key)
            normalized.append(path)
        return normalized

    def open_batch_folders(self, folders):
        """Open Step 3's selector with the exact folders saved by Step 2."""
        folders = self._normalize_batch_input_folders(folders)
        if not folders:
            messagebox.showwarning("Batch Step 3", "No saved Step 3 input folders were found.")
            return False
        if self._busy:
            previously_queued = getattr(self, "_pending_batch_folders", None) or ()
            self._pending_batch_folders = tuple(
                self._normalize_batch_input_folders([*previously_queued, *folders])
            )
            self._pending_input_folder = None
            panel = getattr(self, "r_batch_run_panel", None)
            if panel is not None:
                panel.log(
                    f"Queued {len(self._pending_batch_folders)} Step 2 output folder(s); "
                    "their selector will open when this R batch finishes."
                )
            return True
        self._pending_batch_folders = None
        self._pending_input_folder = None
        try:
            root_dir = Path(os.path.commonpath([str(folder) for folder in folders]))
        except ValueError:
            root_dir = folders[0].parent
        self.current_sdb_dir = str(root_dir)
        self.output_sdb_dir = str(root_dir)
        self._show_r_batch_scanner(root_dir, folders=folders)
        return True

    def _show_r_batch_scanner(self, root_dir, folders=None):
        """Render the Step 3 folder selector, optionally using an exact folder list."""
        self._clear_plot_holder()
        self.r_batch_panel = RBatchSelectionPanel(
            self,
            self.plot_holder,
            Path(root_dir),
            folders=folders,
        )
        self.r_batch_panel.pack(fill="both", expand=True)
        self.progress_text_var.set("Batch scan")
        self.status_var.set(f"Scanning batch root: {root_dir}")

    def _close_r_batch_panel(self, *, render_previous):
        panel = self.r_batch_panel
        self.r_batch_panel = None
        if panel is not None:
            try:
                panel.destroy()
            except Exception:
                pass
        if render_previous:
            self._render()

    def _close_r_batch_run_panel(self, *, render_previous):
        if self._busy:
            return
        panel = self.r_batch_run_panel
        self.r_batch_run_panel = None
        if panel is not None:
            try:
                panel.destroy()
            except Exception:
                pass
        self.progress_text_var.set("Idle")
        if render_previous:
            self._render()

    def _folder_has_r_data(self, folder):
        folder = Path(folder)
        if any((folder / name).is_file() for name in self.R_WORKSPACE_FILES):
            return True
        return any(path.is_file() for path in folder.glob("*.RData"))

    def _r_script_config_for_folder(self, folder):
        folder = Path(folder)
        input_paths = self._find_input_paths(folder)
        missing = self._missing_input_names(input_paths)
        if missing:
            raise RuntimeError("Missing Step 3 inputs: " + ", ".join(missing))
        input_info = self._read_input_stack_info(input_paths)
        requirement_issues = self._input_requirement_issues(input_paths, input_info)
        if requirement_issues:
            raise RuntimeError("Input requirement issue(s): " + "; ".join(requirement_issues))
        self._validate_input_stack_shapes(input_info)
        return {
            "input_dir": str(folder.resolve()),
            "output_dir": str(folder.resolve()),
            # The app produces one LIGHT acquisition. Preserve the original R
            # script's paired array layout by reusing it for the DARK slots.
            "reference_dark": "DARK_MARKED",
            "reference_light": self._analyze_base_name(input_paths["Light_MARKED"]),
            "to_process_dark": "DARK",
            "to_process_light": self._analyze_base_name(input_paths["LIGHT"]),
            "image_index_light": self._r_index_string(input_info["LIGHT"]["shape"][0]),
            "image_index_dark": self._r_index_string(input_info["LIGHT"]["shape"][0]),
            "pixel_width": str(self.PIXEL_WIDTH_UM),
        }

    def _register_r_process(self, process):
        with self._r_process_lock:
            self._active_r_processes.add(process)

    def _unregister_r_process(self, process):
        with self._r_process_lock:
            self._active_r_processes.discard(process)

    @staticmethod
    def _process_is_running(process):
        try:
            return process.poll() is None
        except Exception:
            return True

    def _terminate_r_process(self, process, *, force=False):
        if not self._process_is_running(process):
            return
        pid = getattr(process, "pid", None)
        try:
            if os.name == "nt" and pid:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
            elif pid:
                os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGTERM)
            elif force and hasattr(process, "kill"):
                process.kill()
            else:
                process.terminate()
        except Exception:
            try:
                process.kill() if force and hasattr(process, "kill") else process.terminate()
            except Exception:
                pass

    def _terminate_active_r_processes(self):
        with self._r_process_lock:
            processes = list(self._active_r_processes)
        for process in processes:
            self._terminate_r_process(process)

    def _cancel_batch_r_runs(self):
        if not self._busy:
            return
        self._r_cancel_event.set()
        self.status_var.set("Cancelling Step 3 R processing...")
        if self.r_batch_run_panel is not None:
            self.r_batch_run_panel.log("Cancellation requested by user.")
        threading.Thread(target=self._terminate_active_r_processes, daemon=True).start()

    def _restart_batch_r_runs(
        self,
        folders,
        workers,
        main_script_path,
        output_script_path,
        timeout_seconds,
        output_mode="parallel",
    ):
        restart = (
            [Path(folder) for folder in folders],
            int(workers),
            Path(main_script_path),
            Path(output_script_path),
            int(timeout_seconds),
            self._normalize_r_output_mode(output_mode),
        )
        if self._busy:
            self._pending_batch_restart = restart
            self.status_var.set("Stopping the current batch before restarting...")
            self._cancel_batch_r_runs()
            return
        self._pending_batch_restart = None
        self._start_batch_r_runs(*restart, allow_existing_rdata=True)

    def _run_supervised_r_command(self, command, cwd, env, timeout_seconds, on_line):
        popen_options = {
            "cwd": cwd,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "env": env,
        }
        if os.name == "nt":
            popen_options["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
            )
        else:
            popen_options["start_new_session"] = True

        try:
            process = subprocess.Popen(command, **popen_options)
        except Exception as exc:
            return 1, str(exc), "failed"

        self._register_r_process(process)
        line_queue = queue.Queue()
        sentinel = object()

        def read_output():
            try:
                if process.stdout is not None:
                    for line in process.stdout:
                        line_queue.put(line)
            except Exception as exc:
                line_queue.put(f"ERROR while reading R output: {exc}\n")
            finally:
                line_queue.put(sentinel)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        timeout_seconds = max(1, int(timeout_seconds))
        deadline = time.monotonic() + timeout_seconds
        stop_reason = None
        output_complete = False

        try:
            while True:
                if self._r_cancel_event.is_set():
                    stop_reason = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    stop_reason = "timed_out"
                    break
                if output_complete:
                    if not self._process_is_running(process):
                        break
                    time.sleep(0.05)
                    continue
                try:
                    item = line_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is sentinel:
                    output_complete = True
                    continue
                on_line(item)

            if stop_reason is not None:
                self._terminate_r_process(process)
                try:
                    process.wait(timeout=10)
                except Exception:
                    self._terminate_r_process(process, force=True)
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        pass
                reader.join(timeout=1)
                while True:
                    try:
                        item = line_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not sentinel:
                        on_line(item)
                if stop_reason == "cancelled":
                    return 130, "Cancelled by user.", stop_reason
                return 124, f"R script exceeded the {timeout_seconds}-second timeout.", stop_reason

            try:
                returncode = process.wait()
                return returncode, "", "completed" if returncode == 0 else "failed"
            except Exception as exc:
                return 1, str(exc), "failed"
        finally:
            self._unregister_r_process(process)

    def _start_batch_r_runs(
        self,
        folders,
        workers,
        main_script_path=None,
        output_script_path=None,
        timeout_seconds=None,
        output_mode="parallel",
        allow_existing_rdata=False,
    ):
        folders = [Path(folder) for folder in folders]
        if not folders:
            messagebox.showwarning("Batch Step 3", "Select at least one folder to process.")
            return
        if self._busy:
            return
        timeout_seconds = max(
            1,
            int(timeout_seconds or (self.DEFAULT_R_SCRIPT_TIMEOUT_MINUTES * 60)),
        )
        main_script_path = (
            Path(main_script_path) if main_script_path is not None else self._selected_r_script_path("main")
        )
        output_script_path = (
            Path(output_script_path) if output_script_path is not None else self._selected_r_script_path("output")
        )
        if main_script_path is None or not main_script_path.is_file():
            messagebox.showerror("Batch Step 3", f"Could not find the main R script:\n{main_script_path}")
            return
        if output_script_path is None or not output_script_path.is_file():
            messagebox.showerror("Batch Step 3", f"Could not find the output R script:\n{output_script_path}")
            return
        rscript = self._ensure_r_ready_with_wizard()
        if rscript is None:
            messagebox.showerror(
                "Step 3 R Setup",
                "R or required packages are not ready. Open Settings > R environment to install or repair them.",
                parent=self,
            )
            return

        workers = max(1, min(int(workers), len(folders), self._r_worker_limit()))
        output_mode = self._normalize_r_output_mode(output_mode)
        self._clear_plot_holder()
        self.r_batch_run_panel = RBatchRunPanel(
            self,
            self.plot_holder,
            folders,
            workers,
            main_script_path,
            output_script_path,
            timeout_seconds,
            output_mode,
        )
        self.r_batch_run_panel.pack(fill="both", expand=True)
        self._r_cancel_event.clear()
        self._busy = True
        self._active_r_folder_keys = {self._folder_key(folder) for folder in folders}
        self._set_process_buttons("disabled")
        self.progress_text_var.set("Batch running")
        output_schedule = (
            "in parallel as each folder finishes its first script"
            if output_mode == self.R_OUTPUT_MODE_PARALLEL
            else "sequentially after all first-script jobs finish"
        )
        self.status_var.set(
            f"Running {main_script_path.name} in parallel, then {output_script_path.name} "
            f"{output_schedule}, for {len(folders)} folder(s)."
        )
        worker_thread = threading.Thread(
            target=self._batch_r_worker,
            args=(
                Path(rscript),
                main_script_path,
                output_script_path,
                folders,
                workers,
                timeout_seconds,
                bool(allow_existing_rdata),
                output_mode,
            ),
            daemon=True,
        )
        try:
            worker_thread.start()
        except Exception as exc:
            self._busy = False
            self._active_r_folder_keys = set()
            self._set_process_buttons("normal")
            self.progress_text_var.set("Batch failed to start")
            self.status_var.set(f"Could not start the Step 3 R batch: {exc}")
            if self.r_batch_run_panel is not None:
                self.r_batch_run_panel.set_summary("Batch failed to start.")
                self.r_batch_run_panel.log(str(exc))
                self.r_batch_run_panel.finish()
            messagebox.showerror("Batch Step 3", f"Could not start the R batch worker:\n{exc}")

    def _batch_panel_update(self, folder, status=None, progress=None, log=None):
        panel = self.r_batch_run_panel
        if panel is None:
            return
        if status is not None or progress is not None:
            panel.update_folder(folder, status=status, progress=progress)
        if log:
            panel.log(log)

    def _r_run_env_for_config(self, r_config, r_thread_limit):
        env = self._r_env(thread_limit=r_thread_limit)
        env.update(
            {
                "AIDAS_STEP3_INPUT_DIR": r_config["input_dir"],
                "AIDAS_STEP3_OUTPUT_DIR": r_config["output_dir"],
                "AIDAS_REFERENCE_DARK": r_config["reference_dark"],
                "AIDAS_REFERENCE_LIGHT": r_config["reference_light"],
                "AIDAS_TO_PROCESS_DARK": r_config["to_process_dark"],
                "AIDAS_TO_PROCESS_LIGHT": r_config["to_process_light"],
                "AIDAS_IMAGE_INDEX_LIGHT": r_config["image_index_light"],
                "AIDAS_IMAGE_INDEX_DARK": r_config["image_index_dark"],
                "AIDAS_PIXEL_WIDTH": r_config["pixel_width"],
            }
        )
        return env

    @staticmethod
    def _cancelled_r_stage_result(folder, message, prior_result=None):
        prior_result = prior_result or {}
        return {
            "folder": Path(folder),
            "returncode": 130,
            "stdout": prior_result.get("stdout", ""),
            "stderr": message,
            "cmd": list(prior_result.get("cmd", [])),
            "outcome": "cancelled",
        }

    def _run_main_r_script_for_config(
        self,
        rscript_path,
        main_script_path,
        r_config,
        batch_folder=None,
        timeout_seconds=None,
        r_thread_limit=None,
    ):
        folder = Path(batch_folder or r_config["input_dir"])
        if self._r_cancel_event.is_set():
            return self._cancelled_r_stage_result(
                folder,
                "Cancelled before the main R script started.",
            )

        script_args = [
            r_config["input_dir"],
            r_config["output_dir"],
            r_config["reference_dark"],
            r_config["reference_light"],
            r_config["to_process_dark"],
            r_config["to_process_light"],
            r_config["image_index_light"],
            r_config["image_index_dark"],
            r_config["pixel_width"],
        ]
        main_cmd = self._build_r_run_command(rscript_path, main_script_path, script_args)
        env = self._r_run_env_for_config(r_config, r_thread_limit)
        output_lines = []
        timeout_seconds = max(
            1,
            int(timeout_seconds or (self.DEFAULT_R_SCRIPT_TIMEOUT_MINUTES * 60)),
        )

        self.after(
            0,
            lambda f=folder: self._batch_panel_update(
                f,
                status="Running main R script",
                progress=1,
                log=f"Starting main R script {Path(main_script_path).name}: {f}",
            ),
        )

        def handle_line(line):
            output_lines.append(line)
            progress = self._progress_from_r_line(line)
            if progress is None:
                return
            percent, label = progress
            self.after(
                0,
                lambda f=folder, p=min(97, percent), s=label: self._batch_panel_update(
                    f,
                    status=s,
                    progress=p,
                ),
            )

        returncode, stderr, outcome = self._run_supervised_r_command(
            main_cmd,
            r_config["input_dir"],
            env,
            timeout_seconds,
            handle_line,
        )
        return {
            "folder": folder,
            "returncode": returncode,
            "stdout": "".join(output_lines),
            "stderr": stderr,
            "cmd": [main_cmd],
            "outcome": "completed" if returncode == 0 else outcome,
        }

    def _run_output_r_script_for_config(
        self,
        rscript_path,
        output_script_path,
        r_config,
        prior_result,
        batch_folder=None,
        timeout_seconds=None,
        r_thread_limit=None,
    ):
        folder = Path(batch_folder or r_config["input_dir"])
        if self._r_cancel_event.is_set():
            return self._cancelled_r_stage_result(
                folder,
                "Cancelled before the output R script started.",
                prior_result,
            )

        output_dir = Path(r_config["output_dir"])
        workspace_path = output_dir / self.R_WORKSPACE_FILES[1]
        output_expression = (
            f"setwd({self._r_string(output_dir.resolve())}); "
            f"load({self._r_string(workspace_path.resolve())}); "
            f"source({self._r_string(Path(output_script_path).resolve())}, chdir=FALSE, echo=FALSE)"
        )
        output_cmd = self._build_r_eval_command(rscript_path, output_expression)
        commands = list(prior_result.get("cmd", [])) + [output_cmd]
        output_lines = [prior_result.get("stdout", "")]
        output_lines.append(f"\n--- Output script: {Path(output_script_path).name} ---\n")
        env = self._r_run_env_for_config(r_config, r_thread_limit)
        timeout_seconds = max(
            1,
            int(timeout_seconds or (self.DEFAULT_R_SCRIPT_TIMEOUT_MINUTES * 60)),
        )

        self.after(
            0,
            lambda f=folder, name=Path(output_script_path).name: self._batch_panel_update(
                f,
                status="Running output R script",
                progress=98,
                log=f"Starting output R script {name}: {f}",
            ),
        )
        returncode, output_stderr, outcome = self._run_supervised_r_command(
            output_cmd,
            r_config["output_dir"],
            env,
            timeout_seconds,
            output_lines.append,
        )
        return {
            "folder": folder,
            "returncode": returncode,
            "stdout": "".join(output_lines),
            "stderr": output_stderr or prior_result.get("stderr", ""),
            "cmd": commands,
            "outcome": "completed" if returncode == 0 else outcome,
        }

    def _finalize_r_script_result(self, r_config, result, *, validate_exports):
        result = dict(result)
        folder = Path(result.get("folder") or r_config["input_dir"])
        returncode = result.get("returncode", 1)
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        outcome = result.get("outcome", "completed" if returncode == 0 else "failed")
        commands = list(result.get("cmd", []))

        if returncode == 0 and validate_exports:
            output_dir = Path(r_config["output_dir"])
            required_exports = (
                output_dir / f"_thickness_vs_distance_from_fovea_{r_config['to_process_dark']}.txt",
                output_dir / f"_thickness_vs_distance_from_fovea_{r_config['to_process_light']}.txt",
            )
            missing_exports = [path.name for path in required_exports if not path.is_file()]
            if missing_exports:
                returncode = 1
                outcome = "failed"
                stdout += (
                    "ERROR: R completed without required thickness export(s): "
                    + ", ".join(missing_exports)
                    + "\n"
                )

        log_path = self._write_r_run_log(
            r_config["output_dir"],
            returncode,
            stdout,
            stderr,
            commands,
        )
        if returncode == 0:
            self.after(
                0,
                lambda f=folder, lp=log_path: self._batch_panel_update(
                    f,
                    log=f"Finished: {f}\nLog: {lp}",
                ),
            )
        else:
            short_output = self._short_process_text(stdout)
            outcome_label = (
                "Cancelled"
                if outcome == "cancelled"
                else "Timed out"
                if outcome == "timed_out"
                else "Failed"
            )
            self.after(
                0,
                lambda f=folder, lp=log_path, out=short_output, label=outcome_label: self._batch_panel_update(
                    f,
                    log=f"{label}: {f}\nLog: {lp}\n{out}",
                ),
            )
        result.update(
            {
                "folder": folder,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "cmd": commands,
                "log": log_path,
                "outcome": "completed" if returncode == 0 else outcome,
            }
        )
        return result

    def _run_r_script_for_config(
        self,
        rscript_path,
        main_script_path,
        output_script_path,
        r_config,
        batch_folder=None,
        timeout_seconds=None,
        r_thread_limit=None,
    ):
        main_result = self._run_main_r_script_for_config(
            rscript_path,
            main_script_path,
            r_config,
            batch_folder=batch_folder,
            timeout_seconds=timeout_seconds,
            r_thread_limit=r_thread_limit,
        )
        if main_result["returncode"] != 0:
            return self._finalize_r_script_result(
                r_config,
                main_result,
                validate_exports=False,
            )
        output_result = self._run_output_r_script_for_config(
            rscript_path,
            output_script_path,
            r_config,
            main_result,
            batch_folder=batch_folder,
            timeout_seconds=timeout_seconds,
            r_thread_limit=r_thread_limit,
        )
        return self._finalize_r_script_result(
            r_config,
            output_result,
            validate_exports=True,
        )

    def _batch_r_worker(
        self,
        rscript_path,
        main_script_path,
        output_script_path,
        folders,
        workers,
        timeout_seconds,
        allow_existing_rdata=False,
        output_mode="parallel",
    ):
        folders = [Path(folder) for folder in folders]
        results_by_folder = {}
        main_successes = {}

        def failed_result(folder, error, prior_result=None, *, outcome="failed", returncode=1):
            prior_result = prior_result or {}
            return {
                "folder": Path(folder),
                "returncode": returncode,
                "stdout": prior_result.get("stdout", ""),
                "stderr": str(error),
                "cmd": list(prior_result.get("cmd", [])),
                "outcome": outcome,
            }

        def record_final(result):
            folder = Path(result["folder"])
            results_by_folder[self._folder_key(folder)] = result
            outcome = result.get(
                "outcome",
                "completed" if result.get("returncode") == 0 else "failed",
            )
            status = {
                "completed": "Completed",
                "cancelled": "Cancelled",
                "timed_out": "Timed out",
                "failed": "Failed",
            }.get(outcome, "Failed")
            self.after(
                0,
                lambda f=folder, s=status: self._batch_panel_update(
                    f,
                    status=s,
                    progress=100 if s == "Completed" else None,
                ),
            )

        def run_main(folder):
            folder = Path(folder)
            if self._r_cancel_event.is_set():
                return None, failed_result(
                    folder,
                    "Cancelled before the folder started.",
                    outcome="cancelled",
                    returncode=130,
                )
            self.after(0, lambda f=folder: self._batch_panel_update(f, status="Validating", progress=0))
            r_config = None
            try:
                if not allow_existing_rdata and self._folder_has_r_data(folder):
                    raise RuntimeError("Skipped because this folder contains RData.")
                r_config = self._r_script_config_for_folder(folder)
                result = self._run_main_r_script_for_config(
                    rscript_path,
                    main_script_path,
                    r_config,
                    batch_folder=folder,
                    timeout_seconds=timeout_seconds,
                    r_thread_limit=main_thread_limit,
                )
            except Exception as exc:
                result = failed_result(folder, exc)
            return r_config, result

        def run_parallel_pipeline(folder):
            folder = Path(folder)
            if self._r_cancel_event.is_set():
                return failed_result(
                    folder,
                    "Cancelled before the folder started.",
                    outcome="cancelled",
                    returncode=130,
                )
            self.after(0, lambda f=folder: self._batch_panel_update(f, status="Validating", progress=0))
            try:
                if not allow_existing_rdata and self._folder_has_r_data(folder):
                    raise RuntimeError("Skipped because this folder contains RData.")
                r_config = self._r_script_config_for_folder(folder)
                return self._run_r_script_for_config(
                    rscript_path,
                    main_script_path,
                    output_script_path,
                    r_config,
                    batch_folder=folder,
                    timeout_seconds=timeout_seconds,
                    r_thread_limit=main_thread_limit,
                )
            except Exception as exc:
                return failed_result(folder, exc)

        def run_output(item, output_thread_limit):
            folder, r_config, main_result = item
            try:
                output_result = self._run_output_r_script_for_config(
                    rscript_path,
                    output_script_path,
                    r_config,
                    main_result,
                    batch_folder=folder,
                    timeout_seconds=timeout_seconds,
                    r_thread_limit=output_thread_limit,
                )
                return self._finalize_r_script_result(
                    r_config,
                    output_result,
                    validate_exports=True,
                )
            except Exception as exc:
                failure = failed_result(folder, exc, main_result)
                try:
                    return self._finalize_r_script_result(
                        r_config,
                        failure,
                        validate_exports=False,
                    )
                except Exception:
                    return failure

        try:
            workers = max(1, min(int(workers), len(folders), self._r_worker_limit()))
            output_mode = self._normalize_r_output_mode(output_mode)
            main_thread_limit = self._r_threads_per_process(workers)

            # Preserve the original/default behavior: each worker processes a
            # complete main -> output folder pipeline. That lets the output
            # script for a fast folder overlap another folder's main script.
            if output_mode == self.R_OUTPUT_MODE_PARALLEL:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    future_map = {
                        executor.submit(run_parallel_pipeline, folder): folder
                        for folder in folders
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        folder = future_map[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = failed_result(folder, exc)
                        record_final(result)
                return

            # Sequential-output mode has an explicit phase barrier: every main
            # script settles in parallel before outputs run one folder at a time.
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {executor.submit(run_main, folder): folder for folder in folders}
                for future in concurrent.futures.as_completed(future_map):
                    folder = future_map[future]
                    try:
                        r_config, result = future.result()
                    except Exception as exc:
                        r_config = None
                        result = failed_result(folder, exc)

                    if result["returncode"] == 0 and r_config is not None:
                        main_successes[self._folder_key(folder)] = (folder, r_config, result)
                        self.after(
                            0,
                            lambda f=folder: self._batch_panel_update(
                                f,
                                status="Waiting for output R script",
                                progress=97,
                                log=f"Main R script finished; waiting for the output phase: {f}",
                            ),
                        )
                        continue

                    if r_config is not None:
                        try:
                            result = self._finalize_r_script_result(
                                r_config,
                                result,
                                validate_exports=False,
                            )
                        except Exception:
                            pass
                    record_final(result)

            output_items = [
                main_successes[self._folder_key(folder)]
                for folder in folders
                if self._folder_key(folder) in main_successes
            ]
            if output_items:
                output_thread_limit = self._r_threads_per_process(1)
                for item in output_items:
                    record_final(run_output(item, output_thread_limit))
        except Exception as exc:
            for folder in folders:
                folder_key = self._folder_key(folder)
                if folder_key in results_by_folder:
                    continue
                prior_result = None
                if folder_key in main_successes:
                    prior_result = main_successes[folder_key][2]
                record_final(
                    failed_result(
                        folder,
                        f"Batch coordinator failed: {exc}",
                        prior_result,
                    )
                )
        finally:
            for folder in folders:
                folder_key = self._folder_key(folder)
                if folder_key in results_by_folder:
                    continue
                record_final(
                    failed_result(
                        folder,
                        "Batch coordinator finished without a result for this folder.",
                    )
                )
            finished_results = [
                results_by_folder[self._folder_key(folder)]
                for folder in folders
            ]
            self.after(0, lambda batch_results=finished_results: self._on_batch_r_done(batch_results))

    def _on_batch_r_done(self, results):
        self._busy = False
        self._active_r_folder_keys = set()
        self._set_process_buttons("normal")
        panel = self.r_batch_run_panel
        close_when_finished = bool(panel is not None and panel.close_when_finished)
        outcomes = [
            result.get("outcome", "completed" if result["returncode"] == 0 else "failed")
            for result in results
        ]
        success = outcomes.count("completed")
        cancelled = outcomes.count("cancelled")
        timed_out = outcomes.count("timed_out")
        failed = outcomes.count("failed")
        self.progress_text_var.set("Batch completed")
        summary = (
            f"Batch complete: {success} succeeded, {failed} failed, "
            f"{timed_out} timed out, {cancelled} cancelled."
        )
        self.status_var.set(summary)
        if panel is not None:
            panel.set_summary(summary)
            panel.finish()
            panel.log(summary)
        self.info_var.set(
            "Batch Step 3 R results:\n"
            + "\n".join(
                f"{result.get('outcome', 'completed' if result['returncode'] == 0 else 'failed').upper()}: "
                f"{result['folder']}"
                for result in results
            )
        )

        pending_restart = self._pending_batch_restart
        self._pending_batch_restart = None
        if pending_restart is not None:
            self.progress_text_var.set("Restarting batch")
            self.status_var.set("Starting a clean Batch Step 3 run...")
            if panel is not None:
                panel.set_summary("Starting a clean batch run...")
                panel.log("The previous run stopped. Starting a clean batch run.")
            # Start before returning to Tk's event loop so Step 2 cannot save
            # into the restart folders during a transient unreserved gap.
            self._start_batch_r_runs(
                *pending_restart,
                allow_existing_rdata=True,
            )
            return

        pending_folders = self._pending_batch_folders
        self._pending_batch_folders = None
        if pending_folders:
            folders = self._normalize_batch_input_folders(pending_folders)
            if folders:
                try:
                    root_dir = Path(os.path.commonpath([str(folder) for folder in folders]))
                except ValueError:
                    root_dir = folders[0].parent
                self._pending_input_folder = None
                self.current_sdb_dir = str(root_dir)
                self.output_sdb_dir = str(root_dir)
                self._show_r_batch_scanner(root_dir, folders=folders)
                return

        if close_when_finished:
            self._close_r_batch_run_panel(render_previous=True)
            return

        successful_folders = [Path(result["folder"]) for result in results if result["returncode"] == 0]
        if successful_folders:
            self._open_batch_r_result_tabs(successful_folders)

    def _write_r_run_log(self, output_dir, returncode, stdout, stderr, cmd):
        log_path = app_log_dir() / f"step3_rscript_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
        commands = cmd if cmd and isinstance(cmd[0], (list, tuple)) else [cmd]
        command_text = "\n".join(
            f"{index}. " + " ".join(str(part) for part in command)
            for index, command in enumerate(commands, start=1)
        )
        log_path.write_text(
            "Commands:\n"
            + command_text
            + f"\n\nOutput directory:\n{output_dir}"
            + f"\n\nReturn code: {returncode}\n\nSTDOUT:\n{stdout or ''}\n\nSTDERR:\n{stderr or ''}\n",
            encoding="utf-8",
        )
        return log_path

    @staticmethod
    def _resource_path(relative_path):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        return base_dir / relative_path

    def _tutorial_asset_path(self):
        return self._resource_path(Path("assets") / self.TUTORIAL_IMAGE_NAME)

    def _display_preview_image(self, image, background="#ffffff", parent=None):
        parent = self.plot_holder if parent is None else parent
        label = tk.Label(parent, bg=background, borderwidth=0, highlightthickness=0)
        label.pack(fill="both", expand=True)
        source = image.convert("RGB")

        def redraw(_event=None):
            try:
                if not label.winfo_exists():
                    return
                width = max(1, int(label.winfo_width()))
                height = max(1, int(label.winfo_height()))
            except tk.TclError:
                return
            if width <= 1 or height <= 1:
                return
            fitted = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), background)
            canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
            label.preview_photo = ImageTk.PhotoImage(canvas)
            try:
                label.configure(image=label.preview_photo)
            except tk.TclError:
                return

        label.bind("<Configure>", redraw, add="+")
        if parent is self.plot_holder:
            self.canvas = label
        self.after(0, redraw)

    def _batch_result_tab_name_limit(self):
        notebook = self.batch_results_notebook
        if notebook is None:
            return 18
        try:
            tab_count = max(1, len(notebook.tabs()))
            width = max(260, notebook.winfo_width())
        except tk.TclError:
            return 18
        per_tab = max(70, width // tab_count)
        return max(6, min(18, (per_tab - 54) // 7))

    @staticmethod
    def _compact_batch_result_name(name, limit):
        name = str(name or "Folder")
        if len(name) <= limit:
            return name
        if limit <= 3:
            return name[:limit]
        return f"{name[: limit - 3]}..."

    def _batch_result_tab_text(self, state, *, active=False):
        folder = Path(state.get("folder") or "")
        raw_label = state.get("base_label") or folder.name or str(folder)
        if ". " in raw_label:
            prefix, name = raw_label.split(". ", 1)
            tab_name = name if active else self._compact_batch_result_name(name, self._batch_result_tab_name_limit())
            label = f"{prefix}. {tab_name}"
        else:
            label = raw_label if active else self._compact_batch_result_name(raw_label, self._batch_result_tab_name_limit())
        return label

    def _refresh_batch_result_tab_labels(self):
        notebook = self.batch_results_notebook
        if notebook is None:
            return
        for tab_id in notebook.tabs():
            try:
                tab_key = str(notebook.nametowidget(tab_id))
            except tk.TclError:
                continue
            state = self.batch_result_tab_states.get(tab_key)
            if state is None:
                continue
            try:
                notebook.tab(tab_id, text=self._batch_result_tab_text(state, active=tab_key == self._active_batch_result_tab))
            except tk.TclError:
                pass

    def _on_batch_result_tab_changed(self, _notebook, tab):
        self._active_batch_result_tab = str(tab)
        self._refresh_batch_result_tab_labels()

    def _close_batch_result_tab(self, notebook, tab):
        tab_key = str(tab)
        state = self.batch_result_tab_states.pop(tab_key, None)
        if state is not None:
            folder = Path(state["folder"])
            self.batch_result_folders = [item for item in self.batch_result_folders if Path(item) != folder]
        try:
            notebook.forget(tab)
        except tk.TclError:
            return
        if tab_key == self._active_batch_result_tab:
            self._active_batch_result_tab = None
            tabs = notebook.tabs()
            if tabs:
                notebook.select(tabs[0])
            else:
                self.batch_results_notebook = None
                self.batch_result_folders = []
                self._render()

    @staticmethod
    def _result_png_name_for_view(view):
        if view == "DARK_MARKED_find_vertex":
            return "DARK_MARKED_find_vertex.png"
        if view == "_tissueBorders__DARK":
            return "_tissueBorders__DARK.png"
        return None

    def _render_result_image_for_folder(self, parent, folder):
        view = self.view_var.get()
        filename = self._result_png_name_for_view(view)
        if filename is None:
            image = _placeholder_image(
                f"Unknown Step 3 view:\n{view}",
                size=(1600, 1000),
                title="Step 3 Results",
            )
        else:
            try:
                image = self._load_result_png_from_folder(folder, filename)
            except Exception as exc:
                image = _placeholder_image(
                    f"Could not load {filename}:\n{exc}",
                    size=(1600, 1000),
                    title=filename,
                )
        self._display_preview_image(image, parent=parent)

    def _open_batch_r_result_tabs(self, folders):
        folders = [Path(folder) for folder in folders]
        if not folders:
            return
        self._clear_plot_holder()
        self.batch_result_folders = folders
        self.batch_result_tab_states = {}
        self._active_batch_result_tab = None
        notebook = ClosableTabView(
            self.plot_holder,
            command=self._on_batch_result_tab_changed,
            close_command=self._close_batch_result_tab,
        )
        notebook.pack(fill="both", expand=True)
        notebook.bind(
            "<Configure>",
            lambda _event: self._refresh_batch_result_tab_labels(),
            add="+",
        )
        self.batch_results_notebook = notebook

        for index, folder in enumerate(folders, start=1):
            state = {
                "folder": folder,
                "base_label": f"{index}. {folder.name or folder}",
            }
            frame = notebook.add(text=self._batch_result_tab_text(state))
            tab_key = str(frame)
            self.batch_result_tab_states[tab_key] = state
            ttk.Label(frame, text=str(folder), anchor="w", padding=4).pack(fill="x")
            image_host = ttk.Frame(frame)
            image_host.pack(fill="both", expand=True)
            self._render_result_image_for_folder(image_host, folder)

        first_tab = notebook.tabs()[0] if notebook.tabs() else None
        if first_tab:
            notebook.select(first_tab)

        self.current_sdb_dir = str(folders[0])
        self.output_sdb_dir = str(folders[0])
        self.results = None
        self.original_light_volume = None
        self.status_var.set(f"Opened Step 3 results for {len(folders)} folder(s).")
        self.info_var.set("Batch Step 3 R results opened:\n" + "\n".join(str(folder) for folder in folders))

    def _on_view_selected(self):
        if self.batch_result_folders:
            self._open_batch_r_result_tabs(self.batch_result_folders)
        elif self.results is not None:
            self._render()

    def _render_tutorial(self):
        tutorial_path = self._tutorial_asset_path()
        if tutorial_path.is_file():
            with Image.open(tutorial_path) as img:
                image = img.copy()
        else:
            image = _placeholder_image(
                f"Missing Step 3 tutorial asset:\n{tutorial_path}",
                size=(1800, 1100),
                title="Step 3 Tutorial",
            )
            self.status_var.set(f"Step 3 tutorial image not found: {tutorial_path}")
        self.info_var.set("")
        if tutorial_path.is_file():
            self.status_var.set("Step 3 tutorial: using static asset image.")
        image = self._tutorial_image_for_appearance(image)
        self._display_preview_image(
            image,
            background=resolve_color(COLOR_PAIRS["surface"]),
        )

    @staticmethod
    def _tutorial_image_for_appearance(image, appearance_mode=None):
        """Return the tutorial in its exact light or contrast-safe dark form."""

        surface_hex = resolve_color(COLOR_PAIRS["surface"], appearance_mode)
        if surface_hex.lower() == COLOR_PAIRS["surface"][0].lower():
            return image.convert("RGB")

        source = np.asarray(image.convert("RGB"), dtype=np.float32)
        result = source.copy()
        maximum = source.max(axis=2)
        minimum = source.min(axis=2)
        chroma = maximum - minimum
        luminance = (
            source[:, :, 0] * 0.2126
            + source[:, :, 1] * 0.7152
            + source[:, :, 2] * 0.0722
        )

        surface = np.asarray(ImageColor.getrgb(surface_hex), dtype=np.float32)
        text = np.asarray(
            ImageColor.getrgb(resolve_color(COLOR_PAIRS["text"], "Dark")),
            dtype=np.float32,
        )

        # Invert only neutral luminosity: white becomes the dark surface,
        # black becomes light text, and antialiased edges remain smooth.
        neutral = chroma <= 20
        neutral_mix = luminance[:, :, None] / 255.0
        neutral_result = text * (1.0 - neutral_mix) + surface * neutral_mix
        result[neutral] = neutral_result[neutral]

        # Tone down pale diagram regions and lift dark colored annotations so
        # the blue, red, and gold meanings remain distinct on the dark canvas.
        colored = ~neutral
        pale_colored = colored & (luminance >= 175)
        toned_color = surface + (source - surface) * 0.38
        result[pale_colored] = toned_color[pale_colored]
        dark_colored = colored & (luminance < 135)
        lifted_color = source * 0.78 + text * 0.22
        result[dark_colored] = lifted_color[dark_colored]

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")

    def refresh_appearance(self):
        """Redraw theme-dependent tutorial content without closing active work."""

        panels = (
            self.r_setup_panel,
            self.r_batch_panel,
            self.r_batch_run_panel,
            self.batch_results_notebook,
        )
        if self.results is None and not self.batch_result_folders and not any(panels):
            self._render()

    def _result_info_text(self):
        if self.results is None:
            return ""
        return (
            f"flattened_dark: {self.results['flattened_dark'].shape}\n"
            f"flattened_light: {self.results['flattened_light'].shape}\n"
            f"final_grand_mean: {self.results['final_grand_mean'].shape}\n"
            f"vertex: {self.results['vertex']}"
        )

    def _render(self):
        view = self.view_var.get()
        self._clear_plot_holder()

        if self.results is None:
            self._render_tutorial()
            return

        if view == "DARK_MARKED_find_vertex":
            try:
                image = self._load_result_png("DARK_MARKED_find_vertex.png")
                self.status_var.set("Showing DARK_MARKED_find_vertex.png.")
            except Exception as exc:
                image = _placeholder_image(
                    f"Could not load DARK_MARKED_find_vertex.png:\n{exc}",
                    size=(1600, 1000),
                    title="DARK_MARKED_find_vertex.png",
                )
                self.status_var.set("Could not load DARK_MARKED_find_vertex.png.")
        elif view == "_tissueBorders__DARK":
            try:
                image = self._load_result_png("_tissueBorders__DARK.png")
                self.status_var.set("Showing _tissueBorders__DARK.png.")
            except Exception as exc:
                image = _placeholder_image(
                    f"Could not load _tissueBorders__DARK.png:\n{exc}",
                    size=(1600, 1000),
                    title="_tissueBorders__DARK.png",
                )
                self.status_var.set("Could not load _tissueBorders__DARK.png.")
        else:
            image = _placeholder_image(
                f"Unknown Step 3 view:\n{view}",
                size=(1600, 1000),
                title="Step 3 Results",
            )
            self.status_var.set("Unknown Step 3 results view.")

        self._display_preview_image(image)
        self.info_var.set(self._result_info_text())

    def _tutorial_info_text(self):
        left_px = int(np.ceil(self.MIN_NEGATIVE_UM / self.PIXEL_WIDTH_UM))
        right_px = int(np.ceil(self.MIN_POSITIVE_UM / self.PIXEL_WIDTH_UM))
        source_width_px = int(np.ceil((self.MIN_NEGATIVE_UM + self.MIN_POSITIVE_UM) / self.PIXEL_WIDTH_UM))
        outward_px = int(np.ceil(self.MIN_DEPTH_OUTWARD_UM / self.PIXEL_WIDTH_UM))
        inward_px = int(np.ceil(self.MIN_DEPTH_INWARD_UM / self.PIXEL_WIDTH_UM))
        safe_centered_side_px = right_px + self.CENTERED_FOVEA_GUARD_PX
        return (
            "Step 3 tutorial minimums:\n"
            f"Pixel width: {self.PIXEL_WIDTH_UM:g} um/input px\n"
            f"Fovea to near side: >= {left_px} px ({self.MIN_NEGATIVE_UM:g} um)\n"
            f"Fovea to far side: >= {right_px} px ({self.MIN_POSITIVE_UM:g} um)\n"
            f"Minimum RPE marker coverage: about {source_width_px} px\n"
            f"Centered fovea minimum: >= {right_px * 2} px\n"
            f"Centered fovea recommended: >= {safe_centered_side_px * 2} px "
            f"({safe_centered_side_px} px per side)\n"
            f"Height around RPE: >= {inward_px} px from top and >= {outward_px} px from bottom\n"
            f"Centered RPE height: >= {inward_px * 2} px"
        )
