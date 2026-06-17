"""Step 3 - batch OCT flattening with the original R workflow."""

from __future__ import annotations

import shutil
import subprocess
import re
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

from PIL import Image, ImageOps, ImageTk

try:
    import pyreadr
except Exception:
    pyreadr = None

from aidas.utils.io_utils import read_analyze
from aidas.utils.batch_ui import BatchTable
from aidas.utils.step3_image_utils import (
    placeholder_image as _placeholder_image,
)
from aidas.utils.log_paths import app_log_dir
from aidas.utils.ui_utils import SidebarStepFrame


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

    STEPS = (
        "Welcome",
        "R Program",
        "Download R",
        "Package Library",
        "Packages",
        "Finish",
    )

    def __init__(self, step_frame, parent, on_finish=None):
        super().__init__(parent)
        self.step_frame = step_frame
        self.on_finish = on_finish
        self.result = None
        self.cancelled = True
        self.current_step = 0
        self.busy = False
        self.rscript_path = step_frame._resolve_rscript_executable()
        self.installer_name = ""
        self.installer_url = ""
        self.installer_path = None
        self.package_status = {name: "pending" for name in step_frame.R_REQUIRED_PACKAGES}
        self.package_library_path = Path(
            getattr(step_frame, "r_package_library_path", None)
            or self._default_package_library()
        )
        self.log_path = self._package_log_path()

        self._build_styles()
        self._build_shell()
        self._render_step()
        self.focus_set()

    def _build_styles(self):
        self.style = ttk.Style(self)
        self.style.configure("WizardTitle.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("WizardSubtitle.TLabel", foreground="#555555")
        self.style.configure("WizardStep.TLabel", padding=(10, 7))
        self.style.configure("WizardStepActive.TLabel", padding=(10, 7), font=("Segoe UI", 9, "bold"))
        self.style.configure("WizardStepDone.TLabel", padding=(10, 7), foreground="#1b6e3c")
        self.style.configure("WizardAccent.TButton", padding=(10, 5))

    def _build_shell(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Install R for Step 3", style="WizardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="A guided setup for R, package libraries, and the packages required by this step.",
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

        self.progress = ttk.Progressbar(right, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(6, 0))

        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(10, 0))
        self.back_button = ttk.Button(footer, text="Back", command=self._back)
        self.back_button.pack(side="right", padx=(4, 0))
        self.next_button = ttk.Button(footer, text="Next", command=self._next)
        self.next_button.pack(side="right", padx=(4, 0))
        self.cancel_button = ttk.Button(footer, text="Cancel", command=self._cancel)
        self.cancel_button.pack(side="right")

        self._log("R setup wizard opened.")

    def _clear_content(self):
        for child in self.content.winfo_children():
            child.destroy()

    def _set_busy(self, busy, text=None, indeterminate=False):
        self.busy = bool(busy)
        for button in (self.back_button, self.next_button, self.cancel_button):
            button.configure(state="disabled" if busy else "normal")
        if indeterminate:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
        if text:
            self._log(text)
        if not busy:
            self._update_nav()

    def _set_progress(self, value):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=max(0, min(100, float(value))))

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
        self.next_button.configure(text="Finish" if self.current_step == len(self.STEPS) - 1 else "Next")
        if self.current_step == 1 and self.rscript_path is None:
            self.next_button.configure(state="disabled")
        elif self.current_step == 2 and self.rscript_path is None:
            self.next_button.configure(state="disabled")
        elif self.current_step == 4 and not self._all_packages_ready():
            self.next_button.configure(state="disabled")
        else:
            self.next_button.configure(state="normal")

    def _render_step(self):
        self._clear_content()
        renderers = (
            self._render_welcome,
            self._render_r_program,
            self._render_download,
            self._render_library,
            self._render_packages,
            self._render_finish,
        )
        renderers[self.current_step]()
        self._set_progress((self.current_step / max(1, len(self.STEPS) - 1)) * 100)
        self._update_nav()

    def _section_title(self, title, subtitle):
        ttk.Label(self.content, text=title, style="WizardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            self.content,
            text=subtitle,
            style="WizardSubtitle.TLabel",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(4, 14))

    def _render_welcome(self):
        self._section_title(
            "Welcome",
            "This wizard installs the R runtime and the required R packages for Step 3.",
        )
        body = ttk.Frame(self.content)
        body.pack(fill="both", expand=True)
        items = (
            "Detect an existing Rscript executable.",
            "Download the official Windows R installer from CRAN if R is missing.",
            "Run the R installer and re-check the installed program.",
            "Choose a package-library folder that does not require administrator rights.",
            "Install AnalyzeFMRI and RNiftyReg with binary packages from CRAN.",
        )
        for item in items:
            ttk.Label(body, text=f"- {item}", wraplength=620, justify="left").pack(anchor="w", pady=3)
        ttk.Label(
            body,
            text=f"Full setup log:\n{self.log_path}",
            foreground="#555555",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(18, 0))

    def _render_r_program(self):
        self._section_title(
            "R Program",
            "AIDaS needs Rscript.exe to run the original Step 3 R workflow non-interactively.",
        )
        status = "Not found"
        if self.rscript_path is not None:
            status = str(self.rscript_path)
        self.r_status_var = tk.StringVar(value=status)

        form = ttk.LabelFrame(self.content, text="Detected Rscript")
        form.pack(fill="x")
        ttk.Label(form, textvariable=self.r_status_var, wraplength=620, justify="left").pack(
            anchor="w", padx=10, pady=10
        )

        actions = ttk.Frame(self.content)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Check Again", command=self._detect_rscript).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Locate Rscript...", command=self._locate_rscript).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Download R...", command=self._go_download).pack(side="left")

        ttk.Label(
            self.content,
            text="Choose Download R if R is not installed. Choose Locate Rscript if R is already installed but AIDaS cannot find it.",
            foreground="#555555",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _render_download(self):
        self._section_title(
            "Download And Install R",
            "Download the official Windows installer from CRAN, then run it. The R installer will ask where and how to install R.",
        )
        self.installer_path_var = tk.StringVar(value=str(self.installer_path or ""))
        self.installer_info_var = tk.StringVar(value=self.installer_name or "Installer has not been selected yet.")

        info = ttk.LabelFrame(self.content, text="Installer")
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.installer_info_var, wraplength=620, justify="left").pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        row = ttk.Frame(info)
        row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Entry(row, textvariable=self.installer_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Save As...", command=self._choose_installer_save_path).pack(side="left", padx=(6, 0))

        actions = ttk.Frame(self.content)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Find Latest Installer", command=self._find_latest_installer).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="Download Installer", command=self._download_selected_installer).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="Run Installer", command=self._run_downloaded_installer).pack(side="left")

        ttk.Label(
            self.content,
            text="After the installer finishes, this wizard checks again for Rscript.exe.",
            foreground="#555555",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _render_library(self):
        self._section_title(
            "Package Library",
            "R packages should be installed in a folder the current user can write to.",
        )
        self.library_var = tk.StringVar(value=str(self.package_library_path))
        frame = ttk.LabelFrame(self.content, text="R package-library folder")
        frame.pack(fill="x")
        row = ttk.Frame(frame)
        row.pack(fill="x", padx=10, pady=10)
        ttk.Entry(row, textvariable=self.library_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=self._browse_library).pack(side="left", padx=(6, 0))
        ttk.Label(
            self.content,
            text="Recommended: use the AIDaS folder under Local AppData. This avoids administrator permissions and keeps Step 3 packages separate from system R.",
            foreground="#555555",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _render_packages(self):
        self._section_title(
            "Required Packages",
            "Install the R packages used by the original Step 3 script.",
        )
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
        ttk.Button(actions, text="Install Missing Packages", command=self._install_missing_packages).pack(side="left")

        ttk.Label(
            self.content,
            text=f"Packages will be installed to:\n{self.package_library_path}",
            foreground="#555555",
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
        if self.current_step == 3:
            if not self._save_library_choice():
                return
        if self.current_step == len(self.STEPS) - 1:
            self._finish()
            return
        if self.current_step == 1 and self.rscript_path is not None:
            self.current_step = 3
        else:
            self.current_step = min(len(self.STEPS) - 1, self.current_step + 1)
        if self.current_step == 2 and self.installer_url == "":
            self.after(100, self._find_latest_installer)
        if self.current_step == 4:
            self.after(100, self._check_packages)
        self._render_step()

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
        self.step_frame._close_r_setup_panel(render_previous=True)

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
        self.step_frame._close_r_setup_panel(render_previous=callback is None)
        if callback is not None:
            self.step_frame.after(0, lambda: callback(result))

    def _go_download(self):
        self.current_step = 2
        self._render_step()
        if not self.installer_url:
            self._find_latest_installer()

    def _default_package_library(self):
        configured = getattr(self.step_frame, "r_package_library_path", None)
        if configured:
            return Path(configured)
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "AIDaS" / "R-packages"
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
        self._set_busy(True, title, indeterminate=True)

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
            self.r_status_var.set(str(self.rscript_path) if self.rscript_path else "Not found")
        self._log(f"Rscript detection: {self.rscript_path or 'not found'}")
        self._update_nav()

    def _locate_rscript(self):
        selected = filedialog.askopenfilename(
            title="Locate Rscript executable",
            initialdir=r"C:\Program Files\R" if os.name == "nt" else (self.step_frame.current_sdb_dir or None),
            filetypes=[("Rscript executable", "Rscript*.exe"), ("All files", "*.*")],
        )
        if not selected:
            return
        rscript = self.step_frame._normalize_r_executable(Path(selected))
        if rscript is None:
            messagebox.showerror(
                "Locate Rscript executable",
                "Please select Rscript.exe, not R.exe, Rgui.exe, or RStudio.",
                parent=self,
            )
            return
        self.rscript_path = rscript
        if self.step_frame.preferences is not None:
            self.step_frame.preferences.set("rscript_path", str(rscript))
        self._detect_rscript()

    def _find_latest_installer(self):
        def worker():
            with urllib.request.urlopen(self.step_frame.R_DOWNLOAD_PAGE, timeout=30) as response:
                html = response.read().decode("utf-8", errors="replace")
            installers = sorted(
                set(re.findall(r'href=["\'](R-[0-9][^"\']+-win\.exe)["\']', html)),
                key=lambda name: [int(part) for part in re.findall(r"\d+", name)],
            )
            if not installers:
                raise RuntimeError(f"No Windows installer found at {self.step_frame.R_DOWNLOAD_PAGE}")
            name = installers[-1]
            return name, self.step_frame.R_DOWNLOAD_PAGE + name

        def done(value, error):
            if error:
                self._log(f"Could not find latest R installer: {error}")
                messagebox.showerror("R Setup", f"Could not find the latest R installer.\n{error}", parent=self)
                return
            self.installer_name, self.installer_url = value
            default_dir = Path(self.step_frame.current_sdb_dir or os.getcwd())
            self.installer_path = default_dir / self.installer_name
            if hasattr(self, "installer_info_var"):
                self.installer_info_var.set(f"{self.installer_name}\n{self.installer_url}")
            if hasattr(self, "installer_path_var"):
                self.installer_path_var.set(str(self.installer_path))
            self._log(f"Latest R installer: {self.installer_name}")

        self._run_worker("Finding latest R installer...", worker, done)

    def _choose_installer_save_path(self):
        initial_file = self.installer_name or "R-installer.exe"
        selected = filedialog.asksaveasfilename(
            title="Save R installer as",
            initialdir=self.step_frame.current_sdb_dir or None,
            initialfile=initial_file,
            defaultextension=".exe",
            filetypes=[("Windows installer", "*.exe"), ("All files", "*.*")],
            parent=self,
        )
        if selected:
            self.installer_path = Path(selected)
            self.installer_path_var.set(str(self.installer_path))
            self._log(f"Installer save path selected: {self.installer_path}")

    def _download_selected_installer(self):
        if not self.installer_url:
            messagebox.showwarning("R Setup", "Find the latest installer first.", parent=self)
            return
        path_text = self.installer_path_var.get().strip() if hasattr(self, "installer_path_var") else ""
        if not path_text:
            self._choose_installer_save_path()
            path_text = self.installer_path_var.get().strip()
        if not path_text:
            return
        self.installer_path = Path(path_text)
        if self.installer_path.exists():
            overwrite = messagebox.askyesno(
                "Overwrite Installer",
                f"This file already exists:\n{self.installer_path}\n\nOverwrite it?",
                parent=self,
            )
            if not overwrite:
                return

        def worker():
            self.installer_path.parent.mkdir(parents=True, exist_ok=True)

            def reporthook(block_count, block_size, total_size):
                if total_size > 0:
                    percent = min(100.0, (block_count * block_size / total_size) * 100.0)
                    self.after(0, lambda p=percent: self._set_progress(p))

            urllib.request.urlretrieve(self.installer_url, self.installer_path, reporthook=reporthook)
            return self.installer_path

        def done(value, error):
            if error:
                self._log(f"R installer download failed: {error}")
                messagebox.showerror("R Setup", f"Could not download R.\n{error}", parent=self)
                return
            self._log(f"Downloaded R installer: {value}")
            messagebox.showinfo("R Setup", "R installer downloaded. You can run it now.", parent=self)

        self._run_worker("Downloading R installer...", worker, done)

    def _run_downloaded_installer(self):
        path_text = self.installer_path_var.get().strip() if hasattr(self, "installer_path_var") else ""
        installer_path = Path(path_text) if path_text else self.installer_path
        if not installer_path or not installer_path.is_file():
            messagebox.showwarning("R Setup", "Download or select the R installer first.", parent=self)
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
                messagebox.showinfo("R Setup", "Rscript was found. Continue to package setup.", parent=self)
                self.current_step = 3
                self._render_step()
            else:
                messagebox.showwarning(
                    "R Setup",
                    "AIDaS still cannot find Rscript. Finish the installer if it is still open, then click Check Again or Locate Rscript.",
                    parent=self,
                )

        self._run_worker("Running R installer. Complete the installer window to continue.", worker, done)

    def _browse_library(self):
        selected = filedialog.askdirectory(
            title="Select R package-library folder",
            initialdir=str(Path(self.library_var.get()).parent) if self.library_var.get() else None,
            parent=self,
        )
        if selected:
            self.library_var.set(selected)

    def _save_library_choice(self):
        library_path = Path(self.library_var.get().strip())
        if not str(library_path):
            messagebox.showwarning("Package Library", "Choose a package-library folder.", parent=self)
            return False
        try:
            library_path.mkdir(parents=True, exist_ok=True)
            test_path = library_path / ".aidas_write_test"
            test_path.write_text("ok", encoding="utf-8")
            test_path.unlink()
        except Exception as exc:
            messagebox.showerror(
                "Package Library",
                f"This folder is not writable:\n{library_path}\n\n{exc}",
                parent=self,
            )
            return False
        self.package_library_path = library_path.resolve()
        self.step_frame.r_package_library_path = str(self.package_library_path)
        if self.step_frame.preferences is not None:
            self.step_frame.preferences.set("r_package_library_path", str(self.package_library_path))
        self._log(f"Package library selected: {self.package_library_path}")
        return True

    def _package_check_expression(self, package_name):
        lib = self._r_string(self.package_library_path.resolve())
        return (
            f".libPaths(c({lib}, .libPaths())); "
            f"if (requireNamespace({self._r_string(package_name)}, quietly=TRUE)) "
            "quit(status=0) else quit(status=1)"
        )

    def _package_install_expression(self, package_name):
        lib = self._r_string(self.package_library_path.resolve())
        type_arg = ", type='binary'" if os.name == "nt" else ""
        pkg_type = "options(pkgType='win.binary'); " if os.name == "nt" else ""
        return "".join(
            (
                f".libPaths(c({lib}, .libPaths())); ",
                "options(repos=c(CRAN='https://cloud.r-project.org')); ",
                "options(install.packages.compile.from.source='never'); ",
                pkg_type,
                f"install.packages({self._r_string(package_name)}, ",
                "dependencies=c('Depends','Imports','LinkingTo'), ",
                f"lib={lib}{type_arg})",
            )
        )

    def _check_package_status_worker(self):
        if self.rscript_path is None:
            raise RuntimeError("Rscript is not selected.")
        self.package_library_path.mkdir(parents=True, exist_ok=True)
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
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._log_process_result(f"Check package {package_name}", cmd, result)
            statuses[package_name] = "installed" if result.returncode == 0 else "missing"
        return statuses

    def _check_packages(self):
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
        for package_name, status in list(statuses.items()):
            if status == "installed":
                continue
            expression = self._package_install_expression(package_name)
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
            check_result = subprocess.run(
                self._r_eval_command(self._package_check_expression(package_name)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            statuses[package_name] = "installed" if result.returncode == 0 and check_result.returncode == 0 else "failed"
        return statuses

    def _install_missing_packages(self):
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


class RBatchSelectionTable(ttk.Frame):
    """Fast folder table for Step 3 batch R script selection."""

    COLUMNS = ("folder", "status", "inputs")

    def __init__(self, parent):
        super().__init__(parent)
        self.rows = []
        self._row_by_iid = {}
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
        self.tree.configure(yscrollcommand=self.yscroll.set, xscrollcommand=self.xscroll.set)

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
        self.tree.column("folder", width=520, minwidth=220, stretch=False, anchor="w")
        self.tree.column("status", width=360, minwidth=120, stretch=False, anchor="w")
        self.tree.column("inputs", width=72, minwidth=60, stretch=False, anchor="center")

        self.tree.tag_configure("locked", foreground="#6b7280")
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.yscroll.grid(row=0, column=1, sticky="ns")
        self.xscroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.bind("<Button-1>", self._on_click, add="+")
        self.tree.bind("<Configure>", self._on_tree_configure, add="+")

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
        self.tree.delete(*self.tree.get_children(""))

        if not self.rows:
            self.tree.insert(
                "",
                "end",
                text="",
                values=("No folders with complete Step 3 inputs were found.", "", ""),
                tags=("locked",),
            )
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
        widths = {
            "folder": self._measure_text("Folder", heading=True),
            "status": self._measure_text("Status", heading=True),
            "inputs": self._measure_text("Inputs", heading=True),
        }
        for row in self.rows:
            folder, status, inputs = self._values_for_row(row)
            widths["folder"] = max(widths["folder"], self._measure_text(folder))
            widths["status"] = max(widths["status"], self._measure_text(status))
            widths["inputs"] = max(widths["inputs"], self._measure_text(inputs))

        self.tree.column("folder", width=max(220, widths["folder"]))
        self.tree.column("status", width=max(120, widths["status"]))
        self.tree.column("inputs", width=max(60, widths["inputs"]))
        self._expand_folder_to_view()

    def _on_tree_configure(self, _event=None):
        self._expand_folder_to_view()

    def _expand_folder_to_view(self):
        if not self.rows:
            return
        try:
            view_width = max(1, int(self.tree.winfo_width()))
            checkbox_width = int(self.tree.column("#0", "width"))
            folder_width = int(self.tree.column("folder", "width"))
            status_width = int(self.tree.column("status", "width"))
            inputs_width = int(self.tree.column("inputs", "width"))
        except tk.TclError:
            return

        non_folder_width = checkbox_width + status_width + inputs_width
        desired_folder_width = max(220, view_width - non_folder_width - 2)
        if desired_folder_width > folder_width:
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

    def __init__(self, step_frame, parent, root_dir):
        super().__init__(parent)
        self.step_frame = step_frame
        self.root_dir = Path(root_dir)
        self.rows = []
        self.table = None

        self._build_ui()
        self._start_scan()

    def _build_ui(self):
        wrapper = ttk.Frame(self, padding=12)
        wrapper.pack(fill="both", expand=True)

        ttk.Label(wrapper, text="Batch R Script Processing", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            wrapper,
            text=(
                "AIDaS will search the selected folder and subfolders for complete Step 3 inputs. "
                "Folders containing existing RData are shown as skipped and will not be processed."
            ),
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

        self.table_host = ttk.Frame(wrapper)
        self.table_host.pack(fill="both", expand=True)
        self.scan_label = ttk.Label(
            self.table_host,
            text="Scanning folders...",
            anchor="center",
            justify="center",
        )
        self.scan_label.pack(fill="both", expand=True)

        run_box = ttk.Frame(wrapper)
        run_box.pack(fill="x", pady=(10, 0))
        ttk.Button(run_box, text="Cancel", command=self._cancel).pack(side="left")
        ttk.Label(run_box, text="Batch Size:").pack(side="left", padx=(12, 0))
        max_workers = self._max_worker_count()
        self.workers_var = tk.IntVar(value=min(4, max_workers))
        self.workers_spin = ttk.Spinbox(
            run_box,
            from_=1,
            to=max_workers,
            textvariable=self.workers_var,
            width=5,
        )
        self.workers_spin.pack(side="left", padx=(6, 12))
        self.worker_limit_var = tk.StringVar(value=self._worker_limit_text(max_workers))
        ttk.Label(run_box, textvariable=self.worker_limit_var, foreground="#555555").pack(side="left")
        self.next_button = ttk.Button(run_box, text="Next >", command=self._run_selected)
        self.next_button.pack(side="right")
        self.next_button.state(["disabled"])
        self.workers_spin.configure(state="disabled")

    def _max_worker_count(self, ready_count=None):
        cpu_limit = self.step_frame._cpu_worker_limit()
        if ready_count is None:
            return cpu_limit
        return max(1, min(int(ready_count) or 1, cpu_limit))

    def _worker_limit_text(self, max_workers):
        cpu_limit = self.step_frame._cpu_worker_limit()
        return f"Limit: {max_workers} (CPU cores: {cpu_limit})"

    def _start_scan(self):
        self.step_frame.status_var.set(f"Scanning subfolders under {self.root_dir}...")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        rows = []
        scanned = 0
        missing = 0
        try:
            folders = [self.root_dir]
            folders.extend(path for path in self.root_dir.rglob("*") if path.is_dir())
            for folder in folders:
                scanned += 1
                input_paths = self.step_frame._find_input_paths(folder)
                if any(input_paths.get(label) is None for label, *_rest in self.step_frame.REQUIRED_INPUTS):
                    missing += 1
                    continue
                has_rdata = self.step_frame._folder_has_r_data(folder)
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
                            "inputs": "4",
                        },
                    }
                )
        except Exception as exc:
            self.after(0, lambda: self._scan_failed(exc))
            return
        self.after(0, lambda: self._scan_done(rows, scanned, missing))

    def _scan_failed(self, exc):
        if not self.winfo_exists():
            return
        self.summary_var.set(f"Scan failed: {exc}")
        self.step_frame.status_var.set("Batch scan failed.")
        try:
            self.next_button.state(["disabled"])
            self.workers_spin.configure(state="disabled")
        except tk.TclError:
            pass
        messagebox.showerror("Batch Step 3", f"Could not scan folders.\n{exc}", parent=self)

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

    def _scan_done(self, rows, scanned, missing):
        if not self.winfo_exists():
            return
        self.rows = rows
        self._show_results_table(rows)
        ready = sum(1 for row in rows if not row["locked"])
        skipped = sum(1 for row in rows if row["locked"])
        self.summary_var.set(
            f"Scanned {scanned} folders. Found {ready} ready folder(s), {skipped} skipped folder(s) with RData. "
            f"{missing} folder(s) did not contain all four required inputs."
        )
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
        try:
            workers = max(1, int(self.workers_var.get()))
        except (TypeError, ValueError):
            workers = 1
        max_workers = self._max_worker_count(len(folders))
        workers = min(workers, max_workers)
        self.workers_var.set(workers)
        self.step_frame._start_batch_r_runs(folders, workers)

    def _cancel(self):
        self.step_frame._close_r_batch_panel(render_previous=True)


class RBatchRunPanel(ttk.Frame):
    """Embedded progress panel for concurrent folder-level R script runs."""

    TABLE_COLUMNS = (
        ("folder", "Folder", 560, "w"),
        ("status", "Status", 380, "w"),
        ("progress", "Progress", 92, "center"),
    )
    COLUMN_MIN_WIDTHS = {
        "folder": 320,
        "status": 160,
        "progress": 76,
    }
    COLUMN_MAX_WIDTHS = {
        "progress": 92,
    }

    def __init__(self, step_frame, parent, folders, workers):
        super().__init__(parent)
        self.step_frame = step_frame
        self.folders = [Path(folder) for folder in folders]
        self.workers = workers
        self.row_by_folder = {}
        self._build_ui()

    def _build_ui(self):
        wrapper = ttk.Frame(self, padding=12)
        wrapper.pack(fill="both", expand=True)
        ttk.Label(wrapper, text="Running Batch Step 3", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            wrapper,
            text="AIDaS is running the selected Step 3 R script folders. Progress and logs update as each folder finishes.",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        rows = []
        for folder in self.folders:
            row = {
                "folder": folder,
                "values": {
                    "folder": str(folder),
                    "status": "Queued",
                    "progress": "0%",
                },
            }
            rows.append(row)
            self.row_by_folder[str(folder)] = row

        self.table = BatchTable(
            wrapper,
            columns=self.TABLE_COLUMNS,
            min_widths=self.COLUMN_MIN_WIDTHS,
            max_widths=self.COLUMN_MAX_WIDTHS,
            stretch_column="folder",
            empty_message="No folders are queued.",
        )
        self.table.pack(fill="both", expand=True)
        self.table.set_rows(rows)

        self.summary_var = tk.StringVar(
            value=f"Running {len(self.folders)} folder(s) with up to {self.workers} parallel R process(es)."
        )
        ttk.Label(wrapper, textvariable=self.summary_var, wraplength=760, justify="left").pack(
            anchor="w", pady=(4, 10)
        )

        log_frame = ttk.LabelFrame(wrapper, text="Batch log")
        log_frame.pack(fill="both", expand=False, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def update_folder(self, folder, status=None, progress=None):
        row = self.row_by_folder.get(str(folder))
        if row is None:
            return
        values = dict(row.get("values") or {})
        if status is not None:
            values["status"] = status
        if progress is not None:
            values["progress"] = f"{int(max(0, min(100, float(progress))))}%"
        self.table.update_row(row, values=values)

    def set_summary(self, text):
        self.summary_var.set(text)

    def log(self, text):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {text}"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
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
        ("Dark_MARKED", ("Dark_MARKED", "DARK_MARKED"), "Dark_MARKED.hdr/.img", 8),
        ("Light_MARKED", ("Light_MARKED", "LIGHT_MARKED"), "Light_MARKED.hdr/.img", 8),
        ("DARK", ("DARK", "Dark"), "DARK.hdr/.img", 16),
        ("LIGHT", ("LIGHT", "Light"), "LIGHT.hdr/.img", 16),
    )
    R_SCRIPT_NAME = "RAW_OCT_PROCESSING_2023_09SEP-05_WSU.R"
    R_DOWNLOAD_PAGE = "https://cloud.r-project.org/bin/windows/base/"
    R_REQUIRED_PACKAGES = ("AnalyzeFMRI", "RNiftyReg")
    R_WORKSPACE_FILES = (
        "DARK__and__LIGHT__flat.RData",
        "_done_DARK__and__LIGHT.RData",
    )
    R_ARRAY_EXPORT_DIR = "step3_r_arrays"
    R_PROGRESS_BY_STEP = {
        "startup": (1, "Starting R script"),
        "input-config": (2, "Reading R input configuration"),
        "load-images": (5, "Loading Analyze volumes in R"),
        "fovea-center": (8, "Finding fovea center"),
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
        "layer-borders": (78, "Identifying retinal layer borders"),
        "main-normalization": (86, "Spatially normalizing main retina"),
        "fovea-normalization": (92, "Spatially normalizing fovea"),
        "final-export": (97, "Writing final R outputs"),
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
        self.r_setup_button = None
        self.r_batch_button = None
        self._busy = False
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
            sidebar_pack={"padx": (2, 6), "pady": 6},
            content_pack={"padx": 6, "pady": 6},
            status_var=self.status_var,
        )
        process_section = self.add_sidebar_section("Process", pady=(0, 5))
        process = process_section.body

        self.r_setup_button = ttk.Button(process, text="Setup R and Packages...", command=self._open_r_setup_wizard)
        self.r_setup_button.pack(fill="x", pady=2)

        self.r_batch_button = ttk.Button(process, text="Batch Run R Script...", command=self._open_r_batch_scanner)
        self.r_batch_button.pack(fill="x", pady=2)

        ttk.Button(process, text="Load R Results...", command=self._browse_r_results_folder).pack(fill="x", pady=2)

        ttk.Separator(process, orient="horizontal").pack(fill="x", pady=(6, 4))

        ttk.Label(process, text="View").pack(anchor="w", pady=(6, 2))
        view_combo = ttk.Combobox(
            process,
            textvariable=self.view_var,
            values=["DARK_MARKED_find_vertex", "_tissueBorders__DARK"],
            state="readonly",
        )
        view_combo.pack(fill="x", pady=2)
        view_combo.bind("<<ComboboxSelected>>", lambda _: self._render())

        ttk.Separator(process, orient="horizontal").pack(fill="x", pady=(8, 4))
        self.progress = ttk.Progressbar(process, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=2)
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

    @staticmethod
    def _script_path():
        return Path(__file__).resolve().parents[2] / Step3Frame.R_SCRIPT_NAME

    def _resolve_rscript_executable(self):
        configured = None if self.preferences is None else self.preferences.get("rscript_path")
        candidates = []
        if configured:
            candidates.append(Path(configured))

        env_override = os.environ.get("RSCRIPT_PATH") or os.environ.get("R_SCRIPT_PATH")
        if env_override:
            candidates.append(Path(env_override))

        for name in ("Rscript", "Rscript.exe"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))

        if os.name == "nt":
            for root in (Path(r"C:\Program Files\R"), Path(r"C:\Program Files (x86)\R")):
                if root.is_dir():
                    candidates.extend(root.glob("R*/bin/x64/Rscript.exe"))
                    candidates.extend(root.glob("R*/bin/Rscript.exe"))

        for candidate in candidates:
            candidate = self._normalize_r_executable(candidate)
            if candidate and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _normalize_r_executable(path):
        """Return a non-interactive R executable, preferring Rscript.exe.

        Users often select R.exe or Rgui.exe from the file dialog. Rgui/RStudio
        opens an interactive program and does not run this script as intended.
        If possible, convert those selections to the adjacent Rscript.exe.
        """
        if not path:
            return None
        path = Path(path)
        if not path.is_file():
            return None

        name = path.name.lower()
        if name in {"rscript.exe", "rscript"}:
            return path

        sibling_name = "Rscript.exe" if name.endswith(".exe") else "Rscript"
        sibling = path.with_name(sibling_name)
        if sibling.is_file():
            return sibling

        if name in {"r.exe", "rterm.exe", "r", "rterm"}:
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

    @staticmethod
    def _cpu_worker_limit():
        return max(1, os.cpu_count() or 1)

    def _default_r_package_library(self):
        if self.r_package_library_path:
            return Path(self.r_package_library_path)
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "AIDaS" / "R-packages"
        return Path.home() / "AIDaS_R_packages"

    def _r_env(self):
        env = os.environ.copy()
        library_path = self._default_r_package_library()
        if library_path:
            env["R_LIBS_USER"] = str(library_path.resolve())
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
            self.status_var.set("Rscript was not found. Open the R setup wizard to continue.")
            return None
        if self.preferences is not None:
            self.preferences.set("rscript_path", str(rscript))
        if self._r_packages_ready(rscript):
            self.status_var.set("R and required Step 3 packages are ready.")
            return rscript
        self.status_var.set("Step 3 R packages are missing. Open the R setup wizard to install them.")
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
        self.progress.configure(value=100)
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
        png_path = Path(self.output_sdb_dir or self.current_sdb_dir) / filename
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
        expected = shapes["Dark_MARKED"]
        mismatched = {label: shape for label, shape in shapes.items() if shape != expected}
        if mismatched:
            lines = [f"Dark_MARKED: {expected}"]
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
        self.progress.configure(value=0)
        self.progress_text_var.set("Idle")
        self._render()

    def _refresh_input_status(self):
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
            return
        self.current_sdb_dir = folder
        self.output_sdb_dir = folder
        self.results = None
        self.original_light_volume = None
        self._refresh_input_status()

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
        self.r_setup_panel = None
        self.r_batch_panel = None
        self.r_batch_run_panel = None

    def _open_r_setup_wizard(self, on_finish=None):
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

    def _open_r_batch_scanner(self):
        if self._busy:
            return
        root_dir = filedialog.askdirectory(
            title="Select root folder for batch Step 3 R processing",
            initialdir=self.current_sdb_dir or None,
        )
        if not root_dir:
            return
        self._clear_plot_holder()
        self.r_batch_panel = RBatchSelectionPanel(self, self.plot_holder, Path(root_dir))
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
            "reference_dark": self._analyze_base_name(input_paths["Dark_MARKED"]),
            "reference_light": self._analyze_base_name(input_paths["Light_MARKED"]),
            "to_process_dark": self._analyze_base_name(input_paths["DARK"]),
            "to_process_light": self._analyze_base_name(input_paths["LIGHT"]),
            "image_index_light": self._r_index_string(input_info["LIGHT"]["shape"][0]),
            "image_index_dark": self._r_index_string(input_info["DARK"]["shape"][0]),
            "pixel_width": str(self.PIXEL_WIDTH_UM),
        }

    def _start_batch_r_runs(self, folders, workers):
        folders = [Path(folder) for folder in folders]
        if not folders:
            messagebox.showwarning("Batch Step 3", "Select at least one folder to process.")
            return
        if self._busy:
            return
        script_path = self._script_path()
        if not script_path.is_file():
            messagebox.showerror("Batch Step 3", f"Could not find the R script:\n{script_path}")
            return
        rscript = self._ensure_r_ready_with_wizard()
        if rscript is None:
            self._open_r_setup_wizard(
                on_finish=lambda result: self._start_batch_r_runs(folders, workers) if result else None
            )
            return

        workers = max(1, min(int(workers), len(folders), self._cpu_worker_limit()))
        self._clear_plot_holder()
        self.r_batch_run_panel = RBatchRunPanel(self, self.plot_holder, folders, workers)
        self.r_batch_run_panel.pack(fill="both", expand=True)
        self._busy = True
        self._set_process_buttons("disabled")
        self.progress.configure(value=0)
        self.progress_text_var.set("Batch running")
        self.status_var.set(f"Running Step 3 R script for {len(folders)} folder(s).")
        threading.Thread(
            target=self._batch_r_worker,
            args=(Path(rscript), script_path, folders, workers),
            daemon=True,
        ).start()

    def _batch_panel_update(self, folder, status=None, progress=None, log=None):
        panel = self.r_batch_run_panel
        if panel is None:
            return
        if status is not None or progress is not None:
            panel.update_folder(folder, status=status, progress=progress)
        if log:
            panel.log(log)

    def _run_r_script_for_config(self, rscript_path, script_path, r_config, batch_folder=None):
        folder = Path(batch_folder or r_config["input_dir"])
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
        cmd = self._build_r_run_command(rscript_path, script_path, script_args)
        env = self._r_env()
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

        self.after(
            0,
            lambda f=folder: self._batch_panel_update(
                f,
                status="Running R script",
                progress=1,
                log=f"Starting R script: {f}",
            ),
        )
        output_lines = []
        try:
            process = subprocess.Popen(
                cmd,
                cwd=r_config["input_dir"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if process.stdout is not None:
                for line in process.stdout:
                    output_lines.append(line)
                    progress = self._progress_from_r_line(line)
                    if progress is not None:
                        percent, label = progress
                        self.after(
                            0,
                            lambda f=folder, p=percent, s=label: self._batch_panel_update(
                                f,
                                status=s,
                                progress=p,
                            ),
                        )
            returncode = process.wait()
        except Exception as exc:
            stdout = "".join(output_lines)
            log_path = self._write_r_run_log(r_config["output_dir"], 1, stdout, str(exc), cmd)
            return {"folder": folder, "returncode": 1, "stdout": stdout, "stderr": str(exc), "cmd": cmd, "log": log_path}

        stdout = "".join(output_lines)
        log_path = self._write_r_run_log(r_config["output_dir"], returncode, stdout, "", cmd)
        if returncode == 0:
            self.after(0, lambda f=folder, lp=log_path: self._batch_panel_update(f, log=f"Finished: {f}\nLog: {lp}"))
        else:
            short_output = self._short_process_text(stdout)
            self.after(
                0,
                lambda f=folder, lp=log_path, out=short_output: self._batch_panel_update(
                    f,
                    log=f"Failed: {f}\nLog: {lp}\n{out}",
                ),
            )
        return {"folder": folder, "returncode": returncode, "stdout": stdout, "stderr": "", "cmd": cmd, "log": log_path}

    def _batch_r_worker(self, rscript_path, script_path, folders, workers):
        results = []
        completed = 0
        total = len(folders)

        def run_folder(folder):
            folder = Path(folder)
            self.after(0, lambda f=folder: self._batch_panel_update(f, status="Validating", progress=0))
            try:
                if self._folder_has_r_data(folder):
                    raise RuntimeError("Skipped because this folder contains RData.")
                r_config = self._r_script_config_for_folder(folder)
            except Exception as exc:
                return {"folder": folder, "returncode": 1, "stdout": "", "stderr": str(exc), "cmd": []}
            return self._run_r_script_for_config(rscript_path, script_path, r_config, batch_folder=folder)

        workers = max(1, min(int(workers), len(folders), self._cpu_worker_limit()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(run_folder, folder): folder for folder in folders}
            for future in concurrent.futures.as_completed(future_map):
                folder = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"folder": folder, "returncode": 1, "stdout": "", "stderr": str(exc), "cmd": []}
                results.append(result)
                completed += 1
                overall = (completed / max(1, total)) * 100.0
                status = "Completed" if result["returncode"] == 0 else "Failed"
                self.after(
                    0,
                    lambda f=folder, s=status, o=overall: (
                        self._batch_panel_update(f, status=s, progress=100),
                        self.progress.configure(value=o),
                    ),
                )

        self.after(0, lambda: self._on_batch_r_done(results))

    def _on_batch_r_done(self, results):
        self._busy = False
        self._set_process_buttons("normal")
        success = sum(1 for result in results if result["returncode"] == 0)
        failed = len(results) - success
        self.progress.configure(value=100)
        self.progress_text_var.set("Batch completed")
        self.status_var.set(f"Batch Step 3 complete: {success} succeeded, {failed} failed.")
        if self.r_batch_run_panel is not None:
            self.r_batch_run_panel.set_summary(f"Batch complete: {success} succeeded, {failed} failed.")
            self.r_batch_run_panel.log(f"Batch complete: {success} succeeded, {failed} failed.")
        self.info_var.set(
            "Batch Step 3 R results:\n"
            + "\n".join(
                f"{'OK' if result['returncode'] == 0 else 'FAILED'}: {result['folder']}"
                for result in results
            )
        )

        successful_folders = [Path(result["folder"]) for result in results if result["returncode"] == 0]
        if len(successful_folders) == 1:
            self._load_r_results_from_folder(successful_folders[0], show_errors=False)

    def _write_r_run_log(self, output_dir, returncode, stdout, stderr, cmd):
        log_path = app_log_dir() / f"step3_rscript_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
        log_path.write_text(
            "Command:\n"
            + " ".join(str(part) for part in cmd)
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

    def _display_preview_image(self, image, background="#ffffff"):
        label = tk.Label(self.plot_holder, bg=background, borderwidth=0, highlightthickness=0)
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
            self._preview_photo = ImageTk.PhotoImage(canvas)
            try:
                label.configure(image=self._preview_photo)
            except tk.TclError:
                return

        label.bind("<Configure>", redraw, add="+")
        self.canvas = label
        self.after(0, redraw)

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
        self._display_preview_image(image)

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
