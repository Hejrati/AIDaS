"""Tkinter coordination for AIDaS update checks and installer downloads."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

from aidas.services.update_service import (
    DownloadCancelled,
    GITHUB_RELEASES_URL,
    ReleaseInfo,
    download_installer,
    find_available_update,
    supports_in_app_install,
)


AUTO_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
RELEASE_NOTES_LIMIT = 1600


class DownloadProgressDialog(tk.Toplevel):
    """Small non-modal progress window with a cooperative cancel button."""

    def __init__(self, parent: tk.Misc, release: ReleaseInfo, cancel_command) -> None:
        super().__init__(parent)
        self.title("Downloading AIDaS Update")
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", cancel_command)

        panel = ttk.Frame(self, padding=18)
        panel.pack(fill="both", expand=True)
        ttk.Label(
            panel,
            text=f"Downloading AIDaS {release.version_text}",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        self.detail_var = tk.StringVar(value="Connecting to GitHub...")
        ttk.Label(panel, textvariable=self.detail_var).pack(anchor="w", pady=(7, 8))
        self.progress = ttk.Progressbar(panel, length=390, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.cancel_button = ttk.Button(panel, text="Cancel", command=cancel_command)
        self.cancel_button.pack(anchor="e", pady=(12, 0))

        self.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_width()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_height()) // 2)
        self.geometry(f"+{x}+{y}")

    def set_progress(self, downloaded: int, total: int) -> None:
        percent = 0.0 if total <= 0 else min(100.0, downloaded * 100.0 / total)
        self.progress.configure(value=percent)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        self.detail_var.set(f"{downloaded_mb:,.1f} MB of {total_mb:,.1f} MB ({percent:.0f}%)")

    def mark_cancelling(self) -> None:
        self.detail_var.set("Cancelling download...")
        self.cancel_button.configure(state="disabled")


class UpdateController:
    """Keep network work off the Tk thread and present update results safely."""

    def __init__(
        self,
        root: tk.Misc,
        *,
        preferences,
        current_version: str,
        status_callback,
        restart_blocker_callback,
        install_callback,
    ) -> None:
        self.root = root
        self.preferences = preferences
        self.current_version = current_version
        self.status_callback = status_callback
        self.restart_blocker_callback = restart_blocker_callback
        self.install_callback = install_callback
        self._checking = False
        self._downloading = False
        self._cancel_event: threading.Event | None = None
        self._progress_dialog: DownloadProgressDialog | None = None

    def check_automatically(self) -> None:
        """Check at most daily, and only from an installed Windows build."""
        if not supports_in_app_install() or not self.preferences.get("check_for_updates", True):
            return
        try:
            last_check = float(self.preferences.get("last_successful_update_check", 0) or 0)
        except (TypeError, ValueError):
            last_check = 0
        if time.time() - last_check < AUTO_CHECK_INTERVAL_SECONDS:
            return
        self._begin_check(manual=False)

    def check_now(self) -> None:
        self._begin_check(manual=True)

    def _begin_check(self, *, manual: bool) -> None:
        if self._checking or self._downloading:
            if manual:
                messagebox.showinfo(
                    "AIDaS Updates",
                    "An update check or download is already in progress.",
                    parent=self.root,
                )
            return

        self._checking = True
        self.status_callback("Checking GitHub for AIDaS updates...")

        def worker() -> None:
            try:
                release = find_available_update(self.current_version)
                error = None
            except Exception as exc:  # converted into user-safe UI below
                release = None
                error = exc
            self._schedule(lambda: self._finish_check(release, error, manual=manual))

        threading.Thread(target=worker, name="aidas-update-check", daemon=True).start()

    def _finish_check(self, release, error, *, manual: bool) -> None:
        self._checking = False
        if error is not None:
            self.status_callback("Update check could not be completed")
            if manual:
                messagebox.showerror(
                    "AIDaS Updates",
                    f"AIDaS could not check for updates.\n\n{error}",
                    parent=self.root,
                )
            return

        self.preferences.set("last_successful_update_check", int(time.time()))
        if release is None:
            self.status_callback(f"AIDaS {self.current_version} is up to date")
            if manual:
                messagebox.showinfo(
                    "AIDaS Updates",
                    f"You're using the latest version of AIDaS ({self.current_version}).",
                    parent=self.root,
                )
            return

        self.status_callback(f"AIDaS {release.version_text} is available")
        self._offer_release(release)

    def _offer_release(self, release: ReleaseInfo) -> None:
        notes = release.notes or "See the GitHub release page for details."
        if len(notes) > RELEASE_NOTES_LIMIT:
            notes = notes[:RELEASE_NOTES_LIMIT].rstrip() + "\n..."

        if not supports_in_app_install():
            open_page = messagebox.askyesno(
                "AIDaS Update Available",
                f"AIDaS {release.version_text} is available.\n\n"
                "Automatic installation is enabled only in the packaged Windows app. "
                "Open the release page?",
                parent=self.root,
            )
            if open_page:
                webbrowser.open_new_tab(release.page_url or GITHUB_RELEASES_URL)
            return

        install = messagebox.askyesno(
            "AIDaS Update Available",
            f"AIDaS {release.version_text} is available.\n"
            f"Installed version: {self.current_version}\n\n"
            f"Release notes:\n{notes}\n\n"
            "Download the verified update now? Your preferences, R packages, logs, "
            "and image/output folders will not be removed.",
            parent=self.root,
        )
        if install:
            self._begin_download(release)

    def _begin_download(self, release: ReleaseInfo) -> None:
        self._downloading = True
        self._cancel_event = threading.Event()
        self._progress_dialog = DownloadProgressDialog(self.root, release, self.cancel_download)
        self.status_callback(f"Downloading AIDaS {release.version_text}...")

        def report(downloaded: int, total: int) -> None:
            self._schedule(lambda: self._update_progress(downloaded, total))

        def worker() -> None:
            try:
                installer = download_installer(
                    release,
                    progress=report,
                    cancel_event=self._cancel_event,
                )
                error = None
            except Exception as exc:
                installer = None
                error = exc
            self._schedule(lambda: self._finish_download(release, installer, error))

        threading.Thread(target=worker, name="aidas-update-download", daemon=True).start()

    def cancel_download(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._progress_dialog is not None:
            self._progress_dialog.mark_cancelling()

    def _update_progress(self, downloaded: int, total: int) -> None:
        dialog = self._progress_dialog
        try:
            if dialog is not None and dialog.winfo_exists():
                dialog.set_progress(downloaded, total)
        except tk.TclError:
            pass

    def _finish_download(self, release: ReleaseInfo, installer, error) -> None:
        self._downloading = False
        self._cancel_event = None
        dialog = self._progress_dialog
        self._progress_dialog = None
        try:
            if dialog is not None and dialog.winfo_exists():
                dialog.destroy()
        except tk.TclError:
            pass

        if error is not None:
            if isinstance(error, DownloadCancelled):
                self.status_callback("AIDaS update download cancelled")
                return
            self.status_callback("AIDaS update download failed")
            messagebox.showerror(
                "AIDaS Update",
                f"The update was not installed.\n\n{error}",
                parent=self.root,
            )
            return

        self.status_callback(f"AIDaS {release.version_text} is ready to install")
        restart = messagebox.askyesno(
            "AIDaS Update Ready",
            f"AIDaS {release.version_text} was downloaded and verified.\n\n"
            "Save any work before continuing. AIDaS will close, update in place, "
            "and reopen automatically.\n\nRestart and install now?",
            parent=self.root,
        )
        if not restart:
            self.status_callback("Update downloaded; check again when ready to install")
            return

        blocker = self.restart_blocker_callback()
        if blocker:
            messagebox.showwarning(
                "Finish Current Work First",
                f"The update is downloaded, but AIDaS will not close while this operation is active:\n\n"
                f"{blocker}\n\nFinish it, then choose Help > Check for Updates to install the cached update.",
                parent=self.root,
            )
            self.status_callback("Update downloaded; waiting for active work to finish")
            return
        self.install_callback(installer)

    def _schedule(self, callback) -> None:
        try:
            self.root.after(0, callback)
        except (RuntimeError, tk.TclError):
            pass
