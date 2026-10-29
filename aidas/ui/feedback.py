"""User-reviewed diagnostic ZIPs and email drafts, without server credentials."""
from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import webbrowser

import customtkinter as ctk

from aidas.services.feedback import (
    FEEDBACK_EMAIL, collect_diagnostics, feedback_mailto, report_preview,
    save_report, validate_attachments,
)
from aidas.ui.components import AppButton
from aidas.core.display import work_area_bounds
from aidas.ui.theme import COLOR_PAIRS
from aidas.ui.windowing import centered_logical_geometry, logical_window_size, physical_window_size, synchronize_window_chrome
from aidas.utils.ui_utils import apply_app_icon_to


def _fit_window(window, parent, width, height):
    bounds = work_area_bounds(window, parent=parent)
    physical_width, physical_height = physical_window_size(window, width, height)
    width, height = logical_window_size(window, min(physical_width, round((bounds[2]-bounds[0])*.92)),
                                       min(physical_height, round((bounds[3]-bounds[1])*.9)))
    window.geometry(centered_logical_geometry(window, width, height, bounds=bounds, parent=parent))
    window.minsize(min(560, width), min(450, height))


class _FeedbackWindow(ctk.CTkToplevel):
    def destroy(self):
        # CTkTextbox can queue scrollbar-check lambdas on the textbox itself.
        # Closing a report quickly must cancel them before Tcl deletes them.
        owners = {}
        def visit(widget):
            if not isinstance(widget, tk.Toplevel):
                owners.update((command, widget) for command in (widget._tclCommands or ()))
            for child in widget.winfo_children():
                visit(child)
        try:
            visit(self)
            for timer in self.tk.call("after", "info"):
                script = self.tk.call("after", "info", timer)[0]
                owner = owners.get(self.tk.splitlist(script)[0])
                if owner is not None:
                    owner.after_cancel(timer)
        except tk.TclError:
            pass
        super().destroy()


class FeedbackDialog(_FeedbackWindow):
    def __init__(self, parent, *, context):
        super().__init__(parent)
        self.withdraw()
        self.title("Send AIDaS feedback")
        self.transient(parent)
        self.configure(fg_color=COLOR_PAIRS["application"])
        _fit_window(self, parent, 680, 700)
        apply_app_icon_to(self)
        self._results = queue.Queue()
        self._cancel = threading.Event()
        self._after_id = None
        self._snapshot = None
        self._attachments = []
        self._report_path = None
        self._saved_title = ""
        self._working = False
        self._wrapped_labels = []
        self._wrap_width = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(self, text="Send feedback", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=COLOR_PAIRS["text"]).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 8))
        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=12)
        body.grid_columnconfigure(0, weight=1)
        row = 0

        def label(text):
            nonlocal row
            widget = ctk.CTkLabel(body, text=text, anchor="w", justify="left", wraplength=570,
                                text_color=COLOR_PAIRS["text"])
            widget.grid(row=row, column=0, sticky="ew", padx=8, pady=(8, 4))
            self._wrapped_labels.append(widget)
            row += 1
            return widget

        label(f"To: {FEEDBACK_EMAIL}\nThe app prepares a ZIP and opens your email app. You attach the ZIP and send it.")
        label("Problem summary *")
        self.summary = ctk.CTkEntry(body, placeholder_text="For example: Step 3 fails while flattening an image")
        self.summary.grid(row=row, column=0, sticky="ew", padx=8); row += 1
        label("What happened? Include steps to reproduce, expected results, and actual results. *")
        self.description = ctk.CTkTextbox(body, height=110)
        self.description.grid(row=row, column=0, sticky="ew", padx=8); row += 1
        self.include_logs = tk.BooleanVar(self, value=True)
        self.logs_check = ctk.CTkCheckBox(body, text="Include relevant recent logs (review before sharing)", variable=self.include_logs,
                                         command=self._invalidate_report)
        self.logs_check.grid(row=row, column=0, sticky="w", padx=8, pady=(12, 6)); row += 1
        self.preview_button = AppButton(body, text="Review diagnostics and logs", command=self._preview, state="disabled")
        self.preview_button.grid(row=row, column=0, sticky="w", padx=8, pady=4); row += 1
        label("Includes app and system information, processing settings, and recent errors.\n"
              "Automatic removal of private details is not guaranteed. Review before sharing.")
        label("Optional: screenshots, sample inputs, annotations, or R workspaces (10 MB total).\n"
              "Selected files are included unchanged. Remove patient details and other private information first.")
        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.grid(row=row, column=0, sticky="ew", padx=8); row += 1
        self.add_button = AppButton(actions, text="Add files...", command=self._add_files)
        self.add_button.pack(side="left", padx=(0, 6))
        self.clear_button = AppButton(actions, text="Clear files", command=self._clear_files)
        self.clear_button.pack(side="left")
        self.files_label = label("No data files selected.")
        self.approved = tk.BooleanVar(self, value=False)
        footer = ctk.CTkFrame(self, fg_color=COLOR_PAIRS["surface"])
        footer.grid(row=2, column=0, sticky="ew", padx=12, pady=12)
        footer.grid_columnconfigure(0, weight=1)
        self.approval_check = ctk.CTkCheckBox(footer, text="I reviewed this report and have permission to share\nall selected files with AIDaS support.",
                                            variable=self.approved)
        self.approval_check.grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(10, 4))
        self.status = tk.StringVar(self, value="Collecting diagnostics locally...")
        self.status_label = ctk.CTkLabel(footer, textvariable=self.status, anchor="w", justify="left", wraplength=440)
        self.status_label.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
        self.save_button = AppButton(footer, text="Save report ZIP...", variant="primary", command=self._save, state="disabled")
        self.save_button.grid(row=2, column=0, sticky="w", padx=8, pady=8)
        self.email_button = AppButton(footer, text="Open email draft", command=self._email, state="disabled")
        self.email_button.grid(row=2, column=1, padx=4, pady=8)
        AppButton(footer, text="Close", width=70, command=self.destroy).grid(row=2, column=2, padx=8, pady=8)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda _event: self.destroy())
        self.summary.bind("<KeyRelease>", self._invalidate_report, add="+")
        self.description.bind("<KeyRelease>", self._invalidate_report, add="+")
        def resize_labels(event):
            width = max(240, int(body._reverse_widget_scaling(event.width))-24)
            if width != self._wrap_width:
                self._wrap_width = width
                for widget in self._wrapped_labels:
                    widget.configure(wraplength=width)
        body.bind("<Configure>", resize_labels, add="+")
        self._start_worker("diagnostics", lambda: collect_diagnostics(context))
        synchronize_window_chrome(self, background=COLOR_PAIRS["window_chrome"], foreground=COLOR_PAIRS["text"])
        self.deiconify()
        self.summary.focus_set()

    def _start_worker(self, kind, work):
        self._working = True
        results = self._results
        def run():
            try:
                results.put((kind, work(), None))
            except Exception as exc:
                results.put((kind, None, str(exc) or type(exc).__name__))
        threading.Thread(target=run, daemon=True).start()
        self._after_id = self.after(100, self._poll)

    def _poll(self):
        self._after_id = None
        if self._cancel.is_set():
            return
        try:
            kind, result, error = self._results.get_nowait()
        except queue.Empty:
            self._after_id = self.after(100, self._poll)
            return
        self._working = False
        if kind == "diagnostics":
            self._snapshot = result
            self.status.set("Review the diagnostics, describe the problem, then save the ZIP." if not error else f"Could not collect diagnostics: {error}")
        elif not error:
            self._report_path = result
            self.status.set(f"Saved {result.name}. Attach this ZIP to the email draft; no email has been sent.")
        else:
            self.status.set(f"Could not save report: {error}")
        self._set_editing(True)
        if self._snapshot is not None:
            self.save_button.configure(state="normal")
            self.preview_button.configure(state="normal")
        if self._report_path is not None:
            self.email_button.configure(state="normal")

    def _set_editing(self, enabled):
        state = "normal" if enabled else "disabled"
        for widget in (self.summary, self.description, self.logs_check, self.approval_check,
                       self.add_button, self.clear_button, self.save_button, self.preview_button, self.email_button):
            widget.configure(state=state)
        if enabled:
            self.email_button.configure(state="normal" if self._report_path is not None else "disabled")

    def _invalidate_report(self, _event=None):
        self.approved.set(False)
        self._report_path = None
        self.email_button.configure(state="disabled")

    def _preview(self):
        if self._snapshot is None:
            return
        window = _FeedbackWindow(self)
        window.withdraw()
        window.title("Review AIDaS diagnostics")
        window.transient(self)
        _fit_window(window, self, 740, 560)
        apply_app_icon_to(window)
        synchronize_window_chrome(window, background=COLOR_PAIRS["window_chrome"], foreground=COLOR_PAIRS["text"])
        box = ctk.CTkTextbox(window, wrap="word")
        box.pack(fill="both", expand=True, padx=12, pady=12)
        box.insert("1.0", report_preview(self._snapshot, self.include_logs.get()))
        box.configure(state="disabled")
        AppButton(window, text="Close", command=window.destroy).pack(pady=(0, 12))
        window.deiconify()

    def _add_files(self):
        selected = filedialog.askopenfilenames(parent=self, title="Select files to share with AIDaS support")
        if not selected:
            return
        try:
            self._attachments = list(validate_attachments([*self._attachments, *selected]))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Feedback attachments", str(exc), parent=self)
            return
        self._invalidate_report()
        self.files_label.configure(text="\n".join(f"{p.name} ({p.stat().st_size/1024:.0f} KB)" for p in self._attachments))

    def _clear_files(self):
        self._attachments = []
        self._invalidate_report()
        self.files_label.configure(text="No data files selected.")

    def _save(self):
        if self._working or self._snapshot is None:
            return
        title = self.summary.get().strip()
        description = self.description.get("1.0", "end-1c").strip()
        if not title or not description:
            messagebox.showwarning("Feedback", "Add a summary and describe how to reproduce the problem.", parent=self)
            return
        if not self.approved.get():
            messagebox.showwarning("Review report", "Review the diagnostics and selected files, then check the approval box.", parent=self)
            return
        report_id = self._snapshot["diagnostics"]["report_id"]
        target = filedialog.asksaveasfilename(parent=self, title="Save diagnostic report", defaultextension=".zip",
                    filetypes=[("ZIP report", "*.zip")], initialfile=f"AIDaS-feedback-{report_id}.zip",
                    initialdir=str(Path.home() / "Downloads" if (Path.home() / "Downloads").is_dir() else Path.home()))
        if not target:
            return
        self._saved_title = title
        self._report_path = None
        self._set_editing(False)
        self.status.set("Creating diagnostic ZIP locally...")
        snapshot, files, logs, cancel = self._snapshot, tuple(self._attachments), self.include_logs.get(), self._cancel
        self._start_worker("zip", lambda: save_report(target, snapshot, title=title, description=description,
                                                     attachments=files, include_logs=logs, cancel_event=cancel))

    def _email(self):
        if self._report_path is None or not self._report_path.is_file():
            messagebox.showwarning("Feedback", "Save a report ZIP before opening an email draft.", parent=self)
            return
        url = feedback_mailto(self._snapshot["diagnostics"]["report_id"], self._saved_title, self._report_path.name)
        try:
            opened = webbrowser.open(url)
            if not opened:
                raise OSError("No default email application accepted the draft.")
        except Exception as exc:
            messagebox.showinfo("Email draft", f"Could not open your email app: {exc}\n\n"
                                f"Email {FEEDBACK_EMAIL} and attach:\n{self._report_path}", parent=self)
            return
        self.status.set(f"Draft opened. Attach {self._report_path.name} from {self._report_path.parent}, then click Send in your email app.")

    def destroy(self):
        self._cancel.set()
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        super().destroy()
