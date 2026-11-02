"""Reviewed text reports, optional shared sample data, and user-sent email drafts."""
from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from types import SimpleNamespace
import webbrowser

import customtkinter as ctk

from aidas.services.feedback import (
    FEEDBACK_EMAIL, collect_diagnostics, feedback_mailto, report_preview,
    save_report, validate_attachments, validate_sharing_link,
)
from aidas.ui.components import AppButton
from aidas.ui.scrolling import SettingsScrollableFrame
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


class _FeedbackScrollableFrame(SettingsScrollableFrame):
    """Capture wheel input before child controls consume it, within one dialog."""

    def install_window_scrolling(self, window):
        self._wheel_tag = f"AIDaSFeedbackWheel{self._w}"
        self._wheel_window = window
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            window.bind_class(self._wheel_tag, sequence, self._route_wheel)

        def visit(widget):
            if widget is not window and isinstance(widget, tk.Toplevel):
                return
            widget.bindtags((self._wheel_tag, *widget.bindtags()))
            for child in widget.winfo_children():
                visit(child)

        visit(window)

    def _route_wheel(self, event):
        # Text's native class handler runs before toplevel/global bindings.
        # This capture tag routes exactly once and leaves other windows alone.
        self._mouse_wheel_all(SimpleNamespace(
            widget=self._parent_canvas, delta=event.delta, num=event.num,
        ))
        return "break"

    def destroy(self):
        window = getattr(self, "_wheel_window", None)
        if window is not None:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                try:
                    window.unbind_class(self._wheel_tag, sequence)
                except tk.TclError:
                    pass
            self._wheel_window = None
        super().destroy()


class _FeedbackPreviewWindow(_FeedbackWindow):
    """Keep long diagnostic logs from monopolizing Tk while they are inserted."""

    TEXT_CHUNK_SIZE = 32 * 1024

    def __init__(self, parent):
        super().__init__(parent)
        self._text_after_id = None
        self._text_generation = 0
        self._pending_text = ""
        self._text_offset = 0
        self.box = None

    def set_text(self, text):
        self._cancel_text_load()
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        self.box.configure(state="disabled")
        self.box.yview_moveto(0)
        self._pending_text = text
        self._text_offset = 0
        self._append_text(self._text_generation)

    def _append_text(self, generation):
        self._text_after_id = None
        if generation != self._text_generation:
            return
        end = self._text_offset + self.TEXT_CHUNK_SIZE
        self.box.configure(state="normal")
        self.box.insert("end", self._pending_text[self._text_offset:end])
        self.box.configure(state="disabled")
        self._text_offset = end
        if end < len(self._pending_text):
            self._text_after_id = self.after(1, self._append_text, generation)
        else:
            self._pending_text = ""

    def _cancel_text_load(self):
        self._text_generation += 1
        if self._text_after_id is not None:
            self.after_cancel(self._text_after_id)
            self._text_after_id = None
        self._pending_text = ""

    def destroy(self):
        self._cancel_text_load()
        super().destroy()
        self.box = None


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
        self._saved_sharing_link = ""
        self._sharing_link_value = ""
        self._working = False
        self._review_enabled = False
        self._preview_window = None
        self._preview_text = {}
        self._preview_includes_logs = None
        self.required_markers = []
        self._wrapped_labels = []
        self._wrap_width = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(self, text="Send feedback", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=COLOR_PAIRS["text"]).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 8))
        body = _FeedbackScrollableFrame(self, fg_color="transparent")
        self.body = body
        body.grid(row=1, column=0, sticky="nsew", padx=12)
        body.grid_columnconfigure(0, weight=1)
        row = 0

        def label(text, *, required=False):
            nonlocal row
            holder = ctk.CTkFrame(body, fg_color="transparent") if required else body
            if required:
                holder.grid(row=row, column=0, sticky="w", padx=8, pady=(8, 4))
            widget = ctk.CTkLabel(holder, text=text, anchor="w", justify="left", wraplength=550 if required else 570,
                                text_color=COLOR_PAIRS["text"])
            if required:
                widget.grid(row=0, column=0, sticky="w")
                marker = ctk.CTkLabel(holder, text="*", width=12, text_color=COLOR_PAIRS["danger"])
                marker.grid(row=0, column=1, sticky="ne", padx=(3, 0))
                self.required_markers.append(marker)
            else:
                widget.grid(row=row, column=0, sticky="ew", padx=8, pady=(8, 4))
            self._wrapped_labels.append((widget, 18 if required else 0))
            row += 1
            return widget

        label(f"To: {FEEDBACK_EMAIL}\nSave a text report for email, then open your email app to attach it and send it.")
        label("Problem summary", required=True)
        self.summary = ctk.CTkEntry(body, placeholder_text="For example: Step 3 fails while flattening an image")
        self.summary.grid(row=row, column=0, sticky="ew", padx=8); row += 1
        label("What happened? Include steps to reproduce, expected results, and actual results.", required=True)
        self.description = ctk.CTkTextbox(body, height=110)
        self.description.grid(row=row, column=0, sticky="ew", padx=8); row += 1
        self.include_logs = tk.BooleanVar(self, value=True)
        logs_row = ctk.CTkFrame(body, fg_color="transparent")
        logs_row.grid(row=row, column=0, sticky="ew", padx=8, pady=(12, 6)); row += 1
        self.logs_check = ctk.CTkCheckBox(logs_row, text="Include relevant recent logs", variable=self.include_logs,
                                         command=self._invalidate_report)
        self.logs_check.pack(side="left")
        self.review_link = ctk.CTkLabel(
            logs_row, text="(review before sharing)", text_color=COLOR_PAIRS["disabled_text"],
            font=ctk.CTkFont(underline=True), cursor="arrow",
        )
        self.review_link.pack(side="left", padx=(4, 0))
        self.review_link.bind("<Button-1>", self._review_clicked)
        self.review_link._label.configure(takefocus=False)
        self.review_link.bind("<Return>", self._review_clicked)
        self.review_link.bind("<space>", self._review_clicked)
        label("Includes app and system information, processing settings, and recent errors.\n"
              "Automatic removal of private details is not guaranteed. Review before sharing.")
        label("Optional: screenshots, sample inputs, annotations, or R workspaces (10 MB total).\n"
              "Text reports include anonymous file details and checksums; they do not embed sample contents. "
              "Full data ZIPs include the selected files unchanged. Remove patient details and other private information first.")
        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.grid(row=row, column=0, sticky="ew", padx=8); row += 1
        self.add_button = AppButton(actions, text="Add files...", command=self._add_files)
        self.add_button.pack(side="left", padx=(0, 6))
        self.clear_button = AppButton(actions, text="Clear files", command=self._clear_files)
        self.clear_button.pack(side="left")
        self.files_label = label("No data files selected.")
        label("Secure sample-data link (optional)")
        self._sharing_link_placeholder = "https://... (your approved sharing service)"
        self.sharing_link = ctk.CTkEntry(body, placeholder_text=self._sharing_link_placeholder)
        self.sharing_link.grid(row=row, column=0, sticky="ew", padx=8); row += 1
        label("To share sample data, upload the ZIP yourself to a service approved by your organization "
              "and paste its HTTPS sharing link. The app does not upload files.")
        self.approved = tk.BooleanVar(self, value=False)
        footer = ctk.CTkFrame(self, fg_color=COLOR_PAIRS["surface"])
        footer.grid(row=2, column=0, sticky="ew", padx=12, pady=12)
        footer.grid_columnconfigure(0, weight=1)
        self.approval_check = ctk.CTkCheckBox(footer, text="I reviewed and have permission to share this report,\nselected files, and linked data with AIDaS support.",
                                            variable=self.approved)
        self.approval_check.grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(10, 4))
        self.status = tk.StringVar(self, value="Collecting diagnostics locally...")
        self.status_label = ctk.CTkLabel(footer, textvariable=self.status, anchor="w", justify="left", wraplength=440)
        self.status_label.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
        self.save_button = AppButton(footer, text="Save report...", variant="primary", command=self._save, state="disabled")
        self.save_button.grid(row=2, column=0, sticky="w", padx=8, pady=8)
        self.email_button = AppButton(footer, text="Open email draft", command=self._email, state="disabled")
        self.email_button.grid(row=2, column=1, padx=4, pady=8)
        AppButton(footer, text="Close", width=70, command=self.destroy).grid(row=2, column=2, padx=8, pady=8)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda _event: self.destroy())
        self.summary.bind("<KeyRelease>", self._invalidate_report, add="+")
        self.description.bind("<KeyRelease>", self._invalidate_report, add="+")
        self.sharing_link.configure(
            validate="key", validatecommand=(self.register(self._sharing_link_changed), "%P"),
        )
        def resize_labels(event):
            width = max(240, int(body._reverse_widget_scaling(event.width))-24)
            if width != self._wrap_width:
                self._wrap_width = width
                for widget, inset in self._wrapped_labels:
                    widget.configure(wraplength=max(1, width-inset))
        body.bind("<Configure>", resize_labels, add="+")
        def collect():
            snapshot = collect_diagnostics(context)
            # Prepare both immutable views off Tk, including large log payloads.
            previews = {logs: report_preview(snapshot, logs) for logs in (False, True)}
            return snapshot, previews

        self._start_worker("diagnostics", collect)
        synchronize_window_chrome(self, background=COLOR_PAIRS["window_chrome"], foreground=COLOR_PAIRS["text"])
        body.install_window_scrolling(self)
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
            if not error:
                self._snapshot, self._preview_text = result
            self.status.set("Review the diagnostics, describe the problem, then save a text report." if not error else f"Could not collect diagnostics: {error}")
        elif not error:
            self._report_path = result
            if result.suffix.lower() == ".zip":
                self.status.set(f"Saved {result.name}. Share this ZIP through your approved file-sharing service, "
                                "or save a text report for email. No email has been sent.")
            else:
                self.status.set(f"Saved {result.name}. Attach this text report to the email draft; no email has been sent.")
        else:
            self.status.set(f"Could not save report: {error}")
        self._set_editing(True)

    def _set_editing(self, enabled):
        state = "normal" if enabled else "disabled"
        for widget in (self.summary, self.description, self.sharing_link, self.logs_check, self.approval_check,
                       self.add_button, self.clear_button):
            widget.configure(state=state)
        self.save_button.configure(state="normal" if enabled and self._snapshot is not None else "disabled")
        self._set_review_enabled(enabled and self._snapshot is not None)
        self.email_button.configure(state="normal" if enabled and self._report_path is not None else "disabled")

    def _invalidate_report(self, _event=None):
        if self._report_path is None and not self.approved.get() and not self._saved_sharing_link:
            return
        self.approved.set(False)
        self._report_path = None
        self._saved_sharing_link = ""
        self.email_button.configure(state="disabled")

    def _set_review_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self._review_enabled:
            return
        self._review_enabled = enabled
        self.review_link.configure(
            text_color=COLOR_PAIRS["link"] if enabled else COLOR_PAIRS["disabled_text"],
            cursor="hand2" if enabled else "arrow",
        )
        self.review_link._label.configure(takefocus=enabled)

    def _review_clicked(self, _event=None):
        if self._review_enabled:
            self._preview()
        return "break"

    def _sharing_link_changed(self, value):
        # Validation observes typing, paste, and programmatic edits while keeping
        # CTk's placeholder available (it does not support a textvariable).
        value = "" if value == self._sharing_link_placeholder else value
        if value != self._sharing_link_value:
            self._sharing_link_value = value
            self._invalidate_report()
        return True

    def _preview(self):
        if self._snapshot is None:
            return
        window = self._preview_window
        new_window = window is None or not window.winfo_exists()
        if new_window:
            window = self._preview_window = _FeedbackPreviewWindow(self)
            window.withdraw()
            window.title("Review AIDaS diagnostics")
            window.transient(self)
            _fit_window(window, self, 740, 560)
            apply_app_icon_to(window)
            synchronize_window_chrome(window, background=COLOR_PAIRS["window_chrome"], foreground=COLOR_PAIRS["text"])
            window.box = ctk.CTkTextbox(window, wrap="word")
            window.box.pack(fill="both", expand=True, padx=12, pady=12)
            AppButton(window, text="Close", command=window.destroy).pack(pady=(0, 12))
        include_logs = bool(self.include_logs.get())
        if new_window or self._preview_includes_logs != include_logs:
            window.set_text(self._preview_text[include_logs])
            self._preview_includes_logs = include_logs
        window.deiconify()
        window.lift()

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
        try:
            sharing_link = validate_sharing_link(self.sharing_link.get())
        except ValueError as exc:
            messagebox.showwarning("Secure sample-data link", str(exc), parent=self)
            return
        if not self.approved.get():
            messagebox.showwarning("Review report", "Review the diagnostics and selected files, then check the approval box.", parent=self)
            return
        report_id = self._snapshot["diagnostics"]["report_id"]
        target = filedialog.asksaveasfilename(parent=self, title="Save diagnostic report", defaultextension=".txt",
                    filetypes=[("Text report for email", "*.txt"), ("Full data ZIP for secure sharing", "*.zip")],
                    initialfile=f"AIDaS-feedback-{report_id}.txt",
                    initialdir=str(Path.home() / "Downloads" if (Path.home() / "Downloads").is_dir() else Path.home()))
        if not target:
            return
        self._saved_title = title
        self._saved_sharing_link = sharing_link
        self._report_path = None
        self._set_editing(False)
        self.status.set("Creating full data ZIP locally..." if Path(target).suffix.lower() == ".zip"
                        else "Creating text report locally...")
        snapshot, files, logs, cancel = self._snapshot, tuple(self._attachments), self.include_logs.get(), self._cancel
        self._start_worker("report", lambda: save_report(target, snapshot, title=title, description=description,
                                                        attachments=files, include_logs=logs, cancel_event=cancel,
                                                        sharing_link=sharing_link))

    def _email(self):
        if self._report_path is None or not self._report_path.is_file():
            messagebox.showwarning("Feedback", "Save a report before opening an email draft.", parent=self)
            return
        try:
            sharing_link = validate_sharing_link(self._saved_sharing_link)
        except ValueError as exc:
            messagebox.showwarning("Secure sample-data link", str(exc), parent=self)
            return
        is_zip = self._report_path.suffix.lower() == ".zip"
        if is_zip and not sharing_link:
            messagebox.showwarning(
                "Share sample data securely",
                "Save a text report to attach to email, or upload the ZIP through an approved secure "
                "file-sharing service, paste its HTTPS link, and save the report again. "
                "Do not attach the ZIP to email.",
                parent=self,
            )
            return
        url = feedback_mailto(self._snapshot["diagnostics"]["report_id"], self._saved_title, self._report_path.name,
                              sharing_link=sharing_link)
        try:
            opened = webbrowser.open(url)
            if not opened:
                raise OSError("No default email application accepted the draft.")
        except Exception as exc:
            instructions = (
                f"Email {FEEDBACK_EMAIL} with your secure sample-data link:\n{sharing_link}"
                if is_zip else f"Email {FEEDBACK_EMAIL} and attach the text report:\n{self._report_path}"
            )
            if sharing_link and not is_zip:
                instructions += f"\n\nSecure sample-data link:\n{sharing_link}"
            messagebox.showinfo("Email draft", f"Could not open your email app: {exc}\n\n{instructions}", parent=self)
            return
        if is_zip:
            self.status.set("Draft opened with your secure sample-data link. Click Send in your email app.")
        else:
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
