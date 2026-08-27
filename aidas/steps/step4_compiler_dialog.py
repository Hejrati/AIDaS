"""Themed Step 4 dialog for compiling measurements into one workbook."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import customtkinter as ctk

from aidas.services.step4_compiler import (
    DEFAULT_OUTPUT_FILENAME,
    compile_step4_results,
)
from aidas.ui.components import AppButton
from aidas.ui.theme import COLOR_PAIRS, COLORS, CONTROLS
from aidas.ui.windowing import centered_logical_geometry, synchronize_window_chrome
from aidas.utils.ui_utils import (
    action_button,
    apply_app_icon_to,
    load_color_close_ctk_icon,
    load_ctk_image,
)


class Step4CompilerDialog(ctk.CTkToplevel):
    """Collect compiler options and keep workbook work off the Tk thread."""

    PREFERRED_WIDTH = 720
    PREFERRED_HEIGHT = 620
    MINIMUM_WIDTH = 620
    MINIMUM_HEIGHT = 500
    EVENT_POLL_MS = 75

    def __init__(
        self,
        owner: tk.Misc,
        *,
        initial_input: str | os.PathLike | None = None,
        on_close: Callable[["Step4CompilerDialog"], None] | None = None,
        on_success: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__(owner)
        self.withdraw()
        self.title("Compile Step 4 Measurements")
        self.configure(fg_color=COLOR_PAIRS["surface"])
        self.minsize(self.MINIMUM_WIDTH, self.MINIMUM_HEIGHT)
        self.transient(owner.winfo_toplevel())
        apply_app_icon_to(self)

        self._owner = owner
        self._on_close_callback = on_close
        self._on_success_callback = on_success
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._poll_after_id: str | None = None
        self._worker: threading.Thread | None = None
        self._running = False
        self._closing = False
        self._output_user_selected = False

        input_text = str(initial_input or "")
        output_folder = self._existing_directory(input_text) or Path.home()
        self.input_var = tk.StringVar(master=self, value=input_text)
        self.include_fovea_var = tk.BooleanVar(master=self, value=True)
        self.output_var = tk.StringVar(
            master=self,
            value=str(output_folder / DEFAULT_OUTPUT_FILENAME),
        )
        self.status_var = tk.StringVar(master=self, value="Ready to compile.")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda _event: self.close())
        self.geometry(
            centered_logical_geometry(
                self,
                self.PREFERRED_WIDTH,
                self.PREFERRED_HEIGHT,
                parent=owner.winfo_toplevel(),
            )
        )
        self.deiconify()
        synchronize_window_chrome(
            self,
            background=COLOR_PAIRS["window_chrome"],
            foreground=COLOR_PAIRS["text"],
            border=COLOR_PAIRS["window_chrome"],
        )
        self.grab_set()
        self.input_entry.focus_set()
        self._poll_after_id = self.after(self.EVENT_POLL_MS, self._poll_events)

    @staticmethod
    def _expanded_path(value: str | os.PathLike) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(str(value).strip())))

    @classmethod
    def _existing_directory(cls, value: str | os.PathLike | None) -> Path | None:
        if not value:
            return None
        try:
            path = cls._expanded_path(value)
            if path.is_dir():
                return path
        except (OSError, RuntimeError, ValueError):
            pass
        return None

    def _build_ui(self) -> None:
        panel = ttk.Frame(self, padding=(16, 14, 16, 14))
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(5, weight=1)

        ttk.Label(
            panel,
            text="Compile rrMCP/AR, ELM-RPE, and ONL measurements",
            font=("", 12, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            panel,
            text=(
                "Choose the parent folder containing the LE and RE subject folders, "
                "then choose the name and location of the combined Excel workbook. "
                "Legacy-compatible LIGHT2 cleaned files may be created beside the "
                "source LIGHT text files."
            ),
            justify="left",
            wraplength=660,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 10))

        input_group = ttk.LabelFrame(panel, text="Input folder", padding=(8, 6, 8, 8))
        input_group.grid(row=2, column=0, sticky="ew")
        input_group.columnconfigure(0, weight=1)
        self.input_entry = ttk.Entry(input_group, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.input_browse_button = action_button(
            input_group,
            self,
            "Browse…",
            self._browse_input,
            "folder",
            tooltip="Choose the parent folder containing LE and RE subfolders.",
        )
        self.input_browse_button.grid(row=0, column=1, sticky="e")
        self.include_fovea_checkbox = ttk.Checkbutton(
            input_group,
            variable=self.include_fovea_var,
            text="Add fovea (RPEtoOLM) as an extra top row in ELM-RPE",
        )
        self.include_fovea_checkbox.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(7, 0),
        )

        output_group = ttk.LabelFrame(panel, text="Output workbook", padding=(8, 6, 8, 8))
        output_group.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        output_group.columnconfigure(0, weight=1)
        self.output_entry = ttk.Entry(output_group, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.output_browse_button = action_button(
            output_group,
            self,
            "Browse…",
            self._browse_output,
            "save",
            tooltip="Choose the Excel workbook name and save location.",
        )
        self.output_browse_button.grid(row=0, column=1, sticky="e")

        progress_row = ttk.Frame(panel)
        progress_row.grid(row=4, column=0, sticky="ew", pady=(10, 8))
        progress_row.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(
            progress_row,
            mode="determinate",
            maximum=1,
            value=0,
        )
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            progress_row,
            textvariable=self.status_var,
            style="AIDaS.Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        log_group = ttk.LabelFrame(panel, text="Log", padding=(6, 5, 6, 6))
        log_group.grid(row=5, column=0, sticky="nsew")
        log_group.rowconfigure(0, weight=1)
        log_group.columnconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_group,
            height=10,
            wrap="word",
            state="disabled",
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLORS.border,
            highlightcolor=COLORS.primary,
            background=COLORS.surface_subtle,
            foreground=COLORS.text,
            insertbackground=COLORS.text,
            selectbackground=COLORS.primary,
            selectforeground=COLORS.on_primary,
        )
        log_scrollbar = ttk.Scrollbar(log_group, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")

        footer = ttk.Frame(panel)
        footer.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        self.close_button_icon = load_color_close_ctk_icon(self, size=CONTROLS.icon_size)
        self.close_button = AppButton(
            footer,
            text="Close",
            variant="secondary",
            command=self.close,
            image=self.close_button_icon,
            compound="left",
            width=104,
        )
        self.close_button.pack(side="right")
        self.run_button_icon = load_ctk_image(
            self,
            "flat-color-icons--process.png",
            size=CONTROLS.icon_size,
        )
        self.run_button = AppButton(
            footer,
            text="Run compiler",
            variant="primary",
            command=self._run_clicked,
            image=self.run_button_icon,
            compound="left",
            width=142,
        )
        self.run_button.pack(side="right", padx=(0, 6))

    def _browse_input(self) -> None:
        initial_dir = self._existing_directory(self.input_var.get()) or Path.home()
        selected = filedialog.askdirectory(
            title="Select the parent folder containing LE and RE subfolders",
            initialdir=str(initial_dir),
            parent=self,
        )
        if not selected:
            return
        self.input_var.set(selected)
        if not self._output_user_selected:
            self.output_var.set(str(Path(selected) / DEFAULT_OUTPUT_FILENAME))

    def _browse_output(self) -> None:
        current_text = self.output_var.get().strip()
        current_path = self._expanded_path(current_text) if current_text else None
        initial_dir = None
        if current_path is not None:
            initial_dir = self._existing_directory(current_path.parent)
        if initial_dir is None:
            initial_dir = self._existing_directory(self.input_var.get()) or Path.home()
        initial_file = (
            current_path.name
            if current_path is not None and current_path.name
            else DEFAULT_OUTPUT_FILENAME
        )
        selected = filedialog.asksaveasfilename(
            title="Save compiled workbook as",
            defaultextension=".xlsx",
            initialdir=str(initial_dir),
            initialfile=initial_file,
            filetypes=[("Excel workbook", "*.xlsx")],
            parent=self,
        )
        if not selected:
            return
        self._output_user_selected = True
        self.output_var.set(selected)

    def _validated_paths(self) -> tuple[Path, Path] | None:
        root_text = self.input_var.get().strip()
        output_text = self.output_var.get().strip()
        root_folder = self._expanded_path(root_text) if root_text else None
        if root_folder is None or not root_folder.is_dir():
            messagebox.showerror(
                "Missing folder",
                "Choose a valid parent folder containing the LE and RE subfolders.",
                parent=self,
            )
            return None
        if not output_text:
            messagebox.showerror(
                "Missing output",
                "Choose where to save the compiled Excel workbook.",
                parent=self,
            )
            return None

        output_path = self._expanded_path(output_text)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".xlsx")
        elif output_path.suffix.lower() != ".xlsx":
            messagebox.showerror(
                "Invalid output",
                "The compiled workbook must use the .xlsx file extension.",
                parent=self,
            )
            return None
        if not output_path.parent.is_dir():
            messagebox.showerror(
                "Invalid output folder",
                "Choose an existing folder for the compiled workbook.",
                parent=self,
            )
            return None

        self.input_var.set(str(root_folder))
        self.output_var.set(str(output_path))
        return root_folder, output_path

    def _run_clicked(self) -> None:
        if self._running:
            return
        paths = self._validated_paths()
        if paths is None:
            return
        root_folder, output_path = paths
        include_fovea = bool(self.include_fovea_var.get())

        self._clear_log()
        self._set_running(True)
        self._worker = threading.Thread(
            target=self._compile_worker,
            args=(root_folder, output_path, include_fovea),
            name="aidas-step4-compiler",
            daemon=True,
        )
        self._worker.start()

    def _compile_worker(
        self,
        root_folder: Path,
        output_path: Path,
        include_fovea: bool,
    ) -> None:
        def emit(message: object) -> None:
            self._events.put(("log", str(message)))

        def report_progress(completed: int, total: int, message: str) -> None:
            self._events.put(("progress", (completed, total, message)))

        try:
            result = compile_step4_results(
                root_folder,
                output_path,
                include_fovea=include_fovea,
                log_callback=emit,
                progress_callback=report_progress,
            )
        except Exception as exc:
            self._events.put(("error", exc))
            return
        self._events.put(("success", result))

    def _poll_events(self) -> None:
        self._poll_after_id = None
        if self._closing:
            return
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "progress":
                    self._update_progress(payload)
                elif kind == "success":
                    self._compilation_succeeded(payload)
                elif kind == "error":
                    self._compilation_failed(payload)
        except queue.Empty:
            pass
        if not self._closing:
            self._poll_after_id = self.after(self.EVENT_POLL_MS, self._poll_events)

    def _set_running(self, running: bool) -> None:
        self._running = bool(running)
        state = "disabled" if running else "normal"
        for control in (
            self.input_entry,
            self.input_browse_button,
            self.include_fovea_checkbox,
            self.output_entry,
            self.output_browse_button,
        ):
            control.configure(state=state)
        self.run_button.configure(
            state=state,
            text="Compiling…" if running else "Run compiler",
        )
        if running:
            self.status_var.set("Compiling measurements…")
            self.progress.configure(maximum=1, value=0)

    def _update_progress(self, payload: object) -> None:
        try:
            completed_value, total_value, message_value = payload
            total = max(1, int(total_value))
            completed = min(total, max(0, int(completed_value)))
        except (TypeError, ValueError):
            return

        self.progress.configure(maximum=total, value=completed)
        percentage = round((completed / total) * 100)
        message = str(message_value).strip() or "Compiling measurements…"
        self.status_var.set(f"{message} ({percentage}%)")

    def _compilation_succeeded(self, result: object) -> None:
        self._set_running(False)
        maximum = max(1.0, float(self.progress.cget("maximum")))
        self.progress.configure(value=maximum)
        result_path = Path(getattr(result, "output_path", self.output_var.get()))
        self.output_var.set(str(result_path))
        self.status_var.set(f"Saved workbook: {result_path}")
        self.update_idletasks()
        callback = self._on_success_callback
        if callback is not None:
            callback(result_path)

    def _compilation_failed(self, error: object) -> None:
        self._set_running(False)
        self.status_var.set("Compilation failed. See the log for details.")
        self._append_log(f"\nERROR: {error}")
        messagebox.showerror(
            "Compilation failed",
            str(error),
            parent=self,
        )

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def close(self) -> None:
        if self._closing:
            return
        if self._running:
            messagebox.showwarning(
                "Compilation in progress",
                "Wait for the compiler to finish before closing this window.",
                parent=self,
            )
            return
        self._closing = True
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        try:
            self.grab_release()
        except tk.TclError:
            pass
        callback = self._on_close_callback
        try:
            self.destroy()
        finally:
            if callback is not None:
                callback(self)


__all__ = ["Step4CompilerDialog"]
