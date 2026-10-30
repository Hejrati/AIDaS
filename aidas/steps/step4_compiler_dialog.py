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
from aidas.ui.compiler import CompilerPresentationMixin
from aidas.ui.components import AppButton, AppRunButton
from aidas.ui.path_entry import bind_path_entry
from aidas.ui.theme import COLOR_PAIRS, COLORS
from aidas.ui.windowing import centered_logical_geometry, synchronize_window_chrome
from aidas.utils.filesystem import existing_directory, normalize_path_input
from aidas.utils.ui_utils import (
    action_button,
    apply_app_icon_to,
)


class Step4CompilerDialog(CompilerPresentationMixin, ctk.CTkToplevel):
    """Collect compiler options and keep workbook work off the Tk thread."""

    PREFERRED_WIDTH = 720
    PREFERRED_HEIGHT = 620
    MINIMUM_WIDTH = 620
    MINIMUM_HEIGHT = 500
    EVENT_POLL_MS = 75
    COMPILER_PROGRESS_MESSAGE = "Compiling measurements…"

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
        input_folder = self._existing_directory(input_text)
        self._accepted_input_folder = str(input_folder) if input_folder is not None else ""
        output_folder = input_folder or Path.home()
        self.input_var = tk.StringVar(master=self, value=self._accepted_input_folder or input_text)
        self.include_fovea_var = tk.BooleanVar(master=self, value=True)
        self._default_output_path = str(output_folder / DEFAULT_OUTPUT_FILENAME)
        self.output_var = tk.StringVar(master=self, value=self._default_output_path)
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
        """Use one canonical spelling for inputs, defaults, and workbook paths."""
        return Path(normalize_path_input(value)).resolve()

    @classmethod
    def _existing_directory(cls, value: str | os.PathLike | None) -> Path | None:
        return existing_directory(value)

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
                "Choose a folder to search recursively for group and subject results, "
                "then choose the name and location of the combined Excel workbook. "
                "Thickness tables are cleaned in place and keep their original "
                "LIGHT filenames."
            ),
            justify="left",
            wraplength=660,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 10))

        input_group = ttk.LabelFrame(panel, text="Input folder", padding=(8, 6, 8, 8))
        input_group.grid(row=2, column=0, sticky="ew")
        input_group.columnconfigure(0, weight=1)
        self.input_entry = ttk.Entry(input_group, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        bind_path_entry(self.input_entry, self._open_input_folder)
        self.input_browse_button = action_button(
            input_group,
            self,
            "Browse…",
            self._browse_input,
            "folder",
            tooltip="Choose a folder containing group / subject / nasal or temporal results.",
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
        bind_path_entry(self.output_entry)
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
        self.close_button = AppButton(
            footer,
            text="Close",
            variant="danger",
            command=self.close,
            width=104,
        )
        self.close_button.pack(side="right")
        self.run_button = AppRunButton(
            footer,
            command=self._run_clicked,
        )
        self.run_button.pack(side="right", padx=(0, 6))

    def set_input_folder(self, folder: str | os.PathLike | None) -> bool:
        if self._running:
            return False
        path = self._existing_directory(folder)
        if path is None:
            return False
        self._accepted_input_folder = str(path)
        self.input_var.set(str(path))
        current_output = self.output_var.get().strip()
        previous_default = getattr(self, "_default_output_path", None)
        custom_output = bool(
            current_output and previous_default is not None
            and self._expanded_path(current_output) != self._expanded_path(previous_default)
        )
        if not self._output_user_selected and not custom_output:
            self._default_output_path = str(path / DEFAULT_OUTPUT_FILENAME)
            self.output_var.set(self._default_output_path)
        self.status_var.set(f"Ready to compile measurements in {path}.")
        return True

    def _open_input_folder(self) -> None:
        if self._running:
            return
        if self.set_input_folder(self.input_var.get()):
            return
        self.input_var.set(self._accepted_input_folder)
        messagebox.showerror(
            "Invalid folder",
            "Enter an existing folder path.",
            parent=self,
        )

    def _browse_input(self) -> None:
        initial_dir = (
            self._existing_directory(self.input_var.get())
            or self._existing_directory(self._accepted_input_folder)
            or Path.home()
        )
        selected = filedialog.askdirectory(
            title="Select a folder containing group and subject results",
            initialdir=str(initial_dir),
            parent=self,
        )
        if not selected:
            return
        self.set_input_folder(selected)

    def _browse_output(self) -> None:
        current_text = normalize_path_input(self.output_var.get())
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
        output_text = normalize_path_input(self.output_var.get())
        root_folder = self._existing_directory(self.input_var.get())
        if root_folder is None:
            messagebox.showerror(
                "Missing folder",
                "Choose an existing folder containing group and subject results.",
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

        self._accepted_input_folder = str(root_folder)
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

        self._start_compiler(
            root_folder, output_path, include_fovea,
            thread_name="aidas-step4-compiler",
        )

    def _compiler_function(self):
        return compile_step4_results

    def _poll_events(self) -> None:
        self._poll_after_id = None
        if self._closing:
            return
        self._drain_compile_events()
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
        )
        if running:
            self.status_var.set("Compiling measurements…")
            self.progress.configure(maximum=1, value=0)

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
