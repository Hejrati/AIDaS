"""Step 5 - compile all workflow measurements into one workbook."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from aidas.ui.components import AppButton
from aidas.ui.theme import COLORS, COLOR_PAIRS, CONTROLS
from aidas.utils.ui_layout import LAYOUT
from aidas.utils.ui_utils import (
    SidebarStepFrame,
    action_button,
    load_compile_ctk_icon,
)


DEFAULT_OUTPUT_FILENAME = "LE_RE_rrMCPAR_ELMRPE_ONL.xlsx"


class Step5Frame(SidebarStepFrame):
    """Dedicated final-step UI for compiling Step 3 and Step 4 results."""

    EVENT_POLL_MS = 75

    def __init__(
        self,
        parent,
        preferences=None,
        source_step=None,
        on_compilation_complete=None,
    ):
        super().__init__(parent)
        self.preferences = preferences
        self.source_step = source_step
        self.on_compilation_complete = on_compilation_complete
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False
        self._output_user_selected = False

        initial_input = self._source_input_folder()
        output_folder = self._existing_directory(initial_input) or Path.home()
        self.input_var = tk.StringVar(value=initial_input)
        self.include_fovea_var = tk.BooleanVar(value=True)
        self.output_var = tk.StringVar(
            value=str(output_folder / DEFAULT_OUTPUT_FILENAME)
        )
        self.status_var = tk.StringVar(value="Ready to compile measurements.")

        self.input_var.trace_add("write", self._update_run_button_state)
        self.output_var.trace_add("write", self._update_run_button_state)
        self._build_ui()
        self._update_run_button_state()
        self.after(self.EVENT_POLL_MS, self._poll_events)

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

    def _source_input_folder(self) -> str:
        source = self.source_step
        if source is None:
            return ""
        for value in (
            getattr(source, "batch_roi_root", None),
            getattr(getattr(source, "output_dir_var", None), "get", lambda: "")(),
            getattr(getattr(source, "input_dir_var", None), "get", lambda: "")(),
        ):
            if value:
                return str(value)
        return ""

    def _build_ui(self) -> None:
        self.build_standard_layout(
            sidebar_width=self.SIDEBAR_WIDTH,
            status_var=self.status_var,
            status_bar_content_margin=True,
        )

        input_section = self.add_sidebar_section(
            "Measurement folders",
            pady=(0, LAYOUT.space_xs),
        )
        input_group = input_section.body
        ttk.Label(
            input_group,
            text=(
                "Choose the parent folder containing the LE and RE subject "
                "folders produced by the earlier workflow steps."
            ),
            justify="left",
            wraplength=self.SIDEBAR_TEXT_WRAP,
        ).pack(fill="x", pady=(0, LAYOUT.space_xs))
        input_row = ttk.Frame(input_group)
        input_row.pack(fill="x")
        self.input_entry = ttk.Entry(input_row, textvariable=self.input_var)
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.input_browse_button = action_button(
            input_row,
            self,
            "Browse…",
            self._browse_input,
            "folder",
            tooltip="Choose the parent folder containing LE and RE subfolders.",
        )
        self.input_browse_button.pack(side="right")
        self.include_fovea_checkbox = ttk.Checkbutton(
            input_group,
            variable=self.include_fovea_var,
            text="Add fovea (RPEtoOLM) to ELM-RPE",
        )
        self.include_fovea_checkbox.pack(anchor="w", pady=(7, 0))

        output_section = self.add_sidebar_section(
            "Compiled workbook",
            pady=(0, LAYOUT.space_xs),
        )
        output_group = output_section.body
        output_row = ttk.Frame(output_group)
        output_row.pack(fill="x")
        self.output_entry = ttk.Entry(output_row, textvariable=self.output_var)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.output_browse_button = action_button(
            output_row,
            self,
            "Browse…",
            self._browse_output,
            "save",
            tooltip="Choose the Excel workbook name and save location.",
        )
        self.output_browse_button.pack(side="right")

        footer = ttk.Frame(self.sidebar_shell, style="AIDaS.Sidebar.TFrame")
        footer.pack(
            side="bottom",
            fill="x",
            padx=(LAYOUT.space_sm, LAYOUT.space_xs),
            pady=(0, LAYOUT.space_sm),
            before=self.sidebar,
        )
        self.run_button_icon = load_compile_ctk_icon(
            self,
            size=CONTROLS.icon_size,
        )
        self.run_button_disabled_icon = load_compile_ctk_icon(
            self,
            size=CONTROLS.icon_size,
            color_pair=COLOR_PAIRS["disabled_text"],
        )
        self.run_button = AppButton(
            footer,
            text="Compile all measurements",
            variant="success",
            command=self._run_clicked,
            state="disabled",
            image=self.run_button_disabled_icon,
            compound="left",
        )
        self.run_button.pack(fill="x")

        ttk.Label(
            self.content,
            text="Compile all measurements and results",
            font=("", 14, "bold"),
            anchor="w",
        ).pack(fill="x")
        ttk.Label(
            self.content,
            text=(
                "Step 5 combines rrMCP/AR measurements from Step 4 with the "
                "ELM-RPE and ONL exports from Step 3 into one Excel workbook."
            ),
            justify="left",
            wraplength=760,
        ).pack(fill="x", pady=(4, 12))

        progress_group = ttk.LabelFrame(
            self.content,
            text="Compilation progress",
            padding=(10, 8, 10, 10),
        )
        progress_group.pack(fill="x", pady=(0, 10))
        self.progress = ttk.Progressbar(
            progress_group,
            mode="determinate",
            maximum=1,
            value=0,
        )
        self.progress.pack(fill="x")
        ttk.Label(
            progress_group,
            textvariable=self.status_var,
            style="AIDaS.Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        log_group = ttk.LabelFrame(
            self.content,
            text="Compiler log",
            padding=(6, 5, 6, 6),
        )
        log_group.pack(fill="both", expand=True)
        log_group.rowconfigure(0, weight=1)
        log_group.columnconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_group,
            height=12,
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
        scrollbar = ttk.Scrollbar(log_group, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def set_input_folder(self, folder: str | os.PathLike | None) -> bool:
        path = self._existing_directory(folder)
        if path is None:
            return False
        self.input_var.set(str(path))
        if not self._output_user_selected:
            self.output_var.set(str(path / DEFAULT_OUTPUT_FILENAME))
        self.status_var.set(f"Ready to compile measurements under {path}.")
        return True

    def on_show(self) -> None:
        if not self.input_var.get().strip():
            self.set_input_folder(self._source_input_folder())

    def _browse_input(self) -> None:
        initial_dir = self._existing_directory(self.input_var.get()) or Path.home()
        selected = filedialog.askdirectory(
            title="Select the parent folder containing LE and RE subfolders",
            initialdir=str(initial_dir),
            parent=self,
        )
        if selected:
            self.set_input_folder(selected)

    def _browse_output(self) -> None:
        current_text = self.output_var.get().strip()
        current_path = self._expanded_path(current_text) if current_text else None
        initial_dir = (
            self._existing_directory(current_path.parent)
            if current_path is not None
            else None
        )
        if initial_dir is None:
            initial_dir = self._existing_directory(self.input_var.get()) or Path.home()
        selected = filedialog.asksaveasfilename(
            title="Save compiled workbook as",
            defaultextension=".xlsx",
            initialdir=str(initial_dir),
            initialfile=(
                current_path.name
                if current_path is not None and current_path.name
                else DEFAULT_OUTPUT_FILENAME
            ),
            filetypes=[("Excel workbook", "*.xlsx")],
            parent=self,
        )
        if selected:
            self._output_user_selected = True
            self.output_var.set(selected)

    def _validated_paths(self) -> tuple[Path, Path] | None:
        root_folder = self._existing_directory(self.input_var.get())
        if root_folder is None:
            messagebox.showerror(
                "Missing folder",
                "Choose a valid parent folder containing the LE and RE subfolders.",
                parent=self,
            )
            return None
        output_text = self.output_var.get().strip()
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
        self._clear_log()
        self._set_running(True)
        self._worker = threading.Thread(
            target=self._compile_worker,
            args=(*paths, bool(self.include_fovea_var.get())),
            name="aidas-step5-compiler",
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
            from aidas.services.step4_compiler import compile_step4_results

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
        try:
            self.after(self.EVENT_POLL_MS, self._poll_events)
        except tk.TclError:
            pass

    def _update_run_button_state(self, *_args) -> None:
        if getattr(self, "run_button", None) is None:
            return
        if self._running:
            return
            
        has_input = bool(self.input_var.get().strip())
        has_output = bool(self.output_var.get().strip())
        
        if has_input and has_output:
            self.run_button.configure(image=self.run_button_icon)
            self.run_button.state(["!disabled"])
        else:
            self.run_button.configure(image=self.run_button_disabled_icon)
            self.run_button.state(["disabled"])

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
        
        if running:
            self.run_button.configure(
                state="disabled",
                text="Compiling…",
                image=self.run_button_disabled_icon,
            )
            self.status_var.set("Compiling measurements…")
            self.progress.configure(maximum=1, value=0)
        else:
            self.run_button.configure(text="Compile all measurements")
            self._update_run_button_state()

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
        self.status_var.set(f"Saved compiled workbook: {result_path}")
        self._append_log(f"Saved compiled workbook: {result_path}")
        callback = getattr(self, "on_compilation_complete", None)
        if callable(callback):
            callback(result_path)
        messagebox.showinfo(
            "Compilation complete",
            f"All measurements were compiled into:\n{result_path}",
            parent=self,
        )

    def _compilation_failed(self, error: object) -> None:
        self._set_running(False)
        self.status_var.set("Compilation failed. See the log for details.")
        self._append_log(f"ERROR: {error}")
        messagebox.showerror("Compilation failed", str(error), parent=self)

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


__all__ = ["Step5Frame"]
