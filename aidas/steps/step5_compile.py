"""Step 5 - create an Excel file containing all workflow measurements."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from aidas.ui.compiler import CompilerPresentationMixin
from aidas.ui.components import AppButton
from aidas.ui.measurement_tree import MeasurementDirectoryTree
from aidas.ui.path_entry import bind_path_entry
from aidas.ui.theme import COLOR_PAIRS, CONTROLS
from aidas.utils.filesystem import existing_directory, normalize_path_input
from aidas.utils.ui_layout import LAYOUT
from aidas.utils.ui_utils import (
    SidebarStepFrame,
    action_button,
    load_compile_ctk_icon,
)


DEFAULT_OUTPUT_FILENAME = "rrMCPAR_ELMRPE_ONL.xlsx"


class Step5Frame(CompilerPresentationMixin, SidebarStepFrame):
    """Dedicated final-step UI for compiling Step 3 and Step 4 results."""

    EVENT_POLL_MS = 75
    COMPILER_PROGRESS_MESSAGE = "Creating the Excel file…"

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

        # Step 4 folders arrive only through the explicit Go to Step 5 action.
        initial_input = ""
        input_folder = self._existing_directory(initial_input)
        self._accepted_input_folder = str(input_folder) if input_folder is not None else ""
        output_folder = input_folder or Path.home()
        self.input_var = tk.StringVar(value=self._accepted_input_folder or initial_input)
        self.include_fovea_var = tk.BooleanVar(value=True)
        self._default_output_path = str(output_folder / DEFAULT_OUTPUT_FILENAME)
        self.output_var = tk.StringVar(value=self._default_output_path)
        self.status_var = tk.StringVar(value="Ready to create the Excel file.")

        self.input_var.trace_add("write", self._update_run_button_state)
        self.output_var.trace_add("write", self._update_run_button_state)
        self._build_ui()
        self._update_run_button_state()
        self._refresh_directory_view()
        self.after(self.EVENT_POLL_MS, self._poll_events)

    @staticmethod
    def _expanded_path(value: str | os.PathLike) -> Path:
        """Use one canonical spelling for inputs, defaults, and workbook paths."""
        return Path(normalize_path_input(value)).resolve()

    @classmethod
    def _existing_directory(cls, value: str | os.PathLike | None) -> Path | None:
        return existing_directory(value)

    def _source_input_folder(self) -> str:
        source = self.source_step
        if source is None:
            return ""
        root = getattr(source, "batch_roi_root", None)
        if root:
            return str(root)
        paths = getattr(source, "batch_roi_paths", None)
        if paths:
            # Step 3's handoff supplies exact side folders rather than a batch
            # root. Include all those folders, not just Step 4's active tab.
            try:
                return os.path.commonpath([str(Path(path).parent) for path in paths])
            except ValueError as exc:
                raise ValueError(
                    "The Step 4 folders have no shared root. Choose an input folder in Step 5 "
                    "and use Create Excel File."
                ) from exc
        if not getattr(source, "current_path", None) and not getattr(source, "_last_results_dir", None):
            return ""
        for value in (
            getattr(getattr(source, "output_dir_var", None), "get", lambda: "")(),
            getattr(getattr(source, "input_dir_var", None), "get", lambda: "")(),
        ):
            if value:
                return str(value)
        return ""

    @classmethod
    def _measurement_root(cls, folder: str | os.PathLike) -> Path:
        """Promote a Step 4 side/subject folder to include its group."""
        path = cls._expanded_path(folder)
        for ancestor in (path, *path.parents):
            if ancestor.name.strip().lower() in ("nasal", "temporal"):
                return ancestor.parent.parent
        if path.is_dir() and any(
            child.is_dir() and child.name.strip().lower() in ("nasal", "temporal")
            for child in path.iterdir()
        ):
            return path.parent
        return path

    def start_from_step4(self, folder: str | os.PathLike | None = None) -> None:
        """Accept the workflow handoff and start one background compilation."""
        if self._running:
            return
        source = self.source_step
        if folder is None and source is not None and not any(
            getattr(source, name, None)
            for name in ("batch_roi_root", "batch_roi_paths", "current_path", "_last_results_dir")
        ):
            # Merely visiting empty workflow tabs is not a measurement handoff.
            return
        try:
            source_folder = folder or self._source_input_folder()
            if not source_folder:
                self.status_var.set("Choose an input folder to create the Excel file.")
                return
            source_path = self._existing_directory(source_folder)
            accepted = (
                source_path is not None
                and self.set_input_folder(self._measurement_root(source_path))
            )
        except (OSError, ValueError) as exc:
            self._compilation_failed(exc)
            return
        if not accepted:
            self.status_var.set("The Step 4 input folder is unavailable. Choose an existing folder.")
            return
        self._run_clicked()

    def _build_ui(self) -> None:
        self.build_standard_layout(
            sidebar_width=self.SIDEBAR_WIDTH,
        )

        input_section = self.add_sidebar_section(
            "Input",
            pady=(0, LAYOUT.space_xs),
        )
        input_group = input_section.body
        ttk.Label(
            input_group,
            text=(
                "Choose a folder to search recursively for nasal and temporal Excel results. "
                "Their parent folder supplies the subject ID; its parent supplies the group name."
            ),
            justify="left",
            wraplength=self.SIDEBAR_TEXT_WRAP,
        ).pack(fill="x", pady=(0, LAYOUT.space_xs))
        input_row = ttk.Frame(input_group)
        input_row.pack(fill="x")
        self.input_entry = ttk.Entry(input_row, textvariable=self.input_var)
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        bind_path_entry(self.input_entry, self._open_input_folder)
        self.input_browse_button = action_button(
            input_row,
            self,
            "Browse…",
            self._browse_input,
            "folder",
            tooltip="Search within this folder for group / subject / nasal or temporal results.",
        )
        self.input_browse_button.pack(side="right")
        self.include_fovea_checkbox = ttk.Checkbutton(
            input_group,
            variable=self.include_fovea_var,
            text="Add fovea (RPEtoOLM) to ELM-RPE",
        )
        self.include_fovea_checkbox.pack(anchor="w", pady=(7, 0))

        output_section = self.add_sidebar_section(
            "Output",
            pady=(0, LAYOUT.space_xs),
        )
        output_group = output_section.body
        output_row = ttk.Frame(output_group)
        output_row.pack(fill="x")
        self.output_entry = ttk.Entry(output_row, textvariable=self.output_var)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        bind_path_entry(self.output_entry)
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
        ttk.Separator(footer, orient="horizontal").pack(
            fill="x", pady=(LAYOUT.space_xs, 6),
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
            text="Create Excel File",
            variant="success",
            command=self._run_clicked,
            state="disabled",
            image=self.run_button_disabled_icon,
            compound="left",
        )
        self.run_button.pack(fill="x", padx=2)

        ttk.Label(
            self.content,
            text="Create Excel File",
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

        self.directory_view = MeasurementDirectoryTree(self.content)
        self.directory_view.pack(fill="both", expand=True)
        ttk.Label(
            self.content,
            textvariable=self.status_var,
            style="AIDaS.Muted.TLabel",
            wraplength=850,
        ).pack(fill="x", pady=(8, 0))

    def _refresh_directory_view(self) -> None:
        view = self.__dict__.get("directory_view")
        if view is None:
            return
        folder = self._existing_directory(self.input_var.get())
        if folder is not None:
            output_text = normalize_path_input(self.output_var.get())
            output = self._expanded_path(output_text) if output_text else None
            view.set_directory(folder, output_path=output)

    def set_input_folder(self, folder: str | os.PathLike | None) -> bool:
        if getattr(self, "_running", False):
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
        self.status_var.set(f"Ready to create an Excel file from measurements in {path}.")
        self._refresh_directory_view()
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

    def on_show(self) -> None:
        """Refresh this workspace without accepting a new workflow handoff."""
        self._refresh_directory_view()

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
        if selected:
            self.set_input_folder(selected)

    def _browse_output(self) -> None:
        current_text = normalize_path_input(self.output_var.get())
        current_path = self._expanded_path(current_text) if current_text else None
        initial_dir = (
            self._existing_directory(current_path.parent)
            if current_path is not None
            else None
        )
        if initial_dir is None:
            initial_dir = self._existing_directory(self.input_var.get()) or Path.home()
        selected = filedialog.asksaveasfilename(
            title="Save Excel file as",
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
            self._refresh_directory_view()

    def _validated_paths(self) -> tuple[Path, Path] | None:
        root_folder = self._existing_directory(self.input_var.get())
        if root_folder is None:
            messagebox.showerror(
                "Missing folder",
                "Choose an existing folder containing group and subject results.",
                parent=self,
            )
            return None
        output_text = normalize_path_input(self.output_var.get())
        if not output_text:
            messagebox.showerror(
                "Missing output",
                "Choose where to save the Excel file.",
                parent=self,
            )
            return None
        output_path = self._expanded_path(output_text)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".xlsx")
        elif output_path.suffix.lower() != ".xlsx":
            messagebox.showerror(
                "Invalid output",
                "The Excel file must use the .xlsx file extension.",
                parent=self,
            )
            return None
        if not output_path.parent.is_dir():
            messagebox.showerror(
                "Invalid output folder",
                "Choose an existing folder for the Excel file.",
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
        self._start_compiler(
            *paths, bool(self.include_fovea_var.get()),
            thread_name="aidas-step5-compiler",
        )

    def _poll_events(self) -> None:
        self._drain_compile_events()
        try:
            self.after(self.EVENT_POLL_MS, self._poll_events)
        except tk.TclError:
            pass

    def _clear_log(self) -> None:
        """Step 5 presents the directory canvas instead of a compiler log."""

    def _append_log(self, message: str) -> None:
        """Detailed extraction information remains in the workbook's log sheet."""

    def _update_progress(self, payload: object) -> None:
        """Keep the directory visible while the compiler runs in the background."""

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
                text="Creating Excel File…",
                image=self.run_button_disabled_icon,
            )
            self.status_var.set("Creating the Excel file…")
        else:
            self.run_button.configure(text="Create Excel File")
            self._update_run_button_state()

    def _compilation_succeeded(self, result: object) -> None:
        self._set_running(False)
        result_path = Path(getattr(result, "output_path", self.output_var.get()))
        self.output_var.set(str(result_path))
        self.status_var.set(f"Excel file saved: {result_path}")
        self._refresh_directory_view()
        callback = getattr(self, "on_compilation_complete", None)
        if callable(callback):
            callback(result_path)
        messagebox.showinfo(
            "Excel file created",
            f"All measurements were saved to:\n{result_path}",
            parent=self,
        )

    def _compilation_failed(self, error: object) -> None:
        self._set_running(False)
        self.status_var.set(f"Could not create the Excel file: {error}")
        messagebox.showerror("Excel file creation failed", str(error), parent=self)


__all__ = ["Step5Frame"]
