"""Worker and event handling shared by the two measurement compiler views."""

from __future__ import annotations

from pathlib import Path
import threading


class CompilerPresentationMixin:
    """Keep computation off Tk while leaving presentation and lifecycle to views."""

    def _compiler_function(self):
        # Step 5 keeps the workbook dependency lazy until compilation starts.
        from aidas.services.step4_compiler import compile_step4_results

        return compile_step4_results

    def _start_compiler(self, root_folder, output_path, include_fovea, *, thread_name):
        self._clear_log()
        self._set_running(True)
        self._worker = threading.Thread(
            target=self._compile_worker,
            args=(root_folder, output_path, include_fovea),
            name=thread_name,
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
            result = self._compiler_function()(
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

    def _drain_compile_events(self) -> None:
        # Tk is the sole consumer; workers only publish events. Checking for
        # pending events is safe here and avoids raising Empty on every poll.
        while not self._events.empty():
            kind, payload = self._events.get_nowait()
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "progress":
                self._update_progress(payload)
            elif kind == "success":
                self._compilation_succeeded(payload)
            elif kind == "error":
                self._compilation_failed(payload)

    def _update_progress(self, payload: object) -> None:
        try:
            completed_value, total_value, message_value = payload
            total = max(1, int(total_value))
            completed = min(total, max(0, int(completed_value)))
        except (TypeError, ValueError):
            return
        self.progress.configure(maximum=total, value=completed)
        percentage = round((completed / total) * 100)
        message = str(message_value).strip() or self.COMPILER_PROGRESS_MESSAGE
        self.status_var.set(f"{message} ({percentage}%)")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
