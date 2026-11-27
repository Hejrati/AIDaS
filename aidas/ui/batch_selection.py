"""Shared batch selection tables and folder scan lifecycle."""

import queue
import threading
import tkinter as tk

from aidas.utils.worker_polling import WorkerCompletionGuard


class BatchFolderScanMixin:
    """Deliver a cancellable directory scan to its selection panel."""

    SCAN_POLL_MS = 50

    def _start_folder_scan(self, *scan_args):
        self._stop_scan()
        self._scan_closed = False
        self._scan_ready = False
        cancelled = self._scan_cancel = threading.Event()
        events = self._scan_events = queue.Queue()
        self.next_button.state(["disabled"])
        try:
            worker = threading.Thread(target=self._scan_worker, args=(*scan_args, cancelled, events), daemon=True)
            worker.start()
        except Exception as exc:
            self._scan_failed(exc)
            return
        self._scan_completion = WorkerCompletionGuard(worker)
        self._scan_poll_after_id = self.after(self.SCAN_POLL_MS, self._poll_scan_events)

    def _poll_scan_events(self):
        self._scan_poll_after_id = None
        if self.__dict__.get("_scan_closed", False):
            return
        try:
            kind, payload = self._scan_events.get_nowait()
        except queue.Empty:
            guard = self.__dict__.get("_scan_completion")
            if guard is not None and guard.missing_result():
                self._scan_failed(RuntimeError(
                    "The folder scan stopped without reporting a result. Select another folder to try again."
                ))
            else:
                try:
                    self._scan_poll_after_id = self.after(self.SCAN_POLL_MS, self._poll_scan_events)
                except tk.TclError:
                    pass
            return
        self._scan_completion = None
        if kind == "error":
            self._scan_failed(payload)
        else:
            self._scan_done(*payload)

    def _stop_scan(self):
        cancelled = self.__dict__.get("_scan_cancel")
        if cancelled is not None:
            cancelled.set()
        pending = self.__dict__.get("_scan_poll_after_id")
        self._scan_poll_after_id = None
        self._scan_completion = None
        if pending is not None:
            self.after_cancel(pending)


class BatchSelectionMixin:
    """Use each table's row values, images, and selection callback unchanged."""

    def _make_checkbox_images(self, *, locked_check=False):
        images = {
            state: tk.PhotoImage(master=self, width=16, height=16)
            for state in ("checked", "unchecked", "locked")
        }
        for image in images.values():
            image.put("#ffffff", to=(0, 0, 16, 16))
            for border in ((2, 2, 14, 3), (2, 13, 14, 14), (2, 2, 3, 14), (13, 2, 14, 14)):
                image.put("#6b7280", to=border)

        def draw_check(image, color):
            for x, y in ((4, 8), (5, 9), (6, 10), (7, 9), (8, 8), (9, 7), (10, 6), (11, 5)):
                image.put(color, to=(x, y, x + 1, y + 2))

        draw_check(images["checked"], "#111827")
        locked = images["locked"]
        locked.put("#e5e7eb", to=(3, 3, 13, 13))
        if locked_check:
            draw_check(locked, "#9ca3af")
        else:
            locked.put("#9ca3af", to=(5, 7, 11, 9))
        return images

    def _replace_rows(self, rows):
        # Include detached rows when removing entries from the previous scan.
        previous = tuple(dict.fromkeys((*self._row_by_iid, *self.tree.get_children(""))))
        if previous:
            self.tree.delete(*previous)
        self.rows = list(rows or [])
        self._row_by_iid = {}
        for index, row in enumerate(self.rows):
            iid = str(index)
            self._row_by_iid[iid] = row
            self.tree.insert(
                "", "end", iid=iid, image=self._image_for_row(row),
                values=self._values_for_row(row), tags=("locked",) if row.get("locked") else (),
            )

    def _filter_processed_rows(self, placeholder_iid, empty_message):
        if self.tree.exists(placeholder_iid):
            self.tree.delete(placeholder_iid)
        visible = tuple(
            iid for iid, row in self._row_by_iid.items()
            if not (self._hide_processed and row.get("locked"))
        )
        self.tree.set_children("", *visible)
        if not visible:
            message = "No unprocessed folders to show." if self.rows and self._hide_processed else empty_message
            self.tree.insert("", "end", iid=placeholder_iid, values=(message, "", ""), tags=("locked",))

    def _image_for_row(self, row):
        if row.get("locked"):
            return self._checkbox_images["locked"]
        if row.get("include"):
            return self._checkbox_images["checked"]
        return self._checkbox_images["unchecked"]

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
        self._notify_selection_changed()
        return "break"

    def _toggle_all_ready(self):
        ready_rows = [row for row in self.rows if not row.get("locked")]
        if not ready_rows:
            return
        include = not all(bool(row.get("include")) for row in ready_rows)
        for iid, row in self._row_by_iid.items():
            if row.get("locked"):
                continue
            changed = bool(row.get("include")) != include
            row["include"] = include
            if changed:
                self._refresh_row(iid, row)
        self._refresh_header_checkbox()
        self._notify_selection_changed()

    def selected_rows(self):
        return [row for row in self.rows if row.get("include") and not row.get("locked")]
