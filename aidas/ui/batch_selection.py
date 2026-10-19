"""Shared checkbox selection behavior for workflow batch Treeviews."""

import tkinter as tk


class BatchSelectionMixin:
    """Use each table's row values, images, and selection callback unchanged."""

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
            row["include"] = include
            self._refresh_row(iid, row)
        self._refresh_header_checkbox()
        self._notify_selection_changed()

    def selected_rows(self):
        return [row for row in self.rows if row.get("include") and not row.get("locked")]
