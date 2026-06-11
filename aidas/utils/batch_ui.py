"""Shared batch-list widgets for AIDaS workflow panels."""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont, ttk


class BatchTable(ttk.Frame):
    """Responsive grid table used by embedded batch workflow panels."""

    HEADER_HEIGHT = 34
    ROW_HEIGHT = 34
    GRID_COLOR = "#d1d5db"
    HEADER_BG = "#f3f4f6"
    ROW_BG = "#ffffff"
    ALT_ROW_BG = "#f9fafb"
    TEXT_COLOR = "#111827"
    MUTED_COLOR = "#6b7280"

    def __init__(
        self,
        parent,
        *,
        columns,
        min_widths=None,
        max_widths=None,
        stretch_column="folder",
        select_column=None,
        empty_message="No rows to show.",
        value_getter=None,
        include_key="include",
        locked_key="locked",
    ):
        super().__init__(parent)
        self.columns = tuple(columns)
        self.min_widths = dict(min_widths or {})
        self.max_widths = dict(max_widths or {})
        self.stretch_column = stretch_column
        self.select_column = select_column
        self.empty_message = empty_message
        self.value_getter = value_getter
        self.include_key = include_key
        self.locked_key = locked_key

        self.rows = []
        self.header_cells = {}
        self.empty_message_cell = None
        self.table_filler_cells = []
        self._table_resize_after_id = None
        self._bulk_updating = False
        self._table_font = tkfont.nametofont("TkDefaultFont")
        self._header_font = self._table_font.copy()
        self._header_font.configure(weight="bold")
        self.select_all_ready_var = tk.BooleanVar(value=False)
        self.select_all_ready_check = None
        self.column_widths = {key: width for key, _title, width, _anchor in self.columns}
        self.table_width = sum(self.column_widths.values()) + len(self.columns)

        self._build_ui()

    def _build_ui(self):
        self.table_canvas = tk.Canvas(
            self,
            bg=self.ROW_BG,
            highlightthickness=1,
            highlightbackground=self.GRID_COLOR,
        )
        self.table_yscroll = ttk.Scrollbar(self, orient="vertical", command=self.table_canvas.yview)
        self.table_xscroll = ttk.Scrollbar(self, orient="horizontal", command=self._table_xview)

        self.table_inner = tk.Frame(self.table_canvas, bg=self.GRID_COLOR)
        self.table_window = self.table_canvas.create_window((0, 0), window=self.table_inner, anchor="nw")

        self.table_canvas.configure(yscrollcommand=self.table_yscroll.set, xscrollcommand=self.table_xscroll.set)
        self.table_canvas.grid(row=0, column=0, sticky="nsew")
        self.table_yscroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._build_table_header()
        self.table_inner.bind("<Configure>", self._on_table_inner_configure)
        self.table_canvas.bind("<Configure>", self._on_table_canvas_configure, add="+")
        self.table_canvas.bind("<MouseWheel>", self._on_table_mousewheel, add="+")

    def _build_table_header(self):
        for col, (key, title, _width, anchor) in enumerate(self.columns):
            cell = self._make_table_cell(
                self.table_inner,
                row=0,
                col=col,
                width=self._column_width(key),
                height=self.HEADER_HEIGHT,
                bg=self.HEADER_BG,
            )
            self.header_cells[key] = cell
            if key == self.select_column:
                self.select_all_ready_check = ttk.Checkbutton(
                    cell,
                    variable=self.select_all_ready_var,
                    command=self._toggle_all_ready,
                )
                self.select_all_ready_check.pack(anchor="center", expand=True)
                self.select_all_ready_check.state(["disabled"])
                continue

            label = tk.Label(
                cell,
                text=title,
                anchor=anchor,
                bg=self.HEADER_BG,
                fg=self.TEXT_COLOR,
                font=self._header_font,
                padx=8,
            )
            label.pack(fill="both", expand=True)

    def _make_table_cell(self, parent, row, col, width, height, bg):
        cell = tk.Frame(parent, bg=bg, width=width, height=height)
        cell.grid(row=row, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))
        cell.grid_propagate(False)
        return cell

    def _column_width(self, key):
        return int(self.column_widths.get(key, self.min_widths.get(key, 80)))

    def _row_values(self, row):
        if self.value_getter is not None:
            return dict(self.value_getter(row))
        return dict(row.get("values") or {})

    def _text_width(self, text, *, header=False, padding=18):
        font = self._header_font if header else self._table_font
        return int(font.measure(str(text or ""))) + int(padding)

    def _fit_content_column_width(self, key, title, values):
        measured = [self._text_width(title, header=True)]
        measured.extend(self._text_width(value) for value in values)
        width = max([self.min_widths.get(key, 80), *measured])
        max_width = self.max_widths.get(key)
        if max_width is not None:
            width = min(width, max_width)
        return int(width)

    def _on_table_canvas_configure(self, _event=None):
        if self._bulk_updating:
            return
        if self._table_resize_after_id is not None:
            try:
                self.after_cancel(self._table_resize_after_id)
            except tk.TclError:
                pass
        self._table_resize_after_id = self.after_idle(self.fit_to_window)

    def fit_to_window(self):
        self._table_resize_after_id = None
        try:
            available = max(1, int(self.table_canvas.winfo_width()) - 2)
            visible_height = max(1, int(self.table_canvas.winfo_height()) - 2)
        except tk.TclError:
            return

        row_values = [self._row_values(row) for row in self.rows]
        gap_width = len(self.columns)
        computed = {}
        fixed_width = gap_width
        stretch_key = self.stretch_column if self.stretch_column in {key for key, *_ in self.columns} else None

        for key, title, _default_width, _anchor in self.columns:
            if key == stretch_key:
                continue
            values = [values.get(key, "") for values in row_values]
            width = self._fit_content_column_width(key, title, values)
            computed[key] = width
            fixed_width += width

        if stretch_key is not None:
            title = next(title for key, title, *_rest in self.columns if key == stretch_key)
            values = [values.get(stretch_key, "") for values in row_values]
            content_width = self._fit_content_column_width(stretch_key, title, values)
            min_width = self.min_widths.get(stretch_key, content_width)
            computed[stretch_key] = max(content_width, min_width, available - fixed_width)

        self.column_widths = computed
        self.table_width = sum(self.column_widths.values()) + gap_width
        self._set_horizontal_scrollbar_visible(self.table_width > available + 1)
        self._apply_table_size(visible_height)

    def _set_vertical_scrollbar_visible(self, visible):
        try:
            managed = bool(self.table_yscroll.winfo_manager())
            changed = False
            if visible and not managed:
                self.table_yscroll.grid(row=0, column=1, sticky="ns")
                changed = True
            elif not visible and managed:
                self.table_yscroll.grid_remove()
                self.table_canvas.yview_moveto(0)
                changed = True
            if changed and self._table_resize_after_id is None:
                self._table_resize_after_id = self.after_idle(self.fit_to_window)
        except tk.TclError:
            pass

    def _set_horizontal_scrollbar_visible(self, visible):
        try:
            managed = bool(self.table_xscroll.winfo_manager())
            if visible and not managed:
                self.table_xscroll.grid(row=1, column=0, sticky="ew")
            elif not visible and managed:
                self.table_xscroll.grid_remove()
        except tk.TclError:
            pass

    def _apply_table_size(self, visible_height=None):
        try:
            if visible_height is None:
                visible_height = max(1, int(self.table_canvas.winfo_height()) - 2)
        except tk.TclError:
            return

        for key, cell in self.header_cells.items():
            try:
                cell.configure(width=self._column_width(key))
            except tk.TclError:
                pass
        for row in self.rows:
            for key, cell in (row.get("cells") or {}).items():
                try:
                    cell.configure(width=self._column_width(key))
                except tk.TclError:
                    pass
        if self.empty_message_cell is not None:
            try:
                self.empty_message_cell.configure(width=self.table_width)
            except tk.TclError:
                pass
        content_overflows = self._update_table_filler(visible_height)
        self._set_vertical_scrollbar_visible(content_overflows)
        try:
            table_height = max(visible_height, self._table_content_height())
            self.table_canvas.itemconfigure(
                self.table_window,
                width=self.table_width,
                height=table_height,
            )
        except tk.TclError:
            return
        self._refresh_scrollregion()

    def _table_content_height(self):
        try:
            self.table_inner.update_idletasks()
            return int(self.table_inner.grid_bbox()[3])
        except tk.TclError:
            return 0

    def _visible_body_row_count(self):
        if self.rows:
            return len(self.rows)
        return 1 if self.empty_message_cell is not None else 0

    def _update_table_filler(self, viewport_height):
        for cell in self.table_filler_cells:
            try:
                cell.destroy()
            except tk.TclError:
                pass
        self.table_filler_cells = []

        content_rows = self._visible_body_row_count()
        used_height = self._table_content_height()
        filler_height = max(0, int(viewport_height) - used_height - 1)
        content_overflows = used_height > int(viewport_height) + 1
        if filler_height <= 0:
            return content_overflows

        filler_row = content_rows + 1
        for col, (key, _title, _width, _anchor) in enumerate(self.columns):
            cell = self._make_table_cell(
                self.table_inner,
                row=filler_row,
                col=col,
                width=self._column_width(key),
                height=filler_height,
                bg=self.ROW_BG,
            )
            self.table_filler_cells.append(cell)
        return content_overflows

    def _refresh_scrollregion(self):
        if self._bulk_updating:
            return
        try:
            self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))
        except tk.TclError:
            pass

    def _on_table_inner_configure(self, _event=None):
        self._refresh_scrollregion()

    def _table_xview(self, *args):
        self.table_canvas.xview(*args)

    def _on_table_mousewheel(self, event):
        if event.state & 0x0001:
            self._table_xview("scroll", -1 * int(event.delta / 120), "units")
        else:
            self.table_canvas.yview_scroll(-1 * int(event.delta / 120), "units")
        return "break"

    def set_rows(self, rows, *, empty_message=None):
        self.rows = list(rows or [])
        self.empty_message = self.empty_message if empty_message is None else empty_message
        self._bulk_updating = True
        try:
            try:
                self.table_canvas.itemconfigure(self.table_window, state="hidden")
            except tk.TclError:
                pass
            self._clear_table_body()
            self.empty_message_cell = None
            for idx, row in enumerate(self.rows):
                self._add_row(idx, row)
            if not self.rows:
                self._add_empty_message()
            self._refresh_select_all_ready_checkbox()
        finally:
            self._bulk_updating = False
            try:
                self.table_canvas.itemconfigure(self.table_window, state="normal")
            except tk.TclError:
                pass
        self._refresh_scrollregion()
        self.after_idle(self.fit_to_window)

    def _clear_table_body(self):
        for child in self.table_inner.winfo_children():
            try:
                row = int(child.grid_info().get("row", 0))
            except (TypeError, ValueError):
                row = 0
            if row > 0:
                child.destroy()
        self.table_filler_cells = []

    def _add_empty_message(self):
        cell = self._make_table_cell(
            self.table_inner,
            row=1,
            col=0,
            width=self.table_width,
            height=self.ROW_HEIGHT,
            bg=self.ROW_BG,
        )
        cell.grid(columnspan=len(self.columns))
        self.empty_message_cell = cell
        tk.Label(
            cell,
            text=self.empty_message,
            anchor="w",
            bg=self.ROW_BG,
            fg=self.MUTED_COLOR,
            padx=10,
        ).pack(fill="both", expand=True)

    def _add_row(self, idx, row):
        if self.select_column is not None:
            row["var"] = tk.BooleanVar(value=bool(row.get(self.include_key)))
        row["widgets"] = {}
        row["cells"] = {}
        bg = self.ROW_BG if idx % 2 == 0 else self.ALT_ROW_BG
        values = self._row_values(row)

        for col, (key, _title, _width, anchor) in enumerate(self.columns):
            cell = self._make_table_cell(
                self.table_inner,
                row=idx + 1,
                col=col,
                width=self._column_width(key),
                height=self.ROW_HEIGHT,
                bg=bg,
            )
            row["cells"][key] = cell
            if key == self.select_column:
                checkbutton = ttk.Checkbutton(
                    cell,
                    variable=row["var"],
                    command=lambda item=row: self._on_row_checkbutton_toggled(item),
                )
                checkbutton.pack(anchor="center", expand=True)
                if row.get(self.locked_key):
                    checkbutton.state(["disabled"])
                row["widgets"]["checkbutton"] = checkbutton
                continue

            label = tk.Label(
                cell,
                text=values.get(key, ""),
                anchor=anchor,
                bg=bg,
                fg=self.MUTED_COLOR if row.get(self.locked_key) else self.TEXT_COLOR,
                padx=8,
            )
            label.pack(fill="both", expand=True)
            row["widgets"][key] = label

    def refresh_row(self, row):
        if self.select_column is not None and "var" in row:
            row["var"].set(bool(row.get(self.include_key)))
        values = self._row_values(row)
        widgets = row.get("widgets") or {}
        for key, widget in widgets.items():
            if key == "checkbutton":
                widget.state(["disabled"] if row.get(self.locked_key) else ["!disabled"])
                continue
            try:
                widget.configure(
                    text=values.get(key, ""),
                    fg=self.MUTED_COLOR if row.get(self.locked_key) else self.TEXT_COLOR,
                )
            except tk.TclError:
                pass

    def update_row(self, row, *, values=None, include=None, locked=None):
        if values is not None:
            row.setdefault("values", {}).update(values)
        if include is not None:
            row[self.include_key] = bool(include)
        if locked is not None:
            row[self.locked_key] = bool(locked)
        self.refresh_row(row)
        self._refresh_select_all_ready_checkbox()
        self.after_idle(self.fit_to_window)

    def _on_row_checkbutton_toggled(self, row):
        if row.get(self.locked_key):
            row[self.include_key] = False
            if "var" in row:
                row["var"].set(False)
            return
        row[self.include_key] = bool(row.get("var").get()) if row.get("var") is not None else False
        self._refresh_select_all_ready_checkbox()

    def _toggle_all_ready(self):
        self.set_all_ready_selection(bool(self.select_all_ready_var.get()))

    def set_all_ready_selection(self, include):
        for row in self.rows:
            if not row.get(self.locked_key):
                row[self.include_key] = bool(include)
                self.refresh_row(row)
        self._refresh_select_all_ready_checkbox()

    def _refresh_select_all_ready_checkbox(self):
        if self.select_all_ready_check is None:
            return
        ready_rows = [row for row in self.rows if not row.get(self.locked_key)]
        if not ready_rows:
            self.select_all_ready_var.set(False)
            self.select_all_ready_check.state(["disabled"])
            return
        self.select_all_ready_check.state(["!disabled"])
        self.select_all_ready_var.set(all(bool(row.get(self.include_key)) for row in ready_rows))

    def selected_rows(self):
        return [row for row in self.rows if row.get(self.include_key) and not row.get(self.locked_key)]
