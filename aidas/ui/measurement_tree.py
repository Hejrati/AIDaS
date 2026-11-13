"""Scrollable folder-tree canvas for the Excel compiler's selected directory."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import font as tkfont, ttk

import customtkinter as ctk
from PIL import Image, ImageDraw

from aidas.services.measurement_inventory import InventoryCancelled, scan_measurement_directory
from aidas.services.measurement_scan import MeasurementScanLimitExceeded
from aidas.ui.components import AppButton
from aidas.ui.images import OwnedCTkImage
from aidas.ui.theme import (
    COLORS, COLOR_PAIRS, CONTROLS, TYPOGRAPHY, refresh_interface_widgets,
)


def _tree_action_image(action, color):
    """Draw matching toolbar glyphs from geometry for high-DPI displays."""
    scale = 4
    image = Image.new("RGBA", (16 * scale, 16 * scale))
    draw = ImageDraw.Draw(image)
    stroke = 6
    if action in ("collapse", "expand"):
        draw.rounded_rectangle(
            (3 * scale, 3 * scale, 13 * scale, 13 * scale),
            radius=6, outline=color, width=stroke,
        )
        draw.line((5 * scale, 8 * scale, 11 * scale, 8 * scale),
                  fill=color, width=stroke)
        if action == "expand":
            draw.line((8 * scale, 5 * scale, 8 * scale, 11 * scale),
                      fill=color, width=stroke)
    elif action == "refresh":
        draw.arc((3 * scale, 3 * scale, 13 * scale, 13 * scale),
                 start=45, end=315, fill=color, width=stroke)
        draw.polygon([(54, 10), (54, 26), (38, 26)], fill=color)
    else:
        raise ValueError(f"Unknown tree action: {action}")
    return image


class MeasurementDirectoryTree(ttk.Frame):
    """Draw folder connectors and important files without blocking Tk on scans."""

    def __init__(self, master, *, on_scan_state=None):
        super().__init__(master)
        self._on_scan_state = on_scan_state
        self.scan_state = "empty"
        self._generation = 0
        self._cancel_scan = threading.Event()
        self._events = queue.Queue()
        self._poll_id = self._draw_id = None
        self._inventory = None
        self._rows = []
        self._filtered_children = {}
        self._name_widths = {}
        self._detail_widths = {}
        self._measurement_signature = None
        self._rows_revision = 0
        self._last_draw_signature = None
        self._resize_signature = None
        self._closed = set()
        self._selected = None
        self._directory = self._output = None
        self._name_font = tkfont.Font(family=TYPOGRAPHY.family, size=11)
        self._detail_font = tkfont.Font(family=TYPOGRAPHY.family, size=10)
        self._row_height = max(34, self._name_font.metrics("linespace") + 14)
        self._indent = 26
        self._detail_x = 450
        self._content_width = 800
        self._empty_message = "Choose an input folder to explore its recorded measurements."
        self.path_var = tk.StringVar(master=self, value="")
        self.summary_var = tk.StringVar(master=self, value="")
        self.notice_var = tk.StringVar(master=self, value="")
        self.selection_var = tk.StringVar(master=self, value="")
        self.hide_other_files_var = tk.BooleanVar(master=self, value=True)

        self._toolbar = ttk.Frame(self)
        self._toolbar.pack(fill="x", pady=(0, 6))
        self._toolbar.columnconfigure(0, weight=1)
        self._toolbar_heading = ttk.Frame(self._toolbar)
        self._toolbar_heading.grid(row=0, column=0, sticky="w")
        ttk.Label(self._toolbar_heading, text="Folder contents", font=("", 12, "bold")).pack(side="left")
        self.hide_other_files_checkbox = ttk.Checkbutton(
            self._toolbar_heading, text="Hide other files", variable=self.hide_other_files_var,
            command=self._filter_changed,
        )
        self.hide_other_files_checkbox.pack(side="left", padx=(16, 0))
        self._toolbar_actions = ttk.Frame(self._toolbar)
        self._toolbar_actions.grid(row=0, column=1, sticky="e")
        self._toolbar_row = 0
        self._toolbar_buttons = {}
        action_font = ctk.CTkFont(
            family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size,
            weight=TYPOGRAPHY.semibold_weight,
        )
        for action, label, command in (
            ("refresh", "Refresh", self.refresh),
            ("expand", "Expand all", self.expand_all),
            ("collapse", "Collapse all", self.collapse_all),
        ):
            icon = OwnedCTkImage(
                self,
                light_image=_tree_action_image(action, COLOR_PAIRS["text"][0]),
                dark_image=_tree_action_image(action, COLOR_PAIRS["text"][1]),
                size=(16, 16),
            )
            icon._aidas_monochrome_action = True
            button = AppButton(
                self._toolbar_actions, text=label, command=command, image=icon,
                compound="left", font=action_font, height=CONTROLS.height_md,
                width=max(96, action_font.measure(label) + 16 + CONTROLS.gap + 24),
            )
            button.pack(side="right", padx=(6, 0))
            refresh_interface_widgets(button, recursive=False)
            # Keep native toolbar keyboard activation after adding CTk icons.
            focus_command = button.register(
                lambda _path, target=button: target.cget("state") != "disabled",
            )
            tk.Frame.configure(button, takefocus=focus_command)
            for child in button.winfo_children():
                child.configure(takefocus=False)
            for sequence in ("<Return>", "<space>"):
                activate = lambda _event, target=button: (target.invoke(), "break")[1]
                tk.Misc.bind(button, sequence, activate, add="+")
                button.bind(sequence, activate, add="+")
            self._toolbar_buttons[action] = button
        self._toolbar.bind("<Configure>", self._layout_toolbar)
        self._metadata = ttk.Frame(self)
        self._metadata.columnconfigure(0, weight=1)
        self.path_label = ttk.Label(self._metadata, textvariable=self.path_var, style="AIDaS.Muted.TLabel", wraplength=850)
        self.path_label.grid(row=0, column=0, sticky="ew")
        self.summary_label = ttk.Label(self._metadata, textvariable=self.summary_var)
        self.summary_label.grid(row=1, column=0, sticky="ew", pady=(4, 6))
        self.notice = ttk.Label(self, textvariable=self.notice_var, wraplength=850, foreground=COLORS.warning)

        holder = ttk.Frame(self)
        holder.pack(fill="both", expand=True)
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(holder, highlightthickness=1, borderwidth=0, height=280)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._ybar = ttk.Scrollbar(holder, orient="vertical", command=self._yview)
        self._ybar.grid(row=0, column=1, sticky="ns")
        self._xbar = ttk.Scrollbar(holder, orient="horizontal", command=self.canvas.xview)
        self._xbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=self._on_scroll, xscrollcommand=self._xbar.set)
        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda _event: self._scroll(-3))
        self.canvas.bind("<Button-5>", lambda _event: self._scroll(3))
        self.canvas.bind("<Up>", lambda _event: self._scroll(-1))
        self.canvas.bind("<Down>", lambda _event: self._scroll(1))
        self.canvas.bind("<Next>", lambda _event: self._scroll(10))
        self.canvas.bind("<Prior>", lambda _event: self._scroll(-10))
        self.selection_label = ttk.Label(self, textvariable=self.selection_var, style="AIDaS.Muted.TLabel", wraplength=850)
        self.selection_label.pack(fill="x", pady=(6, 0))
        self.path_var.trace_add("write", self._update_metadata)
        self.summary_var.trace_add("write", self._update_metadata)
        self._apply_aidas_theme()

    def _update_metadata(self, *_args):
        visible = False
        for label, variable in ((self.path_label, self.path_var),
                                (self.summary_label, self.summary_var)):
            if variable.get():
                label.grid()
                visible = True
            else:
                label.grid_remove()
        if visible:
            before = self.notice if self.notice.winfo_manager() else self.canvas.master
            self._metadata.pack(before=before, fill="x")
        else:
            self._metadata.pack_forget()

    def _layout_toolbar(self, event):
        needed = self._toolbar_heading.winfo_reqwidth() + self._toolbar_actions.winfo_reqwidth() + 12
        row = int(event.width < needed)
        if row != self._toolbar_row:
            self._toolbar_row = row
            self._toolbar_actions.grid(
                row=row, column=0 if row else 1, columnspan=2 if row else 1,
                sticky="e", pady=(6, 0) if row else 0,
            )

    def set_directory(self, directory: Path, *, output_path: Path | None = None):
        self._directory, self._output = Path(directory), output_path
        self._closed.clear()
        self._selected = None
        self.selection_var.set("")
        self.canvas.yview_moveto(0)
        self.refresh()

    def refresh(self):
        if self._directory is None:
            return
        self._cancel_scan.set()
        self._cancel_scan = threading.Event()
        self._generation += 1
        generation, cancelled = self._generation, self._cancel_scan
        directory, output, events = self._directory, self._output, self._events
        self.path_var.set(str(directory))
        self.summary_var.set("Reading folder contents…")
        # Clear the previous directory immediately so it cannot be mistaken
        # for the new selection while the worker is reading the filesystem.
        self._inventory = None
        self._rows = []
        self._filtered_children.clear()
        self._name_widths.clear()
        self._detail_widths.clear()
        self._rows_revision += 1
        self._empty_message = "Reading folder contents…"
        self._show_notice("")
        self._schedule_draw()
        self._set_scan_state("scanning")

        def worker():
            try:
                if not directory.is_dir():
                    raise FileNotFoundError(f"The selected folder is unavailable: {directory}")
                inventory = scan_measurement_directory(directory, output_path=output, cancelled=cancelled.is_set)
            except InventoryCancelled:
                return
            except Exception as exc:
                events.put((generation, None, exc))
            else:
                events.put((generation, inventory, ""))

        threading.Thread(target=worker, name="aidas-step5-directory", daemon=True).start()
        if self._poll_id is None:
            self._poll_id = self.after(60, self._poll)

    def _poll(self):
        self._poll_id = None
        finished = False
        while not self._events.empty():
            generation, inventory, error = self._events.get_nowait()
            if generation != self._generation:
                continue
            finished = True
            self._inventory = inventory
            if error:
                limited = isinstance(error, MeasurementScanLimitExceeded)
                self.summary_var.set("Folder limit exceeded." if limited else "Could not read this folder.")
                self._show_notice(str(error))
                self._empty_message = (
                    "Please select another folder with fewer subfolders."
                    if limited else "This folder could not be read. Select another folder or try Refresh."
                )
                self._set_scan_state("limited" if limited else "error", error)
                self._schedule_draw()
                continue
            notices = []
            if inventory.unassigned_workbooks:
                notices.append("Measurement results outside nasal/temporal folders are shown below, but the compiler cannot assign them to a side.")
            if inventory.errors:
                notices.append("Some folders could not be read. Check their access permissions.")
            self._show_notice(" ".join(notices))
            self._rebuild_rows()
            self._set_scan_state("ready")
        if not finished and self.scan_state == "scanning":
            self._poll_id = self.after(60, self._poll)

    def _set_scan_state(self, state, error=None):
        self.scan_state = state
        if self._on_scan_state is not None:
            self._on_scan_state(self._directory, state, error)

    def show_scan_limit(self, error):
        """Keep compiler-discovered folder growth inline with preview failures."""
        self._cancel_scan.set()
        self._generation += 1
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        self._inventory = None
        self._rows = []
        self._selected = None
        self.selection_var.set("")
        self._rows_revision += 1
        self.summary_var.set("Folder limit exceeded.")
        self._show_notice(str(error))
        self._empty_message = "Please select another folder with fewer subfolders."
        self._set_scan_state("limited", error)
        self._schedule_draw()

    def _show_notice(self, text):
        self.notice_var.set(text)
        if text:
            # Keep this explanation above the canvas, not over the folder tree.
            self.notice.pack(before=self.canvas.master, fill="x", pady=(0, 8))
        else:
            self.notice.pack_forget()

    def expand_all(self):
        self._closed.clear()
        self._rebuild_rows()

    def collapse_all(self):
        if self._inventory is not None:
            self._closed = {child.path for child in self._inventory.root.children if child.kind == "folder"}
        self._rebuild_rows()

    def _filter_changed(self):
        self._filtered_children.clear()
        self._selected = None
        self.selection_var.set("")
        self.canvas.yview_moveto(0)
        self._rebuild_rows()

    def _display_children(self, node):
        if not self.hide_other_files_var.get():
            return node.children
        if node.path != self._inventory.root.path and not node.excel_files:
            return ()
        if node.path not in self._filtered_children:
            self._filtered_children[node.path] = tuple(
                child for child in node.children if child.kind in ("folder", "measurements", "excel", "output")
            )
        return self._filtered_children[node.path]

    def _rebuild_rows(self):
        self._rows = []
        self._rows_revision += 1
        if self._inventory is not None:
            inventory = self._inventory
            counts = [(inventory.workbooks, "recorded Step 4 result")]
            outputs = inventory.root.excel_files - inventory.workbooks
            if outputs:
                counts.append((outputs, "compiled workbook"))
            if not self.hide_other_files_var.get():
                counts.extend(((inventory.thickness_files, "thickness table"), (inventory.fovea_files, "fovea file")))
            self.summary_var.set("   ·   ".join(f"{count} {label}{'' if count == 1 else 's'}" for count, label in counts))
            pending = [(self._inventory.root, 0, (), True)]
            while pending:
                node, depth, ancestors, last = pending.pop()
                self._rows.append((node, depth, ancestors, last))
                if node.kind == "folder" and node.path not in self._closed:
                    children = self._display_children(node)
                    for index in range(len(children) - 1, -1, -1):
                        pending.append((children[index], depth + 1, ancestors + (not last,), index == len(children) - 1))
        # A folder toggle changes the visible rows, not their text. Keep the
        # Tcl font measurements for this inventory, with explicit font and
        # display-scaling invalidation so horizontal layout stays exact.
        signature = (
            tuple(self._name_font.actual().items()),
            tuple(self._detail_font.actual().items()),
            self._name_font.metrics("linespace"),
            self._detail_font.metrics("linespace"),
            float(self.tk.call("tk", "scaling")),
        )
        if signature != self._measurement_signature:
            self._name_widths.clear()
            self._detail_widths.clear()
            self._measurement_signature = signature
        for node, _, _, _ in self._rows:
            if node.name not in self._name_widths:
                self._name_widths[node.name] = self._name_font.measure(node.name)
            if node.detail not in self._detail_widths:
                self._detail_widths[node.detail] = self._detail_font.measure(node.detail)
        self._detail_x = max((70 + depth * self._indent + self._name_widths[node.name] + 40 for node, depth, _, _ in self._rows), default=450)
        self._content_width = self._detail_x + max((self._detail_widths[node.detail] for node, _, _, _ in self._rows), default=280) + 24
        self._resize()

    def _resize(self, _event=None):
        canvas_width = self.canvas.winfo_width()
        # Scroll regions describe drawable pixels, excluding the canvas's
        # border. Using the outer width creates a tiny, scrollable overflow
        # even when every row fits inside the viewport.
        inset = sum(self.canvas.winfo_pixels(self.canvas.cget(option))
                    for option in ("borderwidth", "highlightthickness"))
        viewport_width = max(1, canvas_width - 2 * inset)
        viewport_height = max(1, self.canvas.winfo_height() - 2 * inset)
        content_width = self._content_width if self._rows else 0
        content_height = len(self._rows) * self._row_height + 24 if self._rows else 0
        wrap = max(240, canvas_width - 20)
        region = (0, 0, max(viewport_width, content_width),
                  max(viewport_height, content_height))
        signature = (wrap, region)
        if signature != self._resize_signature:
            for label in (self.path_label, self.notice, self.selection_label):
                label.configure(wraplength=wrap)
            self.canvas.configure(scrollregion=region)
            self._resize_signature = signature
        # Reset either axis when filtering, collapsing, or resizing leaves
        # all its content visible. ttk disables a full-range scrollbar.
        if content_width <= viewport_width:
            self.canvas.xview_moveto(0)
        if content_height <= viewport_height:
            self.canvas.yview_moveto(0)
        self._schedule_draw()

    def _schedule_draw(self):
        if self._draw_id is None:
            self._draw_id = self.after_idle(self._draw)

    def _on_scroll(self, first, last):
        self._ybar.set(first, last)
        self._schedule_draw()

    def _yview(self, *args):
        if self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview(*args)
            self._schedule_draw()

    def _scroll(self, units):
        if self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview_scroll(units, "units")
            self._schedule_draw()
        return "break"

    def _wheel(self, event):
        return self._scroll(-int(event.delta / 120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1))

    def _click(self, event):
        self.canvas.focus_set()
        index = int((self.canvas.canvasy(event.y) - 12) // self._row_height)
        if not 0 <= index < len(self._rows):
            return
        node = self._rows[index][0]
        self._selected = (node.path, node.kind)
        self.selection_var.set(str(node.path) if node.kind != "other" else "")
        if node.kind == "folder" and self._display_children(node):
            if node.path in self._closed:
                self._closed.remove(node.path)
            else:
                self._closed.add(node.path)
            self._rebuild_rows()
        else:
            self._schedule_draw()

    def _draw(self):
        self._draw_id = None
        canvas = self.canvas
        height = self._row_height
        first = max(0, int(canvas.canvasy(0) // height) - 1)
        last_visible = min(len(self._rows), first + int(canvas.winfo_height() // height) + 4)
        width = max(canvas.winfo_width(), self._content_width if self._rows else 0)
        signature = (self._rows_revision, first, last_visible, width,
                     self._selected, self._empty_message)
        if signature == self._last_draw_signature:
            return
        self._last_draw_signature = signature
        canvas.delete("all")
        if not self._rows:
            canvas.create_text(24, 28, anchor="nw", text=self._empty_message,
                               width=max(1, canvas.winfo_width() - 48),
                               fill=COLORS.muted_text, font=self._name_font)
            return
        for index in range(first, last_visible):
            node, depth, ancestors, last = self._rows[index]
            top = 12 + index * height
            mid = top + height / 2
            x = 24 + depth * self._indent
            if self._selected == (node.path, node.kind):
                canvas.create_rectangle(6, top, width - 6, top + height, fill=COLORS.selection, outline="")
            for level in range(1, depth):
                if ancestors[level]:
                    line_x = 24 + level * self._indent - 13
                    canvas.create_line(line_x, top, line_x, top + height, fill=COLORS.border_strong)
            if depth:
                line_x = x - 13
                canvas.create_line(line_x, top, line_x, mid if last else top + height, fill=COLORS.border_strong)
                canvas.create_line(line_x, mid, x - 4, mid, fill=COLORS.border_strong)
            color = COLORS.warning if node.kind == "folder" else COLORS.success if node.kind in ("measurements", "excel", "output") else COLORS.primary
            if node.kind == "other":
                color = COLORS.muted_text
                canvas.create_text(x + 6, mid, text="·", fill=color, font=self._name_font)
            elif node.kind == "folder":
                canvas.create_polygon(x, mid - 6, x + 7, mid - 6, x + 10, mid - 3, x + 18, mid - 3, x + 18, mid + 8, x, mid + 8, fill=color, outline="")
                if self._display_children(node):
                    canvas.create_text(x - 12, mid, text="▸" if node.path in self._closed else "▾", fill=COLORS.muted_text, font=self._detail_font)
            else:
                self._draw_file_icon(x, mid, node.kind, color, index)
            canvas.create_text(x + 28, mid, anchor="w", text=node.name, fill=COLORS.muted_text if node.kind == "other" else COLORS.text, font=self._name_font)
            canvas.create_text(self._detail_x, mid, anchor="w", text=node.detail, fill=COLORS.muted_text, font=self._detail_font)

    def _draw_file_icon(self, x, mid, kind, color, index):
        # Draw letters as strokes around the exact center of the file box.
        # Font ascent, descent, and system DPI cannot shift these glyphs.
        tag = f"file-icon-{index}"
        self.canvas.create_rectangle(x + 1, mid - 9, x + 17, mid + 9, outline=color, width=1.5, tags=(tag, "file-icon-box"))
        cx = x + 9
        segments = []
        if kind == "measurements":
            segments = [(cx - 3, mid + 4, cx - 3, mid - 4),
                        (cx - 3, mid - 4, cx, mid),
                        (cx, mid, cx + 3, mid - 4),
                        (cx + 3, mid - 4, cx + 3, mid + 4)]
        elif kind in ("excel", "output"):
            segments = [(cx - 3, mid - 4, cx + 3, mid + 4), (cx - 3, mid + 4, cx + 3, mid - 4)]
        else:
            segments = [(cx - 3, mid - 4, cx + 3, mid - 4), (cx, mid - 4, cx, mid + 4)]
            if kind not in ("thickness", "fovea"):
                segments.append((cx - 3, mid + 4, cx + 3, mid + 4))
        for segment in segments:
            self.canvas.create_line(*segment, fill=color, width=1.25, capstyle="round", tags=(tag, "file-icon-glyph"))

    def _apply_aidas_theme(self):
        self.canvas.configure(background=COLORS.surface, highlightbackground=COLORS.border, yscrollincrement=self._row_height)
        self.notice.configure(foreground=COLORS.warning)
        self._last_draw_signature = None
        self._measurement_signature = None
        self._schedule_draw()

    def destroy(self):
        self._cancel_scan.set()
        self._generation += 1
        for callback in (self._poll_id, self._draw_id):
            if callback is not None:
                try:
                    self.after_cancel(callback)
                except tk.TclError:
                    pass
        super().destroy()
        self._toolbar_buttons.clear()
        # Release Tcl-owned resources on the UI thread. Otherwise Python's
        # cyclic GC can finalize an old view during a later background scan.
        for name in ("path_var", "summary_var", "notice_var", "selection_var", "hide_other_files_var", "_name_font", "_detail_font"):
            setattr(self, name, None)
