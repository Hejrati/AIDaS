"""A compact, theme-aware instruction banner for manual annotation."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk

from aidas.ui.components import AppButton
from aidas.ui.theme import COLOR_PAIRS, COLORS, TYPOGRAPHY


class AnnotationGuide(ctk.CTkFrame):
    """Readable, nonmodal help placed along the image viewport's top edge."""

    preferred_width = 620
    single_column_below = 520
    four_columns_from = 940
    instructions = (
        ("Draw", "Click points. Double-click to finish.", "Click"),
        ("Edit", "Select a completed line. Drag to edit.", "Drag"),
        ("Add point", "Shift-click a handle, then drag.", "Shift"),
        ("Remove point", "Alt-click a handle (at least 3 remain).", "Alt"),
    )

    def __init__(self, master, *, dismiss: Callable, **kwargs):
        super().__init__(
            master,
            height=1,
            corner_radius=0,
            border_width=1,
            border_color=COLOR_PAIRS["primary"],
            fg_color=COLOR_PAIRS["surface"],
            **kwargs,
        )
        self._dismiss_callback = dismiss
        self._layout_signature = None
        self._illustrations = {}
        self._illustration_signature = None
        self._tiles = {}
        self._heading_rows = {}
        self._shortcut_labels = {}
        self.labels = {}
        self.heading_labels = {}
        self.grid_columnconfigure(0, weight=1)

        self.header = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 8))
        self.header.grid_columnconfigure(1, weight=1)
        self.info_icon = tk.Canvas(self.header, highlightthickness=0, takefocus=False)
        self.info_icon.grid(row=0, column=0, padx=(0, 8))
        self.info_icon._apply_aidas_theme = self.redraw_illustrations
        self.title = ctk.CTkLabel(
            self.header,
            text="Manual annotation",
            anchor="w",
            width=1,
            height=22,
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.subtitle_size,
                weight="bold",
            ),
        )
        self.title.grid(row=0, column=1, sticky="ew")
        self.title.bind("<Configure>", lambda event: self._wrap_body(self.title, event.width), add="+")

        self.instruction_grid = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.instruction_grid.grid(row=1, column=0, sticky="ew", padx=12)
        for action, instruction, gesture in self.instructions:
            tile = ctk.CTkFrame(self.instruction_grid, fg_color="transparent", corner_radius=0)
            tile.grid_columnconfigure(0, weight=1)
            heading_row = ctk.CTkFrame(tile, fg_color="transparent", corner_radius=0)
            heading_row.grid(row=0, column=0, sticky="ew", pady=(0, 2))
            heading_row.grid_columnconfigure(0, weight=1)
            heading = ctk.CTkLabel(
                heading_row,
                text=action,
                height=20,
                anchor="w",
                width=1,
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.body_size,
                    weight="bold",
                ),
            )
            heading.grid(row=0, column=0, sticky="ew", padx=(0, 6))
            shortcut = ctk.CTkLabel(
                heading_row,
                text=gesture,
                height=18,
                width=38,
                corner_radius=3,
                fg_color=COLOR_PAIRS["surface_subtle"],
                text_color=COLOR_PAIRS["muted_text"],
                font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.caption_size),
            )
            shortcut.grid(row=0, column=1, sticky="e")
            label = ctk.CTkLabel(
                tile,
                text=instruction,
                width=1,
                height=1,
                wraplength=250,
                justify="left",
                anchor="nw",
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size),
            )
            label.grid(row=1, column=0, sticky="new")
            label.bind("<Configure>", lambda event, body=label: self._wrap_body(body, event.width), add="+")
            self._tiles[action] = tile
            self._heading_rows[action] = heading_row
            self._shortcut_labels[action] = shortcut
            self.labels[action] = label
            self.heading_labels[action] = heading

        self.divider = ctk.CTkFrame(self, height=1, fg_color=COLOR_PAIRS["border"], corner_radius=0)
        self.divider.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 0))
        self.actions = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.actions.grid(row=3, column=0, sticky="w", padx=12, pady=10)
        self.opt_out_button = AppButton(
            self.actions,
            text="Do not show",
            variant="primary",
            width=112,
            height=30,
            command=lambda: self._dismiss(do_not_show_again=True),
        )
        self.opt_out_button.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.ok_button = AppButton(
            self.actions,
            text="OK",
            variant="success",
            width=72,
            height=30,
            command=self._dismiss,
        )
        self.ok_button.grid(row=0, column=1, sticky="w")
        self.bind("<Escape>", self._dismiss_from_keyboard, add="+")
        self.bind("<Return>", self._dismiss_from_keyboard, add="+")
        for button in (self.opt_out_button, self.ok_button):
            button.bind("<Escape>", self._dismiss_from_keyboard, add="+")
            for event in ("<Return>", "<space>"):
                button.bind(event, lambda _event, action=button.invoke: self._invoke_from_keyboard(action), add="+")
        self.configure_layout(self.preferred_width)

    def _dismiss(self, *, do_not_show_again=False):
        if do_not_show_again:
            self._dismiss_callback(do_not_show_again=True)
        else:
            self._dismiss_callback()

    def _dismiss_from_keyboard(self, _event):
        self._dismiss()
        return "break"

    @staticmethod
    def _invoke_from_keyboard(action):
        action()
        return "break"

    def focus_default(self):
        """Acknowledge the guide without changing the user's preference."""
        self.ok_button.focus_set()

    def _apply_aidas_theme(self):
        """Refresh live tokens even when the effective appearance is unchanged."""
        self.configure(fg_color=COLOR_PAIRS["surface"], border_color=COLOR_PAIRS["primary"])
        for label in (self.title, *self.heading_labels.values(), *self.labels.values()):
            label.configure(text_color=COLOR_PAIRS["text"])
        for shortcut in self._shortcut_labels.values():
            shortcut.configure(
                fg_color=COLOR_PAIRS["surface_subtle"],
                text_color=COLOR_PAIRS["muted_text"],
            )
        self.divider.configure(fg_color=COLOR_PAIRS["border"])
        self.opt_out_button.set_variant("primary")
        self.ok_button.set_variant("success")
        self.redraw_illustrations()

    def _wrap_body(self, label, physical_width):
        width = max(1, round(physical_width / self._get_widget_scaling()))
        if label.cget("wraplength") != width:
            label.configure(wraplength=width)

    def configure_layout(self, width):
        """Adapt instruction columns to the viewport's logical width."""
        columns = 4 if width >= self.four_columns_from else 2 if width >= self.single_column_below else 1
        if self._layout_signature == columns:
            return
        self._layout_signature = columns
        for column in range(4):
            self.instruction_grid.grid_columnconfigure(
                column,
                weight=int(column < columns),
                uniform="guide" if column < columns else "",
            )
        rows = len(self.instructions) // columns
        for index, tile in enumerate(self._tiles.values()):
            row, column = divmod(index, columns)
            tile.grid(
                row=row,
                column=column,
                sticky="new",
                padx=(0, 16) if column < columns - 1 else 0,
                pady=(0, 8) if row < rows - 1 else 0,
            )
        self.redraw_illustrations()

    def redraw_illustrations(self):
        """Render the standard information glyph in the current palette."""
        scale = self._get_widget_scaling()
        signature = (scale, COLORS.surface, COLORS.primary_soft, COLORS.primary, TYPOGRAPHY.family)
        if signature == self._illustration_signature:
            return
        size = round(22 * scale)
        self.info_icon.configure(width=size, height=size, background=COLORS.surface)
        if not self._illustrations:
            self._illustrations["circle"] = self.info_icon.create_oval(0, 0, 0, 0)
            self._illustrations["text"] = self.info_icon.create_text(0, 0, text="i")
        self.info_icon.coords(
            self._illustrations["circle"], *(value * scale for value in (1, 1, 21, 21)),
        )
        self.info_icon.itemconfigure(
            self._illustrations["circle"],
            fill=COLORS.primary_soft,
            outline=COLORS.primary,
            width=scale,
        )
        self.info_icon.coords(self._illustrations["text"], 11 * scale, 11 * scale)
        self.info_icon.itemconfigure(
            self._illustrations["text"],
            fill=COLORS.primary,
            font=(TYPOGRAPHY.family, -round(14 * scale), "bold"),
        )
        self._illustration_signature = signature
