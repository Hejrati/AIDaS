"""A modal popup requiring a bright-pixel decision before continuing."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import customtkinter as ctk

from aidas.ui.components import AppButton
from aidas.ui.theme import COLOR_PAIRS, CONTROLS, TYPOGRAPHY, resolve_color
from aidas.utils.ui_utils import HoverToolTip


class BrightPixelNotice(tk.Toplevel):
    """Block application interaction until the user chooses Fix or Ignore."""

    def __init__(
        self,
        master,
        *,
        count: int,
        context: str,
        on_decision: Callable[[bool | None], None],
    ) -> None:
        self._on_decision = on_decision
        self._resolved = False
        self.context = context
        super().__init__(
            master,
            borderwidth=0,
            highlightthickness=0,
        )
        self.withdraw()
        self.title("Bright pixels detected")
        self.transient(master.winfo_toplevel())
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._ignore_without_remembering)
        self._previous_grab = None
        self._content = tk.Frame(self, borderwidth=0, padx=24, pady=24)
        self._content.pack(fill="both", expand=True)
        self._content.columnconfigure(1, weight=1)
        self._icon = tk.Label(
            self._content,
            text="\u26a0",
            font=(TYPOGRAPHY.family, -32),
            anchor="n",
            padx=0,
            pady=0,
        )
        self._icon.grid(row=0, column=0, rowspan=3, sticky="nw", padx=(0, 16))
        self._title = tk.Label(
            self._content,
            text="Bright pixels detected",
            font=(TYPOGRAPHY.family, -TYPOGRAPHY.subtitle_size, TYPOGRAPHY.bold_weight),
            anchor="w",
            justify="left",
            padx=0,
            pady=0,
        )
        self._title.grid(row=0, column=1, sticky="ew")
        pixels = "pixel" if count == 1 else "pixels"
        self._body = tk.Label(
            self._content,
            text=f"Detected {count:,} unusually bright {pixels} that may reduce image contrast.",
            font=(TYPOGRAPHY.family, -TYPOGRAPHY.body_size),
            anchor="w",
            justify="left",
            padx=0,
            pady=0,
            wraplength=480,
        )
        self._body.grid(row=1, column=1, sticky="ew", pady=(8, 0))

        self._divider = tk.Frame(self._content, height=1, borderwidth=0)
        self._divider.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(20, 0))
        self._footer = tk.Frame(self._content, borderwidth=0)
        self._footer.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        self._footer.columnconfigure(0, weight=1)
        self.auto_fix_var = tk.BooleanVar(master=self, value=False)
        self.auto_fix_checkbox = ctk.CTkCheckBox(
            self._footer,
            text="Remember my choice",
            variable=self.auto_fix_var,
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size),
            checkbox_width=18,
            checkbox_height=18,
            corner_radius=4,
            border_width=2,
            bg_color=COLOR_PAIRS["surface_elevated"],
            fg_color=COLOR_PAIRS["primary"],
            hover_color=COLOR_PAIRS["primary_hover"],
            border_color=COLOR_PAIRS["border_strong"],
            text_color=COLOR_PAIRS["text"],
            checkmark_color=COLOR_PAIRS["on_primary"],
        )
        self.auto_fix_checkbox.grid(row=0, column=0, sticky="w", padx=(0, 16))
        self._remember_tooltip = HoverToolTip(
            self.auto_fix_checkbox,
            "Remember Fix or Ignore for future images. "
            "Restore confirmation in Settings > General.",
        )
        self._actions = tk.Frame(self._footer, borderwidth=0)
        self._actions.grid(row=0, column=1, sticky="e")
        options = {
            "bg_color": COLOR_PAIRS["surface_elevated"],
            "height": CONTROLS.height_md,
        }
        self.fix_button = AppButton(
            self._actions,
            text="Fix",
            variant="primary",
            width=88,
            command=lambda: self._finish(True),
            **options,
        )
        self.ignore_button = AppButton(
            self._actions,
            text="Ignore",
            variant="secondary",
            width=88,
            command=lambda: self._finish(False),
            **options,
        )
        self.fix_button.grid(row=0, column=0, padx=(0, 8))
        self.ignore_button.grid(row=0, column=1)
        for button, decision in (
            (self.fix_button, True),
            (self.ignore_button, False),
        ):
            def choose(_event, value=decision):
                self._finish(value)
                return "break"

            button.bind("<Return>", choose, add="+")
            button.bind("<space>", choose, add="+")

        for widget in (
            self,
            self._icon,
            self._title,
            self._body,
            self._actions,
            self.fix_button,
            self.ignore_button,
        ):
            widget.bind("<Escape>", self._cancel_from_keyboard, add="+")
        self.bind("<Configure>", self._resize_text, add="+")
        self.bind("<Destroy>", self._on_destroy, add="+")
        from aidas.ui.title_bar import install_dialog_title_bar

        install_dialog_title_bar(self)
        self._apply_aidas_theme()

    def _resize_text(self, event) -> None:
        if event.widget is not self:
            return
        self._set_text_width(event.width)

    def _set_text_width(self, width) -> None:
        # Leave room for the warning glyph, inset, and warning border.
        wraplength = max(100, width - 108)
        self._body.configure(wraplength=wraplength)
        self.auto_fix_checkbox._text_label.configure(
            wraplength=max(70, width - 280), justify="left", anchor="w",
        )

    def _apply_aidas_theme(self) -> None:
        background = resolve_color(COLOR_PAIRS["surface_elevated"])
        border = resolve_color(COLOR_PAIRS["border"])
        text = resolve_color(COLOR_PAIRS["text"])
        self.configure(
            background=background,
            highlightbackground=border,
            highlightcolor=border,
        )
        self._content.configure(background=background)
        self._footer.configure(background=background)
        self._actions.configure(background=background)
        for label in (self._icon, self._title, self._body):
            label.configure(background=background, foreground=text)
        self.auto_fix_checkbox.configure(
            bg_color=COLOR_PAIRS["surface_elevated"],
            fg_color=COLOR_PAIRS["primary"],
            hover_color=COLOR_PAIRS["primary_hover"],
            border_color=COLOR_PAIRS["border_strong"],
            text_color=COLOR_PAIRS["text"],
            checkmark_color=COLOR_PAIRS["on_primary"],
        )
        self._icon.configure(foreground=resolve_color(COLOR_PAIRS["danger"]))
        self._divider.configure(background=border)
        for button in (self.fix_button, self.ignore_button):
            button.configure(bg_color=COLOR_PAIRS["surface_elevated"])

    def show_modal(self) -> None:
        """Center over the application and route input exclusively to this popup."""
        owner = self.master.winfo_toplevel()
        width = min(560, max(420, owner.winfo_width() - 40), self.winfo_screenwidth())
        self._set_text_width(width)
        self.update_idletasks()
        height = self.winfo_reqheight()
        x = max(0, min(owner.winfo_rootx() + (owner.winfo_width() - width) // 2,
                       self.winfo_screenwidth() - width))
        y = max(0, min(owner.winfo_rooty() + (owner.winfo_height() - height) // 2,
                       self.winfo_screenheight() - height))
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.deiconify()
        self.wait_visibility()
        self.lift()
        self._previous_grab = self.grab_current()
        self.grab_set()
        self.focus_force()
        self.focus_default()

    def focus_default(self) -> None:
        """Focus Ignore so fixing pixels requires an explicit choice."""

        self.ignore_button.focus_set()

    def _cancel_from_keyboard(self, _event):
        self._finish(None)
        return "break"

    def _ignore_without_remembering(self) -> None:
        """Closing the popup keeps original pixels for this image only."""
        if self._resolved:
            return
        self.auto_fix_var.set(False)
        self._finish(False)

    def _finish(self, decision: bool | None) -> None:
        """Resolve once, including when the owner or an action destroys us."""

        if self._resolved:
            return
        self._resolved = True
        self._on_decision(decision)

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            self._finish(None)

    def destroy(self) -> None:
        self._finish(None)
        try:
            if self.grab_current() is self:
                self.grab_release()
                previous = self._previous_grab
                if previous is not None and previous.winfo_exists():
                    previous.grab_set()
        except tk.TclError:
            pass
        super().destroy()
