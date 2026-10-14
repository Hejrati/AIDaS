"""An embedded warning for the bright-pixel decision on the image canvas."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

from aidas.ui.components import AppButton
from aidas.ui.theme import COLOR_PAIRS, CONTROLS, TYPOGRAPHY, resolve_color


class BrightPixelNotice(tk.Frame):
    """Present an explicit choice without creating or grabbing a window.

    The owner places this frame over its image canvas and handles the decision.
    Destruction also resolves the choice, so closing the containing view cannot
    leave a caller waiting for a notice that is no longer visible.
    """

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
            highlightthickness=1,
            padx=12,
            pady=10,
        )
        self.columnconfigure(1, weight=1)

        self._icon = tk.Label(
            self,
            text="\u26a0",
            font=(TYPOGRAPHY.family, -24),
            anchor="n",
            padx=0,
            pady=0,
        )
        self._icon.grid(row=0, column=0, rowspan=3, sticky="nw", padx=(0, 10))
        self._title = tk.Label(
            self,
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
            self,
            text=(
                f"{count:,} unusually bright {pixels} can darken this image. "
                f"Smooth only {'that pixel' if count == 1 else 'those pixels'}?"
            ),
            font=(TYPOGRAPHY.family, -TYPOGRAPHY.body_size),
            anchor="w",
            justify="left",
            padx=0,
            pady=0,
            wraplength=480,
        )
        self._body.grid(row=1, column=1, sticky="ew", pady=(4, 0))
        self._reassurance = tk.Label(
            self,
            text="All other pixel values and the source SDB stay unchanged.",
            font=(TYPOGRAPHY.family, -TYPOGRAPHY.body_size),
            anchor="w",
            justify="left",
            padx=0,
            pady=0,
            wraplength=480,
        )
        self._reassurance.grid(row=2, column=1, sticky="ew", pady=(4, 0))

        self._actions = tk.Frame(self, borderwidth=0)
        self._actions.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
        options = {
            "bg_color": COLOR_PAIRS["warning_soft"],
            "height": CONTROLS.height_sm,
        }
        self.fix_button = AppButton(
            self._actions,
            text="Fix",
            variant="success",
            width=72,
            command=lambda: self._finish(True),
            **options,
        )
        self.cancel_button = AppButton(
            self._actions,
            text="Cancel",
            variant="danger",
            width=72,
            command=lambda: self._finish(None),
            **options,
        )
        self.fix_button.grid(row=0, column=0, padx=(0, 8))
        self.cancel_button.grid(row=0, column=1)
        for button, decision in (
            (self.fix_button, True),
            (self.cancel_button, None),
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
            self._reassurance,
            self._actions,
            self.fix_button,
            self.cancel_button,
        ):
            widget.bind("<Escape>", self._cancel_from_keyboard, add="+")
        self.bind("<Configure>", self._resize_text, add="+")
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._apply_aidas_theme()

    def _resize_text(self, event) -> None:
        if event.widget is not self:
            return
        # Leave room for the warning glyph, inset, and warning border.
        wraplength = max(100, event.width - 64)
        self._body.configure(wraplength=wraplength)
        self._reassurance.configure(wraplength=wraplength)

    def _apply_aidas_theme(self) -> None:
        background = resolve_color(COLOR_PAIRS["warning_soft"])
        warning = resolve_color(COLOR_PAIRS["warning"])
        text = resolve_color(COLOR_PAIRS["text"])
        self.configure(
            background=background,
            highlightbackground=warning,
            highlightcolor=warning,
        )
        self._actions.configure(background=background)
        for label in (self._icon, self._title, self._body, self._reassurance):
            label.configure(background=background, foreground=text)
        self._icon.configure(foreground=warning)
        self._title.configure(foreground=warning)
        for button in (self.fix_button, self.cancel_button):
            button.configure(bg_color=COLOR_PAIRS["warning_soft"])

    def focus_default(self) -> None:
        """Focus Cancel so fixing pixels requires an explicit choice."""

        self.cancel_button.focus_set()

    def _cancel_from_keyboard(self, _event):
        self._finish(None)
        return "break"

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
        super().destroy()
