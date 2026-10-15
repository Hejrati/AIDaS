"""Shared surface for application menus and hover help."""

from __future__ import annotations

import customtkinter as ctk

from aidas.ui.theme import COLOR_PAIRS, get_interface_mode


class PopupPanel(ctk.CTkFrame):
    """Themed popup with a continuous, one-physical-pixel outline."""

    def __init__(self, master, **kwargs) -> None:
        kwargs.setdefault("corner_radius", 0 if get_interface_mode() == "Classic" else 4)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", COLOR_PAIRS["menu_border"])
        kwargs.setdefault("bg_color", COLOR_PAIRS["menu_surface"])
        kwargs.setdefault("fg_color", COLOR_PAIRS["menu_surface"])
        super().__init__(master, **kwargs)

    def _draw(self, no_color_updates=False) -> None:
        # Both sides of the border share one path. Keep the stroke aligned to
        # physical pixels even when the content uses fractional DPI scaling.
        self._border_width = 1 / self._get_widget_scaling()
        super()._draw(no_color_updates)
