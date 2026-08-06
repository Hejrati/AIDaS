"""CustomTkinter application menu bar for the AIDaS window chrome.

The standard Tk menu uses operating-system colors that cannot reliably follow
CustomTkinter's light/dark appearance.  This component keeps the familiar
File/View/Help layout while rendering both the menu row and its popups with the
shared semantic AIDaS palette.
"""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from typing import Callable, Literal, Sequence

import customtkinter as ctk

from aidas.core.display import work_area_bounds
from aidas.ui.theme import COLORS, COLOR_PAIRS, CONTROLS, SHAPES, TYPOGRAPHY


MenuItemKind = Literal["command", "heading", "separator"]


@dataclass(frozen=True)
class _MenuItem:
    """Description of one popup row."""

    kind: MenuItemKind
    label: str = ""
    command: Callable[[], None] | None = None
    checked: bool = False


class _PopupCommandRow(ctk.CTkFrame):
    """One full-width popup command with padded content columns."""

    def __init__(
        self,
        master,
        *,
        width: int,
        height: int,
        label: str,
        checked: bool,
        check_column_width: int,
        on_enter: Callable[[], None],
        on_invoke: Callable[[], None],
    ) -> None:
        super().__init__(
            master,
            width=width,
            height=height,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=0,
            fg_color=COLOR_PAIRS["surface_elevated"],
            cursor="hand2",
        )
        self.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self._on_enter = on_enter
        self._on_invoke = on_invoke
        self._pressed = False

        self.check_label = ctk.CTkLabel(
            self,
            text="\u2713" if checked else "",
            # Leave the parent's rounded left corners visible while keeping
            # the complete check/text alignment column at a fixed width.
            width=max(1, check_column_width - SHAPES.corner_radius_sm),
            height=height,
            anchor="center",
            fg_color=COLOR_PAIRS["surface_elevated"],
            text_color=COLOR_PAIRS["text"],
            cursor="hand2",
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        )
        self.check_label.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(SHAPES.corner_radius_sm, 0),
        )
        self.text_label = ctk.CTkLabel(
            self,
            text=label,
            height=height,
            anchor="w",
            fg_color=COLOR_PAIRS["surface_elevated"],
            text_color=COLOR_PAIRS["text"],
            cursor="hand2",
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
            ),
        )
        self.text_label.grid(row=0, column=1, sticky="nsew", padx=(0, 12))

        for widget in (self, self.check_label, self.text_label):
            widget.bind("<Enter>", self._handle_enter, add="+")
            widget.bind("<ButtonPress-1>", self._handle_press, add="+")
            widget.bind("<ButtonRelease-1>", self._handle_invoke, add="+")

    def _handle_enter(self, _event=None) -> None:
        self._on_enter()

    def _handle_press(self, _event=None) -> str:
        self._pressed = True
        self._on_enter()
        return "break"

    def _handle_invoke(self, event=None) -> str:
        pressed = self._pressed
        self._pressed = False
        if pressed and self._event_is_inside(event):
            self._on_invoke()
        return "break"

    def _event_is_inside(self, event) -> bool:
        if event is None:
            return False
        try:
            x = int(event.x_root)
            y = int(event.y_root)
            left = self.winfo_rootx()
            top = self.winfo_rooty()
            return (
                left <= x < left + self.winfo_width()
                and top <= y < top + self.winfo_height()
            )
        except (AttributeError, tk.TclError, TypeError, ValueError):
            return False

    def set_selected(self, selected: bool) -> None:
        """Apply one background across the complete row and both columns."""

        color = (
            COLOR_PAIRS["primary_soft"]
            if selected
            else COLOR_PAIRS["surface_elevated"]
        )
        self.configure(fg_color=color)
        self.check_label.configure(fg_color=color)
        self.text_label.configure(fg_color=color)


class _PopupMenu(tk.Toplevel):
    """Borderless CTk popup with mouse and keyboard navigation."""

    _MENU_WIDTH = 252
    _ITEM_MARGIN_X = 4
    _ITEM_GAP_Y = 0
    _SEPARATOR_MARGIN_X = 8
    _SEPARATOR_GAP_Y = 2
    _EDGE_PADDING_Y = 4
    _ITEM_HEIGHT = CONTROLS.height_sm
    _CHECK_COLUMN_WIDTH = 26
    _HEADING_HEIGHT = 22

    def __init__(
        self,
        master,
        *,
        items: Sequence[_MenuItem],
        on_close: Callable[["_PopupMenu"], None],
        on_cycle: Callable[[int], None],
    ) -> None:
        super().__init__(master)
        self.withdraw()
        self.overrideredirect(True)
        self.transient(master.winfo_toplevel())
        self.configure(background=COLORS.border_strong)

        self._items = tuple(items)
        self._menu_bar = master
        self._on_close = on_close
        self._on_cycle = on_cycle
        self._closed = False
        self._visible = False
        self._selected = -1
        self._buttons: dict[int, _PopupCommandRow] = {}
        self._command_indices: list[int] = []
        self._separators: list[tk.Frame] = []
        self._outside_binding: str | None = None

        panel = ctk.CTkFrame(
            self,
            width=self._MENU_WIDTH,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=0,
            fg_color=COLOR_PAIRS["surface_elevated"],
        )
        panel.pack(fill="both", expand=True, padx=1, pady=1)

        for index, item in enumerate(self._items):
            if item.kind == "separator":
                separator = tk.Frame(
                    panel,
                    height=1,
                    borderwidth=0,
                    highlightthickness=0,
                    background=COLORS.border_strong,
                )
                separator.pack_propagate(False)
                separator.pack(
                    fill="x",
                    padx=self._SEPARATOR_MARGIN_X,
                    pady=self._SEPARATOR_GAP_Y,
                )
                self._separators.append(separator)
                continue

            if item.kind == "heading":
                ctk.CTkLabel(
                    panel,
                    text=item.label.upper(),
                    height=self._HEADING_HEIGHT,
                    anchor="w",
                    text_color=COLOR_PAIRS["muted_text"],
                    font=ctk.CTkFont(
                        family=TYPOGRAPHY.family,
                        size=TYPOGRAPHY.caption_size,
                        weight=TYPOGRAPHY.semibold_weight,
                    ),
                ).pack(
                    fill="x",
                    padx=(
                        self._ITEM_MARGIN_X + self._CHECK_COLUMN_WIDTH,
                        self._ITEM_MARGIN_X,
                    ),
                    pady=(self._EDGE_PADDING_Y if index == 0 else 0, 0),
                )
                continue

            button = _PopupCommandRow(
                panel,
                width=self._MENU_WIDTH - (2 * self._ITEM_MARGIN_X),
                height=self._ITEM_HEIGHT,
                label=item.label,
                checked=item.checked,
                check_column_width=self._CHECK_COLUMN_WIDTH,
                on_enter=lambda row=index: self._select(row),
                on_invoke=lambda row=index: self._invoke(row),
            )
            button.pack(
                fill="x",
                padx=self._ITEM_MARGIN_X,
                pady=(
                    self._EDGE_PADDING_Y if index == 0 else self._ITEM_GAP_Y,
                    self._EDGE_PADDING_Y
                    if index == len(self._items) - 1
                    else self._ITEM_GAP_Y,
                ),
            )
            self._buttons[index] = button
            self._command_indices.append(index)

        # Let geometry negotiation size the popup to its rows, but retain a
        # stable width so switching File/View/Help does not make the bar jump.
        panel.update_idletasks()
        requested_width = max(1, panel.winfo_reqwidth())
        requested_height = max(1, panel.winfo_reqheight())
        self.geometry(f"{requested_width + 2}x{requested_height + 2}")

        self.bind("<Escape>", self._close_from_event, add="+")
        self.bind("<Up>", lambda _event: self._move_selection(-1), add="+")
        self.bind("<Down>", lambda _event: self._move_selection(1), add="+")
        self.bind("<Home>", lambda _event: self._select_edge(first=True), add="+")
        self.bind("<End>", lambda _event: self._select_edge(first=False), add="+")
        self.bind("<Return>", self._invoke_selected, add="+")
        self.bind("<KP_Enter>", self._invoke_selected, add="+")
        self.bind("<Left>", lambda _event: self._cycle(-1), add="+")
        self.bind("<Right>", lambda _event: self._cycle(1), add="+")
        self.bind("<FocusOut>", self._schedule_focus_check, add="+")

        self._owner = master.winfo_toplevel()

    def set_items(self, items: Sequence[_MenuItem]) -> None:
        """Refresh commands/checkmarks without rebuilding the cached widgets."""

        updated = tuple(items)
        if len(updated) != len(self._items) or any(
            old.kind != new.kind or old.label != new.label
            for old, new in zip(self._items, updated)
        ):
            raise ValueError("Cached popup menu structure cannot change")
        self._items = updated
        for index, row in self._buttons.items():
            text = "\u2713" if updated[index].checked else ""
            if row.check_label.cget("text") != text:
                row.check_label.configure(text=text)

    def show(self, anchor: ctk.CTkButton) -> None:
        """Position and reveal a prebuilt popup without recreating its rows."""

        if self._closed:
            return
        if self._visible:
            self._position_below(anchor)
            self.lift()
            return
        self._visible = True
        self._position_below(anchor)
        # A binding on the owning root receives clicks from its normal widget
        # tree, but not clicks inside this separate popup toplevel. Install it
        # only while visible so the cached hidden menus remain inert.
        self._outside_binding = self._owner.bind(
            "<ButtonPress-1>", self._close_from_owner_click, add="+"
        )
        self.deiconify()
        self.lift()
        if self._command_indices:
            self._select(self._command_indices[0])
        self.after_idle(self._focus_if_visible)

    def hide(self) -> None:
        """Withdraw the popup for fast reuse on the next menu click."""

        if self._closed or not self._visible:
            return
        self._visible = False
        self._remove_outside_binding()
        self.withdraw()
        self._on_close(self)

    def _focus_if_visible(self) -> None:
        if self._visible and not self._closed:
            try:
                self.focus_force()
            except tk.TclError:
                pass

    def _remove_outside_binding(self) -> None:
        binding = self._outside_binding
        self._outside_binding = None
        if not binding:
            return
        try:
            self._owner.unbind("<ButtonPress-1>", binding)
        except tk.TclError:
            pass

    def _position_below(self, anchor: ctk.CTkButton) -> None:
        self.update_idletasks()
        width = max(1, self.winfo_reqwidth())
        height = max(1, self.winfo_reqheight())
        x = anchor.winfo_rootx()
        y = anchor.winfo_rooty() + anchor.winfo_height()
        left, top, right, bottom = work_area_bounds(self, parent=self._owner)
        x = max(left, min(x, right - width))
        if y + height > bottom:
            y = max(top, anchor.winfo_rooty() - height)
        self.geometry(f"+{x}+{y}")

    def _apply_aidas_theme(self) -> None:
        self.configure(background=COLORS.border_strong)
        for separator in self._separators:
            separator.configure(background=COLORS.border_strong)

    def _select(self, index: int) -> None:
        if index not in self._buttons:
            return
        if index == self._selected:
            return
        previous = self._buttons.get(self._selected)
        if previous is not None:
            previous.set_selected(False)
        self._selected = index
        self._buttons[index].set_selected(True)

    def _move_selection(self, delta: int) -> str:
        if not self._command_indices:
            return "break"
        try:
            position = self._command_indices.index(self._selected)
        except ValueError:
            position = 0 if delta >= 0 else len(self._command_indices) - 1
        else:
            position = (position + delta) % len(self._command_indices)
        self._select(self._command_indices[position])
        return "break"

    def _select_edge(self, *, first: bool) -> str:
        if self._command_indices:
            self._select(self._command_indices[0 if first else -1])
        return "break"

    def _invoke(self, index: int) -> None:
        item = self._items[index]
        command = item.command
        self.hide()
        if command is not None:
            command()

    def _invoke_selected(self, _event=None) -> str:
        if self._selected in self._buttons:
            self._invoke(self._selected)
        return "break"

    def _cycle(self, delta: int) -> str:
        # The owner captures the active menu before hiding this cached popup,
        # then reveals the adjacent cached menu.
        self._on_cycle(delta)
        return "break"

    def _close_from_event(self, _event=None) -> str:
        self.hide()
        return "break"

    def _close_from_owner_click(self, event=None) -> None:
        # Menu-bar buttons own their toggle/switch behavior. Let their command
        # run after ButtonPress instead of destroying and reopening the popup.
        widget = getattr(event, "widget", None)
        while widget is not None:
            if widget is self._menu_bar:
                return
            widget = getattr(widget, "master", None)
        self.hide()

    def _schedule_focus_check(self, _event=None) -> None:
        if self._visible and not self._closed:
            self.after(20, self._close_if_focus_outside)

    def _close_if_focus_outside(self) -> None:
        if self._closed or not self._visible:
            return
        focused = self.focus_get()
        widget = focused
        while widget is not None:
            if widget is self:
                return
            widget = getattr(widget, "master", None)
        self.hide()

    def destroy(self) -> None:
        if self._closed:
            return
        was_visible = self._visible
        self._closed = True
        self._visible = False
        self._remove_outside_binding()
        try:
            super().destroy()
        finally:
            if was_visible:
                self._on_close(self)


class ApplicationMenuBar(ctk.CTkFrame):
    """Fully themed File/View/Help application menu.

    The widget is an ordinary CTk frame: callers can ``pack`` or ``grid`` it at
    the top of a window.  ``set_appearance`` updates the checked View entry;
    CustomTkinter itself automatically redraws every dual-color token.
    """

    MENU_NAMES = ("File", "View", "Help")
    _BAR_HEIGHT = 34
    _BUTTON_WIDTH = 50
    _BUTTON_HEIGHT = 28
    _ROW_PADDING_X = 6
    _ROW_PADDING_Y = (2, 3)
    _BUTTON_GAP = 2

    def __init__(
        self,
        master,
        *,
        appearance_modes: Sequence[str],
        current_appearance: str,
        set_appearance_command: Callable[[str], None],
        browse_sdb_command: Callable[[], None],
        check_updates_command: Callable[[], None],
        about_command: Callable[[], None],
        exit_command: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            master,
            height=self._BAR_HEIGHT,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["menu_bar"],
        )
        self.pack_propagate(False)
        self.grid_propagate(False)

        self._appearance_modes = tuple(str(mode) for mode in appearance_modes)
        self._current_appearance = self._canonical_appearance(current_appearance)
        self._set_appearance_command = set_appearance_command
        self._browse_sdb_command = browse_sdb_command
        self._check_updates_command = check_updates_command
        self._about_command = about_command
        self._exit_command = exit_command or master.winfo_toplevel().destroy
        self._popup: _PopupMenu | None = None
        self._popup_cache: dict[str, _PopupMenu] = {}
        self._active_menu: str | None = None
        self._buttons: dict[str, ctk.CTkButton] = {}
        self._root_bindings: list[tuple[str, str]] = []

        row = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        row.pack(
            side="top",
            fill="x",
            padx=self._ROW_PADDING_X,
            pady=self._ROW_PADDING_Y,
        )
        for menu_name in self.MENU_NAMES:
            button = ctk.CTkButton(
                row,
                text=menu_name,
                command=lambda name=menu_name: self._toggle_menu(name),
                width=self._BUTTON_WIDTH,
                height=self._BUTTON_HEIGHT,
                corner_radius=SHAPES.corner_radius_sm,
                border_width=0,
                fg_color="transparent",
                hover_color=COLOR_PAIRS["primary_soft"],
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.body_size,
                ),
            )
            button.pack(side="left", padx=(0, self._BUTTON_GAP))
            button.bind("<Down>", lambda _event, name=menu_name: self._open_from_key(name), add="+")
            button.bind("<Return>", lambda _event, name=menu_name: self._open_from_key(name), add="+")
            button.bind("<space>", lambda _event, name=menu_name: self._open_from_key(name), add="+")
            button.bind("<Left>", lambda _event, name=menu_name: self._focus_adjacent(name, -1), add="+")
            button.bind("<Right>", lambda _event, name=menu_name: self._focus_adjacent(name, 1), add="+")
            self._buttons[menu_name] = button

        ctk.CTkFrame(
            self,
            height=1,
            corner_radius=0,
            fg_color=COLOR_PAIRS["border"],
        ).place(relx=0, rely=1, relwidth=1, anchor="sw")

        owner = master.winfo_toplevel()
        for sequence, name in (("<Alt-f>", "File"), ("<Alt-v>", "View"), ("<Alt-h>", "Help")):
            binding = owner.bind(
                sequence,
                lambda _event, menu=name: self._open_from_key(menu),
                add="+",
            )
            if binding:
                self._root_bindings.append((sequence, binding))
        binding = owner.bind("<F10>", lambda _event: self._open_from_key("File"), add="+")
        if binding:
            self._root_bindings.append(("<F10>", binding))

        # Popup construction is the expensive part of CustomTkinter menus
        # (font/canvas creation and anti-aliased drawing). Build each menu once
        # while the startup splash is visible, then only show/hide it on click.
        for menu_name in self.MENU_NAMES:
            self._popup_cache[menu_name] = self._create_popup(menu_name)

    @property
    def current_appearance(self) -> str:
        """Return the appearance currently marked in the View menu."""

        return self._current_appearance

    def set_appearance(self, mode: str) -> None:
        """Update the checked appearance entry without invoking its callback."""

        self._current_appearance = self._canonical_appearance(mode)
        popup = self._popup_cache.get("View")
        if popup is not None:
            popup.set_items(self._items_for("View"))

    def close_menu(self) -> None:
        """Close the active popup, if any."""

        popup = self._popup
        if popup is not None:
            popup.hide()

    def _canonical_appearance(self, mode: object) -> str:
        requested = str(mode)
        for candidate in self._appearance_modes:
            if candidate.casefold() == requested.casefold():
                return candidate
        return requested

    def _items_for(self, menu_name: str) -> tuple[_MenuItem, ...]:
        if menu_name == "File":
            return (
                _MenuItem("command", "Browse SDB Parent Directory", self._browse_sdb_command),
                _MenuItem("separator"),
                _MenuItem("command", "Exit", self._exit_command),
            )
        if menu_name == "View":
            appearance_items = tuple(
                _MenuItem(
                    "command",
                    mode,
                    lambda selected=mode: self._select_appearance(selected),
                    checked=mode == self._current_appearance,
                )
                for mode in self._appearance_modes
            )
            return (_MenuItem("heading", "Appearance"), *appearance_items)
        return (
            _MenuItem("command", "Check for Updates...", self._check_updates_command),
            _MenuItem("separator"),
            _MenuItem("command", "About", self._about_command),
        )

    def _select_appearance(self, mode: str) -> None:
        self._current_appearance = self._canonical_appearance(mode)
        self._set_appearance_command(self._current_appearance)

    def _toggle_menu(self, menu_name: str) -> None:
        if self._active_menu == menu_name and self._popup is not None:
            self.close_menu()
            return
        self._open_menu(menu_name)

    def _create_popup(self, menu_name: str) -> _PopupMenu:
        return _PopupMenu(
            self,
            items=self._items_for(menu_name),
            on_close=self._popup_closed,
            on_cycle=self._cycle_menu,
        )

    def _open_menu(self, menu_name: str) -> None:
        if menu_name not in self._buttons:
            return
        old_popup = self._popup
        if old_popup is not None:
            # Clear first so its close notification cannot overwrite the new
            # active-menu state while switching between cached popups.
            self._popup = None
            old_popup.hide()
        self._reset_button_colors()
        self._active_menu = menu_name
        anchor = self._buttons[menu_name]
        anchor.configure(fg_color=COLOR_PAIRS["primary_soft"])
        popup = self._popup_cache[menu_name]
        popup.set_items(self._items_for(menu_name))
        self._popup = popup
        popup.show(anchor)

    def _open_from_key(self, menu_name: str) -> str:
        self._open_menu(menu_name)
        return "break"

    def _popup_closed(self, popup: _PopupMenu) -> None:
        if self._popup is popup:
            self._popup = None
            self._active_menu = None
            self._reset_button_colors()

    def _reset_button_colors(self) -> None:
        for button in self._buttons.values():
            if button.cget("fg_color") != "transparent":
                button.configure(fg_color="transparent")

    def _cycle_menu(self, delta: int) -> None:
        active = self._active_menu
        try:
            position = self.MENU_NAMES.index(active) if active is not None else 0
        except ValueError:
            position = 0
        self._open_menu(self.MENU_NAMES[(position + delta) % len(self.MENU_NAMES)])

    def _focus_adjacent(self, menu_name: str, delta: int) -> str:
        position = self.MENU_NAMES.index(menu_name)
        self._buttons[self.MENU_NAMES[(position + delta) % len(self.MENU_NAMES)]].focus_set()
        return "break"

    def destroy(self) -> None:
        self.close_menu()
        for popup in self._popup_cache.values():
            popup.destroy()
        self._popup_cache.clear()
        try:
            owner = self.winfo_toplevel()
            for sequence, binding in self._root_bindings:
                owner.unbind(sequence, binding)
        except tk.TclError:
            pass
        self._root_bindings.clear()
        super().destroy()


__all__ = ["ApplicationMenuBar"]
