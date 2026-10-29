"""Closable, theme-aware tabs built from public Tk and CustomTkinter widgets."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from typing import TYPE_CHECKING, Callable

import customtkinter as ctk
from PIL import Image, ImageDraw
from aidas.ui.theme import COLOR_PAIRS, COLORS, CONTROLS, SHAPES, TYPOGRAPHY, resolve_color

if TYPE_CHECKING:
    from aidas.utils.ui_utils import HoverToolTip


TabCallback = Callable[["ClosableTabView", ctk.CTkFrame], None]


class _TabModel:
    """Display-independent tab identity, order, label, and selection state."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.pages: dict[str, object] = {}
        self.labels: dict[str, str] = {}
        self.tooltips: dict[str, str | None] = {}
        self.selected: str | None = None

    def add(self, page: object, text: object, tooltip: object | None = None) -> str:
        page_id = str(page)
        if page_id in self.pages:
            raise ValueError(f"Tab page already exists: {page_id}")
        self.order.append(page_id)
        self.pages[page_id] = page
        self.labels[page_id] = str(text)
        self.tooltips[page_id] = None if tooltip is None else str(tooltip)
        if self.selected is None:
            self.selected = page_id
        return page_id

    def tooltip_text(self, page_id: str) -> str:
        tooltip = self.tooltips[page_id]
        return self.labels[page_id] if tooltip is None else tooltip

    def resolve(self, tab: object) -> str:
        for page_id, page in self.pages.items():
            if tab is page:
                return page_id
        if isinstance(tab, int):
            if 0 <= tab < len(self.order):
                return self.order[tab]
            raise IndexError(tab)
        text = str(tab)
        if text == "current" and self.selected is not None:
            return self.selected
        if text in self.pages:
            return text
        if text.isdecimal():
            index = int(text)
            if 0 <= index < len(self.order):
                return self.order[index]
        raise KeyError(text)

    def forget(self, page_id: str) -> tuple[object, str | None, bool]:
        index = self.order.index(page_id)
        page = self.pages.pop(page_id)
        self.labels.pop(page_id)
        self.tooltips.pop(page_id)
        self.order.pop(index)
        was_selected = self.selected == page_id
        if was_selected:
            self.selected = self.order[min(index, len(self.order) - 1)] if self.order else None
        return page, self.selected, was_selected


@dataclass
class _TabWidgets:
    frame: ctk.CTkFrame
    title: ctk.CTkButton
    close: ctk.CTkButton
    close_hovered: bool = False
    title_tooltip: HoverToolTip | None = None
    body_tooltip: HoverToolTip | None = None
    close_tooltip: HoverToolTip | None = None
    active: bool = False
    bottom_join: ctk.CTkFrame | None = None
    hovered: bool = False
    hover_after_id: str | None = None
    border_signature: tuple | None = None
    bottom_join_signature: tuple | None = None


class _PixelAlignedTabFrame(ctk.CTkFrame):
    """Paint the full allocated bounds, including odd and scaled pixel sizes."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._draw_engine.set_round_to_even_numbers(False, False)
        self._draw()

    def _update_dimensions_event(self, event) -> None:
        # Retain fractional logical sizes. CTk can truncate integer Configure
        # dimensions when converting them back from physical display pixels.
        scale = self._get_widget_scaling()
        width, height = event.width / scale, event.height / scale
        if self._current_width != width or self._current_height != height:
            self._current_width, self._current_height = width, height
            self._draw(no_color_updates=True)


class ClosableTabView(ctk.CTkFrame):
    """Rounded, theme-aware tabs with connected pages and simple close icons.

    Page IDs are the immutable Tk widget paths returned by :meth:`tabs`, never
    the user-facing labels.  A close click delegates to ``close_command`` so a
    controller can save, cancel, or call :meth:`forget` when it is ready.
    """

    _MAX_TITLE_WIDTH = 360
    _TAB_HEIGHT = 38
    _CLOSE_SIZE = 24
    _CLOSE_WIDTH = _CLOSE_SIZE
    _CLOSE_ICON_SIZE = 12
    _CLOSE_PADDING = CONTROLS.gap
    _TITLE_PADDING = CONTROLS.padding_x
    _TAB_RADIUS = CONTROLS.gap
    _TAB_GAP = CONTROLS.gap // 2
    _SCROLLBAR_HEIGHT = 4
    _BORDER_PIXELS = 1

    def __init__(
        self,
        master,
        *,
        command: TabCallback | None = None,
        close_command: TabCallback | None = None,
        **frame_kwargs,
    ) -> None:
        frame_kwargs.pop("style", None)
        options = {
            "corner_radius": 0,
            "border_width": 0,
            "fg_color": COLOR_PAIRS["surface"],
        }
        options.update(frame_kwargs)
        super().__init__(master, **options)

        self.command = command
        self.close_command = close_command
        self._model = _TabModel()
        self._tab_widgets: dict[str, _TabWidgets] = {}
        self._scrollbar_visible = False
        self._header_overflow = False
        self._header_hovered = False
        self._header_hover_after_id = None

        self._tab_font = ctk.CTkFont(
            family=TYPOGRAPHY.family,
            size=TYPOGRAPHY.body_size + 1,
            weight=TYPOGRAPHY.normal_weight,
        )
        self._selected_tab_font = ctk.CTkFont(
            family=TYPOGRAPHY.family,
            size=TYPOGRAPHY.body_size + 1,
            weight=TYPOGRAPHY.semibold_weight,
        )
        self._close_icon = self._create_close_icon()
        self._header_sync_after_id = None
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._header = ctk.CTkFrame(
            self,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["sidebar"],
        )
        self._header.grid(row=0, column=0, sticky="ew")
        self._header.grid_columnconfigure(0, weight=1)

        # Reserve a thin rail above the tabs. Revealing its scrollbar must
        # leave both the selected-tab join and the image geometry unchanged.
        self._scrollbar_slot = ctk.CTkFrame(
            self._header,
            height=self._SCROLLBAR_HEIGHT,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["sidebar"],
        )
        self._scrollbar_slot.grid(row=0, column=0, sticky="ew")
        self._scrollbar_slot.grid_propagate(False)
        self._scrollbar_slot.grid_columnconfigure(0, weight=1)
        self._scrollbar_slot.grid_rowconfigure(0, weight=1)

        self._header_canvas = tk.Canvas(
            self._header,
            height=self._TAB_HEIGHT + self._TAB_GAP,
            borderwidth=0,
            highlightthickness=0,
            takefocus=False,
            xscrollincrement=1,
        )
        self._header_canvas.grid(row=1, column=0, sticky="ew")

        self._header_scrollbar = ctk.CTkScrollbar(
            self._scrollbar_slot,
            orientation="horizontal",
            height=self._SCROLLBAR_HEIGHT,
            corner_radius=self._SCROLLBAR_HEIGHT // 2,
            border_spacing=0,
            command=self._header_canvas.xview,
            fg_color=COLOR_PAIRS["sidebar"],
            button_color=COLOR_PAIRS["tab_inactive_text"],
            button_hover_color=COLOR_PAIRS["tab_selected_text"],
        )
        self._header_scrollbar.grid(row=0, column=0, sticky="ew")
        self._header_scrollbar.grid_remove()
        self._header_canvas.configure(xscrollcommand=self._on_xscroll)

        self._tab_row = ctk.CTkFrame(
            self._header_canvas,
            width=1,
            height=self._TAB_HEIGHT + self._TAB_GAP,
            corner_radius=0,
            border_width=0,
            bg_color=COLOR_PAIRS["sidebar"],
            fg_color=COLOR_PAIRS["sidebar"],
        )
        self._tab_row_window = self._header_canvas.create_window(
            0,
            0,
            anchor="nw",
            window=self._tab_row,
        )
        self._tab_row.bind("<Configure>", self._sync_header_geometry, add="+")
        self._header_canvas.bind("<Configure>", self._sync_header_geometry, add="+")
        for widget in (self._header, self._scrollbar_slot, self._header_canvas, self._tab_row, self._header_scrollbar):
            self._bind_header_hover(widget)
        for widget in (self._header, self._scrollbar_slot, self._header_canvas, self._tab_row):
            self._bind_horizontal_scroll(widget)

        self._join_frame = _PixelAlignedTabFrame(
            self,
            height=SHAPES.border_width,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["tab_border"],
        )
        self._join_frame.grid(row=1, column=0, sticky="ew")
        self._join_frame.grid_propagate(False)
        self._join_canvas = tk.Canvas(
            self._join_frame, borderwidth=0, highlightthickness=0, takefocus=False,
        )
        self._join_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self._join_canvas.bind("<Configure>", self._draw_content_join, add="+")

        self._content = ctk.CTkFrame(
            self,
            corner_radius=0,
            border_width=0,
            bg_color=COLOR_PAIRS["surface"],
            fg_color=COLOR_PAIRS["surface"],
        )
        self._content.grid(row=2, column=0, sticky="nsew")
        self._content.grid_rowconfigure(0, weight=1)
        self._content.grid_columnconfigure(0, weight=1)
        self._apply_aidas_theme()

    @classmethod
    def _create_close_icon(cls) -> ctk.CTkImage:
        """Draw a symmetric X without the baseline padding of a font glyph."""

        scale = 4
        side = cls._CLOSE_ICON_SIZE * scale
        inset = round(2.5 * scale)
        far = side - inset - 1

        def draw_icon(color):
            icon = Image.new("RGBA", (side, side))
            draw = ImageDraw.Draw(icon)
            stroke = round(1.75 * scale)
            draw.line((inset, inset, far, far), fill=color, width=stroke)
            draw.line((inset, far, far, inset), fill=color, width=stroke)
            return icon

        icon = ctk.CTkImage(
            light_image=draw_icon(COLOR_PAIRS["on_primary"][0]),
            dark_image=draw_icon(COLOR_PAIRS["on_primary"][1]),
            size=(cls._CLOSE_ICON_SIZE, cls._CLOSE_ICON_SIZE),
        )
        icon._aidas_monochrome_action = True
        return icon

    def add(self, *, text: str = "", tooltip: str | None = None) -> ctk.CTkFrame:
        """Create a page with an independent full-path hover description."""

        # The UI package also initializes while shared helpers import theme.
        from aidas.ui.components import AppButton
        from aidas.utils.ui_utils import HoverToolTip

        page = ctk.CTkFrame(
            self._content,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["surface"],
        )
        page.grid(row=0, column=0, sticky="nsew")
        page_id = self._model.add(page, text, tooltip)

        tab_frame = _PixelAlignedTabFrame(
            self._tab_row,
            height=self._TAB_HEIGHT,
            corner_radius=self._TAB_RADIUS,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["tab_border"],
            bg_color=COLOR_PAIRS["sidebar"],
        )
        tab_frame.pack(
            side="left", anchor="s",
            padx=(CONTROLS.gap if len(self._model.order) == 1 else self._TAB_GAP, 0),
            pady=(self._TAB_GAP, 0),
        )
        tab_frame.pack_propagate(False)

        title_button = ctk.CTkButton(
            tab_frame,
            text=str(text),
            height=self._TAB_HEIGHT,
            corner_radius=0,
            border_width=0,
            border_spacing=0,
            hover=False,
            hover_color=COLOR_PAIRS["tab_inactive_hover"],
            font=self._tab_font,
            anchor="w",
            command=lambda tab_id=page_id: self._select_id(tab_id, notify=True),
        )
        title_button.pack(
            side="left", fill="y", padx=(self._TITLE_PADDING, 0), pady=self._TAB_GAP,
        )
        # Keep a long label inside its capped title area, leaving room for
        # the close button even when the header needs horizontal scrolling.
        title_button.grid_propagate(False)

        close_button = AppButton(
            tab_frame,
            text="",
            image=self._close_icon,
            compound="top",
            variant="ghost",
            anchor="center",
            corner_radius=4,
            border_width=0,
            border_spacing=0,
            width=self._CLOSE_WIDTH,
            height=self._CLOSE_SIZE,
            command=lambda tab_id=page_id: self._request_close(tab_id),
        )
        close_button.pack(
            side="left", padx=(0, self._CLOSE_PADDING),
            pady=(self._TAB_HEIGHT - self._CLOSE_SIZE) // 2,
        )

        bottom_join = _PixelAlignedTabFrame(
            tab_frame, height=self._TAB_RADIUS, corner_radius=0, border_width=0,
            bg_color=COLOR_PAIRS["tab_selected"],
            fg_color=COLOR_PAIRS["tab_selected"],
        )
        tooltip_text = self._model.tooltip_text(page_id)
        body_tooltip = HoverToolTip(tab_frame, tooltip_text, wraplength=0)
        bottom_join.bind("<Enter>", body_tooltip._schedule_show, add="+")
        bottom_join.bind("<Leave>", body_tooltip._hide, add="+")
        self._tab_widgets[page_id] = _TabWidgets(
            tab_frame, title_button, close_button,
            title_tooltip=HoverToolTip(title_button, tooltip_text, wraplength=0),
            body_tooltip=body_tooltip,
            close_tooltip=HoverToolTip(
                close_button, tooltip_text if tooltip is not None and tooltip_text else "Close tab", wraplength=0,
            ),
            bottom_join=bottom_join,
        )
        tab_frame.bind(
            "<Configure>",
            lambda _event, tab_id=page_id: self._update_tab_border_geometry(tab_id),
            add="+",
        )
        self._size_tab(page_id)
        self._bind_horizontal_scroll(tab_frame)
        self._bind_horizontal_scroll(title_button)
        self._bind_horizontal_scroll(close_button)
        self._bind_horizontal_scroll(bottom_join)
        for widget in (tab_frame, bottom_join):
            widget.bind(
                "<Button-1>",
                lambda _event, tab_id=page_id: self._select_id(tab_id, notify=True),
                add="+",
            )
        for widget in (tab_frame, title_button, close_button, bottom_join):
            widget.bind(
                "<Enter>",
                lambda event, tab_id=page_id, is_close=widget is close_button: self._on_tab_enter(tab_id, event, is_close=is_close),
                add="+",
            )
            widget.bind(
                "<Leave>",
                lambda event, tab_id=page_id, is_close=widget is close_button: self._on_tab_leave(tab_id, event, is_close=is_close),
                add="+",
            )

        if self._model.selected == page_id:
            self._select_id(page_id, notify=False)
        else:
            page.grid_remove()
            self._style_tab(page_id, active=False)
        self._queue_sync_header_geometry()
        return page

    def tabs(self) -> tuple[str, ...]:
        """Return stable page IDs in display order."""

        return tuple(self._model.order)

    def select(self, tab: object | None = None) -> str:
        """Return the active page ID, or select a page and invoke ``command``."""

        if tab is None:
            return self._model.selected or ""
        page_id = self._resolve_page_id(tab)
        self._select_id(page_id, notify=True)
        return page_id

    def nametowidget(self, name):
        """Resolve a stable tab ID to its CTk content page."""

        try:
            return self._model.pages[self._model.resolve(name)]
        except (IndexError, KeyError, ValueError):
            try:
                return super().nametowidget(name)
            except (KeyError, tk.TclError) as exc:
                raise tk.TclError(f"Invalid tab identifier: {name!r}") from exc

    def tab(self, tab: object, option: str | None = None, **options):
        """Read or update the visible title and its independent hover text."""

        page_id = self._resolve_page_id(tab)
        if option is not None:
            if options:
                raise TypeError("tab() cannot combine an option query with option updates")
            if option == "text":
                return self._model.labels[page_id]
            if option == "tooltip":
                return self._model.tooltip_text(page_id)
            raise tk.TclError(f"Unsupported tab option: {option!r}")

        unknown = set(options) - {"text", "tooltip"}
        if unknown:
            name = sorted(unknown)[0]
            raise tk.TclError(f"Unsupported tab option: {name!r}")
        if "text" in options:
            text = str(options["text"])
            self._model.labels[page_id] = text
            widgets = self._tab_widgets[page_id]
            widgets.title.configure(text=text)
            self._size_tab(page_id)
            self._queue_sync_header_geometry()
        if "tooltip" in options:
            tooltip = options["tooltip"]
            self._model.tooltips[page_id] = None if tooltip is None else str(tooltip)
        if options:
            self._update_tab_tooltips(page_id)
            return None
        return {"text": self._model.labels[page_id], "tooltip": self._model.tooltip_text(page_id)}

    def _update_tab_tooltips(self, page_id: str) -> None:
        widgets = self._tab_widgets[page_id]
        tooltip_text = self._model.tooltip_text(page_id)
        for tooltip in (widgets.title_tooltip, widgets.body_tooltip):
            if tooltip is not None:
                tooltip.text = tooltip_text
        if widgets.close_tooltip is not None:
            widgets.close_tooltip.text = (
                tooltip_text if self._model.tooltips[page_id] is not None and tooltip_text else "Close tab"
            )

    @staticmethod
    def _hide_tab_tooltips(widgets: _TabWidgets) -> None:
        for tooltip in (widgets.title_tooltip, widgets.body_tooltip, widgets.close_tooltip):
            if tooltip is not None:
                tooltip._hide()

    def forget(self, tab: object) -> None:
        """Remove and destroy a page, silently displaying a nearby survivor."""

        page_id = self._resolve_page_id(tab)
        page, replacement_id, was_selected = self._model.forget(page_id)
        widgets = self._tab_widgets.pop(page_id)
        self._cancel_tab_hover(widgets)
        self._hide_tab_tooltips(widgets)
        widgets.frame.destroy()
        try:
            page.destroy()
        except tk.TclError:
            pass

        if was_selected and replacement_id is not None:
            self._show_selected_page(replacement_id)
        elif replacement_id is not None:
            self._style_tab(replacement_id, active=True)
        self._queue_sync_header_geometry()

    def index(self, tab: object) -> int:
        """Return a page index, including ``end``, ``current``, and ``@x,y``."""

        text = str(tab)
        if text == "end":
            return len(self._model.order)
        if text.startswith("@"):
            try:
                x_text, y_text = text[1:].split(",", 1)
                x, y = int(x_text), int(y_text)
            except (TypeError, ValueError) as exc:
                raise tk.TclError(f"Invalid tab coordinate: {tab!r}") from exc
            self.update_idletasks()
            root_x = self.winfo_rootx() + x
            root_y = self.winfo_rooty() + y
            for index, page_id in enumerate(self._model.order):
                frame = self._tab_widgets[page_id].frame
                if (
                    frame.winfo_rootx() <= root_x < frame.winfo_rootx() + frame.winfo_width()
                    and frame.winfo_rooty() <= root_y < frame.winfo_rooty() + frame.winfo_height()
                ):
                    return index
            raise tk.TclError(f"No tab at coordinate: {tab!r}")
        page_id = self._resolve_page_id(tab)
        return self._model.order.index(page_id)

    def _resolve_page_id(self, tab: object) -> str:
        try:
            return self._model.resolve(tab)
        except (IndexError, KeyError, ValueError) as exc:
            raise tk.TclError(f"Invalid tab identifier: {tab!r}") from exc

    def _select_id(self, page_id: str, *, notify: bool) -> None:
        self._model.resolve(page_id)
        previous_id = self._model.selected
        if previous_id is not None and previous_id != page_id:
            previous_page = self._model.pages[previous_id]
            try:
                previous_page.grid_remove()
            except tk.TclError:
                pass
            self._style_tab(previous_id, active=False)

        self._model.selected = page_id
        self._show_selected_page(page_id)
        self.after_idle(lambda selected=page_id: self._scroll_tab_into_view(selected))
        if notify:
            page = self._model.pages[page_id]
            if self.command is not None:
                self.command(self, page)
            try:
                self.event_generate("<<NotebookTabChanged>>", when="tail")
            except tk.TclError:
                pass

    def _show_selected_page(self, page_id: str) -> None:
        page = self._model.pages[page_id]
        page.grid(row=0, column=0, sticky="nsew")
        page.tkraise()
        # Every non-selected tab is styled when it is added or when it stops
        # being active in ``_select_id``. Repainting the full header here made
        # selection cost grow linearly with the number of open result tabs.
        self._style_tab(page_id, active=True)

    def _request_close(self, page_id: str) -> None:
        if page_id not in self._model.pages or self.close_command is None:
            return
        self.close_command(self, self._model.pages[page_id])

    def _style_tab(self, page_id: str, *, active: bool) -> None:
        widgets = self._tab_widgets.get(page_id)
        if widgets is None:
            return
        widgets.active = active
        background = self._tab_background(widgets)
        title_hover = COLOR_PAIRS["tab_selected_hover"] if active else COLOR_PAIRS["tab_inactive_hover"]
        widgets.frame.configure(
            fg_color=background,
            border_color=COLOR_PAIRS["tab_selected_border"] if active else COLOR_PAIRS["tab_border"],
        )
        title_options = {
            "fg_color": background,
            "hover_color": title_hover,
            "text_color": COLOR_PAIRS["tab_selected_text"] if active else COLOR_PAIRS["tab_inactive_text"],
            "text_color_disabled": COLOR_PAIRS["disabled_text"],
        }
        font = self._selected_tab_font if active else self._tab_font
        # Resetting an unchanged CTk font temporarily unmaps the title canvas,
        # which would dismiss a tooltip just as the pointer enters the tab.
        if widgets.title.cget("font") is not font:
            title_options["font"] = font
        widgets.title.configure(**title_options)
        if widgets.bottom_join is not None:
            widgets.bottom_join.configure(fg_color=background, bg_color=background)
        self._update_tab_border_geometry(page_id)
        self._set_close_hover(page_id, widgets.close_hovered)
        self._draw_content_join()

    def _update_tab_border_geometry(self, page_id: str) -> None:
        """Align the open lower corners to a single physical-pixel outline."""

        widgets = self._tab_widgets.get(page_id)
        if widgets is None:
            return
        scale = self._get_widget_scaling()
        inset = self._BORDER_PIXELS / scale
        sidebar = COLOR_PAIRS["sidebar"]
        lower_corner = COLOR_PAIRS["tab_selected_border"] if widgets.active else sidebar
        # Theme token pairs are live lists that mutate in place. Cache their
        # resolved values so an interface change cannot rewrite the old key.
        border_signature = (inset, resolve_color(sidebar), resolve_color(lower_corner))
        if widgets.border_signature != border_signature:
            widgets.frame.configure(
                border_width=inset,
                background_corner_colors=(sidebar, sidebar, lower_corner, lower_corner),
            )
            widgets.border_signature = border_signature
        if widgets.bottom_join is None:
            return
        if not widgets.active:
            if widgets.bottom_join_signature != (False,):
                widgets.bottom_join.place_forget()
                widgets.bottom_join_signature = (False,)
            return
        width = widgets.frame.winfo_width()
        if width <= 1:
            width = round(widgets.frame.cget("width") * scale)
        join_width = max(1, width - 2 * self._BORDER_PIXELS) / scale
        join_signature = (True, join_width, inset, self._TAB_RADIUS)
        if widgets.bottom_join_signature != join_signature:
            widgets.bottom_join.configure(width=join_width)
            widgets.bottom_join.place(x=inset, rely=1, y=-self._TAB_RADIUS)
            widgets.close.tkraise()
            widgets.bottom_join_signature = join_signature
        self._draw_content_join()

    @staticmethod
    def _tab_background(widgets: _TabWidgets):
        if widgets.active:
            return COLOR_PAIRS["tab_selected_hover" if widgets.hovered else "tab_selected"]
        return COLOR_PAIRS["tab_inactive_hover" if widgets.hovered else "tab_inactive"]

    def _set_tab_hover(self, page_id: str, hovered: bool) -> None:
        widgets = self._tab_widgets.get(page_id)
        if widgets is None or widgets.hovered == hovered:
            return
        widgets.hovered = hovered
        self._style_tab(page_id, active=widgets.active)

    def _cancel_tab_hover(self, widgets: _TabWidgets) -> None:
        if widgets.hover_after_id is not None:
            try:
                self.after_cancel(widgets.hover_after_id)
            except tk.TclError:
                pass
            widgets.hover_after_id = None

    def _on_tab_enter(self, page_id: str, _event=None, *, is_close=False) -> None:
        widgets = self._tab_widgets.get(page_id)
        if widgets is None:
            return
        self._cancel_tab_hover(widgets)
        self._on_header_enter()
        if is_close:
            widgets.close_hovered = True
        self._set_tab_hover(page_id, True)
        if is_close:
            self._set_close_hover(page_id, True)

    def _on_tab_leave(self, page_id: str, _event=None, *, is_close=False) -> None:
        widgets = self._tab_widgets.get(page_id)
        if widgets is None:
            return
        if is_close:
            self._set_close_hover(page_id, False)
        self._cancel_tab_hover(widgets)
        widgets.hover_after_id = self.after_idle(lambda: self._finish_tab_leave(page_id))
        self._on_header_leave()

    def _finish_tab_leave(self, page_id: str) -> None:
        widgets = self._tab_widgets.get(page_id)
        if widgets is None:
            return
        widgets.hover_after_id = None
        self._set_tab_hover(page_id, self._pointer_within(widgets.frame))

    @staticmethod
    def _pointer_within(widget) -> bool:
        try:
            if not widget.winfo_ismapped():
                return False
            x, y = widget.winfo_pointerxy()
            return (
                widget.winfo_rootx() <= x < widget.winfo_rootx() + widget.winfo_width()
                and widget.winfo_rooty() <= y < widget.winfo_rooty() + widget.winfo_height()
            )
        except tk.TclError:
            return False

    def _set_close_hover(self, page_id: str, hovered: bool) -> None:
        widgets = self._tab_widgets.get(page_id)
        if widgets is None:
            return
        widgets.close_hovered = bool(hovered)
        background = self._tab_background(widgets)
        foreground = (
            COLOR_PAIRS["tab_close_hover_text"] if hovered else
            COLOR_PAIRS["tab_selected_text"] if widgets.active else COLOR_PAIRS["tab_inactive_text"]
        )
        widgets.close.configure(
            bg_color=background,
            fg_color=background,
            hover_color=COLOR_PAIRS["tab_close_hover"],
            border_width=0,
            text_color=foreground,
        )
        if hovered:
            widgets.close._on_enter()
        else:
            widgets.close._on_leave()

    def _size_tab(self, page_id: str) -> None:
        widgets = self._tab_widgets[page_id]
        text = self._model.labels[page_id]
        try:
            measured = int(self._selected_tab_font.measure(text))
        except tk.TclError:
            measured = len(text) * max(7, TYPOGRAPHY.body_size // 2)
        title_width = min(
            self._MAX_TITLE_WIDTH,
            max(48, measured + CONTROLS.gap),
        )
        widgets.title.configure(width=title_width)
        frame_width = title_width + self._TITLE_PADDING + self._CLOSE_WIDTH + self._CLOSE_PADDING + SHAPES.border_width
        widgets.frame.configure(width=frame_width)
        self._update_tab_border_geometry(page_id)

    def _draw_content_join(self, _event=None) -> None:
        """Open the panel's top border directly beneath the selected tab."""

        try:
            widgets = self._tab_widgets.get(self._model.selected)
            border = COLOR_PAIRS["tab_selected_border" if widgets is not None else "tab_border"]
            scale = self._get_widget_scaling()
            resolved_border = resolve_color(border)
            border_signature = (scale, resolved_border)
            if border_signature != self.__dict__.get("_join_border_signature"):
                self._join_frame.configure(
                    height=self._BORDER_PIXELS / scale, fg_color=border,
                )
                self._join_canvas.configure(background=resolved_border)
                self._join_border_signature = border_signature

            item_id = self.__dict__.get("_selected_join_id")
            previous = self.__dict__.get("_selected_join_signature")
            if widgets is not None:
                left = widgets.frame.winfo_rootx() - self._join_canvas.winfo_rootx()
                right = left + widgets.frame.winfo_width()
                left = max(0, left + self._BORDER_PIXELS)
                right = min(self._join_canvas.winfo_width(), right - self._BORDER_PIXELS)
            if widgets is None or right <= left:
                if item_id is not None and previous is not None:
                    self._join_canvas.itemconfigure(item_id, state="hidden")
                    self._selected_join_signature = None
                return
            height = self._join_canvas.winfo_height()
            fill = resolve_color(self._tab_background(widgets))
            signature = (left, right, height, fill)
            if signature == previous:
                return
            if item_id is None:
                self._selected_join_id = self._join_canvas.create_rectangle(
                    left, 0, right, height, fill=fill, outline="", tags="selected_join",
                )
            else:
                self._join_canvas.coords(item_id, left, 0, right, height)
                self._join_canvas.itemconfigure(item_id, fill=fill, state="normal")
            self._selected_join_signature = signature
        except tk.TclError:
            pass

    def _queue_sync_header_geometry(self) -> None:
        if self._header_sync_after_id is None:
            try:
                self._header_sync_after_id = self.after_idle(self._sync_header_geometry)
            except tk.TclError:
                self._header_sync_after_id = None

    def _sync_header_geometry(self, _event=None) -> None:
        if self._header_sync_after_id is not None:
            try:
                self.after_cancel(self._header_sync_after_id)
            except tk.TclError:
                pass
        self._header_sync_after_id = None
        try:
            canvas_width = max(1, self._header_canvas.winfo_width())
            content_width = max(canvas_width, self._tab_row.winfo_reqwidth())
            content_height = max(1, self._tab_row.winfo_reqheight())
            self._header_canvas.itemconfigure(
                self._tab_row_window,
                width=content_width,
                height=content_height,
            )
            self._header_canvas.configure(
                height=content_height,
                scrollregion=(0, 0, content_width, content_height),
            )
            self._update_header_overflow(content_width > canvas_width + 1)
            self._draw_content_join()
            if _event is not None and self._model.selected is not None:
                self.after_idle(lambda: self._scroll_tab_into_view(self._model.selected))
        except tk.TclError:
            pass

    def _on_xscroll(self, first: str, last: str) -> None:
        self._header_scrollbar.set(first, last)
        self._draw_content_join()

    def _update_header_overflow(self, overflow: bool) -> None:
        was_overflowing = self._header_overflow
        self._header_overflow = overflow
        self._update_scrollbar_visibility()
        if was_overflowing and not overflow:
            self._header_canvas.xview_moveto(0.0)

    def _update_scrollbar_visibility(self) -> None:
        visible = self._header_overflow and self._header_hovered
        if visible == self._scrollbar_visible:
            return
        self._scrollbar_visible = visible
        if visible:
            self._header_scrollbar.grid()
        else:
            self._header_scrollbar.grid_remove()

    def _bind_header_hover(self, widget) -> None:
        widget.bind("<Enter>", self._on_header_enter, add="+")
        widget.bind("<Leave>", self._on_header_leave, add="+")

    def _cancel_header_hover(self) -> None:
        if self._header_hover_after_id is not None:
            try:
                self.after_cancel(self._header_hover_after_id)
            except tk.TclError:
                pass
            self._header_hover_after_id = None

    def _on_header_enter(self, _event=None) -> None:
        self._cancel_header_hover()
        self._header_hovered = True
        self._update_scrollbar_visibility()

    def _on_header_leave(self, _event=None) -> None:
        self._cancel_header_hover()
        self._header_hover_after_id = self.after_idle(self._finish_header_leave)

    def _finish_header_leave(self) -> None:
        self._header_hover_after_id = None
        self._header_hovered = self._pointer_within(self._header)
        self._update_scrollbar_visibility()

    def _scroll_tab_into_view(self, page_id: str) -> None:
        widgets = self._tab_widgets.get(page_id)
        if widgets is None:
            return
        try:
            self._sync_header_geometry()
            visible_width = self._header_canvas.winfo_width()
            total_width = max(visible_width, self._tab_row.winfo_reqwidth())
            view_left = self._header_canvas.canvasx(0)
            view_right = view_left + visible_width
            tab_left = widgets.frame.winfo_x()
            tab_right = tab_left + widgets.frame.winfo_width()
            # On narrow windows a long tab cannot fit in full; keep its
            # close control visible rather than aligning only the title.
            if tab_right - tab_left > visible_width:
                target = tab_right - visible_width
            elif tab_left < view_left:
                target = tab_left
            elif tab_right > view_right:
                target = tab_right - visible_width
            else:
                return
            self._header_canvas.xview_moveto(max(0.0, min(1.0, target / total_width)))
        except tk.TclError:
            pass

    def _bind_horizontal_scroll(self, widget) -> None:
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_mousewheel, add="+")
        widget.bind("<Button-5>", self._on_mousewheel, add="+")

    def _on_mousewheel(self, event):
        if not self._header_overflow:
            return None
        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            delta = int(getattr(event, "delta", 0))
            units = -1 if delta > 0 else 1
        self._header_canvas.xview_scroll(units * CONTROLS.gap * 3, "units")
        return "break"

    def destroy(self) -> None:
        if self._header_sync_after_id is not None:
            try:
                self.after_cancel(self._header_sync_after_id)
            except tk.TclError:
                pass
            self._header_sync_after_id = None
        self._cancel_header_hover()
        for widgets in self._tab_widgets.values():
            self._cancel_tab_hover(widgets)
            self._hide_tab_tooltips(widgets)
        super().destroy()

    def _apply_aidas_theme(self) -> None:
        """Refresh the overflow canvas and retain selected and hovered styles."""

        try:
            self._header_canvas.configure(background=COLORS.sidebar)
        except tk.TclError:
            pass
        for widget in (self._header, self._scrollbar_slot, self._tab_row):
            widget.configure(fg_color=COLOR_PAIRS["sidebar"])
        self._tab_row.configure(bg_color=COLOR_PAIRS["sidebar"])
        self._content.configure(
            fg_color=COLOR_PAIRS["surface"], bg_color=COLOR_PAIRS["surface"],
        )
        for page_id in self._model.order:
            self._model.pages[page_id].configure(fg_color=COLOR_PAIRS["surface"])
            self._style_tab(page_id, active=page_id == self._model.selected)
        self._draw_content_join()


__all__ = ["ClosableTabView"]
