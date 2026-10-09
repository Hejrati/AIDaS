"""Closable, theme-aware tabs built from public Tk and CustomTkinter widgets."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from typing import TYPE_CHECKING, Callable

import customtkinter as ctk
from PIL import Image, ImageChops, ImageDraw

from aidas.ui.theme import COLOR_PAIRS, COLORS, CONTROLS, SHAPES, TYPOGRAPHY, resolve_color

if TYPE_CHECKING:
    from aidas.utils.ui_utils import HoverToolTip


TabCallback = Callable[["ClosableTabView", ctk.CTkFrame], None]


def _close_icon_image(color: str, size: int = 12) -> Image.Image:
    """Draw a centered cross without a font's baseline or side bearings."""

    scale = 4
    extent = size * scale
    image = Image.new("RGBA", (extent, extent), color)
    image.putalpha(0)
    draw = ImageDraw.Draw(image)
    start, end = 2 * scale, extent - 2 * scale - 1
    draw.line((start, start, end, end), fill=color, width=2 * scale)
    draw.line((start, end, end, start), fill=color, width=2 * scale)
    # Pillow's even-width strokes can differ by a subpixel on opposing sides.
    # Mirror the alpha mask to keep the visible ink symmetric about both axes.
    alpha = image.getchannel("A")
    alpha = ImageChops.lighter(alpha, alpha.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    alpha = ImageChops.lighter(alpha, alpha.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    image.putalpha(alpha)
    return image


class _TabModel:
    """Display-independent tab identity, order, label, and selection state."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.pages: dict[str, object] = {}
        self.labels: dict[str, str] = {}
        self.selected: str | None = None

    def add(self, page: object, text: object) -> str:
        page_id = str(page)
        if page_id in self.pages:
            raise ValueError(f"Tab page already exists: {page_id}")
        self.order.append(page_id)
        self.pages[page_id] = page
        self.labels[page_id] = str(text)
        if self.selected is None:
            self.selected = page_id
        return page_id

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
    active: bool = False
    bottom_join: ctk.CTkFrame | None = None
    hovered: bool = False
    hover_after_id: str | None = None


class ClosableTabView(ctk.CTkFrame):
    """Notebook-like content pages with dedicated, theme-native close buttons.

    Page IDs are the immutable Tk widget paths returned by :meth:`tabs`, never
    the user-facing labels.  A close click delegates to ``close_command`` so a
    controller can save, cancel, or call :meth:`forget` when it is ready.
    """

    _MAX_TITLE_WIDTH = 360
    _TAB_HEIGHT = CONTROLS.height_lg
    _CLOSE_SIZE = CONTROLS.height_sm
    _CLOSE_PADDING = CONTROLS.gap
    _TAB_RADIUS = CONTROLS.gap
    _SCROLLBAR_HEIGHT = 8

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
        self._close_images = {
            name: ctk.CTkImage(
                light_image=_close_icon_image(COLOR_PAIRS[name][0]),
                dark_image=_close_icon_image(COLOR_PAIRS[name][1]),
                size=(12, 12),
            )
            for name in ("text", "danger")
        }

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
            height=self._SCROLLBAR_HEIGHT + 2,
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
            height=self._TAB_HEIGHT + CONTROLS.gap // 2,
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
            button_color=COLOR_PAIRS["muted_text"],
            button_hover_color=COLOR_PAIRS["primary"],
        )
        self._header_scrollbar.grid(row=0, column=0, sticky="ew", padx=CONTROLS.gap, pady=1)
        self._header_scrollbar.grid_remove()
        self._header_canvas.configure(xscrollcommand=self._on_xscroll)

        self._tab_row = ctk.CTkFrame(
            self._header_canvas,
            width=1,
            height=self._TAB_HEIGHT + CONTROLS.gap // 2,
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

        self._join_frame = ctk.CTkFrame(
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
            bg_color=COLOR_PAIRS["tab_border"],
            fg_color=COLOR_PAIRS["tab_border"],
        )
        self._content.grid(row=2, column=0, sticky="nsew")
        self._content.grid_rowconfigure(0, weight=1)
        self._content.grid_columnconfigure(0, weight=1)
        self._apply_aidas_theme()

    def add(self, *, text: str = "") -> ctk.CTkFrame:
        """Create, register, and return a new content page."""

        # The UI package also initializes while shared helpers import theme.
        from aidas.utils.ui_utils import HoverToolTip

        page = ctk.CTkFrame(
            self._content,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["surface"],
        )
        page.grid(row=0, column=0, sticky="nsew", padx=SHAPES.border_width, pady=(0, SHAPES.border_width))
        page_id = self._model.add(page, text)

        tab_frame = ctk.CTkFrame(
            self._tab_row,
            height=self._TAB_HEIGHT,
            corner_radius=self._TAB_RADIUS,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["tab_border"],
            bg_color=COLOR_PAIRS["sidebar"],
        )
        tab_frame.pack(
            side="left", anchor="s",
            padx=(CONTROLS.gap if len(self._model.order) == 1 else CONTROLS.gap // 2, 0),
            pady=(CONTROLS.gap // 2, 0),
        )
        tab_frame.pack_propagate(False)

        title_button = ctk.CTkButton(
            tab_frame,
            text=str(text),
            height=self._TAB_HEIGHT,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=0,
            border_spacing=CONTROLS.gap // 2,
            hover=False,
            hover_color=COLOR_PAIRS["surface_subtle"],
            font=self._tab_font,
            anchor="w",
            command=lambda tab_id=page_id: self._select_id(tab_id, notify=True),
        )
        title_button.pack(side="left", fill="y", padx=(CONTROLS.gap, 0), pady=CONTROLS.gap // 2)
        # Keep a long label inside its capped title area, leaving room for
        # the close button even when the header needs horizontal scrolling.
        title_button.grid_propagate(False)

        close_button = ctk.CTkButton(
            tab_frame,
            text="",
            image=self._close_images["text"],
            anchor="center",
            width=self._CLOSE_SIZE,
            height=self._CLOSE_SIZE,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=0,
            fg_color=COLOR_PAIRS["surface"],
            hover_color=COLOR_PAIRS["danger_soft"],
            text_color=COLOR_PAIRS["text"],
            text_color_disabled=COLOR_PAIRS["disabled_text"],
            command=lambda tab_id=page_id: self._request_close(tab_id),
        )
        close_button.pack(
            side="left", padx=(0, self._CLOSE_PADDING),
            pady=(self._TAB_HEIGHT - self._CLOSE_SIZE) // 2,
        )
        HoverToolTip(close_button, "Close tab")

        bottom_join = ctk.CTkFrame(
            tab_frame, height=self._TAB_RADIUS, corner_radius=0, border_width=0,
            bg_color=COLOR_PAIRS["tab_selected"],
            fg_color=COLOR_PAIRS["tab_selected"],
        )
        self._tab_widgets[page_id] = _TabWidgets(
            tab_frame, title_button, close_button,
            title_tooltip=HoverToolTip(title_button, str(text)),
            bottom_join=bottom_join,
        )
        self._size_tab(page_id)
        self._bind_horizontal_scroll(tab_frame)
        self._bind_horizontal_scroll(title_button)
        self._bind_horizontal_scroll(close_button)
        self._bind_horizontal_scroll(bottom_join)
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
        self.after_idle(self._sync_header_geometry)
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
        """Read or update tab options; ``text`` is the supported option."""

        page_id = self._resolve_page_id(tab)
        if option is not None:
            if options:
                raise TypeError("tab() cannot combine an option query with option updates")
            if option == "text":
                return self._model.labels[page_id]
            raise tk.TclError(f"Unsupported tab option: {option!r}")

        unknown = set(options) - {"text"}
        if unknown:
            name = sorted(unknown)[0]
            raise tk.TclError(f"Unsupported tab option: {name!r}")
        if "text" in options:
            text = str(options["text"])
            self._model.labels[page_id] = text
            widgets = self._tab_widgets[page_id]
            widgets.title.configure(text=text)
            if widgets.title_tooltip is not None:
                widgets.title_tooltip.text = text
            self._size_tab(page_id)
            self.after_idle(self._sync_header_geometry)
            return None
        return {"text": self._model.labels[page_id]}

    def forget(self, tab: object) -> None:
        """Remove and destroy a page, silently displaying a nearby survivor."""

        page_id = self._resolve_page_id(tab)
        page, replacement_id, was_selected = self._model.forget(page_id)
        widgets = self._tab_widgets.pop(page_id)
        self._cancel_tab_hover(widgets)
        widgets.frame.destroy()
        try:
            page.destroy()
        except tk.TclError:
            pass

        if was_selected and replacement_id is not None:
            self._show_selected_page(replacement_id)
        elif replacement_id is not None:
            self._style_tab(replacement_id, active=True)
        self.after_idle(self._sync_header_geometry)

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
        page.grid(row=0, column=0, sticky="nsew", padx=SHAPES.border_width, pady=(0, SHAPES.border_width))
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
        title_hover = COLOR_PAIRS["tab_selected_hover"] if active else COLOR_PAIRS["button_hover"]
        widgets.frame.configure(
            fg_color=background,
            border_color=COLOR_PAIRS["tab_selected_border"] if active else COLOR_PAIRS["tab_border"],
            border_width=SHAPES.border_width,
        )
        widgets.title.configure(
            fg_color=background,
            hover_color=title_hover,
            text_color=COLOR_PAIRS["tab_selected_text"] if active else COLOR_PAIRS["text"],
            text_color_disabled=COLOR_PAIRS["disabled_text"],
            font=self._selected_tab_font if active else self._tab_font,
        )
        if widgets.bottom_join is not None:
            widgets.bottom_join.configure(fg_color=background, bg_color=background)
            if active:
                widgets.bottom_join.place(x=SHAPES.border_width, rely=1, y=-self._TAB_RADIUS)
                widgets.close.tkraise()
            else:
                widgets.bottom_join.place_forget()
        self._set_close_hover(page_id, widgets.close_hovered)
        self._draw_content_join()

    @staticmethod
    def _tab_background(widgets: _TabWidgets):
        if widgets.active:
            return COLOR_PAIRS["tab_selected_hover" if widgets.hovered else "tab_selected"]
        return COLOR_PAIRS["button_hover" if widgets.hovered else "button"]

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
        widgets.close.configure(
            fg_color=COLOR_PAIRS["danger_soft"] if hovered else background,
            hover_color=COLOR_PAIRS["danger_soft"],
            text_color=COLOR_PAIRS["danger"] if hovered else COLOR_PAIRS["text"],
            image=self._close_images["danger" if hovered else "text"],
        )

    def _size_tab(self, page_id: str) -> None:
        widgets = self._tab_widgets[page_id]
        text = self._model.labels[page_id]
        try:
            measured = int(self._selected_tab_font.measure(text))
        except tk.TclError:
            measured = len(text) * max(7, TYPOGRAPHY.body_size // 2)
        title_width = min(
            self._MAX_TITLE_WIDTH,
            max(CONTROLS.height_lg * 2, measured + CONTROLS.padding_x * 2),
        )
        widgets.title.configure(width=title_width)
        frame_width = title_width + CONTROLS.gap + self._CLOSE_SIZE + self._CLOSE_PADDING
        widgets.frame.configure(width=frame_width)
        if widgets.bottom_join is not None:
            widgets.bottom_join.configure(width=frame_width - SHAPES.border_width * 2)

    def _draw_content_join(self, _event=None) -> None:
        """Open the panel's top border directly beneath the selected tab."""

        try:
            self._join_canvas.configure(background=COLORS.tab_border)
            self._join_canvas.delete("selected_join")
            widgets = self._tab_widgets.get(self._model.selected)
            if widgets is None:
                return
            left = widgets.frame.winfo_rootx() - self._join_canvas.winfo_rootx()
            right = left + widgets.frame.winfo_width()
            inset = max(1, round(widgets.title.winfo_x() / CONTROLS.gap))
            left = max(0, left + inset)
            right = min(self._join_canvas.winfo_width(), right - inset)
            if right <= left:
                return
            self._join_canvas.create_rectangle(
                left, 0, right, self._join_canvas.winfo_height(),
                fill=resolve_color(self._tab_background(widgets)), outline="", tags="selected_join",
            )
        except tk.TclError:
            pass

    def _sync_header_geometry(self, _event=None) -> None:
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
        self._cancel_header_hover()
        for widgets in self._tab_widgets.values():
            self._cancel_tab_hover(widgets)
        super().destroy()

    def _apply_aidas_theme(self) -> None:
        """Refresh the overflow canvas and retain selected and hovered styles."""

        try:
            self._header_canvas.configure(background=COLORS.sidebar)
        except tk.TclError:
            pass
        for name, icon in self._close_images.items():
            icon.configure(
                light_image=_close_icon_image(COLOR_PAIRS[name][0]),
                dark_image=_close_icon_image(COLOR_PAIRS[name][1]),
            )
        for page_id in self._model.order:
            self._style_tab(page_id, active=page_id == self._model.selected)
        self._draw_content_join()


__all__ = ["ClosableTabView"]
