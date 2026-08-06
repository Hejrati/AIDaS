"""Closable, theme-aware tabs built from public Tk and CustomTkinter widgets."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from typing import Callable

import customtkinter as ctk

from aidas.ui.theme import COLOR_PAIRS, COLORS, CONTROLS, SHAPES, TYPOGRAPHY


TabCallback = Callable[["ClosableTabView", ctk.CTkFrame], None]
_CLOSE_GLYPH = "\u00d7"


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


class ClosableTabView(ctk.CTkFrame):
    """Notebook-like content pages with dedicated, theme-native close buttons.

    Page IDs are the immutable Tk widget paths returned by :meth:`tabs`, never
    the user-facing labels.  A close click delegates to ``close_command`` so a
    controller can save, cancel, or call :meth:`forget` when it is ready.
    """

    _MAX_TITLE_WIDTH = 360

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

        self._tab_font = ctk.CTkFont(
            family=TYPOGRAPHY.family,
            size=TYPOGRAPHY.body_size,
            weight=TYPOGRAPHY.semibold_weight,
        )
        self._close_font = ctk.CTkFont(
            family=TYPOGRAPHY.family,
            size=CONTROLS.icon_size,
            weight=TYPOGRAPHY.normal_weight,
        )

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._header = ctk.CTkFrame(
            self,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["application"],
        )
        self._header.grid(row=0, column=0, sticky="ew")
        self._header.grid_columnconfigure(0, weight=1)

        self._header_canvas = tk.Canvas(
            self._header,
            height=CONTROLS.height_md + CONTROLS.gap // 2,
            borderwidth=0,
            highlightthickness=0,
            takefocus=False,
            xscrollincrement=max(1, CONTROLS.gap * 3),
        )
        self._header_canvas.grid(row=0, column=0, sticky="ew")

        self._header_scrollbar = ctk.CTkScrollbar(
            self._header,
            orientation="horizontal",
            height=max(8, CONTROLS.scrollbar_width - 2),
            command=self._header_canvas.xview,
            fg_color=COLOR_PAIRS["application"],
            button_color=COLOR_PAIRS["border_strong"],
            button_hover_color=COLOR_PAIRS["primary"],
        )
        self._header_scrollbar.grid(row=1, column=0, sticky="ew")
        self._header_scrollbar.grid_remove()
        self._header_canvas.configure(xscrollcommand=self._on_xscroll)

        self._tab_row = ctk.CTkFrame(
            self._header_canvas,
            width=1,
            height=CONTROLS.height_md + CONTROLS.gap // 2,
            corner_radius=0,
            border_width=0,
            bg_color=COLOR_PAIRS["application"],
            fg_color=COLOR_PAIRS["application"],
        )
        self._tab_row_window = self._header_canvas.create_window(
            0,
            0,
            anchor="nw",
            window=self._tab_row,
        )
        self._tab_row.bind("<Configure>", self._sync_header_geometry, add="+")
        self._header_canvas.bind("<Configure>", self._sync_header_geometry, add="+")
        self._bind_horizontal_scroll(self._header_canvas)

        self._content = ctk.CTkFrame(
            self,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["surface"],
        )
        self._content.grid(row=1, column=0, sticky="nsew")
        self._content.grid_rowconfigure(0, weight=1)
        self._content.grid_columnconfigure(0, weight=1)
        self._apply_aidas_theme()

    def add(self, *, text: str = "") -> ctk.CTkFrame:
        """Create, register, and return a new content page."""

        page = ctk.CTkFrame(
            self._content,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["surface"],
        )
        page.grid(row=0, column=0, sticky="nsew")
        page_id = self._model.add(page, text)

        tab_frame = ctk.CTkFrame(
            self._tab_row,
            height=CONTROLS.height_md,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=0,
        )
        tab_frame.pack(side="left", anchor="s", padx=(0, 1), pady=(CONTROLS.gap // 2, 0))
        tab_frame.pack_propagate(False)

        title_button = ctk.CTkButton(
            tab_frame,
            text=str(text),
            height=CONTROLS.height_md,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=0,
            hover_color=COLOR_PAIRS["surface_subtle"],
            font=self._tab_font,
            anchor="w",
            command=lambda tab_id=page_id: self._select_id(tab_id, notify=True),
        )
        title_button.pack(side="left", fill="y")

        close_button = ctk.CTkButton(
            tab_frame,
            text=_CLOSE_GLYPH,
            width=CONTROLS.height_sm,
            height=CONTROLS.height_md,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=0,
            hover_color=COLOR_PAIRS["surface_subtle"],
            text_color=COLOR_PAIRS["muted_text"],
            text_color_disabled=COLOR_PAIRS["disabled_text"],
            font=self._close_font,
            command=lambda tab_id=page_id: self._request_close(tab_id),
        )
        close_button.pack(side="left", fill="y")

        self._tab_widgets[page_id] = _TabWidgets(tab_frame, title_button, close_button)
        self._size_tab(page_id)
        self._bind_horizontal_scroll(tab_frame)
        self._bind_horizontal_scroll(title_button)
        self._bind_horizontal_scroll(close_button)

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
            self._tab_widgets[page_id].title.configure(text=text)
            self._size_tab(page_id)
            self.after_idle(self._sync_header_geometry)
            return None
        return {"text": self._model.labels[page_id]}

    def forget(self, tab: object) -> None:
        """Remove and destroy a page, silently displaying a nearby survivor."""

        page_id = self._resolve_page_id(tab)
        page, replacement_id, was_selected = self._model.forget(page_id)
        widgets = self._tab_widgets.pop(page_id)
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
        page.grid(row=0, column=0, sticky="nsew")
        page.tkraise()
        for candidate in self._model.order:
            self._style_tab(candidate, active=candidate == page_id)

    def _request_close(self, page_id: str) -> None:
        if page_id not in self._model.pages or self.close_command is None:
            return
        self.close_command(self, self._model.pages[page_id])

    def _style_tab(self, page_id: str, *, active: bool) -> None:
        widgets = self._tab_widgets.get(page_id)
        if widgets is None:
            return
        background = COLOR_PAIRS["surface"] if active else COLOR_PAIRS["application"]
        foreground = COLOR_PAIRS["text"] if active else COLOR_PAIRS["muted_text"]
        title_hover = COLOR_PAIRS["surface"] if active else COLOR_PAIRS["surface_subtle"]
        widgets.frame.configure(fg_color=background)
        widgets.title.configure(
            fg_color=background,
            hover_color=title_hover,
            text_color=foreground,
            text_color_disabled=COLOR_PAIRS["disabled_text"],
        )
        widgets.close.configure(
            fg_color=background,
            hover_color=COLOR_PAIRS["danger_soft"],
        )

    def _size_tab(self, page_id: str) -> None:
        widgets = self._tab_widgets[page_id]
        text = self._model.labels[page_id]
        try:
            measured = int(self._tab_font.measure(text))
        except tk.TclError:
            measured = len(text) * max(7, TYPOGRAPHY.body_size // 2)
        title_width = min(
            self._MAX_TITLE_WIDTH,
            max(CONTROLS.height_lg, measured + CONTROLS.padding_x * 2),
        )
        widgets.title.configure(width=title_width)
        widgets.frame.configure(width=title_width + CONTROLS.height_sm)

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
            self._update_scrollbar_visibility(content_width > canvas_width + 1)
        except tk.TclError:
            pass

    def _on_xscroll(self, first: str, last: str) -> None:
        self._header_scrollbar.set(first, last)

    def _update_scrollbar_visibility(self, visible: bool) -> None:
        if visible == self._scrollbar_visible:
            return
        self._scrollbar_visible = visible
        if visible:
            self._header_scrollbar.grid()
        else:
            self._header_scrollbar.grid_remove()
            self._header_canvas.xview_moveto(0.0)

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
            if tab_left < view_left:
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
        if not self._scrollbar_visible:
            return None
        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            delta = int(getattr(event, "delta", 0))
            units = -1 if delta > 0 else 1
        self._header_canvas.xview_scroll(units, "units")
        return "break"

    def _apply_aidas_theme(self) -> None:
        """Refresh the native overflow canvas after appearance changes."""

        try:
            self._header_canvas.configure(background=COLORS.application)
        except tk.TclError:
            pass


__all__ = ["ClosableTabView"]
