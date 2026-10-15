"""Diagnostic results drawn on one Tk surface, without moving child windows."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
import tkinter.font as tkfont

from PIL import Image, ImageTk


class ResultGridCanvas(tk.Canvas):
    """Retain a bounded set of image/text pages and scroll native canvas items."""

    CACHE_PAGES = 3

    def __init__(self, master, *, groups, preview_titles, load_page, preview_layout, on_zoom,
                 on_open_folder, palette, expand_icon, **kwargs):
        super().__init__(master, takefocus=True, **kwargs)
        self._groups = groups
        self._preview_titles = preview_titles
        self._page_count = len(groups)
        self._load_page = load_page
        self._preview_layout = preview_layout
        self._on_zoom = on_zoom
        self._on_open_folder = on_open_folder
        self._palette = palette
        self._expand_icon = expand_icon
        self._font = tkfont.Font(self, family="", size=10, weight="bold")
        self._link_font = tkfont.Font(self, family="", size=10, weight="bold", underline=True)
        self._pages = OrderedDict()
        self._resize_sources = {}
        self._page_size = None
        self._visible_pages = set()
        self._refresh_after_id = None
        self._wheel_after_id = None
        self._wheel_target = None
        self._focus_key = None
        self._hover_key = None
        self._hover_background = None
        self._hover_tooltip = None
        self._generation = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aidas-result-previews")
        self._render_future = None
        self._render_requests = OrderedDict()
        self._poll_after_id = None
        self.bind("<Configure>", self._resize_pages, add="+")
        self.bind("<MouseWheel>", self.scroll_mousewheel, add="+")
        self.bind("<Button-4>", self.scroll_mousewheel, add="+")
        self.bind("<Button-5>", self.scroll_mousewheel, add="+")
        self.bind("<Button-1>", self._on_click, add="+")
        self.bind("<Motion>", self._on_motion, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<Tab>", self._on_tab)
        self.bind("<Shift-Tab>", lambda event: self._on_tab(event, reverse=True))
        self.bind("<ISO_Left_Tab>", lambda event: self._on_tab(event, reverse=True))
        self.bind("<Return>", self._activate_focused)
        self.bind("<space>", self._activate_focused)
        self.bind("<FocusIn>", self._on_focus_in, add="+")
        self.bind("<FocusOut>", lambda _event: self.delete("keyboard-focus"), add="+")

    def _resize_pages(self, event):
        size = (event.width, event.height)
        if size == self._page_size or min(size) <= 1:
            return
        self._dismiss_hover()
        self._generation += 1
        self._render_requests.clear()
        self._page_size = size
        self._resize_sources.update((index, page["data"]) for index, page in self._pages.items()
                                    if page["data"] is not None)
        self._resize_sources = dict(list(self._resize_sources.items())[-self.CACHE_PAGES:])
        self.delete("all")
        self._pages.clear()
        self._visible_pages.clear()
        self.configure(scrollregion=(0, 0, event.width, self._extent()))
        self._queue_refresh()

    def _extent(self):
        height = self._page_size[1] if self._page_size else self.winfo_height()
        return max(1, (height - 20) * self._page_count + 20)

    def _queue_refresh(self):
        if self._refresh_after_id is None:
            self._refresh_after_id = self.after_idle(self._refresh_pages)

    def _refresh_pages(self):
        self._refresh_after_id = None
        if self._page_size is None or not self._page_count:
            return
        height = self._page_size[1]
        page_height = max(1, height - 20)
        first = max(0, min(self._page_count - 1, int((self.canvasy(0) + 2) // page_height)))
        last = max(first, min(self._page_count - 1, int((self.canvasy(height - 1) - 8) // page_height)))
        visible = set(range(first, last + 1))
        self._visible_pages = visible
        self._render_requests = OrderedDict(
            (index, request) for index, request in self._render_requests.items() if index in visible
        )
        for index in range(first, last + 1):
            self._ensure_page(index)
        while len(self._pages) > max(self.CACHE_PAGES, len(visible)):
            index = next(index for index in self._pages if index not in visible)
            self.delete(self._pages.pop(index)["tag"])
            self._render_requests.pop(index, None)
        self._draw_focus()
        self._start_render()

    def _ensure_page(self, index):
        if index not in self._pages:
            group = self._groups[index]
            folders = (
                [(name.title(), folder) for name, folder in group["folders"].items()]
                if group["sided"] else [(None, group["folders"]["result"])]
            )
            pairs = [dict(folder=folder, title=title) for title, folder in folders]
            self._pages[index] = self._paint_page(index, pairs)
        page = self._pages[index]
        running = self._render_future and self._render_future[:2] == (index, self._generation)
        if not page["ready"] and index not in self._render_requests and not running:
            targets = [(preview["pair_index"], preview["column"], preview["target"])
                       for preview in page["previews"]]
            self._render_requests[index] = (
                self._generation, targets, self._resize_sources.pop(index, None),
            )
        self._pages.move_to_end(index)
        return self._pages[index]

    @staticmethod
    def _prepare_page(index, targets, data, load_page, preview_layout, background):
        """Decode and resize with Pillow only; this worker never touches Tk."""
        if data is None:
            data = load_page(index)
        prepared = []
        for pair_index, column, (_x, _y, width, height) in targets:
            pair = data[pair_index]
            title, source, bounds = pair["previews"][column]
            layout = preview_layout(source.size, bounds, pair["margins"], (width, height), pair["aspect"])
            left, top, fitted_width, fitted_height = layout
            rgb = source if source.mode == "RGB" else source.convert("RGB")
            bitmap = Image.new("RGB", (width, height), background)
            bitmap.paste(rgb.resize((fitted_width, fitted_height), Image.Resampling.LANCZOS), (left, top))
            prepared.append(dict(image=source, bitmap=bitmap, layout=layout, bounds=bounds,
                                 margins=pair["margins"], aspect=pair["aspect"], result_title=title,
                                 side_title=pair["title"]))
        return data, prepared

    def _start_render(self):
        if self._render_future is None and self._render_requests:
            index, (generation, targets, data) = self._render_requests.popitem(last=False)
            future = self._executor.submit(
                self._prepare_page, index, targets, data, self._load_page,
                self._preview_layout, self._palette["surface"],
            )
            self._render_future = (index, generation, future)
        if self._render_future is not None and self._poll_after_id is None:
            self._poll_after_id = self.after(12, self._poll_render)

    def _poll_render(self):
        self._poll_after_id = None
        if self._render_future is not None:
            index, generation, future = self._render_future
            if future.done():
                self._render_future = None
                try:
                    data, prepared = future.result()
                except Exception as exc:
                    if generation == self._generation and index in self._pages:
                        page = self._pages[index]
                        page["ready"] = True
                        page["error"] = str(exc)
                        for item in page["loading"]:
                            self.itemconfigure(item, text=f"Could not load results:\n{exc}")
                else:
                    if generation == self._generation and index in self._pages:
                        self._install_previews(index, data, prepared)
                    elif index in self._render_requests:
                        # Resize invalidates thumbnails, but decoded originals
                        # are independent of viewport geometry.
                        current, targets, _source = self._render_requests[index]
                        self._render_requests[index] = (current, targets, data)
        self._start_render()

    def _install_previews(self, index, data, prepared):
        page = self._pages[index]
        page["data"] = data
        for item in page["loading"]:
            self.delete(item)
        page["loading"].clear()
        for preview, pixels in zip(page["previews"], prepared):
            self._paint_preview(page, index, preview, pixels)
        page["actions"].sort(key=lambda action: (
            (-1, -1, 0) if len(action["key"]) == 2 else
            (action["key"][1], -1, 0) if action["kind"] == "folder" else
            (action["key"][1], action["key"][2], action["kind"] == "expand")
        ))
        page["items"] = list(self.find_withtag(page["tag"]))
        page["ready"] = True
        self._draw_focus()

    def _add_action(self, page, bounds, kind, callback, key, tooltip="", **kwargs):
        page["actions"].append(dict(bounds=bounds, kind=kind, callback=callback,
                                    key=key, tooltip=tooltip, **kwargs))

    def _link(self, page, x, y, width, folder, prefix, key):
        text = f"{prefix} — {folder}" if prefix else str(folder)
        item = self.create_text(x, y, anchor="nw", width=max(1, width), text=text,
                                font=self._link_font, fill=self._palette["link"], tags=page["tag"])
        bounds = self.bbox(item)
        mask = self.create_rectangle(bounds[0] - 2, bounds[1], bounds[2] + 2, bounds[3],
                                     fill=self._palette["surface"], outline="", tags=page["tag"])
        self.tag_lower(mask, item)
        self._add_action(page, bounds, "folder", lambda: self._on_open_folder(folder), key)
        return bounds[3]

    def _paint_page(self, index, pairs):
        width, height = self._page_size
        x, y = 12, 8 + index * max(1, height - 20)
        page_width, page_height = max(1, width - 24), max(1, height - 30)
        tag = f"result-page-{index}"
        page = {"tag": tag, "data": None, "previews": [], "actions": [], "ready": False, "loading": []}
        group = self._groups[index]
        line_height = self._font.metrics("linespace")
        self.create_rectangle(x, y + line_height / 2, x + page_width, y + page_height,
                              outline=self._palette["border"], tags=tag)
        top = self._link(page, x + 3, y, page_width - 24, group["root"], None, (index, "root")) + 6
        bottom = y + page_height - 10
        pair_height = max(1, (bottom - top) / max(1, len(pairs)))
        for pair_index, pair in enumerate(pairs):
            pair_top = top + pair_index * pair_height
            pair_bottom = top + (pair_index + 1) * pair_height - 5
            left, right = x + 10, x + page_width - 10
            if pair["title"]:
                self.create_rectangle(left, pair_top + line_height / 2, right, pair_bottom,
                                      outline=self._palette["border"], tags=tag)
                pair_top = self._link(page, left + 3, pair_top, right - left - 24,
                                      pair["folder"], pair["title"], (index, pair_index, "folder")) + 4
                left, right = left + 5, right - 5
                pair_bottom -= 5
            column_width = (right - left) / max(1, len(self._preview_titles))
            for column, title in enumerate(self._preview_titles):
                column_left = left + column * column_width
                self.create_text(column_left + column_width / 2, pair_top, anchor="n",
                                 text=title, font=self._font, fill=self._palette["text"], tags=tag)
                target = (round(column_left + 5), round(pair_top + line_height + 9),
                          max(1, round(column_width - 10)),
                          max(1, round(pair_bottom - pair_top - line_height - 14)))
                if target[1] + target[3] > pair_bottom:
                    continue
                page["previews"].append(dict(target=target, pair_index=pair_index, column=column,
                                             image=None, photo=None, item=None))
                page["loading"].append(self.create_text(
                    target[0] + target[2] / 2, target[1] + target[3] / 2,
                    text="Loading results…", fill=self._palette["text"], tags=tag,
                ))
        page["items"] = list(self.find_withtag(tag))
        return page

    def _paint_preview(self, page, index, preview, pixels):
        x, y, width, height = preview["target"]
        source, bounds, layout = pixels["image"], pixels["bounds"], pixels["layout"]
        title = pixels["result_title"]
        left, top, fitted_width, fitted_height = layout
        photo = ImageTk.PhotoImage(pixels["bitmap"], master=self)
        item = self.create_image(x, y, image=photo, anchor="nw", tags=page["tag"])
        zoom_title = " — ".join(part for part in (self._groups[index]["label"], pixels["side_title"], title) if part)
        callback = lambda: self._on_zoom(source, zoom_title)
        key = (index, preview["pair_index"], preview["column"])
        self._add_action(page, (x, y, x + width, y + height), "preview", callback,
                         (*key, "preview"), f"Click to maximize {zoom_title}.")
        preview.update(image=source, photo=photo, item=item, layout=layout, bounds=bounds,
                       margins=pixels["margins"], aspect=pixels["aspect"], title=zoom_title)
        panel_left = left + (bounds[0] * fitted_width + source.width - 1) // source.width
        panel_top = top + (bounds[1] * fitted_height + source.height - 1) // source.height
        panel_right = left + bounds[2] * fitted_width // source.width
        panel_bottom = top + bounds[3] * fitted_height // source.height
        button_size, inset = 32, 8
        if panel_right - panel_left >= button_size + 2 * inset and panel_bottom - panel_top >= button_size + 2 * inset:
            bx, by = x + panel_left + inset, y + panel_bottom - inset - button_size
            button = self.create_rectangle(bx, by, bx + button_size, by + button_size,
                                           fill=self._palette["button"], outline=self._palette["border"],
                                           tags=page["tag"])
            self.create_image(bx + button_size / 2, by + button_size / 2,
                              image=self._expand_icon, tags=page["tag"])
            self._add_action(page, (bx, by, bx + button_size, by + button_size), "expand",
                             callback, (*key, "expand"), f"Maximize {title} preview.", background_item=button)

    def _action_at(self, x, y):
        x, y = self.canvasx(x), self.canvasy(y)
        for index in sorted(self._visible_pages, reverse=True):
            for action in reversed(self._pages[index]["actions"]):
                left, top, right, bottom = action["bounds"]
                if left <= x <= right and top <= y <= bottom:
                    return action
        return None

    def _on_click(self, event):
        self._dismiss_hover()
        action = self._action_at(event.x, event.y)
        if action is not None:
            self._focus_action(action)
            action["callback"]()
        return "break"

    def _on_motion(self, event):
        action = self._action_at(event.x, event.y)
        key = action["key"] if action else None
        if key != self._hover_key:
            self._clear_hover_background()
            self._hover_key = key
            self.configure(cursor="hand2" if action else "")
            if action and action.get("background_item") is not None:
                self._hover_background = action["background_item"]
                self.itemconfigure(self._hover_background, fill=self._palette["button_hover"])
            if self._hover_tooltip is not None:
                self._hover_tooltip.text = action["tooltip"] if action else ""
                if action and action["tooltip"]:
                    self._hover_tooltip._schedule_show()

    def _on_leave(self, _event=None):
        self._dismiss_hover()

    def _dismiss_hover(self):
        self._clear_hover_background()
        if self._hover_key is not None:
            self.configure(cursor="")
        self._hover_key = None
        if self._hover_tooltip is not None:
            self._hover_tooltip._hide()

    def _clear_hover_background(self):
        if self._hover_background is not None and self.type(self._hover_background):
            self.itemconfigure(self._hover_background, fill=self._palette["button"])
        self._hover_background = None

    def _tooltip_bounds(self):
        for page in self._pages.values():
            for action in page["actions"]:
                if action["key"] == self._hover_key:
                    left, top, right, bottom = action["bounds"]
                    x, y = self.canvasx(0), self.canvasy(0)
                    return (max(0, left - x), max(0, top - y),
                            min(self.winfo_width(), right - x), min(self.winfo_height(), bottom - y))
        return None

    def _focused_action(self):
        for page in self._pages.values():
            for action in page["actions"]:
                if action["key"] == self._focus_key:
                    return action
        return None

    def _focus_action(self, action):
        self._focus_key = action["key"]
        self.focus_set()
        _, top, _, bottom = action["bounds"]
        if top < self.canvasy(0):
            self.yview_moveto(max(0, top - 8) / self._extent())
        elif bottom > self.canvasy(self.winfo_height()):
            self.yview_moveto(max(0, bottom + 8 - self.winfo_height()) / self._extent())
        self._draw_focus()

    def _draw_focus(self):
        self.delete("keyboard-focus")
        action = self._focused_action()
        if action is not None and self.focus_get() is self:
            left, top, right, bottom = action["bounds"]
            self.create_rectangle(left - 1, top - 1, right + 1, bottom + 1,
                                  outline=self._palette["link"], dash=(2, 2), tags="keyboard-focus")

    def _on_focus_in(self, _event=None):
        if self._focused_action() is None and self._visible_pages:
            self._focus_key = self._pages[min(self._visible_pages)]["actions"][0]["key"]
        self._draw_focus()

    def _on_tab(self, _event=None, reverse=False):
        action = self._focused_action()
        if action is None:
            if self._visible_pages:
                actions = self._pages[max(self._visible_pages) if reverse else min(self._visible_pages)]["actions"]
                self._focus_action(actions[-1] if reverse else actions[0])
            return "break"
        index = action["key"][0]
        actions = self._pages[index]["actions"]
        position = actions.index(action) + (-1 if reverse else 1)
        if 0 <= position < len(actions):
            self._focus_action(actions[position])
        else:
            index += -1 if reverse else 1
            if 0 <= index < self._page_count:
                actions = self._ensure_page(index)["actions"]
                self._focus_action(actions[-1] if reverse else actions[0])
                self._queue_refresh()
            else:
                target = self.tk_focusPrev() if reverse else self.tk_focusNext()
                if target is not None:
                    target.focus_set()
        return "break"

    def _activate_focused(self, _event=None):
        self._dismiss_hover()
        action = self._focused_action()
        if action is not None:
            action["callback"]()
        return "break"

    def yview(self, *args):
        if args:
            self._dismiss_hover()
            self._finish_pending_wheel()
        result = super().yview(*args)
        if args:
            self._queue_refresh()
        return result

    def yview_scroll(self, number, what):
        self._dismiss_hover()
        self._finish_pending_wheel()
        super().yview_scroll(number, what)
        self._queue_refresh()

    def yview_moveto(self, fraction):
        self._dismiss_hover()
        self._finish_pending_wheel()
        super().yview_moveto(fraction)
        self._queue_refresh()

    def _finish_pending_wheel(self):
        if self._wheel_after_id is not None:
            self.after_cancel(self._wheel_after_id)
            self._flush_wheel()

    def scroll_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        elif getattr(event, "delta", 0):
            units = -3 if event.delta > 0 else 3
        else:
            return "break"
        if self._hover_tooltip is not None:
            self._hover_tooltip._hide()
        first, last = self.yview()
        if last - first >= 1:
            return "break"
        extent = self._extent()
        increment = float(self.cget("yscrollincrement"))
        distance = units * increment if increment else units * 0.1 * self.winfo_height()
        start = first if self._wheel_target is None else self._wheel_target
        offset = int(round(start * extent) + distance)
        self._wheel_target = max(0.0, min(1.0 - (last - first), offset / extent))
        if self._wheel_after_id is None:
            self._wheel_after_id = self.after_idle(self._flush_wheel)
        return "break"

    def _flush_wheel(self):
        self._wheel_after_id = None
        target, self._wheel_target = self._wheel_target, None
        self.yview_moveto(target)

    def destroy(self):
        self._generation += 1
        self._render_requests.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._render_future = None
        self._dismiss_hover()
        for name in ("_refresh_after_id", "_wheel_after_id", "_poll_after_id"):
            job = getattr(self, name)
            if job is not None:
                self.after_cancel(job)
                setattr(self, name, None)
        super().destroy()
        self._pages.clear()
        self._resize_sources.clear()
        self._load_page = self._on_zoom = self._on_open_folder = None
