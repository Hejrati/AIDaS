"""UI helpers shared across AIDaS Tkinter views."""

import tkinter as tk
from tkinter import ttk


def apply_app_icon_to(window):
    """Apply the application's icon to `window` when available.

    The function looks up the top-level/root window and, if it has either
    a stored PhotoImage reference (`_icon_image_ref`) or an ICO path
    (`_icon_ico_path`), applies that icon to the provided `window`.

    This is intentionally forgiving: it ignores any exceptions so callers
    needn't wrap calls in try/except.
    """
    try:
        root = window
        while getattr(root, "master", None):
            root = root.master
    except Exception:
        try:
            root = window.winfo_toplevel()
        except Exception:
            return

    img = getattr(root, "_icon_image_ref", None)
    if img:
        try:
            window.iconphoto(True, img)
            return
        except Exception:
            pass

    ico = getattr(root, "_icon_ico_path", None)
    if ico:
        try:
            window.iconbitmap(ico)
        except Exception:
            pass


class HoverToolTip:
    """Small hover tooltip for Tk and ttk widgets."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.widget.bind("<Enter>", self._show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self.tipwindow is not None:
            return

        x = self.widget.winfo_pointerx() + 12
        y = self.widget.winfo_pointery() + 12
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        # Apply shared helper so tooltip windows use the app icon when shown
        try:
            apply_app_icon_to(tw)
        except Exception:
            pass
        ttk.Label(
            tw,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            padding=(6, 3),
        ).pack()
        self.tipwindow = tw

    def _hide(self, _event=None):
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


class ScrollableSidebar(ttk.Frame):
    """A vertical sidebar whose content can be scrolled with the mouse."""

    def __init__(self, parent, *, width=None):
        super().__init__(parent)

        canvas_options = {"highlightthickness": 0, "bd": 0}
        if width is not None:
            canvas_options["width"] = width

        self.canvas = tk.Canvas(self, **canvas_options)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)
        self._content_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self._middle_drag_active = False
        self._middle_drag_target = self.canvas
        self._active_nested_scroll = None
        self._refreshing = False

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.content.bind("<Configure>", self._on_content_configure, add="+")
        self.canvas.bind("<Configure>", self._on_canvas_configure, add="+")

        # Mouse-wheel events are delivered to the widget under the pointer, so
        # bind at "all" and only act when the pointer is inside this sidebar.
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self.bind_all("<ButtonPress-2>", self._on_middle_press, add="+")
        self.bind_all("<B2-Motion>", self._on_middle_drag, add="+")
        self.bind_all("<ButtonRelease-2>", self._on_middle_release, add="+")
        self.bind_all("<ButtonPress-1>", self._on_primary_press, add="+")

    def _on_content_configure(self, _event=None):
        self.refresh_scrollregion()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._content_window, width=event.width)
        self.refresh_scrollregion(viewport_height=event.height)

    def refresh_scrollregion(self, *, viewport_height=None, settle=False):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            if settle:
                self.update_idletasks()
            bbox = self._sync_content_geometry(viewport_height)
            self.canvas.configure(scrollregion=bbox)
            self._clamp_yview(bbox)
        finally:
            self._refreshing = False

    def queue_refresh(self):
        self.after_idle(lambda: self.refresh_scrollregion(settle=True))
        self.after(20, lambda: self.refresh_scrollregion(settle=True))
        self.after(120, lambda: self.refresh_scrollregion(settle=True))

    def _clamp_yview(self, bbox):
        viewport_height = max(1, self.canvas.winfo_height())
        region_top = int(bbox[1])
        region_bottom = int(bbox[3])
        region_height = max(1, region_bottom - region_top)
        if region_height <= viewport_height:
            self.canvas.yview_moveto(0.0)
            return

        top = self.canvas.canvasy(0)
        max_top = region_bottom - viewport_height
        if top < region_top:
            self.canvas.yview_moveto(0.0)
        elif top > max_top:
            self.canvas.yview_moveto((max_top - region_top) / region_height)

    def _sync_content_geometry(self, viewport_height=None):
        if viewport_height is None:
            viewport_height = self.canvas.winfo_height()
        viewport_width = max(1, self.canvas.winfo_width())
        content_height = max(self.content.winfo_reqheight(), viewport_height)
        self.canvas.itemconfigure(
            self._content_window,
            width=viewport_width,
            height=content_height,
        )
        return (0, 0, viewport_width, content_height)

    def _contains_pointer(self, widget=None):
        widget = widget or self
        try:
            if not self.winfo_ismapped():
                return False
            pointer_x = self.winfo_pointerx()
            pointer_y = self.winfo_pointery()
            root_x = widget.winfo_rootx()
            root_y = widget.winfo_rooty()
            return (
                root_x <= pointer_x < root_x + widget.winfo_width()
                and root_y <= pointer_y < root_y + widget.winfo_height()
            )
        except tk.TclError:
            return False

    def _contains_scroll_area_pointer(self):
        return self._contains_pointer(getattr(self, "master", None) or self)

    @staticmethod
    def _is_descendant(widget, ancestor):
        while widget is not None:
            if widget is ancestor:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _nested_scroll_owner(self, widget):
        if widget is None or not self._is_descendant(widget, self.content):
            return None
        while widget is not None and widget is not self:
            if isinstance(widget, (tk.Listbox, tk.Text)):
                return widget
            if isinstance(widget, ttk.Treeview):
                return widget
            if isinstance(widget, tk.Canvas) and widget is not self.canvas:
                return widget
            widget = getattr(widget, "master", None)
        return None

    def _nested_scroll_is_active(self, owner):
        if owner is None:
            return False
        focus = self.focus_get()
        return (
            owner is self._active_nested_scroll
            or focus is owner
            or self._is_descendant(focus, owner)
        )

    def _event_widget_owns_scroll(self, widget):
        owner = self._nested_scroll_owner(widget)
        return self._nested_scroll_is_active(owner)

    def _on_primary_press(self, event):
        if not self._contains_scroll_area_pointer():
            return None
        owner = self._nested_scroll_owner(getattr(event, "widget", None))
        self._active_nested_scroll = owner
        if owner is not None:
            try:
                owner.focus_set()
            except tk.TclError:
                pass
        return None

    def _on_mousewheel(self, event):
        if not self._contains_pointer():
            return None
        if self._event_widget_owns_scroll(getattr(event, "widget", None)):
            return None

        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return None
            units = -1 if delta > 0 else 1
        self.canvas.yview_scroll(units, "units")
        return "break"

    def _pointer_canvas_xy(self):
        return (
            self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(),
            self.canvas.winfo_pointery() - self.canvas.winfo_rooty(),
        )

    @staticmethod
    def _pointer_widget_xy(widget):
        return (
            widget.winfo_pointerx() - widget.winfo_rootx(),
            widget.winfo_pointery() - widget.winfo_rooty(),
        )

    def _on_middle_press(self, event):
        if not self._contains_scroll_area_pointer():
            return None
        owner = self._nested_scroll_owner(getattr(event, "widget", None))
        if self._nested_scroll_is_active(owner) and hasattr(owner, "scan_mark"):
            self._middle_drag_target = owner
            x, y = self._pointer_widget_xy(owner)
        else:
            self._middle_drag_target = self.canvas
            x, y = self._pointer_canvas_xy()

        self._middle_drag_active = True
        self._middle_drag_target.scan_mark(x, y)
        try:
            self._middle_drag_target.configure(cursor="sb_v_double_arrow")
        except tk.TclError:
            pass
        return "break"

    def _on_middle_drag(self, _event):
        if not self._middle_drag_active:
            return None
        target = self._middle_drag_target
        x, y = self._pointer_widget_xy(target)
        target.scan_dragto(x, y, gain=1)
        return "break"

    def _on_middle_release(self, _event):
        if not self._middle_drag_active:
            return None
        self._middle_drag_active = False
        try:
            self._middle_drag_target.configure(cursor="")
        except tk.TclError:
            pass
        self._middle_drag_target = self.canvas
        self.queue_refresh()
        return "break"


class CollapsibleSection(ttk.Frame):
    """A titled section that can hide or show its child controls."""

    HEADER_FONT = ("TkDefaultFont", 9, "bold")
    INDICATOR_FONT = ("TkDefaultFont", 12, "bold")
    HEADER_HEIGHT = 26

    def __init__(self, parent, title, *, padding=3, expanded=True):
        super().__init__(parent)
        self.title = title
        self.expanded = bool(expanded)

        style = ttk.Style(self)
        self._header_bg = style.lookup("TFrame", "background") or "#f0f0f0"
        self._header_fill = style.lookup("TButton", "background") or "#f3f4f6"
        self._header_outline = style.lookup("TButton", "bordercolor") or "#9ca3af"
        self._header_text = style.lookup("TLabel", "foreground") or "#111111"

        self.header = tk.Canvas(
            self,
            height=self.HEADER_HEIGHT,
            highlightthickness=0,
            bd=0,
            bg=self._header_bg,
            cursor="hand2",
        )
        self.header.pack(fill="x")
        self.header.bind("<Button-1>", self._on_header_click, add="+")
        self.header.bind("<Configure>", lambda _event: self._draw_header(), add="+")

        self._body_container = ttk.Frame(self, relief="sunken", borderwidth=1, padding=2)
        self._body_container.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 7))
        self.body = ttk.Frame(self._body_container, padding=padding)
        self.body.pack(fill="both", expand=True)
        self._sync_header()

    def _on_header_click(self, _event):
        self.toggle()
        return "break"

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            self._body_container.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 7))
        else:
            self._body_container.pack_forget()
        self._sync_header()
        self._queue_sidebar_refresh()

    def _sync_header(self):
        self._draw_header()

    def _draw_header(self):
        width = max(1, self.header.winfo_width())
        height = max(self.HEADER_HEIGHT, self.header.winfo_height())

        self.header.delete("all")
        self.header.create_rectangle(
            1,
            1,
            width - 1,
            height - 1,
            fill=self._header_fill,
            outline=self._header_outline,
        )
        marker = "\u25be" if self.expanded else "\u25b8"
        self.header.create_text(
            13,
            height / 2,
            text=marker,
            fill=self._header_text,
            font=self.INDICATOR_FONT,
        )
        self.header.create_text(
            28,
            height / 2,
            text=self.title,
            fill=self._header_text,
            font=self.HEADER_FONT,
            anchor="w",
        )

    def _queue_sidebar_refresh(self):
        widget = self.master
        while widget is not None:
            queue_refresh = getattr(widget, "queue_refresh", None)
            if callable(queue_refresh):
                queue_refresh()
                return
            widget = getattr(widget, "master", None)


class SidebarStepFrame(ttk.Frame):
    """Standard left-sidebar/right-content layout for AIDaS step pages."""

    SECTION_PACK = {"fill": "x", "padx": (6, 8), "pady": 2}

    def build_standard_layout(
        self,
        *,
        sidebar_width=None,
        sidebar_pack=None,
        content_pack=None,
        status_var=None,
    ):
        """Create a shared step layout with `self.ctrl` and `self.content`.

        `self.ctrl` is the scrollable sidebar content frame. `self.content` is
        the main right-side work area.
        """
        self.main = ttk.Frame(self)
        self.main.pack(fill="both", expand=True)

        self.sidebar = ScrollableSidebar(self.main, width=sidebar_width)
        sidebar_options = {"side": "left", "fill": "y"}
        if sidebar_pack:
            sidebar_options.update(sidebar_pack)
        self.sidebar.pack(**sidebar_options)
        self.ctrl = self.sidebar.content

        self.content = ttk.Frame(self.main)
        content_options = {"side": "left", "fill": "both", "expand": True}
        if content_pack:
            content_options.update(content_pack)
        self.content.pack(**content_options)

        if status_var is not None:
            self.add_status_bar(status_var)

        return self.ctrl, self.content

    def add_sidebar_section(self, title, *, padding=3, **pack_options):
        """Add a collapsible section to the standard sidebar."""
        section = CollapsibleSection(self.ctrl, title, padding=padding)
        options = dict(self.SECTION_PACK)
        options.update(pack_options)
        section.pack(**options)
        return section

    def add_status_bar(self, status_var, *, parent=None):
        """Add a standard sunken status label."""
        container = parent if parent is not None else self
        label = ttk.Label(
            container,
            textvariable=status_var,
            relief="sunken",
            anchor="w",
            padding=3,
        )
        label.pack(side="bottom", fill="x")
        return label
