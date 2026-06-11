"""UI helpers shared across AIDaS Tkinter views."""

import base64
import os
import sys
import tkinter as tk
from tkinter import ttk


ASSET_DIR_NAME = "assets"
ICON_FOLDER = "glyphs-poly--folder.png"
ICON_HOME = "streamline-flex-color--home-2-flat.png"
ICON_REFRESH = "material-symbols-light--refresh-rounded.png"


def resource_path(relative_path):
    """Resolve a bundled-or-source resource path."""
    base_dir = getattr(sys, "_MEIPASS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    return os.path.join(base_dir, relative_path)


def asset_path(filename):
    """Return the absolute path to an asset file."""
    return resource_path(os.path.join(ASSET_DIR_NAME, filename))


def remember_image(owner, image):
    """Keep a Tk image alive for as long as `owner` exists."""
    refs = getattr(owner, "_ui_image_refs", None)
    if refs is None:
        refs = []
        setattr(owner, "_ui_image_refs", refs)
    refs.append(image)
    return image


def load_ui_icon(owner, filename):
    """Load a PhotoImage from assets and retain it on `owner`."""
    path = asset_path(filename)
    try:
        image = tk.PhotoImage(file=path)
    except tk.TclError:
        with open(path, "rb") as handle:
            data = base64.b64encode(handle.read())
        image = tk.PhotoImage(data=data)
    return remember_image(owner, image)


def icon_button(parent, owner, icon_filename, command, *, tooltip=None, **button_options):
    """Create the flat icon-only Tk button style used by the Step 1 sidebar."""
    icon = load_ui_icon(owner, icon_filename)
    options = {
        "image": icon,
        "command": command,
        "bd": 0,
        "relief": "flat",
        "highlightthickness": 0,
        "cursor": "hand2",
    }
    options.update(button_options)
    button = tk.Button(parent, **options)
    button.image = icon
    if tooltip:
        HoverToolTip(button, tooltip)
    return button


def directory_row(
    parent,
    owner,
    textvariable,
    browse_command,
    *,
    home_command=None,
    refresh_command=None,
    browse_tooltip="Browse folder",
    home_tooltip="Reset folder",
    refresh_tooltip="Refresh",
):
    """Create a Step-1-style directory entry row with shared icon buttons."""
    row = ttk.Frame(parent)
    entry = ttk.Entry(row, textvariable=textvariable)
    entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

    buttons = {
        "browse": icon_button(row, owner, ICON_FOLDER, browse_command, tooltip=browse_tooltip),
        "home": None,
        "refresh": None,
    }
    buttons["browse"].pack(side="left", padx=(0, 4))

    if home_command is not None:
        buttons["home"] = icon_button(row, owner, ICON_HOME, home_command, tooltip=home_tooltip)
        buttons["home"].pack(side="right", padx=(4, 4))

    if refresh_command is not None:
        buttons["refresh"] = icon_button(row, owner, ICON_REFRESH, refresh_command, tooltip=refresh_tooltip)
        buttons["refresh"].pack(side="right")

    return row, entry, buttons


def build_app_menu(root, *, themes, current_theme, set_theme_command, browse_sdb_command, about_command):
    """Build the shared application menu bar."""
    menubar = tk.Menu(root)

    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Browse SDB Directory", command=browse_sdb_command)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.destroy, accelerator="Alt+F4")
    menubar.add_cascade(label="File", menu=file_menu)

    theme_menu = tk.Menu(menubar, tearoff=0)
    for theme in themes:
        prefix = "* " if theme == current_theme else ""
        theme_menu.add_command(label=f"{prefix}{theme.capitalize()}", command=lambda t=theme: set_theme_command(t))

    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_cascade(label="Theme", menu=theme_menu)
    help_menu.add_command(label="About", command=about_command)
    menubar.add_cascade(label="Help", menu=help_menu)

    root.config(menu=menubar)
    return menubar


class NativeNumericSpinbox(tk.Frame):
    """Excel-style numeric entry with tangent up/down arrow buttons."""

    FONT = ("Segoe UI", 9)
    BUTTON_WIDTH = 15

    def __init__(
        self,
        parent,
        textvariable,
        *,
        width=8,
        step=1,
        minimum=0,
        maximum=10_000_000,
        validatecommand=None,
    ):
        super().__init__(
            parent,
            bd=1,
            relief="solid",
            highlightthickness=0,
            bg="#ffffff",
        )
        self.var = textvariable
        self.step = step
        self.minimum = minimum
        self.maximum = maximum

        entry_options = {
            "textvariable": self.var,
            "width": width,
            "justify": "center",
            "font": self.FONT,
            "relief": "flat",
            "bd": 0,
            "highlightthickness": 0,
            "bg": "#ffffff",
        }
        if validatecommand is not None:
            entry_options["validate"] = "key"
            entry_options["validatecommand"] = validatecommand

        self.entry = tk.Entry(self, **entry_options)
        self.entry.pack(side="left", fill="both", expand=True, padx=(3, 1), pady=1, ipady=1)
        self.spinbox = self.entry

        self.button_stack = tk.Frame(self, bd=0, highlightthickness=0, bg="#d2d2d2", width=self.BUTTON_WIDTH)
        self.button_stack.pack(side="right", fill="y")
        self.button_stack.pack_propagate(False)

        self.button_canvas = tk.Canvas(
            self.button_stack,
            width=self.BUTTON_WIDTH,
            highlightthickness=0,
            bd=0,
            bg="#eeeeee",
            cursor="hand2",
        )
        self.button_canvas.pack(fill="both", expand=True)
        self.button_canvas.bind("<Configure>", self._draw_buttons, add="+")
        self.button_canvas.bind("<ButtonPress-1>", self._on_button_press, add="+")
        self.button_canvas.bind("<ButtonRelease-1>", self._on_button_release, add="+")
        self._button_state = "normal"
        self._pressed_half = None

    def _draw_buttons(self, _event=None):
        canvas = self.button_canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        mid = max(1, height // 2)
        disabled = self._button_state == "disabled"
        pressed_up = self._pressed_half == "up"
        pressed_down = self._pressed_half == "down"
        fill_up = "#dedede" if pressed_up else "#eeeeee"
        fill_down = "#dedede" if pressed_down else "#eeeeee"
        outline = "#a7a7a7"
        arrow = "#777777" if disabled else "#202020"

        canvas.delete("all")
        canvas.create_rectangle(0, 0, width - 1, mid, fill=fill_up, outline=outline)
        canvas.create_rectangle(0, mid, width - 1, height - 1, fill=fill_down, outline=outline)

        split = height / 2
        top_center = split / 2
        bottom_center = split + ((height - split) / 2)
        half_w = min(4, max(3, width / 4))
        half_h = min(2.6, max(2.0, (height / 2) * 0.22))
        cx = (width - 1) / 2

        def centered_triangle(center_y, direction):
            # The centroid of an isosceles triangle is 1/3 from the base, so
            # offset the bounding box center to make the icon look symmetric.
            bbox_center_y = center_y + (half_h / 3 if direction == "up" else -half_h / 3)
            if direction == "up":
                return (
                    cx,
                    bbox_center_y - half_h,
                    cx - half_w,
                    bbox_center_y + half_h,
                    cx + half_w,
                    bbox_center_y + half_h,
                )
            return (
                cx,
                bbox_center_y + half_h,
                cx - half_w,
                bbox_center_y - half_h,
                cx + half_w,
                bbox_center_y - half_h,
            )

        canvas.create_polygon(*centered_triangle(top_center, "up"), fill=arrow, outline=arrow)
        canvas.create_polygon(*centered_triangle(bottom_center, "down"), fill=arrow, outline=arrow)

    def _button_half(self, event):
        return "up" if event.y < max(1, self.button_canvas.winfo_height() // 2) else "down"

    def _on_button_press(self, event):
        if self._button_state == "disabled":
            return "break"
        self._pressed_half = self._button_half(event)
        self._draw_buttons()
        return "break"

    def _on_button_release(self, event):
        if self._button_state == "disabled":
            return "break"
        pressed = self._pressed_half
        released = self._button_half(event)
        self._pressed_half = None
        self._draw_buttons()
        if pressed == released:
            self._step(self.step if released == "up" else -self.step)
        return "break"

    def _step(self, delta):
        try:
            current = int(float(self.var.get()))
        except (TypeError, ValueError):
            current = self.minimum if delta > 0 else self.maximum
        next_value = max(self.minimum, min(self.maximum, current + int(delta)))
        self.var.set(str(next_value))

    def configure(self, cnf=None, **kwargs):
        if cnf is None and not kwargs:
            return self.entry.configure()
        options = {}
        if isinstance(cnf, dict):
            options.update(cnf)
        elif cnf is not None:
            return super().configure(cnf)
        options.update(kwargs)

        state = options.pop("state", None)
        result = super().configure(**options) if options else None
        if state is not None:
            self.entry.configure(state=state)
            self._button_state = state
            self._pressed_half = None
            self.button_canvas.configure(cursor="" if state == "disabled" else "hand2")
            self._draw_buttons()
        return result

    config = configure

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

    SIDEBAR_WIDTH = 380
    SIDEBAR_TEXT_WRAP = 350
    SECTION_PADDING = 8
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

        if sidebar_width is None:
            sidebar_width = self.SIDEBAR_WIDTH
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

    def add_sidebar_section(self, title, *, padding=None, **pack_options):
        """Add a collapsible section to the standard sidebar."""
        if padding is None:
            padding = self.SECTION_PADDING
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
