"""UI helpers shared across AIDaS Tkinter views."""

import base64
import math
import os
import sys
import tkinter as tk
from tkinter import ttk

import customtkinter as ctk
from PIL import Image, ImageChops, ImageOps

from aidas.core.display import work_area_bounds
from aidas.ui.theme import (
    COLOR_PAIRS,
    CONTROLS,
    SHAPES,
    TYPOGRAPHY,
    configure_ttk_styles,
)
from aidas.utils.ui_layout import COLORS, LAYOUT, workspace_sidebar_width


ASSET_DIR_NAME = "assets"
ACTION_ICON_SIZE = 20
ACTION_ICON_FILES = {
    "cancel": "flat-color-icons--cancel.png",
    "clear": "flat-color-icons--empty-trash.png",
    "confirm": "flat-color-icons--checkmark.png",
    "folder": "flat-color-icons--folder.png",
    "home": "flat-color-icons--home.png",
    "next": "flat-color-icons--right.png",
    "opened_folder": "flat-color-icons--opened-folder.png",
    "package": "flat-color-icons--package.png",
    "previous": "flat-color-icons--left.png",
    "process": "flat-color-icons--process.png",
    "refresh": "flat-color-icons--refresh.png",
    "results": "flat-color-icons--data-sheet.png",
    "save": "flat-color-icons--download.png",
    "save_all": "flat-color-icons--data-backup.png",
    "settings": "flat-color-icons--settings.png",
    "stack": "flat-color-icons--stack-of-photos.png",
    "undo": "flat-color-icons--undo-action.png",
}
ICON_FOLDER = ACTION_ICON_FILES["folder"]
ICON_HOME = ACTION_ICON_FILES["home"]
ICON_REFRESH = ACTION_ICON_FILES["refresh"]


def configure_aidas_styles(style):
    """Compatibility entry point for the centralized CTk/ttk theme bridge."""

    configure_ttk_styles(style)


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


def load_ui_icon(owner, filename, *, size=None):
    """Load a PhotoImage from assets, optionally resizing it to exact pixels."""
    path = asset_path(filename)
    try:
        image = tk.PhotoImage(file=path)
    except tk.TclError:
        with open(path, "rb") as handle:
            data = base64.b64encode(handle.read())
        image = tk.PhotoImage(data=data)
    if size is not None:
        target_width, target_height = (size, size) if isinstance(size, int) else size
        target_width = max(1, int(target_width))
        target_height = max(1, int(target_height))
        source_width = max(1, image.width())
        source_height = max(1, image.height())
        if (source_width, source_height) != (target_width, target_height):
            width_gcd = math.gcd(source_width, target_width)
            height_gcd = math.gcd(source_height, target_height)
            image = image.zoom(
                target_width // width_gcd,
                target_height // height_gcd,
            ).subsample(
                source_width // width_gcd,
                source_height // height_gcd,
            )
    return remember_image(owner, image)


def load_action_icon(owner, action, *, size=ACTION_ICON_SIZE):
    """Load a semantic icon from the app's unified Flat Color icon family."""

    try:
        filename = ACTION_ICON_FILES[action]
    except KeyError as exc:
        raise ValueError(f"Unknown UI action icon: {action}") from exc
    return load_ui_icon(owner, filename, size=size)


def action_button(
    parent,
    owner,
    text,
    command,
    action,
    *,
    tooltip=None,
    icon_size=ACTION_ICON_SIZE,
    **button_options,
):
    """Create a standard text-and-icon ttk action button.

    Geometry and interaction states come from the global ``TButton`` style;
    this helper standardizes semantic icons, their size, and tooltip behavior.
    """

    icon = load_action_icon(owner, action, size=icon_size)
    options = {
        "text": text,
        "command": command,
        "image": icon,
        "compound": "left",
        "style": "AIDaS.Action.TButton",
    }
    options.update(button_options)
    button = ttk.Button(parent, **options)
    button._aidas_action_icon = icon
    if tooltip:
        HoverToolTip(button, tooltip)
    return button


def icon_action_button(
    parent,
    owner,
    command,
    action,
    *,
    tooltip,
    icon_size=ACTION_ICON_SIZE,
    **button_options,
):
    """Create the standard square icon-only ttk action button."""

    button_options.setdefault("style", "AIDaS.Icon.TButton")
    button_options.setdefault("width", 0)
    return action_button(
        parent,
        owner,
        "",
        command,
        action,
        tooltip=tooltip,
        icon_size=icon_size,
        **button_options,
    )


def load_ctk_image(owner, filename, *, size=CONTROLS.icon_size):
    """Load one DPI-aware image for modern CustomTkinter controls."""

    path = asset_path(filename)
    target_size = (size, size) if isinstance(size, int) else tuple(size)
    with Image.open(path) as source:
        image = source.convert("RGBA").copy()
    # Preserve colored icons while lifting only near-black glyph pixels for
    # dark mode. Transparent pixels stay transparent through the alpha mask.
    luminance = ImageOps.grayscale(image.convert("RGB"))
    dark_pixel_mask = luminance.point(lambda value: 255 if value < 92 else 0)
    dark_pixel_mask = ImageChops.multiply(dark_pixel_mask, image.getchannel("A"))
    light_glyph = Image.new("RGBA", image.size, "#DDE7F0")
    dark_image = Image.composite(light_glyph, image, dark_pixel_mask)
    return remember_image(
        owner,
        ctk.CTkImage(light_image=image, dark_image=dark_image, size=target_size),
    )


def icon_button(parent, owner, icon_filename, command, *, tooltip=None, **button_options):
    """Create the shared modern icon-only button used in compact form rows."""

    icon_size = button_options.pop("icon_size", CONTROLS.icon_size)
    icon = load_ctk_image(owner, icon_filename, size=icon_size)
    options = {
        "text": "",
        "image": icon,
        "command": command,
        "width": CONTROLS.height_md,
        "height": CONTROLS.height_md,
        "corner_radius": SHAPES.corner_radius_sm,
        "border_width": 0,
        "bg_color": COLOR_PAIRS["surface"],
        "fg_color": "transparent",
        "hover_color": COLOR_PAIRS["primary_soft"],
        "text_color": COLOR_PAIRS["text"],
    }
    options.update(button_options)
    button = ctk.CTkButton(parent, **options)
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
    row = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
    entry = ctk.CTkEntry(
        row,
        textvariable=textvariable,
        height=CONTROLS.height_md,
        corner_radius=SHAPES.corner_radius_sm,
        border_width=SHAPES.border_width,
        border_color=COLOR_PAIRS["border_strong"],
        fg_color=COLOR_PAIRS["surface_elevated"],
        text_color=COLOR_PAIRS["text"],
    )
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


class NativeNumericSpinbox(ctk.CTkFrame):
    """Modern numeric entry with compact, keyboard-friendly step controls."""

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
        bg_color=None,
    ):
        pixel_width = max(64, int(width) * 9 + 28)
        super().__init__(
            parent,
            width=pixel_width,
            height=CONTROLS.height_md,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border_strong"],
            bg_color=bg_color or COLOR_PAIRS["surface"],
            fg_color=COLOR_PAIRS["surface_elevated"],
        )
        self.var = textvariable
        self.step = step
        self.minimum = minimum
        self.maximum = maximum
        self._button_state = "normal"
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)
        # This is a fixed-size composite. Without this guard, changing a child
        # entry/button state can briefly alter its requested size and bubble a
        # geometry/redraw cascade through the surrounding sidebar.
        self.grid_propagate(False)

        entry_options = {
            "textvariable": self.var,
            "width": max(42, pixel_width - 24),
            "height": CONTROLS.height_sm,
            "justify": "center",
            "font": ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size),
            "corner_radius": 0,
            "border_width": 0,
            "fg_color": "transparent",
            "text_color": COLOR_PAIRS["text"],
        }
        if validatecommand is not None:
            entry_options["validate"] = "key"
            entry_options["validatecommand"] = validatecommand
        self.entry = ctk.CTkEntry(self, **entry_options)
        self.entry.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(4, 1), pady=2)
        self.spinbox = self.entry

        button_options = {
            "width": 22,
            "height": CONTROLS.height_sm // 2,
            "corner_radius": 3,
            "border_width": 0,
            "fg_color": "transparent",
            "hover_color": COLOR_PAIRS["primary_soft"],
            "text_color": COLOR_PAIRS["muted_text"],
            "font": ctk.CTkFont(family=TYPOGRAPHY.family, size=8, weight="bold"),
        }
        self.up_button = ctk.CTkButton(
            self,
            text="▲",
            command=lambda: self._step(self.step),
            **button_options,
        )
        self.up_button.grid(row=0, column=1, sticky="nsew", padx=(0, 2), pady=(2, 0))
        self.down_button = ctk.CTkButton(
            self,
            text="▼",
            command=lambda: self._step(-self.step),
            **button_options,
        )
        self.down_button.grid(row=1, column=1, sticky="nsew", padx=(0, 2), pady=(0, 2))

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
        if state is not None and state != self._button_state:
            self.entry.configure(state=state)
            self.up_button.configure(state=state)
            self.down_button.configure(state=state)
            self._button_state = state
        return result

    config = configure

    def cget(self, attribute_name):
        """Return the semantic control state without exposing child widgets."""
        if attribute_name == "state":
            return self._button_state
        return super().cget(attribute_name)

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
    """One-at-a-time tooltip that cannot leave orphan hover windows behind."""

    SHOW_DELAY_MS = 450
    POINTER_CHECK_MS = 100
    _active_tooltip = None
    _pending_tooltip = None
    _OWNER_DISMISS_EVENTS = (
        "<ButtonPress>",
        "<Escape>",
        "<FocusOut>",
        "<Unmap>",
        "<<NotebookTabChanged>>",
        "<<AIDaSThemeChanged>>",
    )

    def __init__(self, widget, text):
        self.widget = widget
        self._text = str(text or "")
        self.tipwindow = None
        self._show_after_id = None
        self._pointer_after_id = None
        self.widget.bind("<Enter>", self._schedule_show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")
        self.widget.bind("<Unmap>", self._hide, add="+")
        self.widget.bind("<Destroy>", self._hide, add="+")
        self._bind_owner_dismiss_events()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        updated = str(value or "")
        if updated == self._text:
            return
        was_visible = self.tipwindow is not None
        self._text = updated
        # Never leave stale dynamic text on screen. The current hover can show
        # the updated copy again after the normal short delay.
        self._hide()
        if was_visible and updated.strip() and self._pointer_is_over_widget():
            self._schedule_show()

    @classmethod
    def dismiss_active(cls):
        """Cancel a pending tooltip and dismiss the visible tooltip, if any."""

        pending = cls._pending_tooltip
        if pending is not None:
            pending._cancel_scheduled_show()
        active = cls._active_tooltip
        if active is not None:
            active._hide()

    @classmethod
    def _dismiss_for_owner_event(cls, _event=None):
        cls.dismiss_active()

    def _bind_owner_dismiss_events(self):
        """Install one set of window-wide dismissal hooks per owning window."""

        try:
            owner = self.widget.winfo_toplevel()
        except (AttributeError, tk.TclError):
            return
        marker = "_aidas_hover_tooltip_dismiss_events_bound"
        if getattr(owner, marker, False):
            return
        setattr(owner, marker, True)
        for sequence in self._OWNER_DISMISS_EVENTS:
            try:
                owner.bind(sequence, type(self)._dismiss_for_owner_event, add="+")
            except (AttributeError, tk.TclError):
                pass

    def _schedule_show(self, _event=None):
        self._cancel_scheduled_show()
        if not str(self.text or "").strip():
            return
        pending = type(self)._pending_tooltip
        if pending is not None and pending is not self:
            pending._cancel_scheduled_show()
        try:
            self._show_after_id = self.widget.after(self.SHOW_DELAY_MS, self._show_now)
            type(self)._pending_tooltip = self
        except tk.TclError:
            self._show_after_id = None

    def _show_now(self):
        self._show_after_id = None
        if type(self)._pending_tooltip is self:
            type(self)._pending_tooltip = None
        if self.tipwindow is not None or not str(self.text or "").strip():
            return
        if not self._pointer_is_over_widget():
            return

        active = type(self)._active_tooltip
        if active is not None and active is not self:
            active._hide()

        x = self.widget.winfo_pointerx() + 12
        y = self.widget.winfo_pointery() + 12
        tw = tk.Toplevel(self.widget)
        tw.withdraw()
        tw.wm_overrideredirect(True)
        tw.transient(self.widget.winfo_toplevel())
        tw.configure(background=COLORS.border_strong)
        try:
            tw.attributes("-topmost", True)
        except tk.TclError:
            pass

        shell = ctk.CTkFrame(
            tw,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border_strong"],
            fg_color=COLOR_PAIRS["surface_elevated"],
        )
        shell.pack(fill="both", expand=True)
        ctk.CTkLabel(
            shell,
            text=self.text,
            justify="left",
            anchor="w",
            wraplength=320,
            fg_color="transparent",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.caption_size),
        ).pack(padx=8, pady=5)

        tw.update_idletasks()
        width = max(1, tw.winfo_reqwidth())
        height = max(1, tw.winfo_reqheight())
        left, top, right, bottom = work_area_bounds(
            tw,
            parent=self.widget.winfo_toplevel(),
        )
        x = max(left, min(x, right - width))
        y = max(top, min(y, bottom - height))
        tw.wm_geometry(f"+{x}+{y}")
        self.tipwindow = tw
        type(self)._active_tooltip = self
        tw.deiconify()
        tw.lift()
        self._schedule_pointer_check()

    def _pointer_is_over_widget(self):
        try:
            if not self.widget.winfo_exists() or not self.widget.winfo_viewable():
                return False
            x, y = self.widget.winfo_pointerxy()
            target = self.widget.winfo_containing(x, y)
        except tk.TclError:
            return False
        while target is not None:
            if target is self.widget:
                return True
            target = getattr(target, "master", None)
        return False

    def _schedule_pointer_check(self):
        self._cancel_pointer_check()
        try:
            self._pointer_after_id = self.widget.after(
                self.POINTER_CHECK_MS,
                self._check_pointer,
            )
        except tk.TclError:
            self._pointer_after_id = None

    def _check_pointer(self):
        self._pointer_after_id = None
        if self.tipwindow is None:
            return
        try:
            tip_exists = bool(self.tipwindow.winfo_exists())
        except tk.TclError:
            tip_exists = False
        if not tip_exists or not str(self.text or "").strip() or not self._pointer_is_over_widget():
            self._hide()
            return
        self._schedule_pointer_check()

    def _cancel_scheduled_show(self):
        after_id = self._show_after_id
        self._show_after_id = None
        if type(self)._pending_tooltip is self:
            type(self)._pending_tooltip = None
        if after_id is not None:
            try:
                self.widget.after_cancel(after_id)
            except tk.TclError:
                pass

    def _cancel_pointer_check(self):
        after_id = self._pointer_after_id
        self._pointer_after_id = None
        if after_id is not None:
            try:
                self.widget.after_cancel(after_id)
            except tk.TclError:
                pass

    def _hide(self, _event=None):
        self._cancel_scheduled_show()
        self._cancel_pointer_check()
        tw = self.tipwindow
        self.tipwindow = None
        if type(self)._active_tooltip is self:
            type(self)._active_tooltip = None
        if tw is not None:
            try:
                tw.destroy()
            except tk.TclError:
                pass


class ScrollableSidebar(ctk.CTkFrame):
    """A vertical sidebar whose content can be scrolled with the mouse."""

    def __init__(self, parent, *, width=None):
        super().__init__(parent, fg_color="transparent", corner_radius=0)

        canvas_options = {
            "highlightthickness": 0,
            "bd": 0,
            "background": COLORS.sidebar,
        }
        if width is not None:
            canvas_options["width"] = width

        self.canvas = tk.Canvas(self, **canvas_options)
        self.scrollbar = ctk.CTkScrollbar(
            self,
            orientation="vertical",
            command=self.canvas.yview,
            width=CONTROLS.scrollbar_width,
            fg_color="transparent",
            button_color=COLOR_PAIRS["border_strong"],
            button_hover_color=COLOR_PAIRS["primary"],
        )
        self.content = ctk.CTkFrame(
            self.canvas,
            fg_color=COLOR_PAIRS["sidebar"],
            corner_radius=0,
        )
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
        # CTkBaseClass deliberately blocks bind_all; call Tk's implementation
        # explicitly for this established sidebar-wide wheel routing contract.
        tk.Misc.bind_all(self, "<MouseWheel>", self._on_mousewheel, add="+")
        tk.Misc.bind_all(self, "<Button-4>", self._on_mousewheel, add="+")
        tk.Misc.bind_all(self, "<Button-5>", self._on_mousewheel, add="+")
        tk.Misc.bind_all(self, "<ButtonPress-2>", self._on_middle_press, add="+")
        tk.Misc.bind_all(self, "<B2-Motion>", self._on_middle_drag, add="+")
        tk.Misc.bind_all(self, "<ButtonRelease-2>", self._on_middle_release, add="+")
        tk.Misc.bind_all(self, "<ButtonPress-1>", self._on_primary_press, add="+")

    def _apply_aidas_theme(self):
        self.canvas.configure(background=COLORS.sidebar)

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


class CollapsibleSection(ctk.CTkFrame):
    """A rounded, accessible sidebar card with a collapsible body."""

    HEADER_HEIGHT = 36

    def __init__(self, parent, title, *, padding=3, expanded=True):
        super().__init__(
            parent,
            fg_color=COLOR_PAIRS["surface"],
            corner_radius=SHAPES.corner_radius_md,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
        )
        self.title = title
        self.expanded = bool(expanded)
        self.header = ctk.CTkButton(
            self,
            text="",
            command=self.toggle,
            height=self.HEADER_HEIGHT,
            corner_radius=SHAPES.corner_radius_md,
            border_width=0,
            fg_color="transparent",
            hover_color=COLOR_PAIRS["primary_soft"],
            text_color=COLOR_PAIRS["text"],
            anchor="w",
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        )
        self.header.pack(fill="x", padx=2, pady=2)

        self._body_container = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self._body_container.pack(fill="both", expand=True, padx=2, pady=(0, 4))
        self.body = ctk.CTkFrame(
            self._body_container,
            fg_color="transparent",
            corner_radius=0,
        )
        self.body.pack(fill="both", expand=True, padx=padding, pady=(1, padding))
        self._sync_header()

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            self._body_container.pack(fill="both", expand=True, padx=2, pady=(0, 4))
        else:
            self._body_container.pack_forget()
        self._sync_header()
        self._queue_sidebar_refresh()

    def _sync_header(self):
        marker = "▾" if self.expanded else "▸"
        self.header.configure(text=f"  {marker}   {self.title}")

    def _queue_sidebar_refresh(self):
        widget = self.master
        while widget is not None:
            queue_refresh = getattr(widget, "queue_refresh", None)
            if callable(queue_refresh):
                queue_refresh()
                return
            widget = getattr(widget, "master", None)


class SidebarStepFrame(ctk.CTkFrame):
    """Standard left-sidebar/right-content layout for AIDaS step pages."""

    SIDEBAR_WIDTH = LAYOUT.sidebar_width
    SIDEBAR_TEXT_WRAP = 344
    SIDEBAR_MINIMUM = LAYOUT.sidebar_minimum
    CONTENT_MINIMUM = LAYOUT.content_minimum
    SECTION_PADDING = 8
    SECTION_PACK = {"fill": "x", "padx": (0, 0), "pady": (0, 8)}

    def __init__(self, parent, *args, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        kwargs.setdefault("corner_radius", 0)
        super().__init__(parent, *args, **kwargs)

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
        self.main = ctk.CTkFrame(
            self,
            fg_color=COLOR_PAIRS["application"],
            corner_radius=0,
        )
        self.main.pack(fill="both", expand=True)

        if sidebar_width is None:
            sidebar_width = self.SIDEBAR_WIDTH
        self._sidebar_design_width = max(1, int(sidebar_width))
        self._layout_dpi_scale = None
        self._sync_scaled_layout_values()
        self._last_workspace_width = None

        self.workspace = tk.PanedWindow(
            self.main,
            orient=tk.HORIZONTAL,
            background=COLORS.border,
            borderwidth=0,
            opaqueresize=True,
            sashwidth=LAYOUT.divider_width,
            sashcursor="arrow",
            sashrelief="flat",
            showhandle=False,
        )
        self.workspace.pack(fill="both", expand=True)

        sidebar_padding = self._pane_padding(
            sidebar_pack,
            default=(LAYOUT.space_sm, LAYOUT.space_sm, LAYOUT.space_xs, LAYOUT.space_sm),
        )
        content_padding = self._pane_padding(
            content_pack,
            default=(LAYOUT.space_sm, LAYOUT.space_sm, LAYOUT.space_sm, LAYOUT.space_sm),
        )
        self.sidebar_shell = ctk.CTkFrame(
            self.workspace,
            fg_color=COLOR_PAIRS["sidebar"],
            corner_radius=0,
            border_width=0,
        )
        self.sidebar = ScrollableSidebar(self.sidebar_shell, width=self._sidebar_width)
        self.sidebar.pack(
            fill="both",
            expand=True,
            padx=(sidebar_padding[0], sidebar_padding[2]),
            pady=(sidebar_padding[1], sidebar_padding[3]),
        )
        self.ctrl = self.sidebar.content

        self.content_shell = ctk.CTkFrame(
            self.workspace,
            fg_color=COLOR_PAIRS["surface"],
            corner_radius=0,
            border_width=0,
        )
        if status_var is not None:
            self.add_status_bar(status_var, parent=self.content_shell)
        self.content = ctk.CTkFrame(
            self.content_shell,
            fg_color="transparent",
            corner_radius=0,
        )
        self.content.pack(
            fill="both",
            expand=True,
            padx=(content_padding[0], content_padding[2]),
            pady=(content_padding[1], content_padding[3]),
        )

        self.workspace.add(
            self.sidebar_shell,
            minsize=self._sidebar_minimum,
            stretch="never",
        )
        self.workspace.add(
            self.content_shell,
            minsize=self._content_minimum,
            stretch="always",
        )
        self._pane_minima = (self._sidebar_minimum, self._content_minimum)
        self.workspace.bind("<Configure>", self._on_workspace_configure, add="+")
        self.workspace.bind("<Button-1>", self._lock_sidebar_sash, add="+")
        self.workspace.bind("<B1-Motion>", self._lock_sidebar_sash, add="+")
        self.workspace.bind("<ButtonRelease-1>", self._lock_sidebar_sash, add="+")
        self.after_idle(self._apply_workspace_layout)

        return self.ctrl, self.content

    def _apply_aidas_theme(self):
        self.workspace.configure(background=COLORS.border)

    @staticmethod
    def _pane_padding(options, *, default):
        """Translate legacy pack padding into ttk four-sided frame padding."""

        if not options:
            return default
        padx = options.get("padx", (default[0], default[2]))
        pady = options.get("pady", (default[1], default[3]))
        left, right = padx if isinstance(padx, (tuple, list)) else (padx, padx)
        top, bottom = pady if isinstance(pady, (tuple, list)) else (pady, pady)
        return left, top, right, bottom

    def _on_workspace_configure(self, event):
        scale_changed = self._sync_scaled_layout_values()
        if event.width == self._last_workspace_width and not scale_changed:
            return
        self._last_workspace_width = event.width
        self.after_idle(self._apply_workspace_layout)

    def _sync_scaled_layout_values(self):
        """Keep fixed pane measurements visually stable across monitor DPI levels."""

        try:
            dpi_scale = float(self.winfo_fpixels("1i")) / 96.0
        except (tk.TclError, TypeError, ValueError):
            dpi_scale = 1.0
        dpi_scale = min(3.0, max(0.75, dpi_scale))
        if self._layout_dpi_scale is not None and abs(
            dpi_scale - self._layout_dpi_scale
        ) < 0.01:
            return False

        self._layout_dpi_scale = dpi_scale
        self._sidebar_width = max(1, round(self._sidebar_design_width * dpi_scale))
        self._sidebar_minimum = max(1, round(self.SIDEBAR_MINIMUM * dpi_scale))
        self._content_minimum = max(1, round(self.CONTENT_MINIMUM * dpi_scale))
        return True

    def _lock_sidebar_sash(self, event):
        """Keep the divider fixed while leaving controls in both panes interactive."""

        try:
            if self.workspace.identify(event.x, event.y):
                self.after_idle(self._apply_workspace_layout)
                return "break"
        except tk.TclError:
            pass
        return None

    def _apply_workspace_layout(self):
        """Keep a compact fixed sidebar while respecting both pane minima."""

        try:
            width = self.workspace.winfo_width()
            if width <= 1 or len(self.workspace.panes()) < 2:
                return
            sidebar_width = workspace_sidebar_width(
                width,
                sidebar_width=self._sidebar_width,
                sidebar_minimum=self._sidebar_minimum,
                content_minimum=self._content_minimum,
            )
            available = max(2, width - LAYOUT.divider_width)
            if available >= self._sidebar_minimum + self._content_minimum:
                sidebar_floor = self._sidebar_minimum
                content_floor = self._content_minimum
            else:
                sidebar_floor = max(1, sidebar_width)
                content_floor = max(1, available - sidebar_width)
            pane_minima = (sidebar_floor, content_floor)
            if pane_minima != self._pane_minima:
                self.workspace.paneconfigure(self.sidebar_shell, minsize=sidebar_floor)
                self.workspace.paneconfigure(self.content_shell, minsize=content_floor)
                self._pane_minima = pane_minima
                self.after_idle(self._apply_workspace_layout)
                return
            self.workspace.sash_place(0, sidebar_width, 0)
        except tk.TclError:
            return

    def add_sidebar_section(self, title, *, padding=None, **pack_options):
        """Add a collapsible section to the standard sidebar."""
        if padding is None:
            padding = self.SECTION_PADDING
        section = CollapsibleSection(self.ctrl, title, padding=padding)
        options = dict(self.SECTION_PACK)
        options.update(pack_options)
        section.pack(**options)
        return section

    def add_content_header(self, textvariable, *, parent=None):
        """Add the standard contextual header above a step's main canvas."""

        container = parent if parent is not None else self.content
        frame = ctk.CTkFrame(
            container,
            fg_color=COLOR_PAIRS["surface_subtle"],
            corner_radius=SHAPES.corner_radius_md,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
        )
        frame.pack(fill="x", pady=(0, LAYOUT.space_sm))
        label = ctk.CTkLabel(
            frame,
            textvariable=textvariable,
            anchor="w",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        )
        label.pack(fill="x", padx=LAYOUT.space_md, pady=LAYOUT.space_sm)
        return frame

    def add_status_bar(self, status_var, *, parent=None):
        """Add a standard sunken status label."""
        container = parent if parent is not None else self
        label = ctk.CTkLabel(
            container,
            textvariable=status_var,
            anchor="w",
            height=30,
            corner_radius=SHAPES.corner_radius_sm,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
            fg_color=COLOR_PAIRS["surface_subtle"],
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.caption_size),
        )
        label.pack(side="bottom", fill="x", pady=(0, LAYOUT.space_sm))
        self.content_status_bar = label
        return label
