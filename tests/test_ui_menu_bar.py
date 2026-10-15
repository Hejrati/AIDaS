from __future__ import annotations

from collections import deque
import tkinter as tk
import unittest

import customtkinter as ctk
from PIL import ImageChops, ImageColor, ImageTk

from aidas.ui.menu_bar import (
    ApplicationMenuBar,
    _MenuItem,
    _PopupCommandRow,
    _PopupMenu,
    _PopupSeparator,
)
from aidas.ui.theme import (
    COLOR_PAIRS, COLORS, apply_appearance_mode, get_interface_mode, set_interface_mode,
)


class _SelectionRow:
    def __init__(self):
        self.states = []
        self.enabled_states = []
        self.checked = False

    def set_selected(self, selected):
        self.states.append(bool(selected))

    def set_enabled(self, enabled):
        self.enabled_states.append(bool(enabled))

    def set_checked(self, checked):
        self.checked = bool(checked)


class _ColorTarget:
    def __init__(self):
        self.colors = []

    def configure(self, **options):
        self.colors.append(options["fg_color"])


class _PointerEvent:
    def __init__(self, x_root, y_root):
        self.x_root = x_root
        self.y_root = y_root


class _MenuButton:
    def __init__(self):
        self.fg_color = "transparent"

    def cget(self, name):
        if name != "fg_color":
            raise KeyError(name)
        return self.fg_color

    def configure(self, **options):
        self.fg_color = options["fg_color"]


class _CachedPopup:
    def __init__(self):
        self.items = []
        self.shown_with = []
        self.hide_count = 0

    def set_items(self, items):
        self.items.append(tuple(items))

    def show(self, anchor):
        self.shown_with.append(anchor)

    def hide(self):
        self.hide_count += 1


class MenuBarTests(unittest.TestCase):
    def test_interface_switch_waits_for_popup_callback_to_return(self):
        bar = ApplicationMenuBar.__new__(ApplicationMenuBar)
        bar._interface_modes = ("Modern", "Classic")
        bar._current_interface = "Modern"
        bar._popup_cache = {}
        calls = []
        queued = []
        bar._set_interface_command = calls.append
        bar.after_idle = lambda callback: queued.append(callback) or "after#1"

        bar._select_interface("Classic")

        self.assertEqual(bar.current_interface, "Classic")
        self.assertEqual(calls, [])
        self.assertEqual(len(queued), 1)
        queued.pop()()
        self.assertEqual(calls, ["Classic"])

    def test_unbinding_one_stale_shortcut_still_releases_the_rest(self):
        class Owner:
            def __init__(self):
                self.calls = []

            def unbind(self, sequence, binding):
                self.calls.append((sequence, binding))
                if len(self.calls) == 1:
                    raise tk.TclError("stale command")

        owner = Owner()
        bar = ApplicationMenuBar.__new__(ApplicationMenuBar)
        bar._root_bindings = [("<Alt-f>", "first"), ("<Alt-v>", "second")]
        bar.winfo_toplevel = lambda: owner

        bar._unbind_root_shortcuts()

        self.assertEqual(
            owner.calls,
            [("<Alt-f>", "first"), ("<Alt-v>", "second")],
        )
        self.assertEqual(bar._root_bindings, [])

    def test_unknown_interface_falls_back_to_first_declared_mode(self):
        bar = ApplicationMenuBar.__new__(ApplicationMenuBar)
        bar._interface_modes = ("Modern", "Classic")

        self.assertEqual(bar._canonical_interface("obsolete"), "Modern")

    def test_top_level_menu_targets_share_one_compact_geometry(self):
        self.assertEqual(ApplicationMenuBar.MENU_NAMES, ("File", "View", "Help"))
        self.assertEqual(ApplicationMenuBar._BUTTON_WIDTH, 50)
        self.assertEqual(
            ApplicationMenuBar._BUTTON_HEIGHT
            + sum(ApplicationMenuBar._ROW_PADDING_Y),
            ApplicationMenuBar._BAR_HEIGHT - 1,
        )
        self.assertGreater(ApplicationMenuBar._BUTTON_GAP, 0)

    def test_popup_rows_use_compact_height_and_fixed_alignment_column(self):
        self.assertEqual(_PopupMenu._ITEM_HEIGHT, 24)
        self.assertEqual(_PopupMenu._ITEM_MARGIN_X, 4)
        self.assertEqual(_PopupMenu._ITEM_GAP_Y, 0)
        self.assertGreater(_PopupMenu._CHECK_COLUMN_WIDTH, 0)
        self.assertEqual(
            _PopupMenu._MENU_WIDTH - (2 * _PopupMenu._ITEM_MARGIN_X),
            212,
        )

    def test_popup_selection_surface_uses_uniform_rounded_inset(self):
        self.assertEqual(_PopupCommandRow._CORNER_RADIUS, 4)
        self.assertEqual(_PopupMenu._ITEM_MARGIN_X, 4)
        self.assertEqual(_PopupMenu._EDGE_PADDING_Y, 4)
        self.assertEqual(_PopupMenu._ITEM_GAP_Y, 0)

    def test_popup_separator_is_compact_and_visible(self):
        self.assertEqual(_PopupSeparator._GAP, 4)
        self.assertEqual(_PopupSeparator._INSET + _PopupMenu._ITEM_MARGIN_X, 8)

    def test_popup_row_selection_skips_repainting_unchanged_state(self):
        row = _PopupCommandRow.__new__(_PopupCommandRow)
        target = _ColorTarget()
        row.configure = target.configure
        row._enabled = True
        row._selected = False

        row.set_selected(True)
        row.set_selected(True)
        self.assertEqual(target.colors, [COLOR_PAIRS["menu_hover"]])
        row.set_selected(False)
        self.assertEqual(target.colors[-1], COLOR_PAIRS["menu_surface"])

    def test_keyboard_selection_still_moves_between_command_rows(self):
        popup = _PopupMenu.__new__(_PopupMenu)
        popup._buttons = {0: _SelectionRow(), 2: _SelectionRow(), 4: _SelectionRow()}
        popup._command_indices = [0, 2, 4]
        popup._selected = -1

        self.assertEqual(popup._move_selection(1), "break")
        self.assertEqual(popup._selected, 0)
        self.assertEqual(popup._move_selection(1), "break")
        self.assertEqual(popup._selected, 2)
        self.assertEqual(popup._move_selection(-1), "break")
        self.assertEqual(popup._selected, 0)
        self.assertEqual(popup._move_selection(-1), "break")
        self.assertEqual(popup._selected, 4)

    def test_reentering_selected_popup_row_does_not_toggle_its_surface(self):
        popup = _PopupMenu.__new__(_PopupMenu)
        row = _SelectionRow()
        popup._buttons = {2: row}
        popup._command_indices = [2]
        popup._selected = 2

        popup._select(2)

        self.assertEqual(row.states, [])

    def test_popup_command_requires_press_and_release_inside_same_row(self):
        invoked = []
        row = _PopupCommandRow.__new__(_PopupCommandRow)
        row._pressed = False
        row._on_enter = lambda: None
        row._on_invoke = lambda: invoked.append(True)
        row.winfo_rootx = lambda: 100
        row.winfo_rooty = lambda: 200
        row.winfo_width = lambda: 240
        row.winfo_height = lambda: 28

        row._handle_invoke(_PointerEvent(120, 210))
        self.assertEqual(invoked, [])

        row._handle_press()
        row._handle_invoke(_PointerEvent(400, 210))
        self.assertEqual(invoked, [])

        row._handle_press()
        row._handle_invoke(_PointerEvent(120, 210))
        self.assertEqual(invoked, [True])

    def test_disabled_popup_row_ignores_mouse_selection_and_invocation(self):
        entered = []
        invoked = []
        row = _PopupCommandRow.__new__(_PopupCommandRow)
        row._enabled = False
        row._pressed = False
        row._on_enter = lambda: entered.append(True)
        row._on_invoke = lambda: invoked.append(True)
        row.winfo_rootx = lambda: 100
        row.winfo_rooty = lambda: 200
        row.winfo_width = lambda: 240
        row.winfo_height = lambda: 28

        row._handle_enter()
        row._handle_press()
        row._handle_invoke(_PointerEvent(120, 210))

        self.assertEqual(entered, [])
        self.assertEqual(invoked, [])

    def test_cached_popup_updates_enabled_rows_and_repairs_selection(self):
        popup = _PopupMenu.__new__(_PopupMenu)
        popup._items = (
            _MenuItem("command", "Modern"),
            _MenuItem("command", "System", checked=True),
            _MenuItem("command", "Dark"),
        )
        popup._buttons = {
            0: _SelectionRow(),
            1: _SelectionRow(),
            2: _SelectionRow(),
        }
        popup._command_indices = [0, 1, 2]
        popup._selected = 1
        popup._visible = False

        popup.set_items(
            (
                _MenuItem("command", "Modern"),
                _MenuItem("command", "System", checked=True, enabled=False),
                _MenuItem("command", "Dark", enabled=False),
            )
        )

        self.assertEqual(popup._command_indices, [0])
        self.assertEqual(popup._selected, -1)
        self.assertEqual(popup._buttons[1].enabled_states, [False])
        self.assertEqual(popup._buttons[2].enabled_states, [False])
        self.assertIn(False, popup._buttons[1].states)

    def test_disabled_popup_item_cannot_close_or_invoke_via_keyboard_path(self):
        calls = []
        popup = _PopupMenu.__new__(_PopupMenu)
        popup._items = (
            _MenuItem(
                "command",
                "Dark",
                lambda: calls.append("invoked"),
                enabled=False,
            ),
        )
        popup.hide = lambda: calls.append("hidden")

        popup._invoke(0)

        self.assertEqual(calls, [])

    def test_opening_menus_reuses_prebuilt_popup_widgets(self):
        bar = ApplicationMenuBar.__new__(ApplicationMenuBar)
        file_button = _MenuButton()
        view_button = _MenuButton()
        file_popup = _CachedPopup()
        view_popup = _CachedPopup()
        bar._buttons = {"File": file_button, "View": view_button}
        bar._popup_cache = {"File": file_popup, "View": view_popup}
        bar._popup = None
        bar._active_menu = None
        bar._items_for = lambda name: (name,)

        bar._open_menu("File")
        self.assertIs(bar._popup, file_popup)
        self.assertEqual(file_popup.shown_with, [file_button])

        bar._open_menu("View")
        self.assertEqual(file_popup.hide_count, 1)
        self.assertIs(bar._popup, view_popup)
        self.assertEqual(view_popup.shown_with, [view_button])
        self.assertEqual(tuple(bar._popup_cache), ("File", "View"))

    def test_hiding_popup_withdraws_it_for_reuse_only_once(self):
        closed = []
        withdrawn = []
        popup = _PopupMenu.__new__(_PopupMenu)
        popup._closed = False
        popup._visible = True
        popup._selected = -1
        popup._buttons = {}
        popup._outside_binding = None
        popup._on_close = lambda hidden: closed.append(hidden)
        popup.withdraw = lambda: withdrawn.append(True)

        popup.hide()
        popup.hide()

        self.assertEqual(withdrawn, [True])
        self.assertEqual(closed, [popup])
        self.assertFalse(popup._visible)

    def test_menu_items_retain_commands_and_checked_appearance(self):
        calls = []
        bar = ApplicationMenuBar.__new__(ApplicationMenuBar)
        bar._appearance_modes = ("System", "Light", "Dark")
        bar._current_appearance = "Dark"
        bar._browse_sdb_command = lambda: calls.append("browse")
        bar._tutorial_command = lambda: calls.append("tutorial")
        bar._check_updates_command = lambda: calls.append("updates")
        bar._about_command = lambda: calls.append("about")
        bar._exit_command = lambda: calls.append("exit")
        bar._select_appearance = lambda mode: calls.append(mode)

        file_items = bar._items_for("File")
        help_items = bar._items_for("Help")
        view_items = bar._items_for("View")
        file_items[0].command()
        file_items[2].command()
        help_items[0].command()
        help_items[2].command()
        help_items[4].command()
        next(item for item in view_items if item.label == "Dark").command()

        self.assertEqual(
            calls,
            ["browse", "exit", "tutorial", "updates", "about", "Dark"],
        )
        self.assertEqual(
            [item.label for item in help_items],
            [
                "Workflow Tutorial...",
                "",
                "Check for Updates...",
                "",
                "About",
            ],
        )
        checked = [item.label for item in view_items if item.checked]
        self.assertEqual(checked, ["Dark"])

    def test_view_menu_disables_but_retains_appearance_in_classic(self):
        calls = []
        bar = ApplicationMenuBar.__new__(ApplicationMenuBar)
        bar._interface_modes = ("Modern", "Classic")
        bar._current_interface = "Classic"
        bar._appearance_modes = ("System", "Light", "Dark")
        bar._current_appearance = "Light"
        bar._select_interface = lambda mode: calls.append(("interface", mode))
        bar._select_appearance = lambda mode: calls.append(("appearance", mode))

        items = bar._items_for("View")
        next(item for item in items if item.label == "Modern").command()

        self.assertEqual(
            [item.label for item in items if item.kind == "heading"],
            ["Interface", "Appearance"],
        )
        self.assertEqual(
            [item.label for item in items if item.checked],
            ["Classic", "Light"],
        )
        self.assertEqual(
            [item.label for item in items if item.kind == "command" and item.enabled],
            ["Modern", "Classic"],
        )
        self.assertEqual(
            [item.label for item in items if item.kind == "command" and not item.enabled],
            ["System", "Light", "Dark"],
        )
        self.assertEqual(calls, [("interface", "Modern")])

    def test_set_interface_refreshes_cached_view_state_in_both_directions(self):
        bar = ApplicationMenuBar.__new__(ApplicationMenuBar)
        bar._interface_modes = ("Modern", "Classic")
        bar._current_interface = "Classic"
        bar._appearance_modes = ("System", "Light", "Dark")
        bar._current_appearance = "System"
        popup = _CachedPopup()
        bar._popup_cache = {"View": popup}

        bar.set_interface("Modern")
        modern_items = popup.items[-1]
        self.assertTrue(
            all(
                item.enabled
                for item in modern_items
                if item.label in bar._appearance_modes
            )
        )

        bar.set_interface("Classic")
        classic_items = popup.items[-1]
        self.assertTrue(
            all(
                not item.enabled
                for item in classic_items
                if item.label in bar._appearance_modes
            )
        )

    def test_classic_defensively_ignores_appearance_selection(self):
        calls = []
        bar = ApplicationMenuBar.__new__(ApplicationMenuBar)
        bar._interface_modes = ("Modern", "Classic")
        bar._current_interface = "Classic"
        bar._appearance_modes = ("System", "Light", "Dark")
        bar._current_appearance = "Light"
        bar._set_appearance_command = lambda mode: calls.append(mode)
        bar._popup_cache = {}

        bar._select_appearance("Dark")

        self.assertEqual(bar.current_appearance, "Light")
        self.assertEqual(calls, [])


class MenuRenderingTests(unittest.TestCase):
    def setUp(self):
        self.original_interface = get_interface_mode()
        self.original_appearance = ctk.get_appearance_mode()
        self.original_source = ctk.AppearanceModeTracker.appearance_mode_set_by
        self.original_scaling = ctk.ScalingTracker.widget_scaling
        set_interface_mode("Modern", redraw=False)
        ctk.set_appearance_mode("Dark")
        try:
            self.root = ctk.CTk()
        except tk.TclError as exc:
            set_interface_mode(self.original_interface, redraw=False)
            ctk.set_appearance_mode(
                "System" if self.original_source == "system" else self.original_appearance
            )
            self.skipTest(f"A Tk display is unavailable: {exc}")
        self.root.withdraw()
        self.root.attributes("-alpha", 0)
        self.root.geometry("640x320+0+0")
        self.invoked = []
        self.bar = ApplicationMenuBar(
            self.root,
            appearance_modes=("System", "Light", "Dark"),
            current_appearance="Dark",
            set_appearance_command=self.invoked.append,
            interface_modes=("Modern", "Classic"),
            current_interface="Modern",
            set_interface_command=self.invoked.append,
            browse_sdb_command=lambda: self.invoked.append("browse"),
            check_updates_command=lambda: None,
            about_command=lambda: None,
        )
        self.bar.pack(fill="x")
        for popup in self.bar._popup_cache.values():
            popup.attributes("-alpha", 0)
        self.root.deiconify()
        self.root.update()

    def tearDown(self):
        ctk.set_widget_scaling(self.original_scaling)
        self.root.destroy()
        set_interface_mode(self.original_interface, redraw=False)
        ctk.set_appearance_mode(
            "System" if self.original_source == "system" else self.original_appearance
        )

    def test_command_rows_use_one_canvas_and_keep_all_labels_inside_at_each_scale(self):
        for scaling in (1, 1.25, 1.5, 2):
            with self.subTest(scaling=scaling):
                ctk.set_widget_scaling(scaling)
                self.root.update_idletasks()
                for name, popup in self.bar._popup_cache.items():
                    self.bar._open_menu(name)
                    self.root.update()
                    for row in popup._buttons.values():
                        children = row.tk.call("winfo", "children", str(row))
                        self.assertEqual(len(children), 1)
                        left, top, right, bottom = row._canvas.bbox(row._text_id)
                        self.assertGreaterEqual(left, 0)
                        self.assertLess(right, row.winfo_width())
                        self.assertGreaterEqual(top, 0)
                        self.assertLessEqual(bottom, row.winfo_height())

    def test_selected_row_and_checkmark_follow_appearance_without_rebuilding(self):
        popup = self.bar._popup_cache["View"]
        dark_index = next(i for i, item in enumerate(popup._items) if item.label == "Dark")
        row = popup._buttons[dark_index]
        popup._select(dark_index)
        canvas_items = row._canvas.find_all()
        for mode in ("Light", "Dark", "Light"):
            apply_appearance_mode(mode, root=self.root)
            self.root.update_idletasks()
            self.assertEqual(row._canvas.itemcget(row._text_id, "fill"), COLORS.menu_hover_text)
            self.assertEqual(row._canvas.itemcget(row._check_id, "fill"), COLORS.menu_hover_text)
            self.assertEqual(row._canvas.itemcget(row._check_id, "state"), "normal")
            self.assertEqual(popup.cget("background"), COLORS.menu_surface)
            self.assertEqual(popup._panel.cget("border_color"), COLOR_PAIRS["menu_border"])
            self.assertEqual(row._canvas.find_all(), canvas_items)

    def test_cached_reopen_clears_selection_and_keyboard_can_invoke(self):
        popup = self.bar._popup_cache["View"]
        self.bar._open_from_key("View")
        self.assertEqual(popup._selected, popup._command_indices[0])
        self.bar.close_menu()
        self.assertIsNone(popup._focus_after_id)
        self.assertIsNone(popup._focus_check_after_id)
        self.bar._open_menu("View")
        self.assertIs(self.bar._popup, popup)
        self.assertEqual(popup._selected, -1)
        popup._move_selection(-1)
        self.assertEqual(popup._items[popup._selected].label, "Dark")
        popup._invoke_selected()
        self.assertEqual(self.invoked, ["Dark"])
        self.assertFalse(popup._visible)

    def test_hover_switches_only_while_a_menu_is_open(self):
        self.bar._hover_menu("View")
        self.assertIsNone(self.bar._popup)
        self.bar._open_menu("File")
        file_popup = self.bar._popup
        self.bar._hover_menu("Help")
        self.assertFalse(file_popup._visible)
        self.assertIs(self.bar._popup, self.bar._popup_cache["Help"])

    def test_disabled_rows_ignore_pointer_and_keyboard_and_checks_stay_current(self):
        popup = self.bar._popup_cache["View"]
        self.bar.set_interface("Classic")
        dark_index = next(i for i, item in enumerate(popup._items) if item.label == "Dark")
        row = popup._buttons[dark_index]
        row._handle_enter()
        row._handle_press()
        row._handle_invoke(_PointerEvent(row.winfo_rootx() + 5, row.winfo_rooty() + 5))
        self.assertEqual(self.invoked, [])
        self.assertNotIn(dark_index, popup._command_indices)
        self.assertEqual(row._canvas.itemcget(row._text_id, "fill"), COLORS.menu_disabled_text)
        self.bar.set_interface("Modern")
        self.bar.set_appearance("Light")
        self.assertIn(dark_index, popup._command_indices)
        self.assertEqual(row._canvas.itemcget(row._check_id, "state"), "hidden")

    def test_popup_outline_is_closed_and_one_pixel_thick_at_each_scale(self):
        popup = self.bar._popup_cache["View"]
        for scale in (1, 1.25, 1.5, 2):
            with self.subTest(scale=scale):
                ctk.set_widget_scaling(scale)
                self.bar._open_menu("View")
                self.root.update()
                panel = popup._panel
                rendered = ImageTk.getimage(panel._canvas._aidas_surface_photo).convert("RGB")
                width, height = rendered.size
                surface = ImageColor.getrgb(COLORS.menu_surface)
                stroke = ImageColor.getrgb(COLORS.menu_border)
                # All four straight edges occupy precisely one physical pixel.
                for x in (0, width - 1):
                    self.assertEqual(rendered.getpixel((x, height // 2)), stroke)
                for y in (0, height - 1):
                    self.assertEqual(rendered.getpixel((width // 2, y)), stroke)
                for x, y in ((1, height // 2), (width - 2, height // 2),
                             (width // 2, 1), (width // 2, height - 2)):
                    self.assertEqual(rendered.getpixel((x, y)), surface)

                # A flood from the outside must not leak through a corner join.
                pixels = rendered.load()
                outside = {(0, 0)}
                pending = deque(outside)
                while pending:
                    x, y = pending.popleft()
                    for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                        if not (0 <= nx < width and 0 <= ny < height) or (nx, ny) in outside:
                            continue
                        if max(abs(a-b) for a, b in zip(pixels[nx, ny], surface)) < 4:
                            outside.add((nx, ny))
                            pending.append((nx, ny))
                self.assertNotIn((width // 2, height // 2), outside)
                # Dividers never cover the outline at their left/right edges.
                for separator in popup._separators:
                    self.assertGreater(separator.winfo_x(), 1)
                    self.assertLess(separator.winfo_x() + separator.winfo_width(), width - 1)

    def test_dividers_have_integer_pixel_geometry_at_each_scale(self):
        popup = self.bar._popup_cache["View"]
        for scale in (1, 1.25, 1.5, 2):
            with self.subTest(scale=scale):
                ctk.set_widget_scaling(scale)
                self.bar._open_menu("View")
                self.root.update()
                for separator in popup._separators:
                    x0, y0, x1, y1 = separator._canvas.coords(separator._line_id)
                    self.assertEqual(y1 - y0, 1)
                    self.assertEqual(x0, separator.winfo_width() - x1)
                    self.assertEqual(y0, separator.winfo_height() - y1)
                    self.assertEqual((x0, y0, x1, y1), tuple(map(round, (x0, y0, x1, y1))))
                    self.assertEqual(separator._canvas.itemcget(separator._line_id, "fill"),
                                     COLORS.menu_separator)

    def test_checkmark_is_a_scalable_antialiased_path(self):
        popup = self.bar._popup_cache["View"]
        dark_index = next(i for i, item in enumerate(popup._items) if item.label == "Dark")
        row = popup._buttons[dark_index]
        for scale in (1, 1.25, 1.5, 2):
            with self.subTest(scale=scale):
                ctk.set_widget_scaling(scale)
                self.bar._open_menu("View")
                self.root.update()
                self.assertEqual(row._canvas.type(row._check_id), "line")
                effective_scale = row._get_widget_scaling()
                self.assertAlmostEqual(float(row._canvas.itemcget(row._check_id, "width")),
                                       1.5 * effective_scale)
                checked = ImageTk.getimage(row._canvas._aidas_surface_photo).convert("RGB")
                row.set_checked(False)
                self.root.update_idletasks()
                unchecked = ImageTk.getimage(row._canvas._aidas_surface_photo).convert("RGB")
                bounds = ImageChops.difference(checked, unchecked).getbbox()
                self.assertIsNotNone(bounds)
                left, top, right, bottom = bounds
                self.assertGreater(left, 0)
                self.assertLess(right, row.winfo_width())
                self.assertGreater(top, 0)
                self.assertLess(bottom, row.winfo_height())
                row.set_checked(True)


if __name__ == "__main__":
    unittest.main()
