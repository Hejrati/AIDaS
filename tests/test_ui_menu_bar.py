from __future__ import annotations

import inspect
import unittest

from aidas.ui.menu_bar import (
    ApplicationMenuBar,
    _PopupCommandRow,
    _PopupMenu,
)
from aidas.ui.theme import COLOR_PAIRS, CONTROLS, SHAPES


class _SelectionRow:
    def __init__(self):
        self.states = []

    def set_selected(self, selected):
        self.states.append(bool(selected))


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
    def test_top_level_menu_targets_share_one_compact_geometry(self):
        self.assertEqual(ApplicationMenuBar.MENU_NAMES, ("File", "View", "Help"))
        self.assertEqual(ApplicationMenuBar._BUTTON_WIDTH, 50)
        self.assertEqual(
            ApplicationMenuBar._BUTTON_HEIGHT
            + sum(ApplicationMenuBar._ROW_PADDING_Y),
            ApplicationMenuBar._BAR_HEIGHT - 1,
        )
        self.assertGreater(ApplicationMenuBar._BUTTON_GAP, 0)

    def test_popup_rows_use_standard_height_and_fixed_alignment_column(self):
        self.assertEqual(_PopupMenu._ITEM_HEIGHT, CONTROLS.height_sm)
        self.assertEqual(_PopupMenu._ITEM_MARGIN_X, 4)
        self.assertEqual(_PopupMenu._ITEM_GAP_Y, 0)
        self.assertGreater(_PopupMenu._CHECK_COLUMN_WIDTH, 0)
        self.assertEqual(
            _PopupMenu._MENU_WIDTH - (2 * _PopupMenu._ITEM_MARGIN_X),
            244,
        )

    def test_popup_selection_surface_uses_uniform_rounded_inset(self):
        source = inspect.getsource(_PopupCommandRow.__init__)

        self.assertIn("corner_radius=SHAPES.corner_radius_sm", source)
        self.assertEqual(_PopupMenu._ITEM_MARGIN_X, 4)
        self.assertEqual(_PopupMenu._EDGE_PADDING_Y, 4)
        self.assertEqual(_PopupMenu._ITEM_GAP_Y, 0)

    def test_popup_separator_is_compact_and_visible(self):
        source = inspect.getsource(_PopupMenu.__init__)

        self.assertEqual(_PopupMenu._SEPARATOR_GAP_Y, 2)
        self.assertIn("separator = tk.Frame(", source)
        self.assertIn("background=COLORS.border_strong", source)

    def test_popup_row_selection_colors_the_whole_two_column_surface(self):
        row = _PopupCommandRow.__new__(_PopupCommandRow)
        row.configure = _ColorTarget().configure
        row.check_label = _ColorTarget()
        row.text_label = _ColorTarget()

        row.set_selected(True)
        self.assertEqual(row.check_label.colors[-1], COLOR_PAIRS["primary_soft"])
        self.assertEqual(row.text_label.colors[-1], COLOR_PAIRS["primary_soft"])

        row.set_selected(False)
        self.assertEqual(row.check_label.colors[-1], COLOR_PAIRS["surface_elevated"])
        self.assertEqual(row.text_label.colors[-1], COLOR_PAIRS["surface_elevated"])

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
        next(item for item in view_items if item.label == "Dark").command()

        self.assertEqual(calls, ["browse", "exit", "updates", "about", "Dark"])
        checked = [item.label for item in view_items if item.checked]
        self.assertEqual(checked, ["Dark"])


if __name__ == "__main__":
    unittest.main()
