"""Regression coverage for retained data views and native caption updates."""

from pathlib import Path
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import customtkinter as ctk
from PIL import Image, ImageTk

from aidas.services.measurement_inventory import DirectoryInventory, DirectoryNode
from aidas.ui.measurement_tree import MeasurementDirectoryTree
from aidas.ui.result_grid import ResultGridCanvas
from aidas.ui.theme import COLORS, get_interface_mode, set_interface_mode
from aidas.ui.title_bar import CustomWindowsTitleBar


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class DataViewOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.interface = get_interface_mode()
        self.appearance = ctk.get_appearance_mode()
        self.addCleanup(set_interface_mode, self.interface, redraw=False)
        self.addCleanup(ctk.set_appearance_mode, self.appearance)
        set_interface_mode("Modern", redraw=False)
        ctk.set_appearance_mode("Light")
        try:
            self.root = _QuietRoot()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.attributes("-alpha", 0)
        self.root.geometry("900x600+10000+10000")
        self.errors = []
        self.root.report_callback_exception = lambda *error: self.errors.append(error)
        self.addCleanup(self.root.destroy)

    def tearDown(self):
        self.assertFalse(self.errors, self.errors)

    @staticmethod
    def inventory(name="Example", *, suffix=""):
        directory = Path(name)
        children = tuple(
            DirectoryNode(directory / f"Results_{index}{suffix}.xlsx",
                          f"Results_{index}{suffix}.xlsx", "excel", "Step 4 measurements")
            for index in range(20)
        )
        children += (DirectoryNode(directory / "notes.txt", "notes.txt", "other", "Other file"),)
        root = DirectoryNode(directory, name, "folder", "20 recorded Excel results",
                             children=children, excel_files=20)
        return DirectoryInventory(root, 20, 0, 0, 0, ())

    def tree(self):
        tree = MeasurementDirectoryTree(self.root)
        tree.pack(fill="both", expand=True)
        self.root.update()
        tree._inventory = self.inventory()
        return tree

    def test_reexpanding_folder_reuses_text_measurements_without_changing_widths(self):
        tree = self.tree()
        with patch.object(tree._name_font, "measure", wraps=tree._name_font.measure) as names, \
                patch.object(tree._detail_font, "measure", wraps=tree._detail_font.measure) as details:
            tree._rebuild_rows()
            expected = (tree._detail_x, tree._content_width)
            self.assertLessEqual(details.call_count, 2)
            names.reset_mock()
            details.reset_mock()
            tree._closed.add(tree._inventory.root.path)
            tree._rebuild_rows()
            tree.expand_all()
            names.assert_not_called()
            details.assert_not_called()
        self.assertEqual((tree._detail_x, tree._content_width), expected)

    def test_font_and_dpi_changes_remeasure_text_before_computing_scroll_width(self):
        tree = self.tree()
        tree._rebuild_rows()
        original_scaling = float(self.root.tk.call("tk", "scaling"))
        self.addCleanup(self.root.tk.call, "tk", "scaling", original_scaling)
        for change in (
            lambda: tree._name_font.configure(size=17),
            lambda: self.root.tk.call("tk", "scaling", original_scaling * 1.25),
        ):
            with patch.object(tree._name_font, "measure", wraps=tree._name_font.measure) as measure:
                change()
                tree._rebuild_rows()
                self.assertGreater(measure.call_count, 0)
            expected = max(70 + depth * tree._indent + tree._name_font.measure(node.name) + 40
                           for node, depth, _, _ in tree._rows)
            self.assertEqual(tree._detail_x, expected)

    def test_filter_and_new_scan_keep_child_cache_current(self):
        tree = self.tree()
        tree._rebuild_rows()
        inventory = tree._inventory
        first = tree._display_children(inventory.root)
        self.assertIs(first, tree._display_children(inventory.root))
        self.assertNotIn("notes.txt", [node.name for node in first])
        tree.hide_other_files_var.set(False)
        tree._filter_changed()
        self.assertIn("notes.txt", [node.name for node in tree._display_children(inventory.root)])
        tree.hide_other_files_var.set(True)
        tree._filter_changed()
        updated = self.inventory(suffix="_new")
        tree._directory = inventory.root.path
        with patch("aidas.ui.measurement_tree.scan_measurement_directory", return_value=updated):
            tree.refresh()
            deadline = time.monotonic() + 5
            while tree._inventory is None:
                self.root.update()
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)
        self.root.update()
        self.assertTrue(any(node.name == "Results_0_new.xlsx" for node, _, _, _ in tree._rows))
        self.assertFalse(any(node.name == "Results_0.xlsx" for node, _, _, _ in tree._rows))

    def test_duplicate_viewport_draws_retain_items_and_selection_theme_changes_repaint(self):
        tree = self.tree()
        tree._rebuild_rows()
        self.root.update()
        items = tree.canvas.find_all()
        for _ in range(5):
            tree._schedule_draw()
            self.root.update_idletasks()
        self.assertEqual(tree.canvas.find_all(), items)
        tree._click(SimpleNamespace(x=80, y=int(12 + tree._row_height * 1.5)))
        self.root.update()
        self.assertNotEqual(tree.canvas.find_all(), items)
        rectangles = [item for item in tree.canvas.find_all() if tree.canvas.type(item) == "rectangle"]
        self.assertTrue(any(tree.canvas.itemcget(item, "fill") == COLORS.selection for item in rectangles))
        items = tree.canvas.find_all()
        ctk.set_appearance_mode("Dark")
        tree._apply_aidas_theme()
        self.root.update()
        self.assertNotEqual(tree.canvas.find_all(), items)
        self.assertEqual(tree.canvas.cget("background"), COLORS.surface)

    def grid(self):
        image = Image.new("RGB", (80, 60), "#123456")
        reads = []

        def load(index):
            reads.append(index)
            return [dict(title=None, previews=[("Preview", image, (0, 0, 80, 60))],
                         margins=(0, 0, 0, 0), aspect=0.75)]

        self.expand_icon = ImageTk.PhotoImage(Image.new("RGB", (8, 8), "white"), master=self.root)
        grid = ResultGridCanvas(
            self.root, groups=[dict(root=Path("study"), label="study", sided=False,
                                    folders={"result": Path("study")})],
            preview_titles=["Preview"], load_page=load,
            preview_layout=lambda _size, _bounds, _margins, target, _aspect: (0, 0, *target),
            on_zoom=lambda *_args: None, on_open_folder=lambda *_args: None,
            palette={"surface": "#ffffff", "text": "#000000", "link": "#0000ff",
                     "border": "#888888", "button": "#eeeeee", "button_hover": "#dddddd"},
            expand_icon=self.expand_icon, borderwidth=0, highlightthickness=0,
        )
        grid.pack(fill="both", expand=True)
        self.settle_grid(grid)
        return grid, reads

    def settle_grid(self, grid):
        deadline = time.monotonic() + 5
        while True:
            self.root.update()
            if (grid._page_size is not None and grid._refresh_after_id is None
                    and grid._visible_pages and all(grid._pages[index]["ready"]
                                                    for index in grid._visible_pages)):
                return
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

    def test_resize_burst_uses_last_size_once_and_preserves_decoded_sources(self):
        grid, reads = self.grid()
        generation = grid._generation
        width, height = grid._page_size
        source = grid._pages[0]["previews"][0]["image"]
        for change in range(1, 16):
            grid._resize_pages(SimpleNamespace(width=width - change, height=height - change))
        self.settle_grid(grid)
        self.assertEqual(grid._generation, generation + 1)
        self.assertEqual(grid._page_size, (width - 15, height - 15))
        self.assertIs(grid._pages[0]["previews"][0]["image"], source)
        self.assertEqual(reads, [0])

    def test_resize_burst_returning_to_original_size_preserves_photos_and_items(self):
        grid, reads = self.grid()
        width, height = grid._page_size
        generation = grid._generation
        page = grid._pages[0]
        photo = page["previews"][0]["photo"]
        items = list(page["items"])
        for change in range(1, 16):
            grid._resize_pages(SimpleNamespace(width=width - change, height=height - change))
        grid._resize_pages(SimpleNamespace(width=width, height=height))
        self.settle_grid(grid)
        self.assertEqual(grid._generation, generation)
        self.assertIs(grid._pages[0], page)
        self.assertIs(grid._pages[0]["previews"][0]["photo"], photo)
        self.assertEqual(grid._pages[0]["items"], items)
        self.assertTrue(set(items).issubset(grid.find_all()))
        self.assertEqual(reads, [0])


class TitleBarOptimizationTests(unittest.TestCase):
    def test_same_edge_pointer_motion_skips_cursor_reconfiguration_and_restores_original(self):
        bar = object.__new__(CustomWindowsTitleBar)
        bar._cursor_widget = None
        bar._cursor_before_resize = ""
        bar._resize_cursor = ""
        widget = SimpleNamespace(cget=Mock(return_value="hand2"), configure=Mock())
        for _ in range(10):
            bar._set_resize_cursor(widget, "n")
        widget.configure.assert_called_once_with(cursor="sb_v_double_arrow")
        bar._set_resize_cursor(widget, "ne")
        self.assertEqual(widget.configure.call_args.kwargs, {"cursor": "size_ne_sw"})
        bar._set_resize_cursor(None, None)
        self.assertEqual(widget.configure.call_args.kwargs, {"cursor": "hand2"})

    def test_root_movement_preserves_maximize_glyph_until_window_state_changes(self):
        bar = object.__new__(CustomWindowsTitleBar)
        bar._state_sync_after_id = None
        bar._last_maximized = None
        bar.controller = SimpleNamespace(correct_maximized_bounds=Mock(),
                                         is_maximized=Mock(return_value=False))
        bar.maximize_button = SimpleNamespace(configure=Mock())
        for _ in range(10):
            bar._sync_window_state()
        bar.maximize_button.configure.assert_called_once_with(text="\u25a1")
        self.assertEqual(bar.controller.correct_maximized_bounds.call_count, 10)
        bar.controller.is_maximized.return_value = True
        bar._sync_window_state()
        self.assertEqual(bar.maximize_button.configure.call_count, 2)
        self.assertEqual(bar.maximize_button.configure.call_args.kwargs, {"text": "\u2750"})


if __name__ == "__main__":
    unittest.main()
