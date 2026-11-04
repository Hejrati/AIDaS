from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.services.measurement_inventory import (
    DirectoryInventory, DirectoryNode, scan_measurement_directory,
)
from aidas.steps.step5_compile import Step5Frame
from aidas.ui.measurement_tree import MeasurementDirectoryTree
from aidas.ui.theme import COLORS, apply_appearance_mode, get_interface_mode, set_interface_mode


class MeasurementDirectoryTreeTests(unittest.TestCase):
    def setUp(self):
        self.original_interface = get_interface_mode()
        self.original_appearance = ctk.get_appearance_mode()
        set_interface_mode("Modern", redraw=False)
        self.addCleanup(set_interface_mode, self.original_interface, redraw=False)
        self.addCleanup(ctk.set_appearance_mode, self.original_appearance)
        try:
            self.root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.attributes("-alpha", 0)
        self.root.geometry("1300x820+10000+10000")
        self.addCleanup(self.root.destroy)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)

    def _wait(self, condition):
        deadline = time.monotonic() + 10
        while not condition():
            self.assertLess(time.monotonic(), deadline, "Directory scan did not finish")
            self.root.update()
            time.sleep(.005)
        self.root.update()
        self.assertFalse(self.callback_errors)

    def _tree(self):
        tree = MeasurementDirectoryTree(self.root)
        tree.pack(fill="both", expand=True)
        self.root.update()
        return tree

    def _texts(self, tree):
        return [tree.canvas.itemcget(item, "text") for item in tree.canvas.find_all() if tree.canvas.type(item) == "text"]

    def _show_nodes(self, tree, children):
        count = sum(node.excel_files for node in children)
        root = DirectoryNode(
            self.directory, "Subject", "folder", "Subject ID", tuple(children),
            excel_files=count,
        )
        tree._filtered_children.clear()
        tree._inventory = DirectoryInventory(root, count, 0, 0, 0, ())
        tree._rebuild_rows()
        self.root.update()

    def _assert_scrollbars_disabled(self, tree):
        self.assertTrue(tree._xbar.instate(("disabled",)))
        self.assertTrue(tree._ybar.instate(("disabled",)))
        self.assertEqual(tree.canvas.xview(), (0.0, 1.0))
        self.assertEqual(tree.canvas.yview(), (0.0, 1.0))

    def test_empty_and_short_trees_disable_scrollbars_and_ignore_scroll_input(self):
        tree = self._tree()
        self.root.geometry("650x520+10000+10000")
        self.root.update()
        self.assertEqual(tree.path_var.get(), "")
        self._assert_scrollbars_disabled(tree)
        message = tree.canvas.bbox("all")
        self.assertLessEqual(message[2], tree.canvas.winfo_width() - 1)

        short = DirectoryNode(
            self.directory / "Results.xlsx", "Results.xlsx", "excel",
            "Step 4 measurements", excel_files=1,
        )
        for children in (None, (short,)):
            with self.subTest(empty=children is None):
                if children is not None:
                    self._show_nodes(tree, children)
                self._assert_scrollbars_disabled(tree)
                origin = tree.canvas.canvasx(0), tree.canvas.canvasy(0)
                for units in (-10, 10):
                    tree._scroll(units)
                    tree._wheel(SimpleNamespace(delta=units * 120))
                tree._yview("moveto", 1)
                for bar in (tree._xbar, tree._ybar):
                    self.root.tk.call(bar.cget("command"), "moveto", 1)
                self.root.update()
                self._assert_scrollbars_disabled(tree)
                self.assertEqual((tree.canvas.canvasx(0), tree.canvas.canvasy(0)), origin)

    def test_horizontal_and_vertical_overflow_enable_only_the_needed_scrollbar(self):
        tree = self._tree()
        tall = tuple(DirectoryNode(
            self.directory / f"Results_{index}.xlsx", f"Results_{index}.xlsx",
            "excel", "Step 4 measurements", excel_files=1,
        ) for index in range(40))
        self._show_nodes(tree, tall)
        self.assertTrue(tree._xbar.instate(("disabled",)))
        self.assertFalse(tree._ybar.instate(("disabled",)))
        tree._scroll(10)
        self.root.update()
        self.assertGreater(tree.canvas.yview()[0], 0)

        name = "Results_" + "wide_" * 40 + ".xlsx"
        wide = DirectoryNode(self.directory / name, name, "excel", "Step 4 measurements", excel_files=1)
        self._show_nodes(tree, (wide,))
        self.assertFalse(tree._xbar.instate(("disabled",)))
        self.assertTrue(tree._ybar.instate(("disabled",)))
        self.assertEqual(tree.canvas.yview(), (0.0, 1.0))
        tree.canvas.xview_moveto(1)
        self.root.update()
        self.assertGreater(tree.canvas.xview()[0], 0)

        inset = int(tree.canvas.cget("highlightthickness")) + int(tree.canvas.cget("borderwidth"))
        viewport_width = tree.canvas.winfo_width() - 2 * inset
        for overflow in (0, 1):
            with self.subTest(horizontal_overflow_pixels=overflow):
                tree._content_width = viewport_width + overflow
                tree._resize()
                self.root.update()
                self.assertEqual(tree._xbar.instate(("disabled",)), overflow == 0)

    def test_filter_and_collapse_disable_scrolled_axes_and_expand_restores_them(self):
        tree = self._tree()
        excel = DirectoryNode(
            self.directory / "nasal" / "Results.xlsx", "Results.xlsx", "excel",
            "Step 4 measurements", excel_files=1,
        )
        others = tuple(DirectoryNode(
            self.directory / "nasal" / f"Image_{index}.img",
            f"Image_{index}_" + "wide_" * 40, "image", "Flattened image",
        ) for index in range(30))
        folder = DirectoryNode(
            self.directory / "nasal", "nasal", "folder", "Nasal",
            (excel, *others), excel_files=1,
        )
        tree.hide_other_files_var.set(False)
        self._show_nodes(tree, (folder,))

        def scroll_to_end():
            self.assertFalse(tree._xbar.instate(("disabled",)))
            self.assertFalse(tree._ybar.instate(("disabled",)))
            tree.canvas.xview_moveto(1)
            tree.canvas.yview_moveto(1)
            self.root.update()
            self.assertGreater(tree.canvas.xview()[0], 0)
            self.assertGreater(tree.canvas.yview()[0], 0)

        scroll_to_end()
        tree.hide_other_files_checkbox.invoke()
        self.root.update()
        self._assert_scrollbars_disabled(tree)
        tree.hide_other_files_checkbox.invoke()
        self.root.update()
        scroll_to_end()
        tree.collapse_all()
        self.root.update()
        self._assert_scrollbars_disabled(tree)
        tree.expand_all()
        self.root.update()
        scroll_to_end()

    def test_resizing_to_fit_resets_vertical_scroll_and_shrinking_restores_it(self):
        tree = self._tree()
        self.root.geometry("1300x350+10000+10000")
        children = tuple(DirectoryNode(
            self.directory / f"Results_{index}.xlsx", f"Results_{index}.xlsx",
            "excel", "Step 4 measurements", excel_files=1,
        ) for index in range(10))
        self._show_nodes(tree, children)
        self.assertFalse(tree._ybar.instate(("disabled",)))
        tree._yview("moveto", 1)
        self.root.update()
        self.assertGreater(tree.canvas.yview()[0], 0)
        self.root.geometry("1300x820+10000+10000")
        self.root.update()
        self._assert_scrollbars_disabled(tree)
        tree._scroll(10)
        self.root.update()
        self.assertEqual(tree.canvas.yview(), (0.0, 1.0))
        self.root.geometry("1300x350+10000+10000")
        self.root.update()
        self.assertFalse(tree._ybar.instate(("disabled",)))
        tree._scroll(5)
        self.root.update()
        self.assertGreater(tree.canvas.yview()[0], 0)

    def test_step5_replaces_logs_and_progress_with_actual_directory_contents(self):
        folder = self.directory / "flip human - Copy - Copy (12)"
        folder.mkdir()
        for name in ("rr_MCPAR.xlsx", "_thickness_vs_distance_from_fovea_LIGHT.txt", "_fovea_light_profiles_LIGHT.txt"):
            (folder / name).touch()
        frame = Step5Frame(self.root, source_step=SimpleNamespace(batch_roi_root=folder))
        self.assertEqual(frame.input_var.get(), "")
        self.assertTrue(frame.set_input_folder(folder))
        frame.pack(fill="both", expand=True)
        tree = frame.directory_view
        self._wait(lambda: tree._inventory is not None)
        self.assertNotIn("progress", frame.__dict__)
        self.assertNotIn("log_text", frame.__dict__)
        self.assertIn(folder.name, self._texts(tree))
        self.assertIn("rr_MCPAR.xlsx", self._texts(tree))
        self.assertTrue(tree.hide_other_files_var.get())
        self.assertNotIn("ELM-RPE / ONL thickness", self._texts(tree))
        tree.hide_other_files_checkbox.invoke()
        self.root.update()
        self.assertIn("ELM-RPE / ONL thickness", self._texts(tree))
        self.assertIn("outside nasal/temporal", tree.notice_var.get())
        self.assertEqual(tree.path_var.get(), str(folder))

        changed = self.directory / "New folder"
        changed.mkdir()
        (changed / "Results.xlsx").touch()
        frame.set_input_folder(changed)
        self._wait(lambda: tree._inventory is not None and tree._inventory.root.path == changed)
        self.assertIn("Results.xlsx", self._texts(tree))
        self.assertNotIn("rr_MCPAR.xlsx", self._texts(tree))

    def test_excel_only_filter_hides_other_files_without_rescanning_or_losing_folders(self):
        side = self.directory / "Control" / "Subject_A" / "nasal"
        side.mkdir(parents=True)
        for name in ("rr_MCPAR.xlsx", "_thickness_vs_distance_from_fovea_LIGHT.txt", "_flat_LIGHT.img", "notes.txt"):
            (side / name).touch()
        notes = self.directory / "Notes"
        notes.mkdir()
        (notes / "notes.txt").touch()
        (self.directory / "rrMCPAR_ELMRPE_ONL.xlsx").touch()
        tree = self._tree()
        tree.set_directory(self.directory)
        self._wait(lambda: tree._inventory is not None)
        inventory = tree._inventory
        generation = tree._generation
        files = [node for node, _, _, _ in tree._rows if node.kind != "folder"]
        self.assertEqual({node.name for node in files}, {"rr_MCPAR.xlsx", "rrMCPAR_ELMRPE_ONL.xlsx"})
        self.assertIn("Subject_A", self._texts(tree))
        self.assertIn("No recorded Step 4 measurements", self._texts(tree))
        self.assertNotIn("other files", " ".join(self._texts(tree)))
        self.assertIn("1 recorded Step 4 result", tree.summary_var.get())
        tree.hide_other_files_checkbox.invoke()
        self.root.update()
        self.assertIn("ELM-RPE / ONL thickness", self._texts(tree))
        self.assertIn("Flattened image data", self._texts(tree))
        self.assertTrue(any(node.kind == "other" for node, _, _, _ in tree._rows))
        self.assertIs(tree._inventory, inventory)
        self.assertEqual(tree._generation, generation)
        tree.hide_other_files_checkbox.invoke()
        self.root.update()
        self.assertTrue(all(node.kind in ("folder", "excel", "output") for node, _, _, _ in tree._rows))

    def test_manual_step5_export_stops_polling_and_completion_reentry_only_prefills(self):
        from openpyxl import Workbook, load_workbook
        from aidas.services.step4_compiler import compile_step4_results

        side = self.directory / "Control" / "Subject_A" / "nasal"
        side.mkdir(parents=True)
        workbook = Workbook()
        workbook.active.cell(1, 7, "Round")
        workbook.active.cell(2, 7, 0.75)
        workbook.save(side / "rr_MCPAR.xlsx")
        workbook.close()
        frame = Step5Frame(
            self.root, source_step=SimpleNamespace(batch_roi_root=self.directory),
        )
        frame.pack(fill="both", expand=True)
        frame.on_compilation_complete = lambda _result: frame.start_from_step4(self.directory)
        with mock.patch(
            "aidas.services.step4_compiler.compile_step4_results", wraps=compile_step4_results,
        ) as compile_workbook, mock.patch("aidas.steps.step5_compile.messagebox.showerror") as error:
            frame.start_from_step4(self.directory)
            self.assertIsNone(frame._worker)
            self.assertFalse(frame._running)
            self.assertFalse((self.directory / "rrMCPAR_ELMRPE_ONL.xlsx").exists())
            compile_workbook.assert_not_called()

            self._wait(lambda: frame._input_scan_state == "ready")
            frame._run_clicked()
            self._wait(lambda: (
                not frame._running and frame.directory_view._inventory is not None
                and not frame._pending_step4_handoffs and frame._poll_after_id is None
                and frame.directory_view._poll_id is None
            ))

        compile_workbook.assert_called_once()
        error.assert_not_called()
        output = self.directory / "rrMCPAR_ELMRPE_ONL.xlsx"
        self.assertTrue(output.is_file())
        workbook = load_workbook(output, data_only=True)
        try:
            self.assertEqual(workbook.sheetnames, ["rrMCP-AR", "ELM-RPE", "Extraction_log"])
            self.assertEqual(workbook["rrMCP-AR"]["A3"].value, "Subject_A")
            self.assertEqual(workbook["rrMCP-AR"]["C3"].value, 0.75)
        finally:
            workbook.close()
        self.assertEqual(frame._pending_step4_handoffs, [])
        self.assertIsNone(frame._poll_after_id)
        self.assertIsNone(frame.directory_view._poll_id)
        self.assertIn("Ready to create an Excel file", frame.status_var.get())

    def test_roi_measurements_remain_visible_and_cleaned_excel_is_a_thickness_table(self):
        from PIL import Image
        from aidas.services.step4_results import encode_step4_measurements
        from aidas.services.step4_compiler import create_cleaned_light_thickness_workbook

        side = self.directory / "Control" / "Subject_A" / "nasal"
        side.mkdir(parents=True)
        rows = [dict(Major=10.0, Minor=5.0, Angle=0.0, **{
            "Circ.": 0.5, "AR": 2.0, "Round": 0.5, "Solidity": 0.9,
        })]
        Image.new("L", (8, 8)).save(
            side / "ROI_to_move_stck.tif", description=encode_step4_measurements(rows),
        )
        raw = side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
        raw.write_text("Distance\tWhole\tELM\tONL\n0\t0\t10\t20\n", encoding="utf-8")
        create_cleaned_light_thickness_workbook(raw)
        tree = self._tree()
        tree.set_directory(self.directory)
        self._wait(lambda: tree._inventory is not None)

        self.assertEqual(tree._inventory.workbooks, 1)
        self.assertEqual(tree._inventory.thickness_files, 2)
        self.assertIn("ROI_to_move_stck.tif", self._texts(tree))
        self.assertNotIn("_thickness_vs_distance_from_fovea_LIGHT2.xlsx", self._texts(tree))
        tree.hide_other_files_checkbox.invoke()
        self.root.update()
        self.assertIn("_thickness_vs_distance_from_fovea_LIGHT2.xlsx", self._texts(tree))
        self.assertEqual(
            [node.kind for node, *_rest in tree._rows if node.name.endswith("LIGHT2.xlsx")],
            ["thickness"],
        )
        self.assertIn("1 recorded Step 4 result", tree.summary_var.get())

    def test_folders_expand_collapse_and_new_output_appears_after_refresh(self):
        side = self.directory / "Control" / "Subject_A" / "nasal"
        side.mkdir(parents=True)
        (side / "rr_MCPAR.xlsx").touch()
        tree = self._tree()
        tree.set_directory(self.directory)
        self._wait(lambda: tree._inventory is not None)
        expanded_count = len(tree._rows)
        self.assertIn("rr_MCPAR.xlsx", self._texts(tree))
        tree.collapse_all()
        self.root.update()
        self.assertLess(len(tree._rows), expanded_count)
        self.assertNotIn("rr_MCPAR.xlsx", self._texts(tree))
        tree.expand_all()
        self.root.update()
        self.assertEqual(len(tree._rows), expanded_count)
        self.assertIn("rr_MCPAR.xlsx", self._texts(tree))
        (self.directory / "rrMCPAR_ELMRPE_ONL.xlsx").touch()
        tree.refresh()
        self._wait(lambda: tree._inventory is not None)
        self.assertIn("Compiled Excel workbook", self._texts(tree))

    def test_slow_old_scan_cannot_replace_new_directory_and_scans_stay_off_tk(self):
        old = self.directory / "Old"
        new = self.directory / "New"
        old.mkdir()
        new.mkdir()
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)
        owner = threading.get_ident()
        worker_threads = []

        def scan(directory, **options):
            worker_threads.append(threading.get_ident())
            if directory == old:
                entered.set()
                release.wait(5)
                # Simulate a completed stale response even if cancellation
                # arrived while the filesystem was blocked.
                result = scan_measurement_directory(directory)
                finished.set()
                return result
            return scan_measurement_directory(directory, **options)

        tree = self._tree()
        with mock.patch("aidas.ui.measurement_tree.scan_measurement_directory", side_effect=scan):
            tree.set_directory(old)
            self._wait(entered.is_set)
            tree.set_directory(new)
            self._wait(lambda: tree._inventory is not None)
            self.assertEqual(tree._inventory.root.path, new)
            release.set()
            self._wait(finished.is_set)
            tree.refresh()
            self._wait(lambda: tree._inventory is not None)
        self.assertEqual(tree._inventory.root.path, new)
        self.assertTrue(worker_threads)
        self.assertNotIn(owner, worker_threads)

    def test_large_directory_draws_only_visible_rows_and_supports_scrolling(self):
        for index in range(1000):
            (self.directory / f"Results_{index:04}.xlsx").touch()
        tree = self._tree()
        tree.set_directory(self.directory)
        self._wait(lambda: tree._inventory is not None)
        self.assertEqual(len(tree._rows), 1001)
        self.assertLess(len(tree.canvas.find_all()), 400)
        tree.canvas.yview_moveto(1)
        self.root.update()
        self.assertIn("Results_0999.xlsx", self._texts(tree))
        tree.canvas.yview_moveto(0)
        tree._click(SimpleNamespace(x=80, y=int(12 + tree._row_height * 1.5)))
        self.root.update()
        self.assertTrue(tree.selection_var.get().endswith("Results_0000.xlsx"))

    def test_canvas_tracks_light_dark_and_classic_palette_changes(self):
        tree = self._tree()
        tree.set_directory(self.directory)
        self._wait(lambda: tree._inventory is not None)
        for interface, appearance in (("Modern", "Light"), ("Modern", "Dark"), ("Classic", "Light")):
            set_interface_mode(interface, redraw=False)
            apply_appearance_mode(appearance, root=self.root)
            tree._apply_aidas_theme()
            self.root.update()
            self.assertEqual(tree.canvas.cget("background"), COLORS.surface)
            self.assertIn(self.directory.name, self._texts(tree))


if __name__ == "__main__":
    unittest.main()
