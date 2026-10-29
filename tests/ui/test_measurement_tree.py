from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.services.measurement_inventory import scan_measurement_directory
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
        self.assertIn("No recorded Excel results", self._texts(tree))
        self.assertNotIn("other files", " ".join(self._texts(tree)))
        self.assertIn("1 recorded Excel result", tree.summary_var.get())
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
