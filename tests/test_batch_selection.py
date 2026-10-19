import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest

import customtkinter as ctk

from aidas.steps.step2_annotate import Step2BatchSegmentationTable
from aidas.steps.step3_flatten import RBatchSelectionTable
from aidas.steps.step4_analyze_isez import Step4BatchROITable
from aidas.ui.theme import configure_ttk_styles, get_interface_mode, set_interface_mode


class BatchSelectionTests(unittest.TestCase):
    def test_selection_and_filtering_preserve_locked_rows_in_both_interfaces(self):
        root = None
        previous_mode = get_interface_mode()
        try:
            try:
                root = ctk.CTk()
                root.withdraw()
                root.attributes("-alpha", 0)
                root.geometry("900x300+0+0")
                root.deiconify()
            except tk.TclError as exc:
                self.skipTest(f"Tk is unavailable: {exc}")
            for mode in ("Modern", "Classic"):
                set_interface_mode(mode)
                configure_ttk_styles(ttk.Style(root), "Light")
                # Installing the themed renderer can withdraw the test window.
                root.deiconify()
                root.update()
                for table_type in (Step2BatchSegmentationTable, RBatchSelectionTable, Step4BatchROITable):
                    with self.subTest(mode=mode, table=table_type.__name__):
                        counts = []
                        table = table_type(root, on_selection_changed=counts.append)
                        table.pack(fill="both", expand=True)
                        rows = [
                            {"include": include, "locked": locked,
                             "values": {"folder": name, "status": "Ready", "images": "1", "inputs": "2"},
                             "folder": name, "status": "Ready", "outputs": "0/3"}
                            for name, include, locked in (
                                ("locked", False, True), ("selected", True, False), ("ready", False, False)
                            )
                        ]
                        table.set_rows(rows)
                        root.update()
                        self.assertEqual(counts[-1], 1)
                        self.assertEqual(table.selected_rows(), [rows[1]])

                        def click(iid, x=20):
                            _left, top, _width, height = table.tree.bbox(iid)
                            return table._on_click(SimpleNamespace(x=x, y=top + height // 2))

                        before = len(counts)
                        self.assertEqual(click("0"), "break")
                        self.assertEqual(len(counts), before)
                        self.assertIsNone(click("2", x=100))
                        self.assertEqual(len(counts), before)
                        self.assertEqual(click("2"), "break")
                        self.assertEqual(table.selected_rows(), rows[1:])
                        self.assertEqual(counts[-1], 2)
                        table.set_hide_processed(True)
                        self.assertEqual(table.tree.get_children(""), ("1", "2"))
                        table._toggle_all_ready()
                        self.assertEqual(table.selected_rows(), [])
                        table._toggle_all_ready()
                        self.assertEqual(table.selected_rows(), rows[1:])
                        table.set_hide_processed(False)
                        self.assertEqual(table.tree.get_children(""), ("0", "1", "2"))
                        self.assertFalse(rows[0]["include"])
                        table.set_rows(rows[:1])
                        before = len(counts)
                        table._toggle_all_ready()
                        self.assertEqual(len(counts), before)
                        self.assertEqual(table.selected_rows(), [])
                        table.destroy()
                        root.update()
        finally:
            set_interface_mode(previous_mode)
            if root is not None:
                root.destroy()
