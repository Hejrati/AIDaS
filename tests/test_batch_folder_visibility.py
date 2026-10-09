from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.steps.step2_annotate import Step2BatchSegmentationSelectionPanel
from aidas.steps.step3_flatten import RBatchSelectionPanel
from aidas.steps.step4_analyze_isez import Step4BatchROISelectionPanel


class BatchFolderVisibilityTests(unittest.TestCase):
    def test_filter_layout_preserves_selections_and_restores_processed_folders(self):
        root = None
        try:
            try:
                root = ctk.CTk()
                root.withdraw()
                root.attributes("-alpha", 0)
                root.geometry("1024x680+0+0")
                root.deiconify()
            except tk.TclError as exc:
                self.skipTest(f"A transparent Tk test window is unavailable: {exc}")

            step = SimpleNamespace(
                status_var=tk.StringVar(master=root),
                _shared_core_budget=lambda: (8, 0, 8),
                _gpu_execution_available=lambda: True,
                _available_r_worker_limit=lambda: 8,
                _step2_core_allocation=lambda: 0,
                R_OUTPUT_MODE_PARALLEL="parallel",
                R_OUTPUT_MODE_SEQUENTIAL="sequential",
            )

            with tempfile.TemporaryDirectory() as temporary_directory:
                for panel_type in (
                    Step2BatchSegmentationSelectionPanel,
                    RBatchSelectionPanel,
                    Step4BatchROISelectionPanel,
                ):
                    with self.subTest(panel=panel_type.__name__), mock.patch.object(panel_type, "_start_scan"):
                        panel = panel_type(step, root, temporary_directory)
                        panel.pack(fill="both", expand=True)
                        root.update()
                        self.assertTrue(panel.hide_processed_var.get())
                        self.assertTrue(panel.hide_processed_check.winfo_viewable())
                        self.assertEqual(panel.hide_processed_check.cget("text"), "Hide processed folders")

                        # The first scan hides processed folders by default.
                        rows = [
                            {
                                "folder": str(Path(temporary_directory) / name),
                                "include": include,
                                "locked": locked,
                                "status": "Already processed" if locked else "Ready",
                                "outputs": "3/3" if locked else "0/3",
                                "image_paths": [str(Path(temporary_directory) / name / "Light.img")],
                                "values": {"folder": name, "status": "Ready", "inputs": "2"},
                            }
                            for name, include, locked in (
                                ("processed", False, True),
                                ("ready-a", True, False),
                                ("ready-b", False, False),
                            )
                        ]
                        panel._scan_done(rows, 584, 555,
                                         [(Path(temporary_directory) / "inaccessible", "Access denied")])
                        root.update()
                        table = panel.table
                        run_button = (panel.process_button if panel_type is Step4BatchROISelectionPanel
                                      else panel.next_button)

                        def assert_panel_layout():
                            table_top = table.winfo_rooty()
                            table_bottom = table_top + table.winfo_height()
                            self.assertLessEqual(panel.hide_processed_check.winfo_rooty()
                                                 + panel.hide_processed_check.winfo_height(), table_top)
                            self.assertGreater(panel.summary_row.winfo_rooty(), table_bottom)
                            self.assertLessEqual(panel.summary_label.winfo_rooty()
                                                 + panel.summary_label.winfo_height(),
                                                 run_button.winfo_rooty())
                            self.assertLessEqual(int(panel.summary_label.cget("wraplength")),
                                                 panel.summary_label.winfo_width())
                            self.assertGreaterEqual(panel.summary_label.winfo_height(),
                                                    panel.summary_label.winfo_reqheight())
                            self.assertEqual(panel.more_label.cget("text"), "More")
                            self.assertLessEqual(panel.more_label.winfo_rootx()
                                                 + panel.more_label.winfo_width(),
                                                 panel.summary_row.winfo_rootx()
                                                 + panel.summary_row.winfo_width())

                        assert_panel_layout()
                        root.geometry("620x680+0+0")
                        root.update()
                        assert_panel_layout()
                        root.geometry("1024x680+0+0")
                        root.update()
                        assert_panel_layout()

                        def visible_folders():
                            return [table.tree.item(iid, "values")[0]
                                    for iid in table.tree.get_children("")]

                        self.assertEqual(visible_folders(), ["ready-a", "ready-b"])
                        self.assertEqual(table.selected_rows(), [rows[1]])
                        self.assertEqual(len(panel.rows), 3)

                        # Select an additional ready folder through its checkbox.
                        ready_b = next(iid for iid in table.tree.get_children("")
                                       if table.tree.item(iid, "values")[0] == "ready-b")
                        _x, y, _width, height = table.tree.bbox(ready_b)
                        table.tree.event_generate("<Button-1>", x=20, y=y + height // 2)
                        root.update()
                        self.assertEqual(table.selected_rows(), rows[1:])

                        # Refreshing data also clears old detached row IDs.
                        table.set_rows(rows)
                        root.update()
                        self.assertEqual(visible_folders(), ["ready-a", "ready-b"])
                        self.assertEqual(table.selected_rows(), rows[1:])

                        panel.hide_processed_check.invoke()
                        root.update()
                        self.assertEqual(visible_folders(), ["processed", "ready-a", "ready-b"])
                        self.assertEqual(table.selected_rows(), rows[1:])
                        self.assertFalse(rows[0]["include"])

                        # Hiding every row has a distinct, reversible empty state.
                        panel._scan_done(rows[:1], 1, 0, [])
                        panel.hide_processed_check.invoke()
                        root.update()
                        table = panel.table
                        self.assertEqual(visible_folders(), ["No unprocessed folders to show."])
                        self.assertEqual(table.selected_rows(), [])
                        self.assertIn("disabled", run_button.state())
                        panel.hide_processed_check.invoke()
                        root.update()
                        self.assertEqual(visible_folders(), ["processed"])
                        self.assertTrue(rows[0]["locked"])
                        self.assertEqual(table.selected_rows(), [])

                        # A genuinely empty scan retains its original message.
                        panel._scan_done([], 0, 0, [])
                        panel.hide_processed_check.invoke()
                        root.update()
                        table = panel.table
                        self.assertNotEqual(visible_folders(), ["No unprocessed folders to show."])
                        self.assertEqual(table.selected_rows(), [])
                        panel.destroy()
                        root.update()
        finally:
            if root is not None:
                root.destroy()
