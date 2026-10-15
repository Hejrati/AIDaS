from __future__ import annotations

import tkinter as tk
import unittest

from aidas.steps.step2_annotate import Step2BatchSegmentationTable


class Step2ProcessedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.root = None
        try:
            self.root = tk.Tk()
            self.root.withdraw()
            self.root.attributes("-alpha", 0)
            self.root.geometry("900x360+0+0")
            self.root.deiconify()
        except tk.TclError as exc:
            if self.root is not None:
                self.root.destroy()
            self.skipTest(f"A transparent Tk test window is unavailable: {exc}")

        self.addCleanup(self.root.destroy)
        self.table = Step2BatchSegmentationTable(self.root)
        self.table.pack(fill="both", expand=True)
        self.root.update()

    @staticmethod
    def _row(name, *, locked=False, include=False, status=None):
        return {
            "include": include,
            "locked": locked,
            "values": {
                "folder": name,
                "status": status or ("Already segmented: Light" if locked else "Ready: Light"),
                "images": "0" if locked else "1",
            },
        }

    def _click_checkbox(self, iid):
        x, y, width, height = self.table.tree.bbox(iid, "#0")
        self.table.tree.event_generate(
            "<Button-1>", x=x + width // 2, y=y + height // 2
        )
        self.root.update()

    def _invoke_select_all(self):
        self.root.tk.call(self.table.tree.heading("#0", "command"))
        self.root.update()

    def test_completed_source_and_direct_side_outputs_show_disabled_checks(self):
        rows = [
            self._row(name, locked=True, status=status)
            for name, status in (
                ("source-both", "Already segmented: nasal and temporal"),
                ("source/nasal", "Already segmented: nasal only"),
                ("source/temporal", "Already segmented: temporal only"),
            )
        ]
        ready = self._row("unsegmented")
        self.table.set_rows([*rows, ready])
        self.root.update()

        locked_image = self.table._checkbox_images["locked"]
        # Both arms of the gray check must be present; the old minus lacks them.
        self.assertEqual(locked_image.get(6, 10), (156, 163, 175))
        self.assertEqual(locked_image.get(11, 5), (156, 163, 175))
        self.assertEqual(locked_image.get(5, 5), (229, 231, 235))
        for iid, row in zip(("0", "1", "2"), rows):
            with self.subTest(side=row["values"]["folder"]):
                self.assertEqual(self.table.tree.item(iid, "image"), (str(locked_image),))
                self.assertIn("locked", self.table.tree.item(iid, "tags"))
                self.assertEqual(self.table.tree.item(iid, "values")[1], row["values"]["status"])
                self._click_checkbox(iid)
                self.assertFalse(row["include"])
                self.assertEqual(self.table.selected_rows(), [])

        self._click_checkbox("3")
        self.assertTrue(ready["include"])
        self.assertEqual(self.table.selected_rows(), [ready])

    def test_sources_with_one_saved_side_remain_checked_and_selectable(self):
        rows = [
            self._row(
                f"source-{side}",
                include=True,
                status=f"Ready: Light; segmented: {side} only",
            )
            for side in ("nasal", "temporal")
        ]
        self.table.set_rows(rows)
        self.root.update()

        for iid, row in zip(("0", "1"), rows):
            with self.subTest(side=row["values"]["folder"]):
                self.assertEqual(
                    self.table.tree.item(iid, "image"),
                    (str(self.table._checkbox_images["checked"]),),
                )
                self.assertNotIn("locked", self.table.tree.item(iid, "tags"))
                self.assertEqual(self.table.tree.item(iid, "values")[1], row["values"]["status"])
                self._click_checkbox(iid)
                self.assertFalse(row["include"])
                self.assertNotIn(row, self.table.selected_rows())
                self._click_checkbox(iid)
                self.assertTrue(row["include"])
                self.assertEqual(self.table.selected_rows(), rows)

    def test_select_all_only_changes_ready_rows(self):
        processed = self._row("source-both", locked=True, status="Already segmented: nasal and temporal")
        ready_a = self._row("source-nasal", include=True, status="Ready: Light; segmented: nasal only")
        ready_b = self._row("source-temporal", status="Ready: Light; segmented: temporal only")
        self.table.set_rows([processed, ready_a, ready_b])
        self.root.update()

        self._invoke_select_all()
        self.assertEqual(self.table.selected_rows(), [ready_a, ready_b])
        self.assertFalse(processed["include"])
        self._invoke_select_all()
        self.assertEqual(self.table.selected_rows(), [])
        self.assertFalse(processed["include"])
        self.assertEqual(
            self.table.tree.item("0", "image"),
            (str(self.table._checkbox_images["locked"]),),
        )

        self.table.set_rows([processed])
        self._invoke_select_all()
        self.assertFalse(processed["include"])
        self.assertEqual(self.table.selected_rows(), [])

    def test_hiding_and_restoring_processed_rows_preserves_their_disabled_checks(self):
        processed = self._row("source-both", locked=True, status="Already segmented: nasal and temporal")
        ready = [
            self._row(f"source-{side}", include=True, status=f"Ready: Light; segmented: {side} only")
            for side in ("nasal", "temporal")
        ]
        self.table.set_rows([processed, *ready])
        self.table.set_hide_processed(True)
        self.assertEqual(self.table.tree.get_children(""), ("1", "2"))
        self.assertEqual(self.table.selected_rows(), ready)

        self.table.set_hide_processed(False)
        self.assertEqual(self.table.tree.get_children(""), ("0", "1", "2"))
        self.assertEqual(
            self.table.tree.item("0", "image"),
            (str(self.table._checkbox_images["locked"]),),
        )
        self.assertFalse(processed["include"])
        self.assertEqual(self.table.selected_rows(), ready)


if __name__ == "__main__":
    unittest.main()
