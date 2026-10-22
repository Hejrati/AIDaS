from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import customtkinter as ctk

from aidas.steps.step2_annotate import Step2BatchSegmentationSelectionPanel, Step2Frame


class Step2CompletedFolderPromptTests(unittest.TestCase):
    def setUp(self):
        self.root = None
        try:
            self.root = ctk.CTk()
            self.root.withdraw()
            self.root.attributes("-alpha", 0)
            self.root.geometry("900x600+10000+10000")
            self.root.deiconify()
        except tk.TclError as exc:
            if self.root is not None:
                self.root.destroy()
            self.skipTest(str(exc))
        self.addCleanup(self.root.destroy)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.frame = Step2Frame.__new__(Step2Frame)
        self.step = SimpleNamespace(
            status_var=tk.StringVar(master=self.root),
            on_continue_to_step3=Mock(),
            _close_step2_batch_segmentation_panel=Mock(),
            _step2_override_rows=self.frame._step2_override_rows,
            _start_step2_batch_segmentation_from_rows=Mock(),
        )

    def _source(self, folder, *, marked=False):
        folder.mkdir(parents=True, exist_ok=True)
        names = ["Light.img", "Light.hdr"]
        if marked:
            names += ["Light_MARKED.img", "Light_MARKED.hdr"]
        for name in names:
            (folder / name).write_bytes(name.encode())

    def _completed_sides(self):
        self._source(self.folder)
        for side in ("NaSaL", "TEMPORAL"):
            self._source(self.folder / side, marked=True)
        return self.frame._scan_step2_batch_segmentation_folders(self.folder)[0]

    def _panel(self, rows):
        panel = Step2BatchSegmentationSelectionPanel(
            self.step, self.root, self.folder, initial_rows=rows,
        )
        panel.pack(fill="both", expand=True)
        self.root.update()
        return panel

    def test_completed_images_offer_step3_and_pass_each_output_folder_once(self):
        panel = self._panel(self._completed_sides())
        self.assertTrue(panel.processed_notice.winfo_viewable())
        self.assertTrue(panel.go_to_step3_button.winfo_viewable())
        self.assertTrue(panel.override_button.winfo_viewable())
        self.assertFalse(panel.next_button.winfo_viewable())
        panel.go_to_step3_button.invoke()
        folders = self.step.on_continue_to_step3.call_args.args[0]
        self.assertCountEqual(folders, [str(self.folder / "NaSaL"), str(self.folder / "TEMPORAL")])
        self.assertEqual(len(folders), 2)
        self.step._close_step2_batch_segmentation_panel.assert_called_once_with(restore_previous=True)
        self.step._start_step2_batch_segmentation_from_rows.assert_not_called()

    def test_override_requeues_original_source_and_preserves_saved_files(self):
        rows = self._completed_sides()
        before = {str(path): path.read_bytes() for path in self.folder.rglob("*") if path.is_file()}
        panel = self._panel(rows)
        panel.override_button.invoke()
        self.root.update()
        self.assertIsNone(panel.processed_notice)
        self.assertTrue(panel.next_button.winfo_viewable())
        self.assertNotIn("disabled", panel.next_button.state())
        selected = panel.table.selected_rows()
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["image_paths"], [str(self.folder / "Light.img")])
        self.assertFalse(selected[0]["locked"])
        self.assertTrue(all(row["locked"] for row in rows))
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.folder.rglob("*") if path.is_file()})
        panel.next_button.invoke()
        self.step._start_step2_batch_segmentation_from_rows.assert_called_once_with(selected, str(self.folder))

    def test_directly_segmented_folder_hands_off_itself(self):
        self._source(self.folder, marked=True)
        panel = self._panel([self.frame._step2_batch_segmentation_row(self.folder)])
        panel.go_to_step3_button.invoke()
        self.step.on_continue_to_step3.assert_called_once_with([str(self.folder)])

    def test_empty_folder_offers_retry_and_rescans_new_selection(self):
        panel = self._panel([])
        self.assertTrue(panel.empty_notice.winfo_viewable())
        self.assertFalse(panel.next_button.winfo_viewable())
        notice = panel.empty_notice
        with patch("aidas.steps.step2_annotate.filedialog.askdirectory", return_value=""), patch.object(panel, "_start_scan") as scan:
            notice.select_folder_button.invoke()
            scan.assert_not_called()
        self.assertIs(panel.empty_notice, notice)
        selected = self.folder / "new-selection"
        self._source(selected)
        with patch("aidas.steps.step2_annotate.filedialog.askdirectory", return_value=str(selected)), patch.object(panel, "_start_scan") as scan:
            notice.select_folder_button.invoke()
            scan.assert_called_once_with()
        self.assertEqual(panel.root_dir, str(selected))
        self.assertIsNone(panel.empty_notice)
        panel._scan_done(*self.frame._scan_step2_batch_segmentation_folders(selected))
        self.root.update()
        self.assertTrue(panel.table.winfo_viewable())
        self.assertTrue(panel.next_button.winfo_viewable())
        self.assertNotIn("disabled", panel.next_button.state())
        self.assertNotIn("disabled", panel.hide_processed_check.state())
        self.assertEqual(panel.table.selected_rows()[0]["folder"], str(selected))

    def test_mixed_and_empty_scans_keep_normal_selection(self):
        self._source(self.folder / "complete", marked=True)
        self._source(self.folder / "ready")
        rows = self.frame._scan_step2_batch_segmentation_folders(self.folder)[0]
        panel = self._panel(rows)
        self.assertIsNone(panel.processed_notice)
        self.assertEqual(len(panel.table.selected_rows()), 1)
        self.assertTrue(panel.next_button.winfo_viewable())
        panel._scan_done([], 0, 0, [])
        self.assertIsNone(panel.processed_notice)
        self.assertIn("disabled", panel.next_button.state())


if __name__ == "__main__":
    unittest.main()
