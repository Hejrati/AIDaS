from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import customtkinter as ctk

from aidas.steps.step3_flatten import RBatchSelectionPanel, Step3Frame


class Step3CompletedFolderPromptTests(unittest.TestCase):
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
        self.frame = Step3Frame.__new__(Step3Frame)
        self.scripts = {}
        for role in ("main", "output"):
            path = self.folder / (role + ".R")
            path.write_text("# script", encoding="utf-8")
            self.scripts[role] = path
        self.step = SimpleNamespace(
            status_var=tk.StringVar(master=self.root),
            on_continue_to_step4=Mock(),
            _close_r_batch_panel=Mock(),
            _available_r_worker_limit=lambda: 4,
            _step2_core_allocation=lambda: 0,
            R_OUTPUT_MODE_PARALLEL=Step3Frame.R_OUTPUT_MODE_PARALLEL,
            R_OUTPUT_MODE_SEQUENTIAL=Step3Frame.R_OUTPUT_MODE_SEQUENTIAL,
            REQUIRED_INPUTS=Step3Frame.REQUIRED_INPUTS,
            _find_input_paths=self.frame._find_input_paths,
            _folder_has_r_data=self.frame._folder_has_r_data,
            _selected_r_script_path=lambda role: self.scripts[role],
            _normalize_r_output_mode=Step3Frame._normalize_r_output_mode,
            _start_batch_r_runs=Mock(),
        )

    def _files(self, name, *, processed=True, flattened=True):
        folder = self.folder / name
        folder.mkdir()
        names = ["Light.img", "Light.hdr", "Light_MARKED.img", "Light_MARKED.hdr"]
        if flattened:
            names += ["_flat_LIGHT.img", "_flat_LIGHT.hdr"]
        if processed:
            names += [Step3Frame.R_WORKSPACE_FILES[0]]
        for name in names:
            (folder / name).write_bytes(name.encode())
        return folder

    def _panel(self, folders):
        with patch.object(RBatchSelectionPanel, "_start_scan"):
            panel = RBatchSelectionPanel(self.step, self.root, self.folder, folders=folders)
        panel.pack(fill="both", expand=True)
        panel._scan_worker()
        panel._poll_scan_events()
        self.root.update()
        return panel

    def test_processed_results_offer_green_step4_and_red_override(self):
        folders = [self._files("nasal"), self._files("temporal")]
        panel = self._panel(folders)
        self.assertTrue(panel.processed_notice.winfo_viewable())
        self.assertEqual(panel.go_to_step4_button._variant, "success")
        self.assertEqual(panel.override_button._variant, "danger")
        self.assertFalse(panel.next_button.winfo_viewable())
        self.assertFalse(panel.worker_settings_footer.winfo_viewable())
        panel.go_to_step4_button.invoke()
        self.step.on_continue_to_step4.assert_called_once_with([str(folder) for folder in folders])
        self.step._close_r_batch_panel.assert_called_once_with(render_previous=True)
        self.step._start_batch_r_runs.assert_not_called()

    def test_override_preserves_results_until_run_and_allows_existing_rdata(self):
        folders = [self._files("nasal"), self._files("temporal")]
        panel = self._panel(folders)
        original_rows = panel.rows
        before = {str(path): path.read_bytes() for path in self.folder.rglob("*") if path.is_file()}
        panel.override_button.invoke()
        self.root.update()
        self.assertIsNone(panel.processed_notice)
        self.assertEqual(len(panel.table.selected_rows()), 2)
        self.assertTrue(all(row["locked"] for row in original_rows))
        self.assertTrue(panel.next_button.winfo_viewable())
        self.assertTrue(panel.worker_settings_footer.winfo_viewable())
        self.assertTrue(panel.execution_mode_footer.winfo_viewable())
        self.assertNotIn("disabled", panel.next_button.state())
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.folder.rglob("*") if path.is_file()})
        panel.next_button.invoke()
        self.step._start_batch_r_runs.assert_called_once_with(
            folders, 2, self.scripts["main"], self.scripts["output"],
            output_mode="parallel", allow_existing_rdata=True,
        )

    def test_missing_flattened_images_disable_step4_and_keep_override_available(self):
        panel = self._panel([self._files("incomplete", flattened=False)])
        self.assertTrue(panel.processed_notice.winfo_viewable())
        self.assertEqual(panel.go_to_step4_button.cget("state"), "disabled")
        self.assertEqual(panel.override_button.cget("state"), "normal")
        panel.go_to_step4_button.invoke()
        self.step.on_continue_to_step4.assert_not_called()

    def test_empty_folder_offers_retry_and_scans_new_folder_recursively(self):
        panel = self._panel([])
        self.assertTrue(panel.empty_notice.winfo_viewable())
        self.assertFalse(panel.next_button.winfo_viewable())
        self.assertFalse(panel.worker_settings_footer.winfo_viewable())
        notice = panel.empty_notice
        with patch("aidas.steps.step3_flatten.filedialog.askdirectory", return_value=""), patch.object(panel, "_start_scan") as scan:
            notice.select_folder_button.invoke()
            scan.assert_not_called()
        self.assertIs(panel.empty_notice, notice)
        selected = self._files("new-selection", processed=False)
        with patch("aidas.steps.step3_flatten.filedialog.askdirectory", return_value=str(selected)), patch.object(panel, "_start_scan") as scan:
            notice.select_folder_button.invoke()
            scan.assert_called_once_with()
        self.assertEqual(panel.root_dir, selected)
        self.assertIsNone(panel.input_folders)
        panel._scan_worker()
        panel._poll_scan_events()
        self.root.update()
        self.assertIsNone(panel.empty_notice)
        self.assertTrue(panel.table.winfo_viewable())
        self.assertTrue(panel.next_button.winfo_viewable())
        self.assertTrue(panel.worker_settings_footer.winfo_viewable())
        self.assertTrue(panel.execution_mode_footer.winfo_viewable())
        self.assertNotIn("disabled", panel.next_button.state())
        self.assertNotIn("disabled", panel.hide_processed_check.state())
        self.assertEqual(panel.table.selected_rows()[0]["folder"], selected)

    def test_mixed_and_empty_scans_keep_normal_processing(self):
        folders = [self._files("complete"), self._files("ready", processed=False)]
        panel = self._panel(folders)
        self.assertIsNone(panel.processed_notice)
        self.assertEqual([row["folder"] for row in panel.table.selected_rows()], folders[1:])
        panel.next_button.invoke()
        self.assertNotIn("allow_existing_rdata", self.step._start_batch_r_runs.call_args.kwargs)
        panel._scan_done([], 0, 0, [])
        self.assertIsNone(panel.processed_notice)
        self.assertIn("disabled", panel.next_button.state())


if __name__ == "__main__":
    unittest.main()
