from __future__ import annotations

import os
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

from aidas.steps.step3_flatten import RBatchSelectionPanel, Step3Frame


class Step3InputFreshnessTests(unittest.TestCase):
    INPUT_FILES = ("Light.hdr", "Light.img", "Light_MARKED.hdr", "Light_MARKED.img")
    OLDER = 1_700_000_000_000_000_000
    NEWER = OLDER + 10_000_000_000

    @classmethod
    def _make_folder(cls, root, name, *, newer_input=None):
        folder = Path(root) / name
        folder.mkdir()
        for filename in cls.INPUT_FILES:
            path = folder / filename
            path.write_bytes(b"input")
            timestamp = cls.NEWER if filename == newer_input else cls.OLDER
            os.utime(path, ns=(timestamp, timestamp))
        rdata = folder / Step3Frame.R_WORKSPACE_FILES[1]
        rdata.write_bytes(b"existing RData")
        timestamp = cls.OLDER if newer_input is not None else cls.NEWER
        os.utime(rdata, ns=(timestamp, timestamp))
        return folder

    def test_resaved_header_or_image_makes_existing_rdata_stale(self):
        frame = Step3Frame.__new__(Step3Frame)
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in self.INPUT_FILES:
                with self.subTest(filename=filename):
                    folder = self._make_folder(temp_dir, filename, newer_input=filename)
                    self.assertFalse(frame._folder_has_r_data(folder))
                    self.assertEqual(
                        (folder / Step3Frame.R_WORKSPACE_FILES[1]).read_bytes(),
                        b"existing RData",
                    )

    def test_current_rdata_remains_skipped(self):
        frame = Step3Frame.__new__(Step3Frame)
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = self._make_folder(temp_dir, "current")
            self.assertTrue(frame._folder_has_r_data(folder))

    def test_latest_available_rdata_controls_freshness(self):
        frame = Step3Frame.__new__(Step3Frame)
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = self._make_folder(temp_dir, "mixed", newer_input="Light_MARKED.img")
            newest_rdata = folder / Step3Frame.R_WORKSPACE_FILES[0]
            newest_rdata.write_bytes(b"new RData")
            os.utime(newest_rdata, ns=(self.NEWER, self.NEWER))
            self.assertTrue(frame._folder_has_r_data(folder))

    def test_metadata_error_preserves_existing_rdata_skip(self):
        frame = Step3Frame.__new__(Step3Frame)
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = self._make_folder(temp_dir, "unreadable")
            with mock.patch.object(frame, "_find_input_paths", side_effect=PermissionError("unreadable")):
                self.assertTrue(frame._folder_has_r_data(folder))

    def test_exact_handoff_scan_selects_resaved_inputs_and_locks_current_results(self):
        frame = Step3Frame.__new__(Step3Frame)
        with tempfile.TemporaryDirectory() as temp_dir:
            resaved = self._make_folder(temp_dir, "resaved", newer_input="Light_MARKED.img")
            current = self._make_folder(temp_dir, "current")
            panel = RBatchSelectionPanel.__new__(RBatchSelectionPanel)
            panel.step_frame = frame
            panel.root_dir = Path(temp_dir)
            panel.input_folders = (resaved, current)
            panel._scan_events = queue.Queue()
            rows = []
            panel._scan_done = lambda scanned_rows, *_rest: rows.extend(scanned_rows)
            panel._scan_failed = self.fail

            panel._scan_worker()
            panel._poll_scan_events()

            self.assertEqual([row["folder"] for row in rows], [resaved, current])
            self.assertEqual(rows[0]["status"], "Ready")
            self.assertTrue(rows[0]["include"])
            self.assertFalse(rows[0]["locked"])
            self.assertFalse(rows[1]["include"])
            self.assertTrue(rows[1]["locked"])

    def test_batch_worker_runs_resaved_inputs_without_existing_rdata_override(self):
        for mode in (Step3Frame.R_OUTPUT_MODE_PARALLEL, Step3Frame.R_OUTPUT_MODE_SEQUENTIAL):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp_dir:
                folder = self._make_folder(temp_dir, "resaved", newer_input="Light_MARKED.img")
                frame = Step3Frame.__new__(Step3Frame)
                frame.after = lambda _delay, callback: callback()
                frame._batch_panel_update = lambda *_args, **_kwargs: None
                frame._r_cancel_event = threading.Event()
                frame._r_worker_limit = lambda: 1
                frame._r_threads_per_process = lambda _workers: 1
                config = {"output_dir": str(folder)}
                frame._r_script_config_for_folder = lambda _folder: config
                calls = []
                finished = []
                frame._on_batch_r_done = finished.extend

                def run_pipeline(*_args, **_kwargs):
                    calls.append("run")
                    return {"folder": folder, "returncode": 0, "outcome": "completed"}

                frame._run_r_script_for_config = run_pipeline
                frame._run_main_r_script_for_config = run_pipeline
                frame._run_output_r_script_for_config = lambda *_args, **_kwargs: {
                    "folder": folder, "returncode": 0, "outcome": "completed"
                }
                frame._finalize_r_script_result = lambda _config, result, **_kwargs: result

                frame._batch_r_worker(
                    Path("Rscript.exe"), Path("main.R"), Path("output.R"),
                    [folder], workers=1, output_mode=mode,
                )

                self.assertEqual(calls, ["run"])
                self.assertEqual([result["outcome"] for result in finished], ["completed"])


if __name__ == "__main__":
    unittest.main()
