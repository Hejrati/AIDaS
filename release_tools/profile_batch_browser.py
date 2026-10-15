"""Benchmark batch scan prefiltering and list updates on disposable fixtures.

Run: python release_tools/profile_batch_browser.py
Baselines reproduce the previous full-tree validation, uncached font measuring,
and per-row filtering. These synthetic timings are not clinical-data benchmarks.
"""

from contextlib import ExitStack
from pathlib import Path
import queue
import statistics
import sys
import tempfile
import time
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aidas.steps import step2_annotate as s2, step3_flatten as s3
from aidas.steps import step4_analyze_isez as s4
from aidas.utils.filesystem import walk_accessible_directories


def timed(callback, repeats=3):
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def unfiltered(root, _groups):
    folders, errors = walk_accessible_directories(root)
    return folders, len(folders), errors


def uncached_measure(self, text, *, heading=False, padding=18):
    font = self._heading_font if heading else self._tree_font
    return int(font.measure(str(text or ""))) + int(padding)


def per_row_filter(self):
    for iid, row in self._row_by_iid.items():
        if self._hide_processed and row.get("locked"):
            self.tree.detach(iid)
        else:
            self.tree.move(iid, "", "end")


def main():
    with tempfile.TemporaryDirectory(prefix="aidas-batch-profile-") as directory:
        root = Path(directory)
        for index in range(2651):
            folder = root / f"folder-{index:04}"
            folder.mkdir()
            if index < 6:
                for name in ("Light.img", "Light.hdr", "Light_MARKED.img", "Light_MARKED.hdr", "_flat_LIGHT.hdr"):
                    (folder / name).touch()
        step2 = s2.Step2Frame.__new__(s2.Step2Frame)
        step4 = s4.Step4Frame.__new__(s4.Step4Frame)
        panel = s3.RBatchSelectionPanel.__new__(s3.RBatchSelectionPanel)
        panel.step_frame = s3.Step3Frame.__new__(s3.Step3Frame)
        panel.root_dir = root
        panel.input_folders = None
        panel._scan_events = queue.Queue()

        def scan3():
            panel._scan_worker()
            return panel._scan_events.get_nowait()

        for module, callback in (
            (s2, lambda: step2._scan_step2_batch_segmentation_folders(root)),
            (s3, scan3),
            (s4, lambda: step4._scan_batch_roi_folders(root)),
        ):
            expected = callback()
            with patch.object(module, "find_candidate_directories", unfiltered):
                assert callback() == expected
                baseline = timed(callback)
            optimized = timed(callback)
            print(f"{module.__name__} scan, 2652 folders: {baseline:.3f}s -> {optimized:.3f}s")

    root = tk.Tk()
    root.withdraw()
    try:
        for table_type in (s2.Step2BatchSegmentationTable, s3.RBatchSelectionTable, s4.Step4BatchROITable):
            table = table_type(root)
            rows = [dict(include=i % 3 != 0, locked=i % 3 == 0, values={
                "folder": f"study-{i:05}", "status": "Already complete" if i % 3 == 0 else "Ready",
                "images": "1", "inputs": "2", "outputs": "0/3",
            }) for i in range(3000)]

            def populate():
                table.set_rows(rows)
                root.update_idletasks()

            def toggle():
                table.set_hide_processed(True)
                table.set_hide_processed(False)
                root.update_idletasks()

            with ExitStack() as stack:
                stack.enter_context(patch.object(table_type, "_measure_text", uncached_measure))
                stack.enter_context(patch.object(table_type, "_apply_processed_filter", per_row_filter))
                baseline_load = timed(populate)
                baseline_filter = timed(toggle)
            optimized_load = timed(populate)
            optimized_filter = timed(toggle)
            print(f"{table_type.__name__}, 3000 rows: load {baseline_load:.3f}s -> {optimized_load:.3f}s; "
                  f"filter pair {baseline_filter:.3f}s -> {optimized_filter:.3f}s")
            table.destroy()
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
