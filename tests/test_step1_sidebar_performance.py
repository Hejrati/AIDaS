from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import customtkinter as ctk
import numpy as np
from scipy.signal import find_peaks

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.utils.bright_pixels import _occupied_intensities, detect_bright_pixels_in_bands
from aidas.utils.filesystem import find_sdb_directories
from aidas.utils.oct_crop import _transition_peaks


class PreviewEquivalenceTests(unittest.TestCase):
    def test_transition_peaks_match_scipy_including_plateaus_and_endpoints(self):
        rng = np.random.default_rng(47)
        signals = [np.array(values, dtype=float) for values in (
            [], [0], [0, 1], [0, 2, 2, 0], [0, 2, 2, 2, 0],
            [2, 2, 1, 2, 2], [0, 2, 0, 2, 0], [0, np.nan, 2, 0],
        )]
        signals += [rng.integers(0, 10, size=300).astype(float) for _ in range(100)]
        signals += [rng.random(300) for _ in range(100)]
        for signal in signals:
            for threshold in (0, 0.5, 3, 9, 11):
                expected, _ = find_peaks(signal, height=threshold, distance=2)
                np.testing.assert_array_equal(_transition_peaks(signal, threshold), expected)

    def test_integer_histograms_match_sorted_unique_exactly(self):
        rng = np.random.default_rng(19)
        for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32):
            limits = np.iinfo(dtype)
            samples = rng.integers(limits.min, limits.max, size=10000, dtype=dtype)
            samples[:2] = (limits.min, limits.max)
            expected = np.unique(samples, return_counts=True)
            actual = _occupied_intensities(samples)
            for observed, reference in zip(actual, expected):
                np.testing.assert_array_equal(observed, reference)

    def test_bright_pixel_masks_and_thresholds_match_the_original_histogram(self):
        rng = np.random.default_rng(61)
        for dtype in (np.int16, np.uint16, np.float64):
            for _ in range(5):
                image = rng.integers(0, 2000, size=(256, 128)).astype(dtype)
                image[40, 60] = 30000
                original = image.copy()
                actual = detect_bright_pixels_in_bands(image)
                with patch("aidas.utils.bright_pixels._occupied_intensities",
                           side_effect=lambda samples: np.unique(samples, return_counts=True)):
                    expected = detect_bright_pixels_in_bands(image)
                np.testing.assert_array_equal(actual.mask, expected.mask)
                self.assertEqual(actual.threshold, expected.threshold)
                np.testing.assert_array_equal(image, original)


class SidebarScanTests(unittest.TestCase):
    def _frame(self):
        frame = Step1Frame.__new__(Step1Frame)
        frame._active_sdb_directory = "old"
        frame._sdb_directory_files = {}
        frame._saved_output_directories = set()
        frame.sdb_scan_more_label = Mock()
        frame.sdb_scan_tooltip = SimpleNamespace(text="")
        frame.sdb_dir_var = SimpleNamespace(get=lambda: "root")
        frame._render_sdb_directories = Mock()
        frame._preview_selected_sdb = Mock()
        frame.after = Mock(return_value="scan-poll")
        frame.after_cancel = Mock()
        return frame

    def test_refresh_returns_while_worker_is_blocked_and_applies_result_on_poll(self):
        frame = self._frame()
        started, release = threading.Event(), threading.Event()
        events = queue.Queue()
        worker_threads = []
        result = ({"old": ["old/image.sdb"]}, set(), [])

        def scan(_root, _cancelled):
            worker_threads.append(threading.get_ident())
            started.set()
            release.wait(3)
            return result

        try:
            with patch.object(Step1Frame, "_scan_sdb_directory", side_effect=scan), \
                    patch("aidas.steps.step1_resize_raw.queue.Queue", return_value=events):
                frame.refresh_sdb_list(preview_first=True)
                self.assertTrue(started.wait(1))
                self.assertNotEqual(worker_threads, [threading.get_ident()])
                self.assertTrue(frame._sdb_scan_pending)
                frame._preview_selected_sdb.assert_not_called()
                release.set()
                delivered = events.get(timeout=2)
                events.put(delivered)
                poll = frame.after.call_args.args[1]
                poll()
            self.assertFalse(frame._sdb_scan_pending)
            self.assertEqual(frame._sdb_directory_files, result[0])
            self.assertEqual(frame._active_sdb_directory, "old")
            frame._preview_selected_sdb.assert_called_once()
        finally:
            release.set()
            frame._cancel_sdb_scan()

    def test_cancelled_or_changed_root_never_publishes_a_stale_queue(self):
        frame = self._frame()
        events = queue.Queue()
        events.put(({"stale": []}, set(), []))
        cancelled = frame._sdb_scan_cancel = threading.Event()
        frame._sdb_scan_pending = True
        frame._sdb_scan_after_id = "scan-poll"
        frame._cancel_sdb_scan()
        frame._poll_sdb_scan(events, cancelled, "root", None, True)
        self.assertEqual(frame._sdb_directory_files, {})
        frame._preview_selected_sdb.assert_not_called()
        frame.after_cancel.assert_called_once_with("scan-poll")

        cancelled = frame._sdb_scan_cancel = threading.Event()
        frame._poll_sdb_scan(events, cancelled, "different-root", None, True)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(frame._sdb_directory_files, {})

    def test_scan_keeps_order_saved_pairs_and_access_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.sdb").touch()
            (root / "A.SDB").touch()
            (root / "light.hdr").touch()
            (root / "light.img").touch()
            files, saved, errors = Step1Frame._scan_sdb_directory(root, threading.Event())
            self.assertEqual(files[str(root)], [str(root / "A.SDB"), str(root / "b.sdb")])
            self.assertEqual(saved, {Step1Frame._path_key(root)})
            self.assertEqual(errors, [])
            cancelled = threading.Event()
            cancelled.set()
            self.assertEqual(find_sdb_directories(root, cancel_event=cancelled)[0], {})

    def test_folder_switch_changes_only_two_list_rows_and_keeps_completion_colors(self):
        root = ctk.CTk()
        root.withdraw()
        try:
            with patch.object(Step1Frame, "refresh_sdb_list"):
                frame = Step1Frame(root)
            frame._sdb_directories = [f"folder-{i}" for i in range(1000)]
            frame._sdb_directory_labels = list(frame._sdb_directories)
            frame._sdb_directory_files = {folder: [folder + "/image.sdb"] for folder in frame._sdb_directories}
            frame._active_sdb_directory = "folder-0"
            frame._saved_output_directories = {frame._path_key("folder-0")}
            frame._refresh_sdb_directory_rows()
            completed_color = frame.sdb_directory_listbox.itemcget(0, "background")
            frame._active_sdb_directory = "folder-999"
            with patch.object(frame.sdb_directory_listbox, "delete", wraps=frame.sdb_directory_listbox.delete) as delete:
                frame._refresh_sdb_directory_rows()
                self.assertEqual(delete.call_count, 2)
                self.assertEqual(delete.call_args_list[0].args, (0,))
            self.assertEqual(frame.sdb_directory_listbox.size(), 1000)
            self.assertTrue(frame.sdb_directory_listbox.get(999).startswith("▶"))
            self.assertEqual(frame.sdb_directory_listbox.curselection(), (999,))
            self.assertEqual(frame.sdb_directory_listbox.itemcget(0, "background"), completed_color)
            frame._saved_output_directories.clear()
            frame._refresh_sdb_directory_rows()
            self.assertEqual(frame.sdb_directory_listbox.itemcget(0, "background"), "")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
