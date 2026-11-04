"""Progress text stays synchronous while duplicate rebuilds are avoided."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step3_flatten import RBatchRunPanel
from tests.support.paths import PROJECT_ROOT


class _TextModel:
    def __init__(self):
        self.content = ""
        self.state = "disabled"
        self.rebuilds = 0

    def configure(self, **options):
        self.state = options.get("state", self.state)

    def delete(self, *_):
        self.content = ""
        self.rebuilds += 1

    def insert(self, _, text):
        self.content += text

    def see(self, _):
        pass


def _panel():
    panel = RBatchRunPanel.__new__(RBatchRunPanel)
    panel.folders = [Path("first"), Path("second")]
    panel.row_by_folder = {
        str(folder): {"folder": folder, "values": {"folder": str(folder), "status": "Queued", "inputs": "0%"}}
        for folder in panel.folders
    }
    panel.step_states_by_folder = {str(folder): [] for folder in panel.folders}
    panel.current_step_by_folder = {str(folder): None for folder in panel.folders}
    panel.table = SimpleNamespace(
        _row_by_iid={str(i): row for i, row in enumerate(panel.row_by_folder.values())},
        _refresh_row=mock.Mock(), _schedule_xscroll_visibility_update=mock.Mock(),
    )
    panel.step_text = _TextModel()
    panel._render_step_progress()
    return panel


class Step3ProgressRenderingTests(unittest.TestCase):
    def test_update_returns_with_original_text_history_and_percent_values(self):
        panel = _panel()
        folder = panel.folders[0]
        cases = (
            ({"status": "Validating", "progress": 1}, "Queued...\nQueued...\n", "1%"),
            ({"status": "RPE spline — αβ", "progress": 25}, "RPE spline — αβ...\nQueued...\n", "25%"),
            ({"status": "Flattening", "progress": 50}, "RPE spline — αβ... Done\nFlattening...\nQueued...\n", "50%"),
            ({"status": "Flattening", "progress": 55.9}, "RPE spline — αβ... Done\nFlattening...\nQueued...\n", "55%"),
            ({"progress": 123}, "RPE spline — αβ... Done\nFlattening...\nQueued...\n", "100%"),
            ({"status": "Failed"}, "RPE spline — αβ... Done\nFlattening... Failed\nQueued...\n", "100%"),
            ({"status": "Writing", "progress": -4}, "RPE spline — αβ... Done\nFlattening... Failed\nWriting...\nQueued...\n", "0%"),
            ({"status": "Completed", "progress": 100}, "RPE spline — αβ... Done\nFlattening... Failed\nWriting... Done\nCompleted... Done\nQueued...\n", "100%"),
        )
        for options, expected_text, percent in cases:
            with self.subTest(options=options):
                panel.update_folder(folder, **options)
                # Assert before any Tk idle callback can execute.
                self.assertEqual(panel.step_text.content, expected_text)
                self.assertEqual(panel.step_text.state, "disabled")
                self.assertEqual(panel.row_by_folder[str(folder)]["values"]["inputs"], percent)
        self.assertEqual(panel.step_states_by_folder[str(folder)], [
            {"label": "RPE spline — αβ", "state": "Done"},
            {"label": "Flattening", "state": "Failed"},
            {"label": "Writing", "state": "Done"},
            {"label": "Completed", "state": "Done"},
        ])
        self.assertIsNone(panel.current_step_by_folder[str(folder)])

    def test_direct_append_start_and_finish_keep_immediate_text_contract(self):
        panel = _panel()
        folder = panel.folders[1]
        panel._append_step(folder, "Prepared", "Done")
        self.assertEqual(panel.step_text.content, "Queued...\nPrepared... Done\n")
        panel._start_step(folder, "Reading")
        self.assertEqual(panel.step_text.content, "Queued...\nPrepared... Done\nReading...\n")
        panel._finish_current_step(folder, "Cancelled")
        self.assertEqual(panel.step_text.content, "Queued...\nPrepared... Done\nReading... Cancelled\n")
        before = panel.step_text.content
        panel.update_folder(Path("unknown"), status="Completed", progress=100)
        self.assertEqual(panel.step_text.content, before)

    def test_transition_and_completion_rebuild_text_once_per_operation(self):
        panel = _panel()
        folder = panel.folders[0]
        panel.update_folder(folder, status="Reading")
        previous = panel.step_text.rebuilds
        panel.update_folder(folder, status="Flattening")
        self.assertEqual(panel.step_text.rebuilds - previous, 1)
        self.assertIn("Reading... Done\nFlattening...\n", panel.step_text.content)
        previous = panel.step_text.rebuilds
        panel.update_folder(folder, status="Completed")
        self.assertEqual(panel.step_text.rebuilds - previous, 1)
        self.assertIn("Flattening... Done\nCompleted... Done\n", panel.step_text.content)

    def test_real_read_only_text_updates_before_return_without_new_idle_work(self):
        source = textwrap.dedent(r"""
            from pathlib import Path
            from types import SimpleNamespace
            import customtkinter as ctk
            from aidas.steps.step3_flatten import RBatchRunPanel, Step3Frame
            root = ctk.CTk()
            root.withdraw()
            root.attributes('-alpha', 0)
            try:
                step = SimpleNamespace(_normalize_r_output_mode=Step3Frame._normalize_r_output_mode)
                panel = RBatchRunPanel(step, root, [Path('study')], 1, 'main.R', 'output.R')
                panel.update_folder(Path('study'), status='Reading', progress=10)
                assert panel.step_text.get('1.0', 'end-1c') == 'Reading...\n'
                panel.update_folder(Path('study'), status='Flattening', progress=55)
                assert panel.step_text.get('1.0', 'end-1c') == 'Reading... Done\nFlattening...\n'
                panel.update_folder(Path('study'), status='Completed', progress=100)
                expected = 'Reading... Done\nFlattening... Done\nCompleted... Done\n'
                assert panel.step_text.get('1.0', 'end-1c') == expected
                assert panel.step_text.cget('state') == 'disabled'
                panel.step_text.insert('end', 'must remain read-only')
                assert panel.step_text.get('1.0', 'end-1c') == expected
                panel.destroy()
                root.update_idletasks()
            finally:
                root.destroy()
        """)
        result = subprocess.run([sys.executable, "-c", source], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
