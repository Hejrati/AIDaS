from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from aidas.app import AIDaSApp
from aidas.steps.step3_flatten import Step3Frame
from aidas.steps.step4_analyze_isez import Step4Frame


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Button:
    def __init__(self, state="disabled"):
        self.state = state

    def cget(self, name):
        if name != "state":
            raise KeyError(name)
        return self.state

    def configure(self, **options):
        self.state = options.get("state", self.state)


def _write_flat_light_pair(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "_flat_LIGHT.hdr").write_bytes(b"header")
    (folder / "_flat_LIGHT.img").write_bytes(b"image")
    return folder


class Step3Step4HandoffTests(unittest.TestCase):
    def test_step3_handoff_stays_in_a_fixed_sidebar_footer(self):
        source = inspect.getsource(Step3Frame._build_ui)
        footer_start = source.index("self.step_actions_footer = ttk.Frame(")
        button_start = source.index("self.continue_to_step4_button = AppButton(")
        footer_source = source[footer_start:button_start]

        self.assertIn("self.sidebar_shell,", footer_source)
        self.assertIn('side="bottom"', footer_source)
        self.assertIn("before=self.sidebar", footer_source)

    def test_step3_normalizes_complete_result_pairs_in_stable_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _write_flat_light_pair(root / "first")
            second = _write_flat_light_pair(root / "second")
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "_flat_LIGHT.hdr").write_bytes(b"header")

            folders = Step3Frame._normalize_step4_result_folders(
                [first, first / "_flat_LIGHT.img", incomplete, second, first]
            )

            self.assertEqual(folders, [first.resolve(), second.resolve()])

    def test_step3_handoff_sends_every_available_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _write_flat_light_pair(root / "first")
            second = _write_flat_light_pair(root / "second")
            received = []
            frame = Step3Frame.__new__(Step3Frame)
            frame.step4_result_folders = [first, second]
            frame._busy = False
            frame.on_continue_to_step4 = received.append
            frame.continue_to_step4_button = None

            frame._continue_to_step4()

            self.assertEqual(
                received,
                [[str(first.resolve()), str(second.resolve())]],
            )

    def test_step3_handoff_button_is_disabled_while_r_is_running(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _write_flat_light_pair(Path(temp_dir) / "ready")
            frame = Step3Frame.__new__(Step3Frame)
            frame.step4_result_folders = [folder]
            frame.on_continue_to_step4 = lambda _folders: None
            frame.continue_to_step4_button = _Button()
            frame._busy = False

            frame._update_continue_to_step4_button_state()
            self.assertEqual(frame.continue_to_step4_button.state, "normal")

            frame._busy = True
            frame._update_continue_to_step4_button_state()
            self.assertEqual(frame.continue_to_step4_button.state, "disabled")

    def test_manual_r_result_load_becomes_step4_handoff_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _write_flat_light_pair(Path(temp_dir) / "loaded")
            results = {
                "flattened_dark": np.zeros((1, 2, 3)),
                "flattened_light": np.zeros((1, 2, 3)),
                "final_grand_mean": np.zeros((2, 3)),
                "vertex": 2,
            }
            frame = Step3Frame.__new__(Step3Frame)
            frame._load_r_results_with_fallbacks = lambda _folder: (
                results,
                "test_loader",
                [],
            )
            frame._load_original_light_for_preview = lambda _folder: None
            frame._render = lambda: None
            frame._update_continue_to_step4_button_state = lambda: None
            frame.progress_text_var = _Var()
            frame.view_var = _Var()
            frame.info_var = _Var()
            frame.status_var = _Var()

            loaded = frame._load_r_results_from_folder(folder)

            self.assertTrue(loaded)
            self.assertEqual(frame.step4_result_folders, [folder.resolve()])

    def test_completed_r_batch_records_only_valid_step4_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            complete = _write_flat_light_pair(root / "complete")
            incomplete = root / "incomplete"
            incomplete.mkdir()
            failed = root / "failed"
            failed.mkdir()
            opened = []
            frame = Step3Frame.__new__(Step3Frame)
            frame._busy = True
            frame._active_r_core_allocation = 2
            frame._active_r_folder_keys = {"active"}
            frame.r_batch_run_panel = None
            frame.progress_text_var = _Var()
            frame.status_var = _Var()
            frame.info_var = _Var()
            frame._pending_batch_restart = None
            frame._pending_batch_folders = None
            frame._set_process_buttons = lambda _state: None
            frame._update_continue_to_step4_button_state = lambda: None
            frame._open_batch_r_result_tabs = lambda folders: opened.extend(folders)

            frame._on_batch_r_done(
                [
                    {"folder": complete, "returncode": 0, "outcome": "completed"},
                    {"folder": incomplete, "returncode": 0, "outcome": "completed"},
                    {"folder": failed, "returncode": 1, "outcome": "failed"},
                ]
            )

            self.assertEqual(frame.step4_result_folders, [complete.resolve()])
            self.assertEqual(opened, [complete, incomplete])

    def test_step4_resolves_only_exact_transferred_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected = _write_flat_light_pair(root / "selected")
            _write_flat_light_pair(root / "unrelated-sibling")

            paths = Step4Frame._step3_flat_light_paths(
                [selected, selected / "_flat_LIGHT.img", root / "missing"]
            )

            self.assertEqual(paths, [selected.resolve() / "_flat_LIGHT.hdr"])

    def test_step4_opens_every_transferred_file_as_a_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _write_flat_light_pair(root / "first")
            second = _write_flat_light_pair(root / "second")
            received_rows = []
            frame = Step4Frame.__new__(Step4Frame)
            frame._start_batch_roi_from_rows = received_rows.extend

            opened = frame.open_batch_folders([first, second])

            self.assertTrue(opened)
            self.assertEqual(
                [row["flat_light"] for row in received_rows],
                [
                    first.resolve() / "_flat_LIGHT.hdr",
                    second.resolve() / "_flat_LIGHT.hdr",
                ],
            )

    def test_step4_rejects_a_handoff_without_complete_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "incomplete"
            folder.mkdir()
            (folder / "_flat_LIGHT.hdr").write_bytes(b"header")
            frame = Step4Frame.__new__(Step4Frame)
            frame._start_batch_roi_from_rows = self.fail

            with mock.patch(
                "aidas.steps.step4_analyze_isez.messagebox.showwarning"
            ) as warning:
                opened = frame.open_batch_folders([folder])

            self.assertFalse(opened)
            warning.assert_called_once()

    def test_app_switches_tabs_then_opens_step4_files(self):
        events = []
        step4 = type(
            "Step4",
            (),
            {
                "open_batch_folders": lambda self, folders: events.append(
                    ("open", folders)
                )
            },
        )()
        notebook = type(
            "Notebook",
            (),
            {"select": lambda self, tab: events.append(("select", tab))},
        )()
        app = AIDaSApp.__new__(AIDaSApp)
        app.step4 = step4
        app.notebook = notebook
        app.update_idletasks = lambda: events.append(("update", None))

        folders = ["nasal", "temporal"]
        app._on_step3_continue_to_step4(folders)

        self.assertEqual(
            events,
            [("select", step4), ("update", None), ("open", folders)],
        )


if __name__ == "__main__":
    unittest.main()
