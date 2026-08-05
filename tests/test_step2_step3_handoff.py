from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from aidas.app import AIDaSApp
from aidas.steps.step2_annotate import Step2Frame
from aidas.steps.step3_flatten import RBatchSelectionPanel, Step3Frame


class Step2Step3HandoffTests(unittest.TestCase):
    def test_saved_pairs_are_converted_to_unique_step3_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nasal = root / "nasal"
            temporal = root / "temporal"

            folders = Step2Frame._step3_folders_from_saved_pairs(
                [
                    (nasal / "Light_MARKED", temporal / "Light_MARKED"),
                    (nasal / "Light_MARKED", temporal / "Light_MARKED"),
                ]
            )

            self.assertEqual(folders, [str(nasal.resolve()), str(temporal.resolve())])

    def test_continue_button_saves_all_before_handing_folders_off(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            saved_pairs = [
                (root / "nasal" / "Light_MARKED", root / "temporal" / "Light_MARKED")
            ]
            received = []
            frame = Step2Frame.__new__(Step2Frame)
            frame._batch_result_states = {"segmented": {}}

            def save_all():
                frame._batch_result_states = {}
                return saved_pairs

            frame._save_all_batch_result_tabs = save_all
            frame.on_continue_to_step3 = received.append

            frame._save_all_and_continue_to_step3_button()

            self.assertEqual(
                received,
                [[str((root / "nasal").resolve()), str((root / "temporal").resolve())]],
            )

    def test_handoff_stays_in_step2_when_any_segmented_tab_failed_to_save(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame._batch_result_states = {"failed-tab": {}}
        frame._save_all_batch_result_tabs = lambda: [("nasal/Light_MARKED", "temporal/Light_MARKED")]
        received = []
        frame.on_continue_to_step3 = received.append

        frame._save_all_and_continue_to_step3_button()

        self.assertEqual(received, [])

    def test_step3_normalizes_existing_handoff_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nasal = root / "nasal"
            temporal = root / "temporal"
            nasal.mkdir()
            temporal.mkdir()

            folders = Step3Frame._normalize_batch_input_folders(
                [nasal, nasal, root / "missing", temporal]
            )

            self.assertEqual(folders, [nasal.resolve(), temporal.resolve()])

    def test_exact_handoff_scan_does_not_add_sibling_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected = root / "selected"
            sibling = root / "sibling"
            selected.mkdir()
            sibling.mkdir()

            step_frame = type(
                "StepFrame",
                (),
                {
                    "REQUIRED_INPUTS": (("Light_MARKED",), ("LIGHT",)),
                    "_find_input_paths": lambda self, folder: {
                        "Light_MARKED": folder / "Light_MARKED",
                        "LIGHT": folder / "Light",
                    },
                    "_folder_has_r_data": lambda self, _folder: False,
                },
            )()
            panel = RBatchSelectionPanel.__new__(RBatchSelectionPanel)
            panel.step_frame = step_frame
            panel.root_dir = root
            panel.input_folders = (selected,)
            panel.after = lambda _delay, callback: callback()
            scan_results = []
            panel._scan_done = lambda rows, *_rest: scan_results.extend(rows)
            panel._scan_failed = self.fail

            panel._scan_worker()

            self.assertEqual([row["folder"] for row in scan_results], [selected])

    def test_app_switches_tabs_and_opens_step3_folder_selector(self):
        events = []
        step3 = type(
            "Step3",
            (),
            {"open_batch_folders": lambda self, folders: events.append(("open", folders))},
        )()
        notebook = type(
            "Notebook",
            (),
            {"select": lambda self, tab: events.append(("select", tab))},
        )()
        app = AIDaSApp.__new__(AIDaSApp)
        app.step3 = step3
        app.notebook = notebook
        app.update_idletasks = lambda: events.append(("update", None))

        folders = ["nasal", "temporal"]
        app._on_step2_continue_to_step3(folders)

        self.assertEqual(
            events,
            [("select", step3), ("update", None), ("open", folders)],
        )


if __name__ == "__main__":
    unittest.main()
