from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from aidas.app import AIDaSApp
from aidas.steps.step2_annotate import Step2Frame
from aidas.steps.step3_flatten import RBatchSelectionPanel, Step3Frame


class Step2Step3HandoffTests(unittest.TestCase):
    def test_hidden_step2_defers_external_image_until_selected(self):
        existing_image = np.full((2, 2), 17, dtype=np.int16)
        incoming_image = np.array([[0, 255]], dtype=np.uint8)
        rendered = []
        frame = Step2Frame.__new__(Step2Frame)
        frame.current_file = "existing.img"
        frame.image_data = existing_image
        frame._input_analyze_template = {"existing": True}
        frame._source_was_8bit = False
        frame._pending_external_image = None
        frame._show_image = lambda image, path: rendered.append((image, path))

        frame.load_external_image(
            incoming_image,
            source_path="new.sdb",
            defer_render=True,
        )

        self.assertEqual(rendered, [])
        self.assertEqual(frame.current_file, "existing.img")
        self.assertIs(frame.image_data, existing_image)
        self.assertEqual(frame._input_analyze_template, {"existing": True})
        self.assertFalse(frame._source_was_8bit)

        self.assertTrue(frame.render_pending_external_image())
        self.assertEqual(len(rendered), 1)
        rendered_image, rendered_path = rendered[0]
        np.testing.assert_array_equal(
            rendered_image,
            np.array([[0, 65535]], dtype=np.uint16),
        )
        self.assertEqual(rendered_path, "new.sdb")
        self.assertIsNone(frame._pending_external_image)
        self.assertIsNone(frame._input_analyze_template)
        self.assertTrue(frame._source_was_8bit)
        self.assertFalse(frame.render_pending_external_image())

    def test_latest_deferred_external_image_supersedes_older_crop(self):
        rendered = []
        frame = Step2Frame.__new__(Step2Frame)
        frame._pending_external_image = None
        frame._show_image = lambda image, path: rendered.append((image.copy(), path))

        frame.load_external_image(
            np.full((1, 1), 10, dtype=np.int16),
            source_path="first.sdb",
            defer_render=True,
        )
        frame.load_external_image(
            np.full((1, 1), 20, dtype=np.int16),
            source_path="latest.sdb",
            defer_render=True,
        )
        frame.render_pending_external_image()

        self.assertEqual(len(rendered), 1)
        np.testing.assert_array_equal(rendered[0][0], np.array([[20]], dtype=np.int16))
        self.assertEqual(rendered[0][1], "latest.sdb")

    def test_visible_step2_renders_external_image_immediately(self):
        rendered = []
        frame = Step2Frame.__new__(Step2Frame)
        frame._pending_external_image = ("stale", "stale.sdb", False)
        frame._show_image = lambda image, path: rendered.append((image.copy(), path))

        frame.load_external_image(
            np.full((1, 2), 42, dtype=np.int16),
            source_path="visible.sdb",
        )

        self.assertEqual(len(rendered), 1)
        np.testing.assert_array_equal(
            rendered[0][0],
            np.full((1, 2), 42, dtype=np.int16),
        )
        self.assertEqual(rendered[0][1], "visible.sdb")
        self.assertIsNone(frame._pending_external_image)

    def test_app_defers_step1_crop_only_while_step2_is_hidden(self):
        calls = []
        step2 = type(
            "Step2",
            (),
            {
                "load_external_image": lambda self, image, **kwargs: calls.append(
                    (image, kwargs)
                )
            },
        )()

        class Notebook:
            selected = "step1"

            def select(self):
                return self.selected

            @staticmethod
            def index(tab):
                return 1 if tab is step2 or tab == "step2" else 0

        app = AIDaSApp.__new__(AIDaSApp)
        app.step2 = step2
        app.notebook = Notebook()
        image = np.ones((2, 3), dtype=np.int16)

        app._on_step1_processed_image(image, "source.sdb")
        app.notebook.selected = "step2"
        app._on_step1_processed_image(image, "source.sdb")

        self.assertTrue(calls[0][1]["defer_render"])
        self.assertFalse(calls[1][1]["defer_render"])
        self.assertEqual(calls[0][1]["source_path"], "source.sdb")

    def test_selecting_step2_renders_its_pending_external_image(self):
        events = []
        step2 = type(
            "Step2",
            (),
            {
                "render_pending_external_image": lambda self: events.append("render")
            },
        )()

        class Notebook:
            @staticmethod
            def select():
                return "step2"

            @staticmethod
            def index(tab):
                return 1 if tab is step2 or tab == "step2" else 0

        header = type(
            "Header",
            (),
            {"select_step": lambda self, index: events.append(("header", index))},
        )()
        app = AIDaSApp.__new__(AIDaSApp)
        app.notebook = Notebook()
        app.header = header
        app.step2 = step2

        app._on_workflow_tab_changed()

        self.assertEqual(events, [("header", 1), "render"])

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
