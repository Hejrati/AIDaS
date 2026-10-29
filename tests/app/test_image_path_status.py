from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk
import numpy as np

from aidas.app import AIDaSApp
from aidas.core.config import Config
from aidas.steps.step2_annotate import Step2Frame
from aidas.steps.step4_analyze_isez import Step4Frame


class ImagePathStatusTests(unittest.TestCase):
    def test_step3_elapsed_time_only_appears_while_step3_is_selected(self):
        app = AIDaSApp.__new__(AIDaSApp)
        selection = [2]
        app.notebook = SimpleNamespace(select=lambda: selection[0], index=int)
        app.status = mock.Mock()
        app.status.cget.return_value = ""
        variable = mock.Mock()
        variable.get.return_value = "Elapsed: 00:00:09"
        app.step3 = SimpleNamespace(elapsed_time_var=variable)
        app.step4 = SimpleNamespace(current_path=Path("roi/_flat_LIGHT.hdr"))

        app._refresh_image_path_status("elapsed", "", "write")
        app.status.configure.assert_called_with(text="Elapsed: 00:00:09")
        selection[0] = 3
        app._refresh_image_path_status()
        app.status.configure.assert_called_with(text=os.path.abspath("roi/_flat_LIGHT.hdr"))
        variable.get.return_value = "Elapsed: 00:00:10"
        app._refresh_image_path_status("elapsed", "", "write")
        app.status.configure.assert_called_with(text=os.path.abspath("roi/_flat_LIGHT.hdr"))

        for index in (0, 4):
            selection[0] = index
            app.status.cget.return_value = "previous step's status"
            app._refresh_image_path_status()
            app.status.configure.assert_called_with(text="")

        selection[0] = 2
        variable.get.return_value = "Total elapsed: 00:00:12"
        app._refresh_image_path_status()
        app.status.configure.assert_called_with(text="Total elapsed: 00:00:12")
        variable.get.return_value = ""
        app._refresh_image_path_status("elapsed", "", "write")
        app.status.configure.assert_called_with(text="")

    def test_visible_workflow_owns_footer_and_inactive_work_cannot_replace_it(self):
        app = AIDaSApp.__new__(AIDaSApp)
        selection = [1]
        app.notebook = SimpleNamespace(select=lambda: selection[0], index=int)
        app.status = mock.Mock()
        app.status.cget.return_value = ""
        app.step2 = SimpleNamespace(current_file="temporary-save.img",
            _active_batch_result_tab="first", _batch_result_states={
                "first": {"input": "study-a/Light.img"}, "second": {"input": "study-b/Light.img"}})
        app.step4 = SimpleNamespace(current_path=Path("roi/_flat_LIGHT.hdr"))
        app._refresh_image_path_status()
        app.status.configure.assert_called_with(text=os.path.abspath("study-a/Light.img"))
        app.step2._active_batch_result_tab = "second"
        app._refresh_image_path_status()
        app.status.configure.assert_called_with(text=os.path.abspath("study-b/Light.img"))
        selection[0] = 3
        app._refresh_image_path_status()
        app.status.configure.assert_called_with(text=os.path.abspath("roi/_flat_LIGHT.hdr"))
        app.status.cget.return_value = os.path.abspath("roi/_flat_LIGHT.hdr")
        app.status.configure.reset_mock()
        app.step2.current_file = "background.img"
        app._refresh_image_path_status()
        app.status.configure.assert_not_called()
        app.status.cget.return_value = "old image.img"
        selection[0] = 0
        app._refresh_image_path_status()
        app.status.configure.assert_called_with(text="")
        selection[0] = 1
        app.step2.batch_segmentation_panel = object()
        app._refresh_image_path_status()
        app.status.configure.assert_called_with(text="")

    def test_real_image_tabs_update_and_clear_footer_on_switch_and_close(self):
        original_init = ctk.CTk.__init__

        def transparent_init(window, *args, **kwargs):
            original_init(window, *args, **kwargs)
            window.attributes("-alpha", 0)

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "preferences.json"
            config.write_text(json.dumps({"interface_mode": "Modern", "check_for_updates": False,
                                          "show_step2_annotation_tutorial": False}), encoding="utf-8")
            app = None
            with (mock.patch.object(Config, "CONFIG_DIR", Path(directory)),
                  mock.patch.object(Config, "CONFIG_FILE", config),
                  mock.patch.object(ctk.CTk, "__init__", transparent_init),
                  mock.patch.object(Step2Frame, "_start_ai_device_probe"),
                  mock.patch.object(Step4Frame, "_warm_auto_detection_tools")):
                try:
                    app = AIDaSApp()
                    app._finish_startup()
                    app.update()
                    self.assertEqual(app.status.cget("text"), "")
                    paths = [Path(directory) / folder / "Light.img" for folder in ("first", "second")]
                    image = np.arange(96 * 400, dtype=np.uint16).reshape(96, 400)
                    frame = app.step2
                    app.notebook.select(frame)
                    with mock.patch.object(frame, "_read_image_for_annotation", return_value=(image, None, False)):
                        frame._open_aidas_batch_results_viewer([
                            {"input": str(path), "traces": {
                                "RPE": {"points": [(0, 50), (399, 50)], "color": "#ffb703"}
                            }} for path in paths])
                    tabs = frame.batch_results_notebook.tabs()
                    for index in (0, 1, 0):
                        frame.batch_results_notebook.select(tabs[index])
                        app.update()
                        self.assertEqual(app.status.cget("text"), str(paths[index]))
                    # Closing another tab leaves the active address intact.
                    with mock.patch.object(frame, "_confirm_close_batch_result_tab", return_value=True):
                        frame._close_batch_result_tab(frame.batch_results_notebook, tabs[1])
                        app.update()
                        self.assertEqual(app.status.cget("text"), str(paths[0]))
                        frame._close_batch_result_tab(frame.batch_results_notebook, tabs[0])
                        app.update()
                        self.assertEqual(app.status.cget("text"), "")

                    roi_paths = [path.with_suffix(".hdr") for path in paths]
                    roi = app.step4
                    app.notebook.select(roi)
                    with mock.patch("aidas.steps.step4_analyze_isez.load_oct_volume", return_value=image[None]):
                        roi._open_batch_roi_tabs(roi_paths)
                        tabs = roi.batch_roi_notebook.tabs()
                        for index in (0, 1, 0):
                            roi.batch_roi_notebook.select(tabs[index])
                            app.update()
                            self.assertEqual(app.status.cget("text"), str(roi_paths[index]))
                        app.notebook.select(app.step3)
                        app.update()
                        self.assertEqual(app.status.cget("text"), "")
                        app.step3.elapsed_time_var.set("Elapsed: 00:00:09")
                        app.update()
                        self.assertEqual(app.status.cget("text"), "Elapsed: 00:00:09")
                        app.notebook.select(roi)
                        app.update()
                        self.assertEqual(app.status.cget("text"), str(roi_paths[0]))
                        app.step3.elapsed_time_var.set("Elapsed: 00:00:10")
                        app.update()
                        self.assertEqual(app.status.cget("text"), str(roi_paths[0]))
                        roi._close_batch_roi_tab(roi.batch_roi_notebook, tabs[0])
                        app.update()
                        self.assertEqual(app.status.cget("text"), str(roi_paths[1]))
                        roi._close_batch_roi_tab(roi.batch_roi_notebook, tabs[1])
                        app.update()
                        self.assertEqual(app.status.cget("text"), "")
                finally:
                    if app is not None:
                        app.destroy()


if __name__ == "__main__":
    unittest.main()
