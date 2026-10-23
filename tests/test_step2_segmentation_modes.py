import tkinter as tk
import unittest
from unittest import mock

import customtkinter as ctk
import numpy as np

from aidas.canvas.image_canvas import ImageCanvas
from aidas.steps.step2_annotate import BOUNDARY_NAMES, FOVEA_BOUNDARY_NAME, Step2Frame
from aidas.ui.theme import get_interface_mode, set_interface_mode


class _Var:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Step2SegmentationModeTests(unittest.TestCase):
    def _frame(self):
        frame = Step2Frame.__new__(Step2Frame)
        frame._manual_segmentation_enabled = False
        frame.image_data = np.arange(80, dtype=np.int16).reshape(8, 10)
        frame.image_canvas = mock.Mock()
        frame.image_canvas.get_active_line.return_value = [(0, 2), (9, 4)]
        frame.image_canvas.get_line_handle_indices.return_value = None
        frame._segmenter_running = False
        frame._batch_fovea_picker_active = False
        frame._drawing_locked = False
        frame.active_boundary = BOUNDARY_NAMES[0]
        frame.fovea_x = 5
        frame.boundary_traces, frame.boundary_order = frame._boundary_traces_from_rows(
            np.ones((len(BOUNDARY_NAMES), 10)), frame.image_data.shape
        )
        frame.boundary_traces[FOVEA_BOUNDARY_NAME] = frame._vertical_line_trace(5, 8)
        frame.boundary_order.append(FOVEA_BOUNDARY_NAME)
        frame.boundary_completion_vars = {name: _Var(True) for name in BOUNDARY_NAMES}
        frame.status_var = _Var()
        frame.active_trace_var = _Var()
        frame.fovea_x_entry_var = _Var("5")
        frame.fovea_line_var = _Var()
        frame.boundary_workflow_status_var = _Var()
        frame._updating_fovea_entry = False
        for name in (
            "_refresh_boundary_lists", "_refresh_trace_list", "_select_trace_by_name",
            "_update_boundary_action_buttons", "_sync_boundary_canvas_state",
            "_update_boundary_progress_bar", "_set_fovea_controls_enabled",
            "_sync_active_batch_result_state", "_refresh_live_image_status",
            "_boundary_restore_cache", "_rebuild_saved_overlays",
            "_set_segmentation_running", "_set_runtime_ai_device_status",
        ):
            setattr(frame, name, mock.Mock())
        return frame

    def test_uninitialized_frame_defaults_to_read_only(self):
        frame = Step2Frame.__new__(Step2Frame)

        self.assertFalse(frame.is_manual_segmentation_enabled())

    def test_manual_entry_points_cannot_change_boundaries_when_disabled(self):
        frame = self._frame()
        original = frame._annotation_snapshot()
        with (
            mock.patch("aidas.steps.step2_annotate.messagebox.askyesno", return_value=True) as confirm,
            mock.patch("aidas.steps.step2_annotate.messagebox.showwarning") as warning,
            mock.patch("aidas.steps.step2_annotate.messagebox.showinfo") as information,
        ):
            frame._start_boundary_for_name(BOUNDARY_NAMES[1])
            frame._set_active_boundary_target(BOUNDARY_NAMES[1])
            frame._undo_point()
            frame._clear_active_trace()
            frame._finish_boundary()
            frame._delete_selected_boundary()
            frame._clear_all_traces()
            frame._on_saved_line_edited(BOUNDARY_NAMES[0], [(0, 3), (9, 4)])
            frame._on_active_line_changed([(0, 3), (9, 4)])

        self.assertEqual(frame._annotation_snapshot(), original)
        self.assertEqual(frame.active_boundary, BOUNDARY_NAMES[0])
        frame.image_canvas.undo_active_line_vertex.assert_not_called()
        frame.image_canvas.clear_active_line.assert_not_called()
        frame.image_canvas.commit_active_line.assert_not_called()
        frame._boundary_restore_cache.assert_not_called()
        frame._sync_active_batch_result_state.assert_not_called()
        confirm.assert_not_called()
        warning.assert_not_called()
        information.assert_not_called()

    def test_foveal_entry_nudges_and_drag_callbacks_remain_available(self):
        frame = self._frame()
        boundaries = frame._copy_trace_dict({name: frame.boundary_traces[name] for name in BOUNDARY_NAMES})

        frame._nudge_fovea_x(1)

        self.assertEqual(frame.fovea_x, 6)
        frame.image_canvas.set_vertical_line_x.assert_called_with(6)
        frame._on_vertical_line_changed(7)
        self.assertEqual(frame.fovea_x, 7)
        self.assertEqual(frame.fovea_x_entry_var.get(), "7")
        self.assertEqual(frame.boundary_traces[FOVEA_BOUNDARY_NAME]["points"], [(7, 0), (7, 7)])
        self.assertEqual({name: frame.boundary_traces[name] for name in BOUNDARY_NAMES}, boundaries)
        self.assertFalse(frame.is_manual_segmentation_enabled())

    def test_ai_results_are_imported_while_manual_editing_is_disabled(self):
        frame = self._frame()
        canvas = mock.Mock()
        target = {"canvas": canvas, "image": frame.image_data, "traces": {}, "order": [], "fovea_x": 5}
        frame._active_batch_result_tab = "other"
        frame._batch_result_states = {"target": target}
        frame._batch_result_tab_canvases = {"target": canvas}
        predicted, _order = frame._boundary_traces_from_rows(
            np.full((len(BOUNDARY_NAMES), 10), 3), frame.image_data.shape
        )
        source = {"tab_key": "target", "image": frame.image_data, "source_image": frame.image_data, "fovea_x": 5}

        frame._on_current_image_segmentation_done({"results": [{"traces": predicted}]}, source)

        self.assertEqual(set(target["traces"]), {*BOUNDARY_NAMES, FOVEA_BOUNDARY_NAME})
        for name in BOUNDARY_NAMES:
            self.assertEqual(target["traces"][name], predicted[name])
        canvas.set_line_overlays.assert_called_once()
        self.assertFalse(frame.is_manual_segmentation_enabled())

    def test_live_setting_replaces_workflow_with_legend_and_locks_every_cached_canvas(self):
        root = None
        original_interface = get_interface_mode()
        original_appearance = ctk.get_appearance_mode()
        try:
            set_interface_mode("Modern", redraw=False)
            ctk.set_appearance_mode("Light")
            try:
                root = ctk.CTk()
                root.withdraw()
                root.attributes("-alpha", 0)
                root.geometry("1024x680+10000+10000")
                root.deiconify()
            except tk.TclError as error:
                self.skipTest(f"A transparent Tk test window is unavailable: {error}")
            with mock.patch.object(Step2Frame, "_start_ai_device_probe"):
                frame = Step2Frame(root, preferences={"show_step2_annotation_tutorial": False})
            frame.pack(fill="both", expand=True)
            image = np.zeros((32, 96), dtype=np.int16)
            frame._show_image(image, "Step 1 output")
            root.update()

            self.assertFalse(frame.is_manual_segmentation_enabled())
            self.assertEqual(frame._boundary_workflow_frame.winfo_manager(), "")
            self.assertTrue(frame._boundary_legend.winfo_viewable())
            self.assertEqual(set(frame._boundary_legend.rows), {*BOUNDARY_NAMES, "Fovea center"})
            self.assertFalse(frame.boundary_incomplete_listbox.winfo_viewable())
            self.assertFalse(frame.boundary_completed_listbox.winfo_viewable())
            self.assertFalse(frame.image_canvas._line_on)
            self.assertIsNone(frame.image_canvas._editable_line_label)
            self.assertTrue(frame.image_canvas._vertical_line_on)

            frame.set_manual_segmentation_enabled(True)
            root.update()

            self.assertTrue(frame.is_manual_segmentation_enabled())
            self.assertTrue(frame._boundary_workflow_frame.winfo_viewable())
            self.assertFalse(frame._boundary_legend.winfo_viewable())
            self.assertTrue(frame.boundary_incomplete_listbox.winfo_viewable())
            self.assertTrue(frame.boundary_completed_listbox.winfo_viewable())
            for button in (frame.revert_line_btn, frame.revert_all_lines_btn,
                           frame.ai_line_btn, frame.ai_all_lines_btn):
                self.assertTrue(button.winfo_viewable())

            cached_list_canvas = ImageCanvas(frame.canvas_area)
            cached_tab_canvas = ImageCanvas(frame.canvas_area)
            frame._batch_result_canvases = [cached_list_canvas]
            frame._batch_result_tab_canvases = {"cached": cached_tab_canvas}
            frame._batch_result_states = {
                "list": {"canvas": cached_list_canvas, "traces": {}},
                "cached": {"canvas": cached_tab_canvas, "traces": {}},
            }
            canvases = [frame.single_image_canvas, cached_list_canvas, cached_tab_canvas]
            for canvas in canvases:
                if canvas is not frame.single_image_canvas:
                    canvas.set_image(image)
                canvas.set_line_overlays([{"label": BOUNDARY_NAMES[0], "points": [(1, 5), (20, 7)]}])
                canvas.enable_line(True)
                canvas.set_active_line([(1, 2), (20, 3)])
                canvas.set_editable_line(BOUNDARY_NAMES[0])
                self.assertTrue(canvas.get_active_line())
                self.assertIsNotNone(canvas._editable_line_label)

            frame.set_manual_segmentation_enabled(False)
            root.update()

            self.assertFalse(frame.is_manual_segmentation_enabled())
            self.assertFalse(frame._boundary_workflow_frame.winfo_viewable())
            self.assertTrue(frame._boundary_legend.winfo_viewable())
            for canvas in canvases:
                self.assertFalse(canvas._line_on)
                self.assertEqual(canvas.get_active_line(), [])
                self.assertIsNone(canvas._editable_line_label)
                self.assertIsNone(canvas._line_edit_drag)
            self.assertTrue(frame.image_canvas._vertical_line_on)

            # A single click on a saved line must not enter the saved-line
            # editor. Foveal movement continues through its separate marker.
            canvas = frame.image_canvas
            x, y = canvas._i2c(1, 5)
            event = type("Click", (), {"x": x, "y": y, "state": 0})()
            canvas._on_press(event)
            self.assertIsNone(canvas._line_edit_drag)
            self.assertEqual(canvas.get_active_line(), [])
        finally:
            if root is not None:
                root.destroy()
            set_interface_mode(original_interface, redraw=False)
            ctk.set_appearance_mode(original_appearance)


if __name__ == "__main__":
    unittest.main()
