from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from aidas.steps.step4_analyze_isez import ISezROI, Step4Frame
from aidas.ui.tabs import ClosableTabView


class _VariableStub:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


class _WidgetStub:
    def winfo_exists(self):
        return True


class _CanvasStub:
    def get_tk_widget(self):
        return _WidgetStub()


class _NotebookStub:
    def __init__(self, labels):
        self.labels = dict(labels)
        self.updates = []

    def tabs(self):
        return tuple(self.labels)

    def winfo_width(self):
        return 900

    def tab(self, tab_id, option=None, **options):
        if option == "text":
            return self.labels[tab_id]
        if "text" in options:
            self.labels[tab_id] = options["text"]
            self.updates.append((tab_id, options["text"]))


class _PageStub:
    def __init__(self):
        self.calls = []

    def grid(self, **options):
        self.calls.append(("grid", options))

    def tkraise(self):
        self.calls.append(("raise", None))


class Step4TabSwitchingTests(unittest.TestCase):
    def test_cached_selection_keeps_sidebar_idle_until_source_validation(self):
        frame = Step4Frame.__new__(Step4Frame)
        state = {"source_snapshot": object(), "loaded": True}
        frame.batch_roi_tab_states = {"cached": state}
        frame._active_batch_roi_tab = "previous"
        frame._batch_roi_selection_after_id = "queued"
        frame._leave_batch_roi_tab = mock.Mock()
        frame._set_batch_folder_label = mock.Mock()
        frame.status_var = mock.Mock()
        frame._update_confirm_button_state = mock.Mock()
        frame._close_profile_zoom = mock.Mock()

        frame._activate_batch_roi_tab("cached")

        frame._leave_batch_roi_tab.assert_called_once_with()
        self.assertEqual(frame._batch_roi_selection, ("cached", state))
        frame._set_batch_folder_label.assert_not_called()
        frame.status_var.set.assert_not_called()
        frame._update_confirm_button_state.assert_not_called()
        frame._move_roi(1)
        frame._clear_clicks()
        frame._auto_detect_all_rois()
        frame._close_profile_zoom.assert_not_called()

    def test_unchanged_roi_rows_skip_table_reads_and_mutations(self):
        frame = Step4Frame.__new__(Step4Frame)
        frame.rois = [ISezROI("01", 1, 2)]
        frame.current_roi_idx = 0
        frame.roi_table = mock.Mock()
        frame.roi_table.get_children.return_value = ("0",)
        frame.roi_table.selection.return_value = ("0",)
        frame._roi_needs_review = mock.Mock(return_value=False)
        frame._roi_metric_values = mock.Mock(return_value=("",))
        frame._update_build_stack_button_state = mock.Mock()

        frame._refresh_roi_list()
        frame.roi_table.item.assert_called_once_with("0", values=("01", ""))
        frame.roi_table.item.reset_mock()
        frame._refresh_roi_list()

        frame.roi_table.item.assert_not_called()
        frame.roi_table.selection_set.assert_not_called()
        frame.roi_table.focus.assert_not_called()

    def test_cached_tab_reuses_rendered_canvas_when_theme_is_unchanged(self):
        frame = Step4Frame.__new__(Step4Frame)
        frame.rois = [ISezROI("1", 1, 2)]
        frame.input_dir_var = _VariableStub()
        frame.output_dir_var = _VariableStub()
        frame._output_dir_user_selected = True
        frame._set_batch_folder_label = lambda _folder: None
        frame._sync_entry_vars_from_clicks = lambda: None
        frame._refresh_roi_list = lambda: None
        frame._select_roi_in_list = lambda: None
        frame._plot_theme_signature = lambda: ("dark",)
        frame._update_profile_status = lambda _profile: None
        frame._update_confirm_button_state = lambda: None
        frame._update_continue_to_step5_button_state = mock.Mock()
        frame.status_var = _VariableStub()
        frame._render_current_roi = lambda: self.fail(
            "an unchanged cached canvas must not be redrawn"
        )

        image = np.arange(6, dtype=np.int16).reshape(2, 3)
        state = {
            "path": Path("study") / "_flat_LIGHT.hdr",
            "current_path": Path("study") / "_flat_LIGHT.hdr",
            "current_stem": "_flat_LIGHT",
            "output_dir": str(Path("study") / "custom-results"),
            "volume": image[np.newaxis, ...],
            "image": image,
            "canvas": _CanvasStub(),
            "figure": object(),
            "completed": {},
            "roi_clicks": {},
            "current_roi_idx": 0,
            "profile_clicks": [],
            "current_profile": np.array([1.0, 2.0]),
            "plot_theme_signature": ("dark",),
        }

        self.assertTrue(frame._restore_batch_roi_tab_from_cache(state))
        self.assertIs(frame.image, image)
        self.assertEqual(frame.current_roi_idx, 0)
        self.assertEqual(frame.output_dir_var.value, state["output_dir"])
        frame._update_continue_to_step5_button_state.assert_called_once_with()

    def test_completion_keeps_tab_names_fixed_without_updating_the_header(self):
        frame = Step4Frame.__new__(Step4Frame)
        frame.rois = ("A", "B", "C")
        frame._active_batch_roi_tab = "tab-a"
        frame.batch_roi_tab_states = {
            "tab-a": {"folder": None, "base_label": "1. Alpha", "completed": {}},
            "tab-b": {"folder": None, "base_label": "2. Beta", "completed": {}},
        }
        notebook = _NotebookStub(
            {
                "tab-a": "1. Alpha",
                "tab-b": "2. Beta",
            }
        )
        frame.batch_roi_notebook = notebook

        frame._sync_active_batch_roi_state = mock.Mock()
        frame._mark_active_batch_roi_complete()

        self.assertTrue(frame.batch_roi_tab_states["tab-a"]["complete"])
        frame._sync_active_batch_roi_state.assert_called_once_with()
        self.assertEqual(notebook.updates, [])

    def test_selecting_the_current_loaded_tab_preserves_live_edits_without_reloading(self):
        frame = Step4Frame.__new__(Step4Frame)
        frame._active_batch_roi_tab = "tab-a"
        frame.canvas = _CanvasStub()
        frame._plot_theme_signature = lambda: ("dark",)
        frame.batch_roi_tab_states = {"tab-a": {
            "loaded": True, "canvas": frame.canvas,
            "plot_theme_signature": ("dark",),
        }}
        frame.profile_clicks = [72.25, 98.5]
        frame._sync_active_batch_roi_state = mock.Mock()
        frame._restore_batch_roi_tab_from_cache = mock.Mock()
        frame._cancel_roi_update_animations = mock.Mock()
        frame._load_path = mock.Mock()

        frame._activate_batch_roi_tab("tab-a")

        self.assertEqual(frame.profile_clicks, [72.25, 98.5])
        frame._sync_active_batch_roi_state.assert_not_called()
        frame._restore_batch_roi_tab_from_cache.assert_not_called()
        frame._cancel_roi_update_animations.assert_not_called()
        frame._load_path.assert_not_called()

    def test_showing_page_styles_only_the_selected_tab(self):
        selected = _PageStub()
        other = _PageStub()
        view = ClosableTabView.__new__(ClosableTabView)
        view._model = type(
            "Model",
            (),
            {
                "pages": {"selected": selected, "other": other},
                "order": ["selected", "other"],
            },
        )()
        styled = []
        view._style_tab = lambda page_id, *, active: styled.append((page_id, active))

        view._show_selected_page("selected")

        self.assertEqual(styled, [("selected", True)])
        self.assertEqual(other.calls, [])


if __name__ == "__main__":
    unittest.main()
