from __future__ import annotations

from pathlib import Path
import unittest

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
    def test_cached_tab_reuses_rendered_canvas_when_theme_is_unchanged(self):
        frame = Step4Frame.__new__(Step4Frame)
        frame.rois = [ISezROI("1", 1, 2)]
        frame.input_dir_var = _VariableStub()
        frame._output_dir_user_selected = True
        frame._set_batch_folder_label = lambda _folder: None
        frame._sync_entry_vars_from_clicks = lambda: None
        frame._refresh_roi_list = lambda: None
        frame._select_roi_in_list = lambda: None
        frame._plot_theme_signature = lambda: ("dark",)
        frame._update_profile_status = lambda _profile: None
        frame._update_confirm_button_state = lambda: None
        frame.status_var = _VariableStub()
        frame._render_current_roi = lambda: self.fail(
            "an unchanged cached canvas must not be redrawn"
        )

        image = np.arange(6, dtype=np.int16).reshape(2, 3)
        state = {
            "path": Path("study") / "_flat_LIGHT.hdr",
            "current_path": Path("study") / "_flat_LIGHT.hdr",
            "current_stem": "_flat_LIGHT",
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

    def test_tab_label_refresh_skips_labels_that_are_already_current(self):
        frame = Step4Frame.__new__(Step4Frame)
        frame.rois = ("A", "B", "C")
        frame._active_batch_roi_tab = "tab-a"
        frame.batch_roi_tab_states = {
            "tab-a": {"folder": None, "base_label": "1. Alpha", "completed": {}},
            "tab-b": {"folder": None, "base_label": "2. Beta", "completed": {}},
        }
        notebook = _NotebookStub(
            {
                "tab-a": "1. Alpha (0/3)",
                "tab-b": "stale",
            }
        )
        frame.batch_roi_notebook = notebook

        frame._refresh_batch_roi_tab_labels()

        self.assertEqual(notebook.updates, [("tab-b", "2. Beta (0/3)")])

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
