from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from aidas.steps.step4_analyze_isez import Step4Frame
from tests.support.paths import PROJECT_ROOT


class _Notebook:
    def __init__(self, tabs, current=None, on_select=None):
        self.order = list(tabs)
        self.current = current or self.order[0]
        self.on_select = on_select
        self.selections = []

    def tabs(self):
        return tuple(self.order)

    def select(self, tab=None):
        if tab is not None:
            self.current = str(tab)
            self.selections.append(self.current)
            if self.on_select is not None:
                self.on_select(self.current)
        return self.current


def _frame(*folders: str) -> Step4Frame:
    frame = object.__new__(Step4Frame)
    frame.batch_roi_tab_states = {
        f"tab-{index}": {
            "folder": Path(folder),
            "path": Path(folder) / "_flat_LIGHT.hdr",
            "complete": False,
            "loaded": True,
        }
        for index, folder in enumerate(folders)
    }
    frame.batch_roi_notebook = _Notebook(frame.batch_roi_tab_states)
    frame._active_batch_roi_tab = "tab-0"
    frame.next_image_button = mock.Mock()
    return frame


class Step4ImageNavigationTests(unittest.TestCase):
    def test_single_subject_opens_temporal_before_current_rois_are_finished(self):
        frame = _frame("subject/nasal", "subject/temporal")
        frame.completed = {}

        frame._update_next_image_button_state()
        frame._go_to_next_image()

        self.assertEqual(frame.batch_roi_notebook.selections, ["tab-1"])
        frame.next_image_button.configure.assert_called_with(text="Go to temporal")
        frame.next_image_button.state.assert_called_with(["disabled"])
        self.assertFalse(frame.batch_roi_tab_states["tab-0"]["complete"])

    def test_single_subject_accepts_renamed_nasal_folder_and_side_case(self):
        frame = _frame("subject/Mito490 OD- bad nasal", "subject/Temporal")

        self.assertEqual(frame._next_image_action(), ("Go to temporal", "tab-1"))

    def test_temporal_target_does_not_depend_on_tab_order(self):
        frame = _frame("subject/temporal", "subject/nasal")
        frame.batch_roi_notebook.current = "tab-1"

        self.assertEqual(frame._next_image_action(), ("Go to temporal", "tab-0"))

    def test_single_subject_does_not_offer_completed_temporal(self):
        frame = _frame("subject/nasal", "subject/temporal")
        frame.batch_roi_tab_states["tab-1"]["complete"] = True

        frame._go_to_next_image()

        self.assertEqual(frame.batch_roi_notebook.selections, [])
        frame.next_image_button.state.assert_called_with(["disabled"])

    def test_different_subjects_use_batch_label(self):
        frame = _frame("subject-a/nasal", "subject-b/temporal")

        self.assertEqual(frame._next_image_action(), ("Go to next image", "tab-1"))

    def test_batch_skips_completed_images_and_can_return_to_unfinished_image(self):
        frame = _frame("a/nasal", "a/temporal", "b/nasal", "b/temporal")
        frame.batch_roi_tab_states["tab-1"]["complete"] = True

        frame._go_to_next_image()
        self.assertEqual(frame.batch_roi_notebook.selections, ["tab-2"])
        frame.batch_roi_tab_states["tab-2"]["complete"] = True
        frame._go_to_next_image()
        frame._go_to_next_image()

        self.assertEqual(frame.batch_roi_notebook.selections, ["tab-2", "tab-3", "tab-0"])

    def test_last_unfinished_image_never_navigates_to_itself(self):
        frame = _frame("a/nasal", "b/temporal")
        frame.batch_roi_tab_states["tab-1"]["complete"] = True

        self.assertEqual(frame._next_image_action(), ("Go to next image", None))
        frame._go_to_next_image()
        self.assertEqual(frame.batch_roi_notebook.selections, [])

    def test_one_open_image_or_no_tab_session_disables_navigation(self):
        frame = _frame("subject/nasal")
        self.assertEqual(frame._next_image_action(), ("Go to next image", None))
        frame.batch_roi_notebook = None
        frame._update_next_image_button_state()
        frame.next_image_button.state.assert_called_once_with(["disabled"])

    def test_removed_tab_state_cannot_be_selected(self):
        frame = _frame("a/nasal", "b/temporal")
        del frame.batch_roi_tab_states["tab-1"]

        frame._go_to_next_image()

        self.assertEqual(frame.batch_roi_notebook.selections, [])

    def test_busy_actions_and_initial_loading_block_even_direct_invocation(self):
        for flag in ("_stack_building", "_auto_detecting", "_auto_saving_roi", "_batch_roi_open_cancel"):
            with self.subTest(flag=flag):
                frame = _frame("subject/nasal", "subject/temporal")
                setattr(frame, flag, True)

                frame._go_to_next_image()

                self.assertEqual(frame.batch_roi_notebook.selections, [])
                frame.next_image_button.state.assert_called_with(["disabled"])
                setattr(frame, flag, None if flag == "_batch_roi_open_cancel" else False)
                frame._update_next_image_button_state()
                frame.next_image_button.state.assert_called_with(["!disabled"])

    def test_navigation_uses_normal_tab_activation_to_preserve_unsaved_roi_edits(self):
        frame = _frame("subject/nasal", "subject/temporal")
        frame.batch_roi_notebook.on_select = frame._activate_batch_roi_tab
        frame.completed = {"01": object()}
        frame.auto_detection_reviews = {"02": "review"}
        frame.roi_clicks = {"02": [12.25, 28.5]}
        frame._roi_adjust_bounds = {"02": False}
        frame.current_roi_idx = 1
        frame.profile_clicks = [12.25, 28.5]
        frame.volume = object()
        frame.image = object()
        frame.current_path = Path("subject/nasal/_flat_LIGHT.hdr")
        frame.current_stem = "_flat_LIGHT"
        frame.output_dir_var = mock.Mock(get=lambda: "subject/nasal")
        frame.canvas = None
        frame.figure = None
        frame.ax_profile = None
        frame.ax_roi_grid = None
        frame.empty_placeholder = None
        frame._current_profile = None
        frame._plot_theme_signature = lambda: ("light",)
        frame._close_profile_zoom = mock.Mock()
        frame._hide_roi_grid_hint = mock.Mock()
        frame._cancel_roi_update_animations = mock.Mock()
        frame._restore_batch_roi_tab_from_cache = mock.Mock(return_value=True)
        frame._confirm_current_roi = mock.Mock()
        frame._build_stack_outputs = mock.Mock()

        frame._go_to_next_image()

        state = frame.batch_roi_tab_states["tab-0"]
        self.assertEqual(state["current_roi_idx"], 1)
        self.assertEqual(state["profile_clicks"], [12.25, 28.5])
        self.assertEqual(state["roi_clicks"], {"02": [12.25, 28.5]})
        self.assertEqual(state["completed"], frame.completed)
        self.assertEqual(state["auto_detection_reviews"], {"02": "review"})
        self.assertEqual(state["roi_adjust_bounds"], {"02": False})
        self.assertIsNot(state["profile_clicks"], frame.profile_clicks)
        self.assertIsNot(state["roi_clicks"]["02"], frame.roi_clicks["02"])
        self.assertEqual(frame._active_batch_roi_tab, "tab-1")
        self.assertEqual(frame.current_roi_idx, 1)
        frame._confirm_current_roi.assert_not_called()
        frame._build_stack_outputs.assert_not_called()

    def test_real_button_keeps_edits_when_switching_unfinished_files(self):
        script = textwrap.dedent(
            """
            from pathlib import Path
            from tempfile import TemporaryDirectory
            import customtkinter as ctk
            import numpy as np
            from aidas.steps.step4_analyze_isez import Step4Frame
            from aidas.utils.io_utils import write_analyze

            ctk.set_widget_scaling(1.0)
            ctk.set_window_scaling(1.0)
            root = ctk.CTk()
            root.withdraw()
            root.attributes('-alpha', 0.0)
            root.geometry('980x450+10000+10000')
            frame = Step4Frame(root)
            frame.pack(fill='both', expand=True)
            frame._open_automatic_profile_review = lambda: None
            try:
                with TemporaryDirectory() as temporary:
                    image = np.repeat(np.linspace(20, 80, 450, dtype=np.int16)[:, None], 3000, axis=1)
                    paths = []
                    for side in ('nasal', 'temporal'):
                        path = Path(temporary) / side / '_flat_LIGHT.hdr'
                        path.parent.mkdir()
                        write_analyze(str(path), image)
                        paths.append(path)
                    frame._open_batch_roi_tabs(paths)
                    root.deiconify()
                    root.update_idletasks()
                    button = frame.next_image_button
                    assert button.cget('text') == 'Go to temporal'
                    assert button.cget('state') == 'normal'
                    assert button.winfo_ismapped()
                    assert button.winfo_rooty() >= frame.winfo_rooty()
                    assert button.winfo_rooty() + button.winfo_height() <= frame.winfo_rooty() + frame.winfo_height()
                    tabs = frame.batch_roi_notebook.tabs()
                    frame.profile_clicks = [80.25, 100.5]
                    frame._remember_current_roi_clicks()
                    button.invoke()
                    assert frame._active_batch_roi_tab == tabs[1]
                    assert button.cget('state') == 'disabled'
                    assert frame.batch_roi_tab_states[tabs[0]]['profile_clicks'] == [80.25, 100.5]
                    assert not frame.batch_roi_tab_states[tabs[0]]['complete']
                    frame.batch_roi_notebook.select(tabs[0])
                    assert frame.profile_clicks == [80.25, 100.5]
                    assert frame.current_roi_idx == 0
                    assert button.cget('state') == 'normal'
            finally:
                root.destroy()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
