from __future__ import annotations

from pathlib import Path
import queue
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from aidas.steps.step4_analyze_isez import ISezROI, Step4Frame


class _VariableStub:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _restart_frame() -> Step4Frame:
    frame = object.__new__(Step4Frame)
    frame.rois = [ISezROI("01", 1, 2)]
    frame.volume = np.arange(12).reshape(1, 3, 4)
    frame.image = frame.volume[0]
    frame.current_path = Path("study") / "_flat_LIGHT.hdr"
    frame.current_stem = "_flat_LIGHT"
    frame.current_roi_idx = 0
    frame.completed = {"01": object()}
    frame.auto_detection_reviews = {"01": "old review"}
    frame.roi_clicks = {"01": [2.0, 3.0]}
    frame._roi_adjust_bounds = {"01": False}
    frame.profile_clicks = [2.0, 3.0]
    frame.start_var = _VariableStub("2")
    frame.end_var = _VariableStub("3")
    frame.stats_var = _VariableStub("old measurements")
    frame.status_var = _VariableStub()
    frame.profile_status_var = _VariableStub()
    frame.image_label_var = _VariableStub("old file")
    frame.figure = None
    frame.canvas = None
    frame.ax_profile = None
    frame.ax_roi_grid = None
    frame.empty_placeholder = None
    frame._current_profile = np.arange(4)
    frame._last_results_dir = Path("old results")
    frame._stack_build_complete = True
    frame._stack_building = False
    frame._stack_build_events = None
    frame._auto_detecting = False
    frame._auto_detection_events = None
    frame._auto_saving_roi = False
    frame._active_batch_roi_tab = "tab-a"
    frame.batch_roi_notebook = None
    frame.batch_roi_tab_states = {"tab-a": {"loaded": True, "complete": True}}
    frame._plot_theme_signature = lambda: ("theme",)
    for method in (
        "_hide_grid_notice", "_hide_roi_grid_hint", "_close_profile_zoom", "_cancel_roi_update_animations",
        "_refresh_roi_list", "_select_roi_in_list", "_render_current_roi",
        "_update_active_batch_roi_tab_progress", "_update_build_stack_button_state",
        "_update_continue_to_step5_button_state", "_update_auto_detect_button_state",
        "_update_clear_button_state", "_update_confirm_button_state",
    ):
        setattr(frame, method, mock.Mock())
    frame.after_cancel = mock.Mock()
    return frame


class Step4ResetLifecycleTests(unittest.TestCase):
    def test_restart_clears_both_current_and_cached_roi_state(self):
        frame = _restart_frame()
        original_image = frame.image
        original_volume = frame.volume

        frame._restart_current_file()

        self.assertIs(frame.image, original_image)
        self.assertIs(frame.volume, original_volume)
        self.assertEqual(frame.completed, {})
        self.assertEqual(frame.auto_detection_reviews, {})
        self.assertEqual(frame.roi_clicks, {})
        self.assertEqual(frame._roi_adjust_bounds, {})
        self.assertEqual(frame.profile_clicks, [])
        self.assertEqual((frame.start_var.get(), frame.end_var.get()), ("", ""))
        state = frame.batch_roi_tab_states["tab-a"]
        self.assertFalse(state["complete"])
        self.assertEqual(state["completed"], {})
        self.assertEqual(state["auto_detection_reviews"], {})
        self.assertEqual(state["roi_clicks"], {})
        self.assertEqual(state["roi_adjust_bounds"], {})
        self.assertEqual(state["profile_clicks"], [])
        frame._render_current_roi.assert_called_once()

    def test_restart_detaches_pending_detection_and_stack_results(self):
        frame = _restart_frame()
        detection_events = queue.Queue()
        stack_events = queue.Queue()
        frame._auto_detection_events = detection_events
        frame._stack_build_events = stack_events
        frame._auto_detecting = True
        frame._stack_building = True
        frame._apply_auto_detection_run = mock.Mock()
        frame._finish_stack_build = mock.Mock()

        frame._restart_current_file()
        detection_events.put(("done", object()))
        stack_events.put(("done", None))
        frame._poll_auto_detection_events(detection_events)
        frame._poll_stack_build_events(stack_events, Path("old results"))

        self.assertIsNone(frame._auto_detection_events)
        self.assertIsNone(frame._stack_build_events)
        self.assertFalse(frame._auto_detecting)
        self.assertFalse(frame._stack_building)
        frame._apply_auto_detection_run.assert_not_called()
        frame._finish_stack_build.assert_not_called()
        self.assertEqual(frame.completed, {})

    def test_destroyed_canvas_is_not_restored_from_a_tab_cache(self):
        frame = object.__new__(Step4Frame)
        widget = SimpleNamespace(winfo_exists=lambda: False)
        canvas = SimpleNamespace(get_tk_widget=lambda: widget)

        restored = frame._restore_batch_roi_tab_from_cache(
            {"canvas": canvas, "figure": object(), "path": Path("closed.hdr")}
        )

        self.assertFalse(restored)

    def test_failed_new_file_load_drops_previous_data_and_pending_results(self):
        frame = _restart_frame()
        events = queue.Queue()
        frame._auto_detection_events = events
        frame._auto_detecting = True
        frame._render_empty_canvas = mock.Mock()
        frame._apply_auto_detection_run = mock.Mock()
        with mock.patch(
            "aidas.steps.step4_analyze_isez.load_oct_volume",
            side_effect=ValueError("invalid image header"),
        ), mock.patch(
            "aidas.steps.step4_analyze_isez.messagebox.showerror"
        ) as show_error:
            frame._load_path(Path("broken") / "_flat_LIGHT.hdr")

        events.put(("done", object()))
        frame._poll_auto_detection_events(events)

        self.assertIsNone(frame.image)
        self.assertIsNone(frame.volume)
        self.assertIsNone(frame.current_path)
        self.assertIsNone(frame._current_profile)
        self.assertIsNone(frame._last_results_dir)
        self.assertEqual(frame.completed, {})
        self.assertEqual(frame.auto_detection_reviews, {})
        self.assertEqual(frame.roi_clicks, {})
        self.assertEqual(frame.profile_clicks, [])
        self.assertEqual(frame.stats_var.get(), "")
        self.assertFalse(frame._auto_detecting)
        self.assertIsNone(frame._auto_detection_events)
        self.assertFalse(frame._has_current_roi_input())
        frame._apply_auto_detection_run.assert_not_called()
        frame._render_empty_canvas.assert_called_once_with()
        show_error.assert_called_once_with("Open error", "invalid image header")

    def test_real_widgets_close_restart_and_ignore_detached_callbacks(self):
        # Separate process prevents global CTk theme/DPI callbacks belonging to
        # earlier tests from interfering with this real Tk lifecycle check.
        script = textwrap.dedent(
            """
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from types import SimpleNamespace
            from unittest import mock
            import customtkinter as ctk
            import numpy as np
            from aidas.steps.step4_analyze_isez import AutoDetectionRunResult, Step4Frame
            from aidas.utils.io_utils import write_analyze

            root = ctk.CTk()
            root.withdraw()
            root.attributes('-alpha', 0.0)
            root.geometry('1150x720+10000+10000')
            frame = Step4Frame(root)
            frame.pack(fill='both', expand=True)

            def settle():
                root.update_idletasks()

            def start_detection():
                # Leave the worker unfinished so transitions happen before
                # completion. This makes callback ordering deterministic.
                with mock.patch.object(frame, '_start_background_worker'):
                    frame._auto_detect_all_rois()
                assert frame._auto_detecting
                return frame._auto_detection_events

            def deliver_old_result(events):
                events.put(('done', AutoDetectionRunResult(
                    completed={}, reviews={'01': 'stale worker review'},
                    roi_clicks={'01': [75.0, 99.0]}, removed_clicks=set(),
                    review_count=1, current_roi_idx=0,
                )))
                frame._poll_auto_detection_events(events)
                settle()
                assert not frame.auto_detection_reviews, 'Old worker restored review state.'
                assert not frame.roi_clicks, 'Old worker restored closed-file clicks.'

            def assert_empty():
                assert frame.batch_roi_notebook is None
                assert frame.batch_roi_tab_states == {}
                assert not frame.batch_roi_paths, 'Closed files must leave no batch history.'
                assert frame._active_batch_roi_tab is None
                for name in ('image', 'volume', 'current_path', 'canvas', 'figure',
                             'ax_profile', 'ax_roi_grid', '_current_profile', '_last_results_dir'):
                    assert getattr(frame, name) is None, f'{name} retained closed-file data.'
                for name in ('completed', 'auto_detection_reviews', 'roi_clicks',
                             '_roi_adjust_bounds', 'profile_clicks'):
                    assert not getattr(frame, name), f'{name} retained closed-file state.'
                assert frame.start_var.get() == frame.end_var.get() == ''
                assert frame.stats_var.get() == ''
                assert frame.image_label_var.get() == 'No image loaded'
                assert not frame._auto_detecting and not frame._stack_building
                assert frame._auto_detection_events is None
                assert frame._stack_build_events is None
                for button in (frame.auto_detect_button, frame.build_stacks_button,
                               frame.continue_to_step5_button, frame.clear_button,
                               frame.previous_roi_button, frame.next_roi_button):
                    assert 'disabled' in button.state(), f'{button} remained actionable without an image.'
                assert frame.empty_placeholder.winfo_exists()
                for iid in frame.roi_table.get_children(''):
                    values = frame.roi_table.item(iid, 'values')
                    assert '!' not in values[0], 'Closed-file review badge remained in ROI table.'
                    assert all(value == '' for value in values[1:]), 'Closed-file measurements remained.'

            try:
                with TemporaryDirectory() as temporary:
                    image = np.repeat(np.linspace(20, 80, 450, dtype=np.int16)[:, None], 3000, axis=1)
                    paths = []
                    for side in ('nasal', 'temporal', 'new-study'):
                        path = Path(temporary) / side / '_flat_LIGHT.hdr'
                        path.parent.mkdir()
                        write_analyze(str(path), image)
                        paths.append(path)
                    saved_output = paths[0].parent / 'MAX_Stack.tif'
                    saved_output.write_bytes(b'already saved user output')

                    frame.batch_roi_paths = paths[:2].copy()
                    frame._open_batch_roi_tabs(paths[:2])
                    root.deiconify()
                    settle()
                    tabs = frame.batch_roi_notebook.tabs()
                    first_widget = frame.canvas.get_tk_widget()
                    frame.batch_roi_notebook.select(tabs[1])
                    settle()
                    second_canvas = frame.canvas
                    second_figure = frame.figure
                    second_widget = second_canvas.get_tk_widget()
                    frame.batch_roi_notebook.select(tabs[0])
                    settle()

                    # Closing an inactive cached tab must release its widget
                    # while preserving the active file and its rendered plot.
                    first_canvas = frame.canvas
                    frame._close_batch_roi_tab(frame.batch_roi_notebook, tabs[1])
                    settle()
                    assert not second_widget.winfo_exists()
                    assert second_figure.canvas is None, 'Closed cached figure retained its backend.'
                    assert second_figure.axes == [], 'Closed cached figure retained plot artists.'
                    assert frame.canvas is first_canvas
                    assert frame.current_path == paths[0]
                    assert set(frame.batch_roi_tab_states) == {str(tabs[0])}

                    # Reset clears loaded measurements and cached review state
                    # and detaches a worker already calculating the same file.
                    frame.auto_detection_reviews['01'] = 'old review'
                    frame.roi_clicks['01'] = [20.0, 40.0]
                    frame.profile_clicks = [20.0, 40.0]
                    frame._roi_adjust_bounds['01'] = False
                    events = start_detection()
                    frame._restart_current_file()
                    settle()
                    deliver_old_result(events)
                    state = frame.batch_roi_tab_states[str(tabs[0])]
                    for name in ('completed', 'roi_clicks', 'auto_detection_reviews', 'profile_clicks'):
                        assert not state[name], f'Restart retained cached {name}.'
                    assert frame.image is not None
                    assert 'disabled' not in frame.auto_detect_button.state()

                    # Final close is the reported reproduction: clicking Auto
                    # afterwards must never reconstruct an untabbed plot.
                    events = start_detection()
                    frame._close_batch_roi_tab(frame.batch_roi_notebook, tabs[0])
                    settle()
                    assert not first_widget.winfo_exists()
                    assert_empty()
                    deliver_old_result(events)
                    with mock.patch('aidas.steps.step4_analyze_isez.messagebox.showwarning'), \
                         mock.patch.object(frame, '_start_background_worker') as start:
                        frame._auto_detect_all_rois()
                    start.assert_not_called()
                    assert_empty()
                    assert saved_output.read_bytes() == b'already saved user output'

                    # A new study remains usable, and an active-tab close
                    # restores the surviving cached tab without old results.
                    frame.batch_roi_paths = paths[1:].copy()
                    frame._open_batch_roi_tabs(paths[1:])
                    settle()
                    tabs = frame.batch_roi_notebook.tabs()
                    events = start_detection()
                    frame.batch_roi_notebook.select(tabs[1])
                    settle()
                    deliver_old_result(events)
                    assert frame.current_path == paths[2]
                    assert 'disabled' not in frame.auto_detect_button.state()

                    # Queue a second click's idle save, switch files before it
                    # runs, and deliver that stale callback explicitly. The
                    # new file must not receive the previous file's selection.
                    frame.profile_clicks = [20.0]
                    callbacks = []
                    def capture_idle(callback, *args):
                        callbacks.append(lambda: callback(*args))
                        return 'captured-idle-job'
                    with mock.patch.object(frame, 'after_idle', side_effect=capture_idle):
                        frame._on_profile_click(SimpleNamespace(
                            inaxes=frame.ax_profile, xdata=40.0, button=1))
                    assert callbacks, 'Second click must queue its auto-save.'
                    frame.batch_roi_notebook.select(tabs[0])
                    settle()
                    frame.profile_clicks = [30.0, 50.0]
                    with mock.patch.object(frame, '_save_current_roi') as save:
                        for callback in callbacks:
                            callback()
                    save.assert_not_called()
                    assert frame.current_path == paths[1]
                    assert not frame.completed

                    frame._close_batch_roi_tab(frame.batch_roi_notebook, tabs[0])
                    settle()
                    assert frame.current_path == paths[2]
                    assert frame.canvas is not None

                    # The currently selected file still accepts a completion
                    # belonging to its own worker after old ones were ignored.
                    current_events = start_detection()
                    current_events.put(('done', AutoDetectionRunResult(
                        completed={}, reviews={'01': 'fresh review'},
                        roi_clicks={'01': [75.0, 99.0]}, removed_clicks=set(),
                        review_count=1, current_roi_idx=0,
                    )))
                    frame._poll_auto_detection_events(current_events)
                    settle()
                    assert frame.auto_detection_reviews == {'01': 'fresh review'}
                    assert frame.roi_clicks == {'01': [75.0, 99.0]}
                    assert frame.current_path == paths[2]

                    frame._close_batch_roi_tab(frame.batch_roi_notebook,
                                               frame.batch_roi_notebook.tabs()[0])
                    settle()
                    assert_empty()
                    assert saved_output.read_bytes() == b'already saved user output'

                    # Returning to folder selection also replaces the previous
                    # session, even when its tabs have not been closed first.
                    frame._open_batch_roi_tabs(paths[:2])
                    settle()
                    chooser_old_widget = frame.canvas.get_tk_widget()
                    events = start_detection()
                    frame._open_batch_roi_panel(Path(temporary))
                    settle()
                    deliver_old_result(events)
                    assert not chooser_old_widget.winfo_exists()
                    assert frame.image is None and frame.current_path is None
                    assert not frame.batch_roi_tab_states and not frame.batch_roi_paths
                    assert frame.batch_roi_panel.winfo_exists()
                    assert 'disabled' in frame.auto_detect_button.state()
                    assert saved_output.read_bytes() == b'already saved user output'
            finally:
                root.destroy()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
