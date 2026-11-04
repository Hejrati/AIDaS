"""Real Tk destruction cancels callbacks and releases discarded plot backends."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest

from tests.support.paths import PROJECT_ROOT


class Step4DestroyLifecycleTests(unittest.TestCase):
    def _run(self, body: str) -> None:
        # Separate interpreters isolate Tk/CTk global callbacks from other tests.
        source = textwrap.dedent("""
            from pathlib import Path
            from types import SimpleNamespace
            import queue
            from unittest import mock
            import customtkinter as ctk
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from aidas.steps.step4_analyze_isez import Step4Frame

            root = ctk.CTk()
            root.withdraw()
            def new_frame():
                frame = Step4Frame.__new__(Step4Frame)
                ctk.CTkFrame.__init__(frame, root)
                return frame
            try:
        """) + textwrap.indent(textwrap.dedent(body), "    ") + "\nfinally:\n    root.destroy()\n"
        result = subprocess.run(
            [sys.executable, "-c", source], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_destroy_cancels_pending_frame_and_animation_callbacks(self):
        self._run("""
            frame = new_frame()
            jobs = []
            for name in ('_auto_save_after_id', '_auto_detection_after_id',
                         '_stack_build_after_id', '_batch_roi_open_after_id'):
                job = frame.after(10000, lambda: None)
                jobs.append(job)
                setattr(frame, name, job)
            flash = frame.after(10000, lambda: None)
            jobs.append(flash)
            frame._roi_update_animation_jobs = {'01': flash}
            frame._flashing_updated_rois = {'01'}
            dialog = SimpleNamespace(close=mock.Mock())
            frame._profile_zoom_dialog = dialog
            frame.destroy()
            pending = root.tk.call('after', 'info')
            assert not set(jobs).intersection(pending), 'Destroyed Step 4 retained callbacks.'
            assert frame._roi_update_animation_jobs == {}
            assert frame._flashing_updated_rois == set()
            dialog.close.assert_called_once_with()
            assert frame._profile_zoom_dialog is None
            # Repeated cleanup must not call an already destroyed Tk command.
            frame.destroy()
        """)

    def test_late_worker_results_cannot_render_after_destroy(self):
        self._run("""
            frame = new_frame()
            detection, stack = queue.Queue(), queue.Queue()
            frame._auto_detection_events = detection
            frame._stack_build_events = stack
            frame._auto_detecting = frame._stack_building = True
            frame._apply_auto_detection_run = mock.Mock()
            frame._finish_stack_build = mock.Mock()
            for name in ('_update_auto_detect_button_state',
                         '_update_build_stack_button_state',
                         '_update_continue_to_step5_button_state'):
                setattr(frame, name, mock.Mock())
            frame.destroy()
            detection.put(('done', object()))
            stack.put(('done', None))
            frame._poll_auto_detection_events(detection)
            frame._poll_stack_build_events(stack, Path('unused-output'))
            frame._apply_auto_detection_run.assert_not_called()
            frame._finish_stack_build.assert_not_called()
            assert frame._auto_detection_events is None
            assert frame._stack_build_events is None
        """)

    def test_destroy_disposes_each_cached_plot_once_and_cancels_matplotlib_jobs(self):
        self._run("""
            frame = new_frame()
            plots, jobs = [], []
            for index in range(2):
                figure = Figure()
                axes = figure.add_subplot()
                canvas = FigureCanvasTkAgg(figure, master=frame)
                canvas.draw_idle()
                canvas._event_loop_id = canvas.get_tk_widget().after(10000, lambda: None)
                jobs.extend((canvas._idle_draw_id, canvas._event_loop_id))
                plots.append((canvas, figure, axes))
            frame.canvas, frame.figure, frame.ax_profile = plots[0]
            frame.ax_roi_grid = plots[0][2]
            active = {'canvas': plots[0][0], 'figure': plots[0][1],
                      'ax_profile': plots[0][2], 'ax_roi_grid': plots[0][2]}
            inactive = {'canvas': plots[1][0], 'figure': plots[1][1],
                        'ax_profile': plots[1][2], 'ax_roi_grid': plots[1][2]}
            frame.batch_roi_tab_states = {
                'active': active, 'inactive': inactive, 'shared': dict(inactive),
            }
            dispose = mock.Mock(wraps=frame._dispose_roi_plot)
            frame._dispose_roi_plot = dispose
            frame.destroy()
            assert dispose.call_count == 2, 'Each unique plot must be disposed once.'
            assert not set(jobs).intersection(root.tk.call('after', 'info'))
            for canvas, figure, axes in plots:
                assert figure.canvas is None, 'Figure retained its Tk backend.'
                assert figure.axes == [], 'Figure retained plot artists.'
                assert canvas._idle_draw_id is None
                assert canvas._event_loop_id is None
            for name in ('canvas', 'figure', 'ax_profile', 'ax_roi_grid'):
                assert getattr(frame, name) is None, name
                assert all(state[name] is None for state in frame.batch_roi_tab_states.values()), name
        """)

    def test_partial_startup_can_be_destroyed_repeatedly(self):
        self._run("""
            frame = new_frame()
            frame.destroy()
            frame.destroy()
            # Initialization can fail after animation ownership is established.
            partial = new_frame()
            job = partial.after(10000, lambda: None)
            partial._roi_update_animation_jobs = {'01': job}
            partial.destroy()
            assert job not in root.tk.call('after', 'info')
        """)


if __name__ == '__main__':
    unittest.main()
