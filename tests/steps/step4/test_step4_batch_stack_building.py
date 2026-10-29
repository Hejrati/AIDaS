from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from aidas.steps import step4_analyze_isez as module
from tests.support.paths import PROJECT_ROOT


class BatchStackBuildingTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        frame = self.frame = object.__new__(module.Step4Frame)
        frame.rois = [SimpleNamespace(suffix="01"), SimpleNamespace(suffix="02")]
        self.image = np.ones((8, 12), dtype=np.int16)
        self.states = frame.batch_roi_tab_states = {}
        for key in ("active", "ready", "unfinished", "built"):
            self.states[key] = dict(
                path=self.root / key / "_flat_LIGHT.hdr", complete=key == "built",
                completed={roi.suffix: SimpleNamespace(roi=roi) for roi in frame.rois},
                image=self.image,
            )
        self.states["unfinished"]["completed"].pop("02")
        frame._active_batch_roi_tab = "active"
        frame.completed = self.states["active"]["completed"]
        frame.batch_roi_notebook = object()
        frame._stack_building = False
        frame._stack_build_complete = False
        frame.build_stacks_button = mock.Mock()
        frame.auto_detect_button = mock.Mock()
        frame.status_var = mock.Mock()
        frame.output_dir_var = mock.Mock()
        frame.output_dir_var.get.return_value = str(self.root / "active")
        for name in ("_sync_active_batch_roi_state", "_show_stack_building_notice",
                     "_hide_grid_notice", "_show_processed_grid_notice",
                     "_refresh_batch_roi_tab_labels", "_update_continue_to_step5_button_state",
                     "_update_auto_detect_button_state"):
            setattr(frame, name, mock.Mock())
        self.workers = []
        self.callbacks = []
        frame._start_background_worker = self.workers.append
        frame.after = lambda delay, callback: self.callbacks.append(callback) or "poll"
        frame.after_cancel = mock.Mock()
        frame._write_stack_output_files = mock.Mock()
        self.warning = self.enterContext(mock.patch.object(module.messagebox, "showwarning"))
        self.error = self.enterContext(mock.patch.object(module.messagebox, "showerror"))

    def finish(self):
        self.workers.pop()()
        self.callbacks.pop(0)()

    def test_counter_counts_ready_tabs_and_excludes_unfinished_and_built(self):
        self.frame._update_build_stack_button_state()
        self.frame.build_stacks_button.configure.assert_called_with(
            text="Build Stack (2/4 ready to build)")
        self.frame._stack_building = True
        self.frame._update_build_stack_button_state()
        self.frame.build_stacks_button.configure.assert_called_with(
            text="Build Stack (2/4 ready to build)")
        self.frame.build_stacks_button.state.assert_called_with(["disabled"])

    def test_one_click_builds_every_ready_tab_and_skips_unfinished_and_built(self):
        self.frame._build_stack_outputs()
        self.assertTrue(self.frame._stack_building)
        self.frame.build_stacks_button.state.assert_called_with(["disabled"])
        self.finish()
        calls = self.frame._write_stack_output_files.call_args_list
        self.assertEqual([call.args[2] for call in calls],
                         [self.root / "active", self.root / "ready"])
        self.assertTrue(self.states["active"]["complete"])
        self.assertTrue(self.states["ready"]["complete"])
        self.assertFalse(self.states["unfinished"]["complete"])
        self.assertEqual(self.frame._active_batch_roi_tab, "active")
        self.frame.build_stacks_button.configure.assert_called_with(text="Build Stack (0/4 ready to build)")
        self.assertFalse(self.frame._stack_building)
        self.error.assert_not_called()

    def test_ready_inactive_tab_enables_build_when_active_tab_is_unfinished(self):
        self.frame.completed = self.states["active"]["completed"] = {}
        self.frame._update_build_stack_button_state()
        self.frame.build_stacks_button.state.assert_called_with(["!disabled"])
        self.frame.build_stacks_button.configure.assert_called_with(
            text="Build Stack (1/4 ready to build)")
        self.frame._build_stack_outputs()
        self.finish()
        self.assertEqual(self.frame._write_stack_output_files.call_count, 1)
        self.assertTrue(self.states["ready"]["complete"])
        self.assertFalse(self.states["active"]["complete"])

    def test_failure_does_not_complete_failed_tab_or_stop_other_ready_tabs(self):
        self.frame._write_stack_output_files.side_effect = [OSError("Disk full"), None]
        self.frame._build_stack_outputs()
        self.finish()
        self.assertFalse(self.states["active"]["complete"])
        self.assertTrue(self.states["ready"]["complete"])
        self.assertIn("Disk full", self.error.call_args.args[1])
        self.frame.build_stacks_button.configure.assert_called_with(
            text="Build Stack (1/4 ready to build)")
        self.frame.build_stacks_button.state.assert_called_with(["!disabled"])

    def test_pending_build_survives_tab_context_change_but_session_cancel_detaches_it(self):
        self.frame._build_stack_outputs()
        events = self.frame._stack_build_events
        self.frame._invalidate_roi_work()
        self.assertIs(self.frame._stack_build_events, events)
        self.assertTrue(self.frame._stack_building)
        self.frame._cancel_batch_stack_build()
        self.finish()
        self.frame._write_stack_output_files.assert_not_called()
        self.assertFalse(self.states["active"]["complete"])
        self.assertFalse(self.frame._stack_building)

    def test_changed_or_closed_tabs_are_not_marked_complete_by_stale_results(self):
        self.frame._build_stack_outputs()
        self.workers.pop()()
        self.states["active"]["completed"] = {}
        self.states.pop("ready")
        self.callbacks.pop(0)()
        self.assertFalse(self.states["active"]["complete"])

    def test_prepared_volumes_are_used_without_rereading_the_file(self):
        state = self.states["ready"]
        state.pop("image")
        state["_prepared_load"] = ("loaded", self.image[None, ...])
        with mock.patch.object(module, "load_oct_volume") as read:
            self.frame._build_stack_outputs()
            self.finish()
        read.assert_not_called()
        np.testing.assert_array_equal(self.frame._write_stack_output_files.call_args.args[0], self.image)

    def test_shared_output_folder_is_rejected_before_any_file_is_written(self):
        self.states["ready"]["output_dir"] = str(self.root / "active")
        self.frame._build_stack_outputs()
        self.warning.assert_called_once()
        self.assertFalse(self.workers)

    def test_second_click_does_not_launch_another_worker(self):
        self.frame._build_stack_outputs()
        self.frame._build_stack_outputs()
        self.assertEqual(len(self.workers), 1)

    def test_no_ready_tabs_stays_disabled_and_reports_why(self):
        self.states["active"]["completed"].clear()
        self.states["ready"]["completed"].clear()
        self.frame._update_build_stack_button_state()
        self.frame.build_stacks_button.state.assert_called_with(["disabled"])
        self.frame._build_stack_outputs()
        self.warning.assert_called_once()
        self.assertFalse(self.workers)

    def test_real_button_builds_inactive_ready_tab_without_changing_selection(self):
        script = textwrap.dedent('''
            from pathlib import Path
            from tempfile import TemporaryDirectory
            import threading
            import time
            from unittest import mock
            import customtkinter as ctk
            import numpy as np
            from aidas.steps import step4_analyze_isez as module

            root = ctk.CTk()
            root.attributes('-alpha', 0)
            root.geometry('1100x800+10000+10000')
            frame = module.Step4Frame(root)
            frame.pack(fill='both', expand=True)
            errors = []
            root.report_callback_exception = lambda *error: errors.append(error)
            owner = threading.get_ident()
            samples = np.arange(450)
            profile = 100 + 200 * np.exp(-((samples-88)/8)**2)
            volume = np.repeat(profile[None, :, None], 3000, axis=2)
            writer = frame._write_stack_output_files
            workers = []
            def write(*args):
                workers.append(threading.get_ident())
                time.sleep(.04)
                writer(*args)
            def wait_until(condition):
                deadline = time.monotonic() + 20
                while not condition():
                    assert time.monotonic() < deadline, 'Timed out'
                    root.update()
                    time.sleep(.002)
                root.update()
            try:
                with TemporaryDirectory() as directory, \
                     mock.patch.object(module, 'load_oct_volume', return_value=volume), \
                     mock.patch.object(frame, '_warm_auto_detection_tools'), \
                     mock.patch.object(frame, '_write_stack_output_files', side_effect=write), \
                     mock.patch.object(module.messagebox, 'showerror') as error:
                    frame.rois = frame.rois[:2]
                    paths = [Path(directory)/name/'_flat_LIGHT.hdr' for name in ('active', 'ready')]
                    frame._open_batch_roi_tabs(paths)
                    root.update()
                    keys = frame.batch_roi_notebook.tabs()
                    state = frame.batch_roi_tab_states[keys[1]]
                    state['completed'] = {
                        roi.suffix: module.analyze_and_save_roi(
                            volume[0], roi, start_click=76, end_click=100,
                        ) for roi in frame.rois
                    }
                    state['_prepared_load'] = ('loaded', volume)
                    frame._update_build_stack_button_state()
                    assert 'disabled' not in frame.build_stacks_button.state()
                    assert frame.build_stacks_button.cget('text') == 'Build Stack (1/2 ready to build)'
                    active = frame.batch_roi_notebook.select()
                    frame.build_stacks_button.invoke()
                    assert frame._stack_building
                    beats = []
                    root.after(1, lambda: beats.append(True))
                    wait_until(lambda: not frame._stack_building)
                    assert beats and workers and all(worker != owner for worker in workers)
                    assert state['complete']
                    assert frame.batch_roi_notebook.select() == active
                    assert frame.build_stacks_button.cget('text') == 'Build Stack (0/2 ready to build)'
                    assert frame.continue_to_step5_button.cget('text') == 'Go to Step 5 (1/2 completed)'
                    assert 'disabled' in frame.build_stacks_button.state()
                    for filename in ('MAX_Stack.tif', 'ROI_to_move_stck.tif', module.STEP4_RESULTS_FILENAME):
                        assert (paths[1].parent/filename).is_file(), filename
                        assert not (paths[0].parent/filename).exists()
                    error.assert_not_called()
                    assert not errors, errors
            finally:
                root.destroy()
        ''')
        result = subprocess.run([sys.executable, '-c', script], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
