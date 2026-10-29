from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps import step4_analyze_isez as step4
from tests.support.paths import PROJECT_ROOT


class Step4LoadingProgressTests(unittest.TestCase):
    def _run_ui(self, script):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=75,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_roi_substeps_advance_progress_before_a_file_finishes_and_preview_reserves_completion(self):
        self._run_ui(
            """
            from pathlib import Path
            import threading
            import time
            from unittest import mock
            import customtkinter as ctk
            import numpy as np
            from aidas.steps import step4_analyze_isez as module

            root = ctk.CTk()
            root.attributes('-alpha', 0)
            root.geometry('1000x700+10000+10000')
            frame = module.Step4Frame(root)
            frame.pack(fill='both', expand=True)
            root.update()
            owner = threading.get_ident()
            errors = []
            root.report_callback_exception = lambda *args: errors.append(args)
            reading, read_release = threading.Event(), threading.Event()
            phase_one, phase_one_release = threading.Event(), threading.Event()
            phase_two, phase_two_release = threading.Event(), threading.Event()
            updates, update_threads, preview = [], [], []
            image = np.ones((1, 450, 3000), dtype=np.int16)
            original_show = frame._show_batch_roi_loading_canvas
            original_render = frame._render_current_roi

            def load(path):
                assert threading.get_ident() != owner, 'Read ran on Tk.'
                reading.set()
                assert read_release.wait(20), 'Read was not released.'
                return image

            def detect(_image, _rois, _completed, _parameters, progress_callback=None):
                assert threading.get_ident() != owner, 'Detection ran on Tk.'
                assert progress_callback is not None
                progress_callback(0, 42)
                progress_callback(10, 42)
                phase_one.set()
                assert phase_one_release.wait(20), 'First ROI phase was not released.'
                progress_callback(30, 42)
                phase_two.set()
                assert phase_two_release.wait(20), 'Second ROI phase was not released.'
                progress_callback(42, 42)
                return module.AutoDetectionRunResult({}, {}, {}, set(), 0, 0)

            def show(total):
                surface = original_show(total)
                progress = surface._step4_progress
                original_set = progress.set
                def set_progress(value, *args, **kwargs):
                    update_threads.append(threading.get_ident())
                    updates.append(value)
                    return original_set(value, *args, **kwargs)
                progress.set = set_progress
                for label in (surface._step4_count_label, surface._step4_message_label):
                    original_configure = label.configure
                    def configure(*args, original=original_configure, **kwargs):
                        update_threads.append(threading.get_ident())
                        return original(*args, **kwargs)
                    label.configure = configure
                return surface

            def render():
                surface = frame._batch_roi_loading_surface
                assert surface is not None, 'Preview rendered without loading display.'
                preview.append((surface._step4_progress.get(),
                                surface._step4_message_label.cget('text'),
                                surface._step4_count_label.cget('text')))
                return original_render()

            def wait_until(condition):
                deadline = time.monotonic() + 25
                while not condition():
                    assert time.monotonic() < deadline, 'Loading transition timed out.'
                    root.update()
                    time.sleep(.002)
                root.update()

            try:
                with (
                    mock.patch.object(module, 'load_oct_volume', side_effect=load),
                    mock.patch.object(frame, '_compute_auto_detection_run', side_effect=detect),
                    mock.patch.object(frame, '_show_batch_roi_loading_canvas', side_effect=show),
                    mock.patch.object(frame, '_render_current_roi', side_effect=render),
                    mock.patch.object(frame, '_warm_auto_detection_tools'),
                    mock.patch.object(frame, '_warm_batch_roi_tools'),
                    mock.patch.object(module.messagebox, 'showerror') as show_error,
                ):
                    frame._open_batch_roi_tabs([Path('study/_flat_LIGHT.hdr')], show_progress=True)
                    wait_until(reading.is_set)
                    surface = frame._batch_roi_loading_surface
                    progress = surface._step4_progress
                    animation_phase = progress._indeterminate_value
                    wait_until(lambda: progress._indeterminate_value != animation_phase)
                    reading_value = progress.get()
                    assert '0 of 1 files processed' in surface._step4_count_label.cget('text')
                    assert 'Opening images' in surface._step4_message_label.cget('text')

                    read_release.set()
                    wait_until(phase_one.is_set)
                    wait_until(lambda: progress.get() > reading_value)
                    first_value = progress.get()
                    assert 0 < first_value < 1
                    assert progress.cget('mode') == 'determinate'
                    count = surface._step4_count_label.cget('text')
                    assert '0 of 1 files processed' in count, count
                    assert '1 opened' in count, count
                    # Even when work is blocked between substeps, elapsed time stays live.
                    wait_until(lambda: int(surface._step4_count_label.cget('text')
                                           .rsplit('|', 1)[1].split()[0]) >= 1)

                    phase_one_release.set()
                    wait_until(phase_two.is_set)
                    wait_until(lambda: progress.get() > first_value)
                    assert first_value < progress.get() < 1
                    assert '0 of 1 files processed' in surface._step4_count_label.cget('text')
                    assert 'automatic ROI detection' in surface._step4_message_label.cget('text')

                    phase_two_release.set()
                    wait_until(lambda: frame._batch_roi_open_after_id is None)
                    assert preview, 'The first preview was not rendered.'
                    for value, message, count in preview:
                        assert 0 < value < 1, 'Progress reached completion before rendering.'
                        assert 'Preparing the first preview' in message, message
                        assert '1 of 1 files processed' in count, count
                    assert updates == sorted(updates), updates
                    assert updates[-1] == 1.0, updates
                    assert all(value < 1 for value in updates[:-1]), updates
                    assert set(update_threads) == {owner}, update_threads
                    assert frame._batch_roi_loading_surface is None
                    assert frame.image is not None and frame.canvas is not None
                    assert root.grab_current() is None
                    assert not errors, errors
                    show_error.assert_not_called()
            finally:
                read_release.set()
                phase_one_release.set()
                phase_two_release.set()
                root.destroy()
            """
        )

    def test_failed_background_reads_remain_cached_across_tab_visits_and_retry_in_a_new_session(self):
        self._run_ui(
            """
            from collections import Counter
            from pathlib import Path
            import threading
            import time
            from unittest import mock
            import customtkinter as ctk
            import numpy as np
            from aidas.steps import step4_analyze_isez as module

            root = ctk.CTk()
            root.attributes('-alpha', 0)
            root.geometry('1000x700+10000+10000')
            frame = module.Step4Frame(root)
            frame.pack(fill='both', expand=True)
            root.update()
            owner = threading.get_ident()
            errors = []
            root.report_callback_exception = lambda *args: errors.append(args)
            reads = []
            repaired = threading.Event()
            paths = [Path(name) / '_flat_LIGHT.hdr' for name in ('first', 'broken', 'last')]
            images = {name: np.full((1, 450, 3000), value, dtype=np.int16)
                      for name, value in (('first', 3), ('broken', 7), ('last', 9))}

            def load(path):
                assert threading.get_ident() != owner, 'A cached tab retried I/O on Tk.'
                reads.append(path)
                name = Path(path).parent.name
                if name == 'broken' and not repaired.is_set():
                    raise OSError('Worker read failed: broken')
                return images[name]

            def detect(_image, _rois, _completed, _parameters, progress_callback=None):
                assert threading.get_ident() != owner
                progress_callback(42, 42)
                return module.AutoDetectionRunResult({}, {}, {}, set(), 0, 0)

            def wait_finished():
                deadline = time.monotonic() + 25
                while frame._batch_roi_open_after_id is not None:
                    assert time.monotonic() < deadline, 'Batch did not finish.'
                    root.update()
                    time.sleep(.002)
                root.update()

            try:
                with (
                    mock.patch.object(module, 'load_oct_volume', side_effect=load),
                    mock.patch.object(frame, '_compute_auto_detection_run', side_effect=detect),
                    mock.patch.object(frame, '_warm_auto_detection_tools'),
                    mock.patch.object(frame, '_warm_batch_roi_tools'),
                    mock.patch.object(module.messagebox, 'showerror') as show_error,
                ):
                    frame._open_batch_roi_tabs(paths, show_progress=True)
                    wait_finished()
                    assert Counter(reads) == Counter(paths), reads
                    assert '1 file(s) could not be processed' in frame.status_var.get()
                    keys = list(frame.batch_roi_notebook.tabs())
                    assert frame.batch_roi_tab_states[keys[1]]['_prepared_load'][0] == 'error'
                    assert frame.batch_roi_tab_states[keys[1]]['_prepared_load'][1].__traceback__ is None
                    show_error.assert_not_called()
                    for _ in range(2):
                        frame.batch_roi_notebook.select(keys[1])
                        root.update()
                        assert 'Worker read failed: broken' in show_error.call_args.args[1]
                        assert frame.batch_roi_tab_states[keys[1]]['_prepared_load'][1].__traceback__ is None
                        frame.batch_roi_notebook.select(keys[2])
                        root.update()
                        np.testing.assert_array_equal(frame.volume, images['last'])
                    assert show_error.call_count == 2
                    frame.batch_roi_notebook.select(keys[0])
                    root.update()
                    np.testing.assert_array_equal(frame.volume, images['first'])
                    assert Counter(reads) == Counter(paths), reads

                    # A new batch clears the old failure cache and can read a repaired file.
                    repaired.set()
                    frame._open_batch_roi_tabs(paths, show_progress=True)
                    wait_finished()
                    assert Counter(reads) == Counter({path: 2 for path in paths}), reads
                    keys = list(frame.batch_roi_notebook.tabs())
                    frame.batch_roi_notebook.select(keys[1])
                    root.update()
                    np.testing.assert_array_equal(frame.volume, images['broken'])
                    assert Counter(reads) == Counter({path: 2 for path in paths}), reads
                    assert show_error.call_count == 2
                    assert root.grab_current() is None
                    assert not errors, errors
            finally:
                root.destroy()
            """
        )


class Step4BatchToolPreparationTests(unittest.TestCase):
    def test_plot_preparation_queues_once_and_worker_needs_no_tk(self):
        frame = object.__new__(step4.Step4Frame)
        workers = []
        frame._start_background_worker = workers.append
        frame._warm_auto_detection_tools = mock.Mock()
        with mock.patch.dict(step4.sys.modules, {"matplotlib.backends.backend_tkagg": SimpleNamespace()}):
            frame._warm_batch_roi_tools()
            frame._warm_batch_roi_tools()
        self.assertEqual(len(workers), 1)
        self.assertEqual(frame._warm_auto_detection_tools.call_count, 2)
        # The frame has no Tcl interpreter or widget fields for a worker to use.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(workers[0]).result(timeout=20)

    def test_opening_folder_selection_starts_tool_preparation(self):
        frame = object.__new__(step4.Step4Frame)
        frame.plot_container = mock.Mock()
        frame.plot_container.winfo_children.return_value = []
        frame.empty_placeholder = None
        frame.input_dir_var = mock.Mock()
        for name in (
            "_close_batch_roi_panel", "_clear_batch_roi_session", "_hide_canvas_actions",
            "_set_batch_folder_label", "_notify_image_path_changed", "_refresh_roi_list",
            "_update_confirm_button_state", "_update_continue_to_step5_button_state",
            "_warm_batch_roi_tools",
        ):
            setattr(frame, name, mock.Mock())
        with mock.patch.object(step4, "Step4BatchROISelectionPanel") as panel:
            frame._open_batch_roi_panel(Path("study"))
        frame._warm_batch_roi_tools.assert_called_once_with()
        panel.assert_called_once_with(frame, frame.plot_container, Path("study"))


if __name__ == "__main__":
    unittest.main()
