from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class Step4BatchLoadingTests(unittest.TestCase):
    def test_slow_reads_keep_progress_live_and_cancelled_results_cannot_replace_new_files(self):
        script = textwrap.dedent(
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
            errors = []
            root.report_callback_exception = lambda *args: errors.append(args)
            owner = threading.get_ident()
            reading = threading.Event()
            release = threading.Event()
            old_finished = threading.Event()
            image = np.ones((1, 450, 3000), dtype=np.int16)

            def load(path):
                assert threading.get_ident() != owner, 'File reads blocked the Tk thread.'
                if Path(path).parent.name == 'old':
                    reading.set()
                    assert release.wait(10), 'Test did not release old file read.'
                    old_finished.set()
                    return image * 9
                if Path(path).parent.name == 'broken':
                    raise OSError('Cannot read file')
                return image * 3

            def wait_until(condition):
                deadline = time.monotonic() + 20
                while not condition():
                    assert time.monotonic() < deadline, 'Loading timed out.'
                    root.update()
                    time.sleep(.002)
                root.update()

            try:
                with mock.patch.object(module, 'load_oct_volume', side_effect=load), mock.patch.object(
                        frame, '_warm_auto_detection_tools'), mock.patch.object(
                        module.messagebox, 'showerror') as show_error:
                    frame._open_batch_roi_tabs([Path('old/_flat_LIGHT.hdr')], show_progress=True)
                    wait_until(reading.is_set)
                    notice = frame._batch_roi_loading_surface
                    progress = notice._step4_progress
                    phase = progress._indeterminate_value
                    beats = []
                    def heartbeat():
                        beats.append(time.monotonic())
                        if len(beats) < 10:
                            root.after(10, heartbeat)
                    root.after(10, heartbeat)
                    wait_until(lambda: len(beats) == 10)
                    assert progress._indeterminate_value != phase, 'Progress stopped during the read.'
                    assert notice.winfo_exists()

                    # Replace a run while its worker is still blocked in I/O.
                    frame._open_batch_roi_tabs([Path('new/_flat_LIGHT.hdr')], show_progress=True)
                    wait_until(lambda: frame._batch_roi_open_after_id is None)
                    assert frame.current_path == Path('new/_flat_LIGHT.hdr')
                    np.testing.assert_array_equal(frame.volume, image * 3)
                    assert frame._batch_roi_loading_surface is None
                    assert root.grab_current() is None
                    release.set()
                    wait_until(old_finished.is_set)
                    assert frame.current_path == Path('new/_flat_LIGHT.hdr')
                    np.testing.assert_array_equal(frame.volume, image * 3)
                    show_error.assert_not_called()

                    # Read failures must also release progress and its modal grab.
                    frame._open_batch_roi_tabs([Path('broken/_flat_LIGHT.hdr')], show_progress=True)
                    wait_until(lambda: frame._batch_roi_open_after_id is None)
                    show_error.assert_called_once()
                    assert frame._batch_roi_loading_surface is None
                    assert frame.volume is None
                    assert root.grab_current() is None
                    assert not errors, errors
            finally:
                release.set()
                root.destroy()
            """
        )
        result = subprocess.run([sys.executable, '-c', script],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_run_paints_loading_controls_before_reading_the_first_file(self):
        script = textwrap.dedent(
            """
            import json
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from unittest import mock
            import customtkinter as ctk
            import numpy as np
            from aidas.app import AIDaSApp
            from aidas.core.config import Config
            from aidas.steps.step2_annotate import Step2Frame
            from aidas.steps import step4_analyze_isez as module

            def hidden_initializer(original):
                def initialize(window, *args, **kwargs):
                    original(window, *args, **kwargs)
                    window.attributes('-alpha', 0.0)
                return initialize

            image = np.repeat(np.linspace(20, 80, 450, dtype=np.int16)[:, None], 3000, axis=1)
            app = None
            with TemporaryDirectory() as directory:
                config = Path(directory) / 'preferences.json'
                config.write_text(json.dumps({
                    'interface_mode': 'Modern', 'check_for_updates': False,
                    'show_step2_annotation_tutorial': False,
                }), encoding='utf-8')
                with (
                    mock.patch.object(Config, 'CONFIG_DIR', Path(directory)),
                    mock.patch.object(Config, 'CONFIG_FILE', config),
                    mock.patch.object(ctk.CTk, '__init__', hidden_initializer(ctk.CTk.__init__)),
                    mock.patch.object(ctk.CTkToplevel, '__init__', hidden_initializer(ctk.CTkToplevel.__init__)),
                    mock.patch.object(Step2Frame, '_start_ai_device_probe'),
                    mock.patch.object(module.Step4Frame, '_warm_auto_detection_tools'),
                    mock.patch.object(module.Step4BatchROISelectionPanel, '_start_scan'),
                    mock.patch.object(module.messagebox, 'showerror') as show_error,
                ):
                    try:
                        app = AIDaSApp()
                        app._finish_startup()
                        app.geometry('1150x720+10000+10000')
                        frame = app.step4
                        app.notebook.select(frame)
                        app.update()
                        errors = []
                        app.report_callback_exception = lambda *error: errors.append(error)

                        for total in (1, 18):
                            rows = [dict(
                                folder=Path(directory) / f'study-{index}',
                                flat_light=Path(directory) / f'study-{index}/_flat_LIGHT.hdr',
                                include=True, locked=False, status='Ready', outputs='0/3',
                            ) for index in range(total)]
                            frame._open_batch_roi_panel(Path(directory))
                            panel = frame.batch_roi_panel
                            panel._scan_done(rows, total, 0, [])
                            app.update()
                            observations = []

                            def read_first_file(path):
                                # Workers must never inspect Tk widgets.
                                return image[None, ...]

                            def observe_loading():
                                surface = frame._batch_roi_loading_surface
                                message = surface._step4_message_label._label
                                progress = surface._step4_progress._canvas
                                observations.append({
                                    'message_mapped': bool(message.winfo_ismapped()),
                                    'progress_mapped': bool(progress.winfo_ismapped()),
                                    'progress_width': progress.winfo_width(),
                                    'progress_height': progress.winfo_height(),
                                })

                            def poll_finished():
                                if frame._batch_roi_loading_surface is not None and not observations:
                                    observe_loading()
                                if frame._batch_roi_open_after_id is not None:
                                    app.after(10, poll_finished)
                                else:
                                    app.quit()

                            with mock.patch.object(module, 'load_oct_volume', side_effect=read_first_file):
                                # Invoke the real Run button through Tk, without a
                                # test-owned redraw between Run and file reading.
                                app.after(0, panel.process_button.invoke)
                                app.after(10, poll_finished)
                                timeout = app.after(20000, app.quit)
                                app.mainloop()
                                app.after_cancel(timeout)

                            assert frame._batch_roi_open_after_id is None, 'Loading timed out.'
                            assert len(observations) == 1, observations
                            observed = observations[0]
                            assert observed['message_mapped'], observed
                            assert observed['progress_mapped'], observed
                            assert observed['progress_width'] > 100 and observed['progress_height'] > 1, observed
                            assert frame._batch_roi_loading_surface is None
                            assert len(frame.batch_roi_notebook.tabs()) == total
                            assert frame.image is not None
                            assert not errors, errors
                            show_error.assert_not_called()
                    finally:
                        if app is not None:
                            app.destroy()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_run_shows_responsive_canvas_progress_and_cleans_up_loading(self):
        # Isolate CTk's process-wide theme and timer callbacks from other UI tests.
        script = textwrap.dedent(
            """
            from pathlib import Path
            import time
            from unittest import mock
            import customtkinter as ctk
            import numpy as np
            from aidas.steps import step4_analyze_isez as module

            root = ctk.CTk()
            root.withdraw()
            root.attributes('-alpha', 0.0)
            root.geometry('1150x720+10000+10000')
            frame = module.Step4Frame(root)
            frame.pack(fill='both', expand=True)
            root.deiconify()
            root.update()
            errors = []
            root.report_callback_exception = lambda *error: errors.append(error)
            image = np.repeat(np.linspace(20, 80, 450, dtype=np.int16)[:, None], 3000, axis=1)

            def rows(count):
                return [{'flat_light': Path(f'study-{index}') / '_flat_LIGHT.hdr'}
                        for index in range(count)]

            def wait_for_loading():
                deadline = time.monotonic() + 20
                while frame._batch_roi_open_after_id is not None:
                    assert time.monotonic() < deadline, 'File preparation did not finish.'
                    root.update()
                    time.sleep(0.002)
                root.update()
                assert not errors, errors

            def load_first_file(path):
                return image[None, ...]

            try:
                with (
                    mock.patch.object(frame, '_warm_auto_detection_tools'),
                    mock.patch.object(module, 'load_oct_volume', side_effect=load_first_file) as load,
                    mock.patch.object(module.messagebox, 'showerror') as show_error,
                ):
                    frame._start_batch_roi_from_rows(rows(18))
                    notebook = frame.batch_roi_notebook
                    notice = frame._batch_roi_loading_surface
                    progress = notice._step4_progress
                    assert not notebook.tabs(), 'Run must return before the tabs are built.'
                    assert 'Please wait, files are getting ready' in notice._step4_message_label.cget('text')
                    viewport = frame.plot_container
                    assert notice.winfo_width() == viewport.winfo_width(), 'Loading must replace the entire canvas.'
                    assert notice.winfo_height() == viewport.winfo_height(), 'Loading must replace the entire canvas.'
                    message_label = notice._step4_message_label
                    assert abs(message_label.winfo_rootx() + message_label.winfo_width() / 2
                               - viewport.winfo_rootx() - viewport.winfo_width() / 2) <= 1
                    assert abs(message_label.winfo_rooty() + message_label.winfo_height() / 2
                               - viewport.winfo_rooty() - viewport.winfo_height() / 2) <= 1
                    assert progress.winfo_rooty() > message_label.winfo_rooty() + message_label.winfo_height()
                    assert viewport.winfo_children()[-1] is notice, 'Loading display must be above every canvas child.'

                    heartbeat_counts = []
                    def heartbeat():
                        if frame._batch_roi_open_after_id is not None:
                            assert notice.winfo_exists()
                            heartbeat_counts.append(len(notebook.tabs()))
                            root.after(3, heartbeat)
                    root.after(1, heartbeat)
                    wait_for_loading()

                    assert len(notebook.tabs()) == 18
                    assert any(0 < count < 18 for count in heartbeat_counts), heartbeat_counts
                    assert frame.current_path == rows(18)[0]['flat_light']
                    assert frame.image is not None and frame.canvas is not None
                    assert load.call_count == 1, 'Only the active tab should load its image.'
                    assert not notice.winfo_exists()
                    assert not progress._loop_running
                    assert root.grab_current() is None
                    show_error.assert_not_called()

                    # Replacing a pending run must cancel its callbacks and indicator.
                    frame._start_batch_roi_from_rows(rows(18))
                    old_notebook = frame.batch_roi_notebook
                    old_notice = frame._batch_roi_loading_surface
                    old_progress = old_notice._step4_progress
                    frame._start_batch_roi_from_rows(rows(2))
                    wait_for_loading()
                    assert not old_notebook.winfo_exists()
                    assert not old_notice.winfo_exists() and not old_progress._loop_running
                    assert len(frame.batch_roi_notebook.tabs()) == 2
                    assert load.call_count == 2

                    # A failed tab must release the grab and leave the canvas usable.
                    with mock.patch.object(module.ClosableTabView, 'add', side_effect=RuntimeError('Tab failed')):
                        frame._start_batch_roi_from_rows(rows(3))
                        wait_for_loading()
                    assert frame.batch_roi_notebook is None
                    assert frame.empty_placeholder.winfo_exists()
                    assert root.grab_current() is None
                    show_error.assert_called_once()

                    # Closing the view while preparing must stop its pending work.
                    frame._start_batch_roi_from_rows(rows(18))
                    pending_notice = frame._batch_roi_loading_surface
                    pending_progress = pending_notice._step4_progress
                    frame.destroy()
                    root.after(40, root.quit)
                    root.mainloop()
                    assert not pending_notice.winfo_exists() and not pending_progress._loop_running
                    assert root.grab_current() is None
                    assert load.call_count == 2
                    assert not errors, errors
            finally:
                root.destroy()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
