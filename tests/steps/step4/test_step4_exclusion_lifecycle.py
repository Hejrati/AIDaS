"""Exercise exclusions through real widgets, tab caching, writers and Excel."""

import subprocess
import sys
import textwrap
import unittest

from tests.support.paths import PROJECT_ROOT


class Step4ExclusionLifecycleTests(unittest.TestCase):
    def test_tab_exclusions_round_trip_and_batch_writer_preserves_missing_roi_slots(self):
        script = textwrap.dedent(
            """
            from pathlib import Path
            from tempfile import TemporaryDirectory
            import gc
            import threading
            import time
            from unittest import mock
            import customtkinter as ctk
            import numpy as np
            from matplotlib.backend_bases import MouseEvent
            from openpyxl import load_workbook
            from aidas.services.step4_compiler import compile_step4_results
            from aidas.services.step4_results import STEP4_RESULTS_HEADERS, read_step4_measurements
            from aidas.steps import step4_analyze_isez as module

            # Tk finalizers must stay on this owning thread, not the writer.
            gc.disable()
            root = ctk.CTk()
            root.withdraw()
            root.attributes('-alpha', 0.0)
            root.geometry('1100x800+10000+10000')
            frame = module.Step4Frame(root)
            frame.pack(fill='both', expand=True)
            callback_errors = []
            root.report_callback_exception = lambda *error: callback_errors.append(error)
            owner = threading.get_ident()
            worker_threads = []
            writer = frame._write_stack_output_files
            def write(*args, **kwargs):
                worker_threads.append(threading.get_ident())
                writer(*args, **kwargs)
            def wait_until(condition):
                deadline = time.monotonic() + 30
                while not condition():
                    assert time.monotonic() < deadline, 'Timed out building excluded ROI stacks'
                    root.update()
                    time.sleep(.002)
                root.update()
            def select_roi(index):
                frame._move_roi(index - frame.current_roi_idx)
            def save_roi(index):
                select_roi(index)
                frame.profile_clicks = [76.0, 100.0]
                frame._remember_current_roi_clicks()
                frame._auto_save_current_roi()
                assert frame.rois[index].suffix in frame.completed
                assert frame.current_roi_idx == index
            def tick_roi(index, *, double=False, process_events=True):
                if process_events:
                    root.update()
                cell = frame.ax_roi_grid._aidas_grid_cells[index]
                bounds = cell['exclude_box'].get_window_extent()
                event = MouseEvent(
                    'button_press_event', frame.canvas,
                    (bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2,
                    button=1, dblclick=double,
                )
                frame._on_profile_click(event)
                if process_events:
                    root.update()
            samples = np.arange(450)
            profile = 100 + 200 * np.exp(-((samples - 88) / 8) ** 2)
            volume = np.repeat(profile[None, :, None], 3000, axis=2)
            try:
                with TemporaryDirectory() as temporary, \
                     mock.patch.object(module, 'load_oct_volume', return_value=volume), \
                     mock.patch.object(frame, '_warm_auto_detection_tools'), \
                     mock.patch.object(frame, '_open_automatic_profile_review'), \
                     mock.patch.object(frame, '_write_stack_output_files', side_effect=write), \
                     mock.patch.object(module.messagebox, 'showerror') as error, \
                     mock.patch.object(module.messagebox, 'showwarning') as warning:
                    directory = Path(temporary)
                    subject = directory / 'Control' / 'Subject_1'
                    paths = [subject / side / '_flat_LIGHT.hdr' for side in ('nasal', 'temporal')]
                    frame.rois = frame.rois[:3]
                    frame._open_batch_roi_tabs(paths)
                    root.deiconify()
                    root.update()
                    tabs = frame.batch_roi_notebook.tabs()

                    # All real ROIs start checked and included, even before a
                    # measurement is saved. Unchecking excludes the ROI.
                    assert frame.excluded_rois == set()
                    assert all(cell['exclude_tick'].get_visible()
                               for cell in frame.ax_roi_grid._aidas_grid_cells[:len(frame.rois)])

                    # A measured ROI remains reversible before stack creation.
                    save_roi(0)
                    first_result = frame.completed['01']
                    tick_roi(0)
                    assert frame.excluded_rois == {'01'}
                    assert not frame.ax_roi_grid._aidas_grid_cells[0]['exclude_tick'].get_visible()
                    assert frame.completed['01'] is first_result
                    assert all(value == '' for value in frame.roi_table.item('0', 'values')[1:])
                    # A double-click arrives after a first click. It must not
                    # toggle a second time or open the zoom editor.
                    tick_roi(0, double=True)
                    assert frame.excluded_rois == {'01'}
                    assert frame._profile_zoom_dialog is None
                    save_roi(1)
                    tick_roi(2)
                    assert frame.current_roi_idx == 1
                    assert '03' not in frame.completed
                    assert frame._all_rois_completed()

                    frame.next_image_button.invoke()
                    assert frame._active_batch_roi_tab == tabs[1]
                    assert frame.excluded_rois == set()
                    assert all(cell['exclude_tick'].get_visible()
                               for cell in frame.ax_roi_grid._aidas_grid_cells[:len(frame.rois)])
                    assert frame.current_roi_idx == 0
                    tick_roi(0)
                    assert '01' not in frame.completed
                    save_roi(1)
                    save_roi(2)
                    frame.batch_roi_notebook.select(tabs[0])
                    assert frame.excluded_rois == {'01', '03'}
                    assert frame.current_roi_idx == 1
                    tick_roi(0)
                    assert frame.excluded_rois == {'03'}
                    assert frame.current_roi_idx == 1
                    assert frame.ax_roi_grid._aidas_grid_cells[0]['exclude_tick'].get_visible()
                    assert frame.completed['01'] is first_result
                    assert any(value != '' for value in frame.roi_table.item('0', 'values')[1:])
                    tick_roi(0)
                    assert frame.excluded_rois == {'01', '03'}
                    assert frame.build_stacks_button.cget('text') == 'Build Stack (2/2 ready to build)'

                    # Both real files are written on the worker, including
                    # missing excluded results in active and inactive tabs.
                    frame.build_stacks_button.invoke()
                    assert frame._stack_building
                    assert not frame._can_change_roi_exclusion()
                    tick_roi(0, process_events=False)
                    assert frame.excluded_rois == {'01', '03'}
                    assert frame.next_image_button.cget('state') == 'disabled'
                    wait_until(lambda: not frame._stack_building)
                    assert all(state['complete'] for state in frame.batch_roi_tab_states.values())
                    assert frame._active_batch_roi_tab == tabs[0]
                    assert not frame._can_change_roi_exclusion()
                    tick_roi(0)
                    assert frame.excluded_rois == {'01', '03'}
                    assert worker_threads and all(thread != owner for thread in worker_threads)
                    expected = {}
                    for side, exclusions in (('nasal', {1, 3}), ('temporal', {1})):
                        output = subject / side
                        rows = read_step4_measurements(output / 'ROI_to_move_stck.tif')
                        assert [row[0] for row in rows] == [1, 2, 3]
                        for index, row in enumerate(rows, 1):
                            if index in exclusions:
                                assert row == [index] + [None] * 7
                            else:
                                assert isinstance(row[6], float)
                        workbook = load_workbook(output / 'rr_MCPAR.xlsx', data_only=True)
                        try:
                            saved = list(workbook.active.iter_rows(values_only=True))
                            assert saved[0] == STEP4_RESULTS_HEADERS
                            assert len(saved) == len(rows) + 1
                            for actual, expected_row in zip(saved[1:], rows):
                                assert actual[0] == expected_row[0]
                                for actual_value, expected_value in zip(actual[1:], expected_row[1:]):
                                    if expected_value is None:
                                        assert actual_value is None
                                    else:
                                        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-14)
                        finally:
                            workbook.close()
                        expected[side] = [row[6] for row in rows]
                        assert frame._step4_outputs_complete(output)
                    workbook_path = directory / 'compiled.xlsx'
                    compile_step4_results(directory, workbook_path)
                    workbook = load_workbook(workbook_path, data_only=True)
                    try:
                        sheet = workbook['rrMCP-AR']
                        for side, column in (('nasal', 3), ('temporal', 4)):
                            for row, expected_value in enumerate(expected[side], 3):
                                actual = sheet.cell(row, column).value
                                if expected_value is None:
                                    assert actual is None
                                else:
                                    np.testing.assert_allclose(actual, expected_value, rtol=1e-14)
                        assert [sheet.cell(row, 2).value for row in range(3, 6)] == [1, 2, 3]
                    finally:
                        workbook.close()

                    # Restarting one file clears only that file's exclusions.
                    frame._restart_current_file()
                    assert frame.excluded_rois == set()
                    assert frame.completed == {}
                    assert not frame.batch_roi_tab_states[tabs[0]]['complete']
                    assert frame.batch_roi_tab_states[tabs[1]]['excluded_rois'] == {'01'}
                    frame.batch_roi_notebook.select(tabs[1])
                    assert frame.excluded_rois == {'01'}
                    assert frame._stack_build_complete
                    frame._clear_batch_roi_session()
                    assert frame.excluded_rois == set()
                    assert frame.batch_roi_tab_states == {}
                    error.assert_not_called()
                    warning.assert_not_called()
                    assert callback_errors == [], callback_errors
            finally:
                root.destroy()
                gc.collect()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
