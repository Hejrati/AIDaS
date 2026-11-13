"""External deletion must not revive old source images or completed stacks."""
from pathlib import Path
import shutil
import unittest
from unittest import mock

import numpy as np

from aidas.steps import step4_analyze_isez as module
from aidas.utils.io_utils import write_analyze
from tests.steps.step4 import test_step4_auto_excel as export_fixtures


class DeletedStackLifecycleTests(unittest.TestCase):
    def setUp(self):
        fixture = self.fixture = export_fixtures.Step4ExcelHandoffTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.folder = self.root / 'nasal'
        self.folder.mkdir()
        self.source = self.folder / '_flat_LIGHT.hdr'
        samples = np.arange(450)
        self.image = np.repeat((100 + 200 * np.exp(-((samples - 88) / 8) ** 2))[:, None], 800, axis=1)
        self._write_source()
        self.frame = fixture._single_frame()
        self.frame.current_path = self.source
        self.frame._source_snapshot = module._source_snapshot(self.source)
        self.frame.output_dir_var.set(str(self.folder))
        self.frame.image = self.image
        self.frame.rois = [module.ISezROI('01', 101, 111)]
        self.frame.completed = {'01': module.analyze_and_save_roi(
            self.image, self.frame.rois[0], start_click=76, end_click=100,
        )}
        self.frame._write_stack_output_files = module.Step4Frame._write_stack_output_files
        self.frame.after_cancel = mock.Mock()

    def _write_source(self):
        self.folder.mkdir(exist_ok=True)
        write_analyze(self.source.with_suffix(''), np.stack([self.image, self.image]))

    def _delete_folder(self):
        self.assertTrue(self.folder.resolve().is_relative_to(self.root.resolve()))
        shutil.rmtree(self.folder)

    def _finish(self):
        self.fixture._finish_single(self.frame)

    def test_deleted_input_before_queued_build_does_not_recreate_folder(self):
        self.frame._build_stack_outputs()
        self._delete_folder()
        self._finish()
        self.assertFalse(self.folder.exists())
        self.assertFalse(self.frame._stack_build_complete)
        self.frame.on_continue_to_step5.assert_not_called()
        self.assertIn('deleted or changed', self.fixture.error.call_args.args[1])

    def test_recreated_source_at_same_path_cannot_use_old_measurements(self):
        self._delete_folder()
        self._write_source()
        self.frame._build_stack_outputs()
        self._finish()
        self.assertFalse((self.folder / module.STEP4_RESULTS_FILENAME).exists())
        self.assertFalse(self.frame._stack_build_complete)

    def test_deletion_during_rendering_stops_before_publication(self):
        render = module.make_isez_plot_image
        def delete_after_render(result):
            image = render(result)
            self._delete_folder()
            return image
        with mock.patch.object(module, 'make_isez_plot_image', side_effect=delete_after_render):
            self.frame._build_stack_outputs()
            self._finish()
        self.assertFalse(self.folder.exists())
        self.assertFalse(self.frame._stack_build_complete)
        self.frame.on_continue_to_step5.assert_not_called()

    def test_deleted_results_before_ui_delivery_are_not_reported_complete(self):
        self.frame._build_stack_outputs()
        self.frame._workers.pop(0)()
        (self.folder / 'rr_MCPAR.xlsx').unlink()
        self.frame._callbacks.pop(0)()
        self.assertFalse(self.frame._stack_build_complete)
        self.frame.on_continue_to_step5.assert_not_called()

    def test_deleted_workbook_reenables_build_and_new_handoff(self):
        self.frame._build_stack_outputs()
        self._finish()
        self.assertTrue(self.frame._stack_build_complete)
        generation = self.frame._step5_handoff_generation
        (self.folder / 'rr_MCPAR.xlsx').unlink()
        self.assertTrue(self.frame._revalidate_saved_stacks())
        self.assertFalse(self.frame._stack_build_complete)
        self.assertGreater(self.frame._step5_handoff_generation, generation)
        self.frame._build_stack_outputs()
        self._finish()
        self.assertTrue(self.frame._stack_build_complete)
        self.assertTrue((self.folder / 'rr_MCPAR.xlsx').is_file())
        self.assertEqual(self.frame.on_continue_to_step5.call_count, 2)

    def test_reset_cancels_single_file_write_before_worker_starts(self):
        self.frame._build_stack_outputs()
        self.frame._invalidate_roi_work()
        self._finish()
        self.assertFalse((self.folder / module.STEP4_RESULTS_FILENAME).exists())
        self.assertFalse(self.frame._stack_building)
        self.frame.on_continue_to_step5.assert_not_called()

    def test_repeated_continue_cannot_handoff_deleted_results(self):
        self.frame._build_stack_outputs()
        self._finish()
        self.frame.on_continue_to_step5.reset_mock()
        (self.folder / module.STEP4_RESULTS_FILENAME).unlink()
        self.frame._continue_to_step5()
        self.frame._continue_to_step5()
        self.frame.on_continue_to_step5.assert_not_called()
        self.assertFalse(self.frame._stack_build_complete)

    def test_batch_does_not_recreate_removed_folder_and_builds_other_ready_tab(self):
        fixture = self.fixture._batch_fixture(all_ready=False)
        frame = fixture.frame
        frame.rois = self.frame.rois
        frame.batch_roi_tab_states = {key: fixture.states[key] for key in ('active', 'ready')}
        for state in frame.batch_roi_tab_states.values():
            folder = state['path'].parent
            folder.mkdir()
            write_analyze(state['path'].with_suffix(''), np.stack([self.image, self.image]))
            state.update(source_snapshot=module._source_snapshot(state['path']),
                         completed=dict(self.frame.completed), image=self.image)
        frame.completed = frame.batch_roi_tab_states['active']['completed']
        frame._write_stack_output_files = module.Step4Frame._write_stack_output_files
        frame._build_stack_outputs()
        removed = frame.batch_roi_tab_states['active']['path'].parent
        self.assertTrue(removed.resolve().is_relative_to(fixture.root.resolve()))
        shutil.rmtree(removed)
        fixture.finish()
        self.assertFalse(removed.exists())
        self.assertFalse(frame.batch_roi_tab_states['active']['complete'])
        self.assertTrue(frame.batch_roi_tab_states['ready']['complete'])
        frame.on_continue_to_step5.assert_not_called()


class DeletedSourceTabTests(unittest.TestCase):
    def setUp(self):
        import customtkinter as ctk
        from tempfile import TemporaryDirectory
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.path = self.folder / '_flat_LIGHT.hdr'
        self.image = np.full((2, 450, 800), 50, dtype=np.int16)
        write_analyze(self.path.with_suffix(''), self.image)
        root = self.root = ctk.CTk()
        root.attributes('-alpha', 0)
        root.withdraw()
        self.addCleanup(self._close_root)
        self.errors = self.enterContext(mock.patch.object(module.messagebox, 'showerror'))
        self.frame = module.Step4Frame(root)
        self.frame.rois = [module.ISezROI('01', 101, 111)]
        self.enterContext(mock.patch.object(self.frame, '_warm_auto_detection_tools'))
        self.enterContext(mock.patch.object(self.frame, '_open_automatic_profile_review'))
        self.frame._open_batch_roi_tabs([self.path])
        root.update()
        self.tab = self.frame.batch_roi_notebook.nametowidget(self.frame.batch_roi_notebook.select())
        self.state = self.frame.batch_roi_tab_states[str(self.tab)]

    def _close_root(self):
        # This isolated interpreter owns CTk's process-wide theme timers too.
        for job in self.root.tk.call('after', 'info'):
            self.root.tk.call('after', 'cancel', job)
        self.root.destroy()

    def test_reselecting_same_tab_reloads_replaced_source_and_discards_old_roi_state(self):
        frame = self.frame
        frame.excluded_rois = {'01'}
        frame.roi_clicks = {'01': [2., 8.]}
        frame._sync_active_batch_roi_state()
        self.image[:] = 75
        write_analyze(self.path.with_suffix(''), self.image)
        frame._activate_batch_roi_tab(self.tab)
        np.testing.assert_array_equal(frame.image, self.image[0])
        self.assertEqual(frame.excluded_rois, set())
        self.assertEqual(frame.roi_clicks, {})
        self.assertEqual(frame.completed, {})
        self.errors.assert_not_called()

    def test_missing_source_error_does_not_poison_recreated_input_tab(self):
        self.path.unlink()
        self.path.with_suffix('.img').unlink()
        self.frame._activate_batch_roi_tab(self.tab)
        self.assertIsNone(self.frame.image)
        self.errors.assert_called_once()
        self.image[:] = 90
        write_analyze(self.path.with_suffix(''), self.image)
        self.frame._activate_batch_roi_tab(self.tab)
        np.testing.assert_array_equal(self.frame.image, self.image[0])
        self.assertTrue(self.state['loaded'])
        self.assertFalse(self.state['complete'])
        self.errors.assert_called_once()


if __name__ == '__main__':
    unittest.main()
