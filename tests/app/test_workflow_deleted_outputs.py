import unittest
from unittest import mock

from aidas.app import AIDaSApp


class DeletedOutputProgressTests(unittest.TestCase):
    def test_invalidating_outputs_preserves_upstream_completion_and_active_step(self):
        for step in (1, 2, 3, 4, 5):
            with self.subTest(step=step):
                app = AIDaSApp.__new__(AIDaSApp)
                app._workflow_completed_steps = {0, 1, 2, 3, 4}
                app._workflow_active_step = 4
                app._workflow_cancelled_step = 2
                app._workflow_transition_source = 'old action'
                app._step_transition = mock.Mock()
                app._sync_workflow_progress_strip = mock.Mock()
                app._on_workflow_invalidated(step)
                self.assertEqual(app._workflow_completed_steps, set(range(step - 1)))
                self.assertEqual(app._workflow_active_step, 4)
                self.assertIsNone(app._workflow_cancelled_step)
                self.assertNotIn('_workflow_transition_source', app.__dict__)
                app._step_transition.destroy.assert_called_once()
                app._sync_workflow_progress_strip.assert_called_once()


if __name__ == '__main__':
    unittest.main()
