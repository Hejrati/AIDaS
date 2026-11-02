"""An ended worker without a queue result cannot keep the UI polling forever."""

from types import SimpleNamespace
import unittest

from aidas.utils.worker_polling import WorkerCompletionGuard


class WorkerCompletionGuardTests(unittest.TestCase):
    def test_live_work_has_no_retry_or_duration_limit(self):
        guard = WorkerCompletionGuard(SimpleNamespace(is_alive=lambda: True))
        for _ in range(1000):
            self.assertFalse(guard.missing_result())

    def test_missing_result_is_reported_on_fifth_dead_worker_poll(self):
        guard = WorkerCompletionGuard(SimpleNamespace(is_alive=lambda: False))
        for _ in range(4):
            self.assertFalse(guard.missing_result())
        self.assertTrue(guard.missing_result())

    def test_intervening_live_work_resets_consecutive_missing_polls(self):
        alive = False
        guard = WorkerCompletionGuard(SimpleNamespace(is_alive=lambda: alive))
        for _ in range(4):
            self.assertFalse(guard.missing_result())
        alive = True
        self.assertFalse(guard.missing_result())
        alive = False
        for _ in range(4):
            self.assertFalse(guard.missing_result())
        self.assertTrue(guard.missing_result())
