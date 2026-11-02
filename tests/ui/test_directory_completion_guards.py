"""Folder suggestions stop polling after a failed worker and yield busy queues."""

import queue
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.ui.directory_completion import DirectoryCompletion
from aidas.utils.worker_polling import WorkerCompletionGuard


class DirectoryCompletionGuardTests(unittest.TestCase):
    def _completion(self, *, alive):
        completion = object.__new__(DirectoryCompletion)
        completion._disposed = False
        completion._pending = 1
        completion._generation = 1
        completion._results = queue.Queue()
        completion.owner = SimpleNamespace(after=mock.Mock(return_value="poll"))
        completion._lookup_guard = WorkerCompletionGuard(SimpleNamespace(is_alive=lambda: alive))
        completion.hide = mock.Mock(side_effect=lambda: setattr(completion, "_pending", None))
        return completion

    def test_dead_suggestion_worker_does_not_leave_an_endless_pending_lookup(self):
        completion = self._completion(alive=False)
        for _ in range(5):
            completion._poll_results()
        completion.hide.assert_called_once_with()
        self.assertEqual(completion.owner.after.call_count, 4)
        self.assertIsNone(completion._pending)

    def test_live_suggestion_worker_is_not_timed_out_by_poll_count(self):
        completion = self._completion(alive=True)
        for _ in range(50):
            completion._poll_results()
        completion.hide.assert_not_called()
        self.assertEqual(completion.owner.after.call_count, 50)

    def test_old_lookup_results_are_drained_in_bounded_batches(self):
        completion = self._completion(alive=False)
        for _ in range(500):
            completion._results.put((0, "old", []))
        completion._poll_results()
        self.assertEqual(completion._results.qsize(), 500-completion.MAX_RESULTS_PER_POLL)
        completion.hide.assert_not_called()
        completion.owner.after.assert_called_once()
