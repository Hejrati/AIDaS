"""Detect missing worker completion messages without timing out live work."""

from __future__ import annotations


class WorkerCompletionGuard:
    """Use only after an empty result queue, once per scheduled UI poll."""

    MAX_MISSING_POLLS = 5

    def __init__(self, worker):
        self.worker = worker
        self._missing_polls = 0

    def missing_result(self) -> bool:
        if self.worker.is_alive():
            self._missing_polls = 0
            return False
        self._missing_polls += 1
        return self._missing_polls >= self.MAX_MISSING_POLLS
