from __future__ import annotations

from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest import mock

from aidas.ui.prompt_cue import PromptCue


class _Scheduler:
    """A deterministic event loop with Tk's native binding entry points."""

    _w = ".prompt"

    def __init__(self):
        self.now = 0
        self.jobs = {}
        self.bindings = {}
        self._next_id = 0

    def _bind(self, what, sequence, callback, add):
        self.bindings["destroy"] = callback
        return "destroy"

    def _unbind(self, what, binding):
        self.bindings.pop(binding, None)

    def after(self, delay, callback):
        self._next_id += 1
        job = f"after#{self._next_id}"
        self.jobs[job] = (self.now + delay, callback)
        return job

    def after_cancel(self, job):
        self.jobs.pop(job, None)

    def advance(self, milliseconds):
        end = self.now + milliseconds
        while self.jobs:
            job, (when, callback) = min(self.jobs.items(), key=lambda item: item[1][0])
            if when > end:
                break
            self.now = when
            self.jobs.pop(job)
            callback()
        self.now = end

    def destroy(self, widget=None):
        self.bindings["destroy"](SimpleNamespace(widget=self if widget is None else widget))


class PromptCueTests(unittest.TestCase):
    def setUp(self):
        self.owner = _Scheduler()
        self.cue = PromptCue(self.owner)
        self.offsets = []

    def test_two_finite_oscillations_stay_within_four_logical_pixels(self):
        self.cue.start(self.offsets.append)
        self.owner.advance(600)

        self.assertEqual(self.offsets[0], 0)
        self.assertEqual(self.offsets[-1], 0)
        self.assertLessEqual(max(map(abs, self.offsets)), 4)
        signs = [1 if value > 0 else -1 for value in self.offsets if value]
        runs = [sign for index, sign in enumerate(signs) if not index or signs[index - 1] != sign]
        self.assertEqual(runs, [1, -1, 1, -1])
        self.assertEqual(self.owner.jobs, {})
        resting_offsets = self.offsets.copy()
        self.owner.advance(5000)
        self.assertEqual(self.offsets, resting_offsets)

    def test_stop_resets_motion_and_cancels_pending_work_even_if_a_callback_is_stale(self):
        self.cue.start(self.offsets.append)
        stale_callbacks = [callback for _when, callback in self.owner.jobs.values()]
        self.owner.advance(30)
        self.assertNotEqual(self.offsets[-1], 0)
        self.cue.stop()
        self.assertEqual(self.offsets[-1], 0)
        self.assertEqual(self.owner.jobs, {})
        count = len(self.offsets)
        for callback in stale_callbacks:
            callback()
        self.assertEqual(len(self.offsets), count)
        self.assertEqual(self.owner.jobs, {})

    def test_new_stage_resets_old_motion_and_completes_its_own_cue(self):
        old_offsets, new_offsets = [], []
        self.cue.start(old_offsets.append)
        self.owner.advance(30)
        self.cue.start(new_offsets.append)
        self.assertEqual(old_offsets[-1], 0)
        old_count = len(old_offsets)
        self.owner.advance(600)
        self.assertEqual(len(old_offsets), old_count)
        self.assertEqual(new_offsets[-1], 0)
        self.assertEqual(self.owner.jobs, {})

    def test_owner_destruction_cancels_jobs_but_child_destruction_does_not(self):
        self.cue.start(self.offsets.append)
        self.owner.destroy(widget=object())
        self.assertEqual(len(self.owner.jobs), 1)
        self.owner.advance(30)
        self.owner.destroy()
        self.assertEqual(self.offsets[-1], 0)
        self.assertEqual(self.owner.jobs, {})
        self.cue.start(self.offsets.append)
        self.assertEqual(self.owner.jobs, {})

    def test_destroy_removes_only_the_cues_owner_binding_and_is_idempotent(self):
        self.cue.start(self.offsets.append)
        self.owner.bindings["another_handler"] = mock.Mock()
        self.cue.destroy()
        self.cue.destroy()
        self.assertEqual(self.owner.jobs, {})
        self.assertNotIn("destroy", self.owner.bindings)
        self.assertIn("another_handler", self.owner.bindings)

    def test_destroyed_motion_target_stops_all_pending_work(self):
        move = mock.Mock(side_effect=[None, tk.TclError("widget was destroyed"),
                                      tk.TclError("widget was destroyed")])
        self.cue.start(move)
        self.owner.advance(30)
        self.assertEqual(self.owner.jobs, {})


if __name__ == "__main__":
    unittest.main()
