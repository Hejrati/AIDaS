"""Regression coverage for repeated batch selection and folder input work."""

from __future__ import annotations

import queue
import unittest
from unittest import mock

from aidas.ui.batch_selection import BatchSelectionMixin
from aidas.ui.directory_completion import DirectoryCompletion


class _BatchTable(BatchSelectionMixin):
    def __init__(self, rows):
        self.rows = rows
        self._row_by_iid = {str(index): row for index, row in enumerate(rows)}
        self._checkbox_images = {"checked": "yes", "unchecked": "no", "locked": "locked"}
        self.tree = mock.Mock()
        self._notify_selection_changed = mock.Mock()

    @staticmethod
    def _values_for_row(row):
        return (row["name"],)


class BatchSelectionOptimizationTests(unittest.TestCase):
    def test_select_all_updates_only_changed_rows_and_preserves_locked_rows(self):
        rows = [
            {"name": "selected", "include": True},
            {"name": "ready", "include": False},
            {"name": "locked", "include": False, "locked": True},
        ]
        table = _BatchTable(rows)

        table._toggle_all_ready()

        table.tree.item.assert_called_once_with("1", image="yes", values=("ready",))
        table.tree.heading.assert_called_once_with("#0", image="yes")
        self.assertEqual(table.selected_rows(), rows[:2])
        self.assertFalse(rows[2]["include"])
        table._notify_selection_changed.assert_called_once_with()

        table.tree.reset_mock()
        table._notify_selection_changed.reset_mock()
        table._toggle_all_ready()

        self.assertEqual(
            table.tree.item.call_args_list,
            [mock.call("0", image="no", values=("selected",)),
             mock.call("1", image="no", values=("ready",))],
        )
        self.assertEqual(table.selected_rows(), [])
        self.assertFalse(rows[2]["include"])
        table.tree.heading.assert_called_once_with("#0", image="no")
        table._notify_selection_changed.assert_called_once_with()

    def test_empty_or_locked_selection_does_not_notify_or_redraw(self):
        for rows in ([], [{"name": "locked", "include": True, "locked": True}]):
            with self.subTest(rows=rows):
                table = _BatchTable(rows)
                table._toggle_all_ready()
                table.tree.item.assert_not_called()
                table._notify_selection_changed.assert_not_called()


class DirectoryCompletionOptimizationTests(unittest.TestCase):
    @staticmethod
    def _controller():
        # Exercise the request lifecycle without a Tk window or filesystem IO.
        controller = DirectoryCompletion.__new__(DirectoryCompletion)
        controller.target = mock.Mock()
        controller.target.get.return_value = "C:/example/al"
        controller.target.cget.return_value = "normal"
        controller.target.focus_get.return_value = controller.target
        controller.owner = mock.Mock()
        controller.owner.after.side_effect = ("poll1", "poll2", "poll3", "poll4")
        controller._disposed = False
        controller._generation = 0
        controller._pending = None
        controller._pending_action = None
        controller._query_text = None
        controller._results = queue.Queue()
        controller._delay_id = None
        controller._poll_id = None
        controller._hover_id = None
        controller._hovered = False
        controller._outside_id = None
        controller.popup = None
        controller.suggestions = []
        return controller

    def test_repeated_pending_query_keeps_one_worker_and_keyboard_action(self):
        controller = self._controller()
        with mock.patch("aidas.ui.directory_completion.threading.Thread") as worker:
            controller.refresh_now()
            generation, poll = controller._generation, controller._poll_id
            controller._pending_action = ("complete", -1)
            controller.refresh_now()
            controller.refresh_now()

        worker.assert_called_once()
        worker.return_value.start.assert_called_once_with()
        self.assertEqual(controller._generation, generation)
        self.assertEqual(controller._poll_id, poll)
        self.assertEqual(controller._pending_action, ("complete", -1))

    def test_navigation_while_pending_reuses_query_and_keeps_latest_action(self):
        controller = self._controller()
        with mock.patch("aidas.ui.directory_completion.threading.Thread") as worker:
            controller.refresh_now()
            self.assertEqual(controller._arrow(1), "break")
            self.assertEqual(controller._pending_action, ("highlight", 1))
            self.assertEqual(controller._arrow(-1), "break")
            self.assertEqual(controller._pending_action, ("highlight", -1))
            self.assertEqual(controller._tab(1), "break")

        worker.assert_called_once()
        self.assertEqual(controller._pending_action, ("complete", 1))

    def test_new_path_can_start_without_waiting_for_an_old_lookup(self):
        controller = self._controller()
        with mock.patch("aidas.ui.directory_completion.threading.Thread") as worker:
            controller.refresh_now()
            previous_generation = controller._generation
            controller.target.get.return_value = "C:/example/be"
            controller.refresh_now()

        self.assertEqual(worker.call_count, 2)
        self.assertGreater(controller._generation, previous_generation)
        self.assertEqual(controller._query_text, "C:/example/be")
        self.assertEqual(controller._pending, controller._generation)
        controller.owner.after_cancel.assert_called_once_with("poll1")

    def test_disabled_entry_invalidates_a_pending_request(self):
        controller = self._controller()
        with mock.patch("aidas.ui.directory_completion.threading.Thread") as worker:
            controller.refresh_now()
            controller.target.cget.return_value = "disabled"
            controller.refresh_now()

        worker.assert_called_once()
        self.assertIsNone(controller._pending)
        self.assertIsNone(controller._poll_id)


if __name__ == "__main__":
    unittest.main()
