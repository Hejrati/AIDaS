from __future__ import annotations

import inspect
import unittest

from aidas.utils.ui_utils import HoverToolTip


class _WidgetStub:
    def __init__(self, owner=None) -> None:
        self.bindings = {}
        self.scheduled = []
        self.cancelled = []
        self.owner = owner

    def bind(self, sequence, callback, add=None):
        self.bindings[sequence] = callback

    def after(self, delay, callback):
        after_id = f"after-{len(self.scheduled) + 1}"
        self.scheduled.append((after_id, delay, callback))
        return after_id

    def after_cancel(self, after_id):
        self.cancelled.append(after_id)

    def winfo_toplevel(self):
        if self.owner is None:
            raise AttributeError("stub has no owner")
        return self.owner


class _OwnerStub:
    def __init__(self) -> None:
        self.bindings = {}
        self.bind_calls = []

    def bind(self, sequence, callback, add=None):
        self.bindings[sequence] = callback
        self.bind_calls.append((sequence, add))


class _TipWindowStub:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class HoverToolTipTests(unittest.TestCase):
    def tearDown(self):
        HoverToolTip._active_tooltip = None
        HoverToolTip._pending_tooltip = None

    def test_empty_text_never_schedules_a_window(self):
        widget = _WidgetStub()
        tooltip = HoverToolTip(widget, "")

        tooltip._schedule_show()

        self.assertEqual(widget.scheduled, [])

    def test_hide_cancels_pending_show_and_clears_active_window(self):
        widget = _WidgetStub()
        tooltip = HoverToolTip(widget, "Helpful text")
        tooltip._schedule_show()
        window = _TipWindowStub()
        tooltip.tipwindow = window
        HoverToolTip._active_tooltip = tooltip

        tooltip._hide()

        self.assertEqual(widget.cancelled, ["after-1"])
        self.assertTrue(window.destroyed)
        self.assertIsNone(tooltip.tipwindow)
        self.assertIsNone(HoverToolTip._active_tooltip)

    def test_dynamic_empty_text_immediately_removes_stale_tooltip(self):
        widget = _WidgetStub()
        tooltip = HoverToolTip(widget, "Old text")
        window = _TipWindowStub()
        tooltip.tipwindow = window
        HoverToolTip._active_tooltip = tooltip

        tooltip.text = ""

        self.assertTrue(window.destroyed)
        self.assertIsNone(HoverToolTip._active_tooltip)

    def test_plain_tk_popup_avoids_customtkinter_global_binding_leak(self):
        source = inspect.getsource(HoverToolTip._show_now)
        self.assertIn("tk.Toplevel", source)
        self.assertNotIn("ctk.CTkToplevel", source)

    def test_owner_lifecycle_events_dismiss_the_active_tooltip(self):
        owner = _OwnerStub()
        widget = _WidgetStub(owner)
        tooltip = HoverToolTip(widget, "Helpful text")

        for sequence in HoverToolTip._OWNER_DISMISS_EVENTS:
            with self.subTest(sequence=sequence):
                window = _TipWindowStub()
                tooltip.tipwindow = window
                HoverToolTip._active_tooltip = tooltip

                owner.bindings[sequence](None)

                self.assertTrue(window.destroyed)
                self.assertIsNone(tooltip.tipwindow)
                self.assertIsNone(HoverToolTip._active_tooltip)

    def test_owner_lifecycle_bindings_are_installed_only_once(self):
        owner = _OwnerStub()

        HoverToolTip(_WidgetStub(owner), "First")
        HoverToolTip(_WidgetStub(owner), "Second")

        self.assertEqual(
            owner.bind_calls,
            [(sequence, "+") for sequence in HoverToolTip._OWNER_DISMISS_EVENTS],
        )

    def test_owner_lifecycle_event_cancels_a_pending_tooltip(self):
        owner = _OwnerStub()
        widget = _WidgetStub(owner)
        tooltip = HoverToolTip(widget, "Helpful text")
        tooltip._schedule_show()

        owner.bindings["<FocusOut>"](None)

        self.assertEqual(widget.cancelled, ["after-1"])
        self.assertIsNone(tooltip._show_after_id)
        self.assertIsNone(HoverToolTip._pending_tooltip)


if __name__ == "__main__":
    unittest.main()
