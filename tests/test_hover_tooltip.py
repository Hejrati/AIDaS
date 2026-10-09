from __future__ import annotations

import inspect
import tkinter as tk
import unittest
from unittest import mock

import customtkinter as ctk

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

    def test_hover_schedules_immediately_and_next_button_cancels_pending_show(self):
        first_widget = _WidgetStub()
        second_widget = _WidgetStub()
        first = HoverToolTip(first_widget, "First help")
        second = HoverToolTip(second_widget, "Second help")

        first._schedule_show()

        self.assertEqual(first_widget.scheduled[0][1], 0)
        self.assertIsNone(first.tipwindow)
        self.assertIs(HoverToolTip._pending_tooltip, first)

        second._schedule_show()

        self.assertEqual(first_widget.cancelled, ["after-1"])
        self.assertIsNone(first._show_after_id)
        self.assertEqual(second_widget.scheduled[0][1], 0)
        self.assertIs(HoverToolTip._pending_tooltip, second)

        second._hide()
        self.assertEqual(second_widget.cancelled, ["after-1"])
        self.assertIsNone(HoverToolTip._pending_tooltip)

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

    def test_tooltip_centers_below_button_independently_of_pointer_and_clamps_to_screen(self):
        root = None

        class TransparentTipWindow(tk.Toplevel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.withdraw()
                self.attributes("-alpha", 0)

        try:
            try:
                root = ctk.CTk()
                root.withdraw()
                root.attributes("-alpha", 0)
                root.geometry("640x220+300+200")
            except tk.TclError as exc:
                self.skipTest(f"A transparent Tk test window is unavailable: {exc}")

            first_button = ctk.CTkButton(root, text="First", width=140, height=32)
            second_button = ctk.CTkButton(root, text="Second", width=140, height=32)
            first_button.place(x=4, y=60)
            second_button.place(x=360, y=60)
            first = HoverToolTip(first_button, "Helpful first button text with more detail about this action")
            second = HoverToolTip(second_button, "Helpful second button text")
            root.deiconify()
            root.update()
            self.assertTrue(first_button.winfo_viewable())
            self.assertTrue(second_button.winfo_viewable())

            with (
                mock.patch("aidas.utils.ui_utils.tk.Toplevel", TransparentTipWindow),
                mock.patch("aidas.utils.ui_utils.work_area_bounds", return_value=(0, 0, 10000, 10000)) as bounds,
                mock.patch.object(first, "_pointer_is_over_widget", return_value=True),
                mock.patch.object(second, "_pointer_is_over_widget", return_value=True),
            ):
                def show(tooltip, button, pointer_offset):
                    with (
                        mock.patch.object(button, "winfo_pointerx", return_value=button.winfo_rootx() + pointer_offset),
                        mock.patch.object(button, "winfo_pointery", return_value=button.winfo_rooty() + 2),
                    ):
                        tooltip._schedule_show()
                        self.assertIsNone(tooltip.tipwindow)
                        root.update()
                    popup = tooltip.tipwindow
                    self.assertIsNotNone(popup)
                    self.assertTrue(popup.winfo_ismapped())
                    self.assertIsNotNone(tooltip._pointer_after_id)
                    self.assertIsNone(tooltip._show_after_id)
                    return popup

                positions = []
                for offset in (2, first_button.winfo_width() - 2):
                    popup = show(first, first_button, offset)
                    positions.append((popup.winfo_rootx(), popup.winfo_rooty()))
                    self.assertEqual(popup.winfo_rootx(), first_button.winfo_rootx()
                                     + (first_button.winfo_width() - popup.winfo_width()) // 2)
                    self.assertLess(popup.winfo_rootx(), root.winfo_rootx())
                    self.assertEqual(popup.winfo_rooty(), first_button.winfo_rooty()
                                     + first_button.winfo_height() + 4)
                    if len(positions) == 1:
                        first._hide()
                self.assertEqual(positions[0], positions[1])

                old_popup = first.tipwindow
                popup = show(second, second_button, 2)
                self.assertFalse(old_popup.winfo_exists())
                self.assertIsNone(first.tipwindow)
                self.assertIsNone(first._pointer_after_id)
                self.assertIs(HoverToolTip._active_tooltip, second)
                self.assertEqual(popup.winfo_rootx(), second_button.winfo_rootx()
                                 + (second_button.winfo_width() - popup.winfo_width()) // 2)
                self.assertEqual(popup.winfo_rooty(), second_button.winfo_rooty()
                                 + second_button.winfo_height() + 4)

                x, y = popup.winfo_rootx(), popup.winfo_rooty()
                width, height = popup.winfo_width(), popup.winfo_height()
                for screen, expected in (
                    ((x + 10, y + 10, x + width + 100, y + height + 100), (x + 10, y + 10)),
                    ((x - 100, y - 100, x + width - 10, y + height - 10), (x - 10, y - 10)),
                ):
                    with self.subTest(screen=screen):
                        second._hide()
                        bounds.return_value = screen
                        popup = show(second, second_button, second_button.winfo_width() - 2)
                        self.assertEqual((popup.winfo_rootx(), popup.winfo_rooty()), expected)
                        bounds.assert_called_with(popup)

                root.event_generate("<<AIDaSThemeChanged>>")
                root.update()
                self.assertIsNone(second.tipwindow)
                self.assertIsNone(second._pointer_after_id)
                self.assertIsNone(HoverToolTip._active_tooltip)
        finally:
            HoverToolTip.dismiss_active()
            if root is not None:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
