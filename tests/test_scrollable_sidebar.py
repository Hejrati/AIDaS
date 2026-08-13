from __future__ import annotations

import inspect
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest import mock

import aidas.utils.ui_utils as ui_utils
import customtkinter as ctk
from aidas.ui.theme import CONTROLS
from aidas.utils.ui_utils import ScrollableSidebar


class _WidgetStub:
    def __init__(self, master=None) -> None:
        self.master = master


class _ListboxStub(_WidgetStub):
    def __init__(self, master=None, *, yview=(0.0, 1.0)) -> None:
        super().__init__(master)
        self._yview = yview

    def yview(self):
        return self._yview


class _TextStub(_WidgetStub):
    pass


class _CanvasWidgetStub(_WidgetStub):
    pass


class _TreeviewStub(_WidgetStub):
    pass


class _ScrollbarWidgetStub(_WidgetStub):
    pass


class _OuterCanvasStub:
    def __init__(self) -> None:
        self.scroll_calls = []

    def yview_scroll(self, units, kind):
        self.scroll_calls.append((units, kind))


class ScrollableSidebarTests(unittest.TestCase):
    def _sidebar(self):
        sidebar = object.__new__(ScrollableSidebar)
        sidebar.content = _WidgetStub()
        sidebar.canvas = _OuterCanvasStub()
        sidebar.focus_get = lambda: None
        sidebar._contains_pointer = lambda *_args, **_kwargs: True
        sidebar._contains_scroll_area_pointer = lambda: True
        sidebar._show_scrollbar = mock.Mock()
        return sidebar

    @staticmethod
    def _wheel_event(widget, *, delta=0, num=None):
        return SimpleNamespace(widget=widget, delta=delta, num=num)

    def _nested_widget_types(self):
        return (
            mock.patch.object(ui_utils.tk, "Listbox", _ListboxStub),
            mock.patch.object(ui_utils.tk, "Text", _TextStub),
            mock.patch.object(ui_utils.tk, "Canvas", _CanvasWidgetStub),
            mock.patch.object(ui_utils.tk, "Scrollbar", _ScrollbarWidgetStub),
            mock.patch.object(ui_utils.ttk, "Treeview", _TreeviewStub),
            mock.patch.object(ui_utils.ttk, "Scrollbar", _ScrollbarWidgetStub),
        )

    def test_nested_list_under_pointer_always_owns_wheel_without_prior_click(self):
        sidebar = self._sidebar()
        owner = _ListboxStub(sidebar.content, yview=(0.2, 0.7))
        event_child = _WidgetStub(owner)

        patches = self._nested_widget_types()
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        result = sidebar._on_mousewheel(self._wheel_event(event_child, delta=-120))

        self.assertIsNone(result)
        self.assertEqual(sidebar.canvas.scroll_calls, [])
        sidebar._show_scrollbar.assert_not_called()

    def test_every_nested_scrollable_view_owns_the_wheel_under_its_pointer(self):
        patches = self._nested_widget_types()
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        owner_factories = (
            lambda parent: _ListboxStub(parent, yview=(0.0, 0.5)),
            _TextStub,
            _CanvasWidgetStub,
            _TreeviewStub,
        )
        for owner_factory in owner_factories:
            with self.subTest(owner_factory=owner_factory):
                sidebar = self._sidebar()
                owner = owner_factory(sidebar.content)

                result = sidebar._on_mousewheel(
                    self._wheel_event(owner, delta=-120)
                )

                self.assertIsNone(result)
                self.assertEqual(sidebar.canvas.scroll_calls, [])
                sidebar._show_scrollbar.assert_not_called()

    def test_nested_list_does_not_chain_wheel_to_sidebar_at_either_boundary(self):
        patches = self._nested_widget_types()
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        cases = (
            ((0.0, 0.5), 120),
            ((0.5, 1.0), -120),
        )
        for yview, delta in cases:
            with self.subTest(yview=yview, delta=delta):
                sidebar = self._sidebar()
                owner = _ListboxStub(sidebar.content, yview=yview)

                result = sidebar._on_mousewheel(
                    self._wheel_event(owner, delta=delta)
                )

                self.assertIsNone(result)
                self.assertEqual(sidebar.canvas.scroll_calls, [])
                sidebar._show_scrollbar.assert_not_called()

    def test_non_overflowing_fixed_list_allows_outer_sidebar_scrolling(self):
        patches = self._nested_widget_types()
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        sidebar = self._sidebar()
        fixed_list = _ListboxStub(sidebar.content, yview=(0.0, 1.0))

        result = sidebar._on_mousewheel(
            self._wheel_event(fixed_list, delta=-120)
        )

        self.assertEqual(result, "break")
        self.assertEqual(sidebar.canvas.scroll_calls, [(1, "units")])
        sidebar._show_scrollbar.assert_called_once_with()

    def test_nested_list_scrollbar_does_not_move_or_reveal_outer_sidebar(self):
        patches = self._nested_widget_types()
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        for scrollbar_type in (ui_utils.tk.Scrollbar, ui_utils.ttk.Scrollbar):
            with self.subTest(scrollbar_type=scrollbar_type):
                sidebar = self._sidebar()
                nested_scrollbar = scrollbar_type(sidebar.content)

                result = sidebar._on_mousewheel(
                    self._wheel_event(nested_scrollbar, delta=-120)
                )

                self.assertIsNone(result)
                self.assertEqual(sidebar.canvas.scroll_calls, [])
                sidebar._show_scrollbar.assert_not_called()

    def test_outer_sidebar_wheel_reveals_scrollbar_and_moves_canvas(self):
        sidebar = self._sidebar()
        ordinary_child = _WidgetStub(sidebar.content)

        result = sidebar._on_mousewheel(
            self._wheel_event(ordinary_child, delta=-120)
        )

        self.assertEqual(result, "break")
        self.assertEqual(sidebar.canvas.scroll_calls, [(1, "units")])
        sidebar._show_scrollbar.assert_called_once_with()

    def test_fixed_footer_wheel_still_scrolls_the_outer_sidebar(self):
        sidebar = self._sidebar()
        sidebar_shell = _WidgetStub()
        footer_child = _WidgetStub(sidebar_shell)

        result = sidebar._on_mousewheel(
            self._wheel_event(footer_child, delta=120)
        )

        self.assertEqual(result, "break")
        self.assertEqual(sidebar.canvas.scroll_calls, [(-1, "units")])
        sidebar._show_scrollbar.assert_called_once_with()

    def test_hover_bindings_reveal_and_hide_scrollbar(self):
        source = inspect.getsource(ScrollableSidebar.__init__)
        pointer_source = inspect.getsource(ScrollableSidebar._on_pointer_activity)

        self.assertIn('"<Enter>"', source)
        self.assertIn('"<Leave>"', source)
        self.assertNotIn('"<Motion>"', source)
        self.assertIn("self._on_pointer_activity", source)
        self.assertIn("self._show_scrollbar", pointer_source)
        self.assertIn("self._hide_scrollbar", source)
        self.assertIn("self._scrollbar_visible = False", source)
        self.assertNotIn("self.scrollbar.pack(", source)

    def test_classic_interface_never_reveals_the_sidebar_scrollbar(self):
        sidebar = object.__new__(ScrollableSidebar)
        sidebar.scrollbar = mock.Mock()
        sidebar._scrollbar_visible = True
        sidebar._cancel_scrollbar_hide = mock.Mock()

        with mock.patch.object(ui_utils, "get_interface_mode", return_value="Classic"):
            sidebar._show_scrollbar()

        sidebar.scrollbar.pack_forget.assert_called_once_with()
        self.assertFalse(sidebar._scrollbar_visible)

    def test_modern_interface_can_reveal_the_sidebar_scrollbar(self):
        sidebar = object.__new__(ScrollableSidebar)
        sidebar.scrollbar = mock.Mock()
        sidebar._scrollbar_visible = False
        sidebar._cancel_scrollbar_hide = mock.Mock()

        with mock.patch.object(ui_utils, "get_interface_mode", return_value="Modern"):
            sidebar._show_scrollbar()

        sidebar.scrollbar.pack.assert_called_once_with(fill="both", expand=True)
        self.assertTrue(sidebar._scrollbar_visible)

    def test_leave_does_not_hide_during_transition_to_sidebar_child(self):
        sidebar = self._sidebar()
        sidebar.scrollbar = mock.Mock()
        sidebar._scrollbar_visible = True
        sidebar.after_idle = lambda callback: callback()

        sidebar._contains_scroll_area_pointer = lambda: True
        sidebar._hide_scrollbar()
        self.assertTrue(sidebar._scrollbar_visible)

        sidebar._contains_scroll_area_pointer = lambda: False
        sidebar._hide_scrollbar()
        self.assertFalse(sidebar._scrollbar_visible)

    def test_live_hidden_scrollbar_reserves_width_without_canvas_shift(self):
        try:
            root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display is unavailable: {exc}")

        try:
            root.withdraw()
            root.geometry("240x240")
            sidebar = ScrollableSidebar(root, width=240)
            sidebar.pack(fill="both", expand=True)
            tall_content = ctk.CTkFrame(sidebar.content, height=600)
            tall_content.pack(fill="x")
            tall_content.pack_propagate(False)
            root.update_idletasks()

            initial_canvas_width = sidebar.canvas.winfo_width()
            self.assertEqual(sidebar.scrollbar.winfo_manager(), "")
            self.assertEqual(
                sidebar._scrollbar_slot.winfo_width(),
                CONTROLS.scrollbar_width,
            )

            sidebar._show_scrollbar()
            root.update_idletasks()
            self.assertEqual(sidebar.scrollbar.winfo_manager(), "pack")
            self.assertEqual(sidebar.canvas.winfo_width(), initial_canvas_width)

            sidebar._contains_scroll_area_pointer = lambda: False
            sidebar._hide_scrollbar()
            root.update_idletasks()
            self.assertEqual(sidebar.scrollbar.winfo_manager(), "")
            self.assertEqual(sidebar.canvas.winfo_width(), initial_canvas_width)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
