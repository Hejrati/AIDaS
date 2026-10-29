"""Guard UI work reduction without changing content, colors, or event order."""

from __future__ import annotations

import queue
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.ui.compiler import CompilerPresentationMixin
from aidas.ui.components import AppButton, ProgressCircle, WorkflowHeader
from aidas.ui.tabs import ClosableTabView, _TabModel, _TabWidgets
from aidas.ui.theme import COLOR_PAIRS, resolve_color
from tests.support.paths import PROJECT_ROOT


class CompilerBurstTests(unittest.TestCase):
    def _view(self):
        view = CompilerPresentationMixin()
        view._events = queue.Queue()
        view._append_log = mock.Mock()
        view._update_progress = mock.Mock()
        view._compilation_succeeded = mock.Mock()
        view._compilation_failed = mock.Mock()
        return view

    def test_contiguous_logs_share_one_widget_write_without_crossing_events(self):
        view = self._view()
        result = object()
        for event in (
            ("log", "First"), ("log", "Second\ncontinued"),
            ("progress", (1, 2, "Working")), ("log", "Last"),
            ("success", result),
        ):
            view._events.put(event)
        observer = mock.Mock()
        observer.attach_mock(view._append_log, "log")
        observer.attach_mock(view._update_progress, "progress")
        observer.attach_mock(view._compilation_succeeded, "success")

        view._drain_compile_events()

        self.assertEqual(observer.mock_calls, [
            mock.call.log("First\nSecond\ncontinued"),
            mock.call.progress((1, 2, "Working")),
            mock.call.log("Last"), mock.call.success(result),
        ])
        self.assertTrue(view._events.empty())

    def test_large_backlog_yields_with_remaining_events_in_order(self):
        view = self._view()
        view.MAX_COMPILE_EVENTS_PER_POLL = 3
        for index in range(8):
            view._events.put(("log", str(index)))

        view._drain_compile_events()
        self.assertEqual(view._events.qsize(), 5)
        view._append_log.assert_called_once_with("0\n1\n2")
        view._drain_compile_events()
        self.assertEqual(view._events.qsize(), 2)
        view._drain_compile_events()
        self.assertEqual(view._append_log.call_args_list, [
            mock.call("0\n1\n2"), mock.call("3\n4\n5"), mock.call("6\n7"),
        ])

    def test_newly_published_events_wait_for_next_poll(self):
        view = self._view()
        view._events.put(("progress", (1, 2, "Working")))
        view._update_progress.side_effect = lambda _payload: view._events.put(("log", "Later"))

        view._drain_compile_events()

        view._append_log.assert_not_called()
        self.assertEqual(view._events.qsize(), 1)
        view._drain_compile_events()
        view._append_log.assert_called_once_with("Later")

    def test_logs_are_flushed_before_failure_handler(self):
        view = self._view()
        error = OSError("Disk full")
        for event in (("log", "One"), ("log", "Two"), ("error", error)):
            view._events.put(event)
        view._compilation_failed.side_effect = lambda _error: view._append_log.assert_called_once_with("One\nTwo")

        view._drain_compile_events()

        view._compilation_failed.assert_called_once_with(error)


class ButtonImageUpdateTests(unittest.TestCase):
    def _button(self):
        button = object.__new__(AppButton)
        icon = object()
        button._state = "normal"
        button._enabled_palette = {"text_color": COLOR_PAIRS["text"]}
        button._aidas_action_icon_source = icon
        button._aidas_tinted_action_icon = None
        button._image = icon
        button._action_icon_for_state = mock.Mock(return_value=icon)
        return button, icon

    def test_text_updates_do_not_rebind_an_unchanged_icon(self):
        button, icon = self._button()
        with mock.patch.object(ctk.CTkButton, "configure") as configure:
            button.configure(text="Updated")
        configure.assert_called_once_with(False, text="Updated")
        self.assertIs(button._aidas_action_icon_source, icon)

    def test_replacement_icon_still_reaches_the_button(self):
        button, _icon = self._button()
        replacement = object()
        button._action_icon_for_state.return_value = replacement
        with mock.patch.object(ctk.CTkButton, "configure") as configure:
            button.configure(image=replacement)
        configure.assert_called_once_with(False, image=replacement)
        self.assertIs(button._aidas_action_icon_source, replacement)

    def test_clearing_an_existing_icon_still_reaches_the_button(self):
        button, _icon = self._button()
        button._action_icon_for_state.return_value = None
        with mock.patch.object(ctk.CTkButton, "configure") as configure:
            button.configure(image=None)
        configure.assert_called_once_with(False, image=None)
        self.assertIsNone(button._aidas_action_icon_source)


class ProgressBadgeUpdateTests(unittest.TestCase):
    def _circle(self):
        circle = object.__new__(ProgressCircle)
        circle.master = SimpleNamespace()
        circle.bg_color_ref = ("#eeeeee", "#111111")
        circle._fill_color_ref = ("#abcdef", "#123456")
        circle._text_color_ref = ("#000000", "#ffffff")
        circle._state_text = "2"
        circle.oval_id, circle.text_id = 1, 2
        circle.configure = mock.Mock()
        circle.itemconfig = mock.Mock()
        return circle

    def test_repeated_state_skips_native_canvas_mutations(self):
        circle = self._circle()
        with mock.patch("aidas.ui.components.ctk.get_appearance_mode", return_value="Dark"):
            circle._apply_aidas_theme()
            circle.configure.reset_mock()
            circle.itemconfig.reset_mock()
            for _ in range(20):
                circle.update_state(circle._fill_color_ref, circle._text_color_ref, "2")
        circle.configure.assert_not_called()
        circle.itemconfig.assert_not_called()

    def test_resolved_colors_refresh_after_appearance_changes(self):
        circle = self._circle()
        for mode, fill, foreground, background in (
            ("Dark", "#123456", "#ffffff", "#111111"),
            ("Light", "#abcdef", "#000000", "#eeeeee"),
            ("Dark", "#123456", "#ffffff", "#111111"),
        ):
            with self.subTest(mode=mode), mock.patch(
                "aidas.ui.components.ctk.get_appearance_mode", return_value=mode,
            ):
                circle.configure.reset_mock()
                circle.itemconfig.reset_mock()
                circle._apply_aidas_theme()
                circle.configure.assert_called_once_with(bg=background)
                circle.itemconfig.assert_has_calls([
                    mock.call(1, fill=fill, outline=fill),
                    mock.call(2, fill=foreground, text="2"),
                ])

    def test_text_and_completion_changes_still_repaint(self):
        circle = self._circle()
        with mock.patch("aidas.ui.components.ctk.get_appearance_mode", return_value="Dark"):
            circle._apply_aidas_theme()
            circle.itemconfig.reset_mock()
            circle.update_state("#00aa00", "#ffffff", "\u2713")
        circle.itemconfig.assert_has_calls([
            mock.call(1, fill="#00aa00", outline="#00aa00"),
            mock.call(2, fill="#ffffff", text="\u2713"),
        ])


class TabCanvasUpdateTests(unittest.TestCase):
    def _view(self):
        view = object.__new__(ClosableTabView)
        view._model = _TabModel()
        page_id = view._model.add(".page", "Page")
        widgets = _TabWidgets(mock.Mock(), mock.Mock(), mock.Mock(), active=True, bottom_join=mock.Mock())
        view._tab_widgets = {page_id: widgets}
        widgets.frame.winfo_rootx.return_value = 100
        widgets.frame.winfo_width.return_value = 150
        view._join_frame = mock.Mock()
        view._join_canvas = mock.Mock()
        view._join_canvas.winfo_rootx.return_value = 100
        view._join_canvas.winfo_width.return_value = 500
        view._join_canvas.winfo_height.return_value = 1
        view._join_canvas.create_rectangle.return_value = 42
        view._get_widget_scaling = mock.Mock(return_value=1.0)
        return view, page_id, widgets

    def test_selected_join_reuses_its_canvas_item(self):
        view, _page_id, widgets = self._view()
        view._draw_content_join()
        for width in (175, 150, 180):
            widgets.frame.winfo_width.return_value = width
            view._draw_content_join()
        view._join_canvas.create_rectangle.assert_called_once()
        view._join_canvas.delete.assert_not_called()

    def test_identical_join_geometry_and_palette_skip_all_canvas_writes(self):
        view, _page_id, _widgets = self._view()
        view._draw_content_join()
        view._join_frame.reset_mock()
        view._join_canvas.reset_mock()
        for _ in range(20):
            view._draw_content_join()
        view._join_frame.configure.assert_not_called()
        for name in ("configure", "coords", "itemconfigure", "create_rectangle", "delete"):
            getattr(view._join_canvas, name).assert_not_called()

    def test_join_hover_fill_changes_without_recreating_the_item(self):
        view, _page_id, widgets = self._view()
        view._draw_content_join()
        widgets.hovered = True
        view._draw_content_join()
        view._join_canvas.create_rectangle.assert_called_once()
        view._join_canvas.itemconfigure.assert_called_with(
            42, fill=resolve_color(COLOR_PAIRS["tab_selected_hover"]), state="normal",
        )

    def test_join_hides_with_no_selection_and_reappears_without_recreation(self):
        view, page_id, _widgets = self._view()
        view._draw_content_join()
        view._model.selected = None
        view._draw_content_join()
        view._join_canvas.itemconfigure.assert_called_with(42, state="hidden")
        view._model.selected = page_id
        view._draw_content_join()
        view._join_canvas.itemconfigure.assert_called_with(
            42, fill=resolve_color(COLOR_PAIRS["tab_selected"]), state="normal",
        )
        view._join_canvas.create_rectangle.assert_called_once()

    def test_repeated_tab_border_geometry_does_not_reconfigure_widgets(self):
        view, page_id, widgets = self._view()
        view._draw_content_join = mock.Mock()
        view._update_tab_border_geometry(page_id)
        widgets.frame.reset_mock()
        widgets.bottom_join.reset_mock()
        for _ in range(20):
            view._update_tab_border_geometry(page_id)
        widgets.frame.configure.assert_not_called()
        widgets.bottom_join.configure.assert_not_called()
        widgets.bottom_join.place.assert_not_called()

    def test_tab_border_cache_detects_live_token_pairs_mutating_in_place(self):
        view, page_id, widgets = self._view()
        view._draw_content_join = mock.Mock()
        sidebar = ["#111111", "#222222"]
        lower_corner = ["#333333", "#444444"]
        with mock.patch("aidas.ui.tabs.COLOR_PAIRS", {
            "sidebar": sidebar, "tab_selected_border": lower_corner,
        }):
            view._update_tab_border_geometry(page_id)
            previous = widgets.border_signature
            widgets.frame.configure.reset_mock()
            sidebar[:] = ["#555555", "#666666"]
            lower_corner[:] = ["#777777", "#888888"]

            view._update_tab_border_geometry(page_id)

            self.assertNotEqual(widgets.border_signature, previous)
            self.assertTrue(all(isinstance(value, str) for value in previous[1:]))
            widgets.frame.configure.assert_called_once_with(
                border_width=1.0,
                background_corner_colors=(sidebar, sidebar, lower_corner, lower_corner),
            )

    def test_join_border_cache_stores_immutable_resolved_color(self):
        view, _page_id, _widgets = self._view()
        border = ["#111111", "#222222"]
        with mock.patch("aidas.ui.tabs.COLOR_PAIRS", {
            "tab_selected_border": border, "tab_selected": ["#333333", "#444444"],
        }):
            view._draw_content_join()
            previous = view._join_border_signature
            view._join_frame.configure.reset_mock()
            border[:] = ["#555555", "#666666"]

            view._draw_content_join()

            self.assertNotEqual(view._join_border_signature, previous)
            self.assertIsInstance(previous[1], str)
            view._join_frame.configure.assert_called_once_with(height=1.0, fg_color=border)


class HeaderAssetConstructionTests(unittest.TestCase):
    def test_supplied_assets_do_not_render_discarded_fallbacks(self):
        try:
            root = ctk.CTk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.addCleanup(root.destroy)
        root.withdraw()
        from aidas.ui.components import _header_action_icon

        with mock.patch("aidas.ui.components._header_action_icon", wraps=_header_action_icon) as draw:
            header = WorkflowHeader(
                root, version="test", on_step_selected=lambda _index: None,
                settings_icon_path=str(PROJECT_ROOT / "assets" / "iconify-fluent-color--settings-32.png"),
                help_icon_path=str(PROJECT_ROOT / "assets" / "iconify-fluent-color--question-circle-32.png"),
            )
        self.assertEqual([call.args[0] for call in draw.call_args_list], ["feedback", "feedback"])
        self.assertEqual(header.settings_image.cget("light_image").size, (256, 256))
        self.assertEqual(header.help_image.cget("light_image").size, (256, 256))


if __name__ == "__main__":
    unittest.main()
