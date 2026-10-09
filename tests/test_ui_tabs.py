from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk
from PIL import Image, ImageChops

from aidas.ui import ClosableTabView
from aidas.ui.tabs import _close_icon_image, _TabModel, _TabWidgets
from aidas.ui.theme import (
    CLASSIC_COLOR_PAIRS,
    COLOR_PAIRS,
    CONTROLS,
    MODERN_COLOR_PAIRS,
    resolve_color,
    set_interface_mode,
)


def _contrast_ratio(first: str, second: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _configured_options(widget: mock.Mock) -> dict:
    options = {}
    for call in widget.configure.call_args_list:
        options.update(call.kwargs)
    return options


class _Page:
    def __init__(self, widget_path: str) -> None:
        self.widget_path = widget_path

    def __str__(self) -> str:
        return self.widget_path


class UITabModelTests(unittest.TestCase):
    def test_close_icon_has_centered_symmetric_ink(self):
        alpha = _close_icon_image("#152235").getchannel("A")
        bounds = alpha.getbbox()
        self.assertIsNotNone(bounds)
        left, top, right, bottom = bounds
        self.assertEqual((left + right) / 2, alpha.width / 2)
        self.assertEqual((top + bottom) / 2, alpha.height / 2)
        for direction in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
            self.assertIsNone(ImageChops.difference(alpha, alpha.transpose(direction)).getbbox())

    def test_component_is_a_public_ctk_frame(self):
        self.assertTrue(issubclass(ClosableTabView, ctk.CTkFrame))

    def test_page_paths_are_stable_ids_independent_of_duplicate_labels(self):
        model = _TabModel()
        first = _Page(".tabs.page1")
        second = _Page(".tabs.page2")

        first_id = model.add(first, "Result")
        second_id = model.add(second, "Result")
        model.labels[first_id] = "Renamed result"

        self.assertEqual((first_id, second_id), (str(first), str(second)))
        self.assertEqual(model.order, [str(first), str(second)])
        self.assertEqual(model.labels[first_id], "Renamed result")
        self.assertEqual(model.labels[second_id], "Result")
        self.assertIs(model.pages[model.resolve(first_id)], first)
        self.assertIs(model.pages[model.resolve(second)], second)

    def test_forgetting_inactive_page_preserves_current_selection(self):
        model = _TabModel()
        first = _Page(".tabs.page1")
        second = _Page(".tabs.page2")
        first_id = model.add(first, "One")
        second_id = model.add(second, "Two")

        page, selected, was_selected = model.forget(second_id)

        self.assertIs(page, second)
        self.assertFalse(was_selected)
        self.assertEqual(selected, first_id)
        self.assertEqual(model.order, [first_id])

    def test_forgetting_active_page_silently_selects_nearby_page(self):
        model = _TabModel()
        first = _Page(".tabs.page1")
        second = _Page(".tabs.page2")
        third = _Page(".tabs.page3")
        first_id = model.add(first, "One")
        second_id = model.add(second, "Two")
        third_id = model.add(third, "Three")
        model.selected = second_id

        _, selected, was_selected = model.forget(second_id)

        self.assertTrue(was_selected)
        self.assertEqual(selected, third_id)
        self.assertEqual(model.order, [first_id, third_id])

        model.forget(third_id)
        model.forget(first_id)
        self.assertIsNone(model.selected)
        self.assertEqual(model.order, [])


class UITabContrastTests(unittest.TestCase):
    def test_tab_titles_and_close_glyphs_remain_readable_in_every_palette(self):
        states = (
            ("selected title", "tab_selected_text", "tab_selected"),
            ("selected title hover", "tab_selected_text", "tab_selected_hover"),
            ("inactive title", "text", "button"),
            ("inactive title hover", "text", "button_hover"),
            ("selected close glyph", "text", "tab_selected"),
            ("selected close glyph on whole-tab hover", "text", "tab_selected_hover"),
            ("inactive close glyph", "text", "button"),
            ("inactive close glyph on whole-tab hover", "text", "button_hover"),
            ("close glyph hover", "danger", "danger_soft"),
        )
        for interface, palette in (("Modern", MODERN_COLOR_PAIRS), ("Classic", CLASSIC_COLOR_PAIRS)):
            for appearance in ("Light", "Dark"):
                for state, foreground, background in states:
                    with self.subTest(interface=interface, appearance=appearance, state=state):
                        self.assertGreaterEqual(
                            _contrast_ratio(
                                resolve_color(palette[foreground], appearance),
                                resolve_color(palette[background], appearance),
                            ),
                            4.5,
                        )

    def test_tab_outlines_keep_subtle_inactive_and_distinct_selected_borders(self):
        for interface, palette in (("Modern", MODERN_COLOR_PAIRS), ("Classic", CLASSIC_COLOR_PAIRS)):
            with self.subTest(interface=interface):
                self.assertEqual(palette["tab_border"], palette["border"])
            for appearance in ("Light", "Dark"):
                for background in ("tab_selected", "tab_selected_hover", "sidebar"):
                    with self.subTest(interface=interface, appearance=appearance, background=background):
                        self.assertGreaterEqual(
                            _contrast_ratio(
                                resolve_color(palette["tab_selected_border"], appearance),
                                resolve_color(palette[background], appearance),
                            ),
                            3.0,
                        )


class UITabStyleTests(unittest.TestCase):
    def setUp(self):
        set_interface_mode("Modern", redraw=False)
        self.view = ClosableTabView.__new__(ClosableTabView)
        self.view._model = _TabModel()
        self.view._tab_widgets = {}
        self.view._header_canvas = mock.Mock()
        self.view._header = mock.Mock()
        self.view._header_scrollbar = mock.Mock()
        self.view._header_overflow = False
        self.view._header_hovered = False
        self.view._header_hover_after_id = None
        self.view._scrollbar_visible = False
        self.view._pointer_within = mock.Mock(return_value=False)
        self.hover_callbacks = []

        def queue_hover_check(callback):
            self.hover_callbacks.append(callback)
            return f"hover-check-{len(self.hover_callbacks)}"

        self.view.after_idle = mock.Mock(side_effect=queue_hover_check)
        self.view.after_cancel = mock.Mock()
        self.view._tab_font = mock.Mock()
        self.view._selected_tab_font = mock.Mock()
        self.view._close_images = {name: mock.Mock() for name in ("text", "danger")}
        self.view._draw_content_join = mock.Mock()
        for label in ("One", "Two"):
            page = _Page(f".tabs.{label.lower()}")
            page_id = self.view._model.add(page, label)
            self.view._tab_widgets[page_id] = _TabWidgets(
                mock.Mock(), mock.Mock(), mock.Mock(),
                bottom_join=mock.Mock(),
            )
        self.first_id, self.second_id = self.view._model.order

    def tearDown(self):
        set_interface_mode("Modern", redraw=False)

    def test_selected_and_inactive_tabs_use_distinct_readable_states(self):
        for active, fill, foreground, hover, border in (
            (True, "tab_selected", "tab_selected_text", "tab_selected_hover", "tab_selected_border"),
            (False, "button", "text", "button_hover", "tab_border"),
        ):
            with self.subTest(active=active):
                widgets = self.view._tab_widgets[self.first_id]
                self.view._style_tab(self.first_id, active=active)
                frame = _configured_options(widgets.frame)
                title = _configured_options(widgets.title)
                close = _configured_options(widgets.close)

                self.assertIs(frame["fg_color"], COLOR_PAIRS[fill])
                self.assertIs(frame["border_color"], COLOR_PAIRS[border])
                self.assertEqual(frame["border_width"], 1)
                self.assertIs(title["fg_color"], COLOR_PAIRS[fill])
                self.assertIs(title["text_color"], COLOR_PAIRS[foreground])
                self.assertIs(title["hover_color"], COLOR_PAIRS[hover])
                self.assertIs(title["font"], self.view._selected_tab_font if active else self.view._tab_font)
                self.assertIs(close["fg_color"], COLOR_PAIRS[fill])
                self.assertIs(close["text_color"], COLOR_PAIRS["text"])
                self.assertIs(close["image"], self.view._close_images["text"])
                self.assertIs(close["hover_color"], COLOR_PAIRS["danger_soft"])
                self.assertEqual(widgets.active, active)
                if active:
                    widgets.bottom_join.place.assert_called()
                else:
                    widgets.bottom_join.place_forget.assert_called()

    def test_close_hover_restores_its_normal_icon_and_tab_fill_on_leave(self):
        widgets = self.view._tab_widgets[self.first_id]
        self.view._style_tab(self.first_id, active=True)
        self.view._set_tab_hover(self.first_id, True)

        self.view._set_close_hover(self.first_id, True)

        self.assertTrue(widgets.close_hovered)
        hovered = _configured_options(widgets.close)
        self.assertIs(hovered["fg_color"], COLOR_PAIRS["danger_soft"])
        self.assertIs(hovered["text_color"], COLOR_PAIRS["danger"])
        self.assertIs(hovered["image"], self.view._close_images["danger"])

        self.view._set_close_hover(self.first_id, False)

        self.assertFalse(widgets.close_hovered)
        normal = _configured_options(widgets.close)
        self.assertIs(normal["fg_color"], COLOR_PAIRS["tab_selected_hover"])
        self.assertIs(normal["text_color"], COLOR_PAIRS["text"])
        self.assertIs(normal["image"], self.view._close_images["text"])
        self.assertEqual(self.view._model.selected, self.first_id)

    def test_hover_brightens_the_whole_tab_without_changing_selection(self):
        for active, hover_fill, normal_fill in (
            (True, "tab_selected_hover", "tab_selected"),
            (False, "button_hover", "button"),
        ):
            with self.subTest(active=active):
                widgets = self.view._tab_widgets[self.first_id]
                self.view._model.selected = self.first_id if active else self.second_id
                selected = self.view.select()
                self.view._style_tab(self.first_id, active=active)
                self.view._set_tab_hover(self.first_id, True)

                for target in (widgets.frame, widgets.title, widgets.close, widgets.bottom_join):
                    self.assertIs(_configured_options(target)["fg_color"], COLOR_PAIRS[hover_fill])
                self.assertIs(_configured_options(widgets.bottom_join)["bg_color"], COLOR_PAIRS[hover_fill])
                self.assertEqual(self.view.select(), selected)

                self.view._set_tab_hover(self.first_id, False)
                for target in (widgets.frame, widgets.title, widgets.close, widgets.bottom_join):
                    self.assertIs(_configured_options(target)["fg_color"], COLOR_PAIRS[normal_fill])

    def test_crossing_title_close_and_footer_keeps_the_tab_highlighted(self):
        widgets = self.view._tab_widgets[self.first_id]
        self.view._style_tab(self.first_id, active=True)
        self.view._update_header_overflow(True)
        self.view._on_tab_enter(self.first_id)
        widgets.frame.configure.reset_mock()

        self.view._on_tab_leave(self.first_id)
        tab_check = widgets.hover_after_id
        header_check = self.view._header_hover_after_id
        self.view._on_tab_enter(self.first_id, is_close=True)

        self.view.after_cancel.assert_any_call(tab_check)
        self.view.after_cancel.assert_any_call(header_check)
        self.assertTrue(widgets.hovered)
        self.assertTrue(widgets.close_hovered)
        self.assertTrue(self.view._scrollbar_visible)
        widgets.frame.configure.assert_not_called()

        self.view._on_tab_leave(self.first_id, is_close=True)
        self.view._pointer_within.return_value = True
        self.view._finish_tab_leave(self.first_id)
        self.view._finish_header_leave()
        self.assertTrue(widgets.hovered)
        self.assertFalse(widgets.close_hovered)
        self.assertTrue(self.view._scrollbar_visible)
        self.assertIs(_configured_options(widgets.close)["fg_color"], COLOR_PAIRS["tab_selected_hover"])

        self.view._pointer_within.return_value = False
        self.view._finish_tab_leave(self.first_id)
        self.view._finish_header_leave()
        self.assertFalse(widgets.hovered)
        self.assertFalse(self.view._scrollbar_visible)
        self.assertIs(_configured_options(widgets.frame)["fg_color"], COLOR_PAIRS["tab_selected"])

    def test_scrollbar_reveals_on_header_hover_without_resetting_hidden_scroll(self):
        self.view._update_header_overflow(True)
        self.assertFalse(self.view._scrollbar_visible)
        self.view._header_scrollbar.grid.assert_not_called()

        self.view._on_header_enter()
        self.assertTrue(self.view._scrollbar_visible)
        self.view._header_scrollbar.grid.assert_called_once_with()
        self.view._on_header_leave()
        header_check = self.view._header_hover_after_id
        self.view._on_header_enter()
        self.view.after_cancel.assert_any_call(header_check)
        self.assertTrue(self.view._scrollbar_visible)
        self.view._on_header_leave()
        self.view._pointer_within.return_value = True
        self.view._finish_header_leave()
        self.assertTrue(self.view._scrollbar_visible)

        self.view._pointer_within.return_value = False
        self.view._finish_header_leave()
        self.assertFalse(self.view._scrollbar_visible)
        self.view._header_scrollbar.grid_remove.assert_called_once_with()
        self.view._header_canvas.xview_moveto.assert_not_called()
        self.assertEqual(self.view._on_mousewheel(SimpleNamespace(delta=-120)), "break")
        self.view._header_canvas.xview_scroll.assert_called_once_with(CONTROLS.gap * 3, "units")

        self.view._on_header_enter()
        self.assertTrue(self.view._scrollbar_visible)
        self.view._header_canvas.xview_moveto.assert_not_called()
        self.view._update_header_overflow(False)
        self.assertFalse(self.view._scrollbar_visible)
        self.view._header_canvas.xview_moveto.assert_called_once_with(0.0)
        self.view._on_header_enter()
        self.assertFalse(self.view._scrollbar_visible)

    def test_restyling_a_hovered_close_button_keeps_the_hover_glyph_visible(self):
        widgets = self.view._tab_widgets[self.first_id]
        self.view._set_close_hover(self.first_id, True)

        self.view._style_tab(self.first_id, active=False)

        self.assertTrue(widgets.close_hovered)
        close = _configured_options(widgets.close)
        self.assertIs(close["fg_color"], COLOR_PAIRS["danger_soft"])
        self.assertIs(close["text_color"], COLOR_PAIRS["danger"])
        self.assertIs(close["image"], self.view._close_images["danger"])

    def test_theme_refresh_restyles_all_tabs_without_changing_selection_or_hover(self):
        self.view._model.selected = self.second_id
        self.view._set_tab_hover(self.first_id, True)
        self.view._set_close_hover(self.first_id, True)
        original_order = self.view.tabs()
        original_pages = dict(self.view._model.pages)

        for interface in ("Classic", "Modern"):
            with self.subTest(interface=interface):
                set_interface_mode(interface, redraw=False)
                with (
                    mock.patch("aidas.ui.theme.ctk.get_appearance_mode", return_value="Dark"),
                    mock.patch.object(self.view, "_style_tab", wraps=self.view._style_tab) as style_tab,
                ):
                    self.view._apply_aidas_theme()

                self.assertCountEqual(
                    style_tab.call_args_list,
                    [mock.call(self.first_id, active=False), mock.call(self.second_id, active=True)],
                )
                self.assertEqual(self.view.select(), self.second_id)
                self.assertEqual(self.view.tabs(), original_order)
                self.assertEqual(self.view._model.pages, original_pages)
                self.assertTrue(self.view._tab_widgets[self.first_id].close_hovered)
                self.assertTrue(self.view._tab_widgets[self.first_id].hovered)
                self.assertIs(
                    _configured_options(self.view._tab_widgets[self.first_id].frame)["fg_color"],
                    COLOR_PAIRS["button_hover"],
                )
                self.assertIs(
                    _configured_options(self.view._tab_widgets[self.first_id].close)["text_color"],
                    COLOR_PAIRS["danger"],
                )
                self.view._header_canvas.configure.assert_called_with(
                    background=resolve_color(COLOR_PAIRS["sidebar"], "Dark"),
                )
                for name, icon in self.view._close_images.items():
                    for index, source in enumerate(("light_image", "dark_image")):
                        image = icon.configure.call_args.kwargs[source]
                        expected_color = tuple(int(COLOR_PAIRS[name][index][offset:offset + 2], 16)
                                               for offset in (1, 3, 5))
                        self.assertEqual(image.getpixel((image.width // 2, image.height // 2))[:3],
                                         expected_color)


class UITabGeometryTests(unittest.TestCase):
    def test_long_titles_keep_centered_close_and_joined_panel_after_renaming_and_overflow(self):
        root = None
        try:
            try:
                root = ctk.CTk()
                root.withdraw()
                root.attributes("-alpha", 0)
                root.geometry("640x180+0+0")
            except tk.TclError as exc:
                self.skipTest(f"A transparent Tk test window is unavailable: {exc}")

            view = ClosableTabView(root)
            view.pack(fill="both", expand=True)
            initial_title = "Longer image folder name"
            page = view.add(text=initial_title)
            root.deiconify()

            def assert_close_and_panel_geometry():
                root.update_idletasks()
                widgets = view._tab_widgets[view.select()]
                selected_fill = COLOR_PAIRS["tab_selected_hover" if widgets.hovered else "tab_selected"]
                selected_page = view.nametowidget(view.select())
                frame_left = widgets.frame.winfo_rootx()
                frame_top = widgets.frame.winfo_rooty()
                frame_right = frame_left + widgets.frame.winfo_width()
                frame_bottom = frame_top + widgets.frame.winfo_height()
                close_left = widgets.close.winfo_rootx()
                close_top = widgets.close.winfo_rooty()
                close_width = widgets.close.winfo_width()
                close_height = widgets.close.winfo_height()
                title_right = widgets.title.winfo_rootx() + widgets.title.winfo_width()

                self.assertTrue(widgets.close.winfo_ismapped())
                self.assertGreater(close_width, 1)
                self.assertGreater(close_height, 1)
                self.assertGreaterEqual(close_width, widgets.close.winfo_reqwidth())
                self.assertGreaterEqual(close_height, widgets.close.winfo_reqheight())
                self.assertGreaterEqual(close_left, title_right)
                self.assertGreaterEqual(close_top, frame_top)
                self.assertLessEqual(close_left + close_width, frame_right)
                self.assertLessEqual(close_top + close_height, frame_bottom)
                self.assertEqual(widgets.close.cget("text"), "")
                self.assertEqual(widgets.close.cget("border_width"), 0)
                labels = [child for child in widgets.close.winfo_children()
                          if isinstance(child, tk.Label) and child.cget("image")]
                self.assertEqual(len(labels), 1)
                image_label = labels[0]
                self.assertAlmostEqual(image_label.winfo_x() + image_label.winfo_width() / 2,
                                       close_width / 2, delta=1)
                self.assertAlmostEqual(image_label.winfo_y() + image_label.winfo_height() / 2,
                                       close_height / 2, delta=1)
                self.assertAlmostEqual(close_top + close_height / 2,
                                       frame_top + widgets.frame.winfo_height() / 2, delta=1)
                self.assertEqual(widgets.frame.cget("fg_color"), selected_fill)
                self.assertEqual(widgets.title.cget("fg_color"), selected_fill)
                self.assertFalse(widgets.title.cget("hover"))
                self.assertEqual(widgets.close.cget("fg_color"),
                                 COLOR_PAIRS["danger_soft"] if widgets.close_hovered else selected_fill)
                self.assertEqual(widgets.frame.cget("border_color"), COLOR_PAIRS["tab_selected_border"])
                self.assertEqual(widgets.frame.cget("border_width"), 1)
                self.assertEqual(selected_page.cget("fg_color"), COLOR_PAIRS["surface"])
                self.assertEqual(widgets.frame.cget("corner_radius"), 8)
                self.assertEqual(widgets.title.cget("text_color"), COLOR_PAIRS["tab_selected_text"])
                self.assertEqual(widgets.title.cget("font").cget("size"), 13)
                self.assertEqual(widgets.title.cget("font").cget("weight"), "bold")
                self.assertTrue(widgets.bottom_join.winfo_ismapped())
                self.assertEqual(widgets.bottom_join.cget("fg_color"), selected_fill)
                self.assertEqual(widgets.bottom_join.cget("bg_color"), selected_fill)
                self.assertEqual(frame_bottom, view._join_frame.winfo_rooty())
                self.assertEqual(view._join_frame.winfo_rooty() + view._join_frame.winfo_height(),
                                 view._content.winfo_rooty())
                self.assertEqual(selected_page.winfo_rooty(), view._content.winfo_rooty())
                self.assertGreater(selected_page.winfo_rootx(), view._content.winfo_rootx())
                self.assertLess(selected_page.winfo_rootx() + selected_page.winfo_width(),
                                view._content.winfo_rootx() + view._content.winfo_width())
                joins = view._join_canvas.find_withtag("selected_join")
                self.assertEqual(len(joins), 1)
                left, _, right, _ = view._join_canvas.coords(joins[0])
                self.assertAlmostEqual(left, max(0, frame_left - view._join_canvas.winfo_rootx() + 1),
                                       delta=1)
                self.assertAlmostEqual(right, min(view._join_canvas.winfo_width(),
                                                 frame_right - view._join_canvas.winfo_rootx() - 1),
                                       delta=1)
                self.assertEqual(view._join_canvas.itemcget(joins[0], "fill"),
                                 resolve_color(selected_fill))
                if view._scrollbar_visible:
                    self.assertTrue(view._header_scrollbar.winfo_ismapped())
                    rail_height = view._header_scrollbar.winfo_height()
                    thumb_height = rail_height - 2 * view._header_scrollbar.cget("border_spacing")
                    self.assertGreaterEqual(thumb_height, rail_height / 2)
                    self.assertGreaterEqual(view._header_scrollbar.winfo_rooty(),
                                            view._header.winfo_rooty())
                    self.assertLessEqual(view._header_scrollbar.winfo_rooty() + view._header_scrollbar.winfo_height(),
                                         view._header_canvas.winfo_rooty())
                if view._header_overflow:
                    self.assertGreaterEqual(close_left, view._header_canvas.winfo_rootx())
                    self.assertLessEqual(close_left + close_width,
                                         view._header_canvas.winfo_rootx() + view._header_canvas.winfo_width())

            assert_close_and_panel_geometry()
            title_labels = [
                child for child in view._tab_widgets[view.select()].title.winfo_children()
                if isinstance(child, tk.Label) and child.cget("text") == initial_title
            ]
            self.assertEqual(len(title_labels), 1)
            self.assertGreaterEqual(title_labels[0].winfo_width(), title_labels[0].winfo_reqwidth())
            view.tab(page, text="Renamed tab with a much longer result path " * 50)
            assert_close_and_panel_geometry()
            extra_pages = [view.add(text=f"Additional result {index}") for index in range(8)]
            assert_close_and_panel_geometry()
            self.assertTrue(view._header_overflow)
            self.assertFalse(view._scrollbar_visible)
            panel_geometry = (view._content.winfo_rooty(), view._content.winfo_height())
            view._on_header_enter()
            assert_close_and_panel_geometry()
            self.assertTrue(view._scrollbar_visible)
            self.assertEqual((view._content.winfo_rooty(), view._content.winfo_height()), panel_geometry)
            view.select(extra_pages[-1])
            assert_close_and_panel_geometry()
            scroll_position = view._header_canvas.xview()
            with mock.patch.object(view, "_pointer_within", return_value=False):
                view._finish_header_leave()
            root.update_idletasks()
            self.assertFalse(view._scrollbar_visible)
            self.assertEqual(view._header_canvas.xview(), scroll_position)
            self.assertEqual((view._content.winfo_rooty(), view._content.winfo_height()), panel_geometry)
            view._on_header_enter()
            view._set_tab_hover(view.select(), True)
            assert_close_and_panel_geometry()
            self.assertEqual(view._header_canvas.xview(), scroll_position)
            root.geometry("240x180+0+0")
            view.select(page)
            assert_close_and_panel_geometry()
            view._on_tab_leave(view.select())
            pending_tab = view._tab_widgets[view.select()].hover_after_id
            self.assertIsNotNone(pending_tab)
            with mock.patch.object(view, "after_cancel", wraps=view.after_cancel) as cancel:
                view.forget(page)
            cancel.assert_any_call(pending_tab)
            root.update_idletasks()

            view._on_tab_leave(view.select())
            pending_tab = view._tab_widgets[view.select()].hover_after_id
            pending_header = view._header_hover_after_id
            self.assertIsNotNone(pending_tab)
            self.assertIsNotNone(pending_header)
            with mock.patch.object(view, "after_cancel", wraps=view.after_cancel) as cancel:
                view.destroy()
            cancel.assert_any_call(pending_tab)
            cancel.assert_any_call(pending_header)
            root.update_idletasks()
        finally:
            if root is not None:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
