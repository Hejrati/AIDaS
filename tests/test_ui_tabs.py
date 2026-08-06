from __future__ import annotations

import unittest

import customtkinter as ctk

from aidas.ui import ClosableTabView
from aidas.ui.tabs import _CLOSE_GLYPH, _TabModel


class _Page:
    def __init__(self, widget_path: str) -> None:
        self.widget_path = widget_path

    def __str__(self) -> str:
        return self.widget_path


class UITabModelTests(unittest.TestCase):
    def test_close_control_uses_the_multiplication_glyph(self):
        self.assertEqual(_CLOSE_GLYPH, "\u00d7")
        self.assertNotEqual(_CLOSE_GLYPH, "x")

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


if __name__ == "__main__":
    unittest.main()
