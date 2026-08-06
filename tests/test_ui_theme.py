from __future__ import annotations

import unittest

import customtkinter as ctk

from aidas.ui.theme import (
    APPEARANCE_MODES,
    COLORS,
    COLOR_PAIRS,
    CONTROLS,
    SHAPES,
    apply_appearance_mode,
    normalize_appearance_mode,
    resolve_color,
)


class UIThemeTests(unittest.TestCase):
    def tearDown(self):
        ctk.set_appearance_mode("System")

    def test_supported_appearance_modes_are_user_facing(self):
        self.assertEqual(APPEARANCE_MODES, ("System", "Light", "Dark"))

    def test_legacy_ttk_theme_names_migrate_safely_to_system(self):
        for legacy_value in (None, "", "clam", "vista", "xpnative", "unknown"):
            with self.subTest(value=legacy_value):
                self.assertEqual(normalize_appearance_mode(legacy_value), "System")

    def test_mode_normalization_is_case_insensitive(self):
        self.assertEqual(normalize_appearance_mode("LIGHT"), "Light")
        self.assertEqual(normalize_appearance_mode(" dark "), "Dark")
        self.assertEqual(normalize_appearance_mode("system"), "System")

    def test_dynamic_compatibility_colors_follow_ctk_appearance(self):
        ctk.set_appearance_mode("Light")
        self.assertEqual(COLORS.surface, COLOR_PAIRS["surface"][0])
        ctk.set_appearance_mode("Dark")
        self.assertEqual(COLORS.surface, COLOR_PAIRS["surface"][1])

    def test_explicit_pair_resolution_and_application(self):
        pair = ("#010101", "#fefefe")
        self.assertEqual(resolve_color(pair, "Light"), pair[0])
        self.assertEqual(resolve_color(pair, "Dark"), pair[1])
        self.assertEqual(apply_appearance_mode("dark"), "Dark")

    def test_control_and_shape_tokens_are_usable(self):
        self.assertGreaterEqual(CONTROLS.height_md, CONTROLS.height_sm)
        self.assertGreater(SHAPES.corner_radius_md, SHAPES.corner_radius_sm)
        self.assertIn("primary", COLOR_PAIRS)
        self.assertIn("danger", COLOR_PAIRS)
        self.assertNotEqual(COLOR_PAIRS["window_chrome"][0], "#FFFFFF")
        self.assertNotEqual(COLOR_PAIRS["window_chrome"][1], "#000000")

    def test_title_and_menu_bars_remain_distinct_in_each_mode(self):
        for appearance_mode in ("Light", "Dark"):
            with self.subTest(appearance_mode=appearance_mode):
                self.assertNotEqual(
                    resolve_color(COLOR_PAIRS["window_chrome"], appearance_mode),
                    resolve_color(COLOR_PAIRS["menu_bar"], appearance_mode),
                )


if __name__ == "__main__":
    unittest.main()
