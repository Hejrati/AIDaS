from __future__ import annotations

import unittest
from unittest import mock

import customtkinter as ctk

from aidas.ui.theme import (
    APPEARANCE_MODES,
    CLASSIC_COLOR_PAIRS,
    COLORS,
    COLOR_PAIRS,
    CONTROLS,
    INTERFACE_MODES,
    MODERN_COLOR_PAIRS,
    SHAPES,
    apply_appearance_mode,
    get_interface_mode,
    normalize_appearance_mode,
    normalize_interface_mode,
    refresh_interface_widgets,
    resolve_color,
    set_interface_mode,
)


class UIThemeTests(unittest.TestCase):
    def tearDown(self):
        set_interface_mode("Modern")
        ctk.set_appearance_mode("System")

    def test_supported_appearance_modes_are_user_facing(self):
        self.assertEqual(APPEARANCE_MODES, ("System", "Light", "Dark"))

    def test_supported_interface_modes_are_separate_from_appearance(self):
        self.assertEqual(INTERFACE_MODES, ("Modern", "Classic"))

    def test_interface_mode_normalization_is_safe_and_case_insensitive(self):
        self.assertEqual(normalize_interface_mode(" classic "), "Classic")
        self.assertEqual(normalize_interface_mode("MODERN"), "Modern")
        for invalid in (None, "", "legacy", "dark"):
            with self.subTest(value=invalid):
                self.assertEqual(normalize_interface_mode(invalid), "Modern")

    def test_classic_mode_restores_v2_palette_and_keeps_canonical_shapes(self):
        original_color_keys = set(COLOR_PAIRS)
        live_pair = COLOR_PAIRS["application"]

        self.assertEqual(set_interface_mode("Classic"), "Classic")
        self.assertEqual(get_interface_mode(), "Classic")
        self.assertEqual(dict(COLOR_PAIRS), dict(CLASSIC_COLOR_PAIRS))
        self.assertEqual(set(COLOR_PAIRS), original_color_keys)
        self.assertIs(COLOR_PAIRS["application"], live_pair)
        self.assertEqual(SHAPES.corner_radius_sm, 6)
        self.assertEqual(SHAPES.corner_radius_md, 10)
        self.assertEqual(SHAPES.corner_radius_lg, 14)
        self.assertEqual(COLOR_PAIRS["application"], ("#E9EEF3", "#E9EEF3"))

        self.assertEqual(set_interface_mode("Modern"), "Modern")
        self.assertEqual(dict(COLOR_PAIRS), dict(MODERN_COLOR_PAIRS))
        self.assertIs(COLOR_PAIRS["application"], live_pair)
        self.assertGreater(SHAPES.corner_radius_md, SHAPES.corner_radius_sm)

    def test_interface_switch_can_defer_ctk_redraw_to_appearance_application(self):
        ctk.set_appearance_mode("Light")
        with mock.patch.object(
            ctk.AppearanceModeTracker,
            "update_callbacks",
        ) as redraw:
            set_interface_mode("Classic", redraw=False)
            redraw.assert_not_called()

            apply_appearance_mode("Light", force_ctk_redraw=True)
            redraw.assert_called_once_with()

    def test_deferred_system_mode_restyles_ttk_after_os_mode_is_resolved(self):
        class DeferredRoot:
            def __init__(self):
                self.callback = None

            def after(self, _delay, callback):
                self.callback = callback
                return "appearance-timer"

            def configure(self, **_options):
                return None

            def winfo_children(self):
                return []

            def event_generate(self, *_args, **_options):
                return None

        root = DeferredRoot()
        effective = {"mode": "Light"}

        def resolve_system():
            effective["mode"] = "Dark"

        with (
            mock.patch("aidas.ui.theme.configure_ttk_styles") as configure,
            mock.patch("aidas.ui.theme.refresh_native_widgets"),
            mock.patch("aidas.ui.theme.synchronize_window_chrome"),
            mock.patch.object(ctk, "get_appearance_mode", side_effect=lambda: effective["mode"]),
            mock.patch.object(ctk, "set_appearance_mode"),
            mock.patch.object(
                ctk.AppearanceModeTracker,
                "init_appearance_mode",
                side_effect=resolve_system,
            ),
            mock.patch.object(ctk.AppearanceModeTracker, "update_callbacks"),
        ):
            apply_appearance_mode(
                "System",
                root=root,
                style="style",
                force_ctk_redraw=True,
                defer_ctk_ms=25,
            )
            configure.assert_called_once_with("style", "System")

            root.callback()

        self.assertEqual(effective["mode"], "Dark")
        self.assertEqual(
            configure.call_args_list,
            [mock.call("style", "System"), mock.call("style", "System")],
        )

    def test_interface_widget_refresh_flattens_and_restores_exact_radii(self):
        class CornerWidget:
            def __init__(self, radius, *children):
                self.radius = radius
                self.children = list(children)

            def winfo_children(self):
                return list(self.children)

            def cget(self, name):
                if name != "corner_radius":
                    raise KeyError(name)
                return self.radius

            def configure(self, **options):
                self.radius = options["corner_radius"]

        rounded = CornerWidget(11)
        square = CornerWidget(0)
        root = CornerWidget(14, rounded, square)

        set_interface_mode("Classic")
        self.assertEqual(refresh_interface_widgets(root), 2)
        self.assertEqual((root.radius, rounded.radius, square.radius), (0, 0, 0))

        set_interface_mode("Modern")
        self.assertEqual(refresh_interface_widgets(root), 2)
        self.assertEqual((root.radius, rounded.radius, square.radius), (14, 11, 0))

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
        self.assertIn("success_hover", COLOR_PAIRS)
        self.assertIn("button", COLOR_PAIRS)
        self.assertIn("button_hover", COLOR_PAIRS)
        self.assertNotEqual(COLOR_PAIRS["window_chrome"][0], "#FFFFFF")
        self.assertNotEqual(COLOR_PAIRS["window_chrome"][1], "#000000")

    def test_neutral_buttons_are_visible_against_light_surfaces(self):
        button = resolve_color(COLOR_PAIRS["button"], "Light")
        self.assertNotEqual(button, resolve_color(COLOR_PAIRS["surface"], "Light"))
        self.assertNotEqual(button, resolve_color(COLOR_PAIRS["sidebar"], "Light"))
        self.assertNotEqual(button, resolve_color(COLOR_PAIRS["button_hover"], "Light"))

    def test_title_and_menu_bars_remain_distinct_in_each_mode(self):
        for appearance_mode in ("Light", "Dark"):
            with self.subTest(appearance_mode=appearance_mode):
                self.assertNotEqual(
                    resolve_color(COLOR_PAIRS["window_chrome"], appearance_mode),
                    resolve_color(COLOR_PAIRS["menu_bar"], appearance_mode),
                )


if __name__ == "__main__":
    unittest.main()
