import unittest
import tkinter as tk
from tkinter import ttk

import customtkinter as ctk
from PIL import Image

from aidas.ui.components import AppButton
from aidas.ui.theme import (
    apply_appearance_mode, get_interface_mode, refresh_interface_widgets,
    set_interface_mode,
)
from aidas.utils.ui_utils import (
    ACTION_ICON_FILES, _dark_mode_icon, _tint_icon, action_button, load_ctk_image,
)


def _rgb(color):
    return tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))


def _contrast(foreground, background):
    def luminance(color):
        channels = [value / 255 for value in _rgb(color)]
        channels = [value / 12.92 if value <= 0.04045
                    else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
        return sum(value * weight for value, weight in zip(channels, (0.2126, 0.7152, 0.0722)))

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


class UIIconColorTests(unittest.TestCase):
    def test_build_stack_keeps_source_colors_when_enabled_after_emphasis_changes(self):
        try:
            root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.addCleanup(root.destroy)
        root.withdraw()
        source = load_ctk_image(root, "flat-color-icons--stack-of-photos.png", size=20)
        original = {mode: source.cget(mode).tobytes()
                    for mode in ("light_image", "dark_image")}
        button = AppButton(root, image=source, preserve_icon_colors=True,
                           variant="secondary", state="disabled")
        for _ in range(3):
            button.set_variant("primary")
            button.configure(state="normal")
            self.assertIs(button.cget("image"), source)
            for mode, expected in original.items():
                self.assertEqual(button.cget("image").cget(mode).tobytes(), expected)
            button.configure(state="disabled")
            self.assertIsNot(button.cget("image"), source)
            button.set_variant("secondary")
        for mode, expected in original.items():
            self.assertEqual(source.cget(mode).tobytes(), expected)

    def test_dark_mode_preserves_saturated_red_and_blue_pixels(self):
        source = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
        source.putpixel((0, 0), (255, 0, 0, 255))
        source.putpixel((1, 0), (63, 81, 181, 255))
        source.putpixel((2, 0), (5, 5, 5, 255))

        converted = _dark_mode_icon(source)

        self.assertEqual(converted.getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(converted.getpixel((1, 0)), (63, 81, 181, 255))
        self.assertEqual(converted.getpixel((2, 0)), (221, 231, 240, 255))

    def test_tint_uses_requested_color_and_preserves_alpha(self):
        source = Image.new("RGBA", (2, 1), (10, 20, 30, 0))
        source.putpixel((1, 0), (10, 20, 30, 128))

        tinted = _tint_icon(source, "#07131C")

        self.assertEqual(tinted.getpixel((0, 0)), (7, 19, 28, 0))
        self.assertEqual(tinted.getpixel((1, 0)), (7, 19, 28, 128))

    def test_action_icons_follow_labels_through_live_theme_and_state_changes(self):
        root = None
        original_interface = get_interface_mode()
        original_appearance = ctk.get_appearance_mode()
        try:
            try:
                root = ctk.CTk()
                root.withdraw()
                root.attributes("-alpha", 0)
                root.geometry("700x400+0+0")
                root.deiconify()
            except tk.TclError as exc:
                self.skipTest(f"A transparent Tk test window is unavailable: {exc}")

            style = ttk.Style(root)
            set_interface_mode("Modern", redraw=False)
            apply_appearance_mode("Light", root=root, style=style, force_ctk_redraw=True)
            helper_buttons = []
            modern_buttons = []
            sources = []
            for action in ("previous", "next", "process"):
                helper = action_button(root, root, action, lambda: None, action,
                                       style="AIDaS.PrimaryAction.TButton")
                self.assertIsInstance(helper, AppButton)
                helper.pack()
                helper_buttons.append(helper)
                helper_source = helper._aidas_action_icon_source
                sources.append((helper_source, helper_source.cget("light_image").tobytes(),
                                helper_source.cget("dark_image").tobytes()))
                source = load_ctk_image(root, ACTION_ICON_FILES[action], size=20)
                sources.append((source, source.cget("light_image").tobytes(),
                                source.cget("dark_image").tobytes()))
                modern = AppButton(root, variant="primary", text=action, image=source)
                modern.pack()
                modern_buttons.append(modern)

            folder_button = action_button(root, root, "Select folder", lambda: None, "folder")
            self.assertIsInstance(folder_button, AppButton)
            folder_button.pack()
            folder_icon = folder_button.cget("image")
            folder_sources = {
                mode: folder_icon.cget(mode).copy()
                for mode in ("light_image", "dark_image")
            }
            for image in folder_sources.values():
                self.assertEqual(image.getchannel("A").getextrema(), (0, 255))
                self.assertGreater(len({pixel[:3] for pixel in image.getdata() if pixel[3] == 255}), 1)

            for interface, appearance in (
                ("Modern", "Light"), ("Modern", "Dark"),
                ("Classic", "Light"), ("Classic", "Dark"),
                ("Modern", "Dark"), ("Modern", "Light"),
            ):
                with self.subTest(interface=interface, appearance=appearance):
                    set_interface_mode(interface, redraw=False)
                    apply_appearance_mode(appearance, root=root, style=style, force_ctk_redraw=True)
                    refresh_interface_widgets(root)
                    root.update()
                    mode_index = 1 if appearance == "Dark" else 0
                    for disabled in (False, True, False):
                        for button in helper_buttons + modern_buttons + [folder_button]:
                            button.state(["disabled" if disabled else "!disabled"])
                        root.update()

                        for button in helper_buttons + modern_buttons:
                            color = button.cget("text_color_disabled" if disabled else "text_color")
                            foreground = color[mode_index] if isinstance(color, (list, tuple)) else color
                            image = button.cget("image").cget(
                                "dark_image" if appearance == "Dark" else "light_image")
                            colors = {pixel[:3] for pixel in image.getdata() if pixel[3] >= 240}
                            self.assertEqual(colors, {_rgb(foreground)})
                            if not disabled:
                                for option in ("fg_color", "hover_color"):
                                    background = button.cget(option)
                                    background = (background[mode_index]
                                                  if isinstance(background, (list, tuple)) else background)
                                    self.assertGreaterEqual(_contrast(foreground, background), 4.5)

                        # Colorful secondary glyphs retain their complete alpha
                        # edges and saturated details when disabled; no native
                        # ttk stipple or monochrome action tint is introduced.
                        self.assertIs(folder_button.cget("image"), folder_icon)
                        for mode, expected in folder_sources.items():
                            actual = folder_button.cget("image").cget(mode)
                            self.assertEqual(actual.tobytes(), expected.tobytes())

                    # Moving workflow emphasis must preserve the action glyph's legibility.
                    for button in helper_buttons + modern_buttons:
                        button.set_variant("secondary")
                        image = button.cget("image").cget(
                            "dark_image" if appearance == "Dark" else "light_image")
                        foreground = button.cget("text_color")[mode_index]
                        colors = {pixel[:3] for pixel in image.getdata() if pixel[3] >= 240}
                        self.assertEqual(colors, {_rgb(foreground)})
                        button.set_variant("primary")
                    self.assertIs(folder_button.cget("image"), folder_icon)
                    for source, light_bytes, dark_bytes in sources:
                        self.assertEqual(source.cget("light_image").tobytes(), light_bytes)
                        self.assertEqual(source.cget("dark_image").tobytes(), dark_bytes)
        finally:
            if root is not None:
                root.destroy()
            set_interface_mode(original_interface)
            ctk.set_appearance_mode(original_appearance)


if __name__ == "__main__":
    unittest.main()
