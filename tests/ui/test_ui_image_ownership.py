"""Shared icons must not depend on whichever Tk root was created first."""

import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

import customtkinter as ctk
from PIL import Image

from aidas.ui.components import AppRunButton
from aidas.ui.images import OwnedCTkImage
from aidas.utils.ui_utils import (
    ACTION_ICON_FILES,
    CollapsibleSection,
    action_button,
    load_color_close_ctk_icon,
    load_compile_ctk_icon,
    load_ctk_image,
    load_ui_icon,
)


class UIImageOwnershipTests(unittest.TestCase):
    def setUp(self):
        try:
            self.first_root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        self.addCleanup(self.first_root.destroy)
        self.first_root.withdraw()
        self.owner = tk.Tk()
        self.addCleanup(self.owner.destroy)
        self.owner.withdraw()
        self.assertIsNot(tk._default_root, self.owner)
        self.assertIsNot(self.first_root.tk, self.owner.tk)

    def assertOwnerImage(self, image):
        self.assertIs(image.tk, self.owner.tk)
        self.assertIn(str(image), self.owner.tk.call("image", "names"))
        self.assertNotIn(str(image), self.first_root.tk.call("image", "names"))

    def test_native_icon_and_data_fallback_use_the_owner_interpreter(self):
        icon = load_ui_icon(self.owner, ACTION_ICON_FILES["folder"], size=(17, 19))
        self.assertOwnerImage(icon)
        self.assertEqual((icon.width(), icon.height()), (17, 19))
        ttk.Label(self.owner, image=icon)

        photo_image = tk.PhotoImage
        calls = []

        def retry_from_data(*args, **kwargs):
            calls.append(kwargs)
            if "file" in kwargs:
                raise tk.TclError("Exercise the PNG data fallback")
            return photo_image(*args, **kwargs)

        with mock.patch("aidas.utils.ui_utils.tk.PhotoImage", side_effect=retry_from_data):
            fallback = load_ui_icon(self.owner, ACTION_ICON_FILES["folder"])
        self.assertOwnerImage(fallback)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["master"] is self.owner for call in calls))

    def test_ctk_icon_modes_scaling_and_configuration_keep_owner_and_cache(self):
        light = Image.new("RGBA", (12, 12), "red")
        dark = Image.new("RGBA", (12, 12), "blue")
        icon = OwnedCTkImage(self.owner, light_image=light, dark_image=dark, size=(12, 12))
        self.assertIsInstance(icon, ctk.CTkImage)
        for mode in ("light", "dark"):
            for scale in (1, 1.5, 2):
                with self.subTest(mode=mode, scale=scale):
                    photo = icon.create_scaled_photo_image(scale, mode)
                    self.assertOwnerImage(photo)
                    self.assertEqual((photo.width(), photo.height()), (round(12 * scale),) * 2)
                    self.assertIs(photo, icon.create_scaled_photo_image(scale, mode))
        previous = icon.create_scaled_photo_image(1, "light")
        icon.configure(light_image=Image.new("RGBA", (12, 12), "green"), size=(16, 14))
        updated = icon.create_scaled_photo_image(1, "light")
        self.assertIsNot(updated, previous)
        self.assertOwnerImage(updated)
        self.assertEqual((updated.width(), updated.height()), (16, 14))
        for sources in ({"light_image": light}, {"dark_image": dark}):
            fallback = OwnedCTkImage(self.owner, **sources)
            for mode in ("light", "dark"):
                self.assertOwnerImage(fallback.create_scaled_photo_image(1, mode))

    def test_shared_ctk_helpers_and_button_state_recolors_render_in_owner(self):
        icons = (
            load_ctk_image(self.owner, ACTION_ICON_FILES["folder"]),
            load_color_close_ctk_icon(self.owner),
            load_compile_ctk_icon(self.owner),
        )
        for icon in icons:
            ctk.CTkLabel(self.owner, image=icon, text="")
            for mode in ("light", "dark"):
                self.assertOwnerImage(icon.create_scaled_photo_image(1, mode))

        next_button = action_button(
            self.owner, self.owner, "Next", lambda: None, "next",
            style="AIDaS.PrimaryAction.TButton",
        )
        run_button = AppRunButton(self.owner, command=lambda: None)
        for button in (next_button, run_button):
            for state in ("normal", "disabled", "normal"):
                button.configure(state=state)
                for mode in ("light", "dark"):
                    self.assertOwnerImage(button.cget("image").create_scaled_photo_image(1.5, mode))

        section = CollapsibleSection(self.owner, "Folder options")
        for _ in range(2):
            for mode in ("light", "dark"):
                self.assertOwnerImage(section.header.cget("image").create_scaled_photo_image(1, mode))
            section.toggle()
        self.owner.update_idletasks()


if __name__ == "__main__":
    unittest.main()
