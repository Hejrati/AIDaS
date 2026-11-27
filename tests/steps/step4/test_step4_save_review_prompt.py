"""The Step 4 save prompt uses the application's dialog controls."""

import sys
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.steps.step4_analyze_isez import Step4Frame
from aidas.ui.components import AppButton
from aidas.ui.theme import COLOR_PAIRS, resolve_color


def _buttons(parent):
    for child in parent.winfo_children():
        if isinstance(child, AppButton):
            yield child
        else:
            yield from _buttons(child)


@unittest.skipUnless(sys.platform.startswith("win"), "Windows caption integration")
class Step4SaveReviewPromptTests(unittest.TestCase):
    def setUp(self):
        class AppRoot(ctk.CTk):
            _deactivate_windows_window_header_manipulation = True

        self.addCleanup(ctk.set_appearance_mode, ctk.get_appearance_mode())
        ctk.set_appearance_mode("Dark")
        self.root = AppRoot()
        self.addCleanup(self.root.destroy)
        self.root.interface_mode = "Modern"
        self.root.attributes("-alpha", 0)
        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack()
        self.root.update_idletasks()

    def test_save_review_prompt_uses_app_caption_and_ordered_semantic_buttons(self):
        cases = (
            (True, "Review 2 plots", "review"),
            (True, "Save without these 2 plots", "save"),
            (False, "Cancel", None),
        )
        for can_save_without, clicked, result in cases:
            with self.subTest(clicked=clicked):
                expected = [("Review 2 plots", "warning")]
                if can_save_without:
                    expected.append(("Save without these 2 plots", "success_button"))
                expected.append(("Cancel", "danger"))

                def inspect_prompt(window):
                    try:
                        window.update_idletasks()
                        caption = window._aidas_dialog_title_bar
                        self.assertEqual(caption.title_label.cget("text"), window.title())
                        self.assertEqual(resolve_color(caption.cget("fg_color")),
                                         resolve_color(COLOR_PAIRS["window_chrome"]))
                        self.assertEqual(window.pack_slaves()[0], caption)

                        buttons = sorted(_buttons(window), key=lambda button: button.winfo_rootx())
                        self.assertEqual(
                            [(button.cget("text"), resolve_color(button.cget("fg_color")))
                             for button in buttons],
                            [(label, resolve_color(COLOR_PAIRS[color])) for label, color in expected],
                        )
                        next(button for button in buttons if button.cget("text") == clicked).invoke()
                    finally:
                        if window.winfo_exists():
                            window.destroy()

                with mock.patch.object(self.frame, "wait_window", side_effect=inspect_prompt):
                    choice = Step4Frame._choose_stack_review_action(
                        self.frame, 2, can_save_without=can_save_without,
                    )
                self.assertEqual(choice, result)
