from tkinter import StringVar, ttk
import unittest
from unittest.mock import Mock

import customtkinter as ctk
from PIL import ImageChops, ImageOps, ImageTk

from aidas.steps.step3_flatten import RSetupWizard
from aidas.ui import rendering
from aidas.ui.components import AppButton, AppRunButton
from aidas.ui.theme import COLOR_PAIRS, SHAPES, get_interface_mode, refresh_interface_widgets, set_interface_mode
from aidas.utils.ui_utils import CollapsibleSection, action_button, directory_row, icon_action_button


class ActionButtonInteractionTests(unittest.TestCase):
    def setUp(self):
        self.previous_interface = get_interface_mode()
        set_interface_mode("Modern", redraw=False)
        class AppRoot(ctk.CTk):
            _deactivate_windows_window_header_manipulation = True
        self.root = AppRoot()
        self.root.geometry("640x360")
        self.root.attributes("-alpha", 0)

    def tearDown(self):
        self.root.destroy()
        set_interface_mode(self.previous_interface, redraw=False)

    def test_keyboard_activation_and_focus_skip_disabled_actions(self):
        command = Mock()
        button = action_button(self.root, self.root, "Browse", command, "folder")
        button.pack()
        entry = ttk.Entry(self.root)
        entry.pack()
        self.root.update()
        self.assertEqual(str(self.root.tk.call("tk_focusNext", str(entry))), str(button))
        self.root.tk.call("focus", "-force", str(button))
        button.event_generate("<Return>")
        self.root.update()
        command.assert_called_once_with()
        button.focus_force()
        button._text_label.event_generate("<space>")
        self.root.update()
        self.assertEqual(command.call_count, 2)
        button.state(["disabled"])
        button._text_label.event_generate("<space>")
        self.root.update()
        self.assertEqual(command.call_count, 2)
        self.assertEqual(str(self.root.tk.call("tk_focusNext", str(entry))), str(entry))

    def test_r_setup_busy_state_disables_rounded_and_native_actions(self):
        content = ttk.Frame(self.root)
        content.pack()
        nested = ttk.Frame(content)
        nested.pack()
        modern = action_button(nested, self.root, "Install", lambda: None, "package")
        modern.pack()
        native = ttk.Button(nested, text="Check")
        native.pack()
        wizard = RSetupWizard.__new__(RSetupWizard)
        wizard.content = content
        for state in ("disabled", "normal"):
            wizard._set_content_buttons(state)
            self.assertEqual(modern.cget("state"), state)
            self.assertEqual(str(native.cget("state")), state)

    def test_step1_folder_picker_keeps_the_folder_icon_on_a_green_button(self):
        browse = Mock()
        directory = StringVar(master=self.root)
        row, _entry, buttons = directory_row(
            self.root, self.root, directory, browse,
            home_command=Mock(), refresh_command=Mock(),
        )
        row.pack()
        folder_button = buttons["browse"]
        next_button = AppButton(self.root, text="Next", variant="success")
        self.assertEqual(folder_button.cget("fg_color"), next_button.cget("fg_color"))
        self.assertEqual(folder_button.cget("fg_color"), COLOR_PAIRS["success_button"])
        self.assertEqual(*folder_button.cget("fg_color"))
        self.assertIs(folder_button.cget("image"), folder_button._aidas_action_icon_source)
        for name in ("home", "refresh"):
            self.assertEqual(buttons[name].cget("fg_color"), COLOR_PAIRS["button"])
        folder_button.invoke()
        browse.assert_called_once_with()

    def test_step1_toolbar_has_smooth_borders_at_fractional_display_scales(self):
        self.addCleanup(ctk.set_appearance_mode, ctk.get_appearance_mode())
        self.addCleanup(ctk.set_widget_scaling, ctk.ScalingTracker.widget_scaling)
        self.root.geometry("640x360+10000+10000")
        ctk.set_appearance_mode("Dark")
        section = CollapsibleSection(self.root, "SDB Files")
        section.pack(fill="x")
        directory = StringVar(master=self.root)
        row, _entry, buttons = directory_row(
            section.body, self.root, directory, Mock(),
            home_command=Mock(), refresh_command=Mock(),
        )
        row.pack(fill="x")

        def rgb(canvas, color):
            return tuple(channel // 257 for channel in canvas.winfo_rgb(color))

        for scaling in (1, 1.25, 1.5):
            ctk.set_widget_scaling(scaling)
            self.root.update()
            for name, button in buttons.items():
                with self.subTest(scaling=scaling, button=name):
                    canvas = button._canvas
                    image = ImageTk.getimage(canvas._aidas_surface_photo).convert("RGB")
                    width, height = image.size
                    bg = rgb(canvas, canvas.cget("background"))
                    fill = rgb(canvas, canvas.itemcget("inner_parts", "fill"))
                    border = rgb(canvas, canvas.itemcget("border_parts", "fill"))
                    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
                        self.assertEqual(image.getpixel((x, y)), bg)
                    for reflected in (ImageOps.mirror(image), ImageOps.flip(image)):
                        differences = ImageChops.difference(image, reflected).getextrema()
                        self.assertLessEqual(max(high for _, high in differences), 1)
                    colors = {color for _, color in image.getcolors(maxcolors=width * height)}
                    self.assertTrue({bg, fill, border} <= colors)
                    self.assertTrue(colors - {bg, fill, border})

                    outer_radius = float(canvas.itemcget("border_parts", "width")) / 2
                    inner_radius = float(canvas.itemcget("inner_parts", "width")) / 2
                    effective_scale = button._get_widget_scaling()
                    self.assertAlmostEqual(outer_radius, SHAPES.corner_radius_sm * effective_scale, delta=0.02)
                    self.assertAlmostEqual(outer_radius - inner_radius, SHAPES.border_width * effective_scale, delta=0.02)

                    if name == "browse":
                        outer, inner = (
                            rendering._item_mask(canvas, canvas.find_withtag(tag)[0], image.size)
                            for tag in ("border_parts", "inner_parts")
                        )
                        coverage = zip(outer.tobytes(), inner.tobytes())
                        solid_edge = {
                            image.getpixel((offset % width, offset // width))
                            for offset, (outer_alpha, inner_alpha) in enumerate(coverage)
                            if outer_alpha == 255 and inner_alpha < 255
                        }
                        self.assertEqual(solid_edge, {fill})

    def test_run_uses_the_shared_green_action_palette(self):
        for mode in ("Modern", "Classic"):
            with self.subTest(mode=mode):
                set_interface_mode(mode, redraw=False)
                button = AppRunButton(self.root, command=Mock())
                self.assertEqual(button.cget("fg_color"), COLOR_PAIRS["success_button"])
                self.assertEqual(button.cget("hover_color"), COLOR_PAIRS["success_hover"])
                self.assertEqual(button.cget("text_color"), COLOR_PAIRS["on_success_button"])
                button.destroy()

    def test_retained_actions_round_after_switching_from_classic_to_modern(self):
        set_interface_mode("Classic", redraw=False)
        button = action_button(self.root, self.root, "Save", lambda: None, "save",
                               state="disabled")
        button.pack()
        icon = icon_action_button(self.root, self.root, lambda: None, "folder",
                                  tooltip="Select folder")
        icon.pack()
        self.assertIsInstance(button, AppButton)
        self.assertEqual(button.cget("corner_radius"), 0)
        set_interface_mode("Modern", redraw=False)
        refresh_interface_widgets(self.root)
        self.root.update_idletasks()
        self.assertGreater(button.cget("corner_radius"), 0)
        self.assertEqual(button.cget("state"), "disabled")
        self.assertEqual(icon.winfo_width(), icon.winfo_height())
        self.assertIsNone(icon._text_label)
