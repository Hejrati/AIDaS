"""Step 5 keeps its instructions readable without changing the workspace split."""

from pathlib import Path
import tempfile
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.steps.step5_compile import DEFAULT_OUTPUT_FILENAME, Step5Frame
from aidas.ui.measurement_tree import MeasurementDirectoryTree
from aidas.ui.theme import apply_appearance_mode, get_interface_mode, set_interface_mode
from aidas.utils.ui_layout import LAYOUT, workspace_sidebar_width


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class Step5LayoutTests(unittest.TestCase):
    def setUp(self):
        previous_appearance = (
            "System" if ctk.AppearanceModeTracker.appearance_mode_set_by == "system"
            else ctk.get_appearance_mode()
        )
        self.addCleanup(ctk.set_appearance_mode, previous_appearance)
        self.addCleanup(set_interface_mode, get_interface_mode(), redraw=False)
        self.addCleanup(ctk.set_widget_scaling, ctk.ScalingTracker.widget_scaling)
        self.addCleanup(ctk.set_window_scaling, ctk.ScalingTracker.window_scaling)
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)
        set_interface_mode("Modern", redraw=False)
        try:
            self.root = _QuietRoot()
        except tk.TclError as error:
            self.skipTest(f"Tk is unavailable: {error}")
        self.addCleanup(self.root.destroy)
        self.root.attributes("-alpha", 0)
        self.root.geometry("1024x680+0+0")
        self.original_tk_scaling = float(self.root.tk.call("tk", "scaling"))
        self.addCleanup(self.root.tk.call, "tk", "scaling", self.original_tk_scaling)
        self.style = ttk.Style(self.root)
        apply_appearance_mode("Dark", root=self.root, style=self.style)
        self.frame = Step5Frame(self.root)
        self.frame.pack(fill="both", expand=True)
        self.root.update_idletasks()

    def _assert_description_fits(self):
        frame = self.frame
        label = frame.input_description_label
        row = frame.input_entry.master
        input_body = label.master
        input_card = input_body.master.master
        output_card = frame.output_entry.master.master.master.master
        self.assertGreater(label.winfo_width(), 1)
        self.assertLessEqual(label.winfo_reqwidth(), label.winfo_width())
        self.assertLessEqual(label.winfo_reqheight(), label.winfo_height())
        self.assertLessEqual(
            int(label.cget("wraplength")), label.winfo_width() - 2 * LAYOUT.space_xs,
        )
        self.assertGreaterEqual(
            row.winfo_rooty() - (label.winfo_rooty() + label.winfo_height()),
            LAYOUT.space_sm,
        )
        self.assertLessEqual(
            frame.include_fovea_checkbox.winfo_rooty()
            + frame.include_fovea_checkbox.winfo_height(),
            input_body.winfo_rooty() + input_body.winfo_height(),
        )
        self.assertLessEqual(
            input_card.winfo_rooty() + input_card.winfo_height(), output_card.winfo_rooty(),
        )
        expected_sidebar = workspace_sidebar_width(
            frame.workspace.winfo_width(),
            sidebar_width=frame._sidebar_width,
            sidebar_minimum=frame._sidebar_minimum,
            content_minimum=frame._content_minimum,
        )
        self.assertLessEqual(abs(frame.workspace.sash_coord(0)[0] - expected_sidebar), 1)
        self.assertGreater(frame.directory_view.canvas.winfo_width(), 1)

    def test_description_rewraps_at_narrow_widths_and_display_scales_in_both_interfaces(self):
        for interface, appearance, scale, width, height in (
            ("Modern", "Dark", 1.0, 1024, 680),
            ("Modern", "Light", 1.0, 900, 640),
            ("Modern", "Dark", 1.25, 1100, 720),
            ("Classic", "Light", 1.25, 950, 700),
            ("Modern", "Dark", 1.5, 1280, 820),
            ("Classic", "Dark", 1.5, 1024, 760),
            ("Modern", "Light", 1.0, 1024, 680),
        ):
            with self.subTest(interface=interface, appearance=appearance, scale=scale, width=width):
                self.root.tk.call("tk", "scaling", self.original_tk_scaling * scale)
                ctk.set_widget_scaling(scale)
                set_interface_mode(interface, redraw=False)
                apply_appearance_mode(appearance, root=self.root, style=self.style)
                self.root.geometry(f"{width}x{height}+0+0")
                self.root.update_idletasks()
                self._assert_description_fits()

    def test_unchanged_label_width_does_not_schedule_another_layout_change(self):
        label = self.frame.input_description_label
        with mock.patch.object(label, "configure", wraps=label.configure) as configure:
            label.event_generate(
                "<Configure>", width=label.winfo_width(), height=label.winfo_height(),
            )
            self.root.update_idletasks()
        configure.assert_not_called()
        self._assert_description_fits()

    def test_handoff_only_prepares_fields_and_the_existing_button_starts_compilation(self):
        frame = self.frame
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            output = folder / DEFAULT_OUTPUT_FILENAME
            with (
                mock.patch.object(frame, "_start_compiler") as start,
                mock.patch.object(
                    frame.directory_view, "set_directory", wraps=frame.directory_view.set_directory,
                ) as set_directory,
            ):
                frame.start_from_step4(folder)
                frame.on_show()
                self.root.update_idletasks()
                start.assert_not_called()
                self.assertFalse(output.exists())
                self.assertEqual(Path(frame.input_var.get()), folder)
                self.assertEqual(Path(frame.output_var.get()), output)
                set_directory.assert_called_with(folder, output_path=output)
                deadline = time.monotonic() + 5
                while frame._input_scan_state != "ready":
                    self.assertLess(time.monotonic(), deadline)
                    self.root.update()
                    time.sleep(.005)

                frame.run_button.invoke()

                start.assert_called_once_with(
                    folder, output, True, thread_name="aidas-step5-compiler",
                )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
