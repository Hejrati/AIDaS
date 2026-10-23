"""Windows caption integration for the Step 4 profile editor."""

import sys
import tkinter as tk
from tkinter import ttk
import unittest

import customtkinter as ctk

from aidas.ui.theme import COLOR_PAIRS


@unittest.skipUnless(sys.platform.startswith("win"), "Windows caption integration")
class Step4ProfileZoomCaptionTests(unittest.TestCase):
    def setUp(self):
        class AppRoot(ctk.CTk):
            _deactivate_windows_window_header_manipulation = True

        self.root = AppRoot()
        self.root.interface_mode = "Modern"
        self.root.withdraw()

    def tearDown(self):
        self.root.destroy()

    def _settle_events(self, milliseconds=80):
        finished = tk.BooleanVar(master=self.root, value=False)
        self.root.after(milliseconds, lambda: finished.set(True))
        self.root.wait_variable(finished)

    def test_step4_zoom_remains_visible_during_pointer_motion_and_theme_changes(self):
        import numpy as np
        from aidas.steps.step4_analyze_isez import Step4ProfileZoomDialog, RESULTS_HEADERS

        self.root.deiconify()
        self.root.update()
        dialog = Step4ProfileZoomDialog(
            self.root, roi_index=0, roi_suffix="01",
            profile=np.sin(np.linspace(0, 10, 100)) + 2, start=20, end=70,
            on_preview=lambda *_args, **_kwargs: {k: 1.0 for k in RESULTS_HEADERS[1:]},
            on_apply=lambda *_args, **_kwargs: True,
            on_save=lambda *_args, **_kwargs: False,
            on_close=lambda *_args: None,
        )
        self._settle_events()
        unmapped = []
        dialog.bind("<Unmap>", lambda event: unmapped.append(event)
                    if event.widget is dialog else None, add="+")
        chart = dialog.canvas.get_tk_widget()
        # Real Motion events use the Toplevel bind tag installed on the chart.
        for offset in range(40):
            self.root.after(offset * 2,
                            lambda offset=offset: chart.event_generate("<Motion>",
                                                                       x=80 + offset, y=100))
        self._settle_events(250)
        previous = ctk.get_appearance_mode()
        try:
            for mode in ("Light", "Dark"):
                ctk.set_appearance_mode(mode)
                from aidas.ui.theme import refresh_native_widgets, resolve_color
                from matplotlib.colors import to_hex
                refresh_native_widgets(dialog)
                self._settle_events()
                body_color = resolve_color(COLOR_PAIRS["surface_elevated"])
                self.assertEqual(dialog.cget("bg"), body_color)
                self.assertEqual(to_hex(dialog.figure.get_facecolor()).lower(), body_color.lower())
                self.assertEqual(ttk.Style(dialog).lookup("ProfileZoom.TLabel", "background"), body_color)
                self.assertNotEqual(body_color, resolve_color(COLOR_PAIRS["window_chrome"]))
                self.assertEqual(dialog.state(), "normal")
                self.assertTrue(dialog.winfo_viewable())
                self.assertFalse(unmapped, "The custom caption must never hide the zoom window")
        finally:
            ctk.set_appearance_mode(previous)
        dialog.close()
        self.assertFalse(dialog.winfo_exists())

    def test_step4_navigation_skips_gaps_and_keeps_unapplied_edits(self):
        import numpy as np
        from unittest.mock import Mock
        from aidas.steps.step4_analyze_isez import Step4ProfileZoomDialog, RESULTS_HEADERS

        self.root.deiconify()
        self.root.update()
        profile = np.sin(np.linspace(0, 10, 100)) + 2
        navigate = Mock(side_effect=lambda index: {
            "roi_index": index, "roi_suffix": f"{index + 1:02d}",
            "profile": profile, "start": 20, "end": 70,
        })
        apply = Mock(return_value=True)
        save = Mock(return_value=True)
        dialog = Step4ProfileZoomDialog(
            self.root, roi_index=0, roi_suffix="01", profile=profile, start=20, end=70,
            on_preview=lambda *_args, **_kwargs: {k: 1.0 for k in RESULTS_HEADERS[1:]},
            on_apply=apply, on_save=save, on_close=lambda *_args: None,
            navigation_indices=[0, 3, 5], on_navigate=navigate,
        )
        self._settle_events()
        window_id = dialog.winfo_id()
        canvas = dialog.canvas
        self.assertEqual(dialog.previous_plot_button.cget("state"), "disabled")
        self.assertEqual(dialog.previous_plot_button.cget("text"), "Previous")
        self.assertEqual(dialog.next_plot_button.cget("text"), "Next")
        for button in (dialog.previous_plot_button, dialog.next_plot_button,
                       dialog.save_button, dialog.apply_button, dialog.cancel_button):
            self.assertEqual(button.cget("anchor"), "center")
            content = [label for label in (button._image_label, button._text_label)
                       if label is not None]
            left = min(label.winfo_x() for label in content)
            right = max(label.winfo_x() + label.winfo_width() for label in content)
            top = min(label.winfo_y() for label in content)
            bottom = max(label.winfo_y() + label.winfo_height() for label in content)
            self.assertAlmostEqual((left + right) / 2, button.winfo_width() / 2, delta=2)
            self.assertAlmostEqual((top + bottom) / 2, button.winfo_height() / 2, delta=2)
        self.assertIsNotNone(dialog.previous_plot_button.cget("image"))
        self.assertIsNotNone(dialog.next_plot_button.cget("image"))
        plotted_end = int(dialog._baseline_line.get_xdata()[1])
        dialog._move_boundary("start", 25, refresh_measurements=True)
        dialog.next_plot_button.invoke()
        self.assertEqual(dialog.roi_index, 3)
        self.assertEqual(dialog.plot_number_label.cget("text"), "ROI 04")
        self.assertEqual(dialog.title(), "ROI Profile Editor \u2014 ROI 04")
        self.assertEqual(dialog._aidas_dialog_title_bar.title_label.cget("text"), dialog.title())
        self.assertIs(dialog.canvas, canvas)
        self.assertEqual(dialog.winfo_id(), window_id)
        dialog.next_plot_button.invoke()
        self.assertEqual(dialog.roi_index, 5)
        self.assertEqual(dialog.next_plot_button.cget("state"), "disabled")
        dialog._navigate_profile(1)
        self.assertEqual(navigate.call_count, 2)
        dialog.previous_plot_button.invoke()
        dialog.previous_plot_button.invoke()
        self.assertEqual(dialog.roi_index, 0)
        self.assertEqual(dialog._start, 25)
        self.assertTrue(dialog._has_pending_changes())
        self.assertEqual(dialog.apply_button.cget("state"), "normal")
        self.assertEqual(dialog.previous_plot_button.cget("state"), "disabled")
        apply.assert_not_called()
        save.assert_not_called()
        dialog._apply()
        apply.assert_called_once_with(0, 25, plotted_end, adjust_bounds=False)
        dialog.close()
