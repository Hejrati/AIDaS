"""Real-widget checks for the two required batch fovea-picker stages."""

from __future__ import annotations

import os
import tkinter as tk
import unittest
from unittest import mock

import customtkinter as ctk
import numpy as np

from aidas.steps.step2_annotate import Step2Frame
from aidas.steps import step2_annotate
from aidas.ui.components import AppButton
from aidas.ui.prompt_cue import PromptCue
from aidas.ui.theme import COLOR_PAIRS, resolve_color
from tests.support.paths import PROJECT_ROOT


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class FoveaPickerTests(unittest.TestCase):
    def setUp(self):
        self.scaling = ctk.ScalingTracker.widget_scaling
        self.appearance = ctk.get_appearance_mode()
        self.appearance_source = ctk.AppearanceModeTracker.appearance_mode_set_by
        self.addCleanup(self._restore_settings)
        ctk.set_widget_scaling(1)
        ctk.set_appearance_mode("Dark")
        try:
            self.root = _QuietRoot()
            self.root.attributes("-alpha", 0)
        except tk.TclError as exc:
            self.skipTest(f"A transparent Tk test window is unavailable: {exc}")
        self.addCleanup(self._destroy_root)
        self.root.geometry("860x580+20+20")
        self.root.title("AIDaS batch fovea selection")
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)

        self.frame = Step2Frame.__new__(Step2Frame)
        ctk.CTkFrame.__init__(self.frame, self.root, fg_color=COLOR_PAIRS["surface"])
        self.frame.pack(fill="both", expand=True)
        self.frame.canvas_area = ctk.CTkFrame(self.frame, fg_color=COLOR_PAIRS["surface"])
        self.frame.canvas_area.pack(fill="both", expand=True)
        viewer = ctk.CTkFrame(self.frame.canvas_area, corner_radius=0)
        viewer.pack(fill="both", expand=True)
        viewer.canvas = tk.Canvas(viewer, background="#151719", highlightthickness=0)
        viewer.canvas.pack(fill="both", expand=True)
        self.original_zoom = mock.Mock()
        viewer._cb_zoom = self.original_zoom
        viewer._i2c = lambda x, y: (40 + x, 160 + y * 0.4)
        self.guidance_events = []
        viewer.set_fovea_label_offset = mock.Mock(
            side_effect=lambda offset: self.guidance_events.append(("label", offset)),
        )
        self.guidance_renderer = None

        def record_guidance(target, **kwargs):
            self.guidance_events.append(("arrow", target))
            if target is None:
                if self.guidance_renderer is not None:
                    self.guidance_renderer(None)
                self.guidance_renderer = None
            else:
                self.guidance_renderer = kwargs.get("renderer")

        viewer.set_fovea_guidance_target = mock.Mock(side_effect=record_guidance)
        self.frame.image_canvas = viewer
        self.image = np.zeros((200, 500), dtype=np.uint8)
        self.frame.image_data = self.image
        self.frame.fovea_x = 125
        self.frame.fovea_x_entry_var = tk.StringVar(self.root, value="125")
        self.frame.status_var = tk.StringVar(self.root)
        self.frame._load_image_from_path = mock.Mock(return_value=self.image)
        self.frame._show_image = mock.Mock()
        self.frame._ensure_fovea_line = mock.Mock()
        for name in (
            "_hide_annotation_tutorial", "_set_drawing_locked", "_clear_image_display",
            "_update_boundary_action_buttons", "_update_batch_ai_button_state",
            "_queue_annotation_tutorial",
        ):
            setattr(self.frame, name, mock.Mock())
        self.root.update()
        self.canvas_binding = viewer.canvas.bind("<Configure>")
        self.generated_events = []
        self.cues = []
        self.guidance_arrows = []

    def _destroy_root(self):
        # Use the application's CTk teardown so shared appearance/DPI polling
        # is preserved for any subsequent GUI test in this process.
        self.root.destroy()

    def _restore_settings(self):
        ctk.set_widget_scaling(self.scaling)
        ctk.set_appearance_mode("System" if self.appearance_source == "system" else self.appearance)

    @staticmethod
    def _descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from FoveaPickerTests._descendants(child)

    def _buttons(self):
        return {
            widget.cget("text"): widget
            for widget in self._descendants(self.frame)
            if isinstance(widget, AppButton)
        }

    def _assert_inside(self, child, parent, inset=0):
        self.assertGreaterEqual(child.winfo_rootx(), parent.winfo_rootx() + inset)
        self.assertGreaterEqual(child.winfo_rooty(), parent.winfo_rooty() + inset)
        self.assertLessEqual(child.winfo_rootx() + child.winfo_width(),
                             parent.winfo_rootx() + parent.winfo_width() - inset)
        self.assertLessEqual(child.winfo_rooty() + child.winfo_height(),
                             parent.winfo_rooty() + parent.winfo_height() - inset)

    def _capture(self, filename):
        if not os.environ.get("AIDAS_CAPTURE_FOVEA_CONTROLS"):
            return
        from PIL import ImageGrab

        self.root.attributes("-alpha", 1)
        self.root.lift()
        self.root.update()
        self.root.after(100, self.root.quit)
        self.root.mainloop()
        left, top = self.root.winfo_rootx(), self.root.winfo_rooty()
        target = PROJECT_ROOT / "debug_outputs" / filename
        target.parent.mkdir(exist_ok=True)
        ImageGrab.grab(bbox=(left, top, left + self.root.winfo_width(),
                             top + self.root.winfo_height())).save(target)
        self.root.attributes("-alpha", 0)

    def _run_timers(self, milliseconds):
        self.root.after(milliseconds, self.root.quit)
        self.root.mainloop()

    def _collect(self, paths, stages):
        """Drive real nested Tk waits; a failed callback still releases the wait."""
        stage_errors = []
        waits = []

        def wait(variable):
            index = len(waits)
            waits.append(index)

            def act():
                try:
                    stages[index](self._buttons())
                except BaseException as exc:
                    stage_errors.append(exc)
                    variable.set("cancel")

            self.root.after(140, act)
            tk.Misc.wait_variable(self.frame, variable)

        def record_event(widget, sequence, **kwargs):
            self.generated_events.append((widget, sequence, kwargs))

        def make_cue(owner):
            cue = PromptCue(owner)
            self.cues.append(cue)
            return cue

        def make_arrow(*args, **kwargs):
            arrow = mock.Mock()
            self.guidance_arrows.append(arrow)
            return arrow

        self.frame.wait_variable = wait
        with mock.patch.object(tk.Misc, "event_generate", autospec=True, side_effect=record_event), \
             mock.patch("aidas.steps.step2_annotate.PromptCue", side_effect=make_cue), \
             mock.patch("aidas.steps.step2_annotate.GuidanceArrow", side_effect=make_arrow):
            result = self.frame._collect_folder_fovea_lines(paths)
        self.root.update()
        if stage_errors:
            raise stage_errors[0]
        self.assertEqual(len(waits), len(stages))
        self.assertEqual(self.callback_errors, [])
        self.assertFalse(self.frame._batch_fovea_picker_active)
        self.assertIs(self.frame.image_canvas._cb_zoom, self.original_zoom)
        self.assertEqual(self.frame.image_canvas.canvas.bind("<Configure>"), self.canvas_binding)
        self.assertEqual(self.frame.image_canvas.canvas.find_withtag("side_indicator"), ())
        self.assertEqual(self._buttons(), {})
        for cue in self.cues:
            self.assertIsNone(cue._motion_job)
        self.frame.image_canvas.set_fovea_label_offset.assert_called_with(0)
        self.frame.image_canvas.set_fovea_guidance_target.assert_called_with(None)
        self.guidance_arrows[-1].destroy.assert_called_once_with()
        self.assertEqual(self.generated_events, [])
        return result

    def test_both_images_keep_themed_controls_inside_borders_without_moving_the_pointer(self):
        for scale in (1, 1.5):
            with self.subTest(scale=scale):
                ctk.set_widget_scaling(scale)
                self.root.geometry("860x580+20+20")
                self.root.update()
                self.generated_events.clear()
                stage_visits = []

                def fovea(buttons):
                    button = buttons["Set Fovea"]
                    toolbar = button.master.master.master
                    self.assertTrue(button.winfo_ismapped())
                    self._assert_inside(button, toolbar, inset=2)
                    self._assert_inside(buttons["Exit"], toolbar, inset=2)
                    self.assertEqual(resolve_color(button.cget("bg_color")),
                                     resolve_color(toolbar.cget("fg_color")))
                    self.assertFalse(buttons["Nasal"].winfo_ismapped())
                    stage_visits.append("fovea")
                    if scale == 1 and len(stage_visits) == 1:
                        self._run_timers(step2_annotate.FOVEA_GUIDE_DELAY_MS +
                                         PromptCue.DURATION_MS + 100)
                        for cue in self.cues:
                            self.assertIsNone(cue._motion_job)
                        self.assertEqual(float(button.place_info()["x"]), 0)
                        self._capture("fovea_controls_after.png")
                    self.assertEqual(self.generated_events, [])
                    # Invoking a hidden side action cannot skip fovea confirmation.
                    buttons["Nasal"].invoke()
                    button.invoke()

                def side(buttons):
                    self.assertFalse(buttons["Set Fovea"].winfo_ismapped())
                    overlay = buttons["Nasal"].master.master
                    canvas = self.frame.image_canvas.canvas
                    self.assertTrue(overlay.winfo_ismapped())
                    self._assert_inside(overlay, canvas, inset=2)
                    for name in ("Nasal", "Temporal"):
                        self._assert_inside(buttons[name], overlay, inset=2)
                        self.assertEqual(resolve_color(buttons[name].cget("bg_color")),
                                         resolve_color(overlay.cget("fg_color")))
                    stage_visits.append("side")
                    if scale == 1 and len(stage_visits) == 2:
                        self._capture("side_prompt_after.png")
                    self.root.geometry("420x440+20+20")
                    self.root.update()
                    self._assert_inside(overlay, canvas, inset=2)
                    self.frame.image_canvas._cb_zoom(1.5)
                    self.root.update()
                    self._assert_inside(overlay, canvas, inset=2)
                    self.assertEqual(self.generated_events, [])
                    buttons["Nasal" if len(stage_visits) == 2 else "Temporal"].invoke()
                    self.root.geometry("860x580+20+20")

                paths = ["C:/" + "very long folder/" * 35 + f"retina-{i}.img" for i in (1, 2)]
                result = self._collect(paths, [fovea, side, fovea, side])
                self.assertEqual(result, dict.fromkeys(paths, 125))
                self.assertEqual(stage_visits, ["fovea", "side", "fovea", "side"])
                self.assertEqual(self.generated_events, [])
                self.assertNotEqual(self.frame._collected_image_side_by_path[paths[0]],
                                    self.frame._collected_image_side_by_path[paths[1]])

    def test_fovea_label_moves_before_arrow_and_confirmation_then_resets_for_each_image(self):
        image_number = 0
        prior_arrow_count = 0
        prior_label_count = 0

        def arrow_targets():
            return [value for kind, value in self.guidance_events if kind == "arrow" and value is not None]

        def fovea(buttons):
            nonlocal image_number
            image_number += 1
            button = buttons["Set Fovea"]
            self.assertEqual(len(arrow_targets()), prior_arrow_count)
            self.frame.image_canvas.set_fovea_guidance_target.assert_called_with(None)
            self.assertEqual(float(button.place_info()["x"]), 0)
            label_calls = self.frame.image_canvas.set_fovea_label_offset.call_args_list[prior_label_count:]
            self.assertTrue(any(call.args[0] for call in label_calls))

            # Give the actual label animation time to finish before its delayed
            # arrow hands attention to the confirmation button.
            self._run_timers(step2_annotate.FOVEA_GUIDE_DELAY_MS)
            self.assertGreater(len(arrow_targets()), prior_arrow_count)
            self.assertIsInstance(arrow_targets()[-1], (int, float))
            self.frame.image_canvas.set_fovea_label_offset.assert_called_with(0)
            labels = [value for kind, value in self.guidance_events if kind == "label"]
            self.assertGreater(max(labels), 0)
            self.assertLess(min(labels), 0)
            self.assertIsNotNone(self.cues[0]._motion_job)

            if image_number == 2:
                self._run_timers(step2_annotate.FOVEA_GUIDE_DURATION_MS + PromptCue.DURATION_MS)
                self.frame.image_canvas.set_fovea_guidance_target.assert_called_with(None)
                self.assertEqual(float(button.place_info()["x"]), 0)
            self.assertEqual(self.generated_events, [])
            button.invoke()

        def side(buttons):
            nonlocal prior_arrow_count, prior_label_count
            self.frame.image_canvas.set_fovea_guidance_target.assert_called_with(None)
            self.frame.image_canvas.set_fovea_label_offset.assert_called_with(0)
            prior_arrow_count = len(arrow_targets())
            prior_label_count = self.frame.image_canvas.set_fovea_label_offset.call_count
            self.assertEqual(self.generated_events, [])
            buttons["Nasal"].invoke()

        with mock.patch.object(step2_annotate, "FOVEA_GUIDE_DURATION_MS", 300):
            paths = ["C:/retina-1.img", "C:/retina-2.img"]
            self.assertEqual(self._collect(paths, [fovea, side, fovea, side]),
                             dict.fromkeys(paths, 125))
        self.assertEqual(image_number, 2)

    def test_guidance_arrow_ends_below_the_confirmation_button_at_each_scale(self):
        for scale in (1, 1.5):
            with self.subTest(scale=scale):
                ctk.set_widget_scaling(scale)
                self.root.geometry("860x580+20+20")
                self.root.update()

                def fovea(buttons):
                    self._run_timers(step2_annotate.FOVEA_GUIDE_DELAY_MS)
                    renderer = self.guidance_renderer
                    self.assertTrue(callable(renderer))
                    button = buttons["Set Fovea"]
                    canvas = self.frame.image_canvas.canvas
                    arrow = self.guidance_arrows[-1]

                    for delta in (0, 25):
                        # Moving the window and marker must retain the small
                        # gap below the actual button, centered horizontally.
                        self.root.geometry(f"860x580+{20 + delta}+{20 + delta}")
                        self.root.update_idletasks()
                        origin = (canvas.winfo_rootx() + 160 + delta,
                                  canvas.winfo_rooty() + 55 + delta)
                        renderer(origin)
                        start, end = arrow.show.call_args.args[:2]
                        self.assertEqual(start, origin)
                        self.assertEqual(end[0], button.winfo_rootx() + button.winfo_width() / 2)
                        button_bottom = button.winfo_rooty() + button.winfo_height()
                        self.assertGreater(end[1], button_bottom)
                        self.assertAlmostEqual(end[1] - button_bottom,
                                               4 * button._get_widget_scaling(), delta=1)
                        self.assertLess(end[1], canvas.winfo_rooty())

                    arrow.hide.reset_mock()
                    renderer(None)
                    arrow.hide.assert_called_once_with()
                    self.assertEqual(self.generated_events, [])
                    button.invoke()

                def side(buttons):
                    self.assertIsNone(self.guidance_renderer)
                    self.assertEqual(self.generated_events, [])
                    buttons["Nasal"].invoke()

                self.assertEqual(self._collect(["C:/retina.img"], [fovea, side]),
                                 {"C:/retina.img": 125})

    def test_cancellation_removes_queued_layout_and_cue_callbacks_in_either_stage(self):
        for cancel_stage in ("fovea", "side"):
            with self.subTest(stage=cancel_stage):
                removed_commands = []

                def cancel(buttons):
                    toolbar = buttons["Set Fovea"].master.master.master
                    if cancel_stage == "fovea":
                        # Exit while the delayed arrow and button animation are
                        # active, so its future hide callback must be removed.
                        self._run_timers(step2_annotate.FOVEA_GUIDE_DELAY_MS)
                        self.assertIsNotNone(
                            self.frame.image_canvas.set_fovea_guidance_target.call_args.args[0],
                        )
                    # Queue layout work, then exit before the idle callback runs.
                    self.frame.image_canvas._cb_zoom(1.1)
                    removed_commands.extend(toolbar._tclCommands or [])
                    buttons["Exit"].invoke()

                stages = [cancel] if cancel_stage == "fovea" else [
                    lambda buttons: buttons["Set Fovea"].invoke(), cancel,
                ]
                self.assertIsNone(self._collect(["C:/retina.img"], stages))
                self.assertEqual(self.frame._collected_image_side_by_path, {})
                scripts = [str(self.root.tk.call("after", "info", job))
                           for job in self.root.tk.call("after", "info")]
                for command in removed_commands:
                    self.assertFalse(any(command in script for script in scripts), command)


if __name__ == "__main__":
    unittest.main()
