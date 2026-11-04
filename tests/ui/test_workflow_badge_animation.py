"""Completion badges stay legible while short, interruptible cues run."""

from contextlib import ExitStack
import tkinter as tk
import unittest
from unittest import mock

import customtkinter as ctk
from PIL import ImageTk

from aidas.ui.components import WorkflowProgressStrip
from aidas.ui.theme import (
    COLOR_PAIRS, apply_appearance_mode, get_interface_mode, resolve_color,
    set_interface_mode,
)


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class WorkflowBadgeAnimationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_interface_mode, get_interface_mode(), redraw=False)
        appearance = (
            "System" if ctk.AppearanceModeTracker.appearance_mode_set_by == "system"
            else ctk.get_appearance_mode()
        )
        self.addCleanup(ctk.set_appearance_mode, appearance)
        set_interface_mode("Modern", redraw=False)
        ctk.set_appearance_mode("Light")
        try:
            self.root = _QuietRoot()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.addCleanup(self.root.destroy)
        self.root.attributes("-alpha", 0)
        self.root.geometry("840x160+0+0")
        self.strip = WorkflowProgressStrip(self.root)
        self.strip.pack(fill="x")
        self.strip.set_workflow_state({0}, 1)
        self.root.update()

    def test_status_icons_and_badge_geometry_stay_fixed_during_the_pulse(self):
        circle = self.strip.circles[0]
        self.assertEqual(circle.itemcget(circle.text_id, "text"), "\u2713")
        self.assertEqual(circle.itemcget(circle.text_id, "state"), "hidden")
        self.assertEqual(circle.type(circle._status_icon_id), "image")
        self.assertNotEqual(circle.itemcget(circle._status_icon_id, "state"), "hidden")
        image = circle._status_photo
        self.assertIs(image.tk, self.root.tk)
        alpha = ImageTk.getimage(image).getchannel("A")
        self.assertTrue(any(alpha.histogram()[1:255]))
        original = (
            circle.find_all(), circle.itemcget(circle.text_id, "font"),
            circle.coords(circle.oval_id), circle.coords(circle._status_icon_id),
            circle.winfo_width(), circle.winfo_height(),
        )
        base_fill = circle.itemcget(circle.oval_id, "fill")
        fills = set()
        for amount in (0, 0.08, 0.16, 0.08, 0):
            circle.set_completion_pulse(amount)
            fills.add(circle.itemcget(circle.oval_id, "fill"))
            self.assertIs(circle._status_photo, image)
            self.assertEqual((
                circle.find_all(), circle.itemcget(circle.text_id, "font"),
                circle.coords(circle.oval_id), circle.coords(circle._status_icon_id),
                circle.winfo_width(), circle.winfo_height(),
            ), original)
        self.assertGreater(len(fills), 1)
        self.assertEqual(circle.itemcget(circle.oval_id, "fill"), base_fill)

        self.strip.set_workflow_state(set(), 0, 0)
        self.assertEqual(circle.itemcget(circle.text_id, "text"), "\u00d7")
        self.assertEqual(circle.itemcget(circle.text_id, "state"), "hidden")
        self.strip.set_workflow_state(set(), 1)
        self.assertEqual(circle.itemcget(circle.text_id, "text"), "1")
        self.assertNotEqual(circle.itemcget(circle.text_id, "state"), "hidden")
        self.assertEqual(circle.itemcget(circle._status_icon_id, "state"), "hidden")

    def test_status_icon_colors_follow_the_theme_without_changing_completion(self):
        circle = self.strip.circles[0]
        image_item = circle._status_icon_id
        for appearance in ("Dark", "Light", "Dark"):
            with self.subTest(appearance=appearance):
                apply_appearance_mode(appearance, root=self.root)
                self.root.update()
                color = resolve_color(COLOR_PAIRS["on_primary"], appearance)
                expected = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
                image = ImageTk.getimage(circle._status_photo)
                pixels = (image.getpixel((x, y))
                          for y in range(image.height) for x in range(image.width))
                self.assertEqual({pixel[:3] for pixel in pixels if pixel[3] >= 240}, {expected})
                self.assertEqual(circle._status_icon_id, image_item)
                self.assertEqual(circle.itemcget(circle.text_id, "text"), "\u2713")
                self.assertEqual(self.strip.workflow_progress_label.cget("text"),
                                 "Step 2 of 5: Segment")

    def test_repeated_normalized_workflow_state_does_not_repaint_the_strip(self):
        with ExitStack() as patches:
            circles = [patches.enter_context(mock.patch.object(
                circle, "update_state", wraps=circle.update_state,
            )) for circle in self.strip.circles]
            label = patches.enter_context(mock.patch.object(
                self.strip.workflow_progress_label, "configure",
                wraps=self.strip.workflow_progress_label.configure,
            ))
            for _ in range(20):
                self.strip.set_workflow_state({-1, 0, 10}, 1)
        for circle in circles:
            circle.assert_not_called()
        label.assert_not_called()

    @staticmethod
    def _capture_timer(callbacks):
        def schedule(milliseconds, callback, *args):
            job = f"badge-pulse-{len(callbacks)}"
            callbacks.append((job, milliseconds, lambda: callback(*args)))
            return job
        return schedule

    def test_delayed_pulse_finishes_without_replaying_frames_or_creating_widgets(self):
        callbacks = []
        with (
            mock.patch("aidas.ui.components.time.monotonic", return_value=10) as clock,
            mock.patch.object(self.strip, "after", side_effect=self._capture_timer(callbacks)) as after,
            mock.patch.object(self.strip, "after_cancel"),
            mock.patch.object(self.strip, "update_idletasks", side_effect=AssertionError("Nested idle flush")),
            mock.patch.object(ctk, "CTkLabel", side_effect=AssertionError("Animated overlay label")),
            mock.patch.object(ctk, "CTkFont", side_effect=AssertionError("Animated symbol font")),
        ):
            self.strip.animate_completion(0)
            self.assertEqual(len(callbacks), 1)
            self.assertIsNotNone(self.strip._completion_animation_job)
            clock.return_value = 10.5
            callbacks[-1][2]()
            self.assertIsNone(self.strip._completion_animation_job)
            self.assertIsNone(self.strip._completion_animation_circle)
            self.assertEqual(after.call_count, 1)
            self.assertEqual(self.strip.circles[0].itemcget(
                self.strip.circles[0].oval_id, "fill",
            ), resolve_color(COLOR_PAIRS["success"]))

    def test_replacement_and_cancellation_make_old_pulse_callbacks_harmless(self):
        self.strip.set_workflow_state({0, 1}, 2)
        callbacks = []
        with (
            mock.patch("aidas.ui.components.time.monotonic", return_value=10) as clock,
            mock.patch.object(self.strip, "after", side_effect=self._capture_timer(callbacks)),
            mock.patch.object(self.strip, "after_cancel") as cancel,
        ):
            self.strip.animate_completion(0)
            clock.return_value = 10.032
            callbacks[-1][2]()
            stale_first = callbacks[-1][2]
            self.strip.animate_completion(1)
            new_job = self.strip._completion_animation_job
            stale_second = callbacks[-1][2]
            count = len(callbacks)
            stale_first()
            self.assertEqual(self.strip._completion_animation_job, new_job)
            self.assertIs(self.strip._completion_animation_circle, self.strip.circles[1])
            self.assertEqual(len(callbacks), count)
            self.strip.set_workflow_state(set(), 2, 2)
            stale_second()
            self.assertIsNone(self.strip._completion_animation_job)
            self.assertIsNone(self.strip._completion_animation_circle)
            self.assertGreaterEqual(cancel.call_count, 2)
            cancelled = self.strip.circles[2]
            self.assertEqual(cancelled.itemcget(cancelled.text_id, "text"), "\u00d7")
            self.assertEqual(cancelled.itemcget(cancelled.oval_id, "fill"),
                             resolve_color(COLOR_PAIRS["danger"]))

    def test_unfinished_steps_cannot_start_completion_pulses(self):
        with mock.patch.object(self.strip, "after") as schedule:
            for index in (1, 4, -1, 5, "invalid"):
                self.strip.animate_completion(index)
        schedule.assert_not_called()
        self.assertIsNone(self.strip._completion_animation_job)


if __name__ == "__main__":
    unittest.main()
