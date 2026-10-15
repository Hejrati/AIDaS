"""Rendered regressions for Step 2's in-canvas segmentation progress."""

import tkinter as tk
import time
from tkinter import ttk
import unittest
from unittest import mock

import customtkinter as ctk
import numpy as np

from aidas.steps.step2_annotate import BOUNDARY_NAMES, Step2Frame
from aidas.ui.theme import (
    COLOR_PAIRS,
    apply_appearance_mode,
    get_interface_mode,
    set_interface_mode,
)


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


def _bounds(widget):
    return (
        widget.winfo_rootx(),
        widget.winfo_rooty(),
        widget.winfo_width(),
        widget.winfo_height(),
    )


class Step2CanvasProgressTests(unittest.TestCase):
    def setUp(self):
        self.original_interface = get_interface_mode()
        self.original_appearance = ctk.get_appearance_mode()
        self.original_appearance_source = ctk.AppearanceModeTracker.appearance_mode_set_by
        self.original_scaling = ctk.ScalingTracker.widget_scaling
        self.addCleanup(self._restore_appearance)
        set_interface_mode("Modern", redraw=False)
        ctk.set_appearance_mode("Light")
        ctk.set_widget_scaling(1)
        try:
            self.root = ctk.CTk()
            self.root.withdraw()
            self.root.attributes("-alpha", 0)
            self.root.geometry("1024x680+10000+10000")
        except tk.TclError as exc:
            self.skipTest(f"A transparent Tk test window is unavailable: {exc}")
        self.addCleanup(self._destroy_root)
        self.probe_patch = mock.patch.object(Step2Frame, "_start_ai_device_probe")
        self.probe_patch.start()
        self.addCleanup(self.probe_patch.stop)
        self.root.deiconify()
        self.image = np.zeros((32, 96), dtype=np.uint16)
        self.frame = None

    def _destroy_root(self):
        # CTk schedules appearance and DPI checks on each interpreter. Cancel
        # those before the next test creates a fresh transparent root.
        for job in self.root.tk.call("after", "info"):
            self.root.tk.call("after", "cancel", job)
        self.root.destroy()

    def _settle_events(self, expected_size=None):
        # Native window geometry and the canvas resize debounce settle on
        # different event-loop turns. Let both finish before comparing bounds.
        self.root.update()
        if expected_size is not None:
            # Windows can acknowledge geometry() after the usual debounce.
            # Never snapshot the preceding window size as the expected canvas.
            deadline = time.monotonic() + 5
            while (self.root.winfo_width(), self.root.winfo_height()) != expected_size:
                if time.monotonic() >= deadline:
                    self.fail(f"Window did not reach requested size {expected_size}")
                resized = tk.BooleanVar(master=self.root, value=False)
                self.root.after(20, lambda: resized.set(True))
                self.root.wait_variable(resized)
                self.root.update()
        settled = tk.BooleanVar(master=self.root, value=False)
        self.root.after(150, lambda: settled.set(True))
        self.root.wait_variable(settled)
        self.root.update()

    def _restore_appearance(self):
        ctk.set_widget_scaling(self.original_scaling)
        set_interface_mode(self.original_interface, redraw=False)
        ctk.set_appearance_mode(
            "System" if self.original_appearance_source == "system" else self.original_appearance
        )

    def _make_frame(self, interface="Modern", appearance="Light", scaling=1):
        if self.frame is not None and self.frame.winfo_exists():
            self.frame.destroy()
        set_interface_mode(interface, redraw=False)
        ctk.set_appearance_mode(appearance)
        ctk.set_widget_scaling(scaling)
        apply_appearance_mode(appearance, root=self.root, style=ttk.Style(self.root))
        self.frame = Step2Frame(
            self.root,
            preferences={"show_step2_annotation_tutorial": False},
        )
        self.frame.pack(fill="both", expand=True)
        self.frame._show_image(self.image, "Step 1 output")
        self._settle_events()
        return self.frame

    def _assert_progress_contained(self, frame):
        card = frame.segmentation_progress_card
        surface = frame.segmentation_progress_surface
        viewport = frame.canvas_area
        self.assertTrue(surface.winfo_ismapped())
        self.assertTrue(card.winfo_ismapped())
        self.assertEqual(_bounds(surface), _bounds(viewport))
        left, top, width, height = _bounds(card)
        view_left, view_top, view_width, view_height = _bounds(viewport)
        self.assertGreater(width, 100)
        self.assertGreater(height, 20)
        self.assertGreaterEqual(left, view_left)
        self.assertGreaterEqual(top, view_top)
        self.assertLessEqual(left + width, view_left + view_width)
        self.assertLessEqual(top + height, view_top + view_height)
        self.assertIs(card.master, surface)
        self.assertIs(surface.master, frame.canvas_area)
        self.assertEqual(surface.cget("fg_color"), COLOR_PAIRS["canvas"])
        self.assertIs(frame.canvas_area.winfo_children()[-1], surface)
        scaling = card._get_widget_scaling()
        self.assertEqual((width, height), (round(420 * scaling), round(140 * scaling)))
        self.assertGreater(card.cget("corner_radius"), 0)
        self.assertGreater(frame.segmenter_progress.cget("corner_radius"), 0)

    def test_geometry_wait_handles_a_delayed_native_window_resize(self):
        frame = self._make_frame("Classic", "Light")
        self.root.geometry("1800x1000+10000+10000")
        self._settle_events(expected_size=(1800, 1000))
        self.root.after(250, lambda: self.root.geometry("1280x820+10000+10000"))
        self._settle_events(expected_size=(1280, 820))
        previous_geometry = _bounds(frame.image_canvas.canvas)
        frame._set_segmentation_running(True, animate_progress=False)
        self.root.update()
        self._assert_progress_contained(frame)
        self.assertEqual(_bounds(frame.image_canvas.canvas), previous_geometry)
        frame._set_segmentation_running(False)
        self.root.update()
        self.assertEqual(_bounds(frame.image_canvas.canvas), previous_geometry)

    def test_progress_appears_inside_canvas_without_resizing_image_or_sidebar(self):
        for interface, appearance in (
            ("Modern", "Light"), ("Modern", "Dark"), ("Classic", "Light"),
        ):
            frame = self._make_frame(interface, appearance)
            self.assertFalse(frame.segmentation_progress_card.winfo_ismapped())
            self.assertFalse(any(
                isinstance(widget, (ctk.CTkProgressBar, ttk.Progressbar))
                for widget in _descendants(frame.ctrl)
            ))
            for width, height in ((1800, 1000), (1280, 820), (1024, 680)):
                with self.subTest(interface=interface, appearance=appearance, size=(width, height)):
                    self.root.geometry(f"{width}x{height}+10000+10000")
                    self._settle_events(expected_size=(width, height))
                    previous_geometry = _bounds(frame.image_canvas.canvas)
                    frame._set_segmentation_running(
                        True, status_message="Preparing segmentation…", animate_progress=False,
                    )
                    self.root.update()
                    self._assert_progress_contained(frame)
                    self.assertEqual(_bounds(frame.image_canvas.canvas), previous_geometry)
                    self.assertIn("Preparing", frame.segmenter_progress_var.get())
                    frame._set_segmentation_running(False)
                    self.root.update()
                    self.assertFalse(frame.segmentation_progress_card.winfo_ismapped())
                    self.assertFalse(frame.segmentation_progress_surface.winfo_ismapped())
                    self.assertEqual(frame.segmentation_progress_card.winfo_manager(), "")
                    self.assertEqual(frame.segmentation_progress_surface.winfo_manager(), "")
                    self.assertEqual(_bounds(frame.image_canvas.canvas), previous_geometry)

            with self.subTest(interface=interface, appearance=appearance, scaling=1.5):
                self.root.geometry("1800x1000+10000+10000")
                ctk.set_widget_scaling(1.5)
                frame._sync_scaled_layout_values()
                frame._apply_workspace_layout()
                self._settle_events(expected_size=(1800, 1000))
                frame._set_segmentation_running(True, animate_progress=False)
                self.root.update()
                self._assert_progress_contained(frame)
                frame._set_segmentation_running(False)
                ctk.set_widget_scaling(1)

    def test_progress_keeps_fixed_geometry_during_stage_updates_and_image_redraws(self):
        frame = self._make_frame()
        frame._set_segmentation_running(True, progress_max=12, animate_progress=False)
        self._settle_events()
        self._assert_progress_contained(frame)
        card = frame.segmentation_progress_card
        surface = frame.segmentation_progress_surface
        original_card_bounds = _bounds(card)
        original_bar_bounds = _bounds(frame.segmenter_progress)
        geometry_events = []
        for widget in (surface, card, frame.segmenter_progress):
            for event_name in ("<Configure>", "<Map>", "<Unmap>"):
                widget.bind(
                    event_name,
                    lambda event, name=event_name: geometry_events.append((event.widget, name)),
                    add="+",
                )

        for index in range(1, 13):
            frame._set_segmentation_status(
                f"AI segmentation: image {index} of 12 - " + "tracing a long boundary stage " * 8
            )
            frame._set_segmenter_progress_value(index, 12, smooth=False)
            # Worker previews used to redraw the canvas beneath the card and
            # repeatedly place/configure it, making the entire card flash.
            frame._show_image(self.image + index, f"image {index}")
            frame._cancel_progress_animation()
            frame._animate_progress_bar()
            self.root.update()
            self._assert_progress_contained(frame)
            self.assertEqual(_bounds(card), original_card_bounds)
            self.assertEqual(_bounds(frame.segmenter_progress), original_bar_bounds)

        self._settle_events()
        self.assertEqual(geometry_events, [])
        self.assertEqual(frame.segmenter_progress_percent_var.get(), "100%")
        frame._set_segmentation_running(False)
        self.root.update()
        self.assertFalse(surface.winfo_ismapped())

    def test_canvas_resize_keeps_progress_card_and_bar_the_same_size(self):
        frame = self._make_frame()
        frame._set_segmentation_running(True, animate_progress=False)
        self._settle_events()
        card = frame.segmentation_progress_card
        original_card_size = _bounds(card)[2:]
        original_bar_size = _bounds(frame.segmenter_progress)[2:]
        visibility_events = []
        for event_name in ("<Map>", "<Unmap>"):
            card.bind(event_name, lambda event: visibility_events.append(event.type), add="+")
        for width, height in ((1280, 820), (1800, 1000), (1024, 680)):
            with self.subTest(size=(width, height)):
                self.root.geometry(f"{width}x{height}+10000+10000")
                self._settle_events(expected_size=(width, height))
                self._assert_progress_contained(frame)
                self.assertEqual(_bounds(card)[2:], original_card_size)
                self.assertEqual(_bounds(frame.segmenter_progress)[2:], original_bar_size)
        self.assertEqual(visibility_events, [])
        frame._set_segmentation_running(False)

    def test_batch_progress_keeps_image_units_and_stage_while_smoothing(self):
        frame = self._make_frame()
        frame._set_segmentation_running(True, progress_max=12, animate_progress=False)
        frame._set_segmenter_progress_value(3, 12, smooth=False)
        self.assertAlmostEqual(frame.segmenter_progress.get(), 0.25)
        self.assertIn("25", frame.segmenter_progress_percent_var.get())
        for name in BOUNDARY_NAMES:
            frame.boundary_completion_vars[name].set(True)
        frame._update_boundary_progress_bar()
        self.assertEqual(frame._segmenter_progress_maximum, 12)
        self.assertAlmostEqual(frame.segmenter_progress.get(), 0.25)

        frame._set_segmenter_progress_value(9, 12, smooth=True)
        self.assertGreaterEqual(frame.segmenter_progress.get(), 0.25)
        self.assertLess(frame.segmenter_progress.get(), 0.75)
        for _ in range(25):
            frame._cancel_progress_animation()
            frame._animate_progress_bar()
        self.assertGreater(frame.segmenter_progress.get(), 0.70)
        self.assertLessEqual(frame.segmenter_progress.get(), 0.75)
        frame._set_segmentation_status("AI segmentation: Light.img - tracing boundaries")
        stage_text = frame.segmenter_progress_var.get()
        frame._on_mouse_moved(12, 8, 0)
        frame._on_canvas_zoom_changed(2)
        self.assertEqual(frame.segmenter_progress_var.get(), stage_text)
        frame._set_segmentation_running(False)
        self.root.update()
        self.assertFalse(frame.segmentation_progress_card.winfo_ismapped())
        self.assertFalse(frame.segmentation_progress_surface.winfo_ismapped())
        self.assertIsNone(frame._progress_animation_job)

    def test_progress_stays_over_batch_results_when_the_active_tab_changes(self):
        frame = self._make_frame()
        trace = {"RPE": {"points": [(0, 16), (95, 16)], "color": "#ffb703"}}
        with mock.patch.object(
            frame, "_read_image_for_annotation", return_value=(self.image, None, False),
        ):
            frame._open_aidas_batch_results_viewer([
                {"input": "first.img", "traces": trace},
                {"input": "second.img", "traces": trace},
            ])
        self.root.update()
        frame._set_segmentation_running(True, progress_max=1)
        self.root.update()
        self._assert_progress_contained(frame)
        original_bounds = _bounds(frame.segmentation_progress_card)
        visibility_events = []
        for event_name in ("<Map>", "<Unmap>"):
            frame.segmentation_progress_card.bind(
                event_name, lambda event: visibility_events.append(event.type), add="+",
            )
        frame._set_segmenter_progress_value(0.45, 1, smooth=False)

        for tab_id in frame.batch_results_notebook.tabs():
            with self.subTest(tab=tab_id):
                frame.batch_results_notebook.select(tab_id)
                self.root.update()
                self.assertIs(frame.image_canvas, frame._batch_result_tab_canvases[tab_id])
                self._assert_progress_contained(frame)
                self.assertEqual(_bounds(frame.segmentation_progress_card), original_bounds)
                self.assertEqual(frame._segmenter_progress_maximum, 1)
                self.assertAlmostEqual(frame.segmenter_progress.get(), 0.45)
        self.assertEqual(visibility_events, [])
        frame._set_segmentation_running(False)
        self.root.update()
        self.assertFalse(frame.segmentation_progress_card.winfo_ismapped())
        self.assertFalse(frame.segmentation_progress_surface.winfo_ismapped())

    def test_failure_and_destroy_clear_the_overlay_and_animation_callbacks(self):
        frame = self._make_frame()
        frame._set_segmentation_running(True, progress_max=1)
        self.root.update()
        self.assertTrue(frame.segmentation_progress_card.winfo_ismapped())
        with mock.patch("aidas.steps.step2_annotate.messagebox.showerror") as showerror:
            frame._on_current_image_segmentation_done({"exception": "model error"}, {})
        self.root.update()
        showerror.assert_called_once()
        self.assertFalse(frame.segmentation_progress_card.winfo_ismapped())
        self.assertFalse(frame.segmentation_progress_surface.winfo_ismapped())
        self.assertIsNone(frame._progress_animation_job)

        frame._set_segmentation_running(True, progress_max=1)
        frame._set_segmenter_progress_value(0.6, 1, smooth=True)
        self.root.update()
        job = frame._progress_animation_job
        self.assertIsNotNone(job)
        self.assertIn(job, self.root.tk.call("after", "info"))
        frame.destroy()
        self.assertNotIn(job, self.root.tk.call("after", "info"))


if __name__ == "__main__":
    unittest.main()
