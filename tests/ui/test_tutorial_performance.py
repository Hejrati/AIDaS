"""Regression coverage for tutorial resize work and reusable controls."""

from __future__ import annotations

from contextlib import ExitStack
import math
import sys
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.ui.tutorial import (
    TUTORIAL_PAGES,
    TutorialDialog,
    _TextMeasurements,
    _WorkflowPipeline,
)


class TutorialMeasurementCacheTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def native_height(self, font, copies, width):
        label = tk.Label(
            self.root, font=font, wraplength=width,
            padx=0, pady=0, borderwidth=0, highlightthickness=0,
        )
        try:
            heights = []
            for copy in copies:
                label.configure(text=copy)
                heights.append(label.winfo_reqheight())
            return max(heights) + 8
        finally:
            label.destroy()

    def test_native_height_matches_long_filenames_and_explicit_newlines(self):
        font = ctk.CTkFont(size=14)
        measurements = _TextMeasurements(font, self.root)
        copies = (
            "_thickness_vs_distance_from_fovea_LIGHT.txt" * 3,
            "First line\nA second line with several words that must wrap.\n\nFinal line",
        )
        for width in (105, 180, 570):
            with self.subTest(width=width):
                self.assertEqual(
                    measurements.copy_height(font, copies, width),
                    self.native_height(font, copies, width),
                )
        self.assertFalse(measurements._label.winfo_manager())

    def test_changed_font_signature_remeasures_copy_and_line_height(self):
        font = ctk.CTkFont(size=12)
        label = ctk.CTkLabel(self.root, font=font)
        pool = {}
        copies = ("one two three four five six seven",)
        original = _WorkflowPipeline._copy_height(label, copies, 140, measurements=pool)
        measurements = next(iter(pool.values()))
        with mock.patch.object(
            measurements._label, "configure", wraps=measurements._label.configure,
        ) as configure:
            self.assertEqual(
                _WorkflowPipeline._copy_height(label, copies, 140, measurements=pool),
                original,
            )
            configure.assert_not_called()

        font.configure(size=24)
        changed = _WorkflowPipeline._copy_height(label, copies, 140, measurements=pool)

        self.assertEqual(changed, self.native_height(font, copies, 140))
        self.assertGreater(changed, original)
        self.assertEqual(len(pool), 2)

    def test_scaled_text_uses_the_rendered_font_and_invalidates_cached_layouts(self):
        font = ctk.CTkFont(size=14)
        label = ctk.CTkLabel(self.root, font=font)
        copies = ("Long instructions with several filenames and words to wrap. " * 4,)
        pool = {}
        for scaling in (1.0, 1.25, 1.5, 2.0):
            with (
                self.subTest(scaling=scaling),
                mock.patch.object(label, "_get_widget_scaling", return_value=scaling),
            ):
                actual = _WorkflowPipeline._copy_height(label, copies, 247, measurements=pool)
                physical_height = self.native_height(
                    font.create_scaled_tuple(scaling), copies, 247 * scaling,
                ) - 8
                self.assertEqual(actual, math.ceil(physical_height / scaling) + 8)
        self.assertEqual(len(pool), 4)

    def test_height_cache_stays_bounded_and_evicted_widths_remain_correct(self):
        font = ctk.CTkFont(size=14)
        measurements = _TextMeasurements(font, self.root)
        copies = ("aa aa aa aa",)
        original = measurements.copy_height(font, copies, 50)
        self.assertEqual(original, self.native_height(font, copies, 50))

        for width in range(80, 80 + measurements.MAX_HEIGHTS + 5):
            self.assertEqual(
                measurements.copy_height(font, copies, width),
                self.native_height(font, copies, width),
            )
            self.assertLessEqual(len(measurements._heights), measurements.MAX_HEIGHTS)

        self.assertNotIn((copies, 50), measurements._heights)
        self.assertEqual(measurements.copy_height(font, copies, 50), original)
        self.assertIn((copies, 50), measurements._heights)
        self.assertEqual(len(measurements._heights), measurements.MAX_HEIGHTS)
        with mock.patch.object(
            measurements._label, "configure", wraps=measurements._label.configure,
        ) as configure:
            self.assertEqual(measurements.copy_height(font, copies, 50), original)
            configure.assert_not_called()


class TutorialResizeSchedulingTests(unittest.TestCase):
    def make_dialog(self):
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._closing = False
        dialog._content_resize_after_id = None
        dialog._pending_content_width = None
        dialog._content_viewport_width = 650
        dialog._content_wrap = 600
        dialog._flow_visualizer = mock.Mock()
        dialog._wrapped_labels = [(mock.Mock(), 20)]
        dialog._overview_context_labels = [mock.Mock()]
        dialog._size_page_headings = mock.Mock()
        dialog.after = mock.Mock(return_value="resize-job")
        return dialog

    def test_resize_burst_schedules_once_and_applies_the_latest_width(self):
        dialog = self.make_dialog()
        label = dialog._wrapped_labels[0][0]
        with mock.patch(
            "aidas.ui.tutorial.logical_window_size",
            side_effect=lambda _window, width, height: (width // 2, height),
        ):
            for width in (1200, 1280, 1360):
                dialog._on_content_resize(SimpleNamespace(width=width))

        dialog.after.assert_called_once_with(40, dialog._apply_content_resize)
        self.assertEqual(dialog._pending_content_width, 680)
        dialog._flow_visualizer.set_available_width.assert_not_called()
        label.configure.assert_not_called()
        dialog._size_page_headings.assert_not_called()

        dialog._apply_content_resize()

        self.assertIsNone(dialog._content_resize_after_id)
        self.assertIsNone(dialog._pending_content_width)
        self.assertEqual(dialog._content_viewport_width, 680)
        self.assertEqual(dialog._content_wrap, 630)
        dialog._flow_visualizer.set_available_width.assert_called_once_with(656)
        label.configure.assert_called_once_with(wraplength=610)
        dialog._size_page_headings.assert_called_once_with()

        with mock.patch("aidas.ui.tutorial.logical_window_size", return_value=(700, 1)):
            dialog._on_content_resize(SimpleNamespace(width=700))
        self.assertEqual(dialog.after.call_count, 2)

    def test_close_cancels_pending_resize_and_prevents_further_scheduling(self):
        dialog = self.make_dialog()
        dialog._content_resize_after_id = "resize-job"
        dialog._pending_content_width = 680
        dialog._modal_activation_after_id = "modal-job"
        dialog._scrollbar_sync_after_id = "scrollbar-job"
        dialog._stop_flow_visualizer = mock.Mock()
        dialog._on_close_callback = mock.Mock()
        dialog.after_cancel = mock.Mock()
        dialog.grab_release = mock.Mock()
        dialog.destroy = mock.Mock()

        dialog.close()

        dialog.after_cancel.assert_any_call("resize-job")
        self.assertIsNone(dialog._content_resize_after_id)
        self.assertIsNone(dialog._pending_content_width)
        dialog.destroy.assert_called_once_with()
        dialog._on_close_callback.assert_called_once_with(dialog)

        dialog._on_content_resize(SimpleNamespace(width=700))
        dialog.after.assert_not_called()


class TutorialScrollbarUpdateTests(unittest.TestCase):
    def make_dialog(self, *, deferred=False):
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._aidas_defer_canvas_updates = deferred
        scrollbar = mock.Mock()
        scrollbar.get.return_value = (0.0, 1.0)
        dialog.content_scroll = SimpleNamespace(_scrollbar=scrollbar)
        return dialog, scrollbar

    def test_unchanged_native_fractions_do_not_redraw_the_scrollbar(self):
        dialog, scrollbar = self.make_dialog()
        with mock.patch("aidas.ui.tutorial.finish_deferred_rendering") as finish:
            dialog._update_content_scrollbar("0.0", "1.0")
            scrollbar.set.assert_not_called()
            finish.assert_not_called()
        self.assertFalse(dialog._aidas_defer_canvas_updates)

    def test_changed_fractions_draw_under_deferral_then_finish_rendering(self):
        dialog, scrollbar = self.make_dialog()

        def update_thumb(*_fractions):
            self.assertTrue(dialog._aidas_defer_canvas_updates)

        def finish_rendering(window):
            self.assertIs(window, dialog)
            window._aidas_defer_canvas_updates = False

        scrollbar.set.side_effect = update_thumb
        with mock.patch(
            "aidas.ui.tutorial.finish_deferred_rendering", side_effect=finish_rendering,
        ) as finish:
            dialog._update_content_scrollbar("0.1", "0.8")
            scrollbar.set.assert_called_once_with(0.1, 0.8)
            finish.assert_called_once_with(dialog)
        self.assertFalse(dialog._aidas_defer_canvas_updates)

    def test_scrollbar_update_preserves_outer_construction_deferral(self):
        dialog, scrollbar = self.make_dialog(deferred=True)
        with mock.patch("aidas.ui.tutorial.finish_deferred_rendering") as finish:
            dialog._update_content_scrollbar("0.1", "0.8")
            scrollbar.set.assert_called_once_with(0.1, 0.8)
            finish.assert_not_called()
        self.assertTrue(dialog._aidas_defer_canvas_updates)

    def test_failed_scrollbar_draw_releases_temporary_deferral(self):
        dialog, scrollbar = self.make_dialog()
        scrollbar.set.side_effect = tk.TclError("destroyed scrollbar")

        def finish_rendering(window):
            window._aidas_defer_canvas_updates = False

        with mock.patch(
            "aidas.ui.tutorial.finish_deferred_rendering", side_effect=finish_rendering,
        ) as finish:
            with self.assertRaises(tk.TclError):
                dialog._update_content_scrollbar("0.1", "0.8")
            finish.assert_called_once_with(dialog)
        self.assertFalse(dialog._aidas_defer_canvas_updates)


class TutorialTextAndControlReuseTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def make_guide(self, width=650, *, master=None):
        page = TUTORIAL_PAGES[1]
        return _WorkflowPipeline(
            self.root if master is None else master,
            (page.input_summary, page.function_summary, page.output_summary),
            page.stage_points,
            page.completion_check,
            page.tips,
            available_width=width,
        )

    def test_resizing_within_each_breakpoint_preserves_section_buttons(self):
        guide = self.make_guide()
        for widths in ((620, 680, 640), (320, 380, 360)):
            guide.set_available_width(widths[0])
            buttons = tuple(guide.sections._buttons_dict.values())
            for width in widths[1:]:
                guide.set_available_width(width)
                self.assertEqual(
                    tuple(guide.sections._buttons_dict.values()), buttons,
                    "Resizing within a breakpoint must retain the section buttons.",
                )

    def test_revisited_widths_reuse_native_text_layouts(self):
        guide = self.make_guide()
        for width in (620, 680):
            guide.set_available_width(width)
        with ExitStack() as stack:
            configurations = [
                stack.enter_context(mock.patch.object(
                    measurements._label, "configure",
                    wraps=measurements._label.configure,
                ))
                for measurements in guide._text_measurements.values()
            ]
            for width in (650, 620, 680, 650, 620):
                guide.set_available_width(width)
            self.assertEqual(sum(configure.call_count for configure in configurations), 0)

    def test_section_changes_preserve_visible_number_badges(self):
        guide = self.make_guide()
        with ExitStack() as stack:
            configurations = [
                stack.enter_context(mock.patch.object(
                    badge, "configure", wraps=badge.configure,
                ))
                for badge in guide._badges[:2]
            ]
            for section in guide._section_values[1:]:
                guide._section_selected(section)
            self.assertEqual(sum(configure.call_count for configure in configurations), 0)

    def test_selecting_active_section_does_not_repaint_or_reset_scrolling(self):
        guide = self.make_guide()
        with mock.patch.object(guide, "_select_section") as select:
            guide._section_selected(guide._section_values[0])
            select.assert_not_called()

    def test_closing_before_the_first_idle_leaves_no_destroyed_paint_callbacks(self):
        dialog = TutorialDialog(self.root, initial_page=1)
        self.addCleanup(dialog.close)

        dialog.close()

        for pending in self.root.tk.splitlist(self.root.tk.call("after", "info")):
            script, _kind = self.root.tk.call("after", "info", pending)
            command = self.root.tk.splitlist(script)[0]
            with self.subTest(pending=pending, command=command):
                self.assertTrue(
                    self.root.tk.call("info", "commands", command),
                    "Closing the tutorial must cancel callbacks owned by destroyed widgets.",
                )
        self.root.update()

    @unittest.skipUnless(sys.platform.startswith("win"), "Windows caption integration")
    def test_fast_open_preserves_the_custom_caption_above_the_content(self):
        dialog = TutorialDialog(self.root, initial_page=1)
        self.addCleanup(dialog.close)
        self.root.update_idletasks()

        caption = getattr(dialog, "_aidas_dialog_title_bar", None)
        self.assertIsNotNone(caption)
        self.assertIs(dialog.pack_slaves()[0], caption)
        self.assertGreater(caption.winfo_height(), 1)
        self.assertFalse(dialog._aidas_defer_canvas_updates)

    def test_section_overflow_changes_keep_the_same_viewport_width(self):
        self.root.attributes("-alpha", 0)
        self.root.deiconify()
        with mock.patch.object(TutorialDialog, "_schedule_modal_activation"):
            dialog = TutorialDialog(self.root, initial_page=1)
        dialog.attributes("-alpha", 0)
        self.addCleanup(dialog.close)
        guide = dialog._flow_visualizer
        canvas = dialog.content_scroll._parent_canvas
        scrollbar = dialog.content_scroll._scrollbar

        def settle():
            # Apply the latest debounced width without waiting on a wall clock.
            for _ in range(6):
                self.root.update()
                pending = dialog._content_resize_after_id
                if pending is not None:
                    dialog.after_cancel(pending)
                    dialog._apply_content_resize()
            self.root.update_idletasks()

        def select(index):
            guide._section_selected(guide._section_values[index])
            settle()
            bounds = canvas.bbox("all")
            return bounds[3] - bounds[1]

        settle()
        prepare_height = select(0)
        run_height = select(1)
        self.assertGreater(prepare_height, run_height)
        target_viewport = (prepare_height + run_height) // 2

        widths = []
        visible = []
        # Keep the native window size fixed and place only the overflow cutoff
        # between the two real section heights. This avoids depending on the
        # asynchronous timing of Windows applying a toplevel geometry request.
        with (
            mock.patch.object(canvas, "winfo_height", return_value=target_viewport),
            mock.patch.object(guide, "_size_template", wraps=guide._size_template) as size,
        ):
            for section in (0, 1, 0, 1):
                select(section)
                widths.append(canvas.winfo_width())
                visible.append(bool(scrollbar.winfo_manager()))
            self.assertEqual(visible, [True, False, True, False])
            self.assertEqual(len(set(widths)), 1)
            size.assert_not_called()

    def test_reserved_rows_fit_long_instructions_at_narrow_and_wide_widths(self):
        # The real tutorial gives its instructions an unconstrained content
        # height inside a scrollable viewport. Packing directly into a 1600px
        # toplevel lets the window manager clip the final rows on small CI
        # displays, testing pack's space allocation rather than text sizing.
        self.root.attributes("-alpha", 0)
        self.root.maxsize(1000, 480)
        viewport = ctk.CTkScrollableFrame(self.root)
        viewport.pack(fill="both", expand=True)
        guide = self.make_guide(master=viewport)
        guide.pack(fill="x")
        self.root.deiconify()
        for width in (360, 650):
            self.root.geometry(f"{width}x480")
            guide.set_available_width(width)
            overflow_seen = False
            for page in TUTORIAL_PAGES[1:]:
                self.assertTrue(guide.replace_content(
                    (page.input_summary, page.function_summary, page.output_summary),
                    page.stage_points, page.completion_check, page.tips,
                ))
                for section in guide.SECTIONS:
                    guide._select_section(section)
                    self.root.update_idletasks()
                    for row, label in guide.rows:
                        if row.winfo_manager():
                            with self.subTest(width=width, page=page.key, section=section):
                                self.assertLessEqual(
                                    label.winfo_reqheight() + 24, row.winfo_height(),
                                )
                                self.assertLessEqual(
                                    label._label.winfo_reqheight(), label.winfo_height(),
                                )
                    # Long sections must overflow by scrolling, without
                    # reducing any instruction row to fit the window height.
                    canvas = viewport._parent_canvas
                    bounds = canvas.bbox("all")
                    self.assertIsNotNone(bounds)
                    self.assertEqual(guide.winfo_height(), guide.winfo_reqheight())
                    overflow_seen |= bounds[3] - bounds[1] > canvas.winfo_height()
                    canvas.yview_moveto(1.0)
                    self.root.update_idletasks()
                    self.assertAlmostEqual(canvas.yview()[1], 1.0)
            self.assertTrue(overflow_seen, "Long sections must scroll in the constrained viewport.")


if __name__ == "__main__":
    unittest.main()
