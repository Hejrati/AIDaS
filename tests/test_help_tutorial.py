from __future__ import annotations

import inspect
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest import mock

from aidas.app import AIDaSApp
from aidas.ui.tutorial import (
    CONNECTOR_TRAVEL_MS,
    INPUT_HOLD_MS,
    OUTPUT_HOLD_MS,
    PIPELINE_CYCLE_MS,
    PROCESS_HOLD_MS,
    TUTORIAL_PAGES,
    TutorialDialog,
    _PipelineAnimationState,
    _WorkflowOverviewMap,
    _WorkflowPipeline,
    _pipeline_animation_state,
    _pipeline_orientation,
    tutorial_page_index_for_step,
)


class TutorialContentTests(unittest.TestCase):
    def test_tutorial_has_overview_and_one_ordered_page_per_workflow_step(self):
        self.assertEqual(
            [page.key for page in TUTORIAL_PAGES],
            ["overview", "step1", "step2", "step3", "step4"],
        )
        self.assertEqual(
            [page.step_index for page in TUTORIAL_PAGES],
            [None, 0, 1, 2, 3],
        )
        self.assertEqual(
            [tutorial_page_index_for_step(index) for index in range(4)],
            [1, 2, 3, 4],
        )
        self.assertEqual(tutorial_page_index_for_step(99), 0)
        self.assertEqual(tutorial_page_index_for_step("not-a-step"), 0)

    def test_every_page_has_complete_process_and_handoff_content(self):
        keys = set()
        titles = set()
        for page in TUTORIAL_PAGES:
            with self.subTest(page=page.key):
                self.assertNotIn(page.key, keys)
                self.assertNotIn(page.title, titles)
                keys.add(page.key)
                titles.add(page.title)
                self.assertTrue(page.purpose.strip())
                self.assertTrue(page.input_summary.strip())
                self.assertTrue(page.function_summary.strip())
                self.assertTrue(page.output_summary.strip())
                self.assertTrue(page.completion_check.strip())
                if page.step_index is None:
                    self.assertEqual(tuple(map(len, page.stage_points)), (0, 0, 0))
                else:
                    self.assertEqual(tuple(map(len, page.stage_points)), (4, 2, 3))
                    self.assertTrue(
                        all(
                            point.strip()
                            for group in page.stage_points
                            for point in group
                        )
                    )

    def test_step1_copy_matches_live_crop_queue_and_outputs(self):
        text = self._page_text("step1")
        for token in (
            ".sdb",
            "Width",
            "Height",
            "Offset",
            "Little-endian",
            "Crop & Scale",
            "light.hdr",
            "light.img",
            "Go to Step 2",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_step2_copy_names_every_required_boundary_and_output(self):
        text = self._page_text("step2")
        for token in (
            "RNFL-Vitreous",
            "GCL-RNFL",
            "INL-IPL",
            "ONL-OPL",
            "ELM",
            "RPE",
            "foveal",
            "nasal",
            "temporal",
            "Light_MARKED",
            "Go to Step 3",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_step3_copy_explains_runtime_inputs_scheduling_and_handoff(self):
        text = self._page_text("step3")
        for token in (
            "R 3.3.1",
            "8-bit Light_MARKED",
            "16-bit Light",
            "Batch size",
            "240 minutes",
            "Parallel",
            "Sequential",
            "_flat_LIGHT.hdr",
            "_flat_LIGHT.img",
            "_thickness_vs_distance_from_fovea_DARK.txt",
            "_thickness_vs_distance_from_fovea_LIGHT.txt",
            "Go to Step 4",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_step4_copy_explains_all_rois_measurements_and_final_outputs(self):
        text = self._page_text("step4")
        for token in (
            "21 ROIs",
            "Start",
            "End",
            "Major",
            "Minor",
            "Angle",
            "Circ.",
            "AR",
            "Round",
            "Solidity",
            "ROI_to_move_stck.tif",
            "MAX_Stack.tif",
            "rr_MCPAR.xlsx",
            "slice 0",
            "Apply changes",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    @staticmethod
    def _page_text(key):
        page = next(page for page in TUTORIAL_PAGES if page.key == key)
        return "\n".join(
            (
                page.title,
                page.purpose,
                page.input_summary,
                page.function_summary,
                page.output_summary,
                *(point for group in page.stage_points for point in group),
                page.completion_check,
                *page.tips,
            )
        )


class TutorialNavigationTests(unittest.TestCase):
    @staticmethod
    def _dialog_at(index):
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._page_index = index

        def show_page(next_index):
            dialog._page_index = max(0, min(len(TUTORIAL_PAGES) - 1, next_index))

        dialog.show_page = mock.Mock(side_effect=show_page)
        return dialog

    def test_previous_and_next_stay_inside_page_bounds(self):
        dialog = self._dialog_at(0)
        self.assertEqual(dialog._previous_page(), "break")
        dialog.show_page.assert_not_called()

        self.assertEqual(dialog._next_page(), "break")
        dialog.show_page.assert_called_once_with(1)
        self.assertEqual(dialog._page_index, 1)

        dialog._page_index = len(TUTORIAL_PAGES) - 1
        dialog.show_page.reset_mock()
        self.assertEqual(dialog._next_page(), "break")
        dialog.show_page.assert_not_called()

    def test_open_step_closes_then_invokes_the_real_navigation_callback(self):
        dialog = self._dialog_at(tutorial_page_index_for_step(2))
        calls = []
        dialog._on_step_selected = calls.append
        dialog.close = mock.Mock()

        dialog._open_current_step()

        dialog.close.assert_called_once_with()
        self.assertEqual(calls, [2])

    def test_dialog_has_scroll_keyboard_close_and_fixed_footer_actions(self):
        source = inspect.getsource(TutorialDialog)
        self.assertIn("CTkScrollableFrame", source)
        self.assertIn('self.bind("<Escape>"', source)
        self.assertIn('self.bind("<Alt-Left>"', source)
        self.assertIn('self.bind("<Alt-Right>"', source)
        self.assertIn("self.back_button", source)
        self.assertIn("self.next_button", source)
        self.assertIn("self.close_button", source)
        self.assertIn("self.open_step_button", source)
        self.assertIn("self.resizable(True, True)", source)
        self.assertIn("work_area_bounds", source)
        self.assertIn("MAX_SCREEN_FRACTION", source)
        self.assertIn("logical_window_size(self, int(event.width), 1)", source)
        self.assertIn("_schedule_modal_activation", source)
        self.assertIn("_WorkflowPipeline", source)

    def test_show_page_synchronizes_content_buttons_and_scroll_position(self):
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._page_index = 0
        dialog._render_page = mock.Mock()
        dialog.page_indicator = mock.Mock()
        dialog._navigation_buttons = [mock.Mock() for _page in TUTORIAL_PAGES]
        dialog.back_button = mock.Mock()
        dialog.next_button = mock.Mock()
        dialog.open_step_button = mock.Mock()
        dialog._schedule_scrollbar_sync = mock.Mock()
        canvas = mock.Mock()
        dialog.content_scroll = SimpleNamespace(_parent_canvas=canvas)

        dialog.show_page(tutorial_page_index_for_step(2))

        self.assertEqual(dialog.current_page.key, "step3")
        dialog._render_page.assert_called_once_with(dialog.current_page)
        dialog.page_indicator.configure.assert_called_once_with(text="  4 of 5  ")
        dialog.back_button.configure.assert_called_once_with(state="normal")
        dialog.next_button.configure.assert_called_once_with(state="normal")
        dialog.open_step_button.configure.assert_called_once_with(text="Open Step 3")
        dialog.open_step_button.grid.assert_called_once_with()
        canvas.yview_moveto.assert_called_once_with(0)
        dialog._schedule_scrollbar_sync.assert_called_once_with()

    def test_modal_activation_waits_for_visibility_before_taking_grab(self):
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._modal_activation_after_id = "old-job"
        dialog.after_cancel = mock.Mock()
        dialog.after = mock.Mock(return_value="new-job")
        dialog.deiconify = mock.Mock()

        dialog._schedule_modal_activation(delay_ms=7)

        dialog.after_cancel.assert_called_once_with("old-job")
        dialog.deiconify.assert_called_once_with()
        dialog.after.assert_called_once_with(7, dialog._activate_modal_when_visible)
        self.assertEqual(dialog._modal_activation_after_id, "new-job")

    def test_resize_converts_physical_width_before_updating_ctk_wrapping(self):
        label = mock.Mock()
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._content_wrap = 650
        dialog._content_viewport_width = 650
        dialog._wrapped_labels = [(label, 30)]
        dialog._flow_visualizer = mock.Mock()

        with mock.patch(
            "aidas.ui.tutorial.logical_window_size",
            return_value=(400, 1),
        ) as logical_size:
            dialog._on_content_resize(SimpleNamespace(width=800))
            dialog._on_content_resize(SimpleNamespace(width=800))

        self.assertEqual(
            logical_size.call_args_list,
            [mock.call(dialog, 800, 1), mock.call(dialog, 800, 1)],
        )
        self.assertEqual(dialog._content_viewport_width, 400)
        self.assertEqual(dialog._content_wrap, 350)
        label.configure.assert_called_once_with(wraplength=320)
        dialog._flow_visualizer.set_available_width.assert_called_once_with(376)

    def test_close_is_idempotent_and_releases_modal_state_once(self):
        callback = mock.Mock()
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._closing = False
        dialog._on_close_callback = callback
        dialog._modal_activation_after_id = "pending-job"
        visualizer = mock.Mock()
        dialog._flow_visualizer = visualizer
        dialog.after_cancel = mock.Mock()
        dialog.grab_release = mock.Mock()
        dialog.destroy = mock.Mock()

        dialog.close()
        dialog.close()

        dialog.after_cancel.assert_called_once_with("pending-job")
        dialog.grab_release.assert_called_once_with()
        dialog.destroy.assert_called_once_with()
        callback.assert_called_once_with(dialog)
        visualizer.stop.assert_called_once_with()
        self.assertIsNone(dialog._flow_visualizer)
        self.assertIsNone(dialog._modal_activation_after_id)

    def test_scrollbar_is_hidden_when_the_complete_page_fits(self):
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._closing = False
        dialog._scrollbar_sync_after_id = "sync-job"
        canvas = mock.Mock()
        canvas.bbox.return_value = (0, 0, 500, 420)
        canvas.winfo_height.return_value = 500
        scrollbar = mock.Mock()
        scrollbar.winfo_manager.return_value = "grid"
        dialog.content_scroll = SimpleNamespace(
            _parent_canvas=canvas,
            _scrollbar=scrollbar,
        )

        dialog._sync_content_scrollbar()

        self.assertIsNone(dialog._scrollbar_sync_after_id)
        scrollbar.grid_remove.assert_called_once_with()
        scrollbar.grid.assert_not_called()
        canvas.yview_moveto.assert_called_once_with(0)

    def test_scrollbar_returns_as_a_small_window_fallback(self):
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._closing = False
        dialog._scrollbar_sync_after_id = None
        canvas = mock.Mock()
        canvas.bbox.return_value = (0, 0, 500, 620)
        canvas.winfo_height.return_value = 500
        scrollbar = mock.Mock()
        scrollbar.winfo_manager.return_value = ""
        dialog.content_scroll = SimpleNamespace(
            _parent_canvas=canvas,
            _scrollbar=scrollbar,
        )

        dialog._sync_content_scrollbar()

        scrollbar.grid.assert_called_once_with()
        scrollbar.grid_remove.assert_not_called()
        canvas.yview_moveto.assert_not_called()


class WorkflowPipelineTests(unittest.TestCase):
    def test_orientation_switches_to_vertical_before_cards_can_clip(self):
        self.assertEqual(_pipeline_orientation(649), "vertical")
        self.assertEqual(_pipeline_orientation(650), "horizontal")
        self.assertEqual(_pipeline_orientation(900), "horizontal")

    def test_animation_state_moves_input_through_process_to_output(self):
        self.assertEqual(
            _pipeline_animation_state(0),
            _PipelineAnimationState(0, (0.0, 0.0), 0.0),
        )
        input_midpoint = _pipeline_animation_state(INPUT_HOLD_MS / 2)
        self.assertEqual(input_midpoint.active_stage, 0)
        self.assertEqual(input_midpoint.connector_progress, (0.0, 0.0))
        self.assertAlmostEqual(input_midpoint.stage_progress, 0.5)

        first_travel = _pipeline_animation_state(
            INPUT_HOLD_MS + (CONNECTOR_TRAVEL_MS / 2)
        )
        self.assertEqual(first_travel.active_stage, 0)
        self.assertAlmostEqual(first_travel.connector_progress[0], 0.5)
        self.assertEqual(first_travel.connector_progress[1], 0.0)
        self.assertEqual(first_travel.stage_progress, 1.0)

        process_start = INPUT_HOLD_MS + CONNECTOR_TRAVEL_MS
        self.assertEqual(
            _pipeline_animation_state(process_start),
            _PipelineAnimationState(1, (1.0, 0.0), 0.0),
        )
        second_travel = _pipeline_animation_state(
            process_start + PROCESS_HOLD_MS + (CONNECTOR_TRAVEL_MS / 2)
        )
        self.assertEqual(second_travel.active_stage, 1)
        self.assertEqual(second_travel.connector_progress[0], 1.0)
        self.assertAlmostEqual(second_travel.connector_progress[1], 0.5)

        output_start = (
            process_start + PROCESS_HOLD_MS + CONNECTOR_TRAVEL_MS
        )
        self.assertEqual(
            _pipeline_animation_state(output_start),
            _PipelineAnimationState(2, (1.0, 1.0), 0.0),
        )
        output_midpoint = _pipeline_animation_state(
            output_start + (OUTPUT_HOLD_MS / 2)
        )
        self.assertAlmostEqual(output_midpoint.stage_progress, 0.5)
        self.assertEqual(
            _pipeline_animation_state(PIPELINE_CYCLE_MS),
            _PipelineAnimationState(0, (0.0, 0.0), 0.0),
        )

    def test_start_schedules_only_one_animation_job(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._destroying = False
        pipeline._stopped = True
        pipeline._paused = False
        pipeline._animation_after_id = None
        pipeline.after = mock.Mock(return_value="animation-job")

        pipeline.start()
        pipeline.start()

        pipeline.after.assert_called_once_with(
            pipeline.ANIMATION_INTERVAL_MS,
            pipeline._animation_tick,
        )
        self.assertEqual(pipeline._animation_after_id, "animation-job")

    def test_hidden_animation_uses_slow_retry_without_mutating_cards(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._animation_after_id = "current-job"
        pipeline._destroying = False
        pipeline._stopped = False
        pipeline._paused = False
        pipeline._hidden_since = None
        pipeline.winfo_exists = mock.Mock(return_value=True)
        pipeline.winfo_viewable = mock.Mock(return_value=False)
        pipeline._schedule_tick = mock.Mock()
        pipeline._apply_animation_state = mock.Mock()

        with mock.patch("aidas.ui.tutorial.time.monotonic", return_value=17.5):
            pipeline._animation_tick()

        self.assertIsNone(pipeline._animation_after_id)
        self.assertEqual(pipeline._hidden_since, 17.5)
        pipeline._apply_animation_state.assert_not_called()
        pipeline._schedule_tick.assert_called_once_with(pipeline.HIDDEN_RETRY_MS)

    def test_visible_animation_tick_updates_state_and_schedules_one_successor(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._animation_after_id = "current-job"
        pipeline._destroying = False
        pipeline._stopped = False
        pipeline._paused = False
        pipeline._hidden_since = None
        pipeline._animation_started_at = 9.0
        pipeline.winfo_exists = mock.Mock(return_value=True)
        pipeline.winfo_viewable = mock.Mock(return_value=True)
        pipeline._apply_animation_state = mock.Mock()
        pipeline._schedule_tick = mock.Mock()

        with mock.patch("aidas.ui.tutorial.time.monotonic", return_value=10.0):
            pipeline._animation_tick()

        self.assertIsNone(pipeline._animation_after_id)
        pipeline._apply_animation_state.assert_called_once_with(
            _pipeline_animation_state(1000.0)
        )
        pipeline._schedule_tick.assert_called_once_with(
            pipeline.ANIMATION_INTERVAL_MS
        )

    def test_duplicate_available_width_does_not_queue_another_layout(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._viewport_logical_width = 720
        pipeline._queue_responsive_layout = mock.Mock()

        pipeline.set_available_width(720)

        pipeline._queue_responsive_layout.assert_not_called()

        pipeline.set_available_width(740)
        pipeline._queue_responsive_layout.assert_called_once_with(740)

    def test_same_responsive_width_is_a_complete_noop(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._applied_layout_width = 720
        pipeline._layout_stage_points = mock.Mock()

        pipeline._apply_responsive_layout(720)

        pipeline._layout_stage_points.assert_not_called()

    def test_duplicate_pending_width_does_not_restart_resize_debounce(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._destroying = False
        pipeline._stopped = False
        pipeline._applied_layout_width = 700
        pipeline._pending_logical_width = 720
        pipeline._resize_after_id = "resize-job"
        pipeline._cancel_after_job = mock.Mock()
        pipeline.after = mock.Mock()

        pipeline._queue_responsive_layout(720)

        pipeline._cancel_after_job.assert_not_called()
        pipeline.after.assert_not_called()

    def test_same_active_stage_only_moves_canvas_items(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._cards = [mock.Mock(), mock.Mock(), mock.Mock()]
        pipeline.motion_status = mock.Mock()
        pipeline._paused = False
        pipeline._rendered_active_stage = 1
        pipeline._rendered_paused = False
        pipeline._update_connector = mock.Mock()
        pipeline._apply_stage_reveal = mock.Mock()
        state = _PipelineAnimationState(1, (1.0, 0.4), 0.6)

        pipeline._apply_animation_state(state)

        for card in pipeline._cards:
            card.configure.assert_not_called()
        pipeline.motion_status.configure.assert_not_called()
        pipeline._apply_stage_reveal.assert_called_once_with(1, 0.6)
        self.assertEqual(
            pipeline._update_connector.call_args_list,
            [mock.call(0, 1.0), mock.call(1, 0.4)],
        )

    def test_keyboard_activation_toggles_animation_and_stops_event(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._toggle_animation = mock.Mock()

        self.assertEqual(pipeline._toggle_animation_from_keyboard(), "break")

        pipeline._toggle_animation.assert_called_once_with()

    def test_animation_control_shows_the_available_action_as_an_icon(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pause_icon = object()
        play_icon = object()
        pipeline._pause_animation_icon = pause_icon
        pipeline._play_animation_icon = play_icon
        pipeline.animation_button = mock.Mock()
        pipeline._animation_tooltip = SimpleNamespace(text="")

        pipeline._paused = False
        pipeline._sync_animation_control()
        self.assertIs(pipeline.animation_button.image, pause_icon)
        pipeline.animation_button.configure.assert_called_with(image=pause_icon)
        self.assertEqual(pipeline._animation_tooltip.text, "Pause animation")

        pipeline.animation_button.reset_mock()
        pipeline._paused = True
        pipeline._sync_animation_control()
        self.assertIs(pipeline.animation_button.image, play_icon)
        pipeline.animation_button.configure.assert_called_with(image=play_icon)
        self.assertEqual(pipeline._animation_tooltip.text, "Play animation")

    def test_stage_reveal_raises_only_the_active_group_and_reveals_its_points(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._stage_group_frames = [mock.Mock() for _index in range(3)]
        pipeline._stage_point_widgets = [
            [(mock.Mock(), mock.Mock(), mock.Mock()) for _index in range(count)]
            for count in (4, 2, 3)
        ]
        pipeline._rendered_reveal = None

        pipeline._apply_stage_reveal(0, 0.30)

        pipeline._stage_group_frames[0].tkraise.assert_called_once_with()
        pipeline._stage_group_frames[1].tkraise.assert_not_called()
        input_widgets = pipeline._stage_point_widgets[0]
        self.assertEqual(
            [widget[0].configure.call_count for widget in input_widgets],
            [1, 1, 1, 1],
        )
        self.assertIn("fg_color", input_widgets[0][0].configure.call_args.kwargs)
        self.assertIn("fg_color", input_widgets[1][0].configure.call_args.kwargs)
        self.assertTrue(
            all(
                "font" not in label.configure.call_args.kwargs
                for _point_card, _badge, label in input_widgets
            )
        )
        self.assertEqual(pipeline._rendered_reveal, (0, 2))

        for point_card, badge, label in input_widgets:
            point_card.reset_mock()
            badge.reset_mock()
            label.reset_mock()
        pipeline._apply_stage_reveal(0, 0.31)
        self.assertTrue(
            all(point_card.configure.call_count == 0 for point_card, _b, _l in input_widgets)
        )

    def test_animation_icon_renderer_creates_crisp_transparent_glyphs(self):
        for kind in ("play", "pause"):
            icon = _WorkflowPipeline._render_animation_icon(kind, "#1565C0")
            self.assertEqual(icon.mode, "RGBA")
            self.assertEqual(icon.size, (72, 72))
            alpha_min, alpha_max = icon.getchannel("A").getextrema()
            self.assertEqual(alpha_min, 0)
            self.assertEqual(alpha_max, 255)

        with self.assertRaisesRegex(ValueError, "Unknown animation icon"):
            _WorkflowPipeline._render_animation_icon("stop", "#1565C0")

    def test_stop_cancels_animation_and_resize_jobs_idempotently(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._stopped = False
        pipeline._animation_after_id = "animation-job"
        pipeline._resize_after_id = "resize-job"
        pipeline.after_cancel = mock.Mock()

        pipeline.stop()
        pipeline.stop()

        self.assertTrue(pipeline._stopped)
        self.assertIsNone(pipeline._animation_after_id)
        self.assertIsNone(pipeline._resize_after_id)
        self.assertEqual(
            pipeline.after_cancel.call_args_list,
            [mock.call("animation-job"), mock.call("resize-job")],
        )

    def test_cancel_job_clears_id_even_when_tk_rejects_cancellation(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._animation_after_id = "stale-job"
        pipeline.after_cancel = mock.Mock(side_effect=tk.TclError("gone"))

        pipeline._cancel_after_job("_animation_after_id")

        self.assertIsNone(pipeline._animation_after_id)
        pipeline.after_cancel.assert_called_once_with("stale-job")

    def test_page_replacement_stops_the_previous_visualizer(self):
        dialog = TutorialDialog.__new__(TutorialDialog)
        visualizer = mock.Mock()
        dialog._flow_visualizer = visualizer

        dialog._stop_flow_visualizer()
        dialog._stop_flow_visualizer()

        visualizer.stop.assert_called_once_with()
        self.assertIsNone(dialog._flow_visualizer)

    def test_adjacent_detailed_pages_reuse_the_existing_pipeline(self):
        dialog = TutorialDialog.__new__(TutorialDialog)
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline.replace_content = mock.Mock(return_value=True)
        dialog._flow_visualizer = pipeline
        dialog._page_title_label = mock.Mock()
        dialog._page_purpose_label = mock.Mock()
        dialog._stop_flow_visualizer = mock.Mock()
        page = TUTORIAL_PAGES[3]

        dialog._render_page(page)

        pipeline.replace_content.assert_called_once_with(
            (
                page.input_summary,
                page.function_summary,
                page.output_summary,
            ),
            page.stage_points,
            page.completion_check,
            page.tips,
        )
        dialog._page_title_label.configure.assert_called_once_with(text=page.title)
        dialog._page_purpose_label.configure.assert_called_once_with(text=page.purpose)
        dialog._stop_flow_visualizer.assert_not_called()

    def test_replacing_pipeline_copy_does_not_rebuild_its_layout(self):
        pipeline = _WorkflowPipeline.__new__(_WorkflowPipeline)
        pipeline._summary_labels = [mock.Mock() for _index in range(3)]
        pipeline._stage_point_widgets = [
            [(mock.Mock(), mock.Mock(), mock.Mock()) for _index in range(count)]
            for count in (4, 2, 3)
        ]
        pipeline._completion_label = mock.Mock()
        pipeline._tips_label = mock.Mock()
        pipeline._cancel_after_job = mock.Mock()
        pipeline._sync_animation_control = mock.Mock()
        pipeline._apply_animation_state = mock.Mock()
        pipeline.start = mock.Mock()
        page = TUTORIAL_PAGES[2]

        with mock.patch("aidas.ui.tutorial.time.monotonic", return_value=10.0):
            replaced = pipeline.replace_content(
                (
                    page.input_summary,
                    page.function_summary,
                    page.output_summary,
                ),
                page.stage_points,
                page.completion_check,
                page.tips,
            )

        self.assertTrue(replaced)
        pipeline._cancel_after_job.assert_called_once_with("_animation_after_id")
        pipeline._apply_animation_state.assert_called_once_with(
            _pipeline_animation_state(0.0)
        )
        pipeline.start.assert_called_once_with()
        for widgets in pipeline._stage_point_widgets:
            for point_card, badge, label in widgets:
                point_card.grid.assert_not_called()
                badge.grid.assert_not_called()
                self.assertEqual(set(label.configure.call_args.kwargs), {"text"})

    def test_visualizer_uses_real_labels_vector_icons_and_resolved_canvas_colors(self):
        source = inspect.getsource(_WorkflowPipeline)
        init_source = inspect.getsource(_WorkflowPipeline.__init__)
        reveal_source = inspect.getsource(_WorkflowPipeline._apply_stage_reveal)
        self.assertIn("summary_label = ctk.CTkLabel", source)
        self.assertIn("canvas.create_polygon", source)
        self.assertIn("canvas.create_line", source)
        self.assertIn("resolve_color", source)
        self.assertIn('text=""', source)
        self.assertIn("image=self._pause_animation_icon", source)
        self.assertIn("HoverToolTip", source)
        self.assertIn("self.stage_detail_shell", source)
        self.assertIn("self.context_card", source)
        self.assertIn("self._stage_group_frames", source)
        self.assertIn("self._apply_stage_reveal", source)
        self.assertNotIn("CTkProgressBar", source)
        self.assertNotIn('"packet_text"', source)
        self.assertIn("takefocus=1", source)
        self.assertIn('self.animation_button.bind(\n            "<Return>"', source)
        self.assertIn('self.animation_button.bind(\n            "<space>"', source)
        self.assertIn("_apply_responsive_layout(initial_width)", init_source)
        self.assertNotIn("_on_pipeline_configure", init_source)
        self.assertNotIn("font=", reveal_source)
        self.assertNotIn("border_width", reveal_source)


class WorkflowOverviewMapTests(unittest.TestCase):
    def test_overview_has_exactly_four_ordered_static_step_cards(self):
        self.assertEqual(
            [card[0] for card in _WorkflowOverviewMap.STEP_CARDS],
            ["Load & Crop", "Annotate", "Flatten", "Analyze"],
        )
        source = inspect.getsource(_WorkflowOverviewMap)
        self.assertNotIn("Pause animation", source)
        self.assertNotIn("_animation_after_id", source)
        self.assertNotIn("self.after(", source)

    def test_overview_uses_two_columns_then_one_column_when_narrow(self):
        overview = _WorkflowOverviewMap.__new__(_WorkflowOverviewMap)
        overview._layout = None
        overview._available_width = None
        overview._cards = [mock.Mock() for _index in range(4)]
        overview._summary_labels = [mock.Mock() for _index in range(4)]
        overview.card_host = mock.Mock()

        overview.set_available_width(600)

        self.assertEqual(overview._layout, "grid")
        self.assertEqual(
            [card.grid.call_args.kwargs["row"] for card in overview._cards],
            [0, 0, 1, 1],
        )
        self.assertEqual(
            [card.grid.call_args.kwargs["column"] for card in overview._cards],
            [0, 1, 0, 1],
        )

        for card in overview._cards:
            card.reset_mock()
        overview.set_available_width(400)

        self.assertEqual(overview._layout, "vertical")
        self.assertEqual(
            [card.grid.call_args.kwargs["row"] for card in overview._cards],
            [0, 1, 2, 3],
        )
        self.assertTrue(
            all(card.grid.call_args.kwargs["column"] == 0 for card in overview._cards)
        )

    def test_overview_dispatch_never_constructs_the_animated_pipeline(self):
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._content_viewport_width = 600
        overview_widget = mock.Mock()

        with (
            mock.patch(
                "aidas.ui.tutorial._WorkflowOverviewMap",
                return_value=overview_widget,
            ) as overview_type,
            mock.patch("aidas.ui.tutorial._WorkflowPipeline") as pipeline_type,
        ):
            dialog._render_flow(object(), TUTORIAL_PAGES[0])

        overview_type.assert_called_once_with(mock.ANY, available_width=576)
        pipeline_type.assert_not_called()
        overview_widget.pack.assert_called_once_with(fill="x")
        overview_widget.set_available_width.assert_not_called()
        self.assertIs(dialog._flow_visualizer, overview_widget)

    def test_detailed_step_dispatch_still_uses_its_transformation_visual(self):
        dialog = TutorialDialog.__new__(TutorialDialog)
        dialog._content_viewport_width = 600
        pipeline_widget = mock.Mock()
        page = TUTORIAL_PAGES[1]

        with (
            mock.patch("aidas.ui.tutorial._WorkflowOverviewMap") as overview_type,
            mock.patch(
                "aidas.ui.tutorial._WorkflowPipeline",
                return_value=pipeline_widget,
            ) as pipeline_type,
        ):
            dialog._render_flow(object(), page)

        overview_type.assert_not_called()
        pipeline_type.assert_called_once_with(
            mock.ANY,
            (
                page.input_summary,
                page.function_summary,
                page.output_summary,
            ),
            page.stage_points,
            page.completion_check,
            page.tips,
            available_width=576,
        )
        pipeline_widget.pack.assert_called_once_with(fill="x", pady=(10, 0))
        self.assertIs(dialog._flow_visualizer, pipeline_widget)

    def test_detailed_pages_have_no_duplicate_constant_process_list(self):
        source = inspect.getsource(TutorialDialog._render_page)
        self.assertIn('"Four-step workflow"', source)
        self.assertNotIn("Process at a glance", source)
        self.assertNotIn("How to use this step", source)
        self.assertNotIn("_render_process", inspect.getsource(TutorialDialog))


class TutorialApplicationWiringTests(unittest.TestCase):
    class _Notebook:
        def __init__(self, selected=2):
            self.selected = selected
            self.set_calls = []

        def select(self, value=None):
            if value is None:
                return self.selected
            self.set_calls.append(value)
            self.selected = value

        @staticmethod
        def index(value):
            return int(value)

    def test_help_button_is_wired_to_tutorial_not_about(self):
        source = inspect.getsource(AIDaSApp._new_modern_workflow_header)
        self.assertIn("on_help_selected=self._show_tutorial", source)
        self.assertNotIn("on_help_selected=self._show_about", source)

    def test_show_tutorial_starts_on_active_step_and_reuses_one_dialog(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.notebook = self._Notebook(selected=2)
        app._tutorial_dialog = None
        app._select_workflow_step = mock.Mock()
        existing = mock.Mock()
        existing.winfo_exists.return_value = True

        with mock.patch("aidas.app.TutorialDialog", return_value=existing) as dialog_type:
            app._show_tutorial()
            app._show_tutorial()

        dialog_type.assert_called_once_with(
            app,
            initial_page=tutorial_page_index_for_step(2),
            on_step_selected=app._select_workflow_step,
            on_close=app._tutorial_dialog_closed,
        )
        existing.show_step.assert_called_once_with(2)
        existing._schedule_modal_activation.assert_called_once_with(delay_ms=1)
        self.assertEqual(app.notebook.set_calls, [])

    def test_dialog_close_callback_clears_only_the_matching_instance(self):
        app = AIDaSApp.__new__(AIDaSApp)
        active = object()
        app._tutorial_dialog = active

        app._tutorial_dialog_closed(object())
        self.assertIs(app._tutorial_dialog, active)

        app._tutorial_dialog_closed(active)
        self.assertIsNone(app._tutorial_dialog)


if __name__ == "__main__":
    unittest.main()
