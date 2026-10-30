from __future__ import annotations

import inspect
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest import mock

from aidas.app import AIDaSApp
from aidas.ui.tutorial import (
    TUTORIAL_PAGES,
    TutorialDialog,
    _WorkflowOverviewMap,
    _WorkflowPipeline,
    tutorial_page_index_for_step,
)


class TutorialContentTests(unittest.TestCase):
    def test_tutorial_has_overview_and_one_ordered_page_per_workflow_step(self):
        self.assertEqual(
            [page.key for page in TUTORIAL_PAGES],
            ["overview", "step1", "step2", "step3", "step4", "step5"],
        )
        self.assertEqual(
            [page.step_index for page in TUTORIAL_PAGES],
            [None, 0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [tutorial_page_index_for_step(index) for index in range(5)],
            [1, 2, 3, 4, 5],
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
        page = next(page for page in TUTORIAL_PAGES if page.key == "step2")
        self.assertEqual(page.navigation_label, "Step 2 - Segment")
        self.assertNotIn("Annotate", page.title)

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
            "without an automatic timeout",
            "Parallel",
            "Sequential",
            "_flat_LIGHT.hdr",
            "_flat_LIGHT.img",
            "_thickness_vs_distance_from_fovea_LIGHT.txt",
            "Go to Step 4",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_step4_copy_explains_all_rois_measurements_and_final_outputs(self):
        text = self._page_text("step4")
        for token in (
            "MCP/AR",
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
            "Auto-detect ROIs 1-20",
            "ROI 21",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_step5_copy_explains_final_compilation(self):
        text = self._page_text("step5")
        for token in (
            "LE",
            "RE",
            "rrMCP/AR",
            "ELM-RPE",
            "ONL",
            "Create Excel File",
            "rrMCPAR_ELMRPE_ONL.xlsx",
            "automatically",
            "group",
            "subject ID",
            "Folder contents",
            "Refresh",
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
        dialog.page_indicator.configure.assert_called_once_with(
            text=f"  4 of {len(TUTORIAL_PAGES)}  "
        )
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
        dialog.after = mock.Mock(return_value="resize-job")

        with mock.patch(
            "aidas.ui.tutorial.logical_window_size",
            return_value=(400, 1),
        ) as logical_size:
            dialog._on_content_resize(SimpleNamespace(width=800))
            dialog._on_content_resize(SimpleNamespace(width=800))
        dialog._apply_content_resize()

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
    def setUp(self):
        import customtkinter as ctk
        try:
            self.root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def make_guide(self, page_index=1, width=650):
        page = TUTORIAL_PAGES[page_index]
        return _WorkflowPipeline(
            self.root,
            (page.input_summary, page.function_summary, page.output_summary),
            page.stage_points, page.completion_check, page.tips,
            available_width=width,
        )

    def test_each_section_shows_all_its_instructions_immediately(self):
        guide = self.make_guide()
        page = TUTORIAL_PAGES[1]
        for index, section in enumerate(guide.SECTIONS):
            guide._select_section(section)
            expected = page.tips if index == 3 else page.stage_points[index]
            shown = [label.cget("text") for row, label in guide.rows if label.cget("text")]
            self.assertEqual(shown, list(expected))
            for row, label in guide.rows:
                self.assertEqual(bool(row.winfo_manager()), bool(label.cget("text")))
            self.assertEqual(guide.completion.cget("text"), page.completion_check)

    def test_section_selection_and_navigation_schedule_no_repeating_jobs(self):
        guide = self.make_guide()
        before = set(self.root.tk.call("after", "info"))
        for section in guide.SECTIONS:
            guide._select_section(section)
        self.assertEqual(set(self.root.tk.call("after", "info")), before)

    def test_narrow_sections_remain_selectable_and_content_wraps(self):
        guide = self.make_guide(width=300)
        values = guide.sections.cget("values")
        guide.sections.cget("command")(values[2])
        self.assertEqual(guide._section, 2)
        for label, inset in guide._body_labels:
            self.assertEqual(label.cget("wraplength"), max(100, 300 - inset))
        guide.set_available_width(800)
        self.assertEqual(guide.sections.get(), guide.SECTIONS[2])

    def test_all_step_pages_reuse_rows_and_reset_to_prepare(self):
        guide = self.make_guide()
        rows = tuple(guide.rows)
        for page in TUTORIAL_PAGES[1:]:
            guide._select_section(guide.SECTIONS[3])
            self.assertTrue(guide.replace_content(
                (page.input_summary, page.function_summary, page.output_summary),
                page.stage_points, page.completion_check, page.tips,
            ))
            self.assertEqual(tuple(guide.rows), rows)
            self.assertEqual(guide._section, 0)
            self.assertEqual(guide.sections.get(), guide.sections.cget("values")[0])
            self.assertEqual(guide.rows[0][1].cget("text"), page.stage_points[0][0])

    def test_visible_rows_keep_the_template_positions_and_empty_rows_collapse(self):
        guide = self.make_guide(width=800)
        guide.pack(fill="x")
        self.root.geometry("800x1000")
        self.root.deiconify()
        self.root.update_idletasks()
        before = [(row.winfo_y(), row.winfo_height()) for row, _label in guide.rows]
        ready_y = guide.ready.winfo_y()
        for page in TUTORIAL_PAGES[1:]:
            guide.replace_content(
                (page.input_summary, page.function_summary, page.output_summary),
                page.stage_points, page.completion_check, page.tips,
            )
            for section in guide.SECTIONS:
                guide._select_section(section)
                self.root.update_idletasks()
                visible = [(row, label) for row, label in guide.rows if row.winfo_manager()]
                self.assertEqual(
                    [(row.winfo_y(), row.winfo_height()) for row, _label in visible],
                    before[:len(visible)],
                )
                if len(visible) == len(guide.rows):
                    self.assertEqual(guide.ready.winfo_y(), ready_y)
                else:
                    self.assertLess(guide.ready.winfo_y(), ready_y)
                for row, label in visible:
                    self.assertLessEqual(label.winfo_reqheight() + 24, row.winfo_height())

    def test_overview_navigation_preserves_the_existing_step_template(self):
        dialog = TutorialDialog(self.root, initial_page=1)
        self.addCleanup(dialog.close)
        guide = dialog._flow_visualizer
        rows = tuple(guide.rows)
        dialog.show_page(0)
        overview = dialog._flow_visualizer
        for index in range(1, 6):
            dialog.show_page(index)
            self.assertIs(dialog._flow_visualizer, guide)
            self.assertEqual(tuple(guide.rows), rows)
            dialog.show_page(0)
            self.assertIs(dialog._flow_visualizer, overview)
        self.assertTrue(guide.winfo_exists())

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


class WorkflowOverviewMapTests(unittest.TestCase):
    def test_overview_has_exactly_five_ordered_static_step_cards(self):
        self.assertEqual(
            [card[0] for card in _WorkflowOverviewMap.STEP_CARDS],
            ["Load & Crop", "Segment", "Flatten", "MCP/AR", "Create Excel File"],
        )
        source = inspect.getsource(_WorkflowOverviewMap)
        self.assertNotIn("Pause animation", source)
        self.assertNotIn("_animation_after_id", source)
        self.assertNotIn("self.after(", source)

    def test_overview_uses_two_columns_then_one_column_when_narrow(self):
        overview = _WorkflowOverviewMap.__new__(_WorkflowOverviewMap)
        overview._layout = None
        overview._available_width = None
        overview._cards = [mock.Mock() for _index in range(5)]
        overview._summary_labels = [mock.Mock() for _index in range(5)]
        overview.card_host = mock.Mock()

        overview.set_available_width(600)

        self.assertEqual(overview._layout, "grid")
        self.assertEqual(
            [card.grid.call_args.kwargs["row"] for card in overview._cards],
            [0, 0, 1, 1, 2],
        )
        self.assertEqual(
            [card.grid.call_args.kwargs["column"] for card in overview._cards],
            [0, 1, 0, 1, 0],
        )

        for card in overview._cards:
            card.reset_mock()
        overview.set_available_width(400)

        self.assertEqual(overview._layout, "vertical")
        self.assertEqual(
            [card.grid.call_args.kwargs["row"] for card in overview._cards],
            [0, 1, 2, 3, 4],
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
        self.assertIn('"Five-step workflow"', source)
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
