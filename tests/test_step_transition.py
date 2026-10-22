import tkinter as tk
import time
from tkinter import ttk
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from aidas.app import AIDaSApp
from aidas.ui.step_transition import StepTransition
from aidas.ui.components import AppButton
from aidas.ui.components import WorkflowProgressStrip
from aidas.ui.theme import COLOR_PAIRS, resolve_color
from aidas.steps.step2_annotate import Step2BatchSegmentationSelectionPanel
from aidas.steps.step3_flatten import RBatchSelectionPanel
from aidas.steps.step4_analyze_isez import Step4BatchROISelectionPanel


class WorkflowTransitionTests(unittest.TestCase):
    def test_skipped_steps_never_receive_completion_badges(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.progress_strip = Mock()
        app._advance_workflow_progress(2)
        app._advance_workflow_progress(3)
        self.assertEqual(app._workflow_completed_steps, {1, 2})
        app.notebook = Mock()
        app.notebook.index.return_value = 0
        app._refresh_image_path_status = Mock()
        app._on_workflow_tab_changed()
        app.progress_strip.set_workflow_state.assert_called_with({1, 2}, 0, None)

    def test_cancellation_clears_prior_badges_and_stops_the_badge_flight(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.progress_strip = Mock()
        app._workflow_completed_steps = {0, 1, 2}
        app._step_transition = Mock()
        app._step_transition.winfo_exists.return_value = True
        app._on_workflow_cancelled(3)
        self.assertEqual(app._workflow_completed_steps, set())
        app._step_transition.destroy.assert_called_once_with()
        app.progress_strip.set_workflow_state.assert_called_once_with(set(), 2, 2)
        app._advance_workflow_progress(3)
        self.assertEqual(app._workflow_completed_steps, {2})
        app.progress_strip.set_workflow_state.assert_called_with({2}, 3, None)

    def test_completion_updates_header_badges_without_creating_a_popup(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.progress_strip = Mock()
        app._workflow_completed_steps = 0
        app._workflow_transition_source = (200, 300)
        app._animate_step_transition = Mock()
        app._advance_workflow_progress(1)
        app._advance_workflow_progress(1)
        app.progress_strip.set_workflow_state.assert_called_with({0}, 1, None)
        app._animate_step_transition.assert_called_once_with((200, 300), 0)
        app.progress_strip.animate_completion.assert_not_called()
        self.assertNotIn("_workflow_completion_popup", app.__dict__)
        self.assertFalse(hasattr(AIDaSApp, "_show_workflow_completion_popup"))

    def test_manual_page_changes_do_not_replay_a_completion_badge(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.notebook = Mock()
        app.notebook.index.return_value = 0
        app._refresh_image_path_status = Mock()
        app._animate_step_transition = Mock()
        app._on_workflow_tab_changed()
        app._animate_step_transition.assert_not_called()
        app.notebook.index.return_value = 1
        app._on_workflow_tab_changed()
        app._on_workflow_tab_changed()
        app._animate_step_transition.assert_not_called()


class StepTransitionUITests(unittest.TestCase):
    def setUp(self):
        self.root = None
        try:
            self.root = tk.Tk()
            self.root.withdraw()
            self.root.attributes("-alpha", 0)
            self.root.geometry("640x400+10000+10000")
            self.root.deiconify()
        except tk.TclError as exc:
            if self.root is not None:
                self.root.destroy()
            self.skipTest(str(exc))
        self.addCleanup(self.root.destroy)
        self.host = ttk.Frame(self.root)
        self.host.pack(fill="both", expand=True)
        self.root.update()

    def test_independent_badges_and_red_cancel_icon_render_correctly(self):
        strip = WorkflowProgressStrip(self.host)
        strip.pack(fill="x")
        strip.set_workflow_state({1, 2}, 0)
        self.root.update()
        self.assertEqual(strip.circles[0].itemcget(strip.circles[0].text_id, "text"), "1")
        for index in (1, 2):
            self.assertEqual(strip.circles[index].itemcget(strip.circles[index].text_id, "text"), "\u2713")
        strip.set_workflow_state(set(), 2, 2)
        cancelled = strip.circles[2]
        self.assertEqual(cancelled.itemcget(cancelled.text_id, "text"), "\u00d7")
        self.assertEqual(cancelled.itemcget(cancelled.oval_id, "fill"), resolve_color(COLOR_PAIRS["danger"]))
        self.assertIn("Cancelled", strip.workflow_progress_label.cget("text"))
        self.assertEqual(strip.circles[1].itemcget(strip.circles[1].text_id, "text"), "2")

    def test_cancel_buttons_notify_the_application_for_their_actual_step(self):
        self.root._on_workflow_cancelled = Mock()
        for panel_type, number in (
            (Step2BatchSegmentationSelectionPanel, 2),
            (RBatchSelectionPanel, 3),
            (Step4BatchROISelectionPanel, 4),
        ):
            with self.subTest(step=number):
                panel = panel_type.__new__(panel_type)
                panel.tk = self.root.tk
                panel.master = self.host
                panel.step_frame = SimpleNamespace(
                    _close_step2_batch_segmentation_panel=Mock(),
                    _close_r_batch_panel=Mock(),
                    _close_batch_roi_panel=Mock(),
                )
                panel._cancel()
                self.root._on_workflow_cancelled.assert_called_with(number)

    def test_top_step_title_and_all_five_badges_are_retained(self):
        self.root._pack_shell_surface = lambda widget, **options: widget.pack(**options)
        self.root._sync_workflow_progress_strip = AIDaSApp._sync_workflow_progress_strip.__get__(self.root)
        self.root._workflow_completed_steps = 0
        strip = AIDaSApp._build_progress_strip(self.root)
        self.root.update()
        self.assertTrue(strip.winfo_viewable())
        self.assertEqual(len(strip.circles), 5)
        self.assertEqual(strip.workflow_progress_label.cget("text"), "Step 1 of 5: Load & Crop")
        self.root._workflow_completed_steps = {1}
        self.root._workflow_active_step = 2
        self.root._sync_workflow_progress_strip()
        self.assertEqual(strip.workflow_progress_label.cget("text"), "Step 3 of 5: Flatten")

    def test_badge_flies_to_target_without_blocking_controls(self):
        activated = Mock()
        button = ttk.Button(self.host, text="Continue", command=activated)
        button.pack()
        button.focus_force()
        self.root.update()
        focus = self.root.focus_get()
        target = tk.Canvas(self.host, width=24, height=24)
        target.pack()
        self.root.update()
        landed = Mock()
        transition = StepTransition(self.root)
        transition.start((self.root.winfo_rootx() + 300, self.root.winfo_rooty() + 300), target, 2, on_complete=landed)
        self.root.update()
        self.assertTrue(transition.overrideredirect())
        self.assertEqual(transition.canvas.itemcget(transition._number, "text"), "2")
        self.assertIsNone(self.root.grab_current())
        self.assertIs(self.root.focus_get(), focus)
        button.invoke()
        activated.assert_called_once_with()
        done = tk.BooleanVar(self.root, False)
        self.root.after(transition.DURATION_MS + 150, done.set, True)
        self.root.wait_variable(done)
        self.assertFalse(transition.winfo_exists())
        self.assertIsNone(transition._job)
        landed.assert_called_once_with()

    def test_origin_is_recorded_before_the_next_step_button_is_destroyed(self):
        button = AppButton(self.host, text="Go to Step 3", command=lambda: button.destroy())
        button.pack()
        self.root.update()
        expected = (button.winfo_rootx() + button.winfo_width() / 2,
                    button.winfo_rooty() + button.winfo_height() / 2)
        button.invoke()
        self.assertFalse(button.winfo_exists())
        self.assertEqual(self.root._workflow_transition_source, expected)

    def test_real_mouse_release_starts_the_flying_badge(self):
        root = self.root
        root._pack_shell_surface = lambda widget, **options: widget.pack(**options)
        root._sync_workflow_progress_strip = AIDaSApp._sync_workflow_progress_strip.__get__(root)
        root._animate_step_transition = AIDaSApp._animate_step_transition.__get__(root)
        root._workflow_completed_steps = 0
        AIDaSApp._build_progress_strip(root)
        button = AppButton(self.host, text="Go to Step 2", command=lambda: AIDaSApp._advance_workflow_progress(root, 1))
        button.pack()
        root.update()
        expected = (button.winfo_rootx() + button.winfo_width() / 2,
                    button.winfo_rooty() + button.winfo_height() / 2)
        button._text_label.event_generate("<Enter>")
        button._text_label.event_generate("<ButtonPress-1>")
        button._text_label.event_generate("<ButtonRelease-1>")
        root.update()
        self.assertTrue(root._step_transition.winfo_exists())
        self.assertEqual(root._step_transition._source, expected)
        self.assertIs(root._step_transition._target, root.progress_strip.circles[0])
        root._step_transition.destroy()

    def test_slow_page_loading_does_not_skip_the_badge_flight(self):
        target = tk.Canvas(self.host, width=24, height=24)
        target.pack()
        self.root.update()
        landed = Mock()
        transition = StepTransition(self.root)
        transition.start((300, 300), target, 1, on_complete=landed)
        time.sleep(transition.DURATION_MS / 1000 + 0.1)
        self.root.update()
        self.assertTrue(transition.winfo_exists())
        landed.assert_not_called()
        transition.destroy()

    def test_cached_destination_updates_when_the_target_moves(self):
        holder = ttk.Frame(self.host, width=80, height=80)
        holder.place(x=20, y=20)
        target = tk.Canvas(holder, width=24, height=24, highlightthickness=0)
        target.place(x=0, y=0)
        self.root.update()
        transition = StepTransition(self.root)
        transition.start((300, 300), target, 1)
        original = transition._target_bounds
        holder.place_configure(x=150, y=60)
        self.root.update()
        done = tk.BooleanVar(self.root, False)
        self.root.after(40, done.set, True)
        self.root.wait_variable(done)
        self.assertNotEqual(transition._target_bounds, original)
        self.assertEqual(transition._target_bounds[:2], (
            target.winfo_rootx() + target.winfo_width() / 2,
            target.winfo_rooty() + target.winfo_height() / 2,
        ))
        transition.destroy()

    def test_stable_animation_reuses_target_geometry_and_keeps_window_size_fixed(self):
        target = tk.Canvas(self.host, width=24, height=24)
        target.pack()
        self.root.update()
        transition = StepTransition(self.root)
        with (
            patch.object(target, "winfo_rootx", wraps=target.winfo_rootx) as reads,
            patch.object(transition, "geometry", wraps=transition.geometry) as geometry,
            patch.object(transition, "_render", wraps=transition._render) as frames,
        ):
            transition.start((300, 300), target, 1)
            done = tk.BooleanVar(self.root, False)
            self.root.after(transition.DURATION_MS + 150, done.set, True)
            self.root.wait_variable(done)
            self.assertFalse(transition.winfo_exists())
            self.assertLess(reads.call_count, frames.call_count / 2)
            sizes = [call.args[0] for call in geometry.call_args_list if "x" in call.args[0]]
            self.assertEqual(sizes, [f"{transition.START_SIZE}x{transition.START_SIZE}"])

    def test_rapid_step_changes_replace_and_cancel_the_previous_animation(self):
        app = self.root
        target = tk.Canvas(self.host, width=24, height=24)
        target.pack()
        self.root.update()
        app.progress_strip = SimpleNamespace(circles=[target], animate_completion=Mock())
        AIDaSApp._animate_step_transition(app, (200, 300), 0)
        first = app._step_transition
        job = first._job
        AIDaSApp._animate_step_transition(app, (200, 300), 0)
        self.assertFalse(first.winfo_exists())
        self.assertNotIn(job, self.root.tk.call("after", "info"))
        self.assertTrue(app._step_transition.winfo_exists())
        app._step_transition.destroy()


if __name__ == "__main__":
    unittest.main()
