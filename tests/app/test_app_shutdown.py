from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.app import AIDaSApp
from aidas.steps.step3_flatten import Step3Frame


class AppStatusShutdownTests(unittest.TestCase):
    def test_partial_startup_and_repeated_close_are_safe(self):
        app = AIDaSApp.__new__(AIDaSApp)
        ctk.CTk.__init__(app)
        app.attributes("-alpha", 0)
        app.withdraw()
        try:
            app.destroy()
            app.destroy()
        finally:
            if not app.__dict__.get("_aidas_destroying", False):
                app.destroy()

    def _app_with_status_steps(self, mode):
        # Exercise real Tk destruction in the application's creation order,
        # without loading unrelated workflow controls or application services.
        app = AIDaSApp.__new__(AIDaSApp)
        ctk.CTk.__init__(app)
        app.attributes("-alpha", 0)
        app.withdraw()
        self.addCleanup(lambda: None if app.__dict__.get("_aidas_destroying", False) else app.destroy())
        app.interface_mode = mode
        app._build_status_surface("")
        app.notebook = ttk.Notebook(app)
        for _ in range(2):
            app.notebook.add(ctk.CTkFrame(app.notebook))
        app.step3 = Step3Frame.__new__(Step3Frame)
        ctk.CTkFrame.__init__(app.step3, app.notebook)
        app.step3.elapsed_time_var = tk.StringVar(master=app.step3)
        app._step3_elapsed_trace_id = app.step3.elapsed_time_var.trace_add(
            "write", app._refresh_image_path_status,
        )
        app.notebook.add(app.step3)
        app.notebook.add(ctk.CTkFrame(app.notebook))
        app.step5 = ctk.CTkFrame(app.notebook)
        app.step5.status_var = tk.StringVar(master=app.step5)
        app._step5_status_trace_id = app.step5.status_var.trace_add(
            "write", app._refresh_image_path_status,
        )
        app.notebook.add(app.step5)
        app.notebook.select(app.step3)
        app.report_callback_exception = mock.Mock()
        return app

    def test_closing_detaches_both_status_traces_before_children_update_and_cleans_timer(self):
        for mode in ("Modern", "Classic"):
            with self.subTest(mode=mode):
                app = self._app_with_status_steps(mode)
                variable = app.step3.elapsed_time_var
                trace_id = app._step3_elapsed_trace_id
                step5_variable = app.step5.status_var
                step5_trace_id = app._step5_status_trace_id
                app.step3._start_r_elapsed_timer()
                timer_id = app.step3._r_elapsed_after_id
                self.assertEqual(app.status.cget("text"), "Elapsed: 00:00:00")
                self.assertIn(timer_id, app.tk.call("after", "info"))

                original_destroy = app.step3.destroy

                def destroy_step3():
                    # Both observers must be detached before the first child
                    # cleanup can write a variable after the footer is deleted.
                    self.assertNotIn(trace_id, [name for _modes, name in variable.trace_info()])
                    self.assertNotIn(step5_trace_id, [name for _modes, name in step5_variable.trace_info()])
                    step5_variable.set("Finishing Excel cleanup")
                    original_destroy()

                with mock.patch.object(app.step3, "destroy", side_effect=destroy_step3) as child_destroy:
                    app.destroy()

                child_destroy.assert_called_once_with()

                app.report_callback_exception.assert_not_called()
                self.assertNotIn(trace_id, [name for _modes, name in variable.trace_info()])
                self.assertNotIn(step5_trace_id, [name for _modes, name in step5_variable.trace_info()])
                self.assertEqual(variable.get(), "")
                self.assertEqual(step5_variable.get(), "Finishing Excel cleanup")
                self.assertNotIn(timer_id, app.tk.call("after", "info"))
                # A late completion must not re-enter the deleted status canvas.
                variable.set("Total elapsed: 00:00:01")
                step5_variable.set("Excel file saved: measurements.xlsx")
                app._set_status_message("Finished")
                app._refresh_image_path_status()
                app.report_callback_exception.assert_not_called()

    def test_status_updates_skip_a_destroyed_footer_and_resume_with_replacement(self):
        for mode in ("Modern", "Classic"):
            with self.subTest(mode=mode):
                app = self._app_with_status_steps(mode)
                app.step3.elapsed_time_var.set("Elapsed: 00:00:03")
                surface = app.status_bar if app.status_bar is not None else app.status
                surface.destroy()

                app.step3.elapsed_time_var.set("Elapsed: 00:00:04")
                app._set_status_message("Finished")
                app._refresh_image_path_status()
                app.report_callback_exception.assert_not_called()

                app._build_status_surface("")
                app.step3.elapsed_time_var.set("Elapsed: 00:00:05")
                self.assertEqual(app.status.cget("text"), "Elapsed: 00:00:05")
                app.notebook.select(app.step5)
                app.step5.status_var.set("Excel file saved: C:\\OCT data\\measurements.xlsx")
                self.assertEqual(app.status.cget("text"), app.step5.status_var.get())
                app.destroy()
                app.report_callback_exception.assert_not_called()


if __name__ == "__main__":
    unittest.main()
