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

    def _app_with_step3(self, mode):
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
        app.notebook.select(app.step3)
        app.report_callback_exception = mock.Mock()
        return app

    def test_closing_step3_detaches_status_trace_and_preserves_timer_cleanup(self):
        for mode in ("Modern", "Classic"):
            with self.subTest(mode=mode):
                app = self._app_with_step3(mode)
                variable = app.step3.elapsed_time_var
                trace_id = app._step3_elapsed_trace_id
                app.step3._start_r_elapsed_timer()
                timer_id = app.step3._r_elapsed_after_id
                self.assertEqual(app.status.cget("text"), "Elapsed: 00:00:00")
                self.assertIn(timer_id, app.tk.call("after", "info"))

                app.destroy()

                app.report_callback_exception.assert_not_called()
                self.assertNotIn(trace_id, [name for _modes, name in variable.trace_info()])
                self.assertEqual(variable.get(), "")
                self.assertNotIn(timer_id, app.tk.call("after", "info"))
                # A late completion must not re-enter the deleted status canvas.
                variable.set("Total elapsed: 00:00:01")
                app._set_status_message("Finished")
                app._refresh_image_path_status()
                app.report_callback_exception.assert_not_called()

    def test_status_updates_skip_a_destroyed_footer_and_resume_with_replacement(self):
        for mode in ("Modern", "Classic"):
            with self.subTest(mode=mode):
                app = self._app_with_step3(mode)
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
                app.destroy()
                app.report_callback_exception.assert_not_called()


if __name__ == "__main__":
    unittest.main()
