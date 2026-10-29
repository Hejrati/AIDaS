from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest import mock
from urllib.parse import unquote, urlsplit
import zipfile
from types import SimpleNamespace

import customtkinter as ctk

from aidas.services.feedback import FEEDBACK_EMAIL
from aidas.ui.components import WorkflowHeader
from aidas.ui.feedback import FeedbackDialog
from aidas.ui.menu_bar import ApplicationMenuBar
from aidas.app import AIDaSApp
from aidas.core.config import Config
from aidas.services.feedback import collect_diagnostics


class FeedbackMenuTests(unittest.TestCase):
    def test_application_reuses_one_feedback_window(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app._feedback_dialog = None
        app._feedback_context = mock.Mock(return_value={"workflow": {"active_step": 3}})
        with mock.patch("aidas.ui.feedback.FeedbackDialog") as dialog:
            app._show_feedback()
            app._show_feedback()
        dialog.assert_called_once_with(app, context={"workflow": {"active_step": 3}})
        dialog.return_value.lift.assert_called_once_with()

    def test_application_snapshot_reports_shapes_settings_and_state_without_data(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app._selected_workflow_index = lambda: 2
        preferences = Config.__new__(Config)
        preferences.prefs = {**Config.DEFAULTS, "api_key": "private-secret-value"}
        app.preferences = preferences
        app.step1 = SimpleNamespace(raw_image=SimpleNamespace(shape=(100, 200), dtype="uint16"),
                                    current_file=Path("C:/patients/PrivatePatient.sdb"))
        app.step2 = SimpleNamespace(_active_ai_using_gpu=True, fovea_x=350, boundary_traces={"RPE": {}, "OLM": {}})
        app.step3 = SimpleNamespace(_busy=True, _selected_r_script_path=lambda _role: None,
                                    _default_r_package_library=lambda: Path("C:/missing-library"))
        report = collect_diagnostics(app._feedback_context())["diagnostics"]
        self.assertEqual(report["workflow"]["active_step"], 3)
        self.assertEqual(report["workflow"]["steps"]["1"]["raw_image"]["shape"], [100, 200])
        self.assertEqual(report["workflow"]["steps"]["2"]["boundary_traces_count"], 2)
        self.assertTrue(report["workflow"]["steps"]["3"]["busy"])
        self.assertNotIn("api_key", report["settings"])
        self.assertNotIn("PrivatePatient", str(report))

    def test_modern_help_menu_dispatches_feedback(self):
        bar = ApplicationMenuBar.__new__(ApplicationMenuBar)
        bar._tutorial_command = None
        bar._feedback_command = mock.Mock()
        bar._check_updates_command = mock.Mock()
        bar._about_command = mock.Mock()
        items = bar._items_for("Help")
        next(item for item in items if item.label == "Send feedback...").command()
        bar._feedback_command.assert_called_once_with()


class FeedbackDialogTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.addCleanup(self.root.destroy)
        self.root.interface_mode = "Modern"
        self.root.geometry("950x750")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.snapshot = {"diagnostics": {"report_id": "test123", "app_version": "test"},
                         "logs": [{"name": "log_01.txt", "text": "A failure happened"}]}
        patch = mock.patch("aidas.ui.feedback.collect_diagnostics", return_value=self.snapshot)
        patch.start()
        self.addCleanup(patch.stop)
        self.dialog = FeedbackDialog(self.root, context={})
        self.addCleanup(lambda: self.dialog.destroy() if self.dialog.winfo_exists() else None)
        self.wait(lambda: self.dialog._snapshot is not None)

    def wait(self, condition):
        deadline = time.monotonic()+5
        while not condition():
            self.root.update()
            if time.monotonic() > deadline:
                self.fail("Feedback worker did not finish")
            time.sleep(0.01)
        self.root.update()

    def fill(self):
        self.dialog.summary.insert(0, "Step 3 failed")
        self.dialog.description.insert("1.0", "Load the sample and run Step 3. Expected success; received an error.")

    def test_saved_report_requires_approval_then_opens_only_a_draft(self):
        self.assertEqual(self.dialog.email_button.cget("state"), "disabled")
        self.fill()
        target = self.directory / "report.zip"
        with mock.patch("aidas.ui.feedback.messagebox.showwarning") as warning:
            self.dialog._save()
        warning.assert_called_once()
        self.assertFalse(target.exists())
        self.dialog.approved.set(True)
        with mock.patch("aidas.ui.feedback.filedialog.asksaveasfilename", return_value=str(target)):
            self.dialog._save()
        self.wait(lambda: self.dialog._report_path is not None)
        with zipfile.ZipFile(target) as archive:
            self.assertIn("diagnostics.json", archive.namelist())
            self.assertIn("logs/log_01.txt", archive.namelist())
        with mock.patch("aidas.ui.feedback.webbrowser.open", return_value=True) as open_draft:
            self.dialog._email()
        self.assertEqual(unquote(urlsplit(open_draft.call_args.args[0]).path), FEEDBACK_EMAIL)
        self.assertIn("Attach report.zip", self.dialog.status.get())
        self.dialog._invalidate_report()
        self.assertEqual(self.dialog.email_button.cget("state"), "disabled")
        self.assertFalse(self.dialog.approved.get())

    def test_attachments_are_optional_and_reset_approval(self):
        sample = self.directory / "sample.img"
        sample.write_bytes(b"sample")
        self.dialog.approved.set(True)
        with mock.patch("aidas.ui.feedback.filedialog.askopenfilenames", return_value=(str(sample),)):
            self.dialog._add_files()
        self.assertEqual(self.dialog._attachments, [sample])
        self.assertFalse(self.dialog.approved.get())
        self.assertIn("sample.img", self.dialog.files_label.cget("text"))
        self.dialog._clear_files()
        self.assertEqual(self.dialog._attachments, [])

    def test_closing_cancels_textbox_callbacks_and_collection_poll(self):
        callback = mock.Mock()
        timer = self.dialog.description.after(1000, callback)
        self.dialog.destroy()
        self.assertNotIn(timer, self.root.tk.call("after", "info"))
        self.root.update()
        callback.assert_not_called()

    def test_header_speech_bubble_is_next_to_help_and_dispatches_feedback(self):
        callback = mock.Mock()
        header = WorkflowHeader(self.root, version="test", on_step_selected=lambda _step: None,
                                on_feedback_selected=callback)
        header.pack(fill="x")
        self.root.update()
        self.assertGreater(header.feedback_button.winfo_rootx(), header.help_button.winfo_rootx())
        header.feedback_button.invoke()
        callback.assert_called_once_with()
        self.assertEqual(header.feedback_image.cget("size"), (24, 24))


if __name__ == "__main__":
    unittest.main()
