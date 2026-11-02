from pathlib import Path
import hashlib
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest import mock
from urllib.parse import parse_qs, unquote, urlsplit
import zipfile
from types import SimpleNamespace

import customtkinter as ctk

from aidas.services.feedback import FEEDBACK_EMAIL
from aidas.ui.components import WorkflowHeader
from aidas.ui.feedback import FeedbackDialog
from aidas.ui.scrolling import DeferredScrollbar
from aidas.ui.theme import COLOR_PAIRS
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
        target = self.directory / "report.txt"
        with mock.patch("aidas.ui.feedback.messagebox.showwarning") as warning:
            self.dialog._save()
        warning.assert_called_once()
        self.assertFalse(target.exists())
        self.dialog.approved.set(True)
        with mock.patch("aidas.ui.feedback.filedialog.asksaveasfilename", return_value=str(target)) as choose:
            self.dialog._save()
        self.assertEqual(choose.call_args.kwargs["defaultextension"], ".txt")
        self.assertEqual(choose.call_args.kwargs["filetypes"], [
            ("Text report for email", "*.txt"), ("Full data ZIP for secure sharing", "*.zip"),
        ])
        self.assertTrue(choose.call_args.kwargs["initialfile"].endswith(".txt"))
        self.wait(lambda: self.dialog._report_path is not None)
        self.assertFalse(zipfile.is_zipfile(target))
        content = target.read_text(encoding="utf-8")
        self.assertIn("Step 3 failed", content)
        self.assertIn("test123", content)
        self.assertIn("A failure happened", content)
        with mock.patch("aidas.ui.feedback.webbrowser.open", return_value=True) as open_draft:
            self.dialog._email()
        self.assertEqual(unquote(urlsplit(open_draft.call_args.args[0]).path), FEEDBACK_EMAIL)
        self.assertIn("Attach report.txt", self.dialog.status.get())
        self.dialog._invalidate_report()
        self.assertEqual(self.dialog.email_button.cget("state"), "disabled")
        self.assertFalse(self.dialog.approved.get())

    def _save_report(self, name, *, sharing_link=""):
        self.fill()
        if sharing_link:
            self.dialog.sharing_link.insert(0, sharing_link)
        self.dialog.approved.set(True)
        target = self.directory / name
        with mock.patch("aidas.ui.feedback.filedialog.asksaveasfilename", return_value=str(target)):
            self.dialog._save()
        self.wait(lambda: self.dialog._report_path is not None)
        return target

    def test_text_report_records_sample_checksum_without_embedding_the_sample(self):
        sample = self.directory / "private-patient.img"
        sample_content = b"UNIQUE-RAW-SAMPLE-CONTENT\x00\xff\xfe"
        sample.write_bytes(sample_content)
        with mock.patch("aidas.ui.feedback.filedialog.askopenfilenames", return_value=(str(sample),)):
            self.dialog._add_files()
        target = self._save_report("email-report.txt")

        content = target.read_text(encoding="utf-8")
        self.assertIn(hashlib.sha256(sample_content).hexdigest(), content)
        self.assertNotIn("private-patient", content)
        self.assertNotIn("UNIQUE-RAW-SAMPLE-CONTENT", content)
        self.assertNotIn("\x00", content)
        self.assertEqual(sample.read_bytes(), sample_content)

    def test_zip_includes_selected_files_but_requires_a_sharing_link_for_email(self):
        sample = self.directory / "sample.img"
        sample.write_bytes(b"original sample bytes")
        self.dialog._attachments = [sample]
        target = self._save_report("full-data.zip")
        with zipfile.ZipFile(target) as archive:
            self.assertEqual(archive.read("data/sample_001.img"), sample.read_bytes())
        with (
            mock.patch("aidas.ui.feedback.messagebox.showwarning") as warning,
            mock.patch("aidas.ui.feedback.webbrowser.open") as open_draft,
        ):
            self.dialog._email()
        open_draft.assert_not_called()
        guidance = warning.call_args.args[1]
        self.assertIn("text report", guidance)
        self.assertIn("HTTPS link", guidance)
        self.assertIn("Do not attach the ZIP", guidance)

    def test_zip_with_sharing_link_opens_a_link_draft_without_zip_attachment_instructions(self):
        link = "https://approved.example.org/shared/sample?key=123&view=1"
        self._save_report("full-data.zip", sharing_link=link)
        with mock.patch("aidas.ui.feedback.webbrowser.open", return_value=True) as open_draft:
            self.dialog._email()
        draft_body = parse_qs(urlsplit(open_draft.call_args.args[0]).query)["body"][0]
        self.assertIn(link, draft_body)
        self.assertNotIn("Please attach the diagnostic ZIP", draft_body)
        self.assertIn("secure sample-data link", self.dialog.status.get())
        self.assertNotIn("Attach full-data.zip", self.dialog.status.get())

    def test_text_report_can_include_a_sharing_link_without_embedding_shared_data(self):
        link = "https://approved.example.org/shared/sample"
        target = self._save_report("email-report.txt", sharing_link=link)
        self.assertIn(link, target.read_text(encoding="utf-8"))
        with mock.patch("aidas.ui.feedback.webbrowser.open", return_value=True) as open_draft:
            self.dialog._email()
        draft_body = parse_qs(urlsplit(open_draft.call_args.args[0]).query)["body"][0]
        self.assertIn(link, draft_body)
        self.assertIn("email-report.txt", draft_body)
        self.assertIn("Attach email-report.txt", self.dialog.status.get())

    def test_changing_or_clearing_the_link_requires_fresh_review_and_report(self):
        self._save_report("email-report.txt")
        self.assertEqual(self.dialog.email_button.cget("state"), "normal")
        self.dialog.sharing_link.insert(0, "https://approved.example.org/shared/sample")
        self.assertFalse(self.dialog.approved.get())
        self.assertIsNone(self.dialog._report_path)
        self.assertEqual(self.dialog.email_button.cget("state"), "disabled")
        self.dialog.approved.set(True)
        self.dialog.sharing_link.delete(0, "end")
        self.assertFalse(self.dialog.approved.get())

    def test_long_form_remains_scrollable_and_footer_actions_visible_at_small_size(self):
        self.dialog.geometry("560x450")
        # Windows delivers the native resize after idle geometry work. Settle
        # it before using the scroll range of the newly sized viewport.
        self.root.update()
        body = self.dialog.sharing_link.master
        canvas = body._parent_canvas
        canvas.yview_moveto(1.0)
        self.root.update()
        self.assertLess(canvas.yview()[0], canvas.yview()[1])
        self.assertGreater(canvas.yview()[0], 0)
        self.assertGreaterEqual(self.dialog.sharing_link.winfo_rooty(), canvas.winfo_rooty())
        self.assertLessEqual(
            self.dialog.sharing_link.winfo_rooty() + self.dialog.sharing_link.winfo_height(),
            canvas.winfo_rooty() + canvas.winfo_height(),
        )
        for button in (self.dialog.save_button, self.dialog.email_button):
            self.assertTrue(button.winfo_viewable())
            self.assertLessEqual(
                button.winfo_rooty() + button.winfo_height(),
                self.dialog.winfo_rooty() + self.dialog.winfo_height(),
            )

    def test_wheel_scrolls_the_form_over_text_entry_checkbox_and_footer_once(self):
        dialog = self.dialog
        dialog.geometry("560x450")
        dialog.description.insert("1.0", "\n".join(f"Description line {index}" for index in range(100)))
        self.root.update()
        canvas = dialog.body._parent_canvas
        self.assertIsInstance(dialog.body._scrollbar, DeferredScrollbar)
        for widget in (
            dialog.description._textbox, dialog.summary._entry,
            dialog.logs_check._text_label, dialog.review_link._label,
            dialog.save_button._text_label,
        ):
            with self.subTest(widget=widget):
                canvas.yview_moveto(0)
                dialog.description.yview_moveto(0)
                self.root.update()
                before_text = dialog.description.yview()
                with mock.patch.object(canvas, "yview_moveto", wraps=canvas.yview_moveto) as move:
                    for _ in range(4):
                        widget.event_generate("<MouseWheel>", delta=-120)
                    self.assertEqual(move.call_count, 0)
                    self.root.update()
                    self.assertEqual(move.call_count, 1)
                self.assertGreater(canvas.yview()[0], 0)
                self.assertEqual(dialog.description.yview(), before_text)

    def test_dialog_scrolling_does_not_capture_other_windows_or_preview_text(self):
        other = tk.Text(self.root, height=4)
        other.pack()
        other.insert("1.0", "\n".join(str(index) for index in range(100)))
        self.root.update()
        canvas = self.dialog.body._parent_canvas
        canvas.yview_moveto(0)
        other.event_generate("<MouseWheel>", delta=-120)
        self.root.update()
        self.assertEqual(canvas.yview()[0], 0)
        self.assertGreater(other.yview()[0], 0)
        self.dialog._preview()
        preview = self.dialog._preview_window
        self.wait(lambda: preview._text_after_id is None)
        self.assertNotIn(self.dialog.body._wheel_tag, preview.box._textbox.bindtags())
        self.assertNotIn(self.dialog.body._wheel_tag, other.bindtags())

    def test_review_link_replaces_button_and_reuses_preview_without_changing_approval(self):
        dialog = self.dialog
        self.assertNotIn("preview_button", dialog.__dict__)
        self.assertEqual(dialog.logs_check.cget("text"), "Include relevant recent logs")
        self.assertEqual(dialog.review_link.cget("text"), "(review before sharing)")
        self.assertTrue(dialog.review_link.cget("font").cget("underline"))
        self.assertEqual(dialog.review_link.cget("cursor"), "hand2")
        dialog.approved.set(True)
        dialog.review_link._label.event_generate("<Button-1>")
        self.root.update()
        preview = dialog._preview_window
        self.assertIsNotNone(preview)
        self.wait(lambda: preview._text_after_id is None)
        self.assertIn("A failure happened", preview.box.get("1.0", "end-1c"))
        before = preview.box.get("1.0", "end-1c")
        with mock.patch.object(preview, "set_text", wraps=preview.set_text) as reset:
            dialog.review_link._label.event_generate("<Button-1>")
            self.root.update()
            reset.assert_not_called()
        self.assertIs(dialog._preview_window, preview)
        self.assertEqual(preview.box.get("1.0", "end-1c"), before)
        self.assertTrue(dialog.approved.get())
        dialog.logs_check.toggle()
        self.assertFalse(dialog.approved.get())
        dialog._review_clicked()
        self.wait(lambda: preview._text_after_id is None)
        self.assertNotIn("A failure happened", preview.box.get("1.0", "end-1c"))

    def test_review_link_respects_disabled_state_and_has_keyboard_bindings(self):
        dialog = self.dialog
        self.assertTrue(dialog.review_link._label.bind("<Return>"))
        self.assertTrue(dialog.review_link._label.bind("<space>"))
        with mock.patch.object(dialog, "_preview") as preview:
            dialog._set_editing(False)
            dialog.review_link._label.event_generate("<Button-1>")
            self.root.update()
            preview.assert_not_called()
            dialog._set_editing(True)
            dialog.review_link._label.focus_force()
            self.root.update()
            dialog.review_link._label.event_generate("<Return>")
            self.root.update()
            preview.assert_called_once_with()

    def test_required_stars_use_danger_color_and_fit_narrow_form(self):
        self.dialog.geometry("560x450")
        self.root.update()
        self.assertEqual(len(self.dialog.required_markers), 2)
        body = self.dialog.body
        for marker in self.dialog.required_markers:
            self.assertEqual(marker.cget("text"), "*")
            self.assertEqual(tuple(marker.cget("text_color")), tuple(COLOR_PAIRS["danger"]))
            self.assertLessEqual(
                marker.winfo_rootx()+marker.winfo_width(),
                body._parent_canvas.winfo_rootx()+body._parent_canvas.winfo_width(),
            )

    def test_large_preview_is_complete_and_yields_between_insertions(self):
        dialog = self.dialog
        text = "Report line\n" * 20000
        dialog._preview_text[True] = text
        dialog._preview()
        preview = dialog._preview_window
        first = preview.box.get("1.0", "end-1c")
        self.assertTrue(text.startswith(first))
        self.assertLess(len(first), len(text))
        self.assertIsNotNone(preview._text_after_id)
        tick = mock.Mock()
        self.root.after(0, tick)
        self.wait(lambda: preview._text_after_id is None)
        tick.assert_called_once_with()
        self.assertEqual(preview.box.get("1.0", "end-1c"), text)
        self.assertEqual(preview.box.cget("state"), "disabled")

    def test_replacing_or_closing_a_partial_preview_cancels_previous_text_chunks(self):
        dialog = self.dialog
        dialog._preview_text[True] = "Large log\n" * 20000
        dialog._preview()
        preview = dialog._preview_window
        old_timer = preview._text_after_id
        self.assertIsNotNone(old_timer)
        dialog.include_logs.set(False)
        dialog._preview()
        self.assertNotIn(old_timer, self.root.tk.call("after", "info"))
        self.wait(lambda: preview._text_after_id is None)
        self.assertEqual(preview.box.get("1.0", "end-1c"), dialog._preview_text[False])
        dialog.include_logs.set(True)
        dialog._preview()
        timer = preview._text_after_id
        self.assertIsNotNone(timer)
        preview.destroy()
        self.assertNotIn(timer, self.root.tk.call("after", "info"))
        self.root.update()
        dialog._preview()
        self.assertIsNot(dialog._preview_window, preview)

    def test_close_removes_scoped_wheel_bindings_and_pending_scroll(self):
        dialog = self.dialog
        tag = dialog.body._wheel_tag
        global_wheel = self.root.bind_all("<MouseWheel>")
        dialog.description._textbox.event_generate("<MouseWheel>", delta=-120)
        timer = dialog.body._wheel_after_id
        self.assertIsNotNone(timer)
        dialog.destroy()
        self.assertNotIn(timer, self.root.tk.call("after", "info"))
        self.assertEqual(self.root.bind_class(tag, "<MouseWheel>"), "")
        self.assertEqual(self.root.bind_all("<MouseWheel>"), global_wheel)
        self.root.update()

    def test_unapproved_unsaved_edits_do_not_redraw_the_email_button(self):
        with mock.patch.object(self.dialog.email_button, "configure", wraps=self.dialog.email_button.configure) as configure:
            for _ in range(20):
                self.dialog._invalidate_report()
            configure.assert_not_called()

    def test_closing_dialog_cancels_a_partially_loaded_preview(self):
        dialog = self.dialog
        dialog._preview_text[True] = "Large log\n" * 20000
        dialog._preview()
        preview = dialog._preview_window
        timer = preview._text_after_id
        self.assertIsNotNone(timer)
        dialog.destroy()
        self.assertNotIn(timer, self.root.tk.call("after", "info"))
        self.assertIsNone(preview.box)
        self.root.update()

    def test_attachments_are_optional_and_reset_approval(self):
        sample = self.directory / "sample.img"
        sample.write_bytes(b"sample")
        self.dialog.approved.set(True)
        with mock.patch("aidas.ui.feedback.filedialog.askopenfilenames", return_value=(str(sample),)):
            self.dialog._add_files()
        self.assertEqual(self.dialog._attachments, [sample.resolve()])
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


class FeedbackDialogLogicTests(unittest.TestCase):
    """Exercise thread snapshots and email fallbacks without creating Tk."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dialog = FeedbackDialog.__new__(FeedbackDialog)
        self.dialog._snapshot = {"diagnostics": {"report_id": "headless123"}, "logs": []}
        self.dialog._working = False
        self.dialog.summary = SimpleNamespace(get=mock.Mock(return_value="Problem summary"))
        self.dialog.description = SimpleNamespace(get=mock.Mock(return_value="How to reproduce"))
        self.dialog.sharing_link = SimpleNamespace(get=mock.Mock(return_value=""))
        self.dialog.approved = SimpleNamespace(get=mock.Mock(return_value=True), set=mock.Mock())
        self.dialog.include_logs = SimpleNamespace(get=mock.Mock(return_value=True))
        self.dialog.status = SimpleNamespace(set=mock.Mock())
        self.dialog.email_button = SimpleNamespace(configure=mock.Mock())
        self.dialog._attachments = []
        self.dialog._cancel = threading.Event()
        self.dialog._set_editing = mock.Mock()
        self.dialog._start_worker = mock.Mock()
        self.dialog._saved_title = "Problem summary"
        self.dialog._saved_sharing_link = ""
        self.dialog._report_path = None

    def test_save_worker_receives_plain_values_and_never_reads_tk(self):
        link = "https://approved.example.org/shared/sample"
        self.dialog.sharing_link.get.return_value = link
        target = self.directory / "report.txt"
        with mock.patch("aidas.ui.feedback.filedialog.asksaveasfilename", return_value=str(target)):
            self.dialog._save()
        kind, work = self.dialog._start_worker.call_args.args
        self.assertEqual(kind, "report")
        for widget in (self.dialog.summary, self.dialog.description, self.dialog.sharing_link,
                       self.dialog.approved, self.dialog.include_logs):
            widget.get.side_effect = AssertionError("Worker must not read a Tk control")
        with mock.patch("aidas.ui.feedback.save_report", return_value=target) as save:
            self.assertEqual(work(), target)
        save.assert_called_once_with(
            str(target), self.dialog._snapshot, title="Problem summary", description="How to reproduce",
            attachments=(), include_logs=True, cancel_event=self.dialog._cancel, sharing_link=link,
        )

    def test_invalid_link_stops_before_file_dialog_or_worker(self):
        for link in ("http://example.org/sample", "https://name:secret@example.org/sample", "https://example.org/\ninvalid"):
            with self.subTest(link=link):
                self.dialog.sharing_link.get.return_value = link
                with (
                    mock.patch("aidas.ui.feedback.filedialog.asksaveasfilename") as choose,
                    mock.patch("aidas.ui.feedback.messagebox.showwarning") as warning,
                ):
                    self.dialog._save()
                warning.assert_called_once()
                choose.assert_not_called()
                self.dialog._start_worker.assert_not_called()

    def test_zip_without_link_never_formats_or_opens_a_draft(self):
        target = self.directory / "report.zip"
        target.write_bytes(b"placeholder")
        self.dialog._report_path = target
        with (
            mock.patch("aidas.ui.feedback.messagebox.showwarning") as warning,
            mock.patch("aidas.ui.feedback.feedback_mailto") as mailto,
            mock.patch("aidas.ui.feedback.webbrowser.open") as open_draft,
        ):
            self.dialog._email()
        warning.assert_called_once()
        mailto.assert_not_called()
        open_draft.assert_not_called()

    def test_invalid_saved_link_is_rejected_before_formatting_a_draft(self):
        target = self.directory / "report.txt"
        target.write_text("report", encoding="utf-8")
        self.dialog._report_path = target
        self.dialog._saved_sharing_link = "https://name:secret@example.org/sample"
        with (
            mock.patch("aidas.ui.feedback.messagebox.showwarning") as warning,
            mock.patch("aidas.ui.feedback.feedback_mailto") as mailto,
        ):
            self.dialog._email()
        warning.assert_called_once()
        mailto.assert_not_called()

    def test_email_app_failure_preserves_txt_attachment_and_zip_link_instructions(self):
        link = "https://approved.example.org/shared/sample"
        self.dialog._saved_sharing_link = link
        for suffix in (".txt", ".zip"):
            target = self.directory / ("report" + suffix)
            target.write_bytes(b"placeholder")
            self.dialog._report_path = target
            with (
                self.subTest(suffix=suffix),
                mock.patch("aidas.ui.feedback.webbrowser.open", return_value=False),
                mock.patch("aidas.ui.feedback.messagebox.showinfo") as info,
            ):
                self.dialog._email()
                instructions = info.call_args.args[1]
                self.assertIn(FEEDBACK_EMAIL, instructions)
                self.assertIn(link, instructions)
                if suffix == ".txt":
                    self.assertIn("attach the text report", instructions)
                    self.assertIn(str(target), instructions)
                else:
                    self.assertIn("secure sample-data link", instructions)
                    self.assertNotIn("attach", instructions.lower())

    def test_link_placeholder_does_not_invalidate_review_but_real_changes_do(self):
        self.dialog._sharing_link_placeholder = "https://... (your approved sharing service)"
        self.dialog._sharing_link_value = ""
        self.dialog._invalidate_report = mock.Mock()
        self.assertTrue(self.dialog._sharing_link_changed(self.dialog._sharing_link_placeholder))
        self.dialog._invalidate_report.assert_not_called()
        self.assertTrue(self.dialog._sharing_link_changed("https://approved.example.org/sample"))
        self.dialog._invalidate_report.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
