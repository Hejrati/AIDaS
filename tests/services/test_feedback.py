from collections import deque
from concurrent.futures import CancelledError
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import parse_qs, unquote, urlsplit
import zipfile

from aidas.services import feedback


class FeedbackReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for patch in (
            mock.patch.object(feedback, "app_base_dir", return_value=self.root),
            mock.patch.object(feedback.Config, "CONFIG_DIR", self.root / "config"),
            mock.patch.object(feedback, "_events", deque(maxlen=100)),
            mock.patch.object(feedback, "_logger", None),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def snapshot(self):
        return feedback.collect_diagnostics({"workflow": {"active_step": 3}, "settings": {"sdb_raw_width": 768}})

    def test_scrubs_paths_credentials_emails_and_keeps_traceback_error(self):
        text = ('File "C:\\Users\\Alice\\Patient Smith\\input.sdb", line 30\n'
                'File "/home/alice/Patient Smith/input.sdb", line 10\n'
                'token=very-secret-value\nAuthorization: Bearer abcdefghi\n'
                'alice@example.com\nValueError: boundary is missing')
        clean = feedback.redact(text)
        for private in ("Alice", "alice", "Patient Smith", "very-secret-value", "abcdefghi", "example.com"):
            self.assertNotIn(private, clean)
        self.assertIn("ValueError: boundary is missing", clean)
        self.assertIn("line 30", clean)

    def test_diagnostics_snapshot_includes_versions_script_hashes_and_bounded_scrubbed_logs(self):
        logs = self.root / "logs"
        logs.mkdir()
        (logs / "step3.log").write_text('R failed\nFile "C:\\Users\\Alice\\Patient\\input.img"\nsecret=123456')
        script = self.root / "processing.R"
        script.write_text("print(1)")
        description = self.root / "r_library/AnalyzeFMRI/DESCRIPTION"
        description.parent.mkdir(parents=True)
        description.write_text("Package: AnalyzeFMRI\nVersion: 1.1.16\n")
        snapshot = feedback.collect_diagnostics({"scripts": {"main": script}, "r_library": description.parent.parent,
                                                "workflow": {"status": 'Unable to open "C:\\Users\\Alice\\Patient\\input.img"'}})
        self.assertEqual(snapshot["diagnostics"]["r"]["packages"]["AnalyzeFMRI"], "1.1.16")
        self.assertEqual(snapshot["diagnostics"]["r"]["main_script"]["sha256"], hashlib.sha256(script.read_bytes()).hexdigest())
        self.assertIn("numpy", snapshot["diagnostics"]["python_packages"])
        preview = feedback.report_preview(snapshot)
        for private in ("Alice", "Patient", "123456"):
            self.assertNotIn(private, preview)
        json.loads(json.dumps(snapshot["diagnostics"]))

    def test_bundle_contains_only_approved_files_preserves_analyze_pairs_and_original_bytes(self):
        patient_header = self.root / "Patient Smith.hdr"
        patient_image = self.root / "Patient Smith.img"
        patient_header.write_bytes(b"header data")
        patient_image.write_bytes(b"image data")
        (self.root / "not-selected.img").write_bytes(b"must stay private")
        target = self.root / "report.zip"
        feedback.save_report(target, self.snapshot(), title="R failure", description="Load image, run, error.",
                             attachments=(patient_header, patient_image))
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            self.assertIn("data/sample_001.hdr", names)
            self.assertIn("data/sample_001.img", names)
            self.assertNotIn("Patient", " ".join(names))
            self.assertEqual(archive.read("data/sample_001.hdr"), b"header data")
            manifest = json.loads(archive.read("attachments.json"))
            self.assertEqual(manifest[1]["sha256"], hashlib.sha256(b"image data").hexdigest())
            self.assertNotIn("Patient Smith", archive.read("attachments.json").decode())
            self.assertNotIn("not-selected", " ".join(names))
        self.assertEqual(patient_header.read_bytes(), b"header data")
        self.assertEqual(patient_image.read_bytes(), b"image data")

    def test_logs_are_excluded_when_user_unchecks_option(self):
        snapshot = self.snapshot()
        snapshot["logs"] = [{"name": "log_01.txt", "text": "private debug log"}]
        target = self.root / "report.zip"
        feedback.save_report(target, snapshot, title="Failure", description="Details", include_logs=False)
        with zipfile.ZipFile(target) as archive:
            self.assertFalse(any(name.startswith("logs/") for name in archive.namelist()))
        self.assertNotIn("private debug log", feedback.report_preview(snapshot, False))

    def test_cancel_preserves_existing_report_and_removes_partial_file(self):
        target = self.root / "report.zip"
        target.write_bytes(b"previous report")
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            feedback.save_report(target, self.snapshot(), title="Failure", description="Details", cancel_event=cancel)
        self.assertEqual(target.read_bytes(), b"previous report")
        self.assertEqual(list(self.root.glob(".aidas-feedback-*.zip")), [])

    def test_size_limit_rejects_before_creating_report(self):
        source = self.root / "large.img"
        source.write_bytes(b"a"*101)
        with mock.patch.object(feedback, "MAX_ATTACHMENT_BYTES", 100), self.assertRaises(ValueError):
            feedback.save_report(self.root / "report.zip", self.snapshot(), title="Failure", description="Details", attachments=[source])
        self.assertFalse((self.root / "report.zip").exists())

    def test_archive_limit_failure_preserves_target(self):
        target = self.root / "report.zip"
        target.write_bytes(b"previous report")
        with mock.patch.object(feedback, "MAX_REPORT_BYTES", 10), self.assertRaises(ValueError):
            feedback.save_report(target, self.snapshot(), title="Failure", description="Details")
        self.assertEqual(target.read_bytes(), b"previous report")
        self.assertEqual(list(self.root.glob(".aidas-feedback-*.zip")), [])

    def test_report_cannot_overwrite_its_attachment(self):
        source = self.root / "data.zip"
        source.write_bytes(b"source data")
        with self.assertRaises(ValueError):
            feedback.save_report(source, self.snapshot(), title="Failure", description="Details", attachments=[source])
        self.assertEqual(source.read_bytes(), b"source data")

    def test_report_does_not_overwrite_a_non_zip_data_file(self):
        source = self.root / "data.img"
        source.write_bytes(b"source data")
        with self.assertRaises(ValueError):
            feedback.save_report(source, self.snapshot(), title="Failure", description="Details")
        self.assertEqual(source.read_bytes(), b"source data")

    def test_email_draft_preserves_plus_address_and_has_no_local_paths_or_auto_attachment(self):
        url = feedback.feedback_mailto("report123", "Step 3 & Step 4", self.root / "report.zip")
        parsed = urlsplit(url)
        self.assertEqual(parsed.scheme, "mailto")
        self.assertEqual(unquote(parsed.path), feedback.FEEDBACK_EMAIL)
        query = parse_qs(parsed.query)
        self.assertIn("[AIDaS Feedback]", query["subject"][0])
        self.assertIn("Step 3 & Step 4", query["subject"][0])
        self.assertIn("report.zip", query["body"][0])
        self.assertNotIn(str(self.root), url)
        self.assertNotIn("attachment", query)

    def test_exception_hooks_capture_uncaught_failures_and_preserve_original_handlers(self):
        original_sys, original_thread = mock.Mock(), mock.Mock()
        with mock.patch.object(feedback, "_installed", False), mock.patch.object(feedback, "RotatingFileHandler", side_effect=OSError), \
             mock.patch.object(sys, "excepthook", original_sys), mock.patch.object(threading, "excepthook", original_thread), \
             mock.patch.object(feedback.messagebox, "showerror", mock.Mock()) as original_error:
            feedback.install_exception_capture()
            error = ValueError("layer failed")
            sys.excepthook(ValueError, error, None)
            args = SimpleNamespace(exc_type=RuntimeError, exc_value=RuntimeError("worker failed"), exc_traceback=None)
            threading.excepthook(args)
            feedback.messagebox.showerror("R failed", "Read failed", parent=None)
            self.assertEqual(original_sys.call_count, 1)
            self.assertEqual(original_thread.call_count, 1)
            text = feedback.report_preview(self.snapshot())
            self.assertIn("layer failed", text)
            self.assertIn("worker failed", text)
            self.assertIn("Read failed", text)
            original_error.assert_called_once_with("R failed", "Read failed", parent=None)


if __name__ == "__main__":
    unittest.main()
