from collections import deque
from concurrent.futures import CancelledError
import base64
import hashlib
import json
import os
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
        # Windows CI may expose TEMP through an 8.3 user-profile alias.
        self.root = Path(self.temporary.name).resolve()
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

    def test_text_report_is_readable_utf8_with_diagnostics_logs_and_only_file_metadata(self):
        header = self.root / "Patient Smith_marked.hdr"
        image = header.with_suffix(".img")
        unrelated = self.root / "Not selected.img"
        payloads = {
            header: b"\x00\xff\x80PRIVATE_SELECTED_HEADER\x00" * 5,
            image: b"\x00\xfe\x81PRIVATE_SELECTED_IMAGE\x00" * 9,
        }
        for source, contents in payloads.items():
            source.write_bytes(contents)
        unrelated.write_bytes(b"unselected private data")
        original_hashes = {
            source: hashlib.sha256(contents).hexdigest() for source, contents in payloads.items()
        }
        snapshot = self.snapshot()
        snapshot["logs"] = [{"name": "log_01.txt", "text": "Boundary α detected; 保存 complete."}]
        title = "Step 3 — μ-calibration 保存"
        description = "Open the image, choose fovea, then run.\nObserved: δ = 2 µm."
        target = self.root / "report.txt"

        result = feedback.save_report(
            target, snapshot, title=title, description=description, attachments=(header, image),
        )

        self.assertEqual(result, target)
        self.assertFalse(zipfile.is_zipfile(target))
        text = target.read_bytes().decode("utf-8", errors="strict").replace("\r\n", "\n")
        for expected in (title, description, snapshot["diagnostics"]["report_id"],
                         snapshot["diagnostics"]["app_version"], "python_packages", "active_step",
                         "log_01.txt", snapshot["logs"][0]["text"],
                         "sample_001.hdr", "sample_001.img", "marked input"):
            self.assertIn(expected, text)
        for source, contents in payloads.items():
            self.assertIn(original_hashes[source], text)
            self.assertIn(str(len(contents)), text)
            self.assertEqual(source.read_bytes(), contents)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hashes[source])
            self.assertNotIn(base64.b64encode(contents).decode("ascii"), text)
        for private in ("Patient Smith", str(self.root), "Not selected", "unselected private data",
                        "PRIVATE_SELECTED_HEADER", "PRIVATE_SELECTED_IMAGE", "\x00", "\ufffd"):
            self.assertNotIn(private, text)

        archive_path = self.root / "same-report.zip"
        feedback.save_report(
            archive_path, snapshot, title=title, description=description, attachments=(header, image),
        )
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("attachments.json"))
            for entry in manifest:
                for key in ("file", "sha256", "role_hint"):
                    self.assertIn(entry[key], text)
                self.assertIn(str(entry["bytes"]), text)
            self.assertEqual(archive.read(manifest[0]["file"]), payloads[header])
            self.assertEqual(archive.read(manifest[1]["file"]), payloads[image])

    def test_logs_are_excluded_when_user_unchecks_option(self):
        snapshot = self.snapshot()
        snapshot["logs"] = [{"name": "log_01.txt", "text": "private debug log"}]
        target = self.root / "report.zip"
        feedback.save_report(target, snapshot, title="Failure", description="Details", include_logs=False)
        with zipfile.ZipFile(target) as archive:
            self.assertFalse(any(name.startswith("logs/") for name in archive.namelist()))
        self.assertNotIn("private debug log", feedback.report_preview(snapshot, False))

    def test_text_report_excludes_logs_when_user_unchecks_option(self):
        snapshot = self.snapshot()
        snapshot["logs"] = [{"name": "log_01.txt", "text": "optional private debug log"}]
        target = self.root / "report.txt"

        feedback.save_report(target, snapshot, title="Failure", description="Details", include_logs=False)

        text = target.read_text(encoding="utf-8")
        self.assertIn(snapshot["diagnostics"]["report_id"], text)
        self.assertNotIn("optional private debug log", text)
        self.assertNotIn("log_01.txt", text)

    def test_cancel_preserves_existing_report_and_removes_partial_file(self):
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                target = self.root / ("report" + suffix)
                target.write_bytes(b"previous report")
                cancel = threading.Event()
                cancel.set()
                with self.assertRaises(CancelledError):
                    feedback.save_report(target, self.snapshot(), title="Failure", description="Details", cancel_event=cancel)
                self.assertEqual(target.read_bytes(), b"previous report")
                self.assertEqual(list(self.root.glob(".aidas-feedback-*")), [])

    def test_size_limit_rejects_before_creating_report(self):
        source = self.root / "large.img"
        source.write_bytes(b"a"*101)
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                target = self.root / ("report" + suffix)
                with mock.patch.object(feedback, "MAX_ATTACHMENT_BYTES", 100), self.assertRaises(ValueError):
                    feedback.save_report(target, self.snapshot(), title="Failure", description="Details", attachments=[source])
                self.assertFalse(target.exists())
                self.assertEqual(source.read_bytes(), b"a" * 101)

    def test_report_limit_failure_preserves_target(self):
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                target = self.root / ("report" + suffix)
                target.write_bytes(b"previous report")
                with mock.patch.object(feedback, "MAX_REPORT_BYTES", 10), self.assertRaises(ValueError):
                    feedback.save_report(target, self.snapshot(), title="Failure", description="Details")
                self.assertEqual(target.read_bytes(), b"previous report")
                self.assertEqual(list(self.root.glob(".aidas-feedback-*")), [])

    def test_failed_atomic_replace_preserves_existing_report_and_selected_file(self):
        source = self.root / "sample.img"
        source.write_bytes(b"original image")
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                target = self.root / ("report" + suffix)
                target.write_bytes(b"previous report")
                with mock.patch.object(feedback.os, "replace", side_effect=OSError("file locked")), self.assertRaises(OSError):
                    feedback.save_report(target, self.snapshot(), title="Failure", description="Details", attachments=[source])
                self.assertEqual(target.read_bytes(), b"previous report")
                self.assertEqual(source.read_bytes(), b"original image")
                self.assertEqual(list(self.root.glob(".aidas-feedback-*")), [])

    def test_source_open_failure_preserves_existing_report_and_cleans_partial_file(self):
        source = self.root / "sample.img"
        source.write_bytes(b"original image")
        original_open = Path.open
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                target = self.root / ("report" + suffix)
                target.write_bytes(b"previous report")

                def deny_selected_source(path, *args, **kwargs):
                    mode = args[0] if args else kwargs.get("mode", "r")
                    if path == source and mode == "rb":
                        raise PermissionError("selected sample is locked")
                    return original_open(path, *args, **kwargs)

                with mock.patch.object(Path, "open", new=deny_selected_source), self.assertRaises(PermissionError):
                    feedback.save_report(target, self.snapshot(), title="Failure", description="Details", attachments=[source])

                self.assertEqual(target.read_bytes(), b"previous report")
                self.assertEqual(source.read_bytes(), b"original image")
                self.assertEqual(list(self.root.glob(".aidas-feedback-*")), [])

    def test_report_cannot_overwrite_its_attachment(self):
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                source = self.root / ("data" + suffix)
                source.write_bytes(b"source data")
                with self.assertRaises(ValueError):
                    feedback.save_report(source, self.snapshot(), title="Failure", description="Details", attachments=[source])
                self.assertEqual(source.read_bytes(), b"source data")

    def test_report_does_not_overwrite_a_file_with_an_unsupported_extension(self):
        for suffix in (".img", ".pdf", ".csv", "", ".txt.exe"):
            with self.subTest(suffix=suffix):
                source = self.root / ("data" + suffix)
                source.write_bytes(b"source data")
                with self.assertRaises(ValueError):
                    feedback.save_report(source, self.snapshot(), title="Failure", description="Details")
                self.assertEqual(source.read_bytes(), b"source data")

    def test_attachment_selection_is_revalidated_if_files_grow_or_disappear_before_save(self):
        source = self.root / "sample.img"
        for suffix in (".txt", ".zip"):
            for change in ("grow", "delete"):
                with self.subTest(suffix=suffix, change=change):
                    source.write_bytes(b"sample")
                    target = self.root / ("report" + suffix)
                    target.write_bytes(b"previous report")
                    with mock.patch.object(feedback, "MAX_ATTACHMENT_BYTES", 100):
                        approved = feedback.validate_attachments([source])
                        if change == "grow":
                            source.write_bytes(b"x" * 101)
                        else:
                            source.unlink()
                        with self.assertRaises(ValueError):
                            feedback.save_report(target, self.snapshot(), title="Failure", description="Details", attachments=approved)
                    self.assertEqual(target.read_bytes(), b"previous report")
                    self.assertEqual(list(self.root.glob(".aidas-feedback-*")), [])

    def test_attachment_total_accepts_ten_megabytes_and_rejects_one_more_byte(self):
        first, second = self.root / "first.img", self.root / "second.img"
        first.write_bytes(b"a" * (5 * 1024 * 1024))
        second.write_bytes(b"b" * (5 * 1024 * 1024))
        self.assertEqual(feedback.validate_attachments([first, second]), (first.resolve(), second.resolve()))
        with second.open("ab") as stream:
            stream.write(b"c")

        with self.assertRaises(ValueError):
            feedback.validate_attachments([first, second])

    def test_growth_after_initial_validation_still_cannot_exceed_attachment_limit(self):
        source = self.root / "sample.img"
        validate = feedback.validate_attachments
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                source.write_bytes(b"sample")
                target = self.root / ("report" + suffix)
                target.write_bytes(b"previous report")

                def select_then_grow(paths):
                    approved = validate(paths)
                    source.write_bytes(b"x" * 101)
                    return approved

                with mock.patch.object(feedback, "MAX_ATTACHMENT_BYTES", 100), mock.patch.object(
                    feedback, "validate_attachments", side_effect=select_then_grow,
                ), self.assertRaises(ValueError):
                    feedback.save_report(target, self.snapshot(), title="Failure", description="Details", attachments=[source])

                self.assertEqual(target.read_bytes(), b"previous report")
                self.assertEqual(source.read_bytes(), b"x" * 101)
                self.assertEqual(list(self.root.glob(".aidas-feedback-*")), [])

    def test_attachment_changed_during_collection_is_rejected_even_if_its_size_is_unchanged(self):
        source = self.root / "sample.img"
        original_open = Path.open
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                source.write_bytes(b"original sample")
                original_stat = source.stat()
                target = self.root / ("report" + suffix)
                target.write_bytes(b"previous report")
                changed = []

                class ChangingReader:
                    def __init__(self, stream):
                        self.stream = stream

                    def __enter__(self):
                        self.stream.__enter__()
                        return self

                    def __exit__(self, *args):
                        return self.stream.__exit__(*args)

                    def __getattr__(self, name):
                        return getattr(self.stream, name)

                    def read(self, *args):
                        chunk = self.stream.read(*args)
                        if chunk and not changed:
                            changed.append(True)
                            source.write_bytes(b"modified sample")
                            os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000))
                        return chunk

                def open_with_concurrent_change(path, *args, **kwargs):
                    stream = original_open(path, *args, **kwargs)
                    mode = args[0] if args else kwargs.get("mode", "r")
                    return ChangingReader(stream) if path == source and mode == "rb" else stream

                with mock.patch.object(Path, "open", new=open_with_concurrent_change), self.assertRaises(ValueError):
                    feedback.save_report(target, self.snapshot(), title="Failure", description="Details", attachments=[source])

                self.assertTrue(changed)
                self.assertEqual(target.read_bytes(), b"previous report")
                self.assertEqual(source.read_bytes(), b"modified sample")
                self.assertEqual(list(self.root.glob(".aidas-feedback-*")), [])

    def test_sharing_link_validation_preserves_https_query_and_rejects_unsafe_urls(self):
        link = "https://files.example.org/shared/data?signature=A%2FB+C%3D&download=1#details"
        self.assertEqual(feedback.validate_sharing_link(""), "")
        self.assertEqual(feedback.validate_sharing_link(link), link)
        for invalid in (
            "http://files.example.org/data", "ftp://files.example.org/data", "file:///C:/data.zip",
            "javascript:alert(1)", "data:text/plain,test", "mailto:someone@example.org",
            "https://", "https:///only-a-path", "https://?download=1", "files.example.org/data",
            "https://name:password@files.example.org/data", "https://name@files.example.org/data",
            "https://files.example.org/a\nb", "https://files.example.org/a\rb",
            "https://files.example.org/a\tb", "https://files.example.org/a\x00b",
        ):
            with self.subTest(link=invalid), self.assertRaises(ValueError):
                feedback.validate_sharing_link(invalid)

    def test_sharing_link_is_explicit_in_text_and_zip_feedback_without_query_redaction(self):
        link = "https://files.example.org/shared/data?token=A%2FB+C%3D&download=1"
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                target = self.root / ("report" + suffix)
                feedback.save_report(
                    target, self.snapshot(), title="Failure", description="Details", sharing_link=link,
                )
                if suffix == ".txt":
                    text = target.read_text(encoding="utf-8")
                else:
                    with zipfile.ZipFile(target) as archive:
                        text = archive.read("feedback.txt").decode("utf-8")
                self.assertIn(link, text)

    def test_invalid_sharing_link_preserves_existing_reports_and_rejects_email_draft(self):
        for suffix in (".txt", ".zip"):
            with self.subTest(suffix=suffix):
                target = self.root / ("report" + suffix)
                target.write_bytes(b"previous report")
                with self.assertRaises(ValueError):
                    feedback.save_report(
                        target, self.snapshot(), title="Failure", description="Details",
                        sharing_link="http://files.example.org/private",
                    )
                self.assertEqual(target.read_bytes(), b"previous report")
                with self.assertRaises(ValueError):
                    feedback.feedback_mailto(
                        "report123", "Failure", target, sharing_link="file:///C:/private/data.zip",
                    )

    def test_email_draft_preserves_plus_address_and_has_no_local_paths_or_auto_attachment(self):
        url = feedback.feedback_mailto("report123", "Step 3 & Step 4", self.root / "report.zip")
        parsed = urlsplit(url)
        self.assertEqual(parsed.scheme, "mailto")
        self.assertEqual(unquote(parsed.path), feedback.FEEDBACK_EMAIL)
        query = parse_qs(parsed.query)
        self.assertIn("[AIDaS Feedback]", query["subject"][0])
        self.assertIn("Step 3 & Step 4", query["subject"][0])
        self.assertIn("report.zip", query["body"][0])
        self.assertIn("report123", query["body"][0])
        self.assertNotIn("Please attach the diagnostic ZIP", query["body"][0])
        self.assertNotIn(str(self.root), unquote(url))
        self.assertNotIn("attachment", query)

    def test_text_email_draft_requests_text_report_and_preserves_secure_share_url(self):
        link = "https://files.example.org/data?signature=A%2FB+C%3D&download=1"
        url = feedback.feedback_mailto(
            "report456", "Step 3 — μ-calibration & save", self.root / "diagnostics.txt", sharing_link=link,
        )

        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.scheme, "mailto")
        self.assertEqual(unquote(parsed.path), feedback.FEEDBACK_EMAIL)
        self.assertIn("report456", query["subject"][0])
        self.assertIn("Step 3 — μ-calibration & save", query["subject"][0])
        body = query["body"][0]
        self.assertIn("report456", body)
        self.assertIn("diagnostics.txt", body)
        self.assertIn("attach", body.lower())
        self.assertIn("text", body.lower())
        self.assertIn(link, body)
        self.assertNotIn(str(self.root), body)
        self.assertFalse({"attach", "attachment", "attachments"}.intersection(query))

    def test_zip_email_draft_identifies_securely_shared_data_instead_of_requesting_zip_attachment(self):
        link = "https://files.example.org/data?signature=A%2FB+C%3D&download=1"
        url = feedback.feedback_mailto("data789", "Failure", self.root / "samples.zip", sharing_link=link)
        query = parse_qs(urlsplit(url).query)
        body = query["body"][0]

        self.assertIn("samples.zip", body)
        self.assertIn("data789", body)
        self.assertIn(link, body)
        self.assertIn("shar", body.lower())
        self.assertNotIn("Please attach the diagnostic ZIP", body)
        self.assertFalse({"attach", "attachment", "attachments"}.intersection(query))

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
