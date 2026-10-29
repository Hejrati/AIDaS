"""Reviewable, local diagnostic reports for user-sent feedback email."""
from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import tempfile
import threading
import traceback
from tkinter import messagebox
from urllib.parse import quote, urlencode
import uuid
import zipfile

from aidas import __version__
from aidas.core.config import Config
from aidas.utils.log_paths import app_base_dir

FEEDBACK_EMAIL = "b.hejrati+AIDaSFeedBack@wayne.edu"
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_REPORT_BYTES = 15 * 1024 * 1024
MAX_LOG_BYTES = 256 * 1024
_events = deque(maxlen=100)
_lock = threading.Lock()
_installed = False
_logger = None


def redact(text, private_paths=()):
    """Best-effort text scrubbing; selected binary data require user review."""
    text = str(text)
    # Remove complete paths before shortening known prefixes, so suffixes do
    # not accidentally retain identifying directory/file names.
    text = re.sub(r"[A-Za-z]:[\\/][^\r\n\"'<>]*|\\\\[^\r\n\"'<>]+", "<path>", text)
    text = re.sub(r"(?<=[\"'])/(?:[^\"'\r\n]+)", "<path>", text)
    text = re.sub(r"(?<![\w:/])/(?:home|Users|tmp|var|mnt|media|Volumes|opt|data|srv)/[^\r\n\"'<>]*", "<path>", text)
    for path in sorted({str(Path.home()), *(str(p) for p in private_paths if p)}, key=len, reverse=True):
        for spelling in {path, path.replace("\\", "/"), path.replace("/", "\\")}:
            text = re.sub(re.escape(spelling), "<path>", text, flags=re.I)
    text = re.sub(r"(?i)\b(?:bearer\s+)[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", text)
    text = re.sub(r"(?im)([\"']?(?:password|passwd|token|secret|api[_-]?key|authorization)[\"']?\s*[:=]\s*)([^\r\n]+)",
                  r"\1<redacted>", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "<email>", text)
    # Traceback paths, R command paths, and UNC paths can contain patient names.
    return text


def _clean(value, private_paths):
    if isinstance(value, dict):
        return {key: ("<redacted>" if re.search(r"password|token|secret|api.?key", key, re.I)
                      else _clean(item, private_paths)) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item, private_paths) for item in value]
    return redact(value, private_paths) if isinstance(value, str) else value


def record_event(kind, detail):
    event = {"time_utc": datetime.now(timezone.utc).isoformat(), "kind": kind,
             "detail": redact(detail)[-16384:]}
    with _lock:
        _events.append(event)
    if _logger is not None:
        try:
            _logger.info(json.dumps(event))
        except Exception:
            pass  # Diagnostics must never interrupt the workflow.


def record_exception(exc_type, value, tb, *, source="python"):
    record_event(source, "".join(traceback.format_exception(exc_type, value, tb)))


def install_exception_capture():
    """Capture uncaught errors without altering normal exception handling."""
    global _installed, _logger
    if _installed:
        return
    _installed = True
    try:
        directory = Config.CONFIG_DIR / "diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(directory / "session.log", maxBytes=1024*1024,
                                      backupCount=2, encoding="utf-8")
        _logger = logging.getLogger("aidas.feedback")
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        _logger.addHandler(handler)
    except OSError:
        _logger = None
    previous_sys = sys.excepthook
    previous_thread = threading.excepthook
    previous_showerror = messagebox.showerror

    def system_hook(exc_type, value, tb):
        try:
            if not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
                record_exception(exc_type, value, tb)
        finally:
            previous_sys(exc_type, value, tb)

    def thread_hook(args):
        try:
            if not issubclass(args.exc_type, (KeyboardInterrupt, SystemExit)):
                record_exception(args.exc_type, args.exc_value, args.exc_traceback, source="thread")
        finally:
            previous_thread(args)

    sys.excepthook = system_hook
    threading.excepthook = thread_hook
    def showerror(title=None, message=None, **options):
        try:
            record_event("error_dialog", f"{title or ''}\n{message or ''}")
        except Exception:
            pass
        return previous_showerror(title, message, **options)
    messagebox.showerror = showerror
    record_event("session", f"AIDaS {__version__} started")


def _script_info(path):
    if not path:
        return {"selected": False}
    path = Path(path)
    info = {"name": path.name, "exists": path.is_file()}
    if path.is_file():
        info["bytes"] = path.stat().st_size
        if info["bytes"] <= 10*1024*1024:
            info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return info


def collect_diagnostics(context):
    """Runs off the UI thread; receives a plain snapshot, never Tk widgets."""
    private_paths = context.get("private_paths", ())
    packages = {}
    for name in ("numpy", "scipy", "Pillow", "customtkinter", "matplotlib", "pyreadr",
                 "onnxruntime", "openpyxl", "pandas"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            module = sys.modules.get({"Pillow": "PIL"}.get(name, name))
            packages[name] = getattr(module, "__version__", "version metadata unavailable")
    r = {"version": "not available", "packages": {}}
    executable = context.get("rscript")
    if executable and Path(executable).is_file() and Path(executable).name.lower() in {"rscript", "rscript.exe", "rterm.exe", "r.exe"}:
        try:
            result = subprocess.run([str(executable), "--version"], capture_output=True,
                                    text=True, errors="replace", timeout=5,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            r["version"] = redact((result.stdout + result.stderr).strip(), private_paths)
        except (OSError, subprocess.TimeoutExpired) as exc:
            r["version"] = f"Version check unavailable: {type(exc).__name__}"
    library = context.get("r_library")
    if library:
        for package in ("AnalyzeFMRI", "RNiftyReg", "RNifti", "R.matlab", "R.oo", "R.utils", "R.methodsS3", "fastICA", "Rcpp"):
            try:
                description = (Path(library) / package / "DESCRIPTION").read_text(errors="replace")
                match = re.search(r"^Version:\s*(.+)$", description, re.M)
                r["packages"][package] = match.group(1).strip() if match else "unknown"
            except OSError:
                r["packages"][package] = "not found in configured library"
    for role, path in context.get("scripts", {}).items():
        try:
            r[role + "_script"] = _script_info(path)
        except OSError as exc:
            r[role + "_script"] = {"error": type(exc).__name__}
    with _lock:
        events = list(_events)
    report = {
        "schema_version": 1, "report_id": uuid.uuid4().hex[:12],
        "created_utc": datetime.now(timezone.utc).isoformat(), "app_version": __version__,
        "system": {"os": platform.system(), "release": platform.release(),
                   "os_version": platform.version(), "machine": platform.machine(),
                   "python": platform.python_version(), "python_bits": struct.calcsize("P")*8,
                   "logical_cpus": os.cpu_count(), "packaged_executable": bool(getattr(sys, "frozen", False))},
        "python_packages": packages, "r": r,
        "workflow": context.get("workflow", {}), "settings": context.get("settings", {}),
        "recent_events": events,
    }
    report["input_files"] = {}
    for step, path in context.get("inputs", {}).items():
        try:
            item = Path(path)
            report["input_files"][step] = {"extension": item.suffix, "exists": item.is_file(),
                                           "bytes": item.stat().st_size if item.is_file() else None}
        except OSError:
            report["input_files"][step] = {"exists": False}
    if context.get("model"):
        try:
            report["ai_model"] = _script_info(context["model"])
        except OSError as exc:
            report["ai_model"] = {"error": type(exc).__name__}
    # Serialize and scrub only whitelisted context, never the complete preferences/environment.
    report = _clean(report, private_paths)
    logs = []
    directories = [app_base_dir() / "logs", Config.CONFIG_DIR / "diagnostics"]
    candidates = []
    for directory in directories:
        try:
            candidates.extend(p for p in directory.glob("*.log*") if p.is_file() and not p.is_symlink())
        except OSError:
            continue
    def modified(path):
        try:
            return path.stat().st_mtime
        except OSError:
            return 0
    for path in sorted(candidates, key=modified, reverse=True)[:6]:
        try:
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size-MAX_LOG_BYTES))
                content = handle.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
            logs.append({"name": f"log_{len(logs)+1:02d}.txt", "text": redact(content, private_paths)})
        except OSError:
            continue
    return {"diagnostics": report, "logs": logs}


def report_preview(snapshot, include_logs=True):
    text = json.dumps(snapshot["diagnostics"], indent=2, ensure_ascii=False)
    if include_logs:
        for log in snapshot["logs"]:
            text += f"\n\n--- {log['name']} ---\n{log['text']}"
    return text


def validate_attachments(paths):
    paths = tuple(dict.fromkeys(Path(p).resolve() for p in paths))
    total = 0
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Attachment is no longer available: {path.name}")
        total += path.stat().st_size
    if total > MAX_ATTACHMENT_BYTES:
        raise ValueError("Selected files exceed 10 MB. Choose a smaller sample or share larger data separately.")
    return paths


def save_report(target, snapshot, *, title, description, attachments=(), include_logs=True,
                cancel_event=None):
    """Write an atomic ZIP; never modify originals or embed their full paths."""
    paths = validate_attachments(attachments)
    target = Path(target)
    if target.suffix.lower() != ".zip":
        raise ValueError("Save the diagnostic report with a .zip extension.")
    if target.resolve() in paths:
        raise ValueError("The report cannot overwrite one of its attachments.")
    descriptor, temporary = tempfile.mkstemp(prefix=".aidas-feedback-", suffix=".zip", dir=target.parent)
    os.close(descriptor)
    total = 0
    manifest = []
    groups = {}
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("diagnostics.json", json.dumps(snapshot["diagnostics"], indent=2, ensure_ascii=False))
            archive.writestr("feedback.txt", f"Title: {redact(title)}\n\n{redact(description)}\n")
            archive.writestr("README.txt", "AIDaS diagnostic report. Generated text is scrubbed on a best-effort basis.\n"
                             "User-selected files are included unchanged and require manual review.\n"
                             "Attachment names are anonymized; matching .hdr/.img stems are preserved.\n")
            if include_logs:
                for log in snapshot["logs"]:
                    archive.writestr("logs/" + log["name"], log["text"])
            for path in paths:
                group = str(path.with_suffix(""))
                groups.setdefault(group, len(groups)+1)
                suffix = path.suffix.lower()
                # Only a short extension enters an archive member name.
                suffix = suffix if re.fullmatch(r"\.[a-z0-9]{1,12}", suffix) else ".dat"
                name = f"data/sample_{groups[group]:03d}{suffix}"
                digest = hashlib.sha256()
                size = 0
                with path.open("rb") as source, archive.open(name, "w") as output:
                    before = os.fstat(source.fileno())
                    while chunk := source.read(1024*1024):
                        if cancel_event is not None and cancel_event.is_set():
                            raise CancelledError()
                        total += len(chunk)
                        size += len(chunk)
                        if total > MAX_ATTACHMENT_BYTES:
                            raise ValueError("Attachments grew beyond the 10 MB limit while collecting them.")
                        digest.update(chunk)
                        output.write(chunk)
                    after = os.fstat(source.fileno())
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("An attachment changed while collecting it. Wait until processing finishes and try again.")
                manifest.append({"file": name, "bytes": size, "sha256": digest.hexdigest(),
                                 "role_hint": "marked input" if "marked" in path.stem.lower() else "user-selected file"})
            archive.writestr("attachments.json", json.dumps(manifest, indent=2))
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        if Path(temporary).stat().st_size > MAX_REPORT_BYTES:
            raise ValueError("The ZIP exceeds the 15 MB report limit.")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def feedback_mailto(report_id, title, filename):
    body = (f"AIDaS feedback\nReport ID: {report_id}\nApp version: {__version__}\n\n"
            "Please describe the problem and how to reproduce it:\n\n\n"
            f"Please attach the diagnostic ZIP: {Path(filename).name}\n"
            "The app does not attach or send files automatically.\n")
    clean_title = " ".join(redact(title).split())[:120]
    query = urlencode({"subject": f"[AIDaS Feedback] {clean_title} [{report_id}]", "body": body}, quote_via=quote)
    return f"mailto:{quote(FEEDBACK_EMAIL, safe='@+')}?{query}"
