"""Filesystem helpers shared by batch directory scanners."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


DirectoryScanError = tuple[Path, str]


def normalize_path_input(value: str | os.PathLike) -> str:
    """Accept paths typed or pasted from a file manager's Copy as path action."""
    text = os.fspath(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return os.path.expandvars(os.path.expanduser(text))


def existing_directory(value: str | os.PathLike | None) -> Path | None:
    """Resolve a folder address without accepting an empty path as cwd."""
    if value is None:
        return None
    try:
        text = normalize_path_input(value)
        if text:
            path = Path(text)
            if path.is_dir():
                return path.resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    return None


def find_candidate_directories(
    root: str | os.PathLike,
    required_name_groups: Iterable[Iterable[str]],
) -> tuple[list[Path], int, list[DirectoryScanError]]:
    """Prefilter batch inputs using filenames already returned by os.walk.

    Each group contains alternative filenames; every group must match. Callers
    retain their existing file/type, pair, and freshness checks. No results are
    cached between scans.
    """
    root = Path(root)
    groups = tuple({name.lower() for name in group} for group in required_name_groups)
    candidates = []
    errors = []
    scanned = 0

    def record_error(exc):
        errors.append((Path(exc.filename) if exc.filename else root, str(exc)))

    for folder, dirnames, filenames in os.walk(root, topdown=True, onerror=record_error):
        dirnames.sort(key=str.lower)
        scanned += 1
        names = {name.lower() for name in filenames}
        if all(names.intersection(group) for group in groups):
            candidates.append(Path(folder))
    return candidates, scanned, errors


def find_sdb_directories(
    root: str | os.PathLike,
    *,
    cancel_event=None,
) -> tuple[dict[Path, list[Path]], list[DirectoryScanError]]:
    """Return folders below *root* that directly contain ``.sdb`` files.

    The selected root is included when it contains SDB files itself. File and
    directory ordering is deterministic and case-insensitive so the result can
    be used directly by the Step 1 work-queue browser.
    """

    root = Path(root)
    matches: dict[Path, list[Path]] = {}
    errors: list[DirectoryScanError] = []

    def record_error(exc: OSError) -> None:
        path = Path(exc.filename) if exc.filename else root
        errors.append((path, str(exc)))

    for folder, dirnames, filenames in os.walk(root, topdown=True, onerror=record_error):
        if cancel_event is not None and cancel_event.is_set():
            break
        dirnames.sort(key=str.lower)
        sdb_names = sorted(
            (name for name in filenames if name.lower().endswith(".sdb")),
            key=str.lower,
        )
        if sdb_names:
            directory = Path(folder)
            matches[directory] = [directory / name for name in sdb_names]

    return matches, errors


def walk_accessible_directories(root: str | os.PathLike) -> tuple[list[Path], list[DirectoryScanError]]:
    """Return readable directories below *root* while recording skipped branches.

    ``os.walk`` continues with sibling directories after its ``onerror``
    callback runs.  Keeping the errors as data lets background scanners finish
    normally and lets the Tk UI display one warning on the main thread.
    """

    root = Path(root)
    directories: list[Path] = []
    errors: list[DirectoryScanError] = []

    def record_error(exc: OSError) -> None:
        path = Path(exc.filename) if exc.filename else root
        errors.append((path, str(exc)))

    for folder, dirnames, _filenames in os.walk(root, topdown=True, onerror=record_error):
        dirnames.sort(key=str.lower)
        directories.append(Path(folder))

    return directories, errors


def skipped_directories_warning(errors: list[DirectoryScanError], *, limit: int = 5) -> str:
    """Format a concise warning for directories omitted from a scan."""

    unique: list[DirectoryScanError] = []
    seen: set[str] = set()
    for path, reason in errors:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append((Path(path), reason))

    lines = [
        f"Skipped {len(unique)} folder(s) that could not be opened. The rest of the scan completed.",
        "",
    ]
    lines.extend(f"- {path}: {reason}" for path, reason in unique[:limit])
    if len(unique) > limit:
        lines.append(f"- ...and {len(unique) - limit} more")
    return "\n".join(lines)
