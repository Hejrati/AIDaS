"""Filesystem helpers shared by batch directory scanners."""

from __future__ import annotations

import os
from pathlib import Path


DirectoryScanError = tuple[Path, str]


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
