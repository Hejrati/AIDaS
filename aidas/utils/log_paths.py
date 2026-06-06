from __future__ import annotations

import sys
from pathlib import Path


def app_base_dir() -> Path:
    """Return the installed executable directory, or the project root in source runs."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def app_log_dir() -> Path:
    path = app_base_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
