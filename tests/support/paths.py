"""Stable paths shared by tests, independent of package nesting and cwd."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = PROJECT_ROOT / "tests"
