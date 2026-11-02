"""Bound complete Step 5 scans without limiting legitimate folder depth."""

from __future__ import annotations

import os
from typing import Callable

from aidas.utils.filesystem import walk_unique_directories


MAX_MEASUREMENT_SUBFOLDERS = 500


class MeasurementScanLimitExceeded(ValueError):
    """The selected tree is larger than Step 5 can accept."""

    def __init__(self):
        super().__init__(
            "The selected folder contains too many subfolders and exceeds the "
            f"limit of {MAX_MEASUREMENT_SUBFOLDERS}. Please select another folder."
        )


def walk_measurement_directories(
    root: str | os.PathLike,
    *,
    onerror: Callable[[OSError], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
):
    """Yield a cycle-safe complete scan with at most 500 unique subfolders.

    The selected root does not consume the budget. Count physical directories
    once, including linked descendants the compiler can read, without imposing
    a depth limit.
    Cancellation is checked before the limit so a superseded scan stays silent.
    """
    for subfolders, row in enumerate(walk_unique_directories(root, onerror=onerror, followlinks=True)):
        if check_cancelled is not None:
            check_cancelled()
        if subfolders > MAX_MEASUREMENT_SUBFOLDERS:
            raise MeasurementScanLimitExceeded()
        yield row


def validate_measurement_subfolder_limit(
    root: str | os.PathLike,
    *,
    onerror: Callable[[OSError], None] | None = None,
) -> int:
    """Check the entire unpruned tree before any measurement extraction writes."""
    subfolders = 0
    for subfolders, _row in enumerate(walk_measurement_directories(root, onerror=onerror)):
        pass
    return subfolders


__all__ = [
    "MAX_MEASUREMENT_SUBFOLDERS",
    "MeasurementScanLimitExceeded",
    "validate_measurement_subfolder_limit",
    "walk_measurement_directories",
]
