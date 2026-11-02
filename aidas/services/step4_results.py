"""Preserve Step 4 measurements inside the existing TIFF plot stack.

The first TIFF directory's standard ImageDescription tag contains versioned
JSON. Reading it does not decode the plot pixels. Each data row retains the
ImageJ column order, including its one-based ROI index; ``Round`` is column G
(index 6).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence

from PIL import Image


STEP4_RESULTS_FILENAME = "ROI_to_move_stck.tif"
STEP4_STACK_FILENAMES = (STEP4_RESULTS_FILENAME, "ROI_to_move_stck.tiff")
STEP4_RESULTS_HEADERS = (" ", "Major", "Minor", "Angle", "Circ.", "AR", "Round", "Solidity")
_METADATA_KEY = "aidas_step4_measurements"
_METADATA_VERSION = 1


def encode_step4_measurements(rows: Sequence[Mapping[str, float]]) -> str:
    """Encode the same measured values previously stored in the workbook."""

    values = [
        [index, *(float(row[column]) for column in STEP4_RESULTS_HEADERS[1:])]
        for index, row in enumerate(rows, start=1)
    ]
    if not values:
        raise ValueError("Step 4 measurements must contain at least one ROI.")
    return json.dumps(
        {
            _METADATA_KEY: {
                "version": _METADATA_VERSION,
                "columns": STEP4_RESULTS_HEADERS,
                "rows": values,
            }
        },
        allow_nan=False,
        separators=(",", ":"),
    )


def read_step4_measurements(path: str | os.PathLike) -> list[list[int | float]] | None:
    """Read valid measurement rows, or return ``None`` for other TIFF files.

    Ordinary ImageJ TIFF descriptions and unsupported/corrupt metadata are not
    measurement records. The image handle is always explicitly closed, which
    also releases multi-page TIFF handles on Windows.
    """

    image = None
    try:
        image = Image.open(path)
        if image.format != "TIFF":
            return None
        description = image.tag_v2.get(270)
        if isinstance(description, bytes):
            description = description.decode("utf-8")
        if not isinstance(description, str):
            return None
        payload = json.loads(description)
        if not isinstance(payload, dict):
            return None
        metadata = payload.get(_METADATA_KEY)
        if not isinstance(metadata, dict):
            return None
        version = metadata.get("version")
        if type(version) is not int or version != _METADATA_VERSION:
            return None
        if metadata.get("columns") != list(STEP4_RESULTS_HEADERS):
            return None
        rows = metadata.get("rows")
        if not isinstance(rows, list) or not rows:
            return None
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, list) or len(row) != len(STEP4_RESULTS_HEADERS):
                return None
            if type(row[0]) is not int or row[0] != index:
                return None
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in row[1:]):
                return None
        return rows
    except (OSError, ValueError, TypeError, OverflowError):
        return None
    finally:
        if image is not None:
            image.close()


def has_step4_measurements(path: str | os.PathLike) -> bool:
    """Check a TIFF header for the complete embedded measurement record."""

    return read_step4_measurements(path) is not None
