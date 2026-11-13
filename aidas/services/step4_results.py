"""Preserve Step 4 measurements in the TIFF stack and side-folder workbook.

The first TIFF directory's standard ImageDescription tag contains versioned
JSON. Reading it does not decode the plot pixels. Each data row retains the
ImageJ column order, including its one-based ROI index; ``Round`` is column G
(index 6).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from collections.abc import Iterable, Mapping, Sequence

from PIL import Image


STEP4_RESULTS_FILENAME = "ROI_to_move_stck.tif"
STEP4_STACK_FILENAMES = (STEP4_RESULTS_FILENAME, "ROI_to_move_stck.tiff")
STEP4_RESULTS_HEADERS = (" ", "Major", "Minor", "Angle", "Circ.", "AR", "Round", "Solidity")
_METADATA_KEY = "aidas_step4_measurements"
_METADATA_VERSION = 1
_EXCLUSION_METADATA_VERSION = 2


def _validate_excluded_indices(indices: Iterable[int], row_count: int) -> set[int]:
    """Require explicit, unique one-based ROI numbers within the saved stack."""
    excluded = set()
    for index in indices:
        if type(index) is not int or not 1 <= index <= row_count or index in excluded:
            raise ValueError("Excluded ROI indices must be unique one-based ROI numbers.")
        excluded.add(index)
    return excluded


def encode_step4_measurements(
    rows: Sequence[Mapping[str, float] | None],
    *,
    excluded_roi_indices: Iterable[int] = (),
) -> str:
    """Encode each ROI in place, leaving explicitly excluded measurements blank.

    Exclusions use version 2 so older readers cannot accidentally export them.
    Ordinary records retain the original version 1 representation.
    """

    if not rows:
        raise ValueError("Step 4 measurements must contain at least one ROI.")
    excluded = _validate_excluded_indices(excluded_roi_indices, len(rows))

    values = [
        [index, *(
            [None] * (len(STEP4_RESULTS_HEADERS) - 1)
            if index in excluded
            else [float(row[column]) for column in STEP4_RESULTS_HEADERS[1:]]
        )]
        for index, row in enumerate(rows, start=1)
    ]
    metadata = {
        "version": _EXCLUSION_METADATA_VERSION if excluded else _METADATA_VERSION,
        "columns": STEP4_RESULTS_HEADERS,
        "rows": values,
    }
    if excluded:
        metadata["excluded_rois"] = sorted(excluded)
    return json.dumps(
        {_METADATA_KEY: metadata},
        allow_nan=False,
        separators=(",", ":"),
    )


def write_step4_measurement_workbook(
    path: str | os.PathLike,
    rows: Sequence[Mapping[str, float] | None],
    *,
    excluded_roi_indices: Iterable[int] = (),
    create_parent: bool = True,
    validate_source=None,
) -> None:
    """Atomically save the historical ImageJ-style per-folder workbook.

    The ``in`` worksheet retains every one-based ROI slot, with ``Round`` in
    column G. Excluded slots contain only their ROI number. Reusing the TIFF
    encoder gives both representations the same values and validation rules.
    An existing workbook survives serialization, disk-write, and replacement
    failures, including an Excel lock on Windows.
    """

    measurement_values = json.loads(encode_step4_measurements(
        rows, excluded_roi_indices=excluded_roi_indices,
    ))[_METADATA_KEY]["rows"]
    from openpyxl import Workbook

    output = Path(path)
    if validate_source is not None:
        validate_source()
    if create_parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    temporary_path = None
    try:
        try:
            worksheet = workbook.active
            worksheet.title = "in"
            worksheet.append(STEP4_RESULTS_HEADERS)
            for row_number, values in enumerate(measurement_values, start=2):
                for column, value in enumerate(values, start=1):
                    cell = worksheet.cell(row=row_number, column=column)
                    if value is not None:
                        # openpyxl normally limits numeric serialization to
                        # 16 significant digits. Explicit numeric tokens keep
                        # the full float round trip used by the TIFF encoder.
                        cell.value = repr(value)
                        cell.data_type = "n"
            # The stream's context closes its Windows file handle even if the
            # workbook serializer raises before closing its own ZIP archive.
            with NamedTemporaryFile(
                mode="w+b", dir=output.parent,
                prefix=f".{output.name}.", suffix=".tmp.xlsx", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                workbook.save(temporary)
        finally:
            workbook.close()
        if validate_source is not None:
            validate_source()
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_step4_measurements(path: str | os.PathLike) -> list[list[int | float | None]] | None:
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
        if type(version) is not int or version not in (_METADATA_VERSION, _EXCLUSION_METADATA_VERSION):
            return None
        if metadata.get("columns") != list(STEP4_RESULTS_HEADERS):
            return None
        rows = metadata.get("rows")
        if not isinstance(rows, list) or not rows:
            return None
        if version == _EXCLUSION_METADATA_VERSION:
            excluded_indices = metadata.get("excluded_rois")
            if not isinstance(excluded_indices, list):
                return None
            excluded = _validate_excluded_indices(excluded_indices, len(rows))
        else:
            if "excluded_rois" in metadata:
                return None
            excluded = set()
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, list) or len(row) != len(STEP4_RESULTS_HEADERS):
                return None
            if type(row[0]) is not int or row[0] != index:
                return None
            if any(
                not (index in excluded and value is None)
                and (type(value) not in (int, float) or not math.isfinite(value))
                for value in row[1:]
            ):
                return None
            if index in excluded:
                # The marker remains authoritative even if a producer retained
                # the original numeric measurements in an excluded row.
                row[1:] = [None] * (len(STEP4_RESULTS_HEADERS) - 1)
        return rows
    except (OSError, ValueError, TypeError, OverflowError):
        return None
    finally:
        if image is not None:
            image.close()


def has_step4_measurements(path: str | os.PathLike) -> bool:
    """Check a TIFF header for the complete embedded measurement record."""

    return read_step4_measurements(path) is not None
