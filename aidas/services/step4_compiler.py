"""Compile measurements from recursively discovered group/subject folders.

This module is the non-UI implementation of the legacy
``rrMCPAR_ELMRPE_ONL_Compiler`` utility.  It preserves the workbook layout,
row grouping, extraction order, and legacy thickness cleanup while retaining
the original ``LIGHT`` text file and saving cleaned ``LIGHT2`` Excel tables.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import fnmatch
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Callable
from zipfile import BadZipFile

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from aidas.utils.filesystem import walk_unique_directories
from aidas.services.step4_results import STEP4_STACK_FILENAMES, read_step4_measurements


ID_ROW_START = 3
RRMCPAR_BLOCK_SIZE = 20
ELMRPE_ONL_BLOCK_COUNT = 23
BIN_SIZE = 100
ELMRPE_COL = 3
ONL_COL = 4
FOVEA_LABEL = "RPEtoOLM"
SHEET_NAMES = ("rrMCP-AR", "ELM-RPE", "ONL")
DEFAULT_OUTPUT_FILENAME = "rrMCPAR_ELMRPE_ONL.xlsx"

PathLike = str | os.PathLike[str]
LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str], None]
SubjectProcessedCallback = Callable[[str, str, str], None]
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeasureCompilation:
    """Included and skipped subject counts for one output sheet."""

    name: str
    included: int
    skipped: int


@dataclass(frozen=True)
class CompilationResult:
    """Summary returned after a compiler workbook is saved."""

    output_path: Path
    group_ids: dict[str, tuple[str, ...]]
    measures: tuple[MeasureCompilation, ...]

    @property
    def le_ids(self) -> tuple[str, ...]:
        """Retain access for callers using an actual LE group."""
        return self.group_ids.get("LE", ())

    @property
    def re_ids(self) -> tuple[str, ...]:
        return self.group_ids.get("RE", ())


def _natural_key(value: str):
    """Sort names naturally without changing their spelling in the workbook."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
    ), value


def _raise_walk_error(error: OSError) -> None:
    raise error


def classify_eye_from_path(path: PathLike) -> str:
    """Classify a subject folder as ``LE``, ``RE``, or ``UNKNOWN``."""

    parts = [part.upper() for part in os.path.normpath(os.fspath(path)).split(os.sep)]
    for part in parts:
        if "LE DONE" in part or part in ("LE", "LEFT") or part.startswith("LE "):
            return "LE"
        if "RE DONE" in part or part in ("RE", "RIGHT") or part.startswith("RE "):
            return "RE"

    joined = " | ".join(parts)
    if re.search(r"\bLE\b", joined):
        return "LE"
    if re.search(r"\bRE\b", joined):
        return "RE"
    return "UNKNOWN"


def find_subfolder_case_insensitive(parent_path: PathLike, target_name: str) -> str | None:
    parent = os.fspath(parent_path)
    if not os.path.isdir(parent):
        return None
    for entry in os.listdir(parent):
        full_path = os.path.join(parent, entry)
        if os.path.isdir(full_path) and entry.strip().lower() == target_name.lower():
            return full_path
    return None


def existing_file(*parts: PathLike) -> str | None:
    path = os.path.join(*(os.fspath(part) for part in parts))
    return path if os.path.isfile(path) else None


def find_results_file(folder_path: PathLike | None) -> str | None:
    """Search a side recursively, preferring current and legacy result names."""

    if not folder_path:
        return None
    folder = os.fspath(folder_path)
    if not os.path.isdir(folder):
        return None

    candidates = []
    for directory, subdirectories, filenames in walk_unique_directories(folder, onerror=_raise_walk_error):
        subdirectories.sort(key=_natural_key)
        for filename in filenames:
            lower = filename.lower()
            path = Path(directory) / filename
            relative = path.relative_to(folder)
            if lower in {name.lower() for name in STEP4_STACK_FILENAMES}:
                if read_step4_measurements(path) is not None:
                    candidates.append((0, len(relative.parts), _natural_key(str(relative)), str(path)))
                continue
            if (not lower.endswith(".xlsx") or filename.startswith("~$")
                    or _is_thickness_filename(filename) or _is_light_profile_file(filename)):
                continue
            rank = 1 if lower == "rr_mcpar.xlsx" else 2 if lower.startswith("results") else 3
            candidates.append((rank, len(relative.parts), _natural_key(str(relative)), str(path)))
    return min(candidates)[-1] if candidates else None


def find_nasal_temporal_results(subject_path: PathLike) -> tuple[str | None, str | None]:
    nasal_dir = find_subfolder_case_insensitive(subject_path, "nasal")
    temporal_dir = find_subfolder_case_insensitive(subject_path, "temporal")
    return find_results_file(nasal_dir), find_results_file(temporal_dir)


def get_first20_col_g_values(filepath: PathLike) -> list:
    if Path(filepath).suffix.lower() in (".tif", ".tiff"):
        rows = read_step4_measurements(filepath)
        return [row[6] for row in (rows or ())][:RRMCPAR_BLOCK_SIZE]
    workbook = load_workbook(os.fspath(filepath), data_only=True)
    try:
        worksheet = workbook.active
        values = []
        for row in worksheet.iter_rows(min_row=2, min_col=7, max_col=7, values_only=True):
            value = row[0]
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            values.append(value)
            if len(values) >= RRMCPAR_BLOCK_SIZE:
                break
        return values
    finally:
        workbook.close()


def extract_rrmcpar_from_subject(subject_path: PathLike):
    nasal_file, temporal_file = find_nasal_temporal_results(subject_path)
    nasal_values = get_first20_col_g_values(nasal_file) if nasal_file else []
    temporal_values = get_first20_col_g_values(temporal_file) if temporal_file else []
    return nasal_values, temporal_values, nasal_file, temporal_file


def parse_float_or_none(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_tsv_rows(filepath: PathLike) -> list[list[str]]:
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore", newline="") as handle:
        rows.extend(csv.reader(handle, delimiter="\t"))
    return rows


def write_tsv_rows(filepath: PathLike, rows: list[list[str]]) -> None:
    with open(filepath, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def block_means(
    values: list[float | None],
    block_count: int = ELMRPE_ONL_BLOCK_COUNT,
    bin_size: int = BIN_SIZE,
) -> list[float | None]:
    means = []
    for index in range(block_count):
        chunk = values[index * bin_size:(index + 1) * bin_size]
        numbers = [value for value in chunk if value is not None]
        means.append(sum(numbers) / len(numbers) if numbers else None)
    return means


def _is_light_profile_file(path: PathLike) -> bool:
    name = os.path.basename(path).lower()
    return "profiles" in name or ("fovea" in name and "thickness" not in name)


def _is_thickness_filename(path: PathLike) -> bool:
    name = os.path.basename(path).lower()
    return all(part in name for part in ("thickness", "distance", "fovea", "light"))


def _light_file_priority(path: PathLike) -> int:
    """Use original text to refresh Excel; retain old cleaned text as fallback."""
    name = os.path.basename(path).lower()
    if "light2" not in name:
        return 0
    return 1 if name.endswith(".xlsx") else 2


def find_thickness_file(search_root: PathLike | None) -> str | None:
    if not search_root:
        return None
    root = os.fspath(search_root)
    if not os.path.isdir(root):
        return None
    pattern_groups = (
        ("*_thickness_vs_distance_from_fovea_LIGHT*.txt",
         "*thickness*distance*fovea*LIGHT*.txt",
         "*thickness*distance*fovea*LIGHT2.xlsx"),
        ("*LIGHT*.txt", "*light*.txt"),
    )
    candidates = []
    for directory, subdirectories, filenames in walk_unique_directories(root, followlinks=True):
        # Match recursive glob's default treatment of hidden files/folders.
        subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]
        subdirectories.sort()
        candidates.extend(
            os.path.join(directory, filename)
            for filename in filenames
            if not filename.startswith(".")
            and (filename.lower().endswith(".txt") or (
                filename.lower().endswith(".xlsx") and _is_thickness_filename(filename)
            ))
            and "light" in filename.lower()
            and not _is_light_profile_file(filename)
            and os.path.isfile(os.path.join(directory, filename))
        )
    candidates.sort()
    for patterns in pattern_groups:
        matches = []
        for pattern in patterns:
            matches.extend(path for path in candidates if fnmatch.fnmatch(
                os.path.basename(path).lower(), pattern.lower(),
            ))
        # A known thickness table takes precedence over a generic LIGHT profile.
        if matches:
            return min(matches, key=lambda path: (_light_file_priority(path), path))
    return None


def _prefer_current_light_file(paths) -> str | None:
    """Keep candidate order within original / cleaned Excel / legacy text."""
    existing = [path for path in paths if path]
    return min(existing, key=_light_file_priority) if existing else None


def _is_legacy_light_file(path: str | None) -> bool:
    return bool(path and "light2" in os.path.basename(path).lower())


def find_light_files(subject_path: PathLike) -> tuple[str | None, str | None]:
    subject = os.fspath(subject_path)
    nasal_candidates = (
        existing_file(subject, "nasal", "_thickness_vs_distance_from_fovea_LIGHT2.txt"),
        existing_file(subject, "nasal", "_thickness_vs_distance_from_fovea_LIGHT2.xlsx"),
        existing_file(subject, "nasal", "_thickness_vs_distance_from_fovea_LIGHT.txt"),
        existing_file(subject, "rrMCPAR", "_thickness_vs_distance_from_fovea_LIGHT2.txt"),
        existing_file(subject, "rrMCPAR", "_thickness_vs_distance_from_fovea_LIGHT2.xlsx"),
        existing_file(subject, "rrMCPAR", "_thickness_vs_distance_from_fovea_LIGHT.txt"),
        existing_file(subject, "_thickness_vs_distance_from_fovea_LIGHT2.txt"),
        existing_file(subject, "_thickness_vs_distance_from_fovea_LIGHT2.xlsx"),
        existing_file(subject, "_thickness_vs_distance_from_fovea_LIGHT.txt"),
    )
    temporal_candidates = (
        existing_file(subject, "temporal", "_thickness_vs_distance_from_fovea_LIGHT2.txt"),
        existing_file(subject, "temporal", "_thickness_vs_distance_from_fovea_LIGHT2.xlsx"),
        existing_file(subject, "temporal", "_thickness_vs_distance_from_fovea_LIGHT.txt"),
        existing_file(subject, "temporal", "rrMCPAR", "_thickness_vs_distance_from_fovea_LIGHT2.txt"),
        existing_file(subject, "temporal", "rrMCPAR", "_thickness_vs_distance_from_fovea_LIGHT2.xlsx"),
        existing_file(subject, "temporal", "rrMCPAR", "_thickness_vs_distance_from_fovea_LIGHT.txt"),
    )

    nasal_file = _prefer_current_light_file(nasal_candidates)
    temporal_file = _prefer_current_light_file(temporal_candidates)

    if not nasal_file or _is_legacy_light_file(nasal_file):
        for folder in ("nasal", "Nasal", "rrMCPAR"):
            nasal_file = _prefer_current_light_file((
                nasal_file, find_thickness_file(os.path.join(subject, folder)),
            ))
            if nasal_file and not _is_legacy_light_file(nasal_file):
                break
    if not temporal_file or _is_legacy_light_file(temporal_file):
        for folder in ("temporal", "Temporal", "temp", "Temp", os.path.join("temporal", "rrMCPAR")):
            temporal_file = _prefer_current_light_file((
                temporal_file, find_thickness_file(os.path.join(subject, folder)),
            ))
            if temporal_file and not _is_legacy_light_file(temporal_file):
                break

    if not nasal_file or not temporal_file or any(
        _is_legacy_light_file(path) for path in (nasal_file, temporal_file)
    ):
        all_text_files = []
        for root, _directories, filenames in walk_unique_directories(subject):
            for filename in filenames:
                lower = filename.lower()
                if (
                    lower.endswith((".txt", ".xlsx"))
                    and "thickness" in lower
                    and "distance" in lower
                    and "fovea" in lower
                    and "light" in lower
                    and not _is_light_profile_file(filename)
                ):
                    all_text_files.append(os.path.join(root, filename))
        all_text_files.sort(key=lambda path: (_light_file_priority(path), path.lower()))
        for filepath in all_text_files:
            relative_parts = Path(os.path.relpath(filepath, subject)).parts[:-1]
            is_temporal = any(part.lower() in ("temporal", "temp") for part in relative_parts)
            if is_temporal:
                temporal_file = _prefer_current_light_file((temporal_file, filepath))
            else:
                nasal_file = _prefer_current_light_file((nasal_file, filepath))

    return nasal_file, temporal_file


def _clean_light_rows(rows: list[list]) -> list[list]:
    """Retain the legacy ELM column-3 filter and the first distance-zero row."""
    if not rows:
        return []
    cleaned = [rows[0]]
    kept_zero = False
    for row in rows[1:]:
        if len(row) < ELMRPE_COL:
            continue
        distance = parse_float_or_none(row[0])
        value = parse_float_or_none(row[ELMRPE_COL - 1])
        if distance == 0 and not kept_zero:
            cleaned.append(row)
            kept_zero = True
            continue
        if value is not None:
            cleaned.append(row)

    return cleaned


def read_thickness_rows(filepath: PathLike) -> list[list]:
    """Read raw/legacy text and cleaned Excel without changing either source."""
    if Path(filepath).suffix.lower() != ".xlsx":
        return read_tsv_rows(filepath)
    workbook = load_workbook(filepath, read_only=True, data_only=True)
    try:
        return [list(row) for row in workbook.active.iter_rows(values_only=True)]
    finally:
        workbook.close()


def _excel_thickness_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    number = parse_float_or_none(value)
    if number is not None:
        # Excel has no numeric NaN/Inf representation. Retain these tokens as
        # text so the compiler preserves its legacy parsing/mean semantics.
        return number if math.isfinite(number) else str(value)
    return str(value)


def _remove_replaced_light2_text(source: Path, destination: Path) -> None:
    """Remove only the obsolete sibling after raw LIGHT has a cleaned workbook."""
    if source.suffix.lower() != ".txt" or "light2" in source.name.lower():
        return
    obsolete = destination.with_suffix(".txt")
    if obsolete.is_file() and obsolete.resolve() != source.resolve():
        try:
            obsolete.unlink()
        except OSError as error:
            _LOGGER.warning("Could not remove obsolete cleaned thickness text %s: %s", obsolete, error)


def _cleaned_light_workbook_path(source: Path) -> Path:
    if source.suffix.lower() == ".xlsx":
        return source
    name = source.stem
    if "light2" not in name.lower():
        name = re.sub(r"LIGHT$", "LIGHT2", name, flags=re.IGNORECASE)
        if not name.lower().endswith("light2"):
            name += "2"
    return source.with_name(name + ".xlsx")


def create_cleaned_light_thickness_workbook(light_path: PathLike | None) -> str | None:
    """Save cleaned LIGHT2.xlsx atomically while preserving the raw LIGHT.txt.

    ONL uses the same retained rows as ELM-RPE, matching the former cleanup.
    Numeric cells retain measurement precision and missing values become blank.
    A supplied historical LIGHT2 text table is already cleaned and is converted
    without filtering it again.
    """
    if not light_path:
        return None
    source = Path(light_path)
    if source.suffix.lower() == ".xlsx":
        return str(source)

    rows = read_tsv_rows(source)
    cleaned = rows if "light2" in source.name.lower() else _clean_light_rows(rows)
    excel_rows = ([list(cleaned[0])] if cleaned else []) + [
        [_excel_thickness_value(value) for value in row] for row in cleaned[1:]
    ]
    destination = _cleaned_light_workbook_path(source)

    # Repeated measure extraction must not rewrite a current Excel workbook.
    if destination.is_file():
        try:
            existing = read_thickness_rows(destination)
        except (OSError, ValueError, KeyError, BadZipFile):
            existing = None
        if existing is not None:
            width = max((len(row) for row in excel_rows), default=0)
            expected = [row + [None] * (width - len(row)) for row in excel_rows]
            if existing == expected:
                _remove_replaced_light2_text(source, destination)
                return str(destination)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "LIGHT2"
    for row in excel_rows:
        worksheet.append(row)
    for row in worksheet:
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    worksheet.freeze_panes = "A2"
    if excel_rows:
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".xlsx", dir=source.parent,
        )
        temporary_path = Path(temporary_name)
        os.close(descriptor)
        workbook.save(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        workbook.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    _remove_replaced_light2_text(source, destination)
    return str(destination)


def clean_light_thickness_file(light_path: PathLike | None) -> str | None:
    """Use cleaned Excel, retaining historical LIGHT2.txt as read-only fallback."""
    if not light_path:
        return None
    if "light2" in os.path.basename(light_path).lower() and Path(light_path).suffix.lower() == ".txt":
        return os.fspath(light_path)
    return create_cleaned_light_thickness_workbook(light_path)


def make_light2_from_light(light_path: PathLike | None, required_col: int) -> str | None:
    """Compatibility entry point; save Excel using ELM-RPE column 3 cleanup."""
    return clean_light_thickness_file(light_path)


def get_column_values(light_path: PathLike | None, column_index: int) -> list[float]:
    if not light_path:
        return []
    path = os.fspath(light_path)
    if not os.path.isfile(path):
        return []
    values = []
    for row in read_thickness_rows(path)[1:]:
        if len(row) < column_index:
            continue
        value = parse_float_or_none(row[column_index - 1])
        if value is not None:
            values.append(value)
    return values


def find_fovea_file(folder_path: PathLike | None) -> str | None:
    if not folder_path:
        return None
    folder = os.fspath(folder_path)
    if not os.path.isdir(folder):
        return None
    candidates = (
        "_fovea_light_profiles_LIGHT.txt",
        "_fovea_light_profiles_LIGHT2.txt",
        "_fovea_light_LIGHT.txt",
        "_fovea_light_LIGHT2.txt",
        "fovea_light_profiles_LIGHT.txt",
        "fovea_light_profiles_LIGHT2.txt",
        "fovea_light_LIGHT.txt",
        "fovea_light_LIGHT2.txt",
    )
    for filename in candidates:
        path = existing_file(folder, filename)
        if path:
            return path
    for filename in os.listdir(folder):
        if "fovea_light" in filename.lower() and filename.lower().endswith(".txt"):
            path = os.path.join(folder, filename)
            if os.path.isfile(path):
                return path

    rr_folder = os.path.join(folder, "rrMCPAR")
    if os.path.isdir(rr_folder):
        for filename in candidates:
            path = existing_file(rr_folder, filename)
            if path:
                return path
        for filename in os.listdir(rr_folder):
            if "fovea_light" in filename.lower() and filename.lower().endswith(".txt"):
                path = os.path.join(rr_folder, filename)
                if os.path.isfile(path):
                    return path
    return None


def extract_fovea_value(filepath: PathLike | None, label: str = FOVEA_LABEL) -> float | None:
    if not filepath:
        return None
    path = os.fspath(filepath)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if parts and parts[0].strip() == label and len(parts) >= 2:
                return parse_float_or_none(parts[1])
    return None


def row_count_for_measure(measure_name: str, include_fovea: bool) -> int:
    if measure_name == "rrMCP-AR":
        return RRMCPAR_BLOCK_SIZE
    if measure_name == "ELM-RPE":
        return ELMRPE_ONL_BLOCK_COUNT + 1 if include_fovea else ELMRPE_ONL_BLOCK_COUNT
    return ELMRPE_ONL_BLOCK_COUNT


def extract_elmrpe_onl_from_subject(
    subject_path: PathLike,
    measure_name: str,
    include_fovea: bool,
):
    column_index = ELMRPE_COL if measure_name == "ELM-RPE" else ONL_COL
    nasal_file, temporal_file = find_light_files(subject_path)
    nasal_light = clean_light_thickness_file(nasal_file) if nasal_file else None
    temporal_light = clean_light_thickness_file(temporal_file) if temporal_file else None

    nasal_values = get_column_values(nasal_light, column_index) if nasal_light else []
    temporal_values = get_column_values(temporal_light, column_index) if temporal_light else []
    nasal_blocks = block_means(nasal_values)
    temporal_blocks = block_means(temporal_values)

    if measure_name == "ELM-RPE" and include_fovea:
        subject = os.fspath(subject_path)
        nasal_fovea_file = find_fovea_file(os.path.join(subject, "nasal")) or find_fovea_file(subject)
        temporal_fovea_file = find_fovea_file(os.path.join(subject, "temporal"))
        nasal_blocks = [extract_fovea_value(nasal_fovea_file)] + nasal_blocks
        temporal_blocks = [extract_fovea_value(temporal_fovea_file)] + temporal_blocks

    row_count = row_count_for_measure(measure_name, include_fovea)
    nasal_blocks = (nasal_blocks + [None] * row_count)[:row_count]
    temporal_blocks = (temporal_blocks + [None] * row_count)[:row_count]
    return (
        nasal_blocks,
        temporal_blocks,
        nasal_light,
        temporal_light,
        len(nasal_values),
        len(temporal_values),
    )


def find_subject_folders(root_folder: PathLike) -> dict[str, dict[str, str]]:
    """Find side measurement records, then use their subject and group names.

    Other subject subfolders never become subjects themselves. A subject with
    only thickness tables is ignored; one side's Step 4 measurements suffice.
    """
    subject_folders: dict[str, dict[str, str]] = {}
    for directory, subdirectories, _filenames in walk_unique_directories(
        os.fspath(root_folder), onerror=_raise_walk_error,
    ):
        subdirectories.sort(key=_natural_key)
        side = Path(directory)
        if side.name.strip().lower() not in ("nasal", "temporal"):
            continue
        # find_results_file handles everything below this side, so the outer
        # walk must not reinterpret a nested directory as another subject.
        subdirectories.clear()
        if not find_results_file(side):
            continue
        subject = side.parent
        group_name = subject.parent.name
        subjects = subject_folders.setdefault(group_name, {})
        previous = subjects.get(subject.name)
        if previous is not None and previous != str(subject):
            raise ValueError(
                f"Duplicate subject {subject.name!r} in group {group_name!r}: "
                f"{previous} and {subject}. Choose a narrower input folder or rename the folders."
            )
        subjects[subject.name] = str(subject)
    return subject_folders


def write_group_block(
    worksheet,
    log_worksheet,
    measure_name: str,
    group_name: str,
    start_column: int,
    subject_folders_for_group: dict[str, str],
    include_fovea: bool,
    subject_processed_callback: SubjectProcessedCallback | None = None,
) -> tuple[int, int]:
    id_column = start_column
    nasal_column = start_column + 1
    temporal_column = start_column + 2

    worksheet.cell(1, id_column).value = group_name
    worksheet.cell(1, id_column).data_type = "s"
    worksheet.cell(1, id_column).font = Font(bold=True)
    worksheet.cell(2, id_column).value = "ID"
    worksheet.cell(2, nasal_column).value = "Nasal"
    worksheet.cell(2, temporal_column).value = "Temporal"
    for column in (id_column, nasal_column, temporal_column):
        worksheet.cell(2, column).font = Font(bold=True)
        worksheet.cell(2, column).alignment = Alignment(horizontal="center")

    row_count = row_count_for_measure(measure_name, include_fovea)
    row = ID_ROW_START
    included = 0
    skipped = 0
    for subject_id in sorted(subject_folders_for_group, key=_natural_key):
        subject_path = subject_folders_for_group[subject_id]
        try:
            if measure_name == "rrMCP-AR":
                nasal_values, temporal_values, nasal_file, temporal_file = extract_rrmcpar_from_subject(
                    subject_path
                )
                nasal_count, temporal_count = len(nasal_values), len(temporal_values)
            else:
                (
                    nasal_values,
                    temporal_values,
                    nasal_file,
                    temporal_file,
                    nasal_count,
                    temporal_count,
                ) = extract_elmrpe_onl_from_subject(subject_path, measure_name, include_fovea)
        except Exception as exc:
            skipped += 1
            log_worksheet.append(
                [measure_name, group_name, subject_id, subject_path, "", "", 0, 0, f"Error: {exc}"]
            )
            if subject_processed_callback is not None:
                subject_processed_callback(measure_name, group_name, subject_id)
            continue

        if not any(value is not None for value in nasal_values) and not any(
            value is not None for value in temporal_values
        ):
            skipped += 1
            log_worksheet.append(
                [
                    measure_name,
                    group_name,
                    subject_id,
                    subject_path,
                    nasal_file or "",
                    temporal_file or "",
                    nasal_count,
                    temporal_count,
                    "Skipped - no values extracted",
                ]
            )
            if subject_processed_callback is not None:
                subject_processed_callback(measure_name, group_name, subject_id)
            continue

        id_cell = worksheet.cell(row, id_column)
        id_cell.value = subject_id
        id_cell.data_type = "s"
        id_cell.font = Font(bold=True)
        id_cell.alignment = Alignment(horizontal="center", vertical="center")
        for index in range(row_count):
            output_row = row + index
            nasal_cell = worksheet.cell(output_row, nasal_column)
            temporal_cell = worksheet.cell(output_row, temporal_column)
            nasal_cell.value = nasal_values[index] if index < len(nasal_values) else None
            temporal_cell.value = temporal_values[index] if index < len(temporal_values) else None
            if isinstance(nasal_cell.value, (int, float)):
                nasal_cell.number_format = "0.000"
            if isinstance(temporal_cell.value, (int, float)):
                temporal_cell.number_format = "0.000"

        log_worksheet.append(
            [
                measure_name,
                group_name,
                subject_id,
                subject_path,
                nasal_file or "",
                temporal_file or "",
                nasal_count,
                temporal_count,
                "OK",
            ]
        )
        included += 1
        row += row_count
        if subject_processed_callback is not None:
            subject_processed_callback(measure_name, group_name, subject_id)

    for column in (id_column, nasal_column, temporal_column):
        worksheet.column_dimensions[get_column_letter(column)].width = 12
    return included, skipped


def write_measure_sheet(
    worksheet,
    log_worksheet,
    measure_name: str,
    subject_folders: dict[str, dict[str, str]],
    include_fovea: bool,
    subject_processed_callback: SubjectProcessedCallback | None = None,
) -> tuple[int, int]:
    included = skipped = 0
    for index, group_name in enumerate(sorted(subject_folders, key=_natural_key)):
        group_included, group_skipped = write_group_block(
            worksheet,
            log_worksheet,
            measure_name,
            group_name,
            1 + index * 4,
            subject_folders[group_name],
            include_fovea,
            subject_processed_callback,
        )
        included += group_included
        skipped += group_skipped
    return included, skipped


def compile_step4_results(
    root_folder: PathLike,
    output_path: PathLike,
    *,
    include_fovea: bool = True,
    log_callback: LogCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CompilationResult:
    """Scan ``root_folder`` and save the formatted combined workbook."""

    root = Path(root_folder).expanduser()
    output = Path(output_path).expanduser()
    if not root.is_dir():
        raise ValueError("Choose a valid input folder containing group and subject folders.")
    if output.suffix.lower() != ".xlsx":
        raise ValueError("The compiled output filename must end with .xlsx.")
    if not output.parent.is_dir():
        raise ValueError("The selected output folder does not exist.")

    def emit(message: str) -> None:
        if log_callback is not None:
            log_callback(message)

    def report_progress(completed: int, total: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(int(completed), int(total), message)

    emit(f"Root folder: {root}")
    emit(f"Include fovea in ELM-RPE: {bool(include_fovea)}\n")
    emit("Searching recursively for nasal and temporal Step 4 measurements...")
    report_progress(0, 1, "Scanning input folders...")
    subject_folders = find_subject_folders(root)
    group_ids = {
        group: tuple(sorted(subject_folders[group], key=_natural_key))
        for group in sorted(subject_folders, key=_natural_key)
    }
    for group_name, ids in group_ids.items():
        emit(f"{group_name} subject(s) found: {', '.join(ids)}")
    if not group_ids:
        raise ValueError(
            "No subject folders with nasal or temporal Step 4 measurements were found. "
            "Expected: group / subject / nasal or temporal / ROI_to_move_stck.tif "
            "with saved measurements, or a compatible legacy Excel workbook (.xlsx). "
            "Folders without Step 4 measurements are ignored."
        )

    # A custom destination must never replace a source measurement record.
    for subjects in subject_folders.values():
        for subject in subjects.values():
            for source_file in find_nasal_temporal_results(subject):
                if source_file and Path(source_file).resolve() == output.resolve():
                    raise ValueError("Choose an output path different from the input measurement result files.")
            # Thickness is now also an Excel input. Protect both an existing
            # selected table and the destination about to be created from raw
            # LIGHT text, before any extraction or save can change these files.
            for thickness_file in find_light_files(subject):
                if not thickness_file:
                    continue
                cleaned_output = _cleaned_light_workbook_path(Path(thickness_file))
                if cleaned_output.resolve() == output.resolve():
                    raise ValueError(
                        "Choose an output path different from the input thickness Excel files."
                    )

    total_subjects = sum(len(ids) for ids in group_ids.values())
    total_steps = 2 + (len(SHEET_NAMES) * total_subjects)
    completed_steps = 1
    report_progress(
        completed_steps,
        total_steps,
        f"Found {total_subjects} subject folder{'s' if total_subjects != 1 else ''}.",
    )

    def subject_processed(measure_name: str, group_name: str, subject_id: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        report_progress(
            completed_steps,
            total_steps,
            f"Processed {measure_name}: {group_name} {subject_id}.",
        )

    workbook = Workbook()
    measure_summaries = []
    try:
        workbook.remove(workbook.active)
        log_worksheet = workbook.create_sheet("Extraction_log")
        log_worksheet.append(
            [
                "Measure",
                "Group",
                "ID",
                "Subject folder",
                "Nasal file used",
                "Temporal file used",
                "Nasal raw values read",
                "Temporal raw values read",
                "Status",
            ]
        )

        for measure_name in SHEET_NAMES:
            worksheet = workbook.create_sheet(measure_name)
            included, skipped = write_measure_sheet(
                worksheet,
                log_worksheet,
                measure_name,
                subject_folders,
                bool(include_fovea),
                subject_processed,
            )
            measure_summaries.append(MeasureCompilation(measure_name, included, skipped))
            emit(f"{measure_name}: included {included}, skipped {skipped}")

        workbook.move_sheet(log_worksheet, offset=len(workbook.worksheets) - 1)
        for cell in log_worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        for row in log_worksheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
        for column in range(1, log_worksheet.max_column + 1):
            log_worksheet.column_dimensions[get_column_letter(column)].width = 28

        report_progress(completed_steps, total_steps, "Saving workbook...")
        workbook.save(output)
    finally:
        workbook.close()

    report_progress(total_steps, total_steps, "Saved compiled workbook.")
    emit(f"\nSaved: {output}")
    return CompilationResult(
        output_path=output,
        group_ids=group_ids,
        measures=tuple(measure_summaries),
    )


__all__ = [
    "CompilationResult",
    "DEFAULT_OUTPUT_FILENAME",
    "MeasureCompilation",
    "block_means",
    "classify_eye_from_path",
    "compile_step4_results",
    "create_cleaned_light_thickness_workbook",
    "extract_fovea_value",
    "find_results_file",
    "find_subject_folders",
    "row_count_for_measure",
]
