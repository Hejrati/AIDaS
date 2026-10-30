"""Read-only folder inventory for the Step 5 directory visualization."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

from aidas.utils.filesystem import walk_unique_directories


@dataclass(frozen=True)
class DirectoryNode:
    path: Path
    name: str
    kind: str
    detail: str
    children: tuple[DirectoryNode, ...] = ()
    important_files: int = 0
    total_files: int = 0
    excel_files: int = 0


@dataclass(frozen=True)
class DirectoryInventory:
    root: DirectoryNode
    workbooks: int
    thickness_files: int
    fovea_files: int
    unassigned_workbooks: int
    errors: tuple[str, ...]


class InventoryCancelled(Exception):
    """A newer directory selection superseded this inventory."""


def _file_kind(path: Path, output_path: Path | None) -> tuple[str, str] | None:
    name = path.name.lower()
    if name.startswith("~$"):
        return None
    if name.endswith(".xlsx") and (path == output_path or name in (
        "rrmcpar_elmrpe_onl.xlsx", "le_re_rrmcpar_elmrpe_onl.xlsx",
    )):
        return "output", "Compiled Excel workbook"
    if name.endswith(".xlsx"):
        detail = "Step 4 measurements" if name == "rr_mcpar.xlsx" or name.startswith("results") else "Excel workbook"
        return "excel", detail
    if name.endswith(".txt") and "light" in name:
        if "thickness" in name and "distance" in name and "fovea" in name:
            return "thickness", "ELM-RPE / ONL thickness" + (" · cleaned" if "light2" in name else "")
        if "fovea_light" in name:
            return "fovea", "Fovea measurement"
    if name in ("_flat_light.hdr", "_flat_light.img"):
        return "image", "Flattened image" + (" header" if name.endswith(".hdr") else " data")
    if name in ("roi_to_move_stck.tif", "max_stack.tif"):
        return "image", "Step 4 image stack"
    return None


def scan_measurement_directory(
    root: Path,
    *,
    output_path: Path | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> DirectoryInventory:
    """Inventory actual files, retaining empty branches as compact folder rows.

    Flat workbooks are displayed even though the compiler requires side folders.
    No workbooks or measurement tables are opened or modified by this scan.
    """
    root = Path(root).resolve()
    output_path = Path(output_path).resolve() if output_path is not None else None
    directories = []
    errors = []

    def on_error(error):
        errors.append(str(error))

    for directory, subdirectories, files in walk_unique_directories(root, onerror=on_error):
        if cancelled():
            raise InventoryCancelled()
        subdirectories.sort(key=str.casefold)
        directories.append((Path(directory), tuple(subdirectories), tuple(files)))

    nodes = {}
    subject_paths = set()
    group_paths = set()
    workbooks = thickness = fovea = unassigned = 0
    for directory, subdirectories, files in reversed(directories):
        if cancelled():
            raise InventoryCancelled()
        children = []
        important_count = 0
        excel_count = 0
        total_count = len(files)
        for name in subdirectories:
            path = directory / name
            child = nodes.pop(path, DirectoryNode(path, name, "folder", "Folder unavailable"))
            children.append(child)
            important_count += child.important_files
            excel_count += child.excel_files
            total_count += child.total_files
        other_files = 0
        important_children = []
        for name in sorted(files, key=str.casefold):
            path = directory / name
            classification = _file_kind(path, output_path)
            if classification is None:
                other_files += 1
                continue
            kind, detail = classification
            if kind == "excel":
                workbooks += 1
                side_folder = next((parent for parent in path.parents if parent.name.strip().lower() in ("nasal", "temporal")), None)
                if side_folder is None:
                    unassigned += 1
                    detail += " · no nasal/temporal folder"
                else:
                    subject_paths.add(side_folder.parent)
                    group_paths.add(side_folder.parent.parent)
            thickness += kind == "thickness"
            fovea += kind == "fovea"
            important_count += 1
            is_excel = kind in ("excel", "output")
            excel_count += is_excel
            important_children.append(DirectoryNode(path, name, kind, detail, important_files=1, total_files=1, excel_files=int(is_excel)))
        priority = {"excel": 0, "output": 1, "thickness": 2, "fovea": 3, "image": 4}
        children.extend(sorted(important_children, key=lambda node: (priority[node.kind], node.name.casefold())))
        if other_files:
            children.append(DirectoryNode(directory, f"… {other_files} other files", "other", "Not shown individually"))
        side = directory.name.strip().lower()
        detail = "Nasal" if side == "nasal" else "Temporal" if side == "temporal" else "Folder"
        if directory in subject_paths:
            detail = "Subject ID"
        elif directory in group_paths:
            detail = "Group"
        if not excel_count:
            detail = "No recorded Excel results"
        nodes[directory] = DirectoryNode(
            directory, directory.name or str(directory), "folder", detail,
            tuple(children) if important_count or directory == root else (),
            important_count, total_count, excel_count,
        )
    node = nodes.get(root, DirectoryNode(root, root.name or str(root), "folder", "Folder unavailable"))
    return DirectoryInventory(node, workbooks, thickness, fovea, unassigned, tuple(errors))
