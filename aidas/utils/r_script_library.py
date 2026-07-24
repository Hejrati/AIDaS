"""Discovery and importing for user-selectable Step 3 R scripts."""

from __future__ import annotations

from dataclasses import dataclass
import filecmp
import os
from pathlib import Path
import shutil


@dataclass(frozen=True)
class RScriptChoice:
    """One R script shown in the Step 3 script selector."""

    label: str
    path: Path
    source: str
    is_default: bool = False


def user_r_script_dir(role: str | None = None) -> Path:
    """Return the persistent per-user directory for imported R scripts."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    base_dir = Path(local_app_data) / "AIDaS" if local_app_data else Path.home() / ".aidas"
    script_dir = base_dir / "R-scripts"
    return script_dir / role if role else script_dir


def _r_files(folder: Path) -> list[Path]:
    try:
        return sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".r"),
            key=lambda path: path.name.lower(),
        )
    except OSError:
        return []


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def discover_r_scripts(
    default_script: Path,
    library_dir: Path | None = None,
    *,
    bundled_prefixes: tuple[str, ...] | None = None,
) -> list[RScriptChoice]:
    """List the default, other bundled, and imported user R scripts."""
    default_script = Path(default_script)
    library_dir = Path(library_dir) if library_dir is not None else user_r_script_dir()
    choices: list[RScriptChoice] = []
    seen: set[str] = set()

    def add(path: Path, source: str, *, is_default: bool = False) -> None:
        if not path.is_file():
            return
        key = _path_key(path)
        if key in seen:
            return
        seen.add(key)
        if is_default:
            label = f"Embedded (default) — {path.name}"
        elif source == "bundled":
            label = f"Bundled — {path.name}"
        else:
            label = f"User — {path.name}"
        choices.append(RScriptChoice(label=label, path=path.resolve(), source=source, is_default=is_default))

    add(default_script, "bundled", is_default=True)
    prefixes = tuple(prefix.lower() for prefix in bundled_prefixes or ())
    bundled_paths = _r_files(default_script.parent)
    if prefixes:
        bundled_paths = [path for path in bundled_paths if path.name.lower().startswith(prefixes)]
    for path in bundled_paths:
        add(path, "bundled")
    for path in _r_files(library_dir):
        add(path, "user")
    return choices


def import_r_script(source_path: Path, library_dir: Path | None = None) -> Path:
    """Copy an R script into the user library without overwriting a version."""
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"R script does not exist: {source_path}")
    if source_path.suffix.lower() != ".r":
        raise ValueError("Select a file with the .R extension.")

    library_dir = Path(library_dir) if library_dir is not None else user_r_script_dir()
    library_dir.mkdir(parents=True, exist_ok=True)

    try:
        if source_path.resolve().parent == library_dir.resolve():
            return source_path.resolve()
    except OSError:
        pass

    destination = library_dir / source_path.name
    if destination.exists() and filecmp.cmp(source_path, destination, shallow=False):
        return destination.resolve()

    number = 2
    while destination.exists():
        destination = library_dir / f"{source_path.stem} ({number}){source_path.suffix}"
        number += 1

    shutil.copy2(source_path, destination)
    return destination.resolve()
