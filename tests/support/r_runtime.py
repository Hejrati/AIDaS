"""Locate the optional R 3.3.1 runtime for integration checks."""

import os
from pathlib import Path
import re
import shutil
import subprocess

REQUIRED_R_VERSION = "3.3.1"


def r331_environment() -> dict[str, str]:
    """Use AIDaS's managed library unless the caller supplies an R library."""
    env = os.environ.copy()
    # CI installs its own package library and supplies this override explicitly.
    if "R_LIBS_USER" in env:
        return env

    from aidas.core.config import Config

    configured = Config.peek("r_package_library_path")
    library = Path(configured) if configured else None
    local_app_data = env.get("LOCALAPPDATA")
    legacy = Path(local_app_data) / "AIDaS" / "R-packages" if local_app_data else None
    if library is not None and legacy is not None:
        try:
            if library.resolve() == legacy.resolve():
                library = None
        except OSError:
            pass
    if library is None:
        documents = Path.home() / "Documents"
        library = (
            documents / "AIDaS" / "R-packages"
            if documents.is_dir()
            else Path.home() / "AIDaS_R_packages"
        )
    env["R_LIBS_USER"] = str(library.resolve())
    return env


def _r_version(executable: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\b(?:R|R scripting front-end)\s+version\s+(\d+\.\d+\.\d+)", result.stdout or "")
    return match.group(1) if match else None


def find_r331() -> Path | None:
    candidates = []
    for environment_name in ("AIDAS_R331_PATH", "RSCRIPT_PATH", "R_SCRIPT_PATH"):
        configured = os.environ.get(environment_name)
        if configured:
            candidates.append(Path(configured))

    if os.name == "nt":
        candidates.extend(
            (
                Path(r"C:\Program Files\R\R-3.3.1\bin\x64\Rscript.exe"),
                Path(r"C:\Program Files\R\R-3.3.1\bin\Rscript.exe"),
            )
        )
    discovered = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if discovered:
        candidates.append(Path(discovered))

    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(candidate))
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        if _r_version(candidate) == REQUIRED_R_VERSION:
            return candidate
    return None
