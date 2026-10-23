"""Check an R package library in a single, isolated R process."""

from __future__ import annotations

from concurrent.futures import CancelledError
import os
from pathlib import Path
import subprocess

from aidas.utils.cancellable_process import run_cancellable_process


_RESULT_TAG = "__AIDAS_R_PACKAGE_CHECK__"
_STATUSES = {"installed", "missing"}


def _r_string(value: str) -> str:
    """Quote one R string without interpreting path or package characters."""
    escaped = (
        str(value).replace("\\", "\\\\").replace("'", "\\'")
        .replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    )
    return "'" + escaped + "'"


def check_r_packages(
    r_executable,
    library_path,
    package_names,
    *,
    command_builder,
    log_result=None,
    cancel_event=None,
) -> dict[str, str]:
    """Return installed/missing statuses, rejecting incomplete process output.

    ``command_builder`` supplies the existing Rscript/Rterm command wrapper.
    Tagged records keep R startup messages separate from package results; a
    failed R process cannot turn a partially checked library into valid data.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError()
    names = list(dict.fromkeys(package_names))
    if not names:
        return {}
    try:
        library = Path(library_path).expanduser().resolve()
        library.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Cannot prepare the R package library: {exc}") from exc

    packages = ", ".join(_r_string(name) for name in names)
    expression = (
        f".libPaths(c({_r_string(library.as_posix())}, .libPaths())); "
        f"for (.aidas_pkg in c({packages})) {{ "
        ".aidas_ok <- tryCatch(requireNamespace(.aidas_pkg, quietly=TRUE), "
        "error=function(e) FALSE); "
        f"cat({_r_string(_RESULT_TAG)}, .aidas_pkg, "
        "if (isTRUE(.aidas_ok)) 'installed' else 'missing', sep='\\t'); "
        "cat('\\n'); }"
    )
    command = command_builder(r_executable, expression)
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(library)
    try:
        runner = subprocess.run if cancel_event is None else run_cancellable_process
        cancellation_options = {} if cancel_event is None else {"cancel_event": cancel_event}
        result = runner(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=120,
            **cancellation_options,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"R package check could not complete: {exc}") from exc

    if log_result is not None:
        log_result("Check R packages", command, result)
    if result.returncode != 0:
        raise RuntimeError(f"R package check failed with exit code {result.returncode}.")

    expected = set(names)
    statuses = {}
    for line in (result.stdout or "").splitlines():
        if not line.startswith(_RESULT_TAG):
            continue
        fields = line.split("\t")
        if (
            len(fields) != 3
            or fields[0] != _RESULT_TAG
            or fields[1] not in expected
            or fields[1] in statuses
            or fields[2] not in _STATUSES
        ):
            raise RuntimeError("R package check returned a malformed or duplicate result.")
        statuses[fields[1]] = fields[2]
    if set(statuses) != expected:
        missing = ", ".join(name for name in names if name not in statuses)
        raise RuntimeError(f"R package check returned incomplete results: {missing}.")
    return {name: statuses[name] for name in names}
