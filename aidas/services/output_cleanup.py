"""Remove obsolete generated DARK artifacts after successful LIGHT processing."""

from __future__ import annotations

import logging
from pathlib import Path
import re


_LOGGER = logging.getLogger(__name__)
_ROOT_DARK_OUTPUTS = frozenset({
    "_flat_dark.hdr", "_flat_dark.img",
    "_flat-normed_dark.hdr", "_flat-normed_dark.img",
    "dark_marked_find_vertex.png", "dark_marked_vertex.png",
    "_tissueborders__dark.png",
    "_thickness_vs_distance_from_fovea_dark.txt",
    "_dark_profiles_dark.txt", "_fovea_dark_profiles_dark.txt",
})
_LEGACY_WORKSPACE_PAIRS = {
    suffix: (
        (f"_flat_light.{suffix}", f"_done_light.{suffix}"),
        (f"dark__and__light__flat.{suffix}", f"_done_dark__and__light.{suffix}"),
    )
    for suffix in ("rdata", "npz")
}
_PYTHON_DARK_PLOT = re.compile(
    r"(?:python_dark_.+|python_shift_position_dark_[1-9][0-9]*|"
    r"python_revised_(?:vitreous_retina|rnfl_gcl|inl_ipl|onl_opl|olm)_dark_slice_[1-9][0-9]*)\.png",
    re.IGNORECASE,
)


def remove_legacy_dark_outputs(output_dir: Path) -> list[Path]:
    """Remove only recognized generated files in this completed output folder.

    Call after validating the new LIGHT outputs. This does not scan recursively,
    remove original DARK Analyze inputs, or follow linked output subdirectories
    outside the selected folder. Preserve legacy workspaces until both matching
    LIGHT replacements of the same format exist.

    Cleanup failures are logged without failing an otherwise successful export.
    """
    try:
        root = Path(output_dir).resolve()
        files = {path.name.casefold(): path for path in root.iterdir() if path.is_file()}
    except (OSError, RuntimeError) as error:
        _LOGGER.warning("Could not inspect obsolete output files in %s: %s", output_dir, error)
        return []

    targets = [path for name, path in files.items() if name in _ROOT_DARK_OUTPUTS]
    for replacements, legacy_names in _LEGACY_WORKSPACE_PAIRS.values():
        if all(name in files for name in replacements):
            targets.extend(files[name] for name in legacy_names if name in files)

    for directory_name in ("step3_r_arrays", "python_plots"):
        directory = root / directory_name
        try:
            # Resolve each known directory before listing it. A junction to an
            # unrelated participant or another output folder must be untouched.
            resolved = directory.resolve()
            resolved.relative_to(root)
            if not resolved.is_dir():
                continue
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                if directory_name == "step3_r_arrays":
                    stem = path.stem.casefold()
                    generated = path.suffix.casefold() in (".bin", ".shape") and (
                        "_dark_" in stem or stem.endswith("_dark"))
                else:
                    generated = _PYTHON_DARK_PLOT.fullmatch(path.name) is not None
                if generated:
                    targets.append(path)
        except ValueError:
            # A known subdirectory points outside the resolved output root.
            continue
        except (OSError, RuntimeError) as error:
            _LOGGER.warning("Could not inspect obsolete output files in %s: %s", directory, error)

    removed: list[Path] = []
    for path in targets:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as error:
            _LOGGER.warning("Could not remove obsolete generated output %s: %s", path, error)
        else:
            removed.append(path)
    return removed
