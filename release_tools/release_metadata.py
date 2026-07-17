"""Validate an AIDaS release tag and emit Windows packaging metadata."""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from pathlib import Path
import re

from packaging.version import InvalidVersion, Version


VERSION_PATTERN = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$', re.MULTILINE)


def read_project_version(version_file: Path) -> Version:
    text = version_file.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not find __version__ in {version_file}")
    try:
        version = Version(match.group(1))
    except InvalidVersion as exc:
        raise ValueError(f"Invalid AIDaS version: {match.group(1)!r}") from exc
    if version.dev is not None or version.post is not None or version.local is not None:
        raise ValueError("Release versions cannot contain dev, post, or local segments")
    if len(version.release) != 3:
        raise ValueError("AIDaS releases must use major.minor.patch versions")
    return version


def windows_file_version(version: Version) -> str:
    """Map PEP 440 prereleases to a monotonic four-part Windows version."""
    major, minor, patch = (*version.release, 0, 0, 0)[:3]
    if any(part < 0 or part > 65535 for part in (major, minor, patch)):
        raise ValueError("Windows version components must be between 0 and 65535")

    if version.pre is None:
        stage = 65535
    else:
        label, serial = version.pre
        bases = {"a": 0, "b": 20000, "rc": 40000}
        if label not in bases or serial > 19999:
            raise ValueError(f"Unsupported prerelease version: {version}")
        stage = bases[label] + serial
    return f"{major}.{minor}.{patch}.{stage}"


def metadata(version: Version, tag: str) -> dict[str, str]:
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ValueError(f"Git tag {tag!r} must exactly match {expected_tag!r}")
    return {
        "version": str(version),
        "tag": tag,
        "file_version": windows_file_version(version),
        "installer_name": f"AIDaS-Setup-{version}.exe",
        "prerelease": str(version.is_prerelease).lower(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--version-file", type=Path, default=Path("aidas/__init__.py"))
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    values = metadata(read_project_version(args.version_file), args.tag)
    lines = "".join(f"{key}={value}\n" for key, value in values.items())
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(lines)
    else:
        print(lines, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
