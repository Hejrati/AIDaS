"""Require a successful Windows validation of the exact release commit on main."""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from release_tools.release_metadata import metadata, read_project_version


def github_repository(origin: str) -> str:
    """Accept conventional GitHub HTTPS and SSH origins, never another API host."""
    if origin.startswith("git@github.com:"):
        path = origin.removeprefix("git@github.com:")
    else:
        try:
            parsed = urlsplit(origin)
            allowed = (
                parsed.scheme in {"https", "ssh"}
                and parsed.hostname == "github.com"
                and parsed.password is None
                and parsed.port is None
                and not parsed.query
                and not parsed.fragment
                and (
                    (parsed.scheme == "https" and parsed.username is None)
                    or (parsed.scheme == "ssh" and parsed.username == "git")
                )
            )
        except ValueError:
            allowed = False
        if not allowed:
            raise ValueError("Origin must be a github.com HTTPS or git SSH repository URL.")
        path = parsed.path.removeprefix("/")
    path = path.removesuffix(".git")
    parts = path.split("/")
    if (
        len(parts) != 2
        or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", parts[0])
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", parts[1])
        or parts[1] in {".", ".."}
    ):
        raise ValueError("Origin must identify one GitHub owner and repository.")
    return "/".join(parts)


def fetch_validation_runs(repository: str, commit: str) -> dict:
    query = urlencode({"head_sha": commit, "branch": "main", "event": "push", "per_page": 100})
    endpoint = f"https://api.github.com/repos/{repository}/actions/workflows/release.yml/runs?{query}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AIDaS-release-preflight",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GH_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(endpoint, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise ValueError(
            f"GitHub Actions request failed (HTTP {exc.code}). "
            "Check repository access and GH_TOKEN/GITHUB_TOKEN Actions read permission."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ValueError("GitHub Actions could not be reached. Check the network and try again.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("GitHub did not return a valid workflow-run response.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
        raise ValueError("GitHub did not return a valid workflow-run response.")
    return payload


def run_command(arguments: list[str], *, cwd: Path) -> str:
    """Run without a shell so branch names and API parameters remain literal."""
    try:
        result = subprocess.run(
            arguments, cwd=cwd, text=True, capture_output=True, check=True,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"Required command is not installed: {arguments[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "command failed").strip()
        raise ValueError(f"{arguments[0]}: {detail}") from exc
    return result.stdout.strip()


def require_successful_validation(payload: dict, commit: str) -> dict:
    """Accept only the latest main-push validation for this exact source SHA."""
    runs = [
        run for run in payload.get("workflow_runs", [])
        if run.get("head_sha") == commit
        and run.get("head_branch") == "main"
        and run.get("event") == "push"
    ]
    if not runs:
        raise ValueError(
            f"Commit {commit} has no Windows validation run on main. "
            "Push the commit to main and wait for Windows validation to finish before tagging."
        )
    latest = max(runs, key=lambda run: (run.get("run_number", 0), run.get("run_attempt", 0)))
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        state = latest.get("conclusion") or latest.get("status") or "unknown"
        raise ValueError(
            f"Windows validation for {commit} is {state}, not successful. "
            f"Resolve it before tagging: {latest.get('html_url', '')}"
        )
    return latest


def check_release(tag: str, *, project_root: Path = PROJECT_ROOT) -> tuple[str, dict]:
    metadata(read_project_version(project_root / "aidas" / "__init__.py"), tag)
    if run_command(["git", "status", "--porcelain"], cwd=project_root):
        raise ValueError("The working tree has uncommitted changes. Commit the release source first.")
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=project_root)
    origin = run_command(["git", "remote", "get-url", "origin"], cwd=project_root)
    payload = fetch_validation_runs(github_repository(origin), commit)
    return commit, require_successful_validation(payload, commit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Proposed or existing release tag")
    args = parser.parse_args()
    try:
        commit, run = check_release(args.tag)
    except ValueError as exc:
        print(f"Release preflight failed: {exc}", file=sys.stderr)
        return 1
    print(f"Release preflight passed: {args.tag} at {commit}")
    print(f"Windows tests, executable, and installer validated: {run['html_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
