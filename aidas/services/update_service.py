"""Secure GitHub Release discovery and Windows installer downloads for AIDaS."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable
import urllib.error
import urllib.request

from packaging.version import InvalidVersion, Version
import truststore


GITHUB_REPOSITORY = "Hejrati/AIDaS"
GITHUB_RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases?per_page=30"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPOSITORY}/releases"
GITHUB_API_VERSION = "2022-11-28"
USER_AGENT = "AIDaS-Update-Client"

MAX_API_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_CHECKSUM_BYTES = 64 * 1024
MAX_INSTALLER_BYTES = 2 * 1024 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class UpdateError(RuntimeError):
    """Raised when release metadata or an installer cannot be trusted or used."""


class DownloadCancelled(UpdateError):
    """Raised when the user cancels an installer download."""


@dataclass(frozen=True)
class ReleaseInfo:
    """Validated metadata for one installable GitHub release."""

    version: Version
    version_text: str
    tag_name: str
    title: str
    notes: str
    page_url: str
    installer_name: str
    installer_url: str
    installer_size: int
    sha256: str
    prerelease: bool


def supports_in_app_install() -> bool:
    """Return whether this process can replace itself with the Windows installer."""
    return os.name == "nt" and bool(getattr(sys, "frozen", False))


def update_cache_dir() -> Path:
    """Return a per-user cache that is separate from the installed application."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / ".aidas"
    return base / "AIDaS" / "updates"


def _github_request(url: str, *, accept: str = "application/vnd.github+json") -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
        method="GET",
    )


def _read_limited(response, limit: int) -> bytes:
    declared_length = response.headers.get("Content-Length")
    if declared_length:
        try:
            if int(declared_length) > limit:
                raise UpdateError("The update server returned an unexpectedly large response.")
        except ValueError:
            pass

    data = response.read(limit + 1)
    if len(data) > limit:
        raise UpdateError("The update server returned an unexpectedly large response.")
    return data


def _open_url(request: urllib.request.Request, *, timeout: int = 30):
    try:
        # The packaged app must not depend on an OpenSSL CA file being present
        # beside the executable. Use the operating system's native trust store
        # (Windows CryptoAPI on installed builds), including organization-
        # managed roots, while keeping certificate and hostname checks enabled.
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return urllib.request.urlopen(request, timeout=timeout, context=context)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError(
                "The AIDaS release repository is not publicly accessible. "
                "The publisher must make the release repository public."
            ) from exc
        if exc.code == 403 and exc.headers.get("X-RateLimit-Remaining") == "0":
            raise UpdateError("GitHub's update-check limit was reached. Please try again later.") from exc
        raise UpdateError(f"The update server returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise UpdateError(
                "The system could not validate GitHub's HTTPS certificate. "
                "Check the Windows date and time and ensure any organization-managed "
                "root certificate is installed in the Windows certificate store."
            ) from exc
        raise UpdateError(f"Could not reach GitHub: {reason}") from exc
    except OSError as exc:
        raise UpdateError(f"Could not reach GitHub: {exc}") from exc


def _fetch_release_payloads() -> list[dict[str, Any]]:
    request = _github_request(GITHUB_RELEASES_API)
    with _open_url(request) as response:
        raw = _read_limited(response, MAX_API_RESPONSE_BYTES)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateError("GitHub returned invalid release metadata.") from exc
    if not isinstance(payload, list):
        raise UpdateError("GitHub returned an unexpected release response.")
    releases = [item for item in payload if isinstance(item, dict)]
    if not releases:
        raise UpdateError(
            "No AIDaS releases are currently published on GitHub. "
            "The publisher must restore or publish a release before in-app updates can work."
        )
    return releases


def _version_from_tag(tag_name: str) -> Version:
    candidate = tag_name.strip()
    if candidate.lower().startswith("v"):
        candidate = candidate[1:]
    try:
        version = Version(candidate)
    except InvalidVersion as exc:
        raise UpdateError(f"Release tag {tag_name!r} is not a valid AIDaS version.") from exc
    if (
        len(version.release) != 3
        or version.dev is not None
        or version.post is not None
        or version.local is not None
    ):
        raise UpdateError(f"Release tag {tag_name!r} is not a supported AIDaS version.")
    return version


def _sha256_from_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    algorithm, separator, digest = value.strip().partition(":")
    if separator and algorithm.lower() == "sha256" and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        return digest.lower()
    return None


def _fetch_checksum(url: str, installer_name: str) -> str:
    request = _github_request(url, accept="application/octet-stream")
    with _open_url(request) as response:
        raw = _read_limited(response, MAX_CHECKSUM_BYTES)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UpdateError("The release checksum file is not valid UTF-8.") from exc

    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            continue
        listed_name = parts[1].lstrip("* ").strip()
        if listed_name == installer_name:
            return parts[0].lower()
    raise UpdateError(f"The release checksum does not describe {installer_name}.")


def _release_from_payload(payload: dict[str, Any]) -> ReleaseInfo | None:
    if payload.get("draft"):
        return None

    tag_name = str(payload.get("tag_name") or "").strip()
    if not tag_name:
        return None
    try:
        version = _version_from_tag(tag_name)
    except UpdateError:
        return None

    version_text = str(version)
    expected_installer_name = f"AIDaS-Setup-{version_text}.exe"
    expected_checksum_name = expected_installer_name + ".sha256"
    installer_asset = None
    checksum_asset = None
    assets = payload.get("assets")
    if not isinstance(assets, list):
        return None
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        if name.lower() == expected_installer_name.lower():
            installer_asset = asset
        elif name.lower() == expected_checksum_name.lower():
            checksum_asset = asset

    if installer_asset is None:
        return None

    installer_url = str(installer_asset.get("browser_download_url") or "")
    if not installer_url.startswith("https://github.com/"):
        raise UpdateError(f"Release {tag_name} has an invalid installer URL.")
    try:
        installer_size = int(installer_asset.get("size") or 0)
    except (TypeError, ValueError) as exc:
        raise UpdateError(f"Release {tag_name} has an invalid installer size.") from exc
    if not 0 < installer_size <= MAX_INSTALLER_BYTES:
        raise UpdateError(f"Release {tag_name} has an unsafe installer size.")

    sha256 = _sha256_from_digest(installer_asset.get("digest"))
    if sha256 is None:
        if checksum_asset is None:
            raise UpdateError(f"Release {tag_name} has no SHA-256 checksum.")
        checksum_url = str(checksum_asset.get("browser_download_url") or "")
        if not checksum_url.startswith("https://github.com/"):
            raise UpdateError(f"Release {tag_name} has an invalid checksum URL.")
        sha256 = _fetch_checksum(checksum_url, expected_installer_name)

    page_url = str(payload.get("html_url") or "")
    if not page_url.startswith("https://github.com/"):
        page_url = GITHUB_RELEASES_URL

    return ReleaseInfo(
        version=version,
        version_text=version_text,
        tag_name=tag_name,
        title=str(payload.get("name") or tag_name),
        notes=str(payload.get("body") or "").strip(),
        page_url=page_url,
        installer_name=expected_installer_name,
        installer_url=installer_url,
        installer_size=installer_size,
        sha256=sha256,
        prerelease=bool(payload.get("prerelease")) or version.is_prerelease,
    )


def select_available_update(
    payloads: list[dict[str, Any]],
    current_version: str,
    *,
    include_prereleases: bool = False,
) -> ReleaseInfo | None:
    """Return the highest installable release newer than ``current_version``."""
    try:
        installed = Version(current_version)
    except InvalidVersion as exc:
        raise UpdateError(f"Installed version {current_version!r} is invalid.") from exc

    candidates: list[ReleaseInfo] = []
    metadata_errors: list[UpdateError] = []
    for payload in payloads:
        if not include_prereleases:
            try:
                tagged_version = _version_from_tag(str(payload.get("tag_name") or ""))
            except UpdateError:
                tagged_version = None
            if payload.get("prerelease") or (tagged_version is not None and tagged_version.is_prerelease):
                continue
        try:
            release = _release_from_payload(payload)
        except UpdateError as exc:
            metadata_errors.append(exc)
            continue
        if release is None or release.version <= installed:
            continue
        if release.prerelease and not include_prereleases:
            continue
        candidates.append(release)

    if candidates:
        return max(candidates, key=lambda item: item.version)
    if metadata_errors:
        # A malformed old release should not break update checks. Only surface
        # metadata errors when no valid release can be evaluated at all.
        parseable_versions = []
        for payload in payloads:
            try:
                version = _version_from_tag(str(payload.get("tag_name") or ""))
            except UpdateError:
                pass
            else:
                if include_prereleases or not (payload.get("prerelease") or version.is_prerelease):
                    parseable_versions.append(version)
        if any(version > installed for version in parseable_versions):
            raise metadata_errors[0]
    return None


def find_available_update(current_version: str) -> ReleaseInfo | None:
    """Return the newest validated release, including preview releases."""
    return select_available_update(
        _fetch_release_payloads(),
        current_version,
        include_prereleases=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_existing_installer(path: Path, release: ReleaseInfo) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != release.installer_size:
            return False
        return hmac.compare_digest(_sha256_file(path), release.sha256)
    except OSError:
        return False


def download_installer(
    release: ReleaseInfo,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Download and atomically publish a checksum-verified installer."""
    cache_dir = update_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / release.installer_name

    if _verified_existing_installer(destination, release):
        if progress is not None:
            progress(release.installer_size, release.installer_size)
        return destination

    request = _github_request(release.installer_url, accept="application/octet-stream")
    temp_path: Path | None = None
    try:
        with _open_url(request, timeout=60) as response:
            declared_length = response.headers.get("Content-Length")
            if declared_length:
                try:
                    response_size = int(declared_length)
                except ValueError:
                    response_size = release.installer_size
                if response_size != release.installer_size:
                    raise UpdateError("The downloaded installer size does not match the GitHub release.")

            file_descriptor, temp_name = tempfile.mkstemp(
                prefix=f"{release.installer_name}.",
                suffix=".part",
                dir=cache_dir,
            )
            temp_path = Path(temp_name)
            digest = hashlib.sha256()
            downloaded = 0
            with os.fdopen(file_descriptor, "wb") as handle:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise DownloadCancelled("The update download was cancelled.")
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > release.installer_size or downloaded > MAX_INSTALLER_BYTES:
                        raise UpdateError("The downloaded installer is larger than expected.")
                    handle.write(chunk)
                    digest.update(chunk)
                    if progress is not None:
                        progress(downloaded, release.installer_size)
                handle.flush()
                os.fsync(handle.fileno())

        if downloaded != release.installer_size:
            raise UpdateError("The installer download ended before all bytes were received.")
        if not hmac.compare_digest(digest.hexdigest(), release.sha256):
            raise UpdateError("The installer failed SHA-256 verification and was not opened.")

        os.replace(temp_path, destination)
        temp_path = None
        return destination
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def launch_installer(installer_path: Path) -> subprocess.Popen:
    """Start a verified Inno Setup package; the caller should then exit AIDaS."""
    if not supports_in_app_install():
        raise UpdateError("In-app installation is available only in the packaged Windows application.")
    installer_path = installer_path.resolve()
    if not installer_path.is_file() or installer_path.suffix.lower() != ".exe":
        raise UpdateError("The downloaded update installer is missing.")

    log_path = update_cache_dir() / "latest-install.log"
    command = [
        str(installer_path),
        "/SP-",
        "/SILENT",
        "/NORESTART",
        "/CLOSEAPPLICATIONS",
        "/UPDATEFROMAPP=1",
        f"/LOG={log_path}",
    ]
    try:
        return subprocess.Popen(command, close_fds=True)
    except OSError as exc:
        raise UpdateError(f"Windows could not start the update installer: {exc}") from exc
