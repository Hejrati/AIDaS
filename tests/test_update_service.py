from __future__ import annotations

import hashlib
import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from packaging.version import Version

from aidas.services.update_service import (
    ReleaseInfo,
    UpdateError,
    _fetch_release_payloads,
    _verified_existing_installer,
    download_installer,
    find_available_update,
    select_available_update,
    update_cache_dir,
)


class ReleaseDiscoveryTests(unittest.TestCase):
    def test_empty_github_release_list_is_reported_as_configuration_error(self):
        class Response:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def read(_limit):
                return json.dumps([]).encode("utf-8")

        with mock.patch("aidas.services.update_service._open_url", return_value=Response()):
            with self.assertRaisesRegex(UpdateError, "No AIDaS releases"):
                _fetch_release_payloads()


def release_payload(version: str, *, prerelease: bool = False, digest: str | None = None):
    digest = digest or ("a" * 64)
    installer_name = f"AIDaS-Setup-{version}.exe"
    return {
        "draft": False,
        "prerelease": prerelease,
        "tag_name": f"v{version}",
        "name": f"AIDaS {version}",
        "body": "Release notes",
        "html_url": f"https://github.com/Hejrati/AIDaS/releases/tag/v{version}",
        "assets": [
            {
                "name": installer_name,
                "browser_download_url": (
                    f"https://github.com/Hejrati/AIDaS/releases/download/v{version}/{installer_name}"
                ),
                "size": 1234,
                "digest": f"sha256:{digest}",
            }
        ],
    }


class ReleaseSelectionTests(unittest.TestCase):
    def test_stable_channel_ignores_prereleases(self):
        payloads = [
            release_payload("1.6.0a2", prerelease=True),
            release_payload("1.5.6"),
        ]
        release = select_available_update(payloads, "1.5.5a1")
        self.assertIsNotNone(release)
        self.assertEqual(release.version, Version("1.5.6"))

    def test_preview_channel_selects_highest_version(self):
        payloads = [
            release_payload("1.5.6"),
            release_payload("1.6.0b1", prerelease=True),
            release_payload("1.6.0a2", prerelease=True),
        ]
        release = select_available_update(payloads, "1.5.5", include_prereleases=True)
        self.assertIsNotNone(release)
        self.assertEqual(release.version, Version("1.6.0b1"))

    def test_app_update_discovery_always_includes_preview_releases(self):
        payloads = [
            release_payload("1.5.6"),
            release_payload("1.6.0a1", prerelease=True),
        ]
        with mock.patch("aidas.services.update_service._fetch_release_payloads", return_value=payloads):
            release = find_available_update("1.5.5")
        self.assertIsNotNone(release)
        self.assertEqual(release.version, Version("1.6.0a1"))

    def test_never_downgrades(self):
        release = select_available_update([release_payload("1.5.4")], "1.5.5")
        self.assertIsNone(release)

    def test_same_version_rebuild_is_not_reoffered_forever(self):
        release = select_available_update([release_payload("2.0.0")], "2.0.0")
        self.assertIsNone(release)

    def test_installer_asset_must_match_release_version(self):
        payload = release_payload("1.5.6")
        payload["assets"][0]["name"] = "AIDaS-Setup-1.5.5.exe"
        release = select_available_update([payload], "1.5.5")
        self.assertIsNone(release)

    def test_newer_release_without_checksum_is_rejected(self):
        payload = release_payload("1.5.6")
        payload["assets"][0]["digest"] = None
        with self.assertRaisesRegex(UpdateError, "SHA-256"):
            select_available_update([payload], "1.5.5")

    def test_bad_preview_metadata_does_not_break_stable_channel(self):
        payload = release_payload("1.6.0a1", prerelease=True)
        payload["assets"][0]["digest"] = None
        self.assertIsNone(select_available_update([payload], "1.5.5"))


class InstallerCacheTests(unittest.TestCase):
    def test_verified_installer_requires_matching_size_and_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "AIDaS-Setup-1.5.6.exe"
            content = b"verified installer"
            path.write_bytes(content)
            release = ReleaseInfo(
                version=Version("1.5.6"),
                version_text="1.5.6",
                tag_name="v1.5.6",
                title="AIDaS 1.5.6",
                notes="",
                page_url="https://github.com/Hejrati/AIDaS/releases/tag/v1.5.6",
                installer_name=path.name,
                installer_url="https://github.com/Hejrati/AIDaS/releases/download/v1.5.6/installer.exe",
                installer_size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                prerelease=False,
            )
            self.assertTrue(_verified_existing_installer(path, release))
            path.write_bytes(content + b"changed")
            self.assertFalse(_verified_existing_installer(path, release))

    def test_update_cache_is_separate_from_install_directory(self):
        with tempfile.TemporaryDirectory() as local_app_data:
            with mock.patch.dict(os.environ, {"LOCALAPPDATA": local_app_data}):
                expected = Path(local_app_data) / "AIDaS" / "updates"
                self.assertEqual(update_cache_dir(), expected)

    def test_download_is_published_only_after_checksum_verification(self):
        content = b"complete installer bytes"
        release = ReleaseInfo(
            version=Version("1.5.6"),
            version_text="1.5.6",
            tag_name="v1.5.6",
            title="AIDaS 1.5.6",
            notes="",
            page_url="https://github.com/Hejrati/AIDaS/releases/tag/v1.5.6",
            installer_name="AIDaS-Setup-1.5.6.exe",
            installer_url="https://github.com/Hejrati/AIDaS/releases/download/v1.5.6/AIDaS-Setup-1.5.6.exe",
            installer_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            prerelease=False,
        )

        class Response(io.BytesIO):
            headers = {"Content-Length": str(len(content))}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "aidas.services.update_service.update_cache_dir", return_value=Path(temp_dir)
        ), mock.patch("aidas.services.update_service._open_url", return_value=Response(content)):
            destination = download_installer(release)
            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(list(Path(temp_dir).glob("*.part")), [])

    def test_bad_download_is_deleted_instead_of_replacing_an_installer(self):
        expected = b"expected"
        received = b"tampered"
        release = ReleaseInfo(
            version=Version("1.5.6"),
            version_text="1.5.6",
            tag_name="v1.5.6",
            title="AIDaS 1.5.6",
            notes="",
            page_url="https://github.com/Hejrati/AIDaS/releases/tag/v1.5.6",
            installer_name="AIDaS-Setup-1.5.6.exe",
            installer_url="https://github.com/Hejrati/AIDaS/releases/download/v1.5.6/AIDaS-Setup-1.5.6.exe",
            installer_size=len(received),
            sha256=hashlib.sha256(expected).hexdigest(),
            prerelease=False,
        )

        class Response(io.BytesIO):
            headers = {"Content-Length": str(len(received))}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "aidas.services.update_service.update_cache_dir", return_value=Path(temp_dir)
        ), mock.patch("aidas.services.update_service._open_url", return_value=Response(received)):
            with self.assertRaisesRegex(UpdateError, "SHA-256"):
                download_installer(release)
            self.assertFalse((Path(temp_dir) / release.installer_name).exists())
            self.assertEqual(list(Path(temp_dir).glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
