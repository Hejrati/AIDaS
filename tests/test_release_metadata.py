from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from packaging.version import Version

from aidas.core.single_instance import SingleInstanceGuard
from tools.release_metadata import metadata, read_project_version, windows_file_version


class ReleaseMetadataTests(unittest.TestCase):
    def test_reads_version_without_importing_the_application(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "version.py"
            version_file.write_text('__version__ = "1.5.5a1"\n', encoding="utf-8")
            self.assertEqual(read_project_version(version_file), Version("1.5.5a1"))

    def test_tag_must_exactly_match_project_version(self):
        with self.assertRaisesRegex(ValueError, "must exactly match"):
            metadata(Version("1.5.5"), "v1.5.4")

    def test_windows_versions_keep_prerelease_stages_before_stable(self):
        alpha = tuple(map(int, windows_file_version(Version("1.5.5a2")).split(".")))
        beta = tuple(map(int, windows_file_version(Version("1.5.5b1")).split(".")))
        candidate = tuple(map(int, windows_file_version(Version("1.5.5rc1")).split(".")))
        stable = tuple(map(int, windows_file_version(Version("1.5.5")).split(".")))
        self.assertLess(alpha, beta)
        self.assertLess(beta, candidate)
        self.assertLess(candidate, stable)


class InstallerSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path("installer/AIDaS.iss").read_text(encoding="utf-8")

    def test_installer_is_per_user_and_upgrades_with_stable_identity(self):
        self.assertIn("AppId={{5E514B02-7E97-4F86-8902-DC6EA73A7CB2}", self.script)
        self.assertIn(r"DefaultDirName={localappdata}\Programs\AIDaS", self.script)
        self.assertIn("PrivilegesRequired=lowest", self.script)
        self.assertIn("UsePreviousAppDir=yes", self.script)

    def test_installer_has_no_recursive_deletion_sections(self):
        self.assertNotIn("[InstallDelete]", self.script)
        self.assertNotIn("[UninstallDelete]", self.script)

    def test_installer_checks_the_same_mutex_created_by_aidas(self):
        self.assertIn(f"AppMutex={SingleInstanceGuard.WINDOWS_MUTEX_NAME}", self.script)


if __name__ == "__main__":
    unittest.main()
