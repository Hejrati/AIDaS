from __future__ import annotations


from pathlib import Path
import tempfile
import unittest

from packaging.version import Version

from aidas.core.single_instance import SingleInstanceGuard
from release_tools.release_metadata import metadata, read_project_version, windows_file_version

from tests.support.paths import PROJECT_ROOT


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
        cls.script = (PROJECT_ROOT / "installer" / "AIDaS.iss").read_text(encoding="utf-8")

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

    def test_installer_copies_the_complete_onedir_application(self):
        self.assertIn(r'Source: "..\dist\AIDaS\*"', self.script)
        self.assertIn("recursesubdirs", self.script)
        self.assertIn("createallsubdirs", self.script)


class ReleaseWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    def test_release_publish_step_is_safe_to_rerun(self):
        self.assertIn("gh release view $tag", self.workflow)
        self.assertIn("gh release upload $tag $installer", self.workflow)
        self.assertIn("--clobber", self.workflow)
        self.assertIn("--draft=false", self.workflow)

    def test_release_can_be_started_manually_for_an_existing_tag(self):
        self.assertIn("workflow_dispatch:", self.workflow)
        self.assertIn("ref: ${{ inputs.tag || github.ref }}", self.workflow)
        self.assertIn("RELEASE_TAG: ${{ inputs.tag ||", self.workflow)
        self.assertIn('$tag = $env:RELEASE_TAG', self.workflow)

    def test_main_and_pull_requests_use_the_complete_windows_release_validation(self):
        self.assertIn("branches:\n      - main", self.workflow)
        self.assertIn("  pull_request:", self.workflow)
        self.assertEqual(self.workflow.count("python -m unittest discover -s tests -t . -v"), 1)
        self.assertIn("python -m PyInstaller AIDaS.spec --clean --noconfirm", self.workflow)
        self.assertIn("python release_tools/smoke_test_binary.py dist/AIDaS/AIDaS.exe", self.workflow)
        self.assertIn("python release_tools/smoke_test_binary.py $installedExecutable", self.workflow)

    def test_tag_cannot_build_until_the_exact_commit_has_passed_validation(self):
        gate = self.workflow.index('python release_tools/preflight.py --tag "$env:RELEASE_TAG"')
        build = self.workflow.index("python -m PyInstaller")
        self.assertLess(gate, build)
        self.assertIn("if: env.RELEASE_TAG != ''", self.workflow)

    def test_publication_requires_the_successful_validation_artifact(self):
        validation, publication = self.workflow.split("  publish:\n", 1)
        self.assertNotIn("contents: write", validation)
        self.assertIn("permissions:\n  contents: read\n  actions: read", validation)
        self.assertIn("needs: windows-release", publication)
        self.assertIn("needs.windows-release.result == 'success'", publication)
        self.assertIn("startsWith(github.ref, 'refs/tags/v')", publication)
        self.assertIn("contents: write", publication)
        self.assertIn("actions/upload-artifact@v4", validation)
        self.assertIn("actions/download-artifact@v4", publication)
        self.assertIn("$taggedSha -ne $env:VALIDATED_SHA", publication)

    def test_signing_credentials_are_only_available_to_release_steps(self):
        signing = self.workflow.split("      - name: Optionally sign the application executable\n", 1)[1]
        self.assertTrue(signing.startswith("        if: env.RELEASE_TAG != ''\n"))
        self.assertIn("if: steps.signing.outputs.enabled == 'true'", self.workflow)


if __name__ == "__main__":
    unittest.main()
