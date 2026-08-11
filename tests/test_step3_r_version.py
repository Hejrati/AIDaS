from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest import mock
from types import SimpleNamespace

from aidas.steps.step3_flatten import RSetupWizard, Step3Frame


class Step3RVersionTests(unittest.TestCase):
    def _frame(self, configured=None):
        frame = Step3Frame.__new__(Step3Frame)
        preferences = mock.Mock()
        preferences.get.return_value = configured
        frame.preferences = preferences
        return frame

    def test_resolver_rejects_other_versions_and_finds_r331(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            r4 = root / "R-4.6.1" / "bin" / "x64" / "Rscript.exe"
            r331 = root / "R-3.3.1" / "bin" / "x64" / "Rscript.exe"
            r4.parent.mkdir(parents=True)
            r331.parent.mkdir(parents=True)
            r4.write_text("", encoding="utf-8")
            r331.write_text("", encoding="utf-8")
            frame = self._frame(configured=r4)

            def version(path):
                return "4.6.1" if Path(path) == r4 else "3.3.1"

            with mock.patch("aidas.steps.step3_flatten.shutil.which", return_value=None), mock.patch.object(
                frame, "_installed_r_executable_candidates", return_value=[r4, r331]
            ), mock.patch.object(frame, "_r_version_for_executable", side_effect=version):
                self.assertEqual(frame._resolve_rscript_executable(), r331)

    def test_reported_version_is_parsed_from_rscript_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "Rscript.exe"
            executable.write_text("", encoding="utf-8")
            frame = self._frame()
            completed = mock.Mock(stdout="R scripting front-end version 3.3.1 (2016-05-03)\n")
            with mock.patch("aidas.steps.step3_flatten.subprocess.run", return_value=completed) as run:
                self.assertEqual(frame._r_version_for_executable(executable), "3.3.1")
            self.assertEqual(run.call_args.args[0], [str(executable), "--version"])

    def test_rgui_is_not_accepted_as_the_r_program(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rgui = Path(temp_dir) / "Rgui.exe"
            rscript = Path(temp_dir) / "Rscript.exe"
            rgui.write_text("", encoding="utf-8")
            rscript.write_text("", encoding="utf-8")
            self.assertIsNone(Step3Frame._normalize_r_executable(rgui))
            self.assertEqual(Step3Frame._normalize_r_executable(rscript), rscript)

    def test_installer_is_pinned_to_r331(self):
        self.assertEqual(Step3Frame.R_REQUIRED_VERSION, "3.3.1")
        self.assertEqual(Step3Frame.R_INSTALLER_NAME, "R-3.3.1-win.exe")
        self.assertEqual(
            Step3Frame.R_DOWNLOAD_PAGE,
            "https://cran-archive.r-project.org/bin/windows/base/old/3.3.1/",
        )

    def test_setup_wizard_is_one_unified_page(self):
        self.assertEqual(RSetupWizard.STEPS, ("Setup",))

    def test_setup_status_uses_installed_instead_of_repeated_ok_labels(self):
        self.assertEqual(
            RSetupWizard._status_text("installed"),
            ("Installed", "WizardSuccess.TLabel"),
        )

    def test_open_setup_wizard_rebuilds_styles_on_live_theme_change(self):
        wizard = RSetupWizard.__new__(RSetupWizard)
        wizard._build_styles = mock.Mock()

        wizard._apply_aidas_theme()

        wizard._build_styles.assert_called_once_with()

    def test_installer_is_saved_in_per_user_app_data(self):
        wizard = RSetupWizard.__new__(RSetupWizard)
        wizard.step_frame = SimpleNamespace(R_INSTALLER_NAME="R-3.3.1-win.exe")
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            "os.environ", {"LOCALAPPDATA": temp_dir}, clear=False
        ):
            self.assertEqual(
                wizard._r_installer_cache_path(),
                Path(temp_dir) / "AIDaS" / "R" / "R-3.3.1-win.exe",
            )

    def test_r_packages_are_installed_from_bundled_files_without_cran(self):
        wizard = RSetupWizard.__new__(RSetupWizard)
        wizard.step_frame = SimpleNamespace(R_LOCAL_PACKAGE_FILES=Step3Frame.R_LOCAL_PACKAGE_FILES)
        wizard.package_library_path = Path(tempfile.gettempdir()) / "aidas-test-r-packages"

        expression = wizard._package_install_expression("AnalyzeFMRI")

        self.assertIn("AnalyzeFMRI.zip", expression)
        self.assertIn("repos=NULL", expression)
        self.assertIn("dependencies=FALSE", expression)
        self.assertNotIn("cloud.r-project.org", expression)

    def test_local_package_install_order_places_dependencies_first(self):
        order = Step3Frame.R_LOCAL_PACKAGE_ORDER
        self.assertLess(order.index("R.methodsS3"), order.index("R.matlab"))
        self.assertLess(order.index("R.matlab"), order.index("AnalyzeFMRI"))
        self.assertLess(order.index("Rcpp"), order.index("RNifti"))
        self.assertLess(order.index("RNifti"), order.index("RNiftyReg"))

    def test_default_package_library_migrates_legacy_appdata_path(self):
        frame = self._frame(configured=None)
        frame.r_package_library_path = None
        wizard = RSetupWizard.__new__(RSetupWizard)
        wizard.step_frame = frame
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            "os.environ", {"LOCALAPPDATA": temp_dir}, clear=False
        ):
            frame.r_package_library_path = str(Path(temp_dir) / "AIDaS" / "R-packages")
            with mock.patch("pathlib.Path.home", return_value=Path(temp_dir)):
                self.assertEqual(
                    wizard._default_package_library(),
                    Path(temp_dir) / "AIDaS_R_packages",
                )

    def test_all_local_package_archives_are_bundled_and_valid(self):
        asset_dir = Path(__file__).parents[1] / "assets" / "r_packages"
        for package_name in Step3Frame.R_LOCAL_PACKAGE_ORDER:
            archive_path = asset_dir / Step3Frame.R_LOCAL_PACKAGE_FILES[package_name]
            self.assertTrue(archive_path.is_file(), archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertTrue(
                    any(Path(name).name == "DESCRIPTION" for name in archive.namelist()),
                    archive_path,
                )


if __name__ == "__main__":
    unittest.main()
