from pathlib import Path
from concurrent.futures import CancelledError
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.steps.step3_flatten import RSetupWizard, Step3Frame


class RPackageSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.wizard = RSetupWizard.__new__(RSetupWizard)
        self.wizard._cancel_event = threading.Event()
        self.wizard.rscript_path = Path("Rscript.exe")
        self.wizard.package_library_path = Path(temporary.name)
        self.wizard.setup_package_names = Step3Frame.R_LOCAL_PACKAGE_ORDER
        self.wizard.step_frame = SimpleNamespace(
            R_LOCAL_PACKAGE_ORDER=Step3Frame.R_LOCAL_PACKAGE_ORDER,
            R_REQUIRED_PACKAGES=Step3Frame.R_REQUIRED_PACKAGES,
            _build_r_eval_command=Step3Frame._build_r_eval_command,
        )
        self.statuses = {name: "installed" for name in Step3Frame.R_LOCAL_PACKAGE_ORDER}
        self.wizard._check_package_status_worker = mock.Mock(return_value=self.statuses.copy())
        self.wizard._log = mock.Mock()
        self.wizard._log_process_result = mock.Mock()
        self.wizard._package_install_expression = mock.Mock(return_value="install_missing_package()")

    def test_install_does_not_recheck_packages_confirmed_by_the_batch(self):
        with mock.patch("aidas.steps.step3_flatten.subprocess.run") as run, \
                mock.patch("aidas.steps.step3_flatten.run_cancellable_process") as check:
            statuses = self.wizard._install_missing_packages_worker()
        self.assertEqual(statuses, self.statuses)
        self.wizard._check_package_status_worker.assert_called_once_with()
        self.wizard._package_install_expression.assert_not_called()
        run.assert_not_called()
        check.assert_not_called()

    def test_only_the_missing_package_is_checked_installed_and_verified(self):
        self.wizard._check_package_status_worker.return_value["RNiftyReg"] = "missing"
        results = [subprocess.CompletedProcess([], code, "", "") for code in (1, 0)]
        with mock.patch("aidas.steps.step3_flatten.run_cancellable_process", side_effect=results) as check, \
                mock.patch("aidas.steps.step3_flatten.subprocess.run", return_value=results[-1]) as run:
            statuses = self.wizard._install_missing_packages_worker()
        self.assertEqual(statuses, self.statuses)
        self.assertEqual(check.call_count, 2)
        run.assert_called_once()
        self.wizard._package_install_expression.assert_called_once_with("RNiftyReg")
        for call in check.call_args_list:
            self.assertIn("requireNamespace('RNiftyReg'", call.args[0][-1])
            self.assertIs(call.kwargs["cancel_event"], self.wizard._cancel_event)
        self.assertEqual(run.call_args.args[0][-1], "install_missing_package()")

    def test_package_repaired_by_its_dependencies_is_not_reinstalled(self):
        self.wizard._check_package_status_worker.return_value["RNiftyReg"] = "missing"
        with mock.patch(
            "aidas.steps.step3_flatten.run_cancellable_process",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ) as run:
            statuses = self.wizard._install_missing_packages_worker()
        self.assertEqual(statuses, self.statuses)
        run.assert_called_once()
        self.wizard._package_install_expression.assert_not_called()

    def test_failed_install_cannot_be_reported_as_installed(self):
        self.wizard._check_package_status_worker.return_value["RNiftyReg"] = "missing"
        results = [subprocess.CompletedProcess([], code, "", "") for code in (1, 0)]
        with mock.patch("aidas.steps.step3_flatten.run_cancellable_process", side_effect=results), \
                mock.patch("aidas.steps.step3_flatten.subprocess.run", return_value=results[0]):
            statuses = self.wizard._install_missing_packages_worker()
        self.assertEqual(statuses["RNiftyReg"], "failed")
        self.assertTrue(all(statuses[name] == "installed" for name in statuses if name != "RNiftyReg"))

    def test_cancelling_an_install_stops_after_the_current_package_write(self):
        statuses = self.wizard._check_package_status_worker.return_value
        statuses.update(Rcpp="missing", RNiftyReg="missing")

        def finish_current_package(*args, **kwargs):
            self.wizard._cancel_event.set()
            return subprocess.CompletedProcess([], 0, "", "")

        with mock.patch(
            "aidas.steps.step3_flatten.run_cancellable_process",
            return_value=subprocess.CompletedProcess([], 1, "", ""),
        ) as check, mock.patch(
            "aidas.steps.step3_flatten.subprocess.run", side_effect=finish_current_package,
        ) as install:
            with self.assertRaises(CancelledError):
                self.wizard._install_missing_packages_worker()
        check.assert_called_once()
        install.assert_called_once()
        self.wizard._package_install_expression.assert_called_once_with("Rcpp")

    def test_cancelled_setup_does_not_start_a_package_check(self):
        self.wizard._cancel_event.set()
        with self.assertRaises(CancelledError):
            self.wizard._install_missing_packages_worker()
        self.wizard._check_package_status_worker.assert_not_called()

    def _download_worker(self):
        self.wizard.busy = False
        self.wizard.installer_url = "https://example.invalid/R-installer.exe"
        self.wizard.installer_path = self.wizard.package_library_path / "installer.exe"
        self.wizard._run_worker = mock.Mock()
        self.wizard._download_and_install_r()
        return self.wizard._run_worker.call_args.args[1]

    def test_cancelled_download_removes_partial_files_without_publishing_installer(self):
        worker = self._download_worker()

        def interrupted_read(_size):
            self.wizard._cancel_event.set()
            return b"partial download"

        response = mock.MagicMock()
        response.__enter__.return_value.read1.side_effect = interrupted_read
        with mock.patch("aidas.steps.step3_flatten.urllib.request.urlopen", return_value=response):
            with self.assertRaises(CancelledError):
                worker()
        self.assertEqual(list(self.wizard.package_library_path.iterdir()), [])

    def test_completed_download_is_published_only_after_all_bytes_are_received(self):
        worker = self._download_worker()
        response = mock.MagicMock()
        response.__enter__.return_value = SimpleNamespace(
            read=mock.Mock(side_effect=[b"installer content", b""]),
        )
        with mock.patch("aidas.steps.step3_flatten.urllib.request.urlopen", return_value=response):
            self.assertEqual(worker(), self.wizard.installer_path)
        self.assertEqual(self.wizard.installer_path.read_bytes(), b"installer content")
        self.assertEqual(list(self.wizard.package_library_path.iterdir()), [self.wizard.installer_path])

    def test_processing_requires_a_successful_complete_package_check(self):
        frame = Step3Frame.__new__(Step3Frame)
        frame._default_r_package_library = lambda: self.wizard.package_library_path
        cases = (
            ({"AnalyzeFMRI": "installed", "RNiftyReg": "installed"}, True),
            ({"AnalyzeFMRI": "installed", "RNiftyReg": "missing"}, False),
            (RuntimeError("R returned incomplete results"), False),
        )
        for result, ready in cases:
            with self.subTest(result=result):
                options = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
                with mock.patch("aidas.steps.step3_flatten.check_r_packages", **options) as check:
                    self.assertEqual(frame._r_packages_ready(self.wizard.rscript_path), ready)
                check.assert_called_once_with(
                    self.wizard.rscript_path,
                    self.wizard.package_library_path,
                    Step3Frame.R_REQUIRED_PACKAGES,
                    command_builder=Step3Frame._build_r_eval_command,
                )


if __name__ == "__main__":
    unittest.main()
