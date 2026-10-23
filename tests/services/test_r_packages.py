from __future__ import annotations

import os
from concurrent.futures import CancelledError
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from aidas.utils.r_packages import check_r_packages


TAG = "__AIDAS_R_PACKAGE_CHECK__"


def _record(package, status):
    return f"{TAG}\t{package}\t{status}\n"


def _command(executable, expression):
    return [str(executable), "--vanilla", "-e", expression]


class RPackageCheckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.library = Path(self.temporary.name) / "new library's path"

    def test_empty_input_does_not_prepare_library_or_launch_r(self):
        builder = mock.Mock()
        with mock.patch("aidas.utils.r_packages.subprocess.run") as run:
            self.assertEqual(check_r_packages("Rscript", self.library, [], command_builder=builder), {})
        builder.assert_not_called()
        run.assert_not_called()
        self.assertFalse(self.library.exists())

    def test_pre_cancelled_check_does_not_prepare_library_or_launch_r(self):
        cancellation = threading.Event()
        cancellation.set()
        builder = mock.Mock()
        with mock.patch("aidas.utils.r_packages.subprocess.run") as run:
            with mock.patch("aidas.utils.r_packages.run_cancellable_process") as cancellable:
                with self.assertRaises(CancelledError):
                    check_r_packages("Rscript", self.library, ["stats"], command_builder=builder, cancel_event=cancellation)
        builder.assert_not_called()
        run.assert_not_called()
        cancellable.assert_not_called()
        self.assertFalse(self.library.exists())

    def test_cancellation_propagates_without_becoming_package_status_or_runtime_error(self):
        cancellation = threading.Event()
        with mock.patch("aidas.utils.r_packages.subprocess.run") as run:
            with mock.patch("aidas.utils.r_packages.run_cancellable_process", side_effect=CancelledError) as cancellable:
                with self.assertRaises(CancelledError):
                    check_r_packages("Rscript", self.library, ["stats"], command_builder=_command, cancel_event=cancellation)
        run.assert_not_called()
        self.assertIs(cancellable.call_args.kwargs["cancel_event"], cancellation)
        self.assertEqual(cancellable.call_args.kwargs["timeout"], 120)

    def test_completed_cancellable_check_keeps_status_parsing_and_logging(self):
        cancellation = threading.Event()
        completed = subprocess.CompletedProcess([], 0, _record("stats", "installed"), "")
        logger = mock.Mock()
        with mock.patch("aidas.utils.r_packages.run_cancellable_process", return_value=completed):
            statuses = check_r_packages(
                "Rscript", self.library, ["stats"], command_builder=_command,
                cancel_event=cancellation, log_result=logger,
            )
        self.assertEqual(statuses, {"stats": "installed"})
        logger.assert_called_once()

    def test_one_process_checks_deduplicated_packages_and_ignores_chatter(self):
        stdout = (
            "R startup message\n"
            + _record("missing.package", "missing")
            + "Loading required namespace\n"
            + _record("AnalyzeFMRI", "installed")
        )
        completed = subprocess.CompletedProcess([], 0, stdout, "Startup warning\n")
        builder = mock.Mock(side_effect=_command)
        logger = mock.Mock()
        with mock.patch.dict(os.environ, {"AIDAS_R_CHECK_TEST": "preserved"}):
            with mock.patch("aidas.utils.r_packages.subprocess.run", return_value=completed) as run:
                statuses = check_r_packages(
                    "Rscript", self.library,
                    ["AnalyzeFMRI", "missing.package", "AnalyzeFMRI"],
                    command_builder=builder, log_result=logger,
                )

        self.assertEqual(list(statuses.items()), [("AnalyzeFMRI", "installed"), ("missing.package", "missing")])
        self.assertTrue(self.library.is_dir())
        builder.assert_called_once()
        executable, expression = builder.call_args.args
        self.assertEqual(executable, "Rscript")
        self.assertIn(".libPaths(c(", expression)
        self.assertIn("new library\\'s path", expression)
        self.assertIn("for (.aidas_pkg in c('AnalyzeFMRI', 'missing.package'))", expression)
        self.assertIn("tryCatch(requireNamespace(.aidas_pkg, quietly=TRUE), error=function(e) FALSE)", expression)
        run.assert_called_once()
        command = run.call_args.args[0]
        options = run.call_args.kwargs
        self.assertEqual(command, ["Rscript", "--vanilla", "-e", expression])
        self.assertEqual(options["env"]["R_LIBS_USER"], str(self.library.resolve()))
        self.assertEqual(options["env"]["AIDAS_R_CHECK_TEST"], "preserved")
        self.assertEqual(options["timeout"], 120)
        self.assertEqual(options["creationflags"], getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual((options["stdout"], options["stderr"]), (subprocess.PIPE, subprocess.PIPE))
        self.assertEqual((options["text"], options["encoding"], options["errors"]), (True, "utf-8", "replace"))
        logger.assert_called_once_with("Check R packages", command, completed)

    def test_malformed_or_incomplete_records_raise_after_logging(self):
        cases = {
            "no records": "ordinary R startup output\n",
            "one missing": _record("stats", "installed"),
            "invalid status": _record("stats", "loaded") + _record("utils", "installed"),
            "missing status column": f"{TAG}\tstats\n",
            "extra column": f"{TAG}\tstats\tinstalled\textra\n",
            "wrong package": _record("other", "installed"),
            "duplicate": _record("stats", "installed") * 2 + _record("utils", "installed"),
            "tag without delimiter": TAG + " invalid\n",
        }
        for label, stdout in cases.items():
            with self.subTest(case=label):
                completed = subprocess.CompletedProcess([], 0, stdout, "diagnostic stderr")
                logger = mock.Mock()
                with mock.patch("aidas.utils.r_packages.subprocess.run", return_value=completed):
                    with self.assertRaises(RuntimeError):
                        check_r_packages("Rscript", self.library, ["stats", "utils"], command_builder=_command, log_result=logger)
                logger.assert_called_once()
                self.assertIs(logger.call_args.args[2], completed)

    def test_nonzero_exit_rejects_even_complete_status_records_and_logs_diagnostics(self):
        completed = subprocess.CompletedProcess([], 17, _record("stats", "installed"), "R failed after output")
        logger = mock.Mock()
        with mock.patch("aidas.utils.r_packages.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "exit code 17"):
                check_r_packages("Rscript", self.library, ["stats"], command_builder=_command, log_result=logger)
        logger.assert_called_once()
        self.assertIs(logger.call_args.args[2], completed)

    def test_launch_and_timeout_errors_cannot_report_package_statuses(self):
        for error in (OSError("Cannot launch R"), subprocess.TimeoutExpired(["Rscript"], 120)):
            with self.subTest(error=type(error).__name__):
                with mock.patch("aidas.utils.r_packages.subprocess.run", side_effect=error):
                    with self.assertRaisesRegex(RuntimeError, "could not complete"):
                        check_r_packages("Rscript", self.library, ["stats"], command_builder=_command)

    def test_library_creation_failure_does_not_launch_r(self):
        self.library.write_text("a file prevents creating this library", encoding="utf-8")
        with mock.patch("aidas.utils.r_packages.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "Cannot prepare"):
                check_r_packages("Rscript", self.library, ["stats"], command_builder=_command)
        run.assert_not_called()


class RealR331PackageCheckTests(unittest.TestCase):
    def test_installed_and_nonexistent_packages_with_r331(self):
        from tests.support.r_runtime import find_r331

        executable = find_r331()
        if executable is None:
            message = "R 3.3.1 is required for the real batched package check."
            if os.environ.get("AIDAS_REQUIRE_R331_INTEGRATION") == "1":
                self.fail(message)
            self.skipTest(message)
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "library's package directory"
            statuses = check_r_packages(
                executable, library, ["stats", "utils", "aidas.nonexistent.package.control"],
                command_builder=_command,
            )
        self.assertEqual(statuses, {
            "stats": "installed", "utils": "installed",
            "aidas.nonexistent.package.control": "missing",
        })


if __name__ == "__main__":
    unittest.main()
