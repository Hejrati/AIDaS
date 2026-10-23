from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.steps.step3_flatten import RSetupWizard, Step3Frame


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class RSetupAsyncStartupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve()
        try:
            self.root = _QuietRoot()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        self.addCleanup(self.root.destroy)
        self.root.attributes("-alpha", 0)
        self.root.geometry("1050x650+0+0")
        self.root.update()
        self.callback_errors = mock.Mock()
        self.root.report_callback_exception = self.callback_errors
        self.main_thread = threading.get_ident()
        self.started = threading.Event()
        self.release = threading.Event()
        self.addCleanup(self.release.set)
        self.resolver_threads = []
        self.package_threads = []
        self.completion_threads = []
        self.rscript = self.folder / "Rscript.exe"
        self.rscript.touch()
        self.statuses = {name: "installed" for name in Step3Frame.R_LOCAL_PACKAGE_ORDER}

    def _wait_for(self, predicate, message="The asynchronous R setup did not finish"):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(0.005)
        self.assertTrue(predicate(), message)

    def _create_wizard(self, *, start_initial_check=True, package_worker=None,
                       preferences=None, on_finish=None):
        def resolve(*, cancel_event=None):
            self.resolver_threads.append(threading.current_thread())
            self.started.set()
            if not self.release.wait(timeout=3):
                raise TimeoutError("The test did not release R detection")
            return self.rscript

        def check_packages(_wizard):
            self.package_threads.append(threading.get_ident())
            if package_worker is not None:
                return package_worker()
            return self.statuses.copy()

        finish_worker = RSetupWizard._finish_worker

        def record_completion(wizard, done, value, error):
            self.completion_threads.append(threading.get_ident())
            return finish_worker(wizard, done, value, error)

        step = SimpleNamespace(
            **{name: getattr(Step3Frame, name) for name in (
                "R_REQUIRED_VERSION", "R_REQUIRED_PACKAGES", "R_LOCAL_PACKAGE_ORDER",
                "R_LOCAL_PACKAGE_FILES", "R_INSTALLER_NAME", "R_DOWNLOAD_PAGE",
            )},
            r_package_library_path=str(self.folder / "R-packages"),
            preferences=preferences,
            after=self.root.after,
            current_sdb_dir=str(self.folder),
            _normalize_r_executable=Step3Frame._normalize_r_executable,
            _r_version_for_executable=mock.Mock(return_value="3.3.1"),
            _resolve_rscript_executable=mock.Mock(side_effect=resolve),
        )
        self.enterContext(mock.patch.object(
            RSetupWizard, "_package_log_path", return_value=self.folder / "setup.log",
        ))
        self.package_check = self.enterContext(mock.patch.object(
            RSetupWizard, "_check_package_status_worker", autospec=True, side_effect=check_packages,
        ))
        self.enterContext(mock.patch.object(RSetupWizard, "_finish_worker", record_completion))
        wizard = RSetupWizard(step, self.root, on_finish=on_finish)
        wizard.pack(fill="both", expand=True)
        if not start_initial_check:
            wizard.after_cancel(wizard._initial_check_after_id)
            wizard._initial_check_after_id = None
            wizard._detecting_r = False
        return wizard, step

    def test_blocked_r_detection_keeps_startup_and_the_event_loop_responsive(self):
        wizard, step = self._create_wizard()
        self.assertFalse(self.release.is_set())
        self.assertTrue(wizard._detecting_r)
        self.assertIsNone(wizard.rscript_path)
        step._resolve_rscript_executable.assert_not_called()

        heartbeat = mock.Mock()
        self.root.after(0, heartbeat)
        self._wait_for(lambda: self.started.is_set() and heartbeat.called)

        self.assertTrue(wizard.package_tree.winfo_viewable())
        self.assertTrue(wizard.close_button.winfo_viewable())
        self.assertEqual(wizard.close_button.cget("state"), "normal")
        self.assertEqual(wizard.check_packages_button.cget("state"), "disabled")
        self.assertTrue(wizard._detecting_r)
        self.assertTrue(self.resolver_threads[0].is_alive())
        self.assertNotEqual(self.resolver_threads[0].ident, self.main_thread)
        self.package_check.assert_not_called()
        self.callback_errors.assert_not_called()

        self.release.set()
        self._wait_for(lambda: wizard.package_status == self.statuses and not wizard.busy)

        self.assertEqual(wizard.rscript_path, self.rscript)
        self.assertFalse(wizard._detecting_r)
        step._resolve_rscript_executable.assert_called_once_with(cancel_event=wizard._cancel_event)
        self.package_check.assert_called_once_with(wizard)
        self.assertTrue(self.package_threads)
        self.assertTrue(all(thread != self.main_thread for thread in self.package_threads))
        self.assertEqual(self.completion_threads, [self.main_thread, self.main_thread])
        self.callback_errors.assert_not_called()

    def test_destroy_during_blocked_detection_discards_worker_completion_safely(self):
        wizard, _step = self._create_wizard()
        self._wait_for(self.started.is_set)
        original_after = wizard.after
        after_threads = []

        def record_after(*args, **kwargs):
            after_threads.append(threading.get_ident())
            return original_after(*args, **kwargs)

        with mock.patch.object(wizard, "after", side_effect=record_after), \
                mock.patch.object(threading, "excepthook") as thread_error:
            wizard.destroy()
            self.release.set()
            self._wait_for(lambda: not self.resolver_threads[0].is_alive())
            settled = tk.BooleanVar(master=self.root, value=False)
            self.root.after(80, lambda: settled.set(True))
            self.root.wait_variable(settled)
            self.root.update()

            self.assertFalse(wizard.winfo_exists())
            self.assertEqual(self.completion_threads, [])
            self.package_check.assert_not_called()
            self.callback_errors.assert_not_called()
            thread_error.assert_not_called()
            self.assertTrue(all(thread == self.main_thread for thread in after_threads))

    def test_close_before_initial_detection_dismisses_without_saving_or_callback(self):
        preferences = mock.Mock()
        on_finish = mock.Mock()
        wizard, step = self._create_wizard(preferences=preferences, on_finish=on_finish)
        self.assertEqual(wizard.close_button.cget("state"), "normal")

        wizard.close_button.invoke()
        self.root.update()

        self.assertFalse(wizard.winfo_exists())
        self.assertTrue(wizard._cancel_event.is_set())
        self.assertTrue(wizard.cancelled)
        self.assertIsNone(wizard.result)
        step._resolve_rscript_executable.assert_not_called()
        self.package_check.assert_not_called()
        preferences.set.assert_not_called()
        on_finish.assert_not_called()
        self.callback_errors.assert_not_called()

    def test_close_during_blocked_detection_discards_the_late_result(self):
        preferences = mock.Mock()
        on_finish = mock.Mock()
        wizard, _step = self._create_wizard(preferences=preferences, on_finish=on_finish)
        self._wait_for(self.started.is_set)
        self.assertEqual(wizard.close_button.cget("state"), "normal")
        self.assertEqual(wizard.check_packages_button.cget("state"), "disabled")

        with mock.patch.object(threading, "excepthook") as thread_error:
            wizard.close_button.invoke()
            self.assertFalse(wizard.winfo_exists())
            self.assertFalse(self.release.is_set())
            self.assertTrue(self.resolver_threads[0].is_alive())
            self.assertTrue(wizard._cancel_event.is_set())
            self.release.set()
            self._wait_for(lambda: not self.resolver_threads[0].is_alive())
            self.root.update()

            self.assertEqual(self.completion_threads, [])
            self.package_check.assert_not_called()
            preferences.set.assert_not_called()
            on_finish.assert_not_called()
            self.callback_errors.assert_not_called()
            thread_error.assert_not_called()

    def test_close_during_blocked_package_work_dismisses_before_the_worker_returns(self):
        package_started = threading.Event()
        package_release = threading.Event()
        self.addCleanup(package_release.set)
        workers = []

        def package_worker():
            workers.append(threading.current_thread())
            package_started.set()
            if not package_release.wait(timeout=3):
                raise TimeoutError("The test did not release package checking")
            return self.statuses.copy()

        self.release.set()
        preferences = mock.Mock()
        on_finish = mock.Mock()
        wizard, _step = self._create_wizard(
            package_worker=package_worker, preferences=preferences, on_finish=on_finish,
        )
        self._wait_for(package_started.is_set)
        self.assertEqual(wizard.close_button.cget("state"), "normal")
        self.assertEqual(wizard.check_packages_button.cget("state"), "disabled")
        self.assertEqual(self.completion_threads, [self.main_thread])

        with mock.patch.object(threading, "excepthook") as thread_error:
            wizard.close_button.invoke()
            self.assertFalse(wizard.winfo_exists())
            self.assertFalse(package_release.is_set())
            self.assertTrue(workers[0].is_alive())
            self.assertTrue(wizard._cancel_event.is_set())
            package_release.set()
            self._wait_for(lambda: not workers[0].is_alive())
            self.root.update()

            self.assertEqual(self.completion_threads, [self.main_thread])
            self.assertNotEqual(wizard.package_status, self.statuses)
            preferences.set.assert_not_called()
            on_finish.assert_not_called()
            self.callback_errors.assert_not_called()
            thread_error.assert_not_called()

    def test_ready_close_saves_configuration_and_delivers_the_finish_callback(self):
        self.release.set()
        preferences = mock.Mock()
        on_finish = mock.Mock()
        wizard, step = self._create_wizard(preferences=preferences, on_finish=on_finish)
        self._wait_for(lambda: wizard.package_status == self.statuses and not wizard.busy)
        self.assertEqual(wizard.close_button.cget("state"), "normal")
        self.assertEqual(wizard.check_packages_button.cget("state"), "normal")

        wizard.close_button.invoke()
        self.root.update()

        self.assertFalse(wizard.winfo_exists())
        self.assertFalse(wizard.cancelled)
        self.assertEqual(wizard.result, self.rscript)
        self.assertEqual(step.r_package_library_path, str(wizard.package_library_path))
        self.assertEqual(preferences.set.call_args_list, [
            mock.call("rscript_path", str(self.rscript)),
            mock.call("r_package_library_path", str(wizard.package_library_path)),
        ])
        on_finish.assert_called_once_with(self.rscript)
        self.callback_errors.assert_not_called()

    def test_destroy_before_initial_check_never_starts_r_detection(self):
        wizard, step = self._create_wizard()

        wizard.destroy()
        self.root.update()

        step._resolve_rscript_executable.assert_not_called()
        self.package_check.assert_not_called()
        self.assertEqual(self.completion_threads, [])
        self.callback_errors.assert_not_called()

    def test_installer_waits_for_async_detection_before_starting_package_installation(self):
        wizard, step = self._create_wizard(start_initial_check=False)
        wizard.installer_path = self.folder / "test-installer.exe"
        wizard.installer_path.touch()
        installation_threads = []
        with mock.patch(
            "aidas.steps.step3_flatten.subprocess.run", return_value=SimpleNamespace(returncode=0),
        ) as installer, mock.patch("aidas.steps.step3_flatten.messagebox.showinfo") as ready_message, \
                mock.patch.object(
                    wizard, "_install_missing_packages",
                    side_effect=lambda: installation_threads.append(threading.get_ident()),
                ) as install_packages:
            wizard._run_downloaded_installer()
            self._wait_for(self.started.is_set)

            installer.assert_called_once_with([str(wizard.installer_path)], check=False)
            self.assertTrue(wizard._detecting_r)
            self.assertTrue(wizard.busy)
            self.assertNotEqual(self.resolver_threads[0].ident, self.main_thread)
            ready_message.assert_not_called()
            install_packages.assert_not_called()
            self.package_check.assert_not_called()

            self.release.set()
            self._wait_for(lambda: install_packages.called and not wizard.busy)

            self.assertEqual(wizard.rscript_path, self.rscript)
            self.assertFalse(wizard._detecting_r)
            step._resolve_rscript_executable.assert_called_once_with(cancel_event=wizard._cancel_event)
            ready_message.assert_called_once()
            install_packages.assert_called_once_with()
            self.assertEqual(installation_threads, [self.main_thread])
            self.package_check.assert_not_called()
            self.callback_errors.assert_not_called()

    def test_close_during_manual_r_version_check_discards_the_late_selection(self):
        version_started = threading.Event()
        version_release = threading.Event()
        self.addCleanup(version_release.set)
        workers = []

        def check_version(executable, *, cancel_event=None):
            self.assertEqual(executable, self.rscript)
            workers.append(threading.current_thread())
            version_started.set()
            if not version_release.wait(timeout=3):
                raise TimeoutError("The test did not release R version checking")
            return "3.3.1"

        preferences = mock.Mock()
        on_finish = mock.Mock()
        wizard, step = self._create_wizard(
            start_initial_check=False, preferences=preferences, on_finish=on_finish,
        )
        step._r_version_for_executable.side_effect = check_version
        with mock.patch(
            "aidas.steps.step3_flatten.filedialog.askopenfilename", return_value=str(self.rscript),
        ), mock.patch("aidas.steps.step3_flatten.messagebox.showerror") as selection_error, \
                mock.patch.object(threading, "excepthook") as thread_error:
            wizard._locate_rscript()
            self._wait_for(version_started.is_set)
            self.assertNotEqual(workers[0].ident, self.main_thread)
            self.assertEqual(wizard.close_button.cget("state"), "normal")
            self.assertEqual(wizard.check_packages_button.cget("state"), "disabled")

            wizard.close_button.invoke()
            self.assertFalse(wizard.winfo_exists())
            self.assertFalse(version_release.is_set())
            self.assertTrue(workers[0].is_alive())
            self.assertTrue(wizard._cancel_event.is_set())
            version_release.set()
            self._wait_for(lambda: not workers[0].is_alive())
            self.root.update()

            step._r_version_for_executable.assert_called_once_with(
                self.rscript, cancel_event=wizard._cancel_event,
            )
            self.assertEqual(self.completion_threads, [])
            self.package_check.assert_not_called()
            preferences.set.assert_not_called()
            on_finish.assert_not_called()
            selection_error.assert_not_called()
            self.callback_errors.assert_not_called()
            thread_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
