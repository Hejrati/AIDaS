"""Launch-update behavior without creating Tk windows or contacting GitHub."""

from __future__ import annotations

from collections import deque
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.services.update_ui import UpdateController


class _Preferences(dict):
    def set(self, key, value):
        self[key] = value


class _Root:
    def __init__(self):
        self.callbacks = deque()
        self.scheduled = []
        self._aidas_destroying = False

    def after(self, delay, callback):
        self.scheduled.append(delay)
        self.callbacks.append(callback)
        return str(len(self.scheduled))

    def deliver(self):
        while self.callbacks:
            self.callbacks.popleft()()


class UpdateControllerTests(unittest.TestCase):
    def setUp(self):
        self.root = _Root()
        self.preferences = _Preferences(check_for_updates=True)
        self.status = mock.Mock()
        self.controller = self.make_controller()
        self.dialogs = self.enterContext(mock.patch("aidas.services.update_ui.messagebox"))
        self.dialogs.askyesno.return_value = False
        self.browser = self.enterContext(mock.patch("aidas.services.update_ui.webbrowser"))

    def make_controller(self):
        return UpdateController(
            self.root,
            preferences=self.preferences,
            current_version="3.4.4a4",
            status_callback=self.status,
            restart_blocker_callback=mock.Mock(return_value=None),
            install_callback=mock.Mock(),
        )

    def run_worker(self, thread_factory):
        thread_factory.call_args.kwargs["target"]()
        self.root.deliver()

    def test_each_new_launch_checks_even_after_a_recent_or_invalid_timestamp(self):
        now = int(time.time())
        for timestamp in (now, now + 172_800, "invalid", None, 0):
            with self.subTest(timestamp=timestamp):
                self.preferences["last_successful_update_check"] = timestamp
                controller = self.make_controller()
                with mock.patch.object(controller, "_begin_check") as begin:
                    controller.check_automatically()
                begin.assert_called_once_with(manual=False)

    def test_a_successful_previous_launch_does_not_block_the_next_launch(self):
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory, mock.patch(
            "aidas.services.update_ui.find_available_update", return_value=None,
        ) as discover:
            self.controller.check_automatically()
            self.run_worker(thread_factory)
            self.assertGreater(self.preferences["last_successful_update_check"], 0)
            next_launch = self.make_controller()
            next_launch.check_automatically()
            self.run_worker(thread_factory)
        self.assertEqual(discover.call_count, 2)

    def test_source_and_packaged_launches_both_check(self):
        for packaged in (False, True):
            with self.subTest(packaged=packaged):
                controller = self.make_controller()
                with mock.patch(
                    "aidas.services.update_ui.supports_in_app_install", return_value=packaged,
                ), mock.patch.object(controller, "_begin_check") as begin:
                    controller.check_automatically()
                begin.assert_called_once_with(manual=False)

    def test_automatic_check_is_enabled_by_default_for_older_preferences(self):
        self.preferences.pop("check_for_updates")
        with mock.patch.object(self.controller, "_begin_check") as begin:
            self.controller.check_automatically()
        begin.assert_called_once_with(manual=False)

    def test_automatic_opt_out_keeps_manual_checks_available(self):
        self.preferences["check_for_updates"] = False
        with mock.patch.object(self.controller, "_begin_check") as begin:
            self.controller.check_automatically()
            begin.assert_not_called()
            self.controller.check_now()
        begin.assert_called_once_with(manual=True)

    def test_an_opted_out_check_does_not_consume_a_later_enabled_check(self):
        self.preferences["check_for_updates"] = False
        with mock.patch.object(self.controller, "_begin_check") as begin:
            self.controller.check_automatically()
            self.preferences["check_for_updates"] = True
            self.controller.check_automatically()
        begin.assert_called_once_with(manual=False)

    def test_duplicate_automatic_calls_do_not_repeat_after_completion(self):
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory, mock.patch(
            "aidas.services.update_ui.find_available_update", return_value=None,
        ) as discover:
            self.controller.check_automatically()
            self.controller.check_automatically()
            self.run_worker(thread_factory)
            self.controller.check_automatically()
        thread_factory.assert_called_once()
        discover.assert_called_once_with("3.4.4a4")
        self.dialogs.showinfo.assert_not_called()

    def test_manual_checks_remain_available_after_startup_check(self):
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory, mock.patch(
            "aidas.services.update_ui.find_available_update", return_value=None,
        ) as discover:
            self.controller.check_automatically()
            self.run_worker(thread_factory)
            self.controller.check_now()
            self.run_worker(thread_factory)
        self.assertEqual(discover.call_count, 2)
        self.dialogs.showinfo.assert_called_once()

    def test_in_flight_manual_check_prevents_a_second_startup_worker(self):
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory, mock.patch(
            "aidas.services.update_ui.find_available_update", return_value=None,
        ) as discover:
            self.controller.check_now()
            self.controller.check_automatically()
            self.run_worker(thread_factory)
        thread_factory.assert_called_once()
        discover.assert_called_once()

    def test_in_flight_startup_check_prevents_a_second_manual_worker(self):
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory:
            self.controller.check_automatically()
            self.controller.check_now()
        thread_factory.assert_called_once()
        self.dialogs.showinfo.assert_called_once()

    def test_download_in_progress_prevents_a_startup_worker(self):
        self.controller._downloading = True
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory:
            self.controller.check_automatically()
        thread_factory.assert_not_called()
        self.status.assert_not_called()

    def test_network_work_runs_in_background_and_results_wait_for_ui_delivery(self):
        entered = threading.Event()
        release_worker = threading.Event()
        worker_ids = []
        threads = []
        real_thread = threading.Thread
        calling_thread = threading.get_ident()

        def discover(version):
            worker_ids.append(threading.get_ident())
            entered.set()
            if not release_worker.wait(5):
                raise TimeoutError("Test did not release the update worker")
            return None

        def create_thread(**kwargs):
            thread = real_thread(**kwargs)
            threads.append(thread)
            return thread

        with mock.patch(
            "aidas.services.update_ui.threading.Thread", side_effect=create_thread,
        ), mock.patch(
            "aidas.services.update_ui.find_available_update", side_effect=discover,
        ):
            try:
                self.controller.check_automatically()
                self.assertTrue(entered.wait(5), "The update worker never started")
                self.assertNotEqual(worker_ids, [calling_thread])
                self.assertTrue(self.controller._checking)
                self.assertFalse(self.root.callbacks)
                self.status.assert_called_once_with("Checking GitHub for AIDaS updates...")
            finally:
                release_worker.set()
                for thread in threads:
                    thread.join(5)
            self.assertFalse(threads[0].is_alive())
            self.assertTrue(threads[0].daemon)
            self.assertEqual(self.root.scheduled, [0])
            self.assertTrue(self.controller._checking)
            self.status.assert_called_once()
            self.root.deliver()
        self.assertFalse(self.controller._checking)
        self.status.assert_called_with("AIDaS 3.4.4a4 is up to date")

    def test_offline_automatic_check_is_quiet_and_does_not_record_success(self):
        self.preferences["last_successful_update_check"] = 123
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory, mock.patch(
            "aidas.services.update_ui.find_available_update", side_effect=OSError("offline"),
        ):
            self.controller.check_automatically()
            self.run_worker(thread_factory)
        self.assertEqual(self.preferences["last_successful_update_check"], 123)
        self.assertFalse(self.controller._checking)
        self.status.assert_called_with("Update check could not be completed")
        self.dialogs.showerror.assert_not_called()
        self.dialogs.showinfo.assert_not_called()
        self.dialogs.askyesno.assert_not_called()

    def test_offline_manual_check_explains_the_error(self):
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory, mock.patch(
            "aidas.services.update_ui.find_available_update", side_effect=OSError("offline"),
        ):
            self.controller.check_now()
            self.run_worker(thread_factory)
        self.dialogs.showerror.assert_called_once()
        self.assertIn("offline", self.dialogs.showerror.call_args.args[1])

    def test_newer_release_is_offered_after_ui_delivery(self):
        release = SimpleNamespace(version_text="3.4.4a5")
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory, mock.patch(
            "aidas.services.update_ui.find_available_update", return_value=release,
        ), mock.patch.object(self.controller, "_offer_release") as offer, mock.patch(
            "aidas.services.update_ui.time.time", return_value=456,
        ):
            self.controller.check_automatically()
            thread_factory.call_args.kwargs["target"]()
            offer.assert_not_called()
            self.root.deliver()
        offer.assert_called_once_with(release)
        self.status.assert_called_with("AIDaS 3.4.4a5 is available")
        self.assertEqual(self.preferences["last_successful_update_check"], 456)

    def test_source_run_offers_the_release_page(self):
        release = SimpleNamespace(
            version_text="3.4.4a5", notes="Changes", page_url="https://github.com/Hejrati/AIDaS/releases",
        )
        self.dialogs.askyesno.return_value = True
        with mock.patch(
            "aidas.services.update_ui.supports_in_app_install", return_value=False,
        ), mock.patch.object(self.controller, "_begin_download") as download:
            self.controller._offer_release(release)
        self.browser.open_new_tab.assert_called_once_with(release.page_url)
        download.assert_not_called()

    def test_closing_before_worker_finishes_skips_scheduling_and_dialogs(self):
        release = SimpleNamespace(version_text="3.4.4a5")
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory, mock.patch(
            "aidas.services.update_ui.find_available_update", return_value=release,
        ), mock.patch.object(self.controller, "_offer_release") as offer:
            self.controller.check_automatically()
            self.root._aidas_destroying = True
            self.run_worker(thread_factory)
        self.assertFalse(self.root.scheduled)
        offer.assert_not_called()
        self.status.assert_called_once()
        self.assertNotIn("last_successful_update_check", self.preferences)

    def test_closing_after_result_is_queued_skips_delivery(self):
        release = SimpleNamespace(version_text="3.4.4a5")
        with mock.patch("aidas.services.update_ui.threading.Thread") as thread_factory, mock.patch(
            "aidas.services.update_ui.find_available_update", return_value=release,
        ), mock.patch.object(self.controller, "_offer_release") as offer:
            self.controller.check_automatically()
            thread_factory.call_args.kwargs["target"]()
            self.assertTrue(self.root.callbacks)
            self.root._aidas_destroying = True
            self.root.deliver()
        offer.assert_not_called()
        self.status.assert_called_once()

    def test_scheduling_during_tk_shutdown_is_safe(self):
        for error in (RuntimeError("main loop ended"), tk.TclError("application destroyed")):
            with self.subTest(error=error), mock.patch.object(
                self.root, "after", side_effect=error,
            ):
                self.controller._schedule(mock.Mock())


if __name__ == "__main__":
    unittest.main()
