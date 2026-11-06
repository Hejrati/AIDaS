"""Startup update scheduling without constructing Tk windows or using the network."""

from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.app import AIDaSApp


class StartupUpdateTests(unittest.TestCase):
    def test_every_interface_schedules_one_launch_check_with_legacy_opt_out(self):
        for mode in ("Modern", "Classic"):
            with self.subTest(mode=mode):
                app = mock.Mock()
                app._startup_finished = False
                app._startup_window_size = (1200, 800)
                app.window_title_bar = None
                app.interface_mode = mode
                app.preferences = {"check_for_updates": False}
                check = mock.Mock()
                app.update_controller = SimpleNamespace(check_automatically=check)
                scheduled = []

                def schedule(delay, callback):
                    app.deiconify.assert_called_once_with()
                    scheduled.append((delay, callback))
                    return "update-check"

                app.after.side_effect = schedule
                with (
                    mock.patch("aidas.app._center_geometry", return_value="1200x800+0+0"),
                    mock.patch("aidas.app.synchronize_window_chrome"),
                    mock.patch("aidas.ui.rendering.finish_deferred_rendering"),
                ):
                    AIDaSApp._finish_startup(app)
                    AIDaSApp._finish_startup(app)

                self.assertEqual(scheduled, [(1500, check)])
                check.assert_not_called()
                scheduled[0][1]()
                check.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
