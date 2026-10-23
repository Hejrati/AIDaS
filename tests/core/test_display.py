from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from aidas.core import display
from aidas.core.display import (
    centered_decorated_position,
    centered_position,
    fit_size_to_bounds,
    fractional_size_of_bounds,
)


class DisplayPositionTests(unittest.TestCase):
    def test_centers_on_primary_monitor_work_area(self):
        self.assertEqual(centered_position((0, 0, 1920, 1040), 572, 816), (674, 112))

    def test_centers_on_monitor_to_the_left_of_primary(self):
        self.assertEqual(centered_position((-1920, 0, 0, 1040), 572, 816), (-1246, 112))

    def test_centers_on_monitor_above_primary(self):
        self.assertEqual(centered_position((0, -1200, 1920, 0), 572, 816), (674, -1008))

    def test_keeps_oversized_window_at_work_area_origin(self):
        self.assertEqual(centered_position((1920, 40, 2420, 740), 572, 816), (1920, 40))

    def test_centers_the_complete_windows_frame_not_only_its_client_area(self):
        self.assertEqual(
            centered_decorated_position(
                (0, 0, 1920, 1040),
                1280,
                820,
                frame_left=8,
                frame_top=51,
            ),
            (312, 80),
        )


class DisplaySizingTests(unittest.TestCase):
    def test_fractional_size_uses_both_work_area_dimensions(self):
        self.assertEqual(
            fractional_size_of_bounds((-1920, 40, 0, 1080), 0.75),
            (1440, 780),
        )

    def test_fractional_size_bounds_invalid_fraction(self):
        bounds = (0, 0, 1000, 800)

        self.assertEqual(fractional_size_of_bounds(bounds, -1), (100, 80))
        self.assertEqual(fractional_size_of_bounds(bounds, 2), (1000, 800))

    def test_design_size_is_not_enlarged_on_large_display(self):
        self.assertEqual(
            fit_size_to_bounds((0, 0, 3840, 2160), 480, 620),
            (480, 620, 1.0),
        )

    def test_design_size_scales_down_uniformly_on_small_display(self):
        width, height, scale = fit_size_to_bounds(
            (0, 0, 320, 240),
            480,
            620,
            maximum_fraction=0.9,
        )
        self.assertLess(scale, 1.0)
        self.assertLessEqual(width, 288)
        self.assertLessEqual(height, 216)
        self.assertAlmostEqual(width / height, 480 / 620, places=2)


class MonitorWorkAreaTests(unittest.TestCase):
    @staticmethod
    def _window():
        window = mock.Mock()
        window.winfo_toplevel.return_value = window
        window.winfo_id.return_value = 101
        window.winfo_vrootx.return_value = 0
        window.winfo_vrooty.return_value = 0
        window.winfo_vrootwidth.return_value = 1920
        window.winfo_vrootheight.return_value = 1080
        return window

    @staticmethod
    def _native_api(bounds):
        def get_info(_handle, pointer):
            info = pointer._obj
            area = info.rcWork
            area.left, area.top, area.right, area.bottom = bounds
            return True

        return SimpleNamespace(
            GetAncestor=mock.Mock(return_value=202),
            MonitorFromWindow=mock.Mock(return_value=303),
            GetMonitorInfoW=mock.Mock(side_effect=get_info),
        )

    def test_native_window_query_uses_decorated_frame_and_keeps_negative_origin(self):
        window = self._window()
        api = self._native_api((-1920, 40, 0, 1080))
        with mock.patch.object(display.os, "name", "nt"), mock.patch.object(
            display.ctypes, "WinDLL", return_value=api, create=True,
        ):
            self.assertEqual(
                display.monitor_work_area_for_window(window), (-1920, 40, 0, 1080),
            )
        api.GetAncestor.assert_called_once_with(101, 2)
        api.MonitorFromWindow.assert_called_once_with(202, 2)
        self.assertEqual(api.GetMonitorInfoW.call_args.args[0], 303)

    def test_native_window_query_can_use_tk_handle_without_native_wrapper(self):
        api = self._native_api((0, 0, 1920, 1040))
        api.GetAncestor.return_value = 0
        with mock.patch.object(display.os, "name", "nt"), mock.patch.object(
            display.ctypes, "WinDLL", return_value=api, create=True,
        ):
            self.assertEqual(
                display.monitor_work_area_for_window(self._window()),
                (0, 0, 1920, 1040),
            )
        api.MonitorFromWindow.assert_called_once_with(101, 2)

    def test_native_window_query_rejects_invalid_monitor_results(self):
        for failure in ("no_monitor", "query_failed", "empty_work_area"):
            with self.subTest(failure=failure):
                api = self._native_api((0, 0, 0, 0))
                if failure == "no_monitor":
                    api.MonitorFromWindow.return_value = 0
                elif failure == "query_failed":
                    api.GetMonitorInfoW.side_effect = None
                    api.GetMonitorInfoW.return_value = False
                with mock.patch.object(display.os, "name", "nt"), mock.patch.object(
                    display.ctypes, "WinDLL", return_value=api, create=True,
                ):
                    self.assertIsNone(display.monitor_work_area_for_window(self._window()))

    def test_native_window_query_returns_none_after_window_closes(self):
        window = self._window()
        window.winfo_id.side_effect = display.tk.TclError("window was destroyed")
        api = self._native_api((0, 0, 1920, 1040))
        with mock.patch.object(display.os, "name", "nt"), mock.patch.object(
            display.ctypes, "WinDLL", return_value=api, create=True,
        ):
            self.assertIsNone(display.monitor_work_area_for_window(window))
        api.MonitorFromWindow.assert_not_called()

    def test_owner_monitor_takes_priority_over_pointer_monitor(self):
        window, owner = self._window(), self._window()
        with mock.patch.object(
            display, "monitor_work_area_for_window", return_value=(-1920, 40, 0, 1080),
        ) as native, mock.patch.object(display, "monitor_work_area_at_pointer") as pointer:
            self.assertEqual(
                display.work_area_bounds(window, monitor_window=owner),
                (-1920, 40, 0, 1080),
            )
        native.assert_called_once_with(owner)
        pointer.assert_not_called()

    def test_pointer_failure_still_excludes_taskbar_using_window_monitor(self):
        window = self._window()
        with mock.patch.object(
            display, "monitor_work_area_at_pointer", return_value=None,
        ), mock.patch.object(
            display, "monitor_work_area_for_window", return_value=(0, 0, 1920, 1040),
        ) as native, mock.patch.object(display, "_primary_monitor_work_area") as primary:
            self.assertEqual(display.work_area_bounds(window), (0, 0, 1920, 1040))
        native.assert_called_once_with(window)
        primary.assert_not_called()
        window.winfo_vrootheight.assert_not_called()

    def test_pointer_access_denied_is_a_recoverable_query_failure(self):
        api = self._native_api((0, 0, 1920, 1040))
        api.GetCursorPos = mock.Mock(return_value=False)
        api.MonitorFromPoint = mock.Mock()
        with mock.patch.object(display.os, "name", "nt"), mock.patch.object(
            display.ctypes, "WinDLL", return_value=api, create=True,
        ):
            self.assertEqual(display.work_area_bounds(self._window()), (0, 0, 1920, 1040))
        api.MonitorFromPoint.assert_not_called()
        api.MonitorFromWindow.assert_called_once()

    def test_primary_work_area_query_preserves_taskbar_on_native_window_failure(self):
        def get_primary(_action, _parameter, pointer, _flags):
            area = pointer._obj
            area.left, area.top, area.right, area.bottom = (80, 0, 1920, 1080)
            return True

        spi = mock.Mock(side_effect=get_primary)
        api = SimpleNamespace(SystemParametersInfoW=spi)
        with mock.patch.object(display.os, "name", "nt"), mock.patch.object(
            display.ctypes, "WinDLL", return_value=api, create=True,
        ), mock.patch.object(
            display, "monitor_work_area_at_pointer", return_value=None,
        ), mock.patch.object(display, "monitor_work_area_for_window", return_value=None):
            self.assertEqual(display.work_area_bounds(self._window()), (80, 0, 1920, 1080))
        self.assertEqual(spi.call_args.args[:2], (48, 0))

    def test_virtual_screen_is_last_fallback_after_all_native_queries_fail(self):
        with mock.patch.object(
            display, "monitor_work_area_at_pointer", return_value=None,
        ), mock.patch.object(
            display, "monitor_work_area_for_window", return_value=None,
        ), mock.patch.object(display, "_primary_monitor_work_area", return_value=None):
            self.assertEqual(display.work_area_bounds(self._window()), (0, 0, 1920, 1080))

    def test_visible_parent_bounds_keep_existing_behavior(self):
        parent = self._window()
        parent.winfo_viewable.return_value = True
        parent.winfo_rootx.return_value = 100
        parent.winfo_rooty.return_value = 80
        parent.winfo_width.return_value = 800
        parent.winfo_height.return_value = 600
        with mock.patch.object(display, "monitor_work_area_for_window") as native, mock.patch.object(
            display, "monitor_work_area_at_pointer",
        ) as pointer:
            self.assertEqual(
                display.work_area_bounds(self._window(), parent=parent, monitor_window=parent),
                (100, 80, 900, 680),
            )
        native.assert_not_called()
        pointer.assert_not_called()

    def test_non_windows_window_query_avoids_native_apis(self):
        with mock.patch.object(display.os, "name", "posix"), mock.patch.object(
            display.ctypes, "WinDLL", create=True,
        ) as native:
            self.assertIsNone(display.monitor_work_area_for_window(self._window()))
        native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
