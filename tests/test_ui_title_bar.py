from __future__ import annotations

import unittest
import threading
from types import SimpleNamespace

from aidas.ui.title_bar import (
    CustomWindowsTitleBar,
    WindowsCaptionController,
    _SWP_ASYNCWINDOWPOS,
    _WindowsAPI,
    _REQUIRED_NATIVE_STYLES,
    _RETAINED_NATIVE_STYLES,
    _WS_CAPTION,
    _WS_THICKFRAME,
    _WS_VISIBLE,
    _outer_rect_for_client_work_area,
    _resized_window_rect,
    reassert_client_size,
)


class _FakeNativeAPI:
    def __init__(
        self,
        style: int,
        *,
        refresh_succeeds: bool = True,
        refresh_raises: bool = False,
        frame_applied: bool = True,
    ):
        self.style = style
        self.refresh_succeeds = refresh_succeeds
        self.refresh_raises = refresh_raises
        self.frame_applied = frame_applied
        self.set_styles: list[int] = []
        self.refresh_count = 0
        self.drag_count = 0
        self.zoomed = False
        self.fit_count = 0
        self.close_count = 0
        self.resize_calls: list[tuple[int, int, int, int, int]] = []

    def root_handle(self, _window) -> int:
        return 101

    def get_style(self, _handle: int) -> int:
        return self.style

    def set_style(self, _handle: int, style: int) -> bool:
        self.style = style
        self.set_styles.append(style)
        return True

    def refresh_frame(self, _handle: int) -> bool:
        self.refresh_count += 1
        if self.refresh_raises:
            self.refresh_raises = False
            raise OSError("simulated FRAMECHANGED failure")
        return self.refresh_succeeds

    def frame_insets(self, _handle: int):
        if not self.frame_applied:
            return (0, 0, 0, 0)
        if self.style & _WS_CAPTION:
            return (8, 31, 8, 8)
        return (0, 0, 0, 0)

    def begin_caption_drag(self, _handle: int) -> None:
        self.drag_count += 1

    def resize_window(
        self,
        handle: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bool:
        self.resize_calls.append((handle, x, y, width, height))
        return True

    def is_zoomed(self, _handle: int) -> bool:
        return self.zoomed

    def fit_maximized_to_work_area(self, _handle: int) -> bool:
        self.fit_count += 1
        return True

    def close_window(self, _handle: int) -> None:
        self.close_count += 1


class _ActionWindow:
    def __init__(self, api: _FakeNativeAPI):
        self.api = api
        self.states: list[str] = []
        self.iconify_count = 0
        self.deiconify_count = 0

    def state(self, state: str) -> None:
        self.states.append(state)
        self.api.zoomed = state == "zoomed"

    def iconify(self) -> None:
        self.iconify_count += 1

    def deiconify(self) -> None:
        self.deiconify_count += 1

    def after_idle(self, _callback):
        return "after-id"

    def after(self, _delay, callback):
        callback()
        return "after-id"


class _FrameAdjustedWindow:
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.width = 1
        self.height = 1
        self.geometry_calls: list[str] = []

    def geometry(self, geometry: str) -> None:
        self.geometry_calls.append(geometry)
        width, height = (int(value) for value in geometry.split("x"))
        self.width = int(width * self.scale) + 2
        self.height = int(height * self.scale)

    def _apply_window_scaling(self, value: int) -> int:
        return int(value * self.scale)

    def _reverse_window_scaling(self, value: int) -> int:
        return int(value / self.scale)

    def update_idletasks(self) -> None:
        pass

    def winfo_width(self) -> int:
        return self.width

    def winfo_height(self) -> int:
        return self.height


class WindowsCaptionControllerTests(unittest.TestCase):
    def test_cached_root_handle_skips_a_finished_widget_tree_flush(self):
        calls = []
        window = SimpleNamespace(
            _aidas_native_root_handle=4321,
            update_idletasks=lambda: calls.append("flush"),
        )
        api = _WindowsAPI.__new__(_WindowsAPI)

        self.assertEqual(api.root_handle(window), 4321)
        self.assertEqual(calls, [])

    def test_frame_refresh_is_posted_from_a_non_ui_thread(self):
        api = _WindowsAPI.__new__(_WindowsAPI)
        calls = []
        ui_thread = threading.get_ident()

        def set_window_pos(*args):
            calls.append((threading.get_ident(), args))
            return True

        api._frame_set_window_pos = set_window_pos

        self.assertTrue(api.refresh_frame(101))
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(calls[0][0], ui_thread)
        self.assertTrue(calls[0][1][-1] & _SWP_ASYNCWINDOWPOS)

    def test_descendant_configure_does_not_schedule_title_state_work(self):
        title_bar = object.__new__(CustomWindowsTitleBar)
        root = object()
        title_bar.master = root
        title_bar._state_sync_after_id = None
        scheduled = []
        title_bar.after_idle = lambda callback: scheduled.append(callback) or "after#1"

        title_bar._queue_window_state_sync(SimpleNamespace(widget=object()))

        self.assertEqual(scheduled, [])
        self.assertIsNone(title_bar._state_sync_after_id)

    def test_root_configure_still_schedules_one_debounced_state_sync(self):
        title_bar = object.__new__(CustomWindowsTitleBar)
        root = object()
        title_bar.master = root
        title_bar._state_sync_after_id = None
        scheduled = []
        title_bar.after_idle = lambda callback: scheduled.append(callback) or "after#1"

        root_event = SimpleNamespace(widget=root)
        title_bar._queue_window_state_sync(root_event)
        title_bar._queue_window_state_sync(root_event)

        self.assertEqual(scheduled, [title_bar._sync_window_state])
        self.assertEqual(title_bar._state_sync_after_id, "after#1")

    def test_maximized_outer_rect_accounts_for_all_nonclient_margins(self):
        self.assertEqual(
            _outer_rect_for_client_work_area(
                (0, 0, 1920, 1040),
                (-8, -8, 1928, 1088),
                (0, 0, 1920, 1080),
            ),
            (-8, -8, 1936, 1056),
        )

    def test_resized_window_rect_moves_only_the_selected_edges(self):
        start = (100, 200, 800, 600)
        cases = {
            "e": (50, 25, (100, 200, 850, 600)),
            "s": (50, 25, (100, 200, 800, 625)),
            "nw": (-20, -30, (80, 170, 820, 630)),
            "se": (50, 25, (100, 200, 850, 625)),
        }
        for edge, (delta_x, delta_y, expected) in cases.items():
            with self.subTest(edge=edge):
                self.assertEqual(
                    _resized_window_rect(
                        start, edge, delta_x, delta_y, 300, 250
                    ),
                    expected,
                )

    def test_resized_window_rect_clamps_at_minimum_size(self):
        self.assertEqual(
            _resized_window_rect((100, 200, 800, 600), "nw", 900, 700, 320, 240),
            (580, 560, 320, 240),
        )
        self.assertEqual(
            _resized_window_rect((100, 200, 800, 600), "se", -900, -700, 320, 240),
            (100, 200, 320, 240),
        )

    def test_client_size_is_measured_and_corrected_after_frame_change(self):
        window = _FrameAdjustedWindow()

        self.assertEqual(reassert_client_size(window, 1000, 700), (1000, 700))
        self.assertEqual(window.geometry_calls, ["1000x700", "998x700"])

    def test_client_size_correction_uses_physical_targets_at_fractional_dpi(self):
        window = _FrameAdjustedWindow(scale=1.5)

        self.assertEqual(reassert_client_size(window, 1000, 700), (1500, 1050))
        self.assertEqual(window.geometry_calls, ["1000x700", "999x700"])

    def test_install_removes_caption_and_resize_frame_but_retains_window_commands(self):
        original_style = 0x16000008 | _WS_CAPTION | _REQUIRED_NATIVE_STYLES
        api = _FakeNativeAPI(original_style)
        controller = WindowsCaptionController(object(), native_api=api)

        self.assertTrue(controller.install())
        self.assertEqual(controller.handle, 101)
        self.assertEqual(controller.original_style, original_style)
        expected_style = original_style & ~(_WS_CAPTION | _WS_THICKFRAME)
        self.assertEqual(controller.style, expected_style)
        self.assertEqual(api.style & _WS_CAPTION, 0)
        self.assertEqual(api.style & _WS_THICKFRAME, 0)
        self.assertEqual(
            api.style & _RETAINED_NATIVE_STYLES,
            _RETAINED_NATIVE_STYLES,
        )
        self.assertEqual(api.set_styles, [expected_style])
        self.assertEqual(api.refresh_count, 1)

    def test_unexpected_nonresizable_style_fails_without_mutation(self):
        original_style = _WS_CAPTION
        api = _FakeNativeAPI(original_style)
        controller = WindowsCaptionController(object(), native_api=api)

        self.assertFalse(controller.install())
        self.assertFalse(controller.installed)
        self.assertEqual(api.style, original_style)
        self.assertEqual(api.set_styles, [])
        self.assertEqual(api.refresh_count, 0)

    def test_failed_frame_refresh_rolls_back_original_style(self):
        original_style = 0x16000008 | _WS_CAPTION | _REQUIRED_NATIVE_STYLES
        api = _FakeNativeAPI(original_style, refresh_succeeds=False)
        controller = WindowsCaptionController(object(), native_api=api)

        self.assertFalse(controller.install())
        self.assertEqual(api.style, original_style)
        self.assertEqual(
            api.set_styles,
            [original_style & ~(_WS_CAPTION | _WS_THICKFRAME), original_style],
        )
        self.assertEqual(api.refresh_count, 2)

    def test_frame_refresh_exception_rolls_back_original_style(self):
        original_style = 0x16000008 | _WS_CAPTION | _REQUIRED_NATIVE_STYLES
        api = _FakeNativeAPI(original_style, refresh_raises=True)
        controller = WindowsCaptionController(object(), native_api=api)

        self.assertFalse(controller.install())
        self.assertEqual(api.style, original_style)
        self.assertEqual(
            api.set_styles,
            [original_style & ~(_WS_CAPTION | _WS_THICKFRAME), original_style],
        )
        self.assertEqual(api.refresh_count, 2)

    def test_restore_reinstates_exact_original_style(self):
        original_style = 0x16000008 | _WS_CAPTION | _REQUIRED_NATIVE_STYLES
        api = _FakeNativeAPI(original_style)
        controller = WindowsCaptionController(object(), native_api=api)
        self.assertTrue(controller.install())

        self.assertTrue(controller.restore_native_caption())
        self.assertFalse(controller.installed)
        self.assertEqual(api.style, original_style)
        self.assertEqual(api.refresh_count, 2)

    def test_restore_keeps_client_controls_installed_until_frame_is_visible(self):
        original_style = 0x16000008 | _WS_CAPTION | _REQUIRED_NATIVE_STYLES
        api = _FakeNativeAPI(original_style, frame_applied=False)
        window = SimpleNamespace(_aidas_suppress_native_border=True)
        controller = WindowsCaptionController(window, native_api=api)
        self.assertTrue(controller.install())

        self.assertTrue(controller.begin_restore_native_caption())
        self.assertTrue(controller.installed)
        self.assertTrue(window._aidas_suppress_native_border)
        self.assertIsNone(controller.finish_restore_native_caption())

        api.frame_applied = True
        self.assertTrue(controller.finish_restore_native_caption())
        self.assertFalse(controller.installed)
        self.assertFalse(window._aidas_suppress_native_border)

    def test_failed_restore_refresh_rolls_back_captionless_style(self):
        original_style = 0x16000008 | _WS_CAPTION | _REQUIRED_NATIVE_STYLES
        api = _FakeNativeAPI(original_style)
        window = SimpleNamespace(_aidas_suppress_native_border=True)
        controller = WindowsCaptionController(window, native_api=api)
        self.assertTrue(controller.install())
        captionless_style = controller.style
        api.refresh_succeeds = False

        self.assertFalse(controller.begin_restore_native_caption())

        self.assertTrue(controller.installed)
        self.assertEqual(api.style, captionless_style)
        self.assertTrue(window._aidas_suppress_native_border)
        self.assertIsNone(controller._pending_restore_style)

    def test_restore_preserves_visibility_acquired_after_hidden_install(self):
        original_style = 0x06000008 | _WS_CAPTION | _REQUIRED_NATIVE_STYLES
        api = _FakeNativeAPI(original_style)
        controller = WindowsCaptionController(object(), native_api=api)
        self.assertTrue(controller.install())

        # Startup installs custom chrome while the root is withdrawn.  The
        # visible bit is added later when the completed window is revealed.
        api.style |= _WS_VISIBLE
        self.assertTrue(controller.restore_native_caption())

        self.assertTrue(api.style & _WS_VISIBLE)
        self.assertTrue(api.style & _WS_CAPTION)
        self.assertEqual(controller.style, api.style)

    def test_controls_delegate_to_native_window_management(self):
        original_style = _WS_CAPTION | _REQUIRED_NATIVE_STYLES
        api = _FakeNativeAPI(original_style)
        window = _ActionWindow(api)
        controller = WindowsCaptionController(window, native_api=api)
        self.assertTrue(controller.install())

        controller.begin_drag()
        controller.minimize()
        controller.toggle_maximize()
        self.assertEqual(api.fit_count, 1)
        controller.toggle_maximize()
        controller.restore()
        controller.close()

        self.assertEqual(api.drag_count, 1)
        self.assertEqual(window.iconify_count, 1)
        self.assertEqual(window.states, ["zoomed", "normal"])
        self.assertEqual(window.deiconify_count, 1)
        self.assertEqual(api.close_count, 1)

    def test_client_edge_resize_delegates_to_native_window_api(self):
        api = _FakeNativeAPI(_WS_CAPTION | _REQUIRED_NATIVE_STYLES)
        controller = WindowsCaptionController(object(), native_api=api)

        self.assertFalse(controller.resize_window(10, 20, 640, 480))
        self.assertTrue(controller.install())
        self.assertTrue(controller.resize_window(10, 20, 640, 480))
        self.assertEqual(api.resize_calls, [(101, 10, 20, 640, 480)])

    def test_external_maximize_can_be_constrained_to_monitor_work_area(self):
        api = _FakeNativeAPI(_WS_CAPTION | _REQUIRED_NATIVE_STYLES)
        controller = WindowsCaptionController(object(), native_api=api)
        self.assertTrue(controller.install())

        self.assertFalse(controller.correct_maximized_bounds())
        api.zoomed = True
        self.assertTrue(controller.correct_maximized_bounds())
        self.assertEqual(api.fit_count, 1)

    def test_normal_state_transition_suppresses_maximized_bound_refits(self):
        api = _FakeNativeAPI(_WS_CAPTION | _REQUIRED_NATIVE_STYLES)
        controller = WindowsCaptionController(object(), native_api=api)
        self.assertTrue(controller.install())
        api.zoomed = True
        controller._normal_state_transition = True

        self.assertFalse(controller.correct_maximized_bounds())
        self.assertEqual(api.fit_count, 0)


if __name__ == "__main__":
    unittest.main()
