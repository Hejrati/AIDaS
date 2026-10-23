from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.canvas.image_canvas import ImageCanvas


class _CanvasBindStub:
    def __init__(self):
        self.bindings = {}

    def bind(self, sequence, callback):
        self.bindings[sequence] = callback


class ImageCanvasZoomGestureTests(unittest.TestCase):
    @staticmethod
    def _canvas(windowing_system="win32"):
        canvas = ImageCanvas.__new__(ImageCanvas)
        canvas._zoom = 1.0
        canvas._pending_zoom = None
        canvas._windowing_system = windowing_system
        canvas._queue_zoom = mock.Mock()
        return canvas

    def test_touchpad_pinch_sequences_are_bound_to_zoom(self):
        canvas = ImageCanvas.__new__(ImageCanvas)
        canvas.canvas = _CanvasBindStub()

        canvas._bind_events()

        for sequence in (
            "<Control-MouseWheel>",
            "<Control-Button-4>",
            "<Control-Button-5>",
        ):
            callback = canvas.canvas.bindings[sequence]
            self.assertIs(callback.__func__, ImageCanvas._on_wheel)

    def test_windows_precision_touchpad_delta_uses_fractional_zoom(self):
        canvas = self._canvas()
        event = SimpleNamespace(num=None, delta=30)

        result = canvas._on_wheel(event)

        self.assertEqual(result, "break")
        queued_zoom = canvas._queue_zoom.call_args.args[0]
        self.assertAlmostEqual(queued_zoom, 1.25 ** 0.25)

    def test_touchpad_zoom_directions_are_reciprocal(self):
        zoom_in = ImageCanvas._wheel_zoom_steps(
            SimpleNamespace(num=None, delta=120), "win32"
        )
        zoom_out = ImageCanvas._wheel_zoom_steps(
            SimpleNamespace(num=None, delta=-120), "win32"
        )

        self.assertEqual(zoom_in, 1.0)
        self.assertEqual(zoom_out, -1.0)
        self.assertAlmostEqual(
            ImageCanvas.ZOOM_STEP_FACTOR ** zoom_in
            * ImageCanvas.ZOOM_STEP_FACTOR ** zoom_out,
            1.0,
        )

    def test_legacy_linux_wheel_buttons_still_zoom_both_directions(self):
        self.assertEqual(
            ImageCanvas._wheel_zoom_steps(SimpleNamespace(num=4), "x11"),
            1.0,
        )
        self.assertEqual(
            ImageCanvas._wheel_zoom_steps(SimpleNamespace(num=5), "x11"),
            -1.0,
        )

    def test_zero_delta_is_consumed_without_queuing_a_zoom(self):
        canvas = self._canvas()

        result = canvas._on_wheel(SimpleNamespace(num=None, delta=0))

        self.assertEqual(result, "break")
        canvas._queue_zoom.assert_not_called()

    def test_rapid_gesture_continues_from_the_pending_zoom(self):
        canvas = self._canvas()
        canvas._pending_zoom = 2.0

        canvas._on_wheel(SimpleNamespace(num=None, delta=-120))

        canvas._queue_zoom.assert_called_once_with(2.0 / 1.25)


if __name__ == "__main__":
    unittest.main()
