import unittest
from unittest import mock

import numpy as np

from aidas.steps.step4_analyze_isez import ISezROI, ISezResult, make_isez_plot_image


def _result() -> ISezResult:
    return ISezResult(
        roi=ISezROI("01", 1, 2),
        start=10,
        end=11,
        adjusted_start=10,
        adjusted_end=11,
        center=50.5,
        slope=0.0,
        max_index=1,
        min_intensity=0.0,
        max_intensity=1.0,
        normalized_x=np.array([49.0, 50.0]),
        normalized_y=np.array([0.0, 100.0]),
        baseline_x=np.array([49.0, 50.0]),
        baseline_y=np.array([0.0, 0.0]),
    )


class Step4StackImageAxesTests(unittest.TestCase):
    def test_saved_stack_frame_draws_numeric_x_and_y_tick_labels(self):
        with mock.patch("aidas.steps.step4_analyze_isez.ImageDraw.Draw") as draw_factory:
            make_isez_plot_image(_result())

        text_calls = draw_factory.return_value.text.call_args_list
        x_labels = [call.args[1] for call in text_calls if call.kwargs["anchor"] == "mt"]
        y_labels = [call.args[1] for call in text_calls if call.kwargs["anchor"] == "rm"]

        self.assertEqual(
            x_labels,
            ["10.5", "20.5", "30.5", "40.5", "50.5", "60.5", "70.5", "80.5", "90.5"],
        )
        self.assertEqual(y_labels, ["-20", "0", "20", "40", "60", "80", "100", "120"])

    def test_numeric_labels_render_in_the_saved_frame_margins(self):
        pixels = np.asarray(make_isez_plot_image(_result()).convert("L"))

        self.assertTrue(np.any(pixels[465:490, 100:900] < 255))
        self.assertTrue(np.any(pixels[25:470, 70:116] < 255))


if __name__ == "__main__":
    unittest.main()
