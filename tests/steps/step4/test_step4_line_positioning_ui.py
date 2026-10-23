
import subprocess
import sys
import textwrap
import unittest

from tests.support.paths import PROJECT_ROOT


class Step4LinePositioningUITests(unittest.TestCase):
    def test_checkbox_stays_between_plots_and_shares_mode_across_cached_tabs(self):
        # CTk roots retain process-wide theme callbacks. Isolate this real
        # widget check from suites that previously created or destroyed roots.
        script = textwrap.dedent(
            """
            from pathlib import Path
            from tempfile import TemporaryDirectory
            import customtkinter as ctk
            import numpy as np
            from aidas.steps.step4_analyze_isez import Step4Frame
            from aidas.utils.io_utils import write_analyze

            ctk.set_widget_scaling(1.0)
            ctk.set_window_scaling(1.0)
            root = ctk.CTk()
            root.withdraw()
            # Mapping enables real Tk geometry calculations. Keep the window
            # invisible and offscreen throughout the check.
            root.attributes('-alpha', 0.0)
            root.geometry('1150x720+10000+10000')
            parent = ctk.CTkFrame(root, fg_color='transparent')
            parent.pack(fill='both', expand=True)
            frame = Step4Frame(parent)
            frame.pack(fill='both', expand=True)

            def check_gap():
                root.update_idletasks()
                widget = frame.canvas.get_tk_widget()
                check = widget._aidas_line_positioning_checkbox
                height = widget.winfo_height()
                width = widget.winfo_width()
                grid_bottom = (1.0 - frame.ax_roi_grid.get_position().y0) * height
                profile_top = (1.0 - frame.ax_profile.get_position().y1) * height
                assert check.winfo_ismapped(), 'Checkbox must be visible on the active plot.'
                assert check.winfo_y() + 1 >= grid_bottom, 'Checkbox overlaps ROI grid.'
                assert check.winfo_y() + check.winfo_height() <= profile_top + 1, 'Checkbox overlaps profile.'
                assert abs(check.winfo_x() - frame.ax_profile.get_position().x0 * width) <= 1
                return check

            try:
                with TemporaryDirectory() as temporary:
                    image = np.repeat(np.linspace(20, 80, 450, dtype=np.int16)[:, None], 3000, axis=1)
                    paths = []
                    for side in ('nasal', 'temporal'):
                        path = Path(temporary) / side / '_flat_LIGHT.hdr'
                        path.parent.mkdir()
                        write_analyze(str(path), image)
                        paths.append(path)
                    frame._open_batch_roi_tabs(paths)
                    root.deiconify()
                    first_check = check_gap()
                    assert not frame.disable_line_positioning_var.get()
                    assert first_check.get() == 0
                    first_canvas = frame.canvas
                    tabs = frame.batch_roi_notebook.tabs()

                    first_check.toggle()
                    assert frame.disable_line_positioning_var.get()
                    frame.batch_roi_notebook.select(tabs[1])
                    second_check = check_gap()
                    assert frame.canvas is not first_canvas
                    assert second_check.get() == 1, 'A new tab must share the session mode.'
                    root.geometry('980x450+10000+10000')
                    check_gap()

                    second_check.toggle()
                    assert not frame.disable_line_positioning_var.get()
                    assert first_check.get() == 0, 'A cached checkbox must track the same variable.'
                    frame.batch_roi_notebook.select(tabs[0])
                    cached_check = check_gap()
                    assert frame.canvas is first_canvas, 'The previously rendered canvas must be reused.'
                    assert cached_check is first_check
                    assert cached_check.get() == 0
            finally:
                root.destroy()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
