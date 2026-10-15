"""Profile mapped Step 4 tabs and ROI grids without reading or saving studies.

Run: python release_tools/profile_step4_tabs.py [number-of-tabs]
Uses deterministic synthetic data with 20 accepted ROIs in each tab.
"""

import cProfile
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import customtkinter as ctk
from matplotlib.backend_bases import MouseEvent
from aidas.core.config import STEP4_AUTO_DETECTION_DEFAULTS
from aidas.steps import step4_analyze_isez as module
from release_tools.profile_auto_rois import fixture


def measure(label, callback):
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.runcall(callback)
    print(f"{label}: {time.perf_counter() - started:.3f}s", flush=True)
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(16)


def main():
    image = fixture()
    result = module.Step4Frame._compute_auto_detection_run(
        image, module.default_isez_rois(), set(), dict(STEP4_AUTO_DETECTION_DEFAULTS))
    root = ctk.CTk()
    root.withdraw()
    root.attributes("-alpha", 0)
    root.geometry("1400x900+10000+10000")
    frame = module.Step4Frame(root)
    frame.pack(fill="both", expand=True)
    root.deiconify()
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    try:
        with patch.object(module, "load_oct_volume", return_value=image[None, ...]):
            frame._open_batch_roi_tabs([Path(f"study-{i}") / "_flat_LIGHT.hdr" for i in range(count)])
            notebook = frame.batch_roi_notebook
            for tab in notebook.tabs():
                notebook.select(tab)
                root.update()
                frame._apply_auto_detection_run(result)
                root.update()

            def switch():
                for _ in range(3):
                    for tab in notebook.tabs():
                        notebook.select(tab)
                        root.update()

            def grid():
                for index in range(10):
                    frame.current_roi_idx = index
                    frame._load_current_roi_clicks()
                    frame._render_current_roi()
                    root.update()

            def hover():
                for index in range(21):
                    x, y = frame.ax_roi_grid.transData.transform((index % 7 + .5, 2 - index // 7 + .5))
                    frame._on_plot_motion(MouseEvent("motion_notify_event", frame.canvas, x, y))
                    root.update()

            measure(f"{count * 3} cached tab selections", switch)
            measure("10 ROI selections and grid redraws", grid)
            measure("Hover across 21 cells", hover)
    finally:
        frame._clear_batch_roi_session()
        root.destroy()


if __name__ == "__main__":
    main()
