"""Measure Step 2 tab readiness with dense, synthetic segmented images."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import customtkinter as ctk
import numpy as np
from aidas.steps.step2_annotate import Step2Frame
from aidas.ui.theme import set_interface_mode


def measure(frame, root, results, switches):
    def select(tab):
        start = time.perf_counter()
        frame.batch_results_notebook.select(tab)
        root.update()
        assert frame._active_batch_result_tab == tab
        assert frame.image_canvas._img_id is not None
        return (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    frame._open_aidas_batch_results_viewer(results)
    root.update()
    assert frame.image_canvas._img_id is not None
    first_visible = (time.perf_counter() - start) * 1000
    tabs = frame.batch_results_notebook.tabs()
    cold = [select(tab) for tab in tabs[1:]]
    warm = [select(tabs[index % len(tabs)]) for index in range(switches)]
    return first_visible, cold, warm


def percentiles(samples):
    return {
        "count": len(samples),
        "p50_ms": round(float(np.percentile(samples, 50)), 1),
        "p95_ms": round(float(np.percentile(samples, 95)), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classic", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--switches", type=int, default=32)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.switches < 1:
        parser.error("--runs and --switches must be positive")

    root = ctk.CTk()
    root.attributes('-alpha', 0)
    root.geometry('1400x900+10000+10000')
    mode = "Classic" if args.classic else "Modern"
    set_interface_mode(mode)
    with patch.object(Step2Frame, '_start_ai_device_probe'):
        frame = Step2Frame(root)
    frame.pack(fill='both', expand=True)
    root.update()
    image = np.random.default_rng(42).integers(0, 65535, (256, 3000), dtype=np.uint16)
    rows = np.array([60 + i * 20 + 10 * np.sin(np.arange(3000) / 200) for i in range(6)])
    traces, _ = frame._boundary_traces_from_rows(rows, image.shape)
    results = [
        dict(input=f'study-{index}.hdr', image=image, traces=traces, fovea_x=1500)
        for index in range(4)
    ]
    try:
        opening, cold, warm = [], [], []
        with patch.object(frame, '_read_image_for_annotation', return_value=(image, None, False)):
            for _ in range(args.runs):
                first_visible, first_visits, revisits = measure(frame, root, results, args.switches)
                opening.append(first_visible)
                cold.extend(first_visits)
                warm.extend(revisits)
            print(json.dumps({
                "mode": mode,
                "first_visible": percentiles(opening),
                "first_visits": percentiles(cold),
                "revisits": percentiles(warm),
            }), flush=True)
            if args.trace:
                profiler = cProfile.Profile()
                profiler.runcall(measure, frame, root, results, args.switches)
                pstats.Stats(profiler).sort_stats('cumulative').print_stats(20)
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
