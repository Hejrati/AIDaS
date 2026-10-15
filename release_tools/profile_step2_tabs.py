"""Profile Step 2 with four dense, synthetic segmented-image tabs."""
import cProfile
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import customtkinter as ctk
import numpy as np
from aidas.steps.step2_annotate import Step2Frame


def main():
    root = ctk.CTk()
    root.attributes('-alpha', 0)
    root.geometry('1400x900+10000+10000')
    with patch.object(Step2Frame, '_start_ai_device_probe'):
        frame = Step2Frame(root)
    frame.pack(fill='both', expand=True)
    root.update()
    image = np.random.default_rng(42).integers(0, 65535, (256, 3000), dtype=np.uint16)
    rows = np.array([60 + i * 20 + 10 * np.sin(np.arange(3000) / 200) for i in range(6)])
    try:
        with patch.object(frame, '_read_image_for_annotation', return_value=(image, None, False)):
            frame._open_aidas_batch_results_viewer([
                dict(input=f'study-{i}.hdr', boundaries=rows, fovea_x=1500) for i in range(4)])
        root.update()
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        for _ in range(3):
            for tab in frame.batch_results_notebook.tabs():
                frame.batch_results_notebook.select(tab)
                root.update()
        profiler.disable()
        print(f'12 tab switches: {time.perf_counter() - start:.3f}s')
        pstats.Stats(profiler).sort_stats('cumulative').print_stats(20)
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
