"""Measure Step 4 first-file preparation and longest Tk heartbeat pause."""
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import customtkinter as ctk
import numpy as np
from aidas.steps.step4_analyze_isez import Step4Frame
from aidas.utils.io_utils import write_analyze


def main():
    root = ctk.CTk()
    root.attributes('-alpha', 0)
    root.geometry('1150x720+10000+10000')
    frame = Step4Frame(root)
    frame.pack(fill='both', expand=True)
    root.update()
    try:
        with tempfile.TemporaryDirectory(prefix='aidas-step4-open-') as directory:
            path = Path(directory) / '_flat_LIGHT'
            image = np.repeat(np.linspace(20, 80, 450, dtype=np.int16)[:, None], 3000, axis=1)
            write_analyze(str(path), image[None, ...])
            gaps = []
            started = last = time.perf_counter()
            def heartbeat():
                nonlocal last
                now = time.perf_counter()
                gaps.append(now - last)
                last = now
                if frame._batch_roi_open_after_id is not None:
                    root.after(10, heartbeat)
                else:
                    root.quit()
            frame._open_batch_roi_tabs([path.with_suffix('.hdr')], show_progress=True)
            root.after(10, heartbeat)
            timeout = root.after(30000, root.quit)
            root.mainloop()
            root.after_cancel(timeout)
            assert frame._batch_roi_open_after_id is None, 'Loading timed out'
            assert frame.image is not None and frame.canvas is not None
            print(f'First file and plot: {time.perf_counter() - started:.3f}s')
            print(f'Longest heartbeat pause: {max(gaps):.3f}s')
            print(f'UI heartbeats: {len(gaps)}')
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
