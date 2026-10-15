"""Profile Auto-detect ROIs 1-20 on deterministic synthetic OCT data.

Run: python release_tools/profile_auto_rois.py [optional-flat-image.hdr]
No output images or workbooks are written.
"""

import cProfile
from pathlib import Path
import pstats
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from aidas.core.config import STEP4_AUTO_DETECTION_DEFAULTS
from aidas.steps.step4_analyze_isez import (
    MATLAB_ROI_LOW, Step4Frame, default_isez_rois, load_oct_volume,
)


def fixture():
    rows = np.arange(1, 769, dtype=np.float64) - (MATLAB_ROI_LOW - 1)
    profile = np.full(rows.size, 12.0)
    bell = (rows >= 76) & (rows <= 100)
    profile[bell] = 25.0 - 0.12 * (rows[bell] - 88.0) ** 2
    rng = np.random.default_rng(41)
    image = 3 * profile[:, None] + rng.normal(0, 0.4, size=(768, 3000))
    return image


def main():
    image = load_oct_volume(sys.argv[1])[0] if len(sys.argv) > 1 else fixture()
    rois = default_isez_rois()
    for label in ("First run", "Repeated run"):
        profiler = cProfile.Profile()
        started = time.perf_counter()
        result = profiler.runcall(Step4Frame._compute_auto_detection_run,
                                 image, rois, set(), dict(STEP4_AUTO_DETECTION_DEFAULTS))
        print(f"{label}: {time.perf_counter() - started:.3f}s; "
              f"accepted {len(result.completed)}; review {result.review_count}", flush=True)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(16)


if __name__ == "__main__":
    main()
