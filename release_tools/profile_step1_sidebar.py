"""Profile Step 1 scanning, queue navigation, and an optional SDB preview.

Run: python release_tools/profile_step1_sidebar.py [path/to/image.sdb]
The optional preview keeps bright pixels unchanged and never saves an image.
"""

import cProfile
from pathlib import Path
import pstats
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import customtkinter as ctk
from aidas.steps.step1_resize_raw import Step1Frame
from aidas.utils.filesystem import find_sdb_directories


def profile(label, callback):
    print(label, flush=True)
    profiler = cProfile.Profile()
    profiler.runcall(callback)
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(14)


def main():
    profile("Desktop scan", lambda: find_sdb_directories(Path.home() / "Desktop"))
    frame = Step1Frame.__new__(Step1Frame)
    frame._sdb_directories = [str(Path.cwd() / str(i)) for i in range(100)]
    frame._sdb_directory_files = {
        folder: [str(Path(folder) / f"{i}.sdb") for i in range(100)]
        for folder in frame._sdb_directories
    }
    frame._cropped_sdb_files = set()
    frame._saved_output_directories = set()
    frame.current_file = frame._sdb_directory_files[frame._sdb_directories[50]][50]
    profile("Navigation controls, 10000 queued images", frame._update_queue_controls)
    root = ctk.CTk()
    root.withdraw()
    try:
        with patch.object(Step1Frame, "refresh_sdb_list"):
            frame = Step1Frame(root)
        profile("Sidebar refresh dispatch (scan continues in background)", frame.refresh_sdb_list)
        frame._cancel_sdb_scan()
        if len(sys.argv) <= 1:
            return
        with patch.object(frame, "_ask_bright_pixel_action", return_value=False):
            profile("First preview", lambda: frame._open_raw(path=sys.argv[1], show_saved=False))
            profile("Repeated preview", lambda: frame._open_raw(path=sys.argv[1], show_saved=False))
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
