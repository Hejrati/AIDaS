"""Profile the diagnostic result viewer without modifying study data.

Run with the app's Python: python release_tools/profile_result_grid.py
Use --folder PATH to read existing R PNGs, or --groups N for a larger batch.
The window is transparent and placed outside the visible desktop.
"""

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import customtkinter as ctk
from PIL import Image, ImageDraw

from aidas.steps import step3_flatten as module


def wait_for_images(root, canvas):
    """Include worker completion when comparing full image readiness."""
    deadline = time.monotonic() + 15
    while hasattr(canvas, "_render_future"):
        root.update()
        if canvas._visible_pages and all(canvas._pages[index]["ready"] for index in canvas._visible_pages):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Result previews did not finish loading")
        time.sleep(0.002)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--revision", help="Compare a committed Step 3 viewer, e.g. HEAD")
    args = parser.parse_args()
    frame_type = module.Step3Frame
    if args.revision:
        source_code = subprocess.check_output(
            ["git", "show", f"{args.revision}:aidas/steps/step3_flatten.py"], text=True,
        )
        namespace = dict(vars(module))
        exec(compile(source_code, module.__file__, "exec"), namespace)
        frame_type = namespace["Step3Frame"]
    root = ctk.CTk()
    root.withdraw()
    root.attributes("-alpha", 0)
    root.geometry("1400x900+10000+10000")
    frame = frame_type(root)
    frame.pack(fill="both", expand=True)
    root.deiconify()
    root.update()
    load_png = frame._load_result_png_from_folder
    source = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((90, 50, 1530, 820), outline="black")
    draw.line([(x, 400 + (x % 257)) for x in range(90, 1530)], fill="blue")

    def load(_folder, filename):
        return load_png(args.folder, filename) if args.folder else source.copy()

    profiler = cProfile.Profile()
    try:
        with patch.object(frame, "_load_result_png_from_folder", side_effect=load) as reads, \
                patch.object(module.ImageTk, "PhotoImage", wraps=module.ImageTk.PhotoImage) as photos:
            profiler.enable()
            started = time.perf_counter()
            frame._render_result_grid([Path(f"study-{i}") for i in range(args.groups)])
            root.update()
            interactive = time.perf_counter() - started
            wait_for_images(root, frame.canvas)
            opened = time.perf_counter() - started
            opening_reads, opening_photos = reads.call_count, photos.call_count
            canvas = frame.canvas
            started = time.perf_counter()
            for _ in range(30):
                canvas.yview_scroll(1, "units")
                root.update()
            scrolling = time.perf_counter() - started
            wait_for_images(root, canvas)
            scroll_reads, scroll_photos = reads.call_count, photos.call_count
            # Stay within the pages already visited to isolate native painting
            # from loading results for the first time.
            started = time.perf_counter()
            for index in range(30):
                canvas.yview_scroll(-1 if index % 2 == 0 else 1, "units")
                root.update()
            cached_scrolling = time.perf_counter() - started
            cached_photos = photos.call_count - scroll_photos
            profiler.disable()
            print(json.dumps({
                "groups": args.groups,
                "interactive_seconds": round(interactive, 4),
                "open_seconds": round(opened, 4),
                "opening_png_reads": opening_reads,
                "opening_photos": opening_photos,
                "30_scrolls_seconds": round(scrolling, 4),
                "scroll_png_reads": scroll_reads - opening_reads,
                "scroll_photos": scroll_photos - opening_photos,
                "30_cached_scrolls_seconds": round(cached_scrolling, 4),
                "cached_scroll_photos": cached_photos,
            }, indent=2))
            if args.profile:
                pstats.Stats(profiler).sort_stats("cumulative").print_stats(15)
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
