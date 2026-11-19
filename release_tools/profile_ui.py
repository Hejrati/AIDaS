"""Profile startup and navigation, or use --window for Windows interaction timings.

Window scenarios use synthetic OCT data, temporary preferences and separate
CPU traces. Pass --classic to exercise the Classic interface.
"""

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aidas.core.config import Config
from aidas.app import AIDaSApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classic", action="store_true")
    parser.add_argument("--window", action="store_true", help="Measure Windows window interactions")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--trace", action="store_true", help="Collect a separate window CPU profile")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.window and sys.platform != "win32":
        parser.error("--window requires a Windows desktop")
    if args.trace and not args.window:
        parser.error("--trace requires --window")
    with tempfile.TemporaryDirectory(prefix="aidas-ui-profile-") as directory, patch(
        "aidas.services.update_ui.UpdateController.check_automatically"
    ):
        Config.CONFIG_DIR = Path(directory)
        Config.CONFIG_FILE = Config.CONFIG_DIR / "preferences.json"
        Config.CONFIG_FILE.write_text(json.dumps({
            "interface_mode": "Classic" if args.classic else "Modern",
        }), encoding="utf-8")
        app = None
        try:
            if args.window:
                from release_tools.profile_window import measure_windows
                app = AIDaSApp()
                measure_windows(app, args.runs, args.trace)
                return
            profile = cProfile.Profile()
            started = time.perf_counter()
            profile.enable()
            app = AIDaSApp()
            app.update()
            profile.disable()
            print(f"Startup: {time.perf_counter() - started:.3f}s")
            pstats.Stats(profile).sort_stats("cumulative").print_stats(18)
            for label in ("First navigation", "Retained navigation"):
                profile = cProfile.Profile()
                started = time.perf_counter()
                profile.enable()
                for _ in range(3):
                    for index in range(5):
                        app.notebook.select(index)
                        app.update()
                profile.disable()
                print(f"{label}, 15 selections: {time.perf_counter() - started:.3f}s")
                pstats.Stats(profile).sort_stats("cumulative").print_stats(18)
            for name, show, attribute in (
                ("About", app._show_about, "_about_dialog"),
                ("Settings", app._show_settings, "_settings_dialog"),
            ):
                started = time.perf_counter()
                show()
                app.update()
                print(f"Open {name}: {time.perf_counter() - started:.3f}s")
                dialog = getattr(app, attribute)
                dialog._close()
                app.update()
        finally:
            if app is not None:
                app.destroy()


if __name__ == "__main__":
    main()
