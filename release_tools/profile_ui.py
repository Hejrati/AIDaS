"""Profile source startup and empty workflow navigation without user preferences.

Run with the app's Python: python release_tools/profile_ui.py [--classic]
Timings include cProfile overhead and Tk event processing, but exclude app-module
imports. No OCT data, network update checks, or processing jobs are started.
"""

import cProfile
import json
from pathlib import Path
import pstats
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aidas.core.config import Config
from aidas.app import AIDaSApp


def main():
    with tempfile.TemporaryDirectory(prefix="aidas-ui-profile-") as directory:
        Config.CONFIG_DIR = Path(directory)
        Config.CONFIG_FILE = Config.CONFIG_DIR / "preferences.json"
        Config.CONFIG_FILE.write_text(json.dumps({
            "interface_mode": "Classic" if "--classic" in sys.argv else "Modern",
            "check_for_updates": False,
        }), encoding="utf-8")
        app = None
        try:
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
