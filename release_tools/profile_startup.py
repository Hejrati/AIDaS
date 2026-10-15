"""Profile launch through the first usable window without user data or updates.

Run: python release_tools/profile_startup.py [--classic]
Uses temporary preferences and transparent windows. Includes application module
imports; timings are source-run measurements with cProfile overhead.
"""
import cProfile
import json
from pathlib import Path
import pstats
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import customtkinter as ctk
    from aidas.core.config import Config

    root_init = ctk.CTk.__init__
    window_init = ctk.CTkToplevel.__init__

    def invisible_root(self, *args, **kwargs):
        root_init(self, *args, **kwargs)
        self.attributes('-alpha', 0)

    def invisible_window(self, *args, **kwargs):
        window_init(self, *args, **kwargs)
        self.attributes('-alpha', 0)

    with tempfile.TemporaryDirectory(prefix='aidas-startup-profile-') as directory:
        Config.CONFIG_DIR = Path(directory)
        Config.CONFIG_FILE = Path(directory) / 'preferences.json'
        Config.CONFIG_FILE.write_text(json.dumps({
            'interface_mode': 'Classic' if '--classic' in sys.argv else 'Modern',
            'check_for_updates': False,
        }), encoding='utf-8')
        profiler = cProfile.Profile()
        stages = []
        app = None
        started = time.perf_counter()
        profiler.enable()
        from aidas.app import AIDaSApp
        imported = time.perf_counter()
        original_progress = AIDaSApp._set_splash_progress

        def progress(self, value, message):
            stages.append((time.perf_counter() - started, message))
            original_progress(self, value, message)

        try:
            with patch.object(ctk.CTk, '__init__', invisible_root), patch.object(
                    ctk.CTkToplevel, '__init__', invisible_window), patch.object(
                    AIDaSApp, '_set_splash_progress', progress):
                app = AIDaSApp()
                # Honor the splash's existing minimum duration as real launch does.
                if not app._startup_finished:
                    app.wait_variable_flag = ctk.BooleanVar(master=app, value=False)
                    def wait_for_startup():
                        if app._startup_finished:
                            app.wait_variable_flag.set(True)
                        else:
                            app.after(10, wait_for_startup)
                    app.after(10, wait_for_startup)
                    app.wait_variable(app.wait_variable_flag)
                app.update()
            profiler.disable()
            print(f'App imports: {imported - started:.3f}s')
            print(f'Imports through usable window: {time.perf_counter() - started:.3f}s')
            print(f'Deferred analysis libraries: {[name for name in ("scipy", "matplotlib", "pandas", "pyreadr") if name not in sys.modules]}')
            print('Startup stages (seconds since import began):')
            for elapsed, message in stages:
                print(f'  {elapsed:.3f}  {message}')
            pstats.Stats(profiler).sort_stats('cumulative').print_stats(20)
        finally:
            profiler.disable()
            if app is not None:
                app.destroy()


if __name__ == '__main__':
    main()
