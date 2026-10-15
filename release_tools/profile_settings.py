"""Profile opening and scrolling the real Settings UI with stubbed R inventory."""
import cProfile
from pathlib import Path
import pstats
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import customtkinter as ctk
from aidas.app import SettingsDialog


def measure(label, callback):
    profiler = cProfile.Profile()
    start = time.perf_counter()
    result = profiler.runcall(callback)
    print(f'{label}: {time.perf_counter() - start:.3f}s', flush=True)
    pstats.Stats(profiler).sort_stats('cumulative').print_stats(18)
    return result


def main():
    root = ctk.CTk()
    root.attributes('-alpha', 0)
    root.geometry('1000x850+10000+10000')
    root.update()
    dialog = None
    try:
        def opening():
            with patch.object(SettingsDialog, '_schedule_modal_activation'):
                window = SettingsDialog(root, preferences={}, interface_mode='Modern',
                    appearance_mode='Dark', set_interface_command=lambda value: None,
                    set_appearance_command=lambda value: None, step1=SimpleNamespace(),
                    step3=SimpleNamespace(R_REQUIRED_VERSION='3.3.1',
                        available_r_scripts=lambda role: [], _selected_r_script_path=lambda role: None))
            window.attributes('-alpha', 0)
            root.update()
            return window
        dialog = measure('Open Settings', opening)

        def scrolling(handler=None):
            panel = dialog.settings_panel
            handler = handler or panel._mouse_wheel_all
            panel._parent_canvas.yview_moveto(0)
            root.update()
            for delta in [-120] * 10 + [120] * 10:
                for _ in range(5):
                    handler(SimpleNamespace(widget=panel, delta=delta))
                root.update()
        panel = dialog.settings_panel
        measure('100 upstream wheel events in bursts', lambda: scrolling(
            lambda event: ctk.CTkScrollableFrame._mouse_wheel_all(panel, event)))
        measure('100 optimized wheel events in bursts', scrolling)
    finally:
        if dialog is not None:
            dialog._close()
        root.destroy()


if __name__ == '__main__':
    main()
