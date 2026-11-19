"""Move, resize, maximize, minimize and restore the packaged Windows UI."""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def wait_for(predicate, process):
    deadline = time.monotonic() + 45
    while process.poll() is None and time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.04)
    raise RuntimeError(f"Packaged window did not reach the expected state (exit={process.poll()})")


def exercise(executable, mode):
    user32 = ctypes.windll.user32
    visit_window = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    with tempfile.TemporaryDirectory(prefix="aidas-window-smoke-") as directory:
        prefs = Path(directory, ".aidas", "preferences.json")
        prefs.parent.mkdir()
        prefs.write_text(json.dumps({"interface_mode": mode}), encoding="utf-8")
        environment = dict(os.environ, USERPROFILE=directory)
        process = subprocess.Popen([executable], cwd=directory, env=environment)

        def find_window():
            matches = []

            @visit_window
            def inspect(handle, _):
                window = wintypes.HWND(handle)
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
                if pid.value == process.pid and user32.IsWindowVisible(window):
                    title = ctypes.create_unicode_buffer(256)
                    user32.GetWindowTextW(window, title, len(title))
                    if title.value == "AIDaS — Retinal Image Processing":
                        matches.append(window)
                        return False
                return True

            user32.EnumWindows(inspect, 0)
            return next(iter(matches), None)

        try:
            window = wait_for(find_window, process)
            caption_mask = 0x00C00000  # WS_CAPTION
            wait_for(lambda: bool(user32.GetWindowLongW(window, -16) & caption_mask)
                     == (mode == "Classic"), process)

            def bounds():
                rect = wintypes.RECT()
                if not user32.GetWindowRect(window, ctypes.byref(rect)):
                    raise ctypes.WinError()
                return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top

            # The first visible HWND can still be receiving startup centering
            # and caption corrections. Start the test from settled geometry.
            last = None
            since = time.monotonic()

            def settled_bounds():
                nonlocal last, since
                current = bounds()
                if current != last:
                    last, since = current, time.monotonic()
                if current[2] > 600 and current[3] > 400 and time.monotonic() - since > 1:
                    return current

            original = wait_for(settled_bounds, process)
            moved = (original[0] + 15, original[1] + 10, *original[2:])
            resized = (*moved[:2], moved[2] + 25, moved[3] + 20)
            for target in (moved, resized):
                if not user32.MoveWindow(window, *target, True):
                    raise ctypes.WinError()
                try:
                    wait_for(lambda: bounds() == target, process)
                except RuntimeError as exc:
                    raise RuntimeError(f"{mode} requested {target}, observed {bounds()}") from exc
            for _ in range(2):
                for command, ready in (
                    (3, lambda: user32.IsZoomed(window)),
                    (9, lambda: bounds() == resized and not user32.IsZoomed(window)),
                    (6, lambda: user32.IsIconic(window)),
                    (9, lambda: bounds() == resized and not user32.IsIconic(window)),
                ):
                    user32.ShowWindowAsync(window, command)
                    wait_for(ready, process)
            user32.PostMessageW(window, 0x0010, 0, 0)  # WM_CLOSE
            if process.wait(timeout=20):
                raise RuntimeError("Packaged window exited with an error")
            print(f"PACKAGED_WINDOW_OK {mode}", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    for mode in ("Modern", "Classic"):
        exercise(args.executable.resolve(), mode)
