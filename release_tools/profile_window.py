"""Window-interaction scenarios for profile_ui.py --window (Windows desktop)."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import cProfile
import json
from pathlib import Path
import pstats
from threading import Event
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from aidas.core.config import STEP4_AUTO_DETECTION_DEFAULTS
from aidas.steps.step4_analyze_isez import Step4Frame, default_isez_rois
from aidas.ui import rendering
from aidas.ui.title_bar import _WindowsAPI
from release_tools.profile_auto_rois import fixture


def settle(app, ready=lambda: True):
    deadline = time.monotonic() + 20
    while True:
        app.update()
        if ready():
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Window or visible content failed to settle")
        time.sleep(0.002)


def exercise(app, ready):
    """Use a real Tk timer to sample lateness while native state and layout change."""
    native = _WindowsAPI()
    handle = native.root_handle(app)
    insets = native.frame_insets(handle)
    if insets is None:
        raise OSError("Could not read the initial window bounds")
    left, top, right, bottom = insets
    width = app.winfo_width() + left + right
    height = app.winfo_height() + top + bottom

    def place(x, w, h):
        if not native.resize_window(handle, x, 50, w, h):
            raise OSError("Windows rejected a benchmark geometry request")
        settle(app)

    delays = []
    timer = None
    interval = 0.010
    due = time.perf_counter() + interval

    def heartbeat():
        nonlocal due, timer
        now = time.perf_counter()
        delays.append(max(0, now - due) * 1000)
        due = now + interval
        timer = app.after(10, heartbeat)

    timer = app.after(10, heartbeat)
    try:
        started = time.perf_counter()
        for i in range(12):
            place(40 + i * 3, width, height)
        move_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        for i in range(12):
            place(73, width + i * 8, height + i * 4)
        settle(app, ready)
        resize_ms = (time.perf_counter() - started) * 1000
        motion_delays = delays.copy()
        bar = app.window_title_bar
        for state in ("zoomed", "normal"):
            if bar is None:
                app.state(state)
            else:
                bar.controller.toggle_maximize()
            settle(app, lambda: app.state() == state and ready())
        app.iconify()
        settle(app, lambda: app.state() == "iconic")
        started = time.perf_counter()
        app.deiconify()
        settle(app, lambda: app.state() == "normal" and app.winfo_viewable() and ready())
        restored = (time.perf_counter() - started) * 1000
        samples = len(delays)
        settle(app, lambda: len(delays) > samples)
    finally:
        app.after_cancel(timer)
    return {"p95_ms": round(float(np.percentile(delays, 95)), 1),
            "motion_p95_ms": round(float(np.percentile(motion_delays, 95)), 1) if motion_delays else None,
            "restore_ms": round(restored, 1), "move_ms": round(move_ms, 1),
            "resize_ms": round(resize_ms, 1)}


def process_images(stop, image):
    count = 0
    while not stop.is_set() or count == 0:
        Step4Frame._compute_auto_detection_run(
            image, default_isez_rois(), set(), dict(STEP4_AUTO_DETECTION_DEFAULTS)
        )
        count += 1
    return count


def scenes(app, image):
    app.notebook.select(0)
    yield "empty", app.winfo_viewable

    app.notebook.select(1)
    settle(app)
    frame = app.step2
    curve = 10 * np.sin(np.arange(image.shape[1]) / 200)
    rows = [60 + i * 20 + curve for i in range(6)]
    with patch.object(frame, "_read_image_for_annotation", return_value=(image, None, False)):
        frame._open_aidas_batch_results_viewer([
            dict(input="study.hdr", boundaries=rows, fovea_x=1500)
        ])
    canvas = frame.image_canvas

    def annotation_ready():
        return (canvas.canvas.winfo_viewable() and canvas._img_id is not None
                and not canvas._canvas_size_changed())

    yield "annotated", annotation_ready

    app.notebook.select(2)
    settle(app)
    frame = app.step3
    preview = Image.fromarray(np.uint8(np.clip(image, 0, 255))).convert("RGB")
    with patch.object(frame, "_load_result_png_from_folder", side_effect=lambda *_: preview.copy()):
        frame._render_result_grid([Path(f"study-{i}") for i in range(3)])
        grid = frame.canvas

        def results_ready():
            return (grid._page_size == (grid.winfo_width(), grid.winfo_height())
                    and grid._visible_pages
                    and all(grid._pages[i]["ready"] and not grid._pages[i].get("error")
                            for i in grid._visible_pages))

        yield "results", results_ready


@contextmanager
def processing_load(image, enabled):
    if not enabled:
        yield
        return
    stop = Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = executor.submit(process_images, stop, image)
        try:
            yield
        finally:
            stop.set()
            job.result()


def sample(app, ready):
    app.geometry("1024x680+40+50")
    settle(app, ready)
    repaint = rendering._repaint
    paints = 0

    def counted(canvas):
        nonlocal paints
        paints += 1
        repaint(canvas)

    with patch.object(rendering, "_repaint", counted):
        metrics = exercise(app, ready)
    return dict(metrics, surface_paints=paints)


def profile_resize_grip(app):
    """Replay a burst on the actual Modern grip and verify the released size."""
    bar = app.window_title_bar
    if not bar.controller.resize_window(40, 50, 1100, 720):
        raise OSError("Failed to position the grip benchmark window")
    settle(app)
    width, height = app.winfo_width(), app.winfo_height()
    x = app.winfo_x() + width - 1
    y = app.winfo_y() + height - 1
    if not bar._start_client_resize(SimpleNamespace(widget=app, x_root=x, y_root=y)):
        raise RuntimeError("Failed to grab the southeast resize grip")
    with patch.object(bar.controller, "resize_window", wraps=bar.controller.resize_window) as resize:
        started = time.perf_counter()
        try:
            for delta in range(1, 25):
                bar._drag_client_resize(SimpleNamespace(x_root=x + delta, y_root=y + delta))
        finally:
            bar._end_client_resize()
        settle(app, lambda: (app.winfo_width(), app.winfo_height()) == (width + 24, height + 24))
        return {"resize_requests": resize.call_count,
                "drag_ms": round((time.perf_counter() - started) * 1000, 1)}


def measure_windows(app, runs, trace):
    settle(app, lambda: app._startup_finished and app.winfo_viewable())
    image = fixture()
    screen = app.winfo_screenwidth(), app.winfo_screenheight()
    print(json.dumps(dict(mode=app.interface_mode, screen=screen, dpi=app.winfo_fpixels("1i"))), flush=True)
    for scene, ready in scenes(app, image):
        settle(app, ready)
        for busy in (False, True):
            with processing_load(image, busy):
                for run in range(1, runs + 1):
                    metrics = sample(app, ready)
                    print(json.dumps(dict(scene=scene, busy=busy, run=run, **metrics)), flush=True)
        if trace:
            profile = cProfile.Profile()
            profile.runcall(exercise, app, ready)
            pstats.Stats(profile).sort_stats("cumulative").print_stats(25)
    if app.window_title_bar is not None:
        app.notebook.select(0)
        settle(app)
        for run in range(1, runs + 1):
            print(json.dumps(dict(scene="resize_grip", run=run, **profile_resize_grip(app))), flush=True)
