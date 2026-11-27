"""Measure the selected view becoming interactive on a mapped Tk desktop."""

from collections import defaultdict
import gc
import json
import time
from unittest.mock import patch

import numpy as np
from PIL import Image

from aidas.core.config import STEP4_AUTO_DETECTION_DEFAULTS
from aidas.steps.step2_annotate import Step2Frame
from aidas.steps.step3_flatten import Step3Frame
from aidas.steps.step4_analyze_isez import Step4Frame, default_isez_rois
from aidas.utils.io_utils import write_analyze
from release_tools.profile_auto_rois import fixture
from release_tools.profile_window import annotation_ready, heartbeat_samples, results_ready, settle


def prepare_fixture(root, image):
    preview = Image.fromarray(np.uint8(np.clip(image, 0, 255))).convert("RGB")
    annotation = []
    flattened = []
    folders = []
    curve = 10 * np.sin(np.arange(image.shape[1]) / 200)
    rows = np.array([60 + index * 20 + curve for index in range(6)])
    for index in range(2):
        folder = root / f"Subject {index + 1}" / "nasal"
        folder.mkdir(parents=True)
        folders.append(folder)
        write_analyze(folder / "light", image)
        annotation.append(dict(input=str(folder / "light.hdr"), boundaries=rows, fovea_x=1500))
        write_analyze(folder / "_flat_LIGHT", image)
        flattened.append(folder / "_flat_LIGHT.hdr")
        for _, filename in Step3Frame.RESULT_IMAGE_ROWS:
            preview.save(folder / filename)
        # The inventory labels this file by name and never reads its contents.
        (folder / "rr_MCPAR.xlsx").touch()
    return annotation, flattened, folders


def measure_selection(app, select, ready):
    with heartbeat_samples(app) as delays:
        started = time.perf_counter()
        select()
        settle(app, ready)
        elapsed = (time.perf_counter() - started) * 1000
        settle(app, lambda: bool(delays))
    return elapsed, max(delays)


def run_cycle(app, root, image, prepared, detected, samples):
    annotation, flattened, folders = prepared
    app.geometry("1200x800+40+50")
    settle(app, lambda: app._startup_finished and app.winfo_viewable())
    source = root / "scan.sdb"
    first_annotation = annotation[0]["input"]
    first_flattened = flattened[0]
    step1, step2, step3, step4, step5 = app.step1, app.step2, app.step3, app.step4, app.step5

    def crop_ready():
        return step1.current_file == source and annotation_ready(step1.image_canvas)

    def annotation_tab_ready(path):
        return step2.current_file == path and annotation_ready(step2.image_canvas)

    def diagnostics_ready():
        grid = step3.batch_results_grid
        return grid.winfo_viewable() and results_ready(grid)

    def roi_tab_ready(path):
        canvas = step4.canvas
        notebook = step4.batch_roi_notebook
        return (step4.current_path == path and step4.image is not None
                and notebook is not None and step4._active_batch_roi_tab == notebook.select()
                and canvas is not None
                and canvas.get_tk_widget().winfo_viewable()
                and canvas._idle_draw_id is None)

    def inventory_ready():
        tree = step5.directory_view
        inventory = tree._inventory
        return (tree.canvas.winfo_viewable() and inventory is not None
                and inventory.root.path == root and inventory.workbooks == len(folders))

    workflow = (
        ("Step 1 -> Step 2", 1, lambda: step2._open_aidas_batch_results_viewer(annotation),
         lambda: annotation_tab_ready(first_annotation)),
        ("Step 2 -> Step 3", 2, lambda: step3._open_batch_r_result_tabs(folders), diagnostics_ready),
        ("Step 3 -> Step 4", 3, lambda: step4._open_batch_roi_tabs(flattened, show_progress=True),
         lambda: roi_tab_ready(first_flattened)),
        ("Step 4 -> Step 5", 4, lambda: step5.set_input_folder(root), inventory_ready),
        ("Step 5 -> Step 1", 0, lambda: step1._load_image(image, source, check_bright_pixels=False), crop_ready),
    )

    def workflow_selection(transition, index, ready, phase):
        samples[(transition, phase)].append(measure_selection(
            app, lambda: app._select_workflow_step(index),
            lambda: app._last_workflow_tab_index == index and ready(),
        ))

    def result_selection(label, notebook, index, ready, phase):
        samples[(label, phase)].append(measure_selection(
            app, lambda: notebook.select(index), ready,
        ))

    for transition, index, prepare, ready in workflow:
        prepare()
        workflow_selection(transition, index, ready, "cold")
        if index == 1:
            notebook = step2.batch_results_notebook
            next_ready = lambda: annotation_tab_ready(annotation[1]["input"])
            result_selection("Step 2 result -> next", notebook, 1, next_ready, "cold")
            result_selection("Step 2 result -> previous", notebook, 0, ready, "warm")
            result_selection("Step 2 result -> next", notebook, 1, next_ready, "warm")
            notebook.select(0)
            settle(app, ready)
        elif index == 3:
            notebook = step4.batch_roi_notebook
            settle(app, lambda: step4._batch_roi_open_after_id is None)
            step4._apply_auto_detection_run(detected)
            settle(app, ready)
            next_ready = lambda: roi_tab_ready(flattened[1])
            result_selection("Step 4 result -> next", notebook, 1, next_ready, "cold")
            step4._apply_auto_detection_run(detected)
            settle(app, next_ready)
            result_selection("Step 4 result -> previous", notebook, 0, ready, "warm")
            result_selection("Step 4 result -> next", notebook, 1, next_ready, "warm")
            notebook.select(0)
            settle(app, ready)

    for transition, index, _, ready in workflow:
        workflow_selection(transition, index, ready, "warm")


def measure_tabs(app_factory, directory, runs):
    root = (directory / "OCT fixtures").resolve()
    image = np.rint(fixture()).astype(np.int16)
    prepared = prepare_fixture(root, image)
    detected = Step4Frame._compute_auto_detection_run(
        image, default_isez_rois(), set(), dict(STEP4_AUTO_DETECTION_DEFAULTS),
    )
    samples = defaultdict(list)
    with (
        patch("aidas.app.record_event"),
        patch.object(Step2Frame, "_start_ai_device_probe"),
        patch.object(Step4Frame, "_open_automatic_profile_review"),
    ):
        for run in range(runs):
            app = app_factory()
            try:
                if run == 0:
                    print(json.dumps(dict(
                        mode=app.interface_mode,
                        screen=(app.winfo_screenwidth(), app.winfo_screenheight()),
                        dpi=app.winfo_fpixels("1i"),
                    )), flush=True)
                run_cycle(app, root, image, prepared, detected, samples)
            finally:
                app.destroy()
                gc.collect()
    for (transition, phase), timings in samples.items():
        latencies, delays = zip(*timings)
        print(json.dumps(dict(
            transition=transition, phase=phase, samples=len(timings),
            p50_ms=round(float(np.percentile(latencies, 50)), 1),
            p95_ms=round(float(np.percentile(latencies, 95)), 1),
            heartbeat_p95_ms=round(float(np.percentile(delays, 95)), 1),
        )), flush=True)
