import time


def settle_roi_tab(frame, *, timeout=15):
    """Process Tk events until the currently requested ROI tab is loaded."""
    deadline = time.monotonic() + timeout
    frame.update()
    while frame._batch_roi_selection is not None:
        if time.monotonic() >= deadline:
            raise AssertionError("The selected Step 4 tab did not finish opening.")
        time.sleep(0.002)
        frame.update()
