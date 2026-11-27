"""Prepare a diagnostic preview off the Tk thread and display its latest size."""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from concurrent.futures import ThreadPoolExecutor
import tkinter as tk

from PIL import Image, ImageOps, ImageTk


def show_result_preview(
    parent, image, *, background, layout, on_click, on_image_layout, align_top,
    plot_bounds, plot_margins, plot_box_aspect,
):
    label = tk.Label(
        parent, bg=background, borderwidth=0, highlightthickness=0, padx=0, pady=0
    )
    label.pack(fill="both", expand=True)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aidas-preview")
    redraw_after_id = None
    poll_after_id = None
    pending = None
    label.preview_size = None

    def current_size():
        return label.winfo_width(), label.winfo_height()

    def redraw():
        nonlocal redraw_after_id, pending
        redraw_after_id = None
        if not label.winfo_viewable():
            return
        size = current_size()
        if pending is None and min(size) > 1 and size != label.preview_size:
            pending = size, executor.submit(
                _prepare_preview_image, image, size, background, align_top,
                plot_bounds, plot_margins, plot_box_aspect, layout,
            )
        queue_poll()

    def queue_poll():
        nonlocal poll_after_id
        if pending is not None and poll_after_id is None:
            poll_after_id = label.after(8, install_preview)

    def install_preview():
        nonlocal poll_after_id, pending
        poll_after_id = None
        if not label.winfo_viewable():
            return
        size, future = pending
        if not future.done():
            queue_poll()
            return
        pending = None
        bitmap, bounds = future.result()
        if size != current_size():
            queue_redraw()
            return
        photo = ImageTk.PhotoImage(bitmap, master=label)
        label.configure(image=photo)
        label.preview_photo = photo
        label.preview_size = size
        if on_image_layout is not None:
            on_image_layout(label, bounds)

    def queue_redraw(_event=None):
        nonlocal redraw_after_id
        # Configure also fires when a scrolling window only moves. Those
        # events must reuse the pixels already uploaded to Tk.
        if _event is not None and _event.type == tk.EventType.Configure and (
            _event.width, _event.height
        ) == label.preview_size:
            return
        if redraw_after_id is None:
            redraw_after_id = label.after_idle(redraw)

    def cleanup(event):
        nonlocal redraw_after_id, poll_after_id, pending
        if event.widget is label:
            if redraw_after_id is not None:
                label.after_cancel(redraw_after_id)
                redraw_after_id = None
            if poll_after_id is not None:
                label.after_cancel(poll_after_id)
                poll_after_id = None
            executor.shutdown(wait=False, cancel_futures=True)
            pending = None
            label.preview_photo = None

    label.bind("<Configure>", queue_redraw, add="+")
    label.bind("<Map>", queue_redraw, add="+")
    label.bind("<Destroy>", cleanup, add="+")
    if on_click is not None:
        label.configure(cursor="hand2", takefocus=True)
        label.bind("<Button-1>", lambda _event: on_click(), add="+")
        label.bind("<Return>", lambda _event: on_click(), add="+")
        label.bind("<space>", lambda _event: on_click(), add="+")
    queue_redraw()
    return label


def _prepare_preview_image(image, size, background, align_top, bounds, margins, aspect, layout):
    width, height = size
    source = image if image.mode == "RGB" else image.convert("RGB")
    if bounds is not None and margins is not None:
        left, top, fitted_width, fitted_height = layout(source.size, bounds, margins, size, aspect)
        fitted = source.resize((fitted_width, fitted_height), Image.Resampling.LANCZOS)
    else:
        fitted = ImageOps.contain(source, size, Image.Resampling.LANCZOS)
        left = (width - fitted.width) // 2
        top = 0 if align_top else (height - fitted.height) // 2
    bitmap = Image.new("RGB", size, background)
    bitmap.paste(fitted, (left, top))
    return bitmap, (left, top, fitted.width, fitted.height)
