"""Pillow-backed image helpers for Step 3 previews and plot exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _pil_font(size, bold=False):
    """Return a readable UI font without making a platform-specific font mandatory."""
    candidates = []
    if bold:
        candidates.extend(
            (
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            )
        )
    candidates.extend(
        (
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def placeholder_image(text, size=(1400, 900), title=None):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    title_font = _pil_font(42, bold=True)
    body_font = _pil_font(28)
    if title:
        draw.text((size[0] // 2, 120), title, fill="#333333", font=title_font, anchor="mm")
        y = 220
    else:
        y = size[1] // 2
    draw.multiline_text((size[0] // 2, y), text, fill="#555555", font=body_font, anchor="mm", align="center", spacing=10)
    return image


def _array_to_grayscale_image(array):
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return Image.new("RGB", (1, 1), "black")

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        scaled = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, (1, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            scaled = np.zeros(arr.shape, dtype=np.uint8)
        else:
            normalized = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
            scaled = np.round(normalized * 255.0).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def _fit_image_to_box(image, box, fill="white"):
    left, top, right, bottom = [int(v) for v in box]
    width = max(1, right - left)
    height = max(1, bottom - top)
    fitted = ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    offset = ((width - fitted.width) // 2, (height - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas, (left + offset[0], top + offset[1], fitted.width, fitted.height)


def _draw_image_panel(canvas, source, box, title, line=None, line_axis="h", line_color="#c72525"):
    draw = ImageDraw.Draw(canvas)
    title_font = _pil_font(30, bold=True)
    left, top, right, bottom = [int(v) for v in box]
    draw.text((left, top - 44), title, fill="#333333", font=title_font, anchor="la")
    fitted, placement = _fit_image_to_box(source, box, fill="#f5f5f5")
    canvas.paste(fitted, (left, top))
    draw.rectangle((left, top, right, bottom), outline="#666666", width=2)

    if line is None:
        return

    paste_x, paste_y, draw_w, draw_h = placement
    if line_axis == "h" and source.height > 1:
        y = paste_y + (float(line) / max(1, source.height - 1)) * draw_h
        if paste_y <= y <= paste_y + draw_h:
            draw.line((paste_x, y, paste_x + draw_w, y), fill=line_color, width=4)
    elif line_axis == "v" and source.width > 1:
        x = paste_x + (float(line) / max(1, source.width - 1)) * draw_w
        if paste_x <= x <= paste_x + draw_w:
            draw.line((x, paste_y, x, paste_y + draw_h), fill=line_color, width=4)


def make_profile_plot_image(
    profile_xy,
    title,
    verticals=(),
    spline_xy=None,
    size=(1800, 1100),
    vertical_color=None,
):
    profile_xy = np.asarray(profile_xy, dtype=np.float64)
    if profile_xy.size == 0 or profile_xy.ndim != 2 or profile_xy.shape[1] < 2:
        return placeholder_image("No profile data available.", size=size, title=title)

    x = profile_xy[:, 0]
    y = profile_xy[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return placeholder_image("No finite profile values available.", size=size, title=title)

    x = x[valid]
    y = y[valid]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        pad = max(1.0, abs(y_min) * 0.05)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    title_font = _pil_font(42, bold=True)
    label_font = _pil_font(28)
    tick_font = _pil_font(22)
    plot_left, plot_top = 155, 125
    plot_right, plot_bottom = size[0] - 70, size[1] - 155
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    draw.text((size[0] // 2, 46), title, fill="#222222", font=title_font, anchor="ma")
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#333333", width=2)

    for idx in range(1, 5):
        gx = plot_left + (plot_w * idx / 5.0)
        gy = plot_top + (plot_h * idx / 5.0)
        draw.line((gx, plot_top, gx, plot_bottom), fill="#dddddd", width=1)
        draw.line((plot_left, gy, plot_right, gy), fill="#dddddd", width=1)

    def to_px(px, py):
        sx = plot_left + ((float(px) - x_min) / (x_max - x_min)) * plot_w
        sy = plot_bottom - ((float(py) - y_min) / (y_max - y_min)) * plot_h
        return sx, sy

    points = [to_px(px, py) for px, py in zip(x, y)]
    if len(points) > 1:
        draw.line(points, fill="#111111", width=3, joint="curve")
    else:
        px, py = points[0]
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="#111111")

    for idx, x_val in enumerate(verticals or ()):
        try:
            x_float = float(x_val)
        except Exception:
            continue
        if x_min <= x_float <= x_max:
            vx, _ = to_px(x_float, y_min)
            color = vertical_color or ("red" if idx == len(verticals) - 1 and len(verticals) > 3 else "black")
            draw.line((vx, plot_top, vx, plot_bottom), fill=color, width=3)

    if spline_xy is not None:
        spline_xy = np.asarray(spline_xy, dtype=np.float64)
        if spline_xy.ndim == 2 and spline_xy.shape[1] >= 2:
            for sx, sy in spline_xy[:, :2]:
                if np.isfinite(sx) and np.isfinite(sy) and x_min <= sx <= x_max and y_min <= sy <= y_max:
                    px, py = to_px(sx, sy)
                    draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline="#111111", width=3)

    for idx in range(6):
        tx = x_min + (x_max - x_min) * idx / 5.0
        ty = y_min + (y_max - y_min) * idx / 5.0
        x_px, _ = to_px(tx, y_min)
        _, y_px = to_px(x_min, ty)
        draw.text((x_px, plot_bottom + 20), f"{tx:.0f}", fill="#333333", font=tick_font, anchor="ma")
        draw.text((plot_left - 14, y_px), f"{ty:.2g}", fill="#333333", font=tick_font, anchor="rm")

    draw.text(((plot_left + plot_right) // 2, size[1] - 55), "column", fill="#333333", font=label_font, anchor="mm")
    draw.text((plot_left, 92), "mean intensity", fill="#333333", font=label_font, anchor="la")
    return image


def save_profile_plot(profile_xy, output_path, title, verticals=(), spline_xy=None):
    """Save a simple profile plot matching `main.py`'s `save_profile_plot` without Matplotlib."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile_xy = np.asarray(profile_xy)
    if profile_xy.size == 0 or profile_xy.shape[0] == 0:
        return

    image = make_profile_plot_image(profile_xy, title, verticals=verticals, spline_xy=spline_xy)
    image.save(output_path)


def make_find_vertex_preview_image(results):
    gm = np.asarray(results["final_grand_mean"], dtype=np.float64)
    profile = np.nanmean(gm, axis=0)
    xs = np.arange(1, profile.size + 1, dtype=np.float64)
    profile_image = make_profile_plot_image(
        np.column_stack((xs, profile)),
        "Intensity Profile",
        verticals=(results["vertex"],),
        size=(2200, 720),
        vertical_color="#c72525",
    )
    gm_image = _array_to_grayscale_image(gm.T)

    canvas = Image.new("RGB", (2200, 1450), "white")
    canvas.paste(profile_image, (0, 0))
    _draw_image_panel(
        canvas,
        gm_image,
        (120, 860, 2080, 1340),
        "Final Grand Mean",
        line=results["vertex"],
        line_axis="h",
        line_color="#c72525",
    )
    return canvas


def make_comparison_preview_image(original_light, flattened_light, slice_idx):
    original_image = _array_to_grayscale_image(np.rot90(original_light[:, :, slice_idx], k=1))
    flattened_image = _array_to_grayscale_image(flattened_light[slice_idx].T)

    canvas = Image.new("RGB", (2200, 1400), "white")
    title_font = _pil_font(44, bold=True)
    draw = ImageDraw.Draw(canvas)
    draw.text((1100, 50), f"Step 3 comparison - slice {slice_idx}", fill="#222222", font=title_font, anchor="ma")
    _draw_image_panel(canvas, original_image, (120, 160, 2080, 640), "Original LIGHT")
    _draw_image_panel(canvas, flattened_image, (120, 820, 2080, 1300), "Flattened LIGHT")
    return canvas


def make_main_results_summary_image(results):
    canvas = Image.new("RGB", (2400, 1800), "white")
    title_font = _pil_font(46, bold=True)
    draw = ImageDraw.Draw(canvas)
    draw.text((1200, 50), "OCT flattening results", fill="#222222", font=title_font, anchor="ma")

    first = _array_to_grayscale_image(results["first_grand_mean"])
    second = _array_to_grayscale_image(results["second_grand_mean"])
    final = _array_to_grayscale_image(results["final_grand_mean"])
    profile = make_profile_plot_image(
        results["grand_profile"],
        "Final Grand Profile",
        verticals=(results["vertex"],),
        size=(1040, 680),
        vertical_color="#c72525",
    )

    _draw_image_panel(canvas, first, (100, 170, 1100, 780), "First Grand Mean")
    _draw_image_panel(canvas, second, (1300, 170, 2300, 780), "Second Grand Mean")
    _draw_image_panel(
        canvas,
        final,
        (100, 1030, 1100, 1660),
        "Final Grand Mean",
        line=results["vertex"],
        line_axis="h",
        line_color="#c72525",
    )
    canvas.paste(profile, (1230, 940))
    return canvas
