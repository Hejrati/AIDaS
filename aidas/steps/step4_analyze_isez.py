"""Step 4 - Analyze ISez.

This module is intentionally a close translation of the lab's original Step 4
workflow:

1. ``Rotated_Rescaled_Human_ISez6_8bit_JS.m`` computes the ROI intensity
   profile, adjusts the clicked IS band bounds, rotates the selected IS band
   baseline, rescales the surrounding profile, and saves an ISez plot.
2. ``Step 4 analyze ISez.txt`` is the ImageJ macro that opens those ISez plot
   frames, runs ``doWand(512, 179)``, measures with
   ``Set Measurements... fit shape invert redirect=None decimal=3``, and then
   saves ``ROI_to_move_stck.tif``, ``MAX_Stack.tif``, and the results table.

ImageJ reference source used for the measurement port:
- Wand selection:
  https://github.com/imagej/ImageJ/blob/master/ij/gui/Wand.java
- Ellipse fit:
  https://github.com/imagej/ImageJ/blob/master/ij/process/EllipseFitter.java
- Result columns and shape descriptors:
  https://github.com/imagej/ImageJ/blob/master/ij/plugin/filter/Analyzer.java
- Statistics-to-ellipse handoff:
  https://github.com/imagej/ImageJ/blob/master/ij/process/ImageStatistics.java

Important limitation: the macro pauses after each ``doWand`` call with
``waitForUser("Pause","Do Wand")``. If the user manually edits the wand ROI in
ImageJ during that pause, the saved Results.xlsx can contain values that cannot
be reproduced from the plot image alone because that manual ROI state is not
stored in ``ROI_to_move_stck.tif``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import zipfile
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from aidas.utils.io_utils import read_analyze, read_tiff
from aidas.utils.ui_utils import SidebarStepFrame, directory_row, load_ui_icon


MATLAB_ROI_LOW = 300
MATLAB_ROI_HIGH = 450
MATLAB_ROI_TOP_LINE = 450
MATLAB_A_LIMIT = 1.0

# These dimensions are the generated plot-frame geometry used before applying
# the ImageJ macro measurement. They are kept fixed so the same wand seed
# location, doWand(512, 179), lands inside the drawn ISez shape.
IMAGEJ_ISEZ_CANVAS_SIZE = (969, 513)
IMAGEJ_PLOT_BOX = (126.0, 38.0, 876.0, 456.0)

# Column order and text match the ImageJ Results table produced by the macro's
# "fit shape" measurement options and 3-decimal precision.
RESULTS_HEADERS = [" ", "Major", "Minor", "Angle", "Circ.", "AR", "Round", "Solidity"]


@dataclass(frozen=True)
class ISezROI:
    """One MATLAB-style ISez ROI definition.

    Coordinates are 1-based and inclusive to match the source MATLAB script.
    """

    suffix: str
    left: int
    right: int
    low: int = MATLAB_ROI_LOW
    high: int = MATLAB_ROI_HIGH


@dataclass
class ISezResult:
    roi: ISezROI
    start: int
    end: int
    adjusted_start: int
    adjusted_end: int
    center: float
    slope: float
    max_index: int
    min_intensity: float
    max_intensity: float
    normalized_x: np.ndarray
    normalized_y: np.ndarray
    baseline_x: np.ndarray
    baseline_y: np.ndarray


def default_isez_rois() -> list[ISezROI]:
    """Return the 20 peripheral ROIs plus the fovea ROI from the MATLAB file."""

    rois = [
        ISezROI(
            suffix=f"{idx:02d}",
            left=2851 - (120 * idx),
            right=2850 - (120 * (idx - 1)),
        )
        for idx in range(1, 21)
    ]
    rois.append(ISezROI(suffix="fovea", left=95, right=105))
    return rois


def load_oct_volume(path: str | os.PathLike) -> np.ndarray:
    """Load a supported OCT image as a stack shaped ``(slices, height, width)``."""

    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".hdr", ".img"}:
        data = read_analyze(path)
    elif ext in {".tif", ".tiff"}:
        data = read_tiff(path)
    else:
        data = np.asarray(Image.open(path).convert("L"))

    data = np.asarray(data)
    if data.ndim == 2:
        return data[np.newaxis, :, :]
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        gray = Image.fromarray(data).convert("L")
        return np.asarray(gray)[np.newaxis, :, :]
    if data.ndim == 3:
        return data
    raise ValueError("Step 4 expects a 2-D grayscale image or a 3-D image stack.")


def roi_bounds_for_image(roi: ISezROI, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Convert a 1-based/inclusive ROI into clipped Python slice bounds."""

    height, width = image_shape
    left = max(0, roi.left - 1)
    right = min(width, roi.right)
    low = max(0, roi.low - 1)
    high = min(height, roi.high)
    if left >= right or low >= high:
        raise ValueError(
            f"ROI {roi.suffix} is outside this image "
            f"({width} x {height}); MATLAB bounds are x={roi.left}:{roi.right}, "
            f"y={roi.low}:{roi.high}."
        )
    return left, right, low, high


def intensity_profile(image: np.ndarray, roi: ISezROI) -> np.ndarray:
    """Return MATLAB ``Int=mean(oct(low:450,left:right)/3,2)`` for one ROI."""

    arr = np.asarray(image, dtype=np.float64) / 3.0
    left, right, low, high = roi_bounds_for_image(roi, arr.shape)
    return np.nanmean(arr[low:high, left:right], axis=1)


def to_uint8_display(image: np.ndarray) -> np.ndarray:
    """Scale any grayscale array to uint8 for JPEG/preview output."""

    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return np.array(arr, copy=True)

    work = arr.astype(np.float64, copy=False)
    finite = np.isfinite(work)
    if not np.any(finite):
        return np.zeros(work.shape, dtype=np.uint8)
    lo = float(np.nanmin(work[finite]))
    hi = float(np.nanmax(work[finite]))
    if hi > lo:
        work = (work - lo) / (hi - lo) * 255.0
    return np.nan_to_num(np.clip(work, 0, 255), nan=0.0).astype(np.uint8)


def roi_overlay_image(image: np.ndarray, roi: ISezROI) -> np.ndarray:
    """Return a uint8 copy of the image with the MATLAB ROI box drawn in white."""

    out = to_uint8_display(image)
    left, right, low, high = roi_bounds_for_image(roi, out.shape)
    high_line = min(out.shape[0] - 1, max(0, MATLAB_ROI_TOP_LINE - 1))
    low_line = low
    out[high_line, left:right] = 255
    out[low_line, left:right] = 255
    out[low: high_line + 1, left] = 255
    out[low: high_line + 1, right - 1] = 255
    return out


def _clamp_profile_index(value: float | int, n_points: int) -> int:
    return max(1, min(int(round(float(value))), n_points))


def adjust_isez_bounds(
    profile: np.ndarray,
    start: int,
    end: int,
    *,
    right_click_x: float | None = None,
    a_limit: float = MATLAB_A_LIMIT,
) -> tuple[int, int, float]:
    """Apply the four MATLAB while-loop adjustments to the selected bounds.

    Reference: ``Rotated_Rescaled_Human_ISez6_8bit_JS.m`` after the two
    ``ginput`` calls. MATLAB indexes are 1-based, so this function keeps the
    public ``s`` and ``e`` variables as 1-based values and only subtracts one
    when reading from the NumPy array.
    """

    values = np.asarray(profile, dtype=np.float64)
    n_points = values.size
    if n_points < 4:
        raise ValueError("The selected ROI profile is too short to analyze.")

    s = _clamp_profile_index(start, n_points)
    e = _clamp_profile_index(end, n_points)
    if s > e:
        s, e = e, s
    if s == e:
        e = min(n_points, s + 1)
    s = max(2, min(s, n_points - 1))
    e = max(s + 1, min(e, n_points - 1))
    x_right = float(right_click_x if right_click_x is not None else e)

    def slope(current_s: int, current_e: int) -> float:
        if current_e == current_s:
            return 0.0
        return float((values[current_e - 1] - values[current_s - 1]) / (current_e - current_s))

    slop = slope(s, e)
    max_iterations = n_points * 4

    iterations = 0
    while (
        s < n_points
        and s > 1
        and e < x_right + a_limit
        and values[s - 1] + slop - values[s] > 0
        and values[s - 1] - slop - values[s - 2] < 0
    ):
        if e > s + 1:
            s += 1
            slop = slope(s, e)
        else:
            break
        iterations += 1
        if iterations > max_iterations:
            break

    iterations = 0
    while (
        s < n_points
        and s > 1
        and e < x_right + a_limit
        and values[s - 1] + slop - values[s] < 0
        and values[s - 1] - slop - values[s - 2] > 0
    ):
        s -= 1
        slop = slope(s, e)
        iterations += 1
        if iterations > max_iterations:
            break

    iterations = 0
    while (
        e < n_points
        and e > 1
        and e < x_right + a_limit
        and values[e - 1] + slop - values[e] < 0
        and values[e - 1] - slop - values[e - 2] > 0
    ):
        if e > s + 1:
            e -= 1
            slop = slope(s, e)
        else:
            break
        iterations += 1
        if iterations > max_iterations:
            break

    iterations = 0
    while (
        e < n_points
        and e > 1
        and e < x_right + a_limit
        and values[e - 1] + slop - values[e] > 0
        and values[e - 1] - slop - values[e - 2] < 0
    ):
        e += 1
        slop = slope(s, e)
        iterations += 1
        if iterations > max_iterations:
            break

    return s, e, slop


def rotated_rescaled_isez(profile: np.ndarray, start: int, end: int) -> dict[str, np.ndarray | float | int]:
    """Rotate the IS band baseline and rescale the surrounding region to 0-100.

    This is the Python equivalent of the MATLAB block beginning at
    ``IS=Int(s:e);`` and ending after:

    ``Int_region = (Int_region - min_Int_region) / ... * 100``.

    The matrix multiplication from MATLAB,
    ``R * [x_points - x_rotation_point; y_points - y_rotation_point]``, is
    expanded below into scalar/vector operations. The math is identical, but it
    avoids a native BLAS crash seen on the Windows test environment.
    """

    values = np.asarray(profile, dtype=np.float64)
    n_points = values.size
    s = _clamp_profile_index(start, n_points)
    e = _clamp_profile_index(end, n_points)
    if s > e:
        s, e = e, s
    if e <= s:
        raise ValueError("ISez end must be greater than start.")

    is_region = values[s - 1:e]

    # MATLAB:
    # x_rotation_point = (e + s)/2;
    # y_rotation_point = (Int(s) + Int(e)) / 2;
    # theta_radians = acos((e-s)/sqrt((e-s)^2 + (IS(end)-IS(1))^2));
    x_rotation_point = (e + s) / 2.0
    y_rotation_point = (values[s - 1] + values[e - 1]) / 2.0
    adjacent_length = float(e - s)
    hypotenuse_length = float(np.sqrt(adjacent_length**2 + (is_region[-1] - is_region[0]) ** 2))
    if hypotenuse_length <= 0:
        theta_radians = 0.0
    else:
        ratio = max(-1.0, min(1.0, adjacent_length / hypotenuse_length))
        theta_radians = float(np.arccos(ratio))

    x_points = np.linspace(s, e, is_region.size)
    x_shifted = x_points - x_rotation_point
    y_shifted = is_region - y_rotation_point

    # MATLAB clockwise rotation matrix:
    # R = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)]
    # We only need new_y for the peak index used during rescaling.
    cos_t = float(np.cos(-theta_radians))
    sin_t = float(np.sin(-theta_radians))
    new_y = (sin_t * x_shifted) + (cos_t * y_shifted) + y_rotation_point
    max_index_zero = int(np.nanargmax(new_y))

    # MATLAB uses Int(s-3:e+5), then normalizes by the minimum of IS and the
    # original IS intensity at the rotated maximum index.
    region_start = max(1, s - 3)
    region_end = min(n_points, e + 5)
    x_region = np.arange(region_start, region_end + 1, dtype=np.float64)
    int_region = values[region_start - 1:region_end]

    max_intensity = float(is_region[max_index_zero])
    min_intensity = float(np.nanmin(is_region))
    denom = max_intensity - min_intensity
    if abs(denom) < np.finfo(np.float64).eps:
        normalized = np.zeros_like(int_region, dtype=np.float64)
    else:
        normalized = ((int_region - min_intensity) / denom) * 100.0

    baseline_y = np.array(
        [
            normalized[int(s - region_start)] if region_start <= s <= region_end else np.nan,
            normalized[int(e - region_start)] if region_start <= e <= region_end else np.nan,
        ],
        dtype=np.float64,
    )

    return {
        "x": x_region,
        "y": normalized,
        "baseline_x": np.array([s, e], dtype=np.float64),
        "baseline_y": baseline_y,
        "max_index": max_index_zero + 1,
        "max_intensity": max_intensity,
        "min_intensity": min_intensity,
    }


def make_isez_plot_image(result: ISezResult) -> Image.Image:
    """Render one result as the ImageJ-stack ISez plot frame.

    The original MATLAB script used ``saveas(gcf, *_ISez_*.png)`` and the
    ImageJ macro later opened those PNGs. This function creates the equivalent
    final frame in memory so Step 4 only writes ``ROI_to_move_stck.tiff`` when
    the user presses the stack button.
    """

    image = Image.new("RGB", IMAGEJ_ISEZ_CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = IMAGEJ_PLOT_BOX
    x_min = result.center - 40.0
    x_max = result.center + 40.0
    y_min = -20.0
    y_max = 120.0

    def to_pixel(px: float, py: float) -> tuple[float, float]:
        x = x0 + ((float(px) - x_min) / (x_max - x_min)) * (x1 - x0)
        y = y0 + ((y_max - float(py)) / (y_max - y_min)) * (y1 - y0)
        return x, y

    draw.rectangle((x0, y0, x1, y1), outline="black", width=1)
    for tick in np.linspace(x_min, x_max, 5):
        tx, _ = to_pixel(tick, y_min)
        draw.line((tx, y1, tx, y1 + 5), fill="black")
    for tick in [-20, 0, 50, 100, 120]:
        _, ty = to_pixel(x_min, tick)
        draw.line((x0 - 5, ty, x0, ty), fill="black")

    curve = [to_pixel(px, py) for px, py in zip(result.normalized_x, result.normalized_y)]
    curve = [(x, y) for x, y in curve if x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2]
    if len(curve) >= 2:
        draw.line(curve, fill="black", width=2)
    baseline = [to_pixel(px, py) for px, py in zip(result.baseline_x, result.baseline_y)]
    if len(baseline) >= 2:
        draw.line(baseline, fill="black", width=2)
    return image


def analyze_and_save_roi(
    image: np.ndarray,
    roi: ISezROI,
    *,
    start_click: float,
    end_click: float,
) -> ISezResult:
    """Run the MATLAB ISez analysis for one ROI without writing files.

    Confirming a ROI stores the data in memory. The ImageJ-style workbook and
    TIFF stacks are generated only by ``_build_stack_outputs``.
    """

    profile = intensity_profile(image, roi)
    start = _clamp_profile_index(start_click, profile.size)
    end = _clamp_profile_index(end_click, profile.size)
    if start > end:
        start, end = end, start
        start_click, end_click = end_click, start_click

    adjusted_start, adjusted_end, slope = adjust_isez_bounds(
        profile,
        start,
        end,
        right_click_x=end_click,
    )
    center = (start + end) / 2.0
    rescaled = rotated_rescaled_isez(profile, adjusted_start, adjusted_end)

    return ISezResult(
        roi=roi,
        start=start,
        end=end,
        adjusted_start=adjusted_start,
        adjusted_end=adjusted_end,
        center=center,
        slope=slope,
        max_index=int(rescaled["max_index"]),
        min_intensity=float(rescaled["min_intensity"]),
        max_intensity=float(rescaled["max_intensity"]),
        normalized_x=np.asarray(rescaled["x"], dtype=np.float64),
        normalized_y=np.asarray(rescaled["y"], dtype=np.float64),
        baseline_x=np.asarray(rescaled["baseline_x"], dtype=np.float64),
        baseline_y=np.asarray(rescaled["baseline_y"], dtype=np.float64),
    )


def _worksheet_cell_ref(row: int, col: int) -> str:
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


def _xlsx_cell_xml(row: int, col: int, value) -> str:
    ref = _worksheet_cell_ref(row, col)
    if isinstance(value, str):
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
    if value is None:
        return f'<c r="{ref}"/>'
    return f'<c r="{ref}"><v>{float(value):.15g}</v></c>'


def write_imagej_results_xlsx(rows: list[dict[str, float]], path: str | os.PathLike) -> None:
    """Write the ImageJ-style Step 4 Results.xlsx workbook.

    ImageJ's Results table is tabular text internally; the old workflow saved
    it as an Excel workbook. ``openpyxl`` is not a project dependency here, so
    this writes the small XLSX package directly. The worksheet name is ``in``
    and the column order follows the GT workbook from ``Examples/step4``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = [RESULTS_HEADERS]
    for idx, row in enumerate(rows, start=1):
        sheet_rows.append(
            [
                idx,
                row["Major"],
                row["Minor"],
                row["Angle"],
                row["Circ."],
                row["AR"],
                row["Round"],
                row["Solidity"],
            ]
        )

    last_ref = _worksheet_cell_ref(len(sheet_rows), len(RESULTS_HEADERS))
    row_xml = []
    for r_idx, values in enumerate(sheet_rows, start=1):
        cells = "".join(_xlsx_cell_xml(r_idx, c_idx, value) for c_idx, value in enumerate(values, start=1))
        row_xml.append(f'<row r="{r_idx}">{cells}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:{last_ref}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        '<cols><col min="1" max="8" width="12" customWidth="1"/></cols>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="in" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)


def _imagej_wand_mask(image: Image.Image, point: tuple[int, int] = (512, 179)) -> np.ndarray:
    """Select the macro's wand region at doWand(512,179).

    Source reference: ImageJ ``ij.gui.Wand``. For these generated black/white
    ISez frames, the wand selection is equivalent to choosing the connected
    component with the same grayscale value as the seed pixel. The macro uses
    the fixed seed point ``(512, 179)`` on every slice.
    """

    arr = np.asarray(image.convert("L"))
    x, y = point
    if not (0 <= y < arr.shape[0] and 0 <= x < arr.shape[1]):
        raise ValueError("ImageJ wand point is outside the ISez plot frame.")

    seed = int(arr[y, x])
    mask = arr == seed
    labels, _count = ndimage.label(mask)
    label = int(labels[y, x])
    if label == 0:
        return np.zeros(arr.shape, dtype=bool)
    return labels == label


def _imagej_ellipse_fit(mask: np.ndarray) -> tuple[float, float, float]:
    """Port of ImageJ EllipseFitter.getEllipseParam() for an ROI mask.

    Source reference:
    ``ij.process.EllipseFitter.getEllipseParam()``.

    Key details copied from ImageJ:
    - compute second-order central moments from mask pixels;
    - derive ``a11``, ``a12``, ``a22`` from ``u20``, ``u02``, ``u11``;
    - use the same theta quadrant logic;
    - scale the ellipse so its area equals the selected pixel count;
    - return major/minor lengths and angle in degrees.
    """

    ys, xs = np.where(mask)
    bit_count = int(xs.size)
    if bit_count == 0:
        return 0.0, 0.0, 0.0

    left = int(xs.min())
    top = int(ys.min())
    local_x = xs.astype(np.float64) - left
    local_y = ys.astype(np.float64) - top

    xsum = float(np.sum(local_x))
    ysum = float(np.sum(local_y))
    x2sum = float(np.sum(local_x * local_x))
    y2sum = float(np.sum(local_y * local_y))
    xysum = float(np.sum(local_x * local_y))
    n = float(bit_count)
    xm = xsum / n
    ym = ysum / n
    u20 = x2sum / n - xm * xm
    u02 = y2sum / n - ym * ym
    u11 = xysum / n - xm * ym

    # The following block mirrors ImageJ's getEllipseParam() variable names.
    # Keeping those names makes it easier to compare line-by-line with the
    # Java source while debugging GT differences.
    half_pi = 1.5707963267949
    m4 = 4.0 * abs(u02 * u20 - u11 * u11)
    if m4 < 0.000001:
        m4 = 0.000001
    a11 = u02 / m4
    a12 = u11 / m4
    a22 = u20 / m4
    tmp = a11 - a22
    if tmp == 0.0:
        tmp = 0.000001
    theta = 0.5 * math.atan(2.0 * a12 / tmp)
    if theta < 0.0:
        theta += half_pi
    if a12 > 0.0:
        theta += half_pi
    elif a12 == 0.0:
        if a22 > a11:
            theta = 0.0
            tmp = a22
            a22 = a11
            a11 = tmp
        elif a11 != a22:
            theta = half_pi

    sin_theta = math.sin(theta)
    if sin_theta == 0.0:
        sin_theta = 0.000001
    z = a12 * math.cos(theta) / sin_theta
    major = math.sqrt(1.0 / abs(a22 + z))
    minor = math.sqrt(1.0 / abs(a11 - z))
    scale = math.sqrt(bit_count / (math.pi * major * minor))
    major = major * scale * 2.0
    minor = minor * scale * 2.0
    angle = 180.0 * theta / math.pi
    if angle == 180.0:
        angle = 0.0
    if major < minor:
        major, minor = minor, major
    return major, minor, angle


def _imagej_boundary_points(mask: np.ndarray) -> np.ndarray:
    """Trace the selected wand region boundary using Moore-neighbor tracing.

    ImageJ circularity uses ``roi.getLength()`` as perimeter in
    ``Analyzer.saveResults``. For the binary wand region here, tracing the outer
    boundary with 8-neighbor steps reproduces the perimeter used by ImageJ
    closely enough to match the GT rows that were not manually edited.
    """

    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    order = np.lexsort((xs, ys))
    start = (int(ys[order[0]]), int(xs[order[0]]))
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    height, width = mask.shape
    point = start
    backtrack = 4
    boundary = []

    for _ in range(mask.size * 4):
        boundary.append(point)
        found = False
        for offset in range(8):
            idx = (backtrack + 1 + offset) % 8
            dy, dx = directions[idx]
            candidate = (point[0] + dy, point[1] + dx)
            if 0 <= candidate[0] < height and 0 <= candidate[1] < width and mask[candidate]:
                backtrack = (idx + 4) % 8
                point = candidate
                found = True
                break
        if not found or (point == start and len(boundary) > 1):
            break

    return np.asarray([(x, y) for y, x in boundary], dtype=np.float64)


def _imagej_perimeter(boundary_points: np.ndarray) -> float:
    """Length of the wand boundary polygon used for ImageJ circularity."""

    if len(boundary_points) < 2:
        return 0.0
    shifted = np.roll(boundary_points, -1, axis=0)
    return float(np.sum(np.hypot(shifted[:, 0] - boundary_points[:, 0], shifted[:, 1] - boundary_points[:, 1])))


def _imagej_convex_hull_area(mask: np.ndarray) -> float:
    """Return the convex hull area used for ImageJ Solidity.

    Source reference: ``Analyzer.saveResults`` computes:
    ``Solidity = stats.pixelCount / getArea(roi.getConvexHull())``.

    Pixel centers make the hull slightly too small. ImageJ's polygon hull is
    effectively around pixel edges for these filled wand regions, so the hull
    is built from each selected pixel's four corners.
    """

    ys, xs = np.where(mask)
    if xs.size < 3:
        return 0.0

    corners = np.empty((xs.size * 4, 2), dtype=np.float64)
    corners[0::4, 0] = xs - 0.5
    corners[0::4, 1] = ys - 0.5
    corners[1::4, 0] = xs + 0.5
    corners[1::4, 1] = ys - 0.5
    corners[2::4, 0] = xs + 0.5
    corners[2::4, 1] = ys + 0.5
    corners[3::4, 0] = xs - 0.5
    corners[3::4, 1] = ys + 0.5

    try:
        from scipy.spatial import ConvexHull

        return float(ConvexHull(corners).volume)
    except Exception:
        return 0.0


def imagej_shape_measurements_from_frame(image: Image.Image) -> dict[str, float]:
    """Measure one ISez plot frame using the Step 4 ImageJ macro workflow.

    Source reference: ``Step 4 analyze ISez.txt`` does:
    ``run("Set Measurements...", "fit shape invert redirect=None decimal=3")``
    then ``doWand(512, 179)`` and ``run("Measure")`` for each stack slice.

    Source reference: ``Analyzer.saveResults`` writes these shape columns:
    - Circ. = min(1, 4*pi*area/perimeter^2)
    - AR = major/minor
    - Round = minor/major for the current uncalibrated square-pixel data,
      equivalent to ImageJ's 4*area/(pi*major^2) after the area-equalized
      ellipse fit
    - Solidity = pixelCount/convexHullArea
    """

    mask = _imagej_wand_mask(image)
    area = float(np.count_nonzero(mask))
    if area <= 0:
        return {key: 0.0 for key in RESULTS_HEADERS[1:]}

    major, minor, angle = _imagej_ellipse_fit(mask)
    boundary_points = _imagej_boundary_points(mask)
    perimeter = _imagej_perimeter(boundary_points)
    circularity = 0.0 if perimeter <= 0 else min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
    aspect_ratio = 0.0 if minor <= 0 else major / minor
    roundness = 0.0 if major <= 0 else minor / major
    hull_area = _imagej_convex_hull_area(mask)
    solidity = min(1.0, area / hull_area) if hull_area > 0 else 0.0

    return {
        "Major": round(major, 3),
        "Minor": round(minor, 3),
        "Angle": round(angle, 3),
        "Circ.": round(circularity, 3),
        "AR": round(aspect_ratio, 3),
        "Round": round(roundness, 3),
        "Solidity": round(solidity, 3),
    }


class Step4Frame(SidebarStepFrame):
    """Step 4 tab UI for interactive ISez profile selection and output saving."""

    def __init__(self, parent, preferences=None, source_step=None):
        super().__init__(parent)
        self.preferences = preferences
        self.source_step = source_step

        self.rois = default_isez_rois()
        self.volume = None
        self.image = None
        self.current_path = None
        self.current_stem = "_flat_LIGHT"
        self.current_roi_idx = 0
        self.completed: dict[str, ISezResult] = {}
        self.roi_clicks: dict[str, list[float]] = {}
        self.profile_clicks: list[float] = []
        self.figure = None
        self.canvas = None
        self.empty_placeholder = None
        self.ax_profile = None
        self.ax_isez = None
        self._current_profile = None
        self._updating_roi_selection = False
        self._input_dir_user_selected = False
        self._output_dir_user_selected = False

        self.input_dir_var = tk.StringVar(value=self._default_input_folder())
        self.output_dir_var = tk.StringVar(value=self._default_input_folder())
        self.image_label_var = tk.StringVar(value="No image loaded")
        self.status_var = tk.StringVar(value="Ready - load a flattened OCT image to begin.")
        self.profile_status_var = tk.StringVar(value="Click start and end on the profile canvas.")
        self.stats_var = tk.StringVar(value="")
        self.slice_var = tk.StringVar(value="0")
        self.start_var = tk.StringVar(value="")
        self.end_var = tk.StringVar(value="")
        self.confirm_button = None
        # self.auto_advance_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._refresh_roi_list()

    def _build_ui(self) -> None:
        self.build_standard_layout(
            sidebar_width=self.SIDEBAR_WIDTH,
            sidebar_pack={"padx": (2, 6), "pady": 6},
            content_pack={"padx": 6, "pady": 6},
            status_var=self.status_var,
        )

        source_section = self.add_sidebar_section("Input", pady=(0, 5))
        source = source_section.body
        # Temporarily disable the file-open and Step 3 load buttons to prevent
        # loading files while the feature is disabled for maintenance/testing.

        ttk.Label(source, text="Input dir:").pack(anchor="w", pady=(6, 0))
        folder_row, _input_entry, input_buttons = directory_row(
            source,
            self,
            self.input_dir_var,
            self._browse_input_folder,
            browse_tooltip="Browse Step 3 output folder",
        )
        folder_row.pack(fill="x", pady=(0, 8))
        self.input_search_btn = input_buttons["browse"]

        ttk.Label(
            source,
            textvariable=self.image_label_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            foreground="gray",
            justify="left",
        ).pack(fill="x", pady=(6, 0))

        slice_row = ttk.Frame(source)
        slice_row.pack(fill="x", pady=(6, 0))
        ttk.Label(slice_row, text="Slice").pack(side="left")
        self.slice_combo = ttk.Combobox(slice_row, textvariable=self.slice_var, values=["0"], state="readonly", width=8)
        self.slice_combo.pack(side="right")
        self.slice_combo.bind("<<ComboboxSelected>>", lambda _event: self._set_current_slice())

        # output_section = self.add_sidebar_section("Output", padding=3, pady=(0, 5))
        # output = output_section.body
        # ttk.Label(source, text="Output dir:").pack(anchor="w", pady=(6, 0))
        # out_row = ttk.Frame(source)
        # out_row.pack(fill="x")
        # ttk.Entry(out_row, textvariable=self.output_dir_var).pack(side="left", fill="x", expand=True)
        # ttk.Button(out_row, text="...", width=3, command=self._browse_output_folder).pack(side="right", padx=(2, 0))

        ttk.Label(source, text="Output dir:").pack(anchor="w", pady=(6, 0))
        save_dir_row, _output_entry, output_buttons = directory_row(
            source,
            self,
            self.output_dir_var,
            self._browse_output_folder,
            browse_tooltip="Browse output folder",
        )
        save_dir_row.pack(fill="x", pady=(0, 10))
        self.out_search_btn = output_buttons["browse"]


        roi_section = self.add_sidebar_section("ROIs", fill="both", expand=True, pady=(0, 5))
        roi_box = roi_section.body
        list_frame = ttk.Frame(roi_box)
        list_frame.pack(fill="both", expand=True)
        self.roi_listbox = tk.Listbox(list_frame, height=10, exportselection=False)
        roi_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.roi_listbox.yview)
        self.roi_listbox.configure(yscrollcommand=roi_scroll.set)
        self.roi_listbox.pack(side="left", fill="both", expand=True)
        roi_scroll.pack(side="right", fill="y")
        self.roi_listbox.bind("<<ListboxSelect>>", self._on_roi_selected)

        nav = ttk.Frame(roi_box)
        nav.pack(fill="x", pady=(6, 0))
        ttk.Button(nav, text="Prev", command=lambda: self._move_roi(-1)).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(nav, text="Next", command=lambda: self._move_roi(1)).pack(side="right", expand=True, fill="x", padx=(2, 0))

        # Create a SINGLE frame for the horizontal controls
        control_row = ttk.Frame(roi_box)
        # Packs the row at the top (default) of the available space
        control_row.pack(fill="x", pady=5) 

        # 1. Start Label & Entry
        ttk.Label(control_row, text="Start").pack(side="left", padx=(0, 2))
        ttk.Entry(control_row, textvariable=self.start_var, width=8).pack(side="left", padx=(0, 10))

        # 2. End Label & Entry
        ttk.Label(control_row, text="End").pack(side="left", padx=(0, 2))
        ttk.Entry(control_row, textvariable=self.end_var, width=8).pack(side="left", padx=(0, 10))

        # 3. Apply Button
        self.apply_icon = load_ui_icon(self, "streamline-stickies-color--validation-1-duo.png")
        self.apply_button = ttk.Button(control_row, image=self.apply_icon, command=self._apply_entry_clicks)
        self.apply_button.pack(side="left", expand=False) 

        # 4. Build Stacks Button
        # Change the parent from 'control_row' to 'roi_box'
        # Use side="bottom" to anchor it to the bottom of the roi_box frame
        self.build_stacks_icon = load_ui_icon(self, "tabler--stack-push.png")
        self.build_stacks_button = ttk.Button(roi_box, image=self.build_stacks_icon, text="Build Stacks", command=self._build_stack_outputs)
        self.build_stacks_button.pack(side="bottom", fill="x", pady=(6, 2))
        # stats_section = self.add_sidebar_section("Stats", padding=3, pady=(0, 5))
        # stats = stats_section.body
        # ttk.Label(stats, textvariable=self.stats_var, wraplength=self.SIDEBAR_TEXT_WRAP, justify="left").pack(fill="x")

        self.plot_holder = ttk.Frame(self.content)
        self.plot_holder.pack(fill="both", expand=True)

        confirm_row = ttk.Frame(self.content)
        confirm_row.pack(fill="x", pady=(6, 0))
        self.confirm_button = ttk.Button(
            confirm_row,
            text="Confirm and Next ROI",
            command=self._confirm_current_roi,
        )
        self.confirm_button.pack(side="right")
        self._update_confirm_button_state()

        self._render_empty_canvas()

    def on_show(self) -> None:
        folder = self._default_input_folder()
        if folder and not self._input_dir_user_selected:
            self.input_dir_var.set(folder)
        if folder and not self._output_dir_user_selected:
            self.output_dir_var.set(folder)

    def _default_input_folder(self) -> str:
        if self.source_step is not None:
            folder = getattr(self.source_step, "output_sdb_dir", None) or getattr(self.source_step, "current_sdb_dir", None)
            if folder:
                return str(folder)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


    def _browse_input_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Select folder containing Step 3 _flat_LIGHT output",
            initialdir=self.input_dir_var.get() or None,
        )
        if folder:
            self.input_dir_var.set(folder)
            self._input_dir_user_selected = True
            if folder:
                self.input_dir_var.set(folder)
                self._input_dir_user_selected = True
                # 1. Construct the expected full file path
                light_hdr_path = os.path.join(folder, "_flat_LIGHT.hdr")
                # 2. Check if that specific file actually exists on the computer
                if os.path.exists(light_hdr_path):
                    self._load_path(light_hdr_path)
                else:
                    # Optional: You can add an else statement here to print a warning
                    # or show a tkinter messagebox if the file wasn't found.
                    print(f"Warning: _flat_LIGHT.hdr not found in {folder}")
    def _browse_output_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Select folder for Step 4 ISez outputs",
            initialdir=self.output_dir_var.get() or None,
        )
        if folder:
            self.output_dir_var.set(folder)
            self._output_dir_user_selected = True

    def _load_path(self, path: str | os.PathLike) -> None:
        try:
            volume = load_oct_volume(path)
        except Exception as exc:
            messagebox.showerror("Open error", str(exc))
            return

        self.volume = volume
        self.current_path = Path(path)
        self.current_stem = self.current_path.stem
        self.input_dir_var.set(str(self.current_path.parent))
        self._input_dir_user_selected = True
        if not self._output_dir_user_selected:
            self.output_dir_var.set(str(self.current_path.parent))

        slice_values = [str(idx) for idx in range(volume.shape[0])]
        self.slice_combo.configure(values=slice_values)
        self.slice_var.set("0")
        self.completed.clear()
        self.roi_clicks.clear()
        self.current_roi_idx = 0
        self.profile_clicks.clear()
        self._set_current_slice()
        self._refresh_roi_list()
        self._select_roi_in_list()
        self.image_label_var.set(
            f"{self.current_path.name}\nstack: {volume.shape[0]} slice(s), "
            f"{volume.shape[2]} x {volume.shape[1]}, {volume.dtype}"
        )
        self.status_var.set(f"Loaded {self.current_path}. Click start/end on the profile.")

    def _set_current_slice(self) -> None:
        if self.volume is None:
            return
        try:
            idx = int(self.slice_var.get())
        except ValueError:
            idx = 0
        idx = max(0, min(idx, self.volume.shape[0] - 1))
        self.slice_var.set(str(idx))
        self.image = np.asarray(self.volume[idx])
        self.profile_clicks.clear()
        self.roi_clicks.clear()
        self.start_var.set("")
        self.end_var.set("")
        self.completed.clear()
        self._load_current_roi_clicks()
        self._refresh_roi_list()
        self._render_current_roi()

    def _on_roi_selected(self, _event=None) -> None:
        if self._updating_roi_selection:
            return
        selection = self.roi_listbox.curselection()
        if not selection:
            return
        selected_idx = int(selection[0])
        if selected_idx == self.current_roi_idx:
            return
        self._remember_current_roi_clicks()
        self.current_roi_idx = selected_idx
        self._load_current_roi_clicks()
        self._render_current_roi()

    def _move_roi(self, delta: int) -> None:
        self._remember_current_roi_clicks()
        self.current_roi_idx = max(0, min(len(self.rois) - 1, self.current_roi_idx + delta))
        self._load_current_roi_clicks()
        self._select_roi_in_list()
        self._render_current_roi()

    def _current_roi_suffix(self) -> str:
        return self.rois[self.current_roi_idx].suffix

    def _remember_current_roi_clicks(self) -> None:
        suffix = self._current_roi_suffix()
        if self.profile_clicks:
            self.roi_clicks[suffix] = list(self.profile_clicks[:2])
        else:
            self.roi_clicks.pop(suffix, None)

    def _load_current_roi_clicks(self) -> None:
        suffix = self._current_roi_suffix()
        if suffix in self.roi_clicks:
            self.profile_clicks = list(self.roi_clicks[suffix][:2])
        elif suffix in self.completed:
            result = self.completed[suffix]
            self.profile_clicks = [float(result.start), float(result.end)]
        else:
            self.profile_clicks = []
        self._sync_entry_vars_from_clicks()

    def _sync_entry_vars_from_clicks(self) -> None:
        if self.profile_clicks:
            self.start_var.set(str(_clamp_profile_index(self.profile_clicks[0], self._current_profile_size())))
        else:
            self.start_var.set("")
        if len(self.profile_clicks) >= 2:
            self.end_var.set(str(_clamp_profile_index(self.profile_clicks[1], self._current_profile_size())))
        else:
            self.end_var.set("")

    def _select_roi_in_list(self) -> None:
        self._updating_roi_selection = True
        self.roi_listbox.selection_clear(0, tk.END)
        self.roi_listbox.selection_set(self.current_roi_idx)
        self.roi_listbox.see(self.current_roi_idx)
        self.after_idle(lambda: setattr(self, "_updating_roi_selection", False))

    def _refresh_roi_list(self) -> None:
        self.roi_listbox.delete(0, tk.END)
        for roi in self.rois:
            mark = "✅" if roi.suffix in self.completed else "❌"
            self.roi_listbox.insert(tk.END, f"{roi.suffix:>5}   {mark}")

    def _render_empty_canvas(self) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.figure = None
        self.ax_profile = None
        self.ax_isez = None
        self._current_profile = None
        self._update_confirm_button_state()
        if self.empty_placeholder is not None:
            self.empty_placeholder.destroy()
            self.empty_placeholder = None

        self.empty_placeholder = ttk.Label(
            self.plot_holder,
            text="Load a flattened OCT image",
            anchor="center",
        )
        self.empty_placeholder.pack(fill="both", expand=True)

    def _ensure_plot_canvas(self) -> None:
        if self.empty_placeholder is not None:
            self.empty_placeholder.destroy()
            self.empty_placeholder = None
        if self.canvas is not None and self.figure is not None:
            return

        self.figure = Figure(figsize=(11, 7), dpi=100)
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
        self.ax_isez = self.figure.add_subplot(grid[0, 0])
        self.ax_profile = self.figure.add_subplot(grid[1, 0])
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, hspace=0.28)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_holder)
        self.canvas.mpl_connect("button_press_event", self._on_profile_click)

    def _render_current_roi(self) -> None:
        if self.image is None:
            self._render_empty_canvas()
            return

        roi = self.rois[self.current_roi_idx]
        try:
            profile = intensity_profile(self.image, roi)
        except Exception as exc:
            self.status_var.set(str(exc))
            return
        self._current_profile = profile

        self._ensure_plot_canvas()

        xs = np.arange(1, profile.size + 1, dtype=np.float64)
        ymin = float(np.nanmin(profile))
        self.ax_profile.clear()
        self.ax_profile.plot(xs, profile, color="black", linewidth=1.2)
        self.ax_profile.set_xlim(0, 140)
        self.ax_profile.set_ylim(ymin, ymin + 20)
        self.ax_profile.set_title(f"ROI {roi.suffix}: click start, then end")
        self.ax_profile.set_xlabel("profile row")
        for idx, click in enumerate(self.profile_clicks[:2]):
            color = "#1f77b4" if idx == 0 else "#d62728"
            self.ax_profile.axvline(click, color=color, linewidth=1.2)

        self._draw_isez_preview(profile)

        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        if not canvas_widget.winfo_manager():
            canvas_widget.pack(fill="both", expand=True)
        self._update_profile_status(profile)
        self._update_confirm_button_state()

    def _draw_isez_preview(self, profile: np.ndarray) -> None:
        self.ax_isez.clear()
        self.ax_isez.set_title("Rotated / rescaled ISez")
        self.ax_isez.set_ylim(-20, 120)
        if len(self.profile_clicks) < 2:
            self.ax_isez.text(
                0.5,
                0.5,
                "Waiting for two profile clicks",
                ha="center",
                va="center",
                transform=self.ax_isez.transAxes,
            )
            self.ax_isez.set_axis_off()
            return

        start = _clamp_profile_index(self.profile_clicks[0], profile.size)
        end = _clamp_profile_index(self.profile_clicks[1], profile.size)
        if start > end:
            start, end = end, start
        try:
            adj_start, adj_end, _slope = adjust_isez_bounds(
                profile,
                start,
                end,
                right_click_x=max(self.profile_clicks[:2]),
            )
            data = rotated_rescaled_isez(profile, adj_start, adj_end)
        except Exception as exc:
            self.ax_isez.text(0.5, 0.5, str(exc), ha="center", va="center", transform=self.ax_isez.transAxes)
            self.ax_isez.set_axis_off()
            return

        center = (start + end) / 2.0
        self.ax_isez.plot(data["x"], data["y"], color="black", linewidth=1.2)
        self.ax_isez.plot(data["baseline_x"], data["baseline_y"], color="black", linewidth=1.0)
        self.ax_isez.set_xlim(center - 40, center + 40)
        self.ax_isez.set_ylim(-20, 120)
        self.ax_isez.set_xlabel("profile row")
        self.ax_isez.set_ylabel("rescaled intensity")

    def _on_profile_click(self, event) -> None:
        if event.inaxes is not self.ax_profile or event.xdata is None:
            return
        if self.image is None:
            return

        click = float(event.xdata)
        if len(self.profile_clicks) >= 2:
            self.profile_clicks = [click]
        else:
            self.profile_clicks.append(click)

        if len(self.profile_clicks) >= 1:
            self.start_var.set(str(_clamp_profile_index(self.profile_clicks[0], self._current_profile_size())))
        if len(self.profile_clicks) >= 2:
            self.end_var.set(str(_clamp_profile_index(self.profile_clicks[1], self._current_profile_size())))

        self._remember_current_roi_clicks()
        self._render_current_roi()

    def _current_profile_size(self) -> int:
        if self._current_profile is not None:
            return int(self._current_profile.size)
        if self.image is None:
            return 1
        try:
            return int(intensity_profile(self.image, self.rois[self.current_roi_idx]).size)
        except Exception:
            return 1

    def _apply_entry_clicks(self) -> None:
        try:
            start = float(self.start_var.get())
            end = float(self.end_var.get())
        except ValueError:
            messagebox.showerror("Profile Selection", "Enter numeric start and end values.")
            return
        self.profile_clicks = [start, end]
        self._remember_current_roi_clicks()
        self._render_current_roi()

    def _clear_clicks(self) -> None:
        self.profile_clicks.clear()
        self.roi_clicks.pop(self._current_roi_suffix(), None)
        self.start_var.set("")
        self.end_var.set("")
        self._render_current_roi()

    def _update_confirm_button_state(self) -> None:
        if self.confirm_button is None:
            return
        label = "Confirm ROI" if self.current_roi_idx >= len(self.rois) - 1 else "Confirm and Next ROI"
        self.confirm_button.configure(text=label)
        if len(self.profile_clicks) >= 2 and self.image is not None:
            self.confirm_button.state(["!disabled"])
        else:
            self.confirm_button.state(["disabled"])

    def _confirm_current_roi(self) -> None:
        self._save_current_roi(auto_advance=True)

    def _update_profile_status(self, profile: np.ndarray) -> None:
        roi = self.rois[self.current_roi_idx]
        if len(self.profile_clicks) < 2:
            self.profile_status_var.set(f"ROI {roi.suffix}: click start and end on the profile.")
            self.stats_var.set("")
            return

        start = _clamp_profile_index(self.profile_clicks[0], profile.size)
        end = _clamp_profile_index(self.profile_clicks[1], profile.size)
        if start > end:
            start, end = end, start
        try:
            adj_start, adj_end, slope = adjust_isez_bounds(profile, start, end, right_click_x=max(self.profile_clicks[:2]))
            data = rotated_rescaled_isez(profile, adj_start, adj_end)
        except Exception as exc:
            self.profile_status_var.set(str(exc))
            return

        self.profile_status_var.set(
            f"ROI {roi.suffix}: selected {start}-{end}; adjusted {adj_start}-{adj_end}. "
            "Review or edit, then confirm."
        )
        self.stats_var.set(
            f"center: {(start + end) / 2.0:.1f}\n"
            f"slope: {slope:.4f}\n"
            f"max index: {int(data['max_index'])}\n"
            f"scale min/max: {float(data['min_intensity']):.3f} / {float(data['max_intensity']):.3f}"
        )

    def _save_current_roi(self, *, auto_advance: bool) -> None:
        if self.image is None:
            messagebox.showwarning("Step 4", "Load an OCT image first.")
            return
        if len(self.profile_clicks) < 2:
            messagebox.showwarning("Step 4", "Click or enter both start and end profile positions.")
            return

        roi = self.rois[self.current_roi_idx]
        try:
            result = analyze_and_save_roi(
                self.image,
                roi,
                start_click=self.profile_clicks[0],
                end_click=self.profile_clicks[1],
            )
            self.completed[roi.suffix] = result
            self.roi_clicks[roi.suffix] = [float(result.start), float(result.end)]
        except Exception as exc:
            messagebox.showerror("Step 4", f"Could not save ROI {roi.suffix}.\n{exc}")
            return

        self._refresh_roi_list()
        self.status_var.set(f"Confirmed ROI {roi.suffix}.")

        if auto_advance and self.current_roi_idx < len(self.rois) - 1:
            self.current_roi_idx += 1
            self._load_current_roi_clicks()
            self._select_roi_in_list()
            self._render_current_roi()
        else:
            self._select_roi_in_list()

    def _build_stack_outputs(self) -> None:
        if not self.completed:
            messagebox.showwarning("Step 4", "Save at least one ROI before building stack outputs.")
            return

        ordered = [self.completed[roi.suffix] for roi in self.rois if roi.suffix in self.completed]
        outdir = Path(self.output_dir_var.get() or ".")
        try:
            outdir.mkdir(parents=True, exist_ok=True)

            # ImageJ macro equivalent:
            # open *_ISez_*.png -> run("Images to Stack") ->
            # saveAs("Tiff", "ROI_to_move_stck.tif").
            # We render the plot frames in memory and save only the final stack.
            isez_images = [make_isez_plot_image(result) for result in ordered]
            if isez_images:
                isez_images[0].save(outdir / "ROI_to_move_stck.tiff", save_all=True, append_images=isez_images[1:])

            # MATLAB produced one ROI-overlay JPG per ROI, then the ImageJ macro
            # stacked those overlays with the original image and ran a max
            # projection. We skip the intermediate JPGs and write only the final
            # MAX_Stack.tiff requested for this framework.
            roi_arrays = [
                roi_overlay_image(self.image, result.roi)
                for result in ordered
            ]
            if roi_arrays:
                max_projection = np.maximum.reduce(roi_arrays)
                Image.fromarray(max_projection).save(outdir / "MAX_Stack.tiff")

            # ImageJ macro equivalent:
            # setTool("wand"); doWand(512, 179); run("Measure") per slice.
            # The workbook is created here, after the stack button, not during
            # individual ROI confirmation.
            write_imagej_results_xlsx(
                [imagej_shape_measurements_from_frame(image) for image in isez_images],
                outdir / "Results.xlsx",
            )
        except Exception as exc:
            messagebox.showerror("Step 4", f"Could not build stack outputs.\n{exc}")
            return

        self.status_var.set(f"Built stack outputs in {outdir}.")
        messagebox.showinfo("Step 4", "Built MAX_Stack.tiff, Results.xlsx, and ROI_to_move_stck.tiff.")


def main() -> None:
    root = tk.Tk()
    root.title("AIDaS Step 4 - Analyze ISez")
    root.geometry("1200x800")
    frame = Step4Frame(root)
    frame.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
