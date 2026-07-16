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
import threading
import math
import os
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
import zipfile
from io import BytesIO
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from PIL import ImageDraw

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from aidas.utils.io_utils import read_analyze, read_tiff
from aidas.utils.filesystem import skipped_directories_warning, walk_accessible_directories
from aidas.utils.ui_utils import HoverToolTip, SidebarStepFrame, load_ui_icon


MATLAB_ROI_LOW = 300
MATLAB_ROI_HIGH = 450
MATLAB_ROI_TOP_LINE = 450
MATLAB_A_LIMIT = 1.0

# These are the Step 4 macro/MATLAB plot properties. The GT
# ``ROI_to_move_stck.tif`` frames are 969 x 513 px, the macro applies
# ``doWand(512, 179)``, then measures with
# ``fit shape invert redirect=None decimal=3``.
IMAGEJ_ISEZ_CANVAS_SIZE = (969, 513)
IMAGEJ_WAND_POINT = (512, 179)
IMAGEJ_RESULTS_DECIMALS = 3
MATLAB_ISEZ_X_HALF_WIDTH = 40.0
MATLAB_ISEZ_Y_LIMITS = (-20.0, 120.0)
MATLAB_AXIS_LINE_WIDTH = 1
MATLAB_DATA_LINE_WIDTH = 2
IMAGEJ_PLOT_BOX = (126.0, 38.0, 876.0, 456.0)
IMAGEJ_PLOT_BOX_ASPECT = (IMAGEJ_PLOT_BOX[3] - IMAGEJ_PLOT_BOX[1]) / (IMAGEJ_PLOT_BOX[2] - IMAGEJ_PLOT_BOX[0])

# Column order and text match the ImageJ Results table produced by the macro's
# "fit shape" measurement options and 3-decimal precision.
RESULTS_HEADERS = [" ", "Major", "Minor", "Angle", "Circ.", "AR", "Round", "Solidity"]
STEP4_OUTPUT_GROUPS = (
    ("MAX_Stack.tif",),
    ("ROI_to_move_stck.tif", "ROI_to_move_stck.tiff"),
    ("Results.xlsx", "Results_org.xlsx"),
)


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


def _jpeg_roundtrip_uint8(image: np.ndarray) -> np.ndarray:
    """Match MATLAB ``imwrite(...jpg)`` before ImageJ stacks the ROI overlays."""

    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="JPEG")
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L"), dtype=np.uint8)


def max_stack_projection_image(image: np.ndarray, rois: list[ISezROI]) -> np.ndarray:
    """Create ``MAX_Stack`` from the original image plus MATLAB ROI JPGs."""

    original = to_uint8_display(image)
    roi_arrays = [_jpeg_roundtrip_uint8(roi_overlay_image(image, roi)) for roi in rois]
    return np.maximum.reduce([original, *roi_arrays])



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
    final frame in memory so Step 4 only writes ``ROI_to_move_stck.tif`` when
    the user presses the stack button.
    """

    image = Image.new("RGB", IMAGEJ_ISEZ_CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = IMAGEJ_PLOT_BOX
    x_min = result.center - MATLAB_ISEZ_X_HALF_WIDTH
    x_max = result.center + MATLAB_ISEZ_X_HALF_WIDTH
    y_min, y_max = MATLAB_ISEZ_Y_LIMITS

    def to_pixel(px: float, py: float) -> tuple[float, float]:
        x = x0 + ((float(px) - x_min) / (x_max - x_min)) * (x1 - x0)
        y = y0 + ((y_max - float(py)) / (y_max - y_min)) * (y1 - y0)
        return x, y

    draw.rectangle((x0, y0, x1, y1), outline="black", width=MATLAB_AXIS_LINE_WIDTH)
    for tick in np.linspace(x_min, x_max, 5):
        tx, _ = to_pixel(tick, y_min)
        draw.line((tx, y1, tx, y1 + 5), fill="black", width=MATLAB_AXIS_LINE_WIDTH)
    for tick in [-20, 0, 50, 100, 120]:
        _, ty = to_pixel(x_min, tick)
        draw.line((x0 - 5, ty, x0, ty), fill="black", width=MATLAB_AXIS_LINE_WIDTH)

    curve = [to_pixel(px, py) for px, py in zip(result.normalized_x, result.normalized_y)]
    curve = [(x, y) for x, y in curve if x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2]
    if len(curve) >= 2:
        draw.line(curve, fill="black", width=MATLAB_DATA_LINE_WIDTH)
    baseline = [to_pixel(px, py) for px, py in zip(result.baseline_x, result.baseline_y)]
    if len(baseline) >= 2:
        draw.line(baseline, fill="black", width=MATLAB_DATA_LINE_WIDTH)
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


def _imagej_wand_mask(image: Image.Image, point: tuple[int, int] = IMAGEJ_WAND_POINT) -> np.ndarray:
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
    # SciPy is only needed when Step 4 performs this measurement. Importing it
    # here keeps application and debugger startup out of SciPy's large import
    # graph without changing the measurement implementation.
    from scipy import ndimage

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
        "Major": round(major, IMAGEJ_RESULTS_DECIMALS),
        "Minor": round(minor, IMAGEJ_RESULTS_DECIMALS),
        "Angle": round(angle, IMAGEJ_RESULTS_DECIMALS),
        "Circ.": round(circularity, IMAGEJ_RESULTS_DECIMALS),
        "AR": round(aspect_ratio, IMAGEJ_RESULTS_DECIMALS),
        "Round": round(roundness, IMAGEJ_RESULTS_DECIMALS),
        "Solidity": round(solidity, IMAGEJ_RESULTS_DECIMALS),
    }


class Step4BatchROITable(ttk.Frame):
    """Folder selection table for Step 4 batch ROI review."""

    COLUMNS = ("folder", "status", "outputs")

    def __init__(self, parent):
        super().__init__(parent)
        self.rows = []
        self._row_by_iid = {}
        self._checkbox_images = self._make_checkbox_images()
        self._tree_font = tkfont.nametofont("TkDefaultFont")
        self._heading_font = self._tree_font.copy()
        self._heading_font.configure(weight="bold")

        self._tree_style = "Step4BatchROI.Treeview"
        self._style = ttk.Style(self)
        try:
            self._style.configure(self._tree_style, indent=0)
        except tk.TclError:
            pass

        self.tree = ttk.Treeview(
            self,
            columns=self.COLUMNS,
            show=("tree", "headings"),
            selectmode="none",
            style=self._tree_style,
        )
        self.yscroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.yscroll.set, xscrollcommand=self.xscroll.set)

        self.tree.heading("#0", text="", image=self._checkbox_images["unchecked"], anchor="center", command=self._toggle_all_ready)
        self.tree.heading("folder", text="Folder")
        self.tree.heading("status", text="Status")
        self.tree.heading("outputs", text="Outputs")
        self.tree.column("#0", width=40, minwidth=40, stretch=False, anchor="center")
        self.tree.column("folder", width=520, minwidth=220, stretch=False, anchor="w")
        self.tree.column("status", width=220, minwidth=120, stretch=False, anchor="w")
        self.tree.column("outputs", width=120, minwidth=80, stretch=False, anchor="center")
        self.tree.tag_configure("locked", foreground="#6b7280")

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.yscroll.grid(row=0, column=1, sticky="ns")
        self.xscroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.tree.bind("<Button-1>", self._on_click, add="+")
        self.tree.bind("<Configure>", self._on_tree_configure, add="+")

    def _make_checkbox_images(self):
        images = {
            "checked": tk.PhotoImage(width=16, height=16),
            "unchecked": tk.PhotoImage(width=16, height=16),
            "locked": tk.PhotoImage(width=16, height=16),
        }
        for image in images.values():
            image.put("#ffffff", to=(0, 0, 16, 16))
            image.put("#6b7280", to=(2, 2, 14, 3))
            image.put("#6b7280", to=(2, 13, 14, 14))
            image.put("#6b7280", to=(2, 2, 3, 14))
            image.put("#6b7280", to=(13, 2, 14, 14))
        checked = images["checked"]
        for x, y in ((4, 8), (5, 9), (6, 10), (7, 9), (8, 8), (9, 7), (10, 6), (11, 5)):
            checked.put("#111827", to=(x, y, x + 1, y + 1))
            checked.put("#111827", to=(x, y + 1, x + 1, y + 2))
        locked = images["locked"]
        locked.put("#e5e7eb", to=(3, 3, 13, 13))
        locked.put("#9ca3af", to=(5, 7, 11, 9))
        return images

    def set_rows(self, rows):
        self.rows = list(rows or [])
        self._row_by_iid = {}
        self.tree.delete(*self.tree.get_children(""))
        if not self.rows:
            self.tree.insert("", "end", values=("No _flat_LIGHT folders were found.", "", ""), tags=("locked",))
            self._refresh_header_checkbox()
            return
        for index, row in enumerate(self.rows):
            iid = str(index)
            self._row_by_iid[iid] = row
            self.tree.insert(
                "",
                "end",
                iid=iid,
                image=self._image_for_row(row),
                values=self._values_for_row(row),
                tags=("locked",) if row.get("locked") else (),
            )
        self._fit_columns_to_content()
        self._refresh_header_checkbox()

    def _image_for_row(self, row):
        if row.get("locked"):
            return self._checkbox_images["locked"]
        return self._checkbox_images["checked"] if row.get("include") else self._checkbox_images["unchecked"]

    @staticmethod
    def _values_for_row(row):
        values = row.get("values") or {}
        return values.get("folder", ""), values.get("status", ""), values.get("outputs", "")

    def _measure_text(self, text, *, heading=False, padding=18):
        font = self._heading_font if heading else self._tree_font
        return int(font.measure(str(text or ""))) + int(padding)

    def _fit_columns_to_content(self):
        widths = {
            "folder": self._measure_text("Folder", heading=True),
            "status": self._measure_text("Status", heading=True),
            "outputs": self._measure_text("Outputs", heading=True),
        }
        for row in self.rows:
            folder, status, outputs = self._values_for_row(row)
            widths["folder"] = max(widths["folder"], self._measure_text(folder))
            widths["status"] = max(widths["status"], self._measure_text(status))
            widths["outputs"] = max(widths["outputs"], self._measure_text(outputs))
        self.tree.column("folder", width=max(220, widths["folder"]))
        self.tree.column("status", width=max(120, widths["status"]))
        self.tree.column("outputs", width=max(80, widths["outputs"]))
        self._expand_folder_to_view()

    def _on_tree_configure(self, _event=None):
        self._expand_folder_to_view()

    def _expand_folder_to_view(self):
        try:
            view_width = max(1, int(self.tree.winfo_width()))
            checkbox_width = int(self.tree.column("#0", "width"))
            folder_width = int(self.tree.column("folder", "width"))
            status_width = int(self.tree.column("status", "width"))
            outputs_width = int(self.tree.column("outputs", "width"))
        except tk.TclError:
            return
        desired = max(220, view_width - checkbox_width - status_width - outputs_width - 2)
        if desired > folder_width:
            try:
                self.tree.column("folder", width=desired)
            except tk.TclError:
                pass

    def _refresh_row(self, iid, row):
        try:
            self.tree.item(iid, image=self._image_for_row(row), values=self._values_for_row(row))
        except tk.TclError:
            pass

    def _refresh_header_checkbox(self):
        ready_rows = [row for row in self.rows if not row.get("locked")]
        image = self._checkbox_images["checked"] if ready_rows and all(row.get("include") for row in ready_rows) else self._checkbox_images["unchecked"]
        try:
            self.tree.heading("#0", image=image)
        except tk.TclError:
            pass

    def _on_click(self, event):
        if self.tree.identify_region(event.x, event.y) not in {"cell", "tree"}:
            return None
        if self.tree.identify_column(event.x) != "#0":
            return None
        iid = self.tree.identify_row(event.y)
        row = self._row_by_iid.get(iid)
        if not row or row.get("locked"):
            return "break"
        row["include"] = not bool(row.get("include"))
        self._refresh_row(iid, row)
        self._refresh_header_checkbox()
        return "break"

    def _toggle_all_ready(self):
        ready_rows = [row for row in self.rows if not row.get("locked")]
        if not ready_rows:
            return
        include = not all(row.get("include") for row in ready_rows)
        for iid, row in self._row_by_iid.items():
            if row.get("locked"):
                continue
            row["include"] = include
            self._refresh_row(iid, row)
        self._refresh_header_checkbox()

    def selected_rows(self):
        return [row for row in self.rows if row.get("include") and not row.get("locked")]


class Step4BatchROISelectionPanel(ttk.Frame):
    """Embedded panel for selecting Step 4 ROI folders to process."""

    def __init__(self, step_frame, parent, root_dir):
        super().__init__(parent)
        self.step_frame = step_frame
        self.root_dir = Path(root_dir)
        self.table = None
        self.rows = []
        self._build_ui()
        self._start_scan()

    def _build_ui(self):
        wrapper = ttk.Frame(self, padding=12)
        wrapper.pack(fill="both", expand=True)
        ttk.Label(wrapper, text="Batch ROI", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            wrapper,
            text=(
                "AIDaS will search the selected folder and subfolders for _flat_LIGHT.img/_flat_LIGHT.hdr. "
                "Folders with existing Step 4 outputs are locked and skipped."
            ),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        top = ttk.Frame(wrapper)
        top.pack(fill="x", pady=(0, 8))
        self.summary_var = tk.StringVar(value=f"Scanning: {self.root_dir}")
        ttk.Label(top, textvariable=self.summary_var, wraplength=760, justify="left").pack(side="left", fill="x", expand=True)
        self.more_label = ttk.Label(top, text="", foreground="#0066cc", cursor="hand2")
        self.more_label.pack(side="right", padx=(8, 0))
        self.more_tooltip = HoverToolTip(self.more_label, "")

        self.table_host = ttk.Frame(wrapper)
        self.table_host.pack(fill="both", expand=True)
        self.scan_label = ttk.Label(self.table_host, text="Scanning folders...", anchor="center", justify="center")
        self.scan_label.pack(fill="both", expand=True)

        run_box = ttk.Frame(wrapper)
        run_box.pack(fill="x", pady=(10, 0))
        self.process_button = ttk.Button(run_box, text="Process Selected", command=self._process_selected)
        self.process_button.pack(side="right")
        self.process_button.state(["disabled"])
        ttk.Button(run_box, text="Cancel", command=self._cancel).pack(side="left")

    def _start_scan(self):
        self.step_frame.status_var.set(f"Scanning Step 4 ROI folders under {self.root_dir}...")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            rows, scanned, skipped, access_errors = self.step_frame._scan_batch_roi_folders(self.root_dir)
        except Exception as exc:
            self.after(0, lambda exc=exc: self._scan_failed(exc))
            return
        self.after(0, lambda: self._scan_done(rows, scanned, skipped, access_errors))

    def _scan_failed(self, exc):
        if not self.winfo_exists():
            return
        self.summary_var.set("Scan failed. Move the mouse over More for details.")
        self.more_label.configure(text="More")
        self.more_tooltip.text = f"Could not scan folders.\n{exc}"
        self.step_frame.status_var.set("Batch ROI scan failed.")
        self.process_button.state(["disabled"])

    def _show_results_table(self, rows):
        for child in self.table_host.winfo_children():
            try:
                child.destroy()
            except tk.TclError:
                pass
        table = Step4BatchROITable(self.table_host)
        table.set_rows(rows)
        table.pack(fill="both", expand=True)
        self.table = table

    def _scan_done(self, rows, scanned, skipped, access_errors):
        if not self.winfo_exists():
            return
        self.rows = rows
        for row in rows:
            try:
                folder_text = os.path.relpath(row["folder"], self.root_dir)
                if folder_text == ".":
                    folder_text = str(self.root_dir)
            except ValueError:
                folder_text = str(row["folder"])
            row["values"] = {
                "folder": folder_text,
                "status": row.get("status", ""),
                "outputs": row.get("outputs", ""),
            }
        self._show_results_table(rows)

        ready = sum(1 for row in rows if not row.get("locked"))
        complete = sum(1 for row in rows if row.get("locked"))
        summary = (
            f"Scanned {scanned} folder(s). Found {ready} ready folder(s), "
            f"{complete} already complete, {skipped} without _flat_LIGHT. "
            f"{len(access_errors)} inaccessible folder(s) skipped."
        )
        self.summary_var.set(summary)
        self.more_label.configure(text="More" if access_errors else "")
        self.more_tooltip.text = skipped_directories_warning(access_errors) if access_errors else ""
        self.step_frame.status_var.set("Batch ROI scan complete. Select folders to process.")
        if ready:
            self.process_button.state(["!disabled"])
        else:
            self.process_button.state(["disabled"])
    def _process_selected(self):
        if self.table is None:
            return
        rows = self.table.selected_rows()
        if not rows:
            messagebox.showwarning("Batch ROI", "Select at least one ready folder.", parent=self)
            return
        self.step_frame._start_batch_roi_from_rows(rows)

    def _cancel(self):
        self.step_frame._close_batch_roi_panel(restore_previous=True)


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
        self._flashing_updated_rois: set[str] = set()
        self._roi_update_animation_jobs: dict[str, str] = {}
        self.roi_clicks: dict[str, list[float]] = {}
        self.profile_clicks: list[float] = []
        self.figure = None
        self.canvas = None
        self.empty_placeholder = None
        self.ax_profile = None
        self.ax_roi_grid = None
        self._current_profile = None
        self._updating_roi_selection = False
        self._input_dir_user_selected = False
        self._output_dir_user_selected = False
        self.batch_roi_root = None
        self.batch_roi_paths: list[Path] = []
        self.batch_roi_index = -1
        self.batch_roi_skipped = 0
        self.batch_roi_panel = None
        self.batch_roi_notebook = None
        self.batch_roi_tab_states: dict[str, dict] = {}
        self._active_batch_roi_tab = None
        self.plot_container = None

        self.input_dir_var = tk.StringVar(value=self._default_input_folder())
        self.output_dir_var = tk.StringVar(value=self._default_input_folder())
        self.image_label_var = tk.StringVar(value="No image loaded")
        self.status_var = tk.StringVar(value="Ready - load a flattened OCT image to begin.")
        self.profile_status_var = tk.StringVar(value="Click start and end on the profile canvas.")
        self.stats_var = tk.StringVar(value="")
        self.start_var = tk.StringVar(value="")
        self.end_var = tk.StringVar(value="")
        self.confirm_button = None
        self.build_stacks_button = None
        self.roi_table = None
        self._auto_saving_roi = False
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

        self.batch_roi_button = ttk.Button(
            source,
            text="Batch ROI...",
            image=load_ui_icon(self, "glyphs-poly--folder.png"),
            compound="left",
            command=self._browse_batch_roi_root,
        )
        self.batch_roi_button.pack(fill="x", pady=(6, 8))

        ttk.Label(
            source,
            textvariable=self.image_label_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            foreground="gray",
            justify="left",
        ).pack(fill="x", pady=(6, 0))

        roi_section = self.add_sidebar_section("ROIs", fill="both", expand=True, pady=(0, 5))
        roi_box = roi_section.body
        list_frame = ttk.Frame(roi_box)
        list_frame.pack(fill="both", expand=True)
        columns = ("roi", *RESULTS_HEADERS[1:])
        self.roi_table = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=10,
        )
        roi_yscroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.roi_table.yview)
        roi_xscroll = ttk.Scrollbar(list_frame, orient="horizontal", command=self.roi_table.xview)
        self.roi_table.configure(yscrollcommand=roi_yscroll.set, xscrollcommand=roi_xscroll.set)
        self.roi_table.heading("roi", text="ROI")
        self.roi_table.column("roi", width=44, minwidth=40, stretch=False, anchor="center")
        for column in RESULTS_HEADERS[1:]:
            self.roi_table.heading(column, text=column)
            self.roi_table.column(column, width=70, minwidth=58, stretch=False, anchor="center")
        self.roi_table.grid(row=0, column=0, sticky="nsew")
        roi_yscroll.grid(row=0, column=1, sticky="ns")
        roi_xscroll.grid(row=1, column=0, sticky="ew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.roi_table.bind("<<TreeviewSelect>>", self._on_roi_selected)

        nav = ttk.Frame(roi_box)
        nav.pack(fill="x", pady=(6, 0))
        ttk.Button(nav, text="< Prev", command=lambda: self._move_roi(-1)).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(nav, text="Next >", command=lambda: self._move_roi(1)).pack(side="right", expand=True, fill="x", padx=(2, 0))

        # Create a SINGLE frame for the horizontal controls
        control_row = ttk.Frame(roi_box)
        # Packs the row at the top (default) of the available space
        control_row.pack(fill="x", pady=5) 

        # 1. Start Label & Entry
        ttk.Label(control_row, text="Start").pack(side="left", padx=(0, 2))
        self.start_entry = ttk.Entry(control_row, textvariable=self.start_var, width=8)
        self.start_entry.pack(side="left", padx=(0, 10))

        # 2. End Label & Entry
        ttk.Label(control_row, text="End").pack(side="left", padx=(0, 2))
        self.end_entry = ttk.Entry(control_row, textvariable=self.end_var, width=8)
        self.end_entry.pack(side="left", padx=(0, 10))

        # 3. Apply Button
        self.apply_icon = load_ui_icon(self, "streamline-stickies-color--validation-1-duo.png")
        self.apply_button = ttk.Button(control_row, image=self.apply_icon, command=self._apply_entry_clicks)
        self.apply_button.pack(side="left", expand=False) 
        self.clear_icon = load_ui_icon(self, "solar--eraser-bold-duotone.png")
        self.clear_button = ttk.Button(control_row, image=self.clear_icon, command=self._clear_clicks)
        self.clear_button.pack(side="left", padx=(4, 0), expand=False)
        for entry in (self.start_entry, self.end_entry):
            entry.bind("<Return>", self._apply_entry_clicks)
            entry.bind("<KP_Enter>", self._apply_entry_clicks)
            entry.bind("<FocusOut>", self._apply_entry_clicks_if_complete)

        # 4. Build Stacks Button
        # Change the parent from 'control_row' to 'roi_box'
        # Use side="bottom" to anchor it to the bottom of the roi_box frame
        self.build_stacks_icon = load_ui_icon(self, "tabler--stack-push.png")
        self.build_stacks_button = ttk.Button(
            roi_box,
            image=self.build_stacks_icon,
            text="Build Stack",
            compound="left",
            command=self._build_stack_outputs,
        )
        self.build_stacks_button.pack(side="bottom", fill="x", pady=(6, 2))
        # stats_section = self.add_sidebar_section("Stats", padding=3, pady=(0, 5))
        # stats = stats_section.body
        # ttk.Label(stats, textvariable=self.stats_var, wraplength=self.SIDEBAR_TEXT_WRAP, justify="left").pack(fill="x")

        self.plot_container = ttk.Frame(self.content)
        self.plot_container.pack(fill="both", expand=True)
        self.plot_holder = self.plot_container

        self.confirm_row = ttk.Frame(self.content)
        self.confirm_row.pack(fill="x", pady=(6, 0))
        self._update_confirm_button_state()
        self._update_build_stack_button_state()

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

    def _hide_canvas_actions(self) -> None:
        row = getattr(self, "confirm_row", None)
        if row is not None and row.winfo_manager():
            row.pack_forget()

    def _show_canvas_actions(self) -> None:
        row = getattr(self, "confirm_row", None)
        if row is not None and not row.winfo_manager():
            row.pack(fill="x", pady=(6, 0))

    def _set_batch_folder_label(self, folder: Path | None) -> None:
        if folder is None:
            self.image_label_var.set("No image loaded")
            return
        self.image_label_var.set(f"Parent: {folder.parent.name}\nCurrent: {folder.name}")


    @staticmethod
    def _step4_outputs_complete(folder: str | os.PathLike) -> bool:
        folder = Path(folder)
        return all(any((folder / name).is_file() for name in names) for names in STEP4_OUTPUT_GROUPS)

    @staticmethod
    def _flat_light_path_for_folder(folder: Path) -> Path | None:
        img_path = folder / "_flat_LIGHT.img"
        hdr_path = folder / "_flat_LIGHT.hdr"
        if img_path.is_file():
            return hdr_path if hdr_path.is_file() else img_path
        if hdr_path.is_file():
            return hdr_path
        return None

    @staticmethod
    def _step4_output_count(folder: str | os.PathLike) -> int:
        folder = Path(folder)
        return sum(any((folder / name).is_file() for name in names) for names in STEP4_OUTPUT_GROUPS)

    def _scan_batch_roi_folders(self, root_dir: str | os.PathLike) -> tuple[list[dict], int, int, list]:
        root = Path(root_dir)
        candidate_dirs, access_errors = walk_accessible_directories(root)

        rows: list[dict] = []
        skipped_without_flat_light = 0
        for folder in sorted(candidate_dirs, key=lambda path: str(path).lower()):
            try:
                flat_light = self._flat_light_path_for_folder(folder)
            except OSError as exc:
                access_errors.append((folder, str(exc)))
                continue
            if flat_light is None:
                skipped_without_flat_light += 1
                continue
            try:
                output_count = self._step4_output_count(folder)
                outputs_complete = self._step4_outputs_complete(folder)
            except OSError as exc:
                access_errors.append((folder, str(exc)))
                continue
            if outputs_complete:
                rows.append(
                    {
                        "folder": folder,
                        "flat_light": flat_light,
                        "include": False,
                        "locked": True,
                        "status": "Already complete",
                        "outputs": "3/3",
                    }
                )
                continue
            rows.append(
                {
                    "folder": folder,
                    "flat_light": flat_light,
                    "include": True,
                    "locked": False,
                    "status": "Ready",
                    "outputs": f"{output_count}/3",
                }
            )
        return rows, len(candidate_dirs), skipped_without_flat_light, access_errors

    def _browse_batch_roi_root(self) -> None:
        folder = filedialog.askdirectory(
            title="Select root folder for batch Step 4 ROI",
            initialdir=str(Path.home() / "Desktop"),
        )
        if not folder:
            return

        self._open_batch_roi_panel(Path(folder))

    def _open_batch_roi_panel(self, root_dir: Path) -> None:
        self._close_batch_roi_panel(restore_previous=False)
        self._hide_canvas_actions()
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.batch_roi_notebook is not None:
            try:
                self.batch_roi_notebook.destroy()
            except tk.TclError:
                pass
            self.batch_roi_notebook = None
        for child in self.plot_container.winfo_children():
            try:
                child.destroy()
            except tk.TclError:
                pass
        self.plot_holder = self.plot_container
        self.figure = None
        self.ax_profile = None
        self._current_profile = None
        if self.empty_placeholder is not None:
            self.empty_placeholder.destroy()
            self.empty_placeholder = None

        self.batch_roi_root = Path(root_dir)
        self.input_dir_var.set(str(self.batch_roi_root))
        self._input_dir_user_selected = True
        self._set_batch_folder_label(self.batch_roi_root)
        self.batch_roi_panel = Step4BatchROISelectionPanel(self, self.plot_container, self.batch_roi_root)
        self.batch_roi_panel.pack(fill="both", expand=True)

    def _close_batch_roi_panel(self, *, restore_previous: bool) -> None:
        panel = self.batch_roi_panel
        self.batch_roi_panel = None
        if panel is not None:
            try:
                panel.destroy()
            except tk.TclError:
                pass
        if restore_previous:
            self._show_canvas_actions()
            if self.image is not None:
                self._render_current_roi()
            else:
                self._render_empty_canvas()

    def _start_batch_roi_from_rows(self, rows: list[dict]) -> None:
        paths = [Path(row["flat_light"]) for row in rows if row.get("flat_light")]
        if not paths:
            messagebox.showwarning("Batch ROI", "Select at least one ready folder.")
            return
        self._close_batch_roi_panel(restore_previous=False)
        self.batch_roi_paths = paths
        self.batch_roi_index = -1
        self.batch_roi_skipped = 0
        self._open_batch_roi_tabs(paths)

    def _open_batch_roi_tabs(self, paths: list[Path]) -> None:
        self._show_canvas_actions()
        for child in self.plot_container.winfo_children():
            try:
                child.destroy()
            except tk.TclError:
                pass
        self.canvas = None
        self.figure = None
        self.ax_profile = None
        self.ax_roi_grid = None
        self.empty_placeholder = None
        self._current_profile = None
        self.batch_roi_tab_states = {}
        self._active_batch_roi_tab = None

        self._configure_batch_roi_tab_style()
        notebook = ttk.Notebook(self.plot_container, style="Step4Batch.TNotebook")
        notebook.pack(fill="both", expand=True)
        notebook.bind("<Button-1>", self._on_batch_roi_notebook_click, add="+")
        notebook.bind("<Motion>", self._on_batch_roi_notebook_motion, add="+")
        notebook.bind("<<NotebookTabChanged>>", self._on_batch_roi_tab_changed, add="+")
        notebook.bind("<Configure>", self._on_batch_roi_notebook_configure, add="+")
        self.batch_roi_notebook = notebook

        for idx, path in enumerate(paths, start=1):
            frame = ttk.Frame(notebook)
            tab_key = str(frame)
            folder = path.parent
            self.batch_roi_tab_states[tab_key] = {
                "path": path,
                "folder": folder,
                "base_label": f"{idx}. {folder.name}",
                "loaded": False,
                "completed": {},
                "roi_clicks": {},
                "current_roi_idx": 0,
                "profile_clicks": [],
                "complete": False,
            }
            notebook.add(frame, text=self._batch_roi_tab_text(self.batch_roi_tab_states[tab_key]))

        first_tab = notebook.tabs()[0] if notebook.tabs() else None
        if first_tab:
            self._activate_batch_roi_tab(notebook.nametowidget(first_tab))

    def _configure_batch_roi_tab_style(self) -> None:
        try:
            style = ttk.Style(self)
            style.configure("Step4Batch.TNotebook", background="#f0f0f0", borderwidth=1)
        except tk.TclError:
            pass

    def _batch_roi_tab_name_limit(self) -> int:
        notebook = self.batch_roi_notebook
        if notebook is None:
            return 18
        try:
            tab_count = max(1, len(notebook.tabs()))
            width = max(260, notebook.winfo_width())
        except tk.TclError:
            return 18
        per_tab = max(70, width // tab_count)
        return max(6, min(18, (per_tab - 54) // 7))

    @staticmethod
    def _compact_batch_roi_name(name: str, limit: int) -> str:
        name = str(name or "Folder")
        if len(name) <= limit:
            return name
        if limit <= 3:
            return name[:limit]
        return f"{name[: limit - 3]}..."

    def _batch_roi_tab_text(self, state: dict, done: int | None = None, *, active: bool = False) -> str:
        total = len(self.rois)
        if done is None:
            done = len(state.get("completed") or {})
        folder = state.get("folder")
        raw_label = state.get("base_label") or (folder.name if folder else "Folder")
        if ". " in raw_label:
            prefix, name = raw_label.split(". ", 1)
            tab_name = name if active else self._compact_batch_roi_name(name, self._batch_roi_tab_name_limit())
            label = f"{prefix}. {tab_name}"
        else:
            label = raw_label if active else self._compact_batch_roi_name(raw_label, self._batch_roi_tab_name_limit())
        done_prefix = "[Done] " if state.get("complete") or done >= total else ""
        return f"{done_prefix}{label} ({done}/{total})    ×"

    def _refresh_batch_roi_tab_labels(self) -> None:
        notebook = self.batch_roi_notebook
        if notebook is None:
            return
        for tab_id in notebook.tabs():
            try:
                tab_key = str(notebook.nametowidget(tab_id))
            except tk.TclError:
                continue
            state = self.batch_roi_tab_states.get(tab_key)
            if state is None:
                continue
            done = len(state.get("completed") or {})
            active = tab_key == self._active_batch_roi_tab
            try:
                notebook.tab(tab_id, text=self._batch_roi_tab_text(state, done, active=active))
            except tk.TclError:
                pass

    def _on_batch_roi_notebook_configure(self, _event) -> None:
        self._refresh_batch_roi_tab_labels()

    def _sync_active_batch_roi_state(self) -> None:
        tab_key = self._active_batch_roi_tab
        if not tab_key:
            return
        state = self.batch_roi_tab_states.get(tab_key)
        if state is None:
            return
        state["completed"] = dict(self.completed)
        state["roi_clicks"] = {key: list(value) for key, value in self.roi_clicks.items()}
        state["current_roi_idx"] = int(self.current_roi_idx)
        state["profile_clicks"] = list(self.profile_clicks[:2])
        state["volume"] = self.volume
        state["image"] = self.image
        state["current_path"] = self.current_path
        state["current_stem"] = self.current_stem
        state["canvas"] = self.canvas
        state["figure"] = self.figure
        state["ax_profile"] = self.ax_profile
        state["ax_roi_grid"] = self.ax_roi_grid
        state["empty_placeholder"] = self.empty_placeholder
        state["current_profile"] = self._current_profile

    def _activate_batch_roi_tab(self, tab) -> None:
        self._cancel_roi_update_animations(redraw=True)
        self._sync_active_batch_roi_state()
        tab_key = str(tab)
        state = self.batch_roi_tab_states.get(tab_key)
        if state is None:
            return
        self._active_batch_roi_tab = tab_key
        self._refresh_batch_roi_tab_labels()
        self.plot_holder = tab
        self._set_batch_folder_label(state.get("folder"))
        if state.get("loaded") and self._restore_batch_roi_tab_from_cache(state):
            return
        self.canvas = None
        self.figure = None
        self.ax_profile = None
        self.ax_roi_grid = None
        self.empty_placeholder = None
        self._current_profile = None
        self._load_path(state["path"], restore_state=state)

    def _restore_batch_roi_tab_from_cache(self, state: dict) -> bool:
        canvas = state.get("canvas")
        figure = state.get("figure")
        if canvas is None or figure is None:
            return False
        try:
            widget = canvas.get_tk_widget()
            widget.winfo_exists()
        except tk.TclError:
            return False

        self.volume = state.get("volume")
        self.image = state.get("image")
        self.current_path = state.get("current_path") or Path(state["path"])
        self.current_stem = state.get("current_stem") or self.current_path.stem
        self.canvas = canvas
        self.figure = figure
        self.ax_profile = state.get("ax_profile")
        self.ax_roi_grid = state.get("ax_roi_grid")
        self.empty_placeholder = state.get("empty_placeholder")
        self._current_profile = state.get("current_profile")
        self.completed = dict(state.get("completed") or {})
        self.roi_clicks = {key: list(value) for key, value in (state.get("roi_clicks") or {}).items()}
        self.current_roi_idx = int(state.get("current_roi_idx") or 0)
        self.current_roi_idx = max(0, min(len(self.rois) - 1, self.current_roi_idx))
        self.profile_clicks = list((state.get("profile_clicks") or [])[:2])

        self.input_dir_var.set(str(self.current_path.parent))
        if not self._output_dir_user_selected:
            self.output_dir_var.set(str(self.current_path.parent))
        self._set_batch_folder_label(state.get("folder"))
        self._sync_entry_vars_from_clicks()
        self._refresh_roi_list()
        self._select_roi_in_list()
        self._update_profile_status(self._current_profile)
        self._update_confirm_button_state()
        self.status_var.set(f"Loaded {self.current_path}. Click start/end on the profile.")
        return True

    def _on_batch_roi_tab_changed(self, event) -> None:
        notebook = event.widget
        try:
            selected = notebook.select()
        except tk.TclError:
            return
        if selected:
            self._activate_batch_roi_tab(notebook.nametowidget(selected))

    @staticmethod
    def _batch_roi_tab_bounds(notebook, index: int, y: int):
        try:
            bbox = notebook.bbox(index)
        except tk.TclError:
            bbox = None
        if bbox and bbox[2] > 0:
            x0, _y0, width, _height = bbox
            return x0, x0 + width

        try:
            width = max(1, notebook.winfo_width())
        except tk.TclError:
            return None
        first_x = None
        last_x = None
        probe_y = max(1, int(y))
        for probe_x in range(width):
            try:
                probe_index = notebook.index(f"@{probe_x},{probe_y}")
            except tk.TclError:
                if first_x is not None:
                    break
                continue
            if probe_index != index:
                if first_x is not None:
                    break
                continue
            if first_x is None:
                first_x = probe_x
            last_x = probe_x
        if first_x is None or last_x is None:
            return None
        return first_x, last_x + 1

    @classmethod
    def _batch_roi_close_tab_at(cls, notebook, x: int, y: int):
        try:
            index = notebook.index(f"@{x},{y}")
        except tk.TclError:
            return None
        bounds = cls._batch_roi_tab_bounds(notebook, index, y)
        if not bounds:
            return None
        left, right = bounds
        close_width = min(24, max(14, right - left))
        if right - close_width <= x <= right:
            try:
                return notebook.nametowidget(notebook.tabs()[index])
            except (tk.TclError, IndexError):
                return None
        return None

    def _on_batch_roi_notebook_click(self, event):
        notebook = event.widget
        tab = self._batch_roi_close_tab_at(notebook, event.x, event.y)
        if tab is None:
            return None
        self._close_batch_roi_tab(notebook, tab)
        return "break"

    def _on_batch_roi_notebook_motion(self, event):
        try:
            event.widget.configure(cursor="hand2" if self._batch_roi_close_tab_at(event.widget, event.x, event.y) is not None else "")
        except tk.TclError:
            pass

    def _close_batch_roi_tab(self, notebook, tab) -> None:
        tab_key = str(tab)
        if tab_key == self._active_batch_roi_tab:
            self._cancel_roi_update_animations(redraw=False)
            self._sync_active_batch_roi_state()
        self.batch_roi_tab_states.pop(tab_key, None)
        try:
            notebook.forget(tab)
        except tk.TclError:
            return
        try:
            tab.destroy()
        except tk.TclError:
            pass
        if tab_key == self._active_batch_roi_tab:
            self._active_batch_roi_tab = None
            if self.canvas is not None:
                try:
                    self.canvas.get_tk_widget().destroy()
                except tk.TclError:
                    pass
                self.canvas = None
            tabs = notebook.tabs()
            if tabs:
                self._activate_batch_roi_tab(notebook.nametowidget(tabs[0]))
            else:
                self.batch_roi_notebook = None
                self.plot_holder = self.plot_container
                self._render_empty_canvas()

    def _update_active_batch_roi_tab_progress(self) -> None:
        tab_key = self._active_batch_roi_tab
        notebook = self.batch_roi_notebook
        if not tab_key or notebook is None:
            return
        state = self.batch_roi_tab_states.get(tab_key)
        if state is None:
            return
        done = len(state.get("completed") or self.completed)
        try:
            notebook.tab(tab_key, text=self._batch_roi_tab_text(state, done, active=True))
        except tk.TclError:
            pass

    def _mark_active_batch_roi_complete(self) -> None:
        tab_key = self._active_batch_roi_tab
        if not tab_key:
            return
        state = self.batch_roi_tab_states.get(tab_key)
        if state is None:
            return
        state["complete"] = True
        self._sync_active_batch_roi_state()
        self._update_active_batch_roi_tab_progress()

    def _select_next_incomplete_batch_roi_tab(self) -> bool:
        notebook = self.batch_roi_notebook
        if notebook is None:
            return False
        tabs = list(notebook.tabs())
        if not tabs:
            return False
        try:
            current = notebook.select()
            start_index = tabs.index(current) if current in tabs else -1
        except (tk.TclError, ValueError):
            start_index = -1

        for offset in range(1, len(tabs) + 1):
            tab_id = tabs[(start_index + offset) % len(tabs)]
            state = self.batch_roi_tab_states.get(str(notebook.nametowidget(tab_id)))
            if state is not None and not state.get("complete"):
                notebook.select(tab_id)
                return True
        return False

    def _load_next_batch_roi(self) -> None:
        next_index = self.batch_roi_index + 1
        if next_index >= len(self.batch_roi_paths):
            self.status_var.set(
                f"Batch ROI complete. Processed {len(self.batch_roi_paths)} folder(s); "
                f"skipped {self.batch_roi_skipped} already complete."
            )
            messagebox.showinfo("Batch ROI", "All incomplete Step 4 folders in this batch are done.")
            return

        self.batch_roi_index = next_index
        path = self.batch_roi_paths[self.batch_roi_index]
        self.status_var.set(
            f"Batch ROI {self.batch_roi_index + 1}/{len(self.batch_roi_paths)}: {path.parent}"
        )
        self._load_path(path)

    def _load_path(self, path: str | os.PathLike, *, restore_state: dict | None = None) -> None:
        self._cancel_roi_update_animations(redraw=False)
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

        self.completed.clear()
        self.roi_clicks.clear()
        self.current_roi_idx = 0
        self.profile_clicks.clear()
        self._set_slice_zero()
        self._refresh_roi_list()
        self._select_roi_in_list()
        if restore_state is not None:
            self.completed = dict(restore_state.get("completed") or {})
            self.roi_clicks = {key: list(value) for key, value in (restore_state.get("roi_clicks") or {}).items()}
            self.current_roi_idx = int(restore_state.get("current_roi_idx") or 0)
            self.current_roi_idx = max(0, min(len(self.rois) - 1, self.current_roi_idx))
            self._load_current_roi_clicks()
            saved_clicks = list((restore_state.get("profile_clicks") or [])[:2])
            suffix = self._current_roi_suffix()
            if saved_clicks and suffix not in self.roi_clicks and suffix not in self.completed:
                self.profile_clicks = saved_clicks
                self._sync_entry_vars_from_clicks()
            self._refresh_roi_list()
            self._select_roi_in_list()
            self._render_current_roi()
            restore_state["loaded"] = True
            self._sync_active_batch_roi_state()
            self._update_active_batch_roi_tab_progress()

        if restore_state is not None or self.batch_roi_paths:
            self._set_batch_folder_label(self.current_path.parent)
        else:
            self.image_label_var.set(
                f"{self.current_path.name}\nusing slice 0 of {volume.shape[0]} slice(s), "
                f"{volume.shape[2]} x {volume.shape[1]}, {volume.dtype}"
            )
        self.status_var.set(f"Loaded {self.current_path}. Click start/end on the profile.")

    def _set_slice_zero(self) -> None:
        if self.volume is None:
            return
        self.image = np.asarray(self.volume[0])
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
        if self.roi_table is None:
            return
        selection = self.roi_table.selection()
        if not selection:
            return
        try:
            selected_idx = int(selection[0])
        except (TypeError, ValueError):
            return
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
        if self.roi_table is None:
            return
        self._updating_roi_selection = True
        iid = str(self.current_roi_idx)
        self.roi_table.selection_set(iid)
        self.roi_table.focus(iid)
        self.roi_table.see(iid)
        self.after_idle(lambda: setattr(self, "_updating_roi_selection", False))

    def _roi_metric_values(self, roi: ISezROI) -> tuple[str, ...]:
        result = self.completed.get(roi.suffix)
        if result is None:
            return tuple("" for _name in RESULTS_HEADERS[1:])
        try:
            measurements = imagej_shape_measurements_from_frame(make_isez_plot_image(result))
        except Exception:
            return tuple("" for _name in RESULTS_HEADERS[1:])
        return tuple(f"{float(measurements.get(name, 0.0)):.3f}" for name in RESULTS_HEADERS[1:])

    def _refresh_roi_list(self) -> None:
        if self.roi_table is None:
            return
        self.roi_table.delete(*self.roi_table.get_children(""))
        for index, roi in enumerate(self.rois):
            self.roi_table.insert(
                "",
                "end",
                iid=str(index),
                values=(roi.suffix, *self._roi_metric_values(roi)),
            )
        self._select_roi_in_list()
        self._update_build_stack_button_state()

    def _render_empty_canvas(self) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.figure = None
        self.ax_profile = None
        self.ax_roi_grid = None
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

        self.figure = Figure(figsize=(8.5, 5.2), dpi=100)
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1.35, 0.95])
        self.ax_roi_grid = self.figure.add_subplot(grid[0, 0])
        self.ax_profile = self.figure.add_subplot(grid[1, 0])
        self.figure.subplots_adjust(left=0.065, right=0.965, bottom=0.085, top=0.985, hspace=0.13)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_holder)
        self.canvas.mpl_connect("button_press_event", self._on_profile_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.get_tk_widget().bind("<Configure>", self._on_plot_canvas_configure, add="+")

    def _on_plot_motion(self, event) -> None:
        if self.canvas is None:
            return
        try:
            self.canvas.get_tk_widget().configure(cursor="hand2" if event.inaxes is self.ax_roi_grid else "")
        except tk.TclError:
            pass

    def _on_plot_canvas_configure(self, event) -> None:
        if self.figure is None:
            return
        width = max(320, int(event.width))
        height = max(260, int(event.height))
        dpi = float(self.figure.dpi or 100)
        current_w, current_h = self.figure.get_size_inches()
        next_w = width / dpi
        next_h = height / dpi
        if abs(current_w - next_w) < 0.02 and abs(current_h - next_h) < 0.02:
            return
        self.figure.set_size_inches(next_w, next_h, forward=False)
        if self.canvas is not None:
            self.canvas.draw_idle()

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
        self._draw_roi_overview_grid()

        xs = np.arange(1, profile.size + 1, dtype=np.float64)
        ymin = float(np.nanmin(profile))
        self.ax_profile.clear()
        self.ax_profile.plot(xs, profile, color="black", linewidth=1.2)
        self.ax_profile.set_xlim(0, 140)
        self.ax_profile.set_ylim(ymin, ymin + 20)
        self.ax_profile.set_position([0.12, 0.085, 0.78, 0.30])
        self.ax_profile.text(
            0.01,
            0.96,
            f"ROI {roi.suffix}",
            ha="left",
            va="top",
            transform=self.ax_profile.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2},
        )
        for idx, click in enumerate(self.profile_clicks[:2]):
            color = "#1f77b4" if idx == 0 else "#d62728"
            self.ax_profile.axvline(click, color=color, linewidth=1.2)

        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        if not canvas_widget.winfo_manager():
            canvas_widget.pack(fill="both", expand=True)
        self._update_profile_status(profile)
        self._update_confirm_button_state()

    def _roi_overview_result(self, index: int) -> ISezResult | None:
        if self.image is None or index >= len(self.rois):
            return None
        roi = self.rois[index]
        if index == self.current_roi_idx:
            clicks = list(self.profile_clicks[:2])
            # A completed ROI can be edited.  Prefer the current two-point
            # selection so the overview previews the replacement result before
            # the idle auto-save updates ``self.completed`` and the table.
            if len(clicks) < 2:
                return self.completed.get(roi.suffix)
        elif roi.suffix in self.completed:
            return self.completed[roi.suffix]
        else:
            clicks = list((self.roi_clicks.get(roi.suffix) or [])[:2])
        if len(clicks) < 2:
            return None

        try:
            profile = intensity_profile(self.image, roi)
            start = _clamp_profile_index(clicks[0], profile.size)
            end = _clamp_profile_index(clicks[1], profile.size)
            if start > end:
                start, end = end, start
            adj_start, adj_end, slope = adjust_isez_bounds(profile, start, end, right_click_x=max(clicks))
            data = rotated_rescaled_isez(profile, adj_start, adj_end)
        except Exception:
            return None

        center = (start + end) / 2.0
        return ISezResult(
            roi=roi,
            start=start,
            end=end,
            adjusted_start=adj_start,
            adjusted_end=adj_end,
            center=center,
            slope=slope,
            max_index=int(data["max_index"]),
            min_intensity=float(data["min_intensity"]),
            max_intensity=float(data["max_intensity"]),
            normalized_x=np.asarray(data["x"], dtype=np.float64),
            normalized_y=np.asarray(data["y"], dtype=np.float64),
            baseline_x=np.asarray(data["baseline_x"], dtype=np.float64),
            baseline_y=np.asarray(data["baseline_y"], dtype=np.float64),
        )

    def _draw_roi_overview_grid(self) -> None:
        ax = self.ax_roi_grid
        if ax is None:
            return

        ax.clear()
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 3)
        ax.set_axis_off()
        ax.set_facecolor("#fbfbfb")

        for index in range(21):
            col = index % 7
            row = 2 - (index // 7)
            active = index == self.current_roi_idx
            updated = index < len(self.rois) and self.rois[index].suffix in self._flashing_updated_rois
            edge = "#b45309" if updated else ("#111827" if active else "#d1d5db")
            width = 2.0 if updated else (1.2 if active else 0.6)
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor="#fbfbfb", edgecolor=edge, linewidth=width))

            if index >= len(self.rois):
                continue

            roi = self.rois[index]
            done = roi.suffix in self.completed
            ax.text(
                col + 0.04,
                row + 0.94,
                str(index + 1),
                ha="left",
                va="top",
                fontsize=7,
                color="#b45309" if updated else "#111827",
                weight="bold" if updated else "normal",
            )
            ax.text(
                col + 0.96,
                row + 0.94,
                "\u21bb" if updated else ("\u2713" if done else "x"),
                ha="right",
                va="top",
                fontsize=7,
                color="#b45309" if updated else ("#047857" if done else "#b91c1c"),
                weight="bold",
            )

            result = self._roi_overview_result(index)
            if result is None:
                continue

            center = float(result.center)
            x_min = center - MATLAB_ISEZ_X_HALF_WIDTH
            x_max = center + MATLAB_ISEZ_X_HALF_WIDTH
            y_min, y_max = MATLAB_ISEZ_Y_LIMITS

            def to_cell(xs, ys):
                xs = np.asarray(xs, dtype=np.float64)
                ys = np.asarray(ys, dtype=np.float64)
                x_scaled = col + 0.08 + ((xs - x_min) / (x_max - x_min)) * 0.84
                y_scaled = row + 0.10 + ((ys - y_min) / (y_max - y_min)) * 0.78
                mask = (
                    np.isfinite(x_scaled)
                    & np.isfinite(y_scaled)
                    & (x_scaled >= col + 0.04)
                    & (x_scaled <= col + 0.96)
                    & (y_scaled >= row + 0.04)
                    & (y_scaled <= row + 0.90)
                )
                return x_scaled[mask], y_scaled[mask]

            x_curve, y_curve = to_cell(result.normalized_x, result.normalized_y)
            if x_curve.size >= 2:
                ax.plot(x_curve, y_curve, color="black", linewidth=0.75)
            x_base, y_base = to_cell(result.baseline_x, result.baseline_y)
            if x_base.size >= 2:
                ax.plot(x_base, y_base, color="black", linewidth=0.55)

    def _redraw_roi_update_flash(self) -> None:
        if self.ax_roi_grid is None:
            return
        self._draw_roi_overview_grid()
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _cancel_roi_update_animation(self, suffix: str, *, redraw: bool) -> None:
        job = self._roi_update_animation_jobs.pop(suffix, None)
        if job is not None:
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass
        self._flashing_updated_rois.discard(suffix)
        if redraw:
            self._redraw_roi_update_flash()

    def _cancel_roi_update_animations(self, *, redraw: bool) -> None:
        jobs = list(self._roi_update_animation_jobs.values())
        self._roi_update_animation_jobs.clear()
        for job in jobs:
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass
        had_flashing_rois = bool(self._flashing_updated_rois)
        self._flashing_updated_rois.clear()
        if redraw and had_flashing_rois:
            self._redraw_roi_update_flash()

    def _start_roi_update_animation(self, suffix: str) -> None:
        """Briefly pulse an updated ROI cell, then restore normal styling."""
        self._cancel_roi_update_animation(suffix, redraw=False)
        flashes_remaining = 8

        def flash() -> None:
            nonlocal flashes_remaining
            if suffix not in self.completed:
                self._roi_update_animation_jobs.pop(suffix, None)
                self._flashing_updated_rois.discard(suffix)
                self._redraw_roi_update_flash()
                return

            if suffix in self._flashing_updated_rois:
                self._flashing_updated_rois.discard(suffix)
            else:
                self._flashing_updated_rois.add(suffix)
            self._redraw_roi_update_flash()
            flashes_remaining -= 1

            if flashes_remaining > 0:
                self._roi_update_animation_jobs[suffix] = self.after(140, flash)
            else:
                self._flashing_updated_rois.discard(suffix)
                self._roi_update_animation_jobs.pop(suffix, None)

        flash()

    def _on_profile_click(self, event) -> None:
        if event.inaxes is self.ax_roi_grid:
            self._select_roi_from_grid_click(event)
            return
        if event.inaxes is not self.ax_profile or event.xdata is None:
            return
        if self.image is None:
            return

        click = float(event.xdata)
        if len(self.profile_clicks) >= 2:
            self.profile_clicks = [click]
        else:
            self.profile_clicks.append(click)

        self._sync_entry_vars_from_clicks()
        self._remember_current_roi_clicks()
        self._render_current_roi()
        if len(self.profile_clicks) >= 2:
            self.after_idle(self._auto_save_current_roi)

    def _select_roi_from_grid_click(self, event) -> None:
        """Select the ROI represented by a click anywhere in its grid cell."""
        if event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        if not (0.0 <= x < 7.0 and 0.0 <= y < 3.0):
            return

        column = int(math.floor(x))
        row_from_bottom = int(math.floor(y))
        index = (2 - row_from_bottom) * 7 + column
        if not (0 <= index < len(self.rois)):
            return

        if index != self.current_roi_idx:
            self._remember_current_roi_clicks()
            self.current_roi_idx = index
            self._load_current_roi_clicks()
        self._select_roi_in_list()
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

    def _entry_clicks_from_vars(self, *, show_errors: bool) -> list[float] | None:
        start_text = self.start_var.get().strip()
        end_text = self.end_var.get().strip()
        if not start_text and not end_text:
            return None
        if not start_text or not end_text:
            if show_errors:
                messagebox.showerror("Profile Selection", "Enter both start and end values.")
            return None

        try:
            start = float(start_text)
            end = float(end_text)
        except ValueError:
            if show_errors:
                messagebox.showerror("Profile Selection", "Enter numeric start and end values.")
            return None

        n_points = self._current_profile_size()
        return [
            float(_clamp_profile_index(start, n_points)),
            float(_clamp_profile_index(end, n_points)),
        ]

    def _apply_entry_clicks(self, _event=None, *, show_errors: bool = True, auto_save: bool = True) -> str | None:
        clicks = self._entry_clicks_from_vars(show_errors=show_errors)
        if clicks is None:
            return None

        self.profile_clicks = clicks
        self._sync_entry_vars_from_clicks()
        self._remember_current_roi_clicks()
        self._render_current_roi()
        if auto_save and len(self.profile_clicks) >= 2:
            self.after_idle(self._auto_save_current_roi)
        return "break"

    def _apply_entry_clicks_if_complete(self, _event=None) -> None:
        if self.start_var.get().strip() and self.end_var.get().strip():
            self._apply_entry_clicks(show_errors=False)

    def _clear_clicks(self) -> None:
        suffix = self._current_roi_suffix()
        self._cancel_roi_update_animation(suffix, redraw=False)
        self.profile_clicks.clear()
        self.roi_clicks.pop(suffix, None)
        self.completed.pop(suffix, None)
        self.start_var.set("")
        self.end_var.set("")
        self._refresh_roi_list()
        if self.batch_roi_notebook is not None and self._active_batch_roi_tab:
            self._sync_active_batch_roi_state()
            self._update_active_batch_roi_tab_progress()
        self._render_current_roi()

    def _update_confirm_button_state(self) -> None:
        self._update_clear_button_state()

    def _update_clear_button_state(self) -> None:
        enabled = bool(self.profile_clicks) and self.image is not None
        for button in (getattr(self, "clear_button", None),):
            if button is None:
                continue
            if enabled:
                button.state(["!disabled"])
            else:
                button.state(["disabled"])

    def _all_rois_completed(self) -> bool:
        return bool(self.rois) and all(roi.suffix in self.completed for roi in self.rois)

    def _update_build_stack_button_state(self) -> None:
        if self.build_stacks_button is None:
            return
        if self._all_rois_completed():
            self.build_stacks_button.state(["!disabled"])
        else:
            self.build_stacks_button.state(["disabled"])

    def _auto_save_current_roi(self) -> None:
        if self._auto_saving_roi or self.image is None:
            return
        if len(self.profile_clicks) < 2:
            return
        replacing_completed_roi = self._current_roi_suffix() in self.completed
        self._auto_saving_roi = True
        try:
            # New ROIs retain the quick auto-advance workflow.  When revising
            # an existing ROI, stay on it so the user can verify the updated
            # plot and measurement row.
            self._save_current_roi(auto_advance=not replacing_completed_roi)
        finally:
            self._auto_saving_roi = False

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

    def _save_current_roi(self, *, auto_advance: bool, build_after: bool = False) -> None:
        if self.image is None:
            messagebox.showwarning("Step 4", "Load an OCT image first.")
            return
        self._apply_entry_clicks(show_errors=False, auto_save=False)
        if len(self.profile_clicks) < 2:
            messagebox.showwarning("Step 4", "Click or enter both start and end profile positions.")
            return

        roi = self.rois[self.current_roi_idx]
        replacing_completed_roi = roi.suffix in self.completed
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
        action = "Updated" if replacing_completed_roi else "Confirmed"
        self.status_var.set(f"{action} ROI {roi.suffix}.")
        if replacing_completed_roi:
            self._start_roi_update_animation(roi.suffix)
        if self.batch_roi_notebook is not None and self._active_batch_roi_tab:
            self._sync_active_batch_roi_state()
            self._update_active_batch_roi_tab_progress()

        if build_after:
            self._select_roi_in_list()
            self._render_current_roi()
            self._build_stack_outputs()
            return

        if auto_advance and self.current_roi_idx < len(self.rois) - 1:
            self.current_roi_idx += 1
            self._load_current_roi_clicks()
            self._select_roi_in_list()
            self._render_current_roi()
        else:
            self._select_roi_in_list()
            self._draw_roi_overview_grid()
            if self.canvas is not None:
                self.canvas.draw()
            self._update_build_stack_button_state()

    def _build_stack_outputs(self) -> None:
        if not self.completed:
            messagebox.showwarning("Step 4", "Save at least one ROI before building stack outputs.")
            return
        if not self._all_rois_completed():
            missing = [roi.suffix for roi in self.rois if roi.suffix not in self.completed]
            messagebox.showwarning(
                "Step 4",
                "Build Stack is available after all ROI plots are confirmed.\n"
                f"Missing: {', '.join(missing)}",
            )
            self._update_build_stack_button_state()
            return

        ordered = [self.completed[roi.suffix] for roi in self.rois if roi.suffix in self.completed]
        outdir = Path(self.output_dir_var.get() or ".")
        try:
            outdir.mkdir(parents=True, exist_ok=True)

            # ImageJ macro equivalent:
            # open *_ISez_*.png -> run("Images to Stack") ->
            # saveAs("Tiff", "ROI_to_move_stck.tif").
            # We render the plot frames in memory because this workflow does
            # not save the intermediate MATLAB PNG files.
            isez_images = [make_isez_plot_image(result) for result in ordered]
            if isez_images:
                isez_images[0].save(outdir / "ROI_to_move_stck.tif", save_all=True, append_images=isez_images[1:])

            # MATLAB produced one ROI-overlay JPG per ROI, then the ImageJ macro
            # stacked the original image plus those JPGs and ran a max
            # projection before saving MAX_Stack.tif. The app does not apply
            # extra ROI image processing here.
            max_projection = max_stack_projection_image(self.image, [result.roi for result in ordered])
            Image.fromarray(max_projection).save(outdir / "MAX_Stack.tif")

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
        messagebox.showinfo("Step 4", "Built MAX_Stack.tif, Results.xlsx, and ROI_to_move_stck.tif.")
        if self.batch_roi_notebook is not None and self._active_batch_roi_tab:
            self._mark_active_batch_roi_complete()
            if not self._select_next_incomplete_batch_roi_tab():
                self.status_var.set("Batch ROI complete. All selected folder tabs are done.")
                messagebox.showinfo("Batch ROI", "All selected Step 4 folders are done.")
            return
        if self.batch_roi_paths and self.batch_roi_index >= 0:
            self._load_next_batch_roi()


def main() -> None:
    root = tk.Tk()
    root.title("AIDaS Step 4 - Analyze ISez")
    root.geometry("1200x800")
    frame = Step4Frame(root)
    frame.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
