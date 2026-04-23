"""Step 1 translation of the OCT R script into Python.

This pass mirrors only the R script's startup/configuration and image-loading
block so we can verify the loaded data before translating later sections.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline

from aidas.utils.io_utils import read_analyze, write_analyze


REFERENCE_DARK = "DARK_MARKED"
REFERENCE_LIGHT = "LIGHT_MARKED"
TO_PROCESS_DARK = "DARK"
TO_PROCESS_LIGHT = "LIGHT"
PIXEL_WIDTH = 3.89

IMAGE_INDEX_LIGHT = [1, 2]
IMAGE_INDEX_DARK = [1, 2]

DFonINITIALspline = 10
DFforSECONDfit = 10

DEFAULT_INPUT_DIR = Path(r"C:\Users\behzad\Desktop\flat")
DEBUG_STEP = "startup"


class StopTranslationBoundary(Exception):
    """Raised when we intentionally stop at the current translation boundary."""


def dbg(step: str, *parts: object) -> None:
    """Print debug messages in the same style as the R script."""
    global DEBUG_STEP
    DEBUG_STEP = step
    message = " ".join(str(part) for part in parts)
    print(f"DEBUG [{step}] {message}")


def stop_at_boundary(step: str, *parts: object) -> None:
    """Stop execution at the current translated boundary without calling exit()."""
    dbg(step, *parts)
    raise StopTranslationBoundary


def format_number(value: float) -> str:
    """Format numeric output consistently for cross-checking with R."""
    return f"{float(value):.6f}"


def format_dim(array: np.ndarray) -> str:
    """Format a NumPy shape like R's paste(dim(x), collapse='x')."""
    return "x".join(str(size) for size in array.shape)


def show_scalar_stats(name: str, value: object) -> None:
    """Print a compact summary for scalar/string variables."""
    print(f"STAT {name}: type={type(value).__name__} value={value}")


def show_vector_stats(name: str, values: list[int]) -> None:
    """Print summary statistics for a small numeric vector."""
    arr = np.asarray(values, dtype=np.float64)
    joined = ",".join(str(int(v)) for v in arr)
    print(
        f"STAT {name}: type=int_vector length={arr.size} values={joined} "
        f"min={format_number(arr.min())} max={format_number(arr.max())} "
        f"mean={format_number(arr.mean())} sum={format_number(arr.sum())}"
    )


def show_array_stats(name: str, array: np.ndarray) -> None:
    """Print summary statistics for an ndarray."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        print(f"STAT {name}: type=ndarray dtype={array.dtype} dim={format_dim(array)} empty=1")
        return

    if np.isnan(arr).all():
        min_value = max_value = mean_value = sum_value = "nan"
    else:
        min_value = format_number(np.nanmin(arr))
        max_value = format_number(np.nanmax(arr))
        mean_value = format_number(np.nanmean(arr))
        sum_value = format_number(np.nansum(arr))

    print(
        f"STAT {name}: type=ndarray dtype={array.dtype} dim={format_dim(array)} "
        f"min={min_value} max={max_value} "
        f"mean={mean_value} sum={sum_value} "
        f"na={int(np.isnan(arr).sum())}"
    )


def load_analyze_volume_r_layout(path: Path) -> np.ndarray:
    """Load an Analyze volume and match the R script's x/y/slice layout."""
    volume = np.asarray(read_analyze(path), dtype=np.uint8)

    # read_analyze() returns (slice, x, y) for stacks. For this OCT dataset,
    # the R script is operating as if the loaded array is (y, x, slice), with
    # the in-slice x axis reversed relative to the local Analyze reader.
    if volume.ndim == 3:
        volume = np.transpose(volume, (2, 1, 0))[:, ::-1, :]

    # Keep the same safeguard as the R script in case a future reader returns 4D.
    if volume.ndim == 4:
        volume = volume[:, :, :, 0]

    return volume


def load_input_volumes(input_dir: Path) -> dict[str, np.ndarray]:
    """Load the four Analyze volumes used by the R script."""
    return {
        "REF_DARK": load_analyze_volume_r_layout(input_dir / f"{REFERENCE_DARK}.hdr"),
        "REF_LIGHT": load_analyze_volume_r_layout(input_dir / f"{REFERENCE_LIGHT}.hdr"),
        "DARK": load_analyze_volume_r_layout(input_dir / f"{TO_PROCESS_DARK}.hdr"),
        "LIGHT": load_analyze_volume_r_layout(input_dir / f"{TO_PROCESS_LIGHT}.hdr"),
    }


def save_r_image_matlines_plot(dark_slice: np.ndarray, rpe_info_2: np.ndarray, output_path: Path) -> None:
    """Save the Python equivalent of R's image(...); matlines(...)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.imshow(
        np.asarray(dark_slice, dtype=np.float64).T,
        cmap="gray",
        origin="lower",
        aspect="auto",
        extent=(1, dark_slice.shape[0], 1, dark_slice.shape[1]),
    )
    ax.plot(rpe_info_2[:, 0], rpe_info_2[:, 1], color="red", linewidth=1.0)
    ax.set_title("DARK[,,1] with RPE.info.2")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_overlay_series_plot(
    volume: np.ndarray,
    rpe_info_2_series: list[np.ndarray],
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    """Save one overlay image per translated R image(...); matlines(...) call."""
    saved_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, rpe_info_2 in enumerate(rpe_info_2_series, start=1):
        output_path = output_dir / f"{prefix}_slice_{index}.png"
        save_r_image_matlines_plot(volume[:, :, index - 1], rpe_info_2, output_path)
        saved_paths.append(output_path)
    return saved_paths


def save_shift_position_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    spline_xy: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save the Python equivalent of plot(...); matlines(..., col='red')."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(x_values, y_values, color="black", linewidth=1.0)
    ax.plot(
        spline_xy[:, 0],
        spline_xy[:, 1],
        linestyle="None",
        marker="o",
        markersize=6.0,
        markerfacecolor="red",
        markeredgecolor="red",
    )
    ax.set_ylim(430, 470)
    ax.set_title(title)
    ax.set_xlabel("dist.on.spline.microns")
    ax.set_ylabel("shift target")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_profile_plot(
    profile_xy: np.ndarray,
    output_path: Path,
    title: str,
    verticals: tuple[float, ...] = (),
    spline_xy: np.ndarray | None = None,
) -> None:
    """Save simple line plots used by the later R verification steps."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(profile_xy[:, 0], profile_xy[:, 1], color="black", linewidth=1.0)
    if spline_xy is not None:
        ax.plot(
            spline_xy[:, 0],
            spline_xy[:, 1],
            linestyle="None",
            marker="o",
            markersize=6.0,
            markerfacecolor="none",
            markeredgecolor="black",
        )
    for x in verticals:
        ax.axvline(float(x), color="red" if x == verticals[-1] and len(verticals) > 3 else "black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("mean intensity")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def get_recon_value(unwrapped_recon: np.ndarray, upper_x: int, upper_y: int, point: np.ndarray) -> float:
    """Python equivalent of the R GETrecon() helper."""
    col = float(point[0])
    row = float(point[1])
    if 1 <= col <= upper_x and 1 <= row <= upper_y:
        idx = int((col - 1) * upper_y + row - 1)
        return float(unwrapped_recon[idx])
    return np.nan


def correlation_estimate(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Pearson correlation estimate used by R's cor.test(... )$est."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0.0 or y_std == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def build_floor_sample_line(start_x: float, end_x: float, start_y: float, end_y: float) -> np.ndarray:
    """Mirror the R seq(...); cbind(...); floor(...) sampling line construction."""
    dx = (end_x - start_x) / 500.0
    dy = (end_y - start_y) / 500.0
    line_x = np.arange(start_x, end_x + dx, dx)
    line_y = np.arange(start_y, end_y + dy, dy)

    if line_x.size < 501:
        line_x = np.linspace(start_x, end_x, 501)
    else:
        line_x = line_x[:501]

    if line_y.size < 501:
        line_y = np.linspace(start_y, end_y, 501)
    else:
        line_y = line_y[:501]

    return np.floor(np.column_stack((line_x, line_y)))


def compute_retina_points_for_marked_slice(
    marked_slice: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Translate the repeated REF.* slice block that builds Retina.Points and angles."""
    r = np.asarray(marked_slice, dtype=np.float64).copy()
    r[r < 243] = np.nan
    r[r > 243] = np.nan
    r[r == 243] = 1.0

    xcoords = xs * r
    ycoords = ys * r
    mask = ~np.isnan(xcoords)
    fovea_line = np.column_stack((xcoords[mask], ycoords[mask]))
    if fovea_line.size:
        fovea_line = fovea_line[np.argsort(fovea_line[:, 0])]
    if fovea_line.shape[0] > 0 and np.unique(fovea_line[:, 0]).size == 1:
        fovea_line[0, 0] = fovea_line[0, 0] + 0.1

    r = np.asarray(marked_slice, dtype=np.float64).copy()
    r[r < 255] = np.nan
    r[r > 255] = np.nan
    r[r == 255] = 1.0

    xcoords = xs * r
    ycoords = ys * r
    mask = ~np.isnan(xcoords)
    rpe_line = np.column_stack((xcoords[mask], ycoords[mask]))
    if rpe_line.size:
        rpe_line = rpe_line[np.argsort(rpe_line[:, 0])]

    rpe_sp = fit_smooth_spline_like_r(rpe_line[:, 0], rpe_line[:, 1], df=DFonINITIALspline)
    pred_x = np.arange(0.0, marked_slice.shape[0] + 0.02, 0.02)
    pred_y = rpe_sp(pred_x)
    pred_dy = rpe_sp.derivative()(pred_x)
    rpe_spline = np.column_stack((pred_x, pred_y, pred_dy))

    rpe_spline_compare = np.vstack((rpe_spline[1:], rpe_spline[-1:]))
    rpe_spline = np.column_stack((rpe_spline[:, 0], rpe_spline))
    rpe_spline[:, 0] = np.sqrt(
        (rpe_spline[:, 1] - rpe_spline_compare[:, 0]) ** 2
        + (rpe_spline[:, 2] - rpe_spline_compare[:, 1]) ** 2
    )
    rpe_spline = np.column_stack((np.cumsum(rpe_spline[:, 0]), rpe_spline))

    fovea_curve = fit_line_coefficients(fovea_line)
    compare_fovea_and_rpe = rpe_spline[:, [2, 3, 3, 3]].copy()
    compare_fovea_and_rpe[:, 2] = compare_fovea_and_rpe[:, 0] * fovea_curve[1] + fovea_curve[0]
    compare_fovea_and_rpe[:, 3] = np.abs(compare_fovea_and_rpe[:, 1] - compare_fovea_and_rpe[:, 2])
    center = int(np.where(compare_fovea_and_rpe[:, 3] == np.min(compare_fovea_and_rpe[:, 3]))[0][0] + 1)
    center_value = float(rpe_spline[center - 1, 0])

    rpe_info = rpe_spline[:, [2, 3, 4, 0]].copy()
    rpe_info[:, 3] = np.round((rpe_info[:, 3] - center_value) * PIXEL_WIDTH, 0)
    rpe_info = rpe_info[(rpe_info[:, 3] > -200.9) & (rpe_info[:, 3] < 3000.9)]

    unique_dist = np.unique(rpe_info[:, 3])
    rpe_info_2 = np.column_stack((unique_dist, unique_dist, unique_dist, unique_dist))
    for x in range(rpe_info_2.shape[0]):
        first = np.where(rpe_info[:, 3] == rpe_info_2[x, 3])[0][0]
        rpe_info_2[x, 0:3] = rpe_info[first, 0:3]

    rpe_info_2[:, 2] = (-1.0) / rpe_info_2[:, 2]

    apparent_angle = rpe_info_2[(rpe_info_2[:, 3] > 0) & (rpe_info_2[:, 3] < 2750), 0:2].copy()
    slopey_0_to_2750_deg = float(np.arctan(fit_line_coefficients(apparent_angle)[1]) * 180.0 / np.pi)

    apparent_angle_fovea = rpe_info_2[(rpe_info_2[:, 3] > -100) & (rpe_info_2[:, 3] < 100), 0:2].copy()
    slopey_neg100_to_100_deg = float(np.arctan(fit_line_coefficients(apparent_angle_fovea)[1]) * 180.0 / np.pi)

    deltas = rpe_info_2[:, [0, 1]].copy()
    deltas[:, 0] = np.cos(np.arctan(rpe_info_2[:, 2]))
    deltas[:, 1] = np.sin(np.arctan(rpe_info_2[:, 2]))
    pixel_move = round(500 / PIXEL_WIDTH, 1)
    deltas = deltas * pixel_move

    add = deltas.copy()
    flip = deltas * (-1.0)
    add[:, 0] = np.where(flip[:, 1] > 0, flip[:, 0], add[:, 0])
    add[:, 1] = np.where(flip[:, 1] > 0, flip[:, 1], add[:, 1])
    sub = add / (-10.0)
    add = add - (add / 10.0)

    retina_points = rpe_info_2[:, [3, 0, 1, 2, 2, 2, 2, 2]].copy()
    retina_points[:, 4:6] = add + retina_points[:, 1:3]
    retina_points[:, 6:8] = sub + retina_points[:, 1:3]

    return retina_points, rpe_info_2, slopey_neg100_to_100_deg, slopey_0_to_2750_deg


def shift_rows_to_border(data: np.ndarray, borders: np.ndarray, fill_value: float) -> np.ndarray:
    """Translate the repeated border-based row shifting used in the R script."""
    source = np.asarray(data, dtype=np.float64)
    out = np.full_like(source, fill_value, dtype=np.float64)

    for row_index in range(source.shape[0]):
        border = int(borders[row_index])
        if border < 450:
            left_len = 500 + (border - 449)
            out[row_index, :left_len, ...] = source[row_index, (450 - border - 1):500, ...]
        elif border == 450:
            out[row_index, ...] = source[row_index, ...]
        else:
            offset = border - 450
            out[row_index, offset:500, ...] = source[row_index, : (500 - offset), ...]

    return out


def fit_line_coefficients(points: np.ndarray) -> np.ndarray:
    """Return R-style linear-model coefficients [intercept, slope]."""
    arr = np.asarray(points, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    design = np.column_stack((np.ones_like(x), x))
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeffs.astype(np.float64)


def fit_smooth_spline_like_r(x: np.ndarray, y: np.ndarray, df: int, degree: int = 3):
    """Approximate R smooth.spline(df=...) with a weighted cubic regression spline.

    R's `smooth.spline(df=10)` targets effective degrees of freedom, not the same
    smoothing penalty used by SciPy's `UnivariateSpline(..., s=...)`.
    A cubic regression spline with roughly `df` coefficients is usually much
    closer for this OCT workflow.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    unique_x, inverse = np.unique(x, return_inverse=True)
    y_sum = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse).astype(np.float64)
    unique_y = y_sum / counts

    if unique_x.size <= degree + 1:
        return UnivariateSpline(unique_x, unique_y, k=min(degree, unique_x.size - 1), s=0)

    n_coeff = max(int(df), degree + 1)
    n_internal = max(0, n_coeff - (degree + 1))

    if n_internal == 0:
        return UnivariateSpline(unique_x, unique_y, k=degree, s=0)

    probs = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    knots = np.quantile(unique_x, probs)
    knots = np.unique(knots)

    # Interior knots must lie strictly inside the boundary knots.
    eps = np.finfo(np.float64).eps * max(1.0, float(unique_x[-1] - unique_x[0]))
    knots = knots[(knots > unique_x[0] + eps) & (knots < unique_x[-1] - eps)]

    while knots.size > 0:
        try:
            return LSQUnivariateSpline(
                unique_x,
                unique_y,
                t=knots,
                w=np.sqrt(counts),
                k=degree,
            )
        except ValueError:
            knots = knots[1:-1]

    return UnivariateSpline(unique_x, unique_y, k=degree, s=len(unique_x))


def main() -> dict[str, np.ndarray]:
    parser = argparse.ArgumentParser(
        description="Step 1: load the four OCT Analyze volumes used by the R script."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing DARK/LIGHT and *_MARKED Analyze files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path.cwd()

    # dbg("startup", "Script started")
    # dbg("input-config", "Working directory:", outdir)
    # dbg(
    #     "input-config",
    #     "Input directory:",
    #     input_dir,
    #     "length(IMAGE.INDEX.LIGHT)=",
    #     len(IMAGE_INDEX_LIGHT),
    #     "length(IMAGE.INDEX.DARK)=",
    #     len(IMAGE_INDEX_DARK),
    # )
    # dbg("load-images", "Loading Analyze volumes")

    volumes = load_input_volumes(input_dir)

    # dbg(
    #     "load-images",
    #     "REF.DARK dim:",
    #     format_dim(volumes["REF_DARK"]),
    #     "REF.LIGHT dim:",
    #     format_dim(volumes["REF_LIGHT"]),
    # )
    # dbg(
    #     "load-images",
    #     "DARK dim:",
    #     format_dim(volumes["DARK"]),
    #     "LIGHT dim:",
    #     format_dim(volumes["LIGHT"]),
    # )
    # dbg("variable-stats", "Printing summary statistics for current variables")
    # show_scalar_stats("REFERENCE.DARK", REFERENCE_DARK)
    # show_scalar_stats("REFERENCE.LIGHT", REFERENCE_LIGHT)
    # show_scalar_stats("TO.PROCESS.DARK", TO_PROCESS_DARK)
    # show_scalar_stats("TO.PROCESS.LIGHT", TO_PROCESS_LIGHT)
    # show_scalar_stats("PIXEL.WIDTH", PIXEL_WIDTH)
    # show_scalar_stats("DFonINITIALspline", DFonINITIALspline)
    # show_scalar_stats("DFforSECONDfit", DFforSECONDfit)
    # show_vector_stats("IMAGE.INDEX.LIGHT", IMAGE_INDEX_LIGHT)
    # show_vector_stats("IMAGE.INDEX.DARK", IMAGE_INDEX_DARK)
    # show_array_stats("REF.DARK", volumes["REF_DARK"])
    # show_array_stats("REF.LIGHT", volumes["REF_LIGHT"])
    # show_array_stats("DARK", volumes["DARK"])
    # show_array_stats("LIGHT", volumes["LIGHT"])

    # dbg("fovea-center", "Translating the first R block for fovea center detection")

    r = np.asarray(volumes["REF_DARK"][:, :, 0], dtype=np.float64)
    xs = r.copy()
    insert = np.arange(1, r.shape[0] + 1, dtype=np.float64)
    for x in range(xs.shape[1]):
        xs[:, x] = insert

    ys = r.copy()
    insert = np.arange(1, r.shape[1] + 1, dtype=np.float64)
    for x in range(ys.shape[0]):
        ys[x, :] = insert

    r[r < 243] = np.nan
    r[r > 243] = np.nan
    r[r == 243] = 1.0

    xcoords = xs * r
    ycoords = ys * r

    mask = ~np.isnan(xcoords)
    fovea_line = np.column_stack((xcoords[mask], ycoords[mask]))
    if fovea_line.size:
        fovea_line = fovea_line[np.argsort(fovea_line[:, 0])]

    if fovea_line.shape[0] > 0 and np.unique(fovea_line[:, 0]).size == 1:
        fovea_line[0, 0] = fovea_line[0, 0] + 0.1

    # dbg("variable-stats", "Printing summary statistics for translated fovea variables")
    # show_array_stats("R", r)
    # show_array_stats("Xs", xs)
    # show_array_stats("Ys", ys)
    # show_array_stats("Xcoords", xcoords)
    # show_array_stats("Ycoords", ycoords)
    # show_array_stats("fovea.line", fovea_line)

    # dbg("rpe-line", "Translating the next R block for RPE.line detection")

    r = np.asarray(volumes["REF_DARK"][:, :, 0], dtype=np.float64)
    r[r < 255] = np.nan
    r[r > 255] = np.nan
    r[r == 255] = 1.0

    xcoords = xs * r
    ycoords = ys * r

    mask = ~np.isnan(xcoords)
    rpe_line = np.column_stack((xcoords[mask], ycoords[mask]))
    if rpe_line.size:
        rpe_line = rpe_line[np.argsort(rpe_line[:, 0])]

    # dbg("variable-stats", "Printing summary statistics for translated RPE variables")
    # show_array_stats("R", r)
    # show_array_stats("Xcoords", xcoords)
    # show_array_stats("Ycoords", ycoords)
    # show_array_stats("RPE.line", rpe_line)

    # dbg("rpe-spline", "Translating the spline/center/RPE.info block through line 305")

    rpe_sp = fit_smooth_spline_like_r(rpe_line[:, 0], rpe_line[:, 1], df=DFonINITIALspline)
    pred_x = np.arange(0.0, r.shape[0] + 0.02, 0.02)
    pred_y = rpe_sp(pred_x)
    pred_dy = rpe_sp.derivative()(pred_x)
    rpe_spline = np.column_stack((pred_x, pred_y, pred_dy))

    rpe_spline_compare = np.vstack((rpe_spline[1:], rpe_spline[-1:]))
    rpe_spline = np.column_stack((rpe_spline[:, 0], rpe_spline))
    rpe_spline[:, 0] = np.sqrt(
        (rpe_spline[:, 1] - rpe_spline_compare[:, 0]) ** 2
        + (rpe_spline[:, 2] - rpe_spline_compare[:, 1]) ** 2
    )
    rpe_spline = np.column_stack((np.cumsum(rpe_spline[:, 0]), rpe_spline))

    fovea_curve = fit_line_coefficients(fovea_line)
    compare_fovea_and_rpe = rpe_spline[:, [2, 3, 3, 3]].copy()
    compare_fovea_and_rpe[:, 2] = compare_fovea_and_rpe[:, 0] * fovea_curve[1] + fovea_curve[0]
    compare_fovea_and_rpe[:, 3] = np.abs(compare_fovea_and_rpe[:, 1] - compare_fovea_and_rpe[:, 2])
    center = int(np.where(compare_fovea_and_rpe[:, 3] == np.min(compare_fovea_and_rpe[:, 3]))[0][0] + 1)
    center_value = float(rpe_spline[center - 1, 0])

    rpe_info = rpe_spline[:, [2, 3, 4, 0]].copy()
    rpe_info[:, 3] = np.round((rpe_info[:, 3] - center_value) * PIXEL_WIDTH, 0)
    rpe_info = rpe_info[(rpe_info[:, 3] > -200.9) & (rpe_info[:, 3] < 3000.9)]

    unique_dist = np.unique(rpe_info[:, 3])
    rpe_info_2 = np.column_stack((unique_dist, unique_dist, unique_dist, unique_dist))
    for x in range(rpe_info_2.shape[0]):
        first = np.where(rpe_info[:, 3] == rpe_info_2[x, 3])[0][0]
        rpe_info_2[x, 0:3] = rpe_info[first, 0:3]

    rpe_info_2[:, 2] = (-1.0) / rpe_info_2[:, 2]
    plots_dir = outdir / "python_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rpe_overlay_plot_path = plots_dir / "python_rpe_info_2_overlay.png"
    save_r_image_matlines_plot(volumes["DARK"][:, :, 0], rpe_info_2, rpe_overlay_plot_path)
    saved_dark_overlay_paths = [rpe_overlay_plot_path]

    # dbg("variable-stats", "Printing summary statistics for translated spline variables")
    # show_array_stats("RPE.spline.compare", rpe_spline_compare)
    # show_array_stats("RPE.spline", rpe_spline)
    # show_array_stats("fovea.curve", fovea_curve)
    # show_array_stats("compare.fovea.and.RPE", compare_fovea_and_rpe)
    # show_scalar_stats("CENTER", center)
    # show_scalar_stats("CENTER.value", center_value)
    # show_array_stats("RPE.info", rpe_info)
    # show_array_stats("RPE.info.2", rpe_info_2)
    # show_scalar_stats("PLOT.RPE.info.2", rpe_overlay_plot_path)

    # dbg("apparent-angle", "Translating the APPARENT.ANGLE block through line 331")

    apparent_angle = rpe_info_2[(rpe_info_2[:, 3] > 500) & (rpe_info_2[:, 3] < 2750), 0:2].copy()
    slopey_500_to_2750 = float(fit_line_coefficients(apparent_angle)[1])

    apparent_angle_fovea = rpe_info_2[(rpe_info_2[:, 3] > -100) & (rpe_info_2[:, 3] < 100), 0:2].copy()
    slopey_neg100_to_100 = float(fit_line_coefficients(apparent_angle_fovea)[1])

    # dbg("variable-stats", "Printing summary statistics for translated APPARENT.ANGLE variables")
    # show_array_stats("APPARENT.ANGLE.500.to.2750", apparent_angle)
    # show_scalar_stats("SLOPEY.500.to.2750", slopey_500_to_2750)
    # show_array_stats("APPARENT.ANGLE.neg100.to.100", apparent_angle_fovea)
    # show_scalar_stats("SLOPEY.neg100.to.100", slopey_neg100_to_100)

    # dbg("perpendiculars", "Translating the perpendicular-setup block through line 386")

    intercepts = rpe_info_2[:, 0].copy()
    intercepts[:] = rpe_info_2[:, 1] - (rpe_info_2[:, 0] * rpe_info_2[:, 2])

    deltas = rpe_info_2[:, [0, 1]].copy()
    deltas[:, 0] = np.cos(np.arctan(rpe_info_2[:, 2]))
    deltas[:, 1] = np.sin(np.arctan(rpe_info_2[:, 2]))

    pixel_move = round(500 / PIXEL_WIDTH, 1)
    deltas = deltas * pixel_move

    add = deltas.copy()
    flip = deltas * (-1.0)
    add[:, 0] = np.where(flip[:, 1] > 0, flip[:, 0], add[:, 0])
    add[:, 1] = np.where(flip[:, 1] > 0, flip[:, 1], add[:, 1])

    sub = add / (-10.0)
    add = add - (add / 10.0)

    retina_points = rpe_info_2[:, [3, 0, 1, 2, 2, 2, 2, 2]].copy()
    retina_points[:, 4:6] = add + retina_points[:, 1:3]
    retina_points[:, 6:8] = sub + retina_points[:, 1:3]

    # dbg("variable-stats", "Printing summary statistics for translated perpendicular variables")
    # show_array_stats("INTERCEPTS", intercepts)
    # show_array_stats("DELTAS", deltas)
    # show_scalar_stats("pixel.move", pixel_move)
    # show_array_stats("ADD", add)
    # show_array_stats("FLIP", flip)
    # show_array_stats("SUB", sub)
    # show_array_stats("Retina.Points", retina_points)

    dbg("flattened-markers", "Translating the FLATTENED.MARKERS block through line 438")
    flattened_markers = np.full((retina_points.shape[0], 500), np.nan, dtype=np.float64)
    dbg(
        "flattened-markers",
        "FLATTENED.MARKERS dim:",
        format_dim(flattened_markers),
        "Retina.Points rows:",
        retina_points.shape[0],
    )
    upper_x = int(volumes["DARK"].shape[1])
    upper_y = int(volumes["DARK"].shape[0])
    unwrapped_recon = np.ravel(volumes["REF_DARK"][:, :, 0], order="F").astype(np.float64)

    for x in range(retina_points.shape[0]):
        line = build_floor_sample_line(
            retina_points[x, 4],
            retina_points[x, 6],
            retina_points[x, 5],
            retina_points[x, 7],
        )
        f = np.array(
            [
                get_recon_value(unwrapped_recon, upper_x, upper_y, np.array([line[i, 1], line[i, 0]]))
                for i in range(line.shape[0])
            ],
            dtype=np.float64,
        )
        flattened_markers[x, :] = f[1:]

    # dbg("variable-stats", "Printing summary statistics for translated flattened-marker variables")
    # show_scalar_stats("UpperX", upper_x)
    # show_scalar_stats("UpperY", upper_y)
    # show_array_stats("unwrapped.recon", unwrapped_recon)
    # show_array_stats("FLATTENED.MARKERS", flattened_markers)
    # dbg("dark-loop", "Translating the dark-image loop through line 599")

    apparent_angles_for_light = np.column_stack(
        (
            np.asarray(IMAGE_INDEX_LIGHT, dtype=np.float64),
            np.full(len(IMAGE_INDEX_LIGHT), np.nan, dtype=np.float64),
            np.full(len(IMAGE_INDEX_LIGHT), np.nan, dtype=np.float64),
        )
    )
    apparent_angles_for_dark = np.column_stack(
        (
            np.asarray(IMAGE_INDEX_DARK, dtype=np.float64),
            np.full(len(IMAGE_INDEX_DARK), np.nan, dtype=np.float64),
            np.full(len(IMAGE_INDEX_DARK), np.nan, dtype=np.float64),
        )
    )
    dark_rpe_info_2_series: list[np.ndarray] = []
    flattened_dark_retina = np.full(
        (retina_points.shape[0], 500, volumes["DARK"].shape[2]),
        np.nan,
        dtype=np.float64,
    )

    for z in range(len(IMAGE_INDEX_DARK)):
        dbg(
            "dark-loop",
            "Processing z=",
            z + 1,
            "of",
            len(IMAGE_INDEX_DARK),
            "REF.DARK slice dim:",
            format_dim(volumes["REF_DARK"][:, :, z]),
        )
        (
            slice_retina_points,
            slice_rpe_info_2,
            slopey_neg100_to_100_deg,
            slopey_0_to_2750_deg,
        ) = compute_retina_points_for_marked_slice(
            volumes["REF_DARK"][:, :, z],
            xs,
            ys,
        )
        dark_rpe_info_2_series.append(slice_rpe_info_2.copy())
        apparent_angles_for_dark[z, 1] = slopey_neg100_to_100_deg
        apparent_angles_for_dark[z, 2] = slopey_0_to_2750_deg

        unwrapped_retina = np.ravel(volumes["DARK"][:, :, z], order="F").astype(np.float64)
        for x in range(slice_retina_points.shape[0]):
            line = build_floor_sample_line(
                slice_retina_points[x, 4],
                slice_retina_points[x, 6],
                slice_retina_points[x, 5],
                slice_retina_points[x, 7],
            )
            f = np.array(
                [
                    get_recon_value(unwrapped_retina, upper_x, upper_y, np.array([line[i, 1], line[i, 0]]))
                    for i in range(line.shape[0])
                ],
                dtype=np.float64,
            )
            flattened_dark_retina[x, :, z] = f[1:]

    saved_dark_overlay_paths = save_overlay_series_plot(
        volumes["DARK"],
        dark_rpe_info_2_series,
        plots_dir,
        "python_dark_rpe_info_2_overlay",
    )

    # dbg("light-loop", "Translating the light-image loop through line 755")
    light_rpe_info_2_series: list[np.ndarray] = []
    flattened_light_retina = np.full(
        (retina_points.shape[0], 500, volumes["LIGHT"].shape[2]),
        np.nan,
        dtype=np.float64,
    )

    for z in range(len(IMAGE_INDEX_LIGHT)):
        dbg(
            "light-loop",
            "Processing z=",
            z + 1,
            "of",
            len(IMAGE_INDEX_LIGHT),
            "REF.LIGHT slice dim:",
            format_dim(volumes["REF_LIGHT"][:, :, z]),
        )
        (
            slice_retina_points,
            slice_rpe_info_2,
            slopey_neg100_to_100_deg,
            slopey_0_to_2750_deg,
        ) = compute_retina_points_for_marked_slice(
            volumes["REF_LIGHT"][:, :, z],
            xs,
            ys,
        )
        light_rpe_info_2_series.append(slice_rpe_info_2.copy())
        apparent_angles_for_light[z, 1] = slopey_neg100_to_100_deg
        apparent_angles_for_light[z, 2] = slopey_0_to_2750_deg

        unwrapped_retina = np.ravel(volumes["LIGHT"][:, :, z], order="F").astype(np.float64)
        for x in range(slice_retina_points.shape[0]):
            line = build_floor_sample_line(
                slice_retina_points[x, 4],
                slice_retina_points[x, 6],
                slice_retina_points[x, 5],
                slice_retina_points[x, 7],
            )
            f = np.array(
                [
                    get_recon_value(unwrapped_retina, upper_x, upper_y, np.array([line[i, 1], line[i, 0]]))
                    for i in range(line.shape[0])
                ],
                dtype=np.float64,
            )
            flattened_light_retina[x, :, z] = f[1:]

    saved_light_overlay_paths = save_overlay_series_plot(
        volumes["LIGHT"],
        light_rpe_info_2_series,
        plots_dir,
        "python_light_rpe_info_2_overlay",
    )

    # dbg("variable-stats", "Printing summary statistics for translated dark/light loop variables")
    # show_array_stats("APPARENT.ANGLES.FOR.LIGHT", apparent_angles_for_light)
    # show_array_stats("APPARENT.ANGLES.FOR.DARK", apparent_angles_for_dark)
    # show_array_stats("FLATTENED.DARK.RETINA", flattened_dark_retina)
    # show_array_stats("FLATTENED.LIGHT.RETINA", flattened_light_retina)
    # show_scalar_stats(
    #     "PLOTS.DARK.RPE.info.2",
    #     ",".join(str(path) for path in saved_dark_overlay_paths),
    # )
    # show_scalar_stats(
    #     "PLOTS.LIGHT.RPE.info.2",
    #     ",".join(str(path) for path in saved_light_overlay_paths),
    # )
    # dbg("post-log-convert", "Translating the post-log conversion block through line 800")

    flattened_dark_retina[np.isnan(flattened_dark_retina)] = -32768.0
    flattened_dark_retina = flattened_dark_retina + 32768.0
    flattened_dark_retina[flattened_dark_retina < 0] = 0.0
    flattened_dark_retina_raw = np.power(2.0, flattened_dark_retina / 5000.0)

    flattened_light_retina[np.isnan(flattened_light_retina)] = -32768.0
    flattened_light_retina = flattened_light_retina + 32768.0
    flattened_light_retina[flattened_light_retina < 0] = 0.0
    flattened_light_retina_raw = np.power(2.0, flattened_light_retina / 5000.0)

    dbg("post-log-convert", "Converted DARK and LIGHT flattened arrays back to raw scale")
    dbg("grand-mean", "Building FIRST.GRAND.MEAN from DARK and LIGHT volumes")

    first_grand_mean = flattened_dark_retina_raw[:, :, 0].copy()
    for z in range(1, flattened_dark_retina_raw.shape[2]):
        first_grand_mean = first_grand_mean + flattened_dark_retina_raw[:, :, z]
    for z in range(1, flattened_light_retina_raw.shape[2]):
        first_grand_mean = first_grand_mean + flattened_light_retina_raw[:, :, z]
    first_grand_mean = first_grand_mean / (
        flattened_dark_retina_raw.shape[2] + flattened_light_retina_raw.shape[2]
    )

    rough_vit_retina_position = np.column_stack(
        (
            np.arange(-200.0, 3001.0, 1.0),
            np.full(flattened_markers.shape[0], np.nan, dtype=np.float64),
        )
    )
    dbg(
        "rough-vit-loop",
        "ROUGH.VIT.RETINA.POSITION rows:",
        rough_vit_retina_position.shape[0],
        "FLATTENED.MARKERS rows:",
        flattened_markers.shape[0],
    )
    for x in range(rough_vit_retina_position.shape[0]):
        if x >= flattened_markers.shape[0]:
            raise RuntimeError(
                f"Index exceeds FLATTENED.MARKERS in rough-vit-loop: x={x + 1}, "
                f"rows={flattened_markers.shape[0]}. Adjust loop bounds to marker rows."
            )
        a = np.where(flattened_markers[x, :] == 249)[0]
        if a.size > 0:
            rough_vit_retina_position[x, 1] = float(a[-1] + 1)

    look_to = rough_vit_retina_position.copy()
    look_to[:, 1] = np.round((250.0 - (0.7 * (250.0 - look_to[:, 1]))), 0)

    window_width_in_pixels = 400
    start_move = 201
    end_move = (flattened_markers.shape[0] - start_move) - 1

    shift_position_dark = np.full(
        (flattened_dark_retina_raw.shape[0], flattened_dark_retina_raw.shape[2] + 1),
        np.nan,
        dtype=np.float64,
    )
    shift_position_dark[:, 0] = np.arange(-200.0, 3001.0, 1.0)

    for z in range(flattened_dark_retina.shape[2]):
        revise_dark = flattened_dark_retina_raw[:, :, z]
        for x in range(start_move, end_move + 1, 50):
            row_window = slice(x - 200, x + 200)
            profile = np.nanmean(revise_dark[row_window, :], axis=0)
            comparison = np.nanmean(first_grand_mean[row_window, :], axis=0)
            top_range = int(np.nanmax(look_to[row_window, 1]))
            check_rows = np.arange(480, top_range - 1, -1, dtype=np.int64)
            check_idx = check_rows - 1
            check = np.column_stack((check_rows, profile[check_idx], comparison[check_idx]))

            moves = np.arange(-10, 11, 1, dtype=np.int64)
            slide_corr = np.full(moves.shape, np.nan, dtype=np.float64)
            for move_index, move in enumerate(moves):
                if move < 0:
                    slide_corr[move_index] = correlation_estimate(
                        check[(-move):, 1],
                        check[: (check.shape[0] + move), 2],
                    )
                elif move == 0:
                    slide_corr[move_index] = correlation_estimate(check[:, 1], check[:, 2])
                else:
                    slide_corr[move_index] = correlation_estimate(
                        check[: (check.shape[0] - move), 1],
                        check[move:, 2],
                    )
            if not np.all(np.isnan(slide_corr)):
                shift_position_dark[x - 1, z + 1] = float(moves[int(np.nanargmax(slide_corr))])

    shift_position_dark[:, 1:] = 450.0 - shift_position_dark[:, 1:]

    shift_position_dark_refined = shift_position_dark.copy()
    dark_shift_plot_paths: list[Path] = []
    shift_x = np.arange(-200.0, 3001.0, 1.0)
    for y in range(1, shift_position_dark.shape[1]):
        shift_position_dark_sp_maker = np.column_stack((shift_position_dark[:, 0], shift_position_dark[:, y]))
        valid = ~np.isnan(shift_position_dark_sp_maker[:, 1])
        shift_position_dark_sp_maker = shift_position_dark_sp_maker[valid]
        shift_position_dark_sp = fit_smooth_spline_like_r(
            shift_position_dark_sp_maker[:, 0],
            shift_position_dark_sp_maker[:, 1],
            df=DFforSECONDfit,
        )
        shift_position_dark_spline = np.column_stack((shift_x, shift_position_dark_sp(shift_x)))
        dark_shift_plot_path = plots_dir / f"python_shift_position_dark_{y}.png"
        save_shift_position_plot(
            shift_position_dark[:, 0],
            shift_position_dark[:, y],
            shift_position_dark_spline,
            dark_shift_plot_path,
            f"SHIFT.POSITION.DARK column {y + 1}",
        )
        dark_shift_plot_paths.append(dark_shift_plot_path)
        shift_position_dark_refined[:, y] = shift_position_dark_spline[:, 1]

    shift_position_dark_refined[:, 1:] = np.round(shift_position_dark_refined[:, 1:], 0)
    shift_position_dark_refined[:199, 1:] = shift_position_dark_refined[199, 1:]

    flattened_dark_retina_raw_refined = np.zeros_like(flattened_dark_retina_raw)
    for z in range(flattened_dark_retina_raw.shape[2]):
        flattened_dark_retina_raw_refined[:, :, z] = shift_rows_to_border(
            flattened_dark_retina_raw[:, :, z],
            shift_position_dark_refined[:, z + 1],
            fill_value=0.0,
        )

    flattened_markers_refined = shift_rows_to_border(
        flattened_markers,
        shift_position_dark_refined[:, 1],
        fill_value=0.0,
    )

    shift_position_light = np.full(
        (flattened_light_retina_raw.shape[0], flattened_light_retina_raw.shape[2] + 1),
        np.nan,
        dtype=np.float64,
    )
    shift_position_light[:, 0] = np.arange(-200.0, 3001.0, 1.0)

    for z in range(flattened_light_retina.shape[2]):
        revise_light = flattened_light_retina_raw[:, :, z]
        for x in range(start_move, end_move + 1, 50):
            row_window = slice(x - 200, x + 200)
            profile = np.nanmean(revise_light[row_window, :], axis=0)
            comparison = np.nanmean(first_grand_mean[row_window, :], axis=0)
            top_range = int(np.nanmax(look_to[row_window, 1]))
            check_rows = np.arange(480, top_range - 1, -1, dtype=np.int64)
            check_idx = check_rows - 1
            check = np.column_stack((check_rows, profile[check_idx], comparison[check_idx]))

            moves = np.arange(-10, 11, 1, dtype=np.int64)
            slide_corr = np.full(moves.shape, np.nan, dtype=np.float64)
            for move_index, move in enumerate(moves):
                if move < 0:
                    slide_corr[move_index] = correlation_estimate(
                        check[(-move):, 1],
                        check[: (check.shape[0] + move), 2],
                    )
                elif move == 0:
                    slide_corr[move_index] = correlation_estimate(check[:, 1], check[:, 2])
                else:
                    slide_corr[move_index] = correlation_estimate(
                        check[: (check.shape[0] - move), 1],
                        check[move:, 2],
                    )
            if not np.all(np.isnan(slide_corr)):
                shift_position_light[x - 1, z + 1] = float(moves[int(np.nanargmax(slide_corr))])

    shift_position_light[:, 1:] = 450.0 - shift_position_light[:, 1:]

    shift_position_light_refined = shift_position_light.copy()
    light_shift_plot_paths: list[Path] = []
    for y in range(1, shift_position_light.shape[1]):
        shift_position_light_sp_maker = np.column_stack((shift_position_light[:, 0], shift_position_light[:, y]))
        valid = ~np.isnan(shift_position_light_sp_maker[:, 1])
        shift_position_light_sp_maker = shift_position_light_sp_maker[valid]
        shift_position_light_sp = fit_smooth_spline_like_r(
            shift_position_light_sp_maker[:, 0],
            shift_position_light_sp_maker[:, 1],
            df=DFforSECONDfit,
        )
        shift_position_light_spline = np.column_stack((shift_x, shift_position_light_sp(shift_x)))
        light_shift_plot_path = plots_dir / f"python_shift_position_light_{y}.png"
        save_shift_position_plot(
            shift_position_light[:, 0],
            shift_position_light[:, y],
            shift_position_light_spline,
            light_shift_plot_path,
            f"SHIFT.POSITION.LIGHT column {y + 1}",
        )
        light_shift_plot_paths.append(light_shift_plot_path)
        shift_position_light_refined[:, y] = shift_position_light_spline[:, 1]

    shift_position_light_refined[:, 1:] = np.round(shift_position_light_refined[:, 1:], 0)
    shift_position_light_refined[:199, 1:] = shift_position_light_refined[199, 1:]

    flattened_light_retina_raw_refined = np.zeros_like(flattened_light_retina_raw)
    for z in range(flattened_light_retina_raw.shape[2]):
        flattened_light_retina_raw_refined[:, :, z] = shift_rows_to_border(
            flattened_light_retina_raw[:, :, z],
            shift_position_light_refined[:, z + 1],
            fill_value=0.0,
        )

    second_grand_mean = flattened_dark_retina_raw_refined[:, :, 0].copy()
    for z in range(1, flattened_dark_retina_raw_refined.shape[2]):
        second_grand_mean = second_grand_mean + flattened_dark_retina_raw_refined[:, :, z]
    for z in range(1, flattened_light_retina_raw_refined.shape[2]):
        second_grand_mean = second_grand_mean + flattened_light_retina_raw_refined[:, :, z]
    second_grand_mean = second_grand_mean / (
        flattened_dark_retina_raw_refined.shape[2] + flattened_light_retina_raw_refined.shape[2]
    )

    sgm = np.asarray(second_grand_mean[39:(second_grand_mean.shape[0] - 39), :], dtype=np.float64)
    best_lat_move_dark = np.column_stack(
        (
            np.arange(1.0, flattened_dark_retina_raw_refined.shape[2] + 1.0, 1.0),
            np.arange(1.0, flattened_dark_retina_raw_refined.shape[2] + 1.0, 1.0),
        )
    )
    for z in range(flattened_dark_retina_raw_refined.shape[2]):
        refine = flattened_dark_retina_raw_refined[:, :, z]
        slide = np.column_stack((np.arange(-39.0, 40.0, 1.0), np.arange(-39.0, 40.0, 1.0)))
        for x in range(-39, 40):
            shifted = refine[(39 + x):((refine.shape[0] - 39) + x), :]
            slide[x + 39, 1] = correlation_estimate(
                np.ravel(sgm, order="F"),
                np.ravel(shifted, order="F"),
            )
        best_lat_move_dark[z, 1] = slide[int(np.nanargmax(slide[:, 1])), 0]

    sgm = np.asarray(second_grand_mean[39:(second_grand_mean.shape[0] - 39), :], dtype=np.float64)
    best_lat_move_light = np.column_stack(
        (
            np.arange(1.0, flattened_light_retina_raw_refined.shape[2] + 1.0, 1.0),
            np.arange(1.0, flattened_light_retina_raw_refined.shape[2] + 1.0, 1.0),
        )
    )
    for z in range(flattened_light_retina_raw_refined.shape[2]):
        refine = flattened_light_retina_raw_refined[:, :, z]
        slide = np.column_stack((np.arange(-39.0, 40.0, 1.0), np.arange(-39.0, 40.0, 1.0)))
        for x in range(-39, 40):
            shifted = refine[(39 + x):((refine.shape[0] - 39) + x), :]
            slide[x + 39, 1] = correlation_estimate(
                np.ravel(sgm, order="F"),
                np.ravel(shifted, order="F"),
            )
        best_lat_move_light[z, 1] = slide[int(np.nanargmax(slide[:, 1])), 0]

    flattened_light_retina_rrc = np.full((2851, 500, flattened_light_retina_raw_refined.shape[2]), np.nan, dtype=np.float64)
    flattened_dark_retina_rrc = np.full((2851, 500, flattened_dark_retina_raw_refined.shape[2]), np.nan, dtype=np.float64)
    flattened_markers_rrc = np.full((2851, 500), np.nan, dtype=np.float64)

    dark_marker_shift = int(best_lat_move_dark[0, 1])
    flattened_markers_rrc = flattened_markers_refined[
        (100 - dark_marker_shift - 1):(2950 - dark_marker_shift),
        :,
    ]
    for z in range(best_lat_move_dark.shape[0]):
        crop_shift = int(best_lat_move_dark[z, 1])
        flattened_dark_retina_rrc = flattened_dark_retina_raw_refined[
            (100 - crop_shift - 1):(2950 - crop_shift),
            :,
            :,
        ]
    for z in range(best_lat_move_light.shape[0]):
        crop_shift = int(best_lat_move_light[z, 1])
        flattened_light_retina_rrc = flattened_light_retina_raw_refined[
            (100 - crop_shift - 1):(2950 - crop_shift),
            :,
            :,
        ]

    final_grand_mean = flattened_dark_retina_rrc[:, :, 0].copy()
    for z in range(1, flattened_dark_retina_rrc.shape[2]):
        final_grand_mean = final_grand_mean + flattened_dark_retina_rrc[:, :, z]
    for z in range(1, flattened_light_retina_rrc.shape[2]):
        final_grand_mean = final_grand_mean + flattened_light_retina_rrc[:, :, z]
    final_grand_mean = final_grand_mean / (
        flattened_dark_retina_rrc.shape[2] + flattened_light_retina_rrc.shape[2]
    )

    grand_profile = np.column_stack(
        (
            np.arange(1.0, 501.0, 1.0),
            np.nanmean(final_grand_mean, axis=0),
        )
    )
    find_vertex_plot_path = plots_dir / f"{REFERENCE_DARK}_find_vertex.png"
    save_profile_plot(
        grand_profile,
        find_vertex_plot_path,
        "Find Vertex",
        verticals=(450.0, 434.0, 466.0),
    )

    grand_profile = grand_profile[433:466, :]
    check_sp = fit_smooth_spline_like_r(grand_profile[:, 0], grand_profile[:, 1], df=10)
    check_x = np.arange(434.0, 467.0, 1.0)
    check_spline = np.column_stack((check_x, check_sp(check_x), check_sp.derivative()(check_x)))
    threshold = float(np.quantile(check_spline[:, 1], 0.25))
    check_spline[:, 2] = np.where(check_spline[:, 1] < threshold, np.nan, check_spline[:, 2])

    positive = np.where(check_spline[:, 2] > 0)[0]
    if positive.size > 0:
        vertex = float(check_spline[positive[-1], 0] + 1.0)
    else:
        check_spline[:, 2] = check_spline[:, 2] - np.nanmedian(check_spline[:, 2])
        nonnegative = np.where(check_spline[:, 2] >= 0)[0]
        if nonnegative.size == 0:
            raise RuntimeError("Could not identify vertex in check.spline.")
        vertex = float(check_spline[nonnegative[-1], 0] + 1.0)

    vertex_plot_path = plots_dir / f"{REFERENCE_DARK}_vertex.png"
    save_profile_plot(
        grand_profile,
        vertex_plot_path,
        "Vertex",
        verticals=(vertex,),
        spline_xy=check_spline[:, 0:2],
    )

    vertex_start = int(vertex) - 431
    vertex_stop = int(vertex) + 30
    flattened_markers_rrc = flattened_markers_rrc[:, vertex_start:vertex_stop]
    flattened_dark_retina_rrc = flattened_dark_retina_rrc[:, vertex_start:vertex_stop, :]
    flattened_light_retina_rrc = flattened_light_retina_rrc[:, vertex_start:vertex_stop, :]

    dark_export = np.transpose(np.nan_to_num(flattened_dark_retina_rrc[:, ::-1, :], nan=0.0), (2, 1, 0)).astype(np.float32)
    light_export = np.transpose(np.nan_to_num(flattened_light_retina_rrc[:, ::-1, :], nan=0.0), (2, 1, 0)).astype(np.float32)
    dark_export_base = outdir / f"_flat_{TO_PROCESS_DARK}"
    light_export_base = outdir / f"_flat_{TO_PROCESS_LIGHT}"
    write_analyze(str(dark_export_base), dark_export)
    write_analyze(str(light_export_base), light_export)

    session_save_path = outdir / f"{TO_PROCESS_DARK}__and__{TO_PROCESS_LIGHT}__flat.npz"
    np.savez_compressed(
        session_save_path,
        flattened_dark_retina_rrc=flattened_dark_retina_rrc,
        flattened_light_retina_rrc=flattened_light_retina_rrc,
        flattened_markers_rrc=flattened_markers_rrc,
        apparent_angles_for_dark=apparent_angles_for_dark,
        apparent_angles_for_light=apparent_angles_for_light,
        best_lat_move_dark=best_lat_move_dark,
        best_lat_move_light=best_lat_move_light,
        vertex=np.asarray([vertex], dtype=np.float64),
    )

    dbg("variable-stats", "Printing summary statistics for translated line-1161 variables")
    show_array_stats("FIRST.GRAND.MEAN", first_grand_mean)
    show_array_stats("ROUGH.VIT.RETINA.POSITION", rough_vit_retina_position)
    show_array_stats("LOOK.TO", look_to)
    show_scalar_stats("window.width.in.pixels", window_width_in_pixels)
    show_scalar_stats("start.move", start_move)
    show_scalar_stats("end.move", end_move)
    show_array_stats("SHIFT.POSITION.DARK", shift_position_dark)
    show_array_stats("SHIFT.POSITION.DARK.REFINED", shift_position_dark_refined)
    show_array_stats("FLATTENED.DARK.RETINA.RAW.REFINED", flattened_dark_retina_raw_refined)
    show_array_stats("FLATTENED.MARKERS.REFINED", flattened_markers_refined)
    show_array_stats("SHIFT.POSITION.LIGHT", shift_position_light)
    show_array_stats("SHIFT.POSITION.LIGHT.REFINED", shift_position_light_refined)
    show_array_stats("FLATTENED.LIGHT.RETINA.RAW.REFINED", flattened_light_retina_raw_refined)
    show_array_stats("SECOND.GRAND.MEAN", second_grand_mean)
    show_array_stats("BEST.LAT.MOVE.DARK", best_lat_move_dark)
    show_array_stats("BEST.LAT.MOVE.LIGHT", best_lat_move_light)
    show_array_stats("FLATTENED.DARK.RETINA.RRC", flattened_dark_retina_rrc)
    show_array_stats("FLATTENED.LIGHT.RETINA.RRC", flattened_light_retina_rrc)
    show_array_stats("FLATTENED.MARKERS.RRC", flattened_markers_rrc)
    show_array_stats("FINAL.GRAND.MEAN", final_grand_mean)
    show_array_stats("GRAND.PROFILE", grand_profile)
    show_array_stats("check.spline", check_spline)
    show_scalar_stats("threshold", threshold)
    show_scalar_stats("vertex", vertex)
    show_scalar_stats("PLOT.SHIFT.POSITION.DARK", ",".join(str(path) for path in dark_shift_plot_paths))
    show_scalar_stats("PLOT.SHIFT.POSITION.LIGHT", ",".join(str(path) for path in light_shift_plot_paths))
    show_scalar_stats("PLOT.FIND.VERTEX", find_vertex_plot_path)
    show_scalar_stats("PLOT.VERTEX", vertex_plot_path)
    show_scalar_stats("EXPORT.DARK", dark_export_base)
    show_scalar_stats("EXPORT.LIGHT", light_export_base)
    show_scalar_stats("SAVE.IMAGE", session_save_path)
    stop_at_boundary(
        "exit-after-save-image",
        "Reached current translated boundary through line 1161; stopping here.",
    )

    return volumes


if __name__ == "__main__":
    try:
        main()
    except StopTranslationBoundary:
        pass
    except Exception:
        print(f"ERROR: failure at step '{DEBUG_STEP}'")
        raise
