"""Step 1 translation of the OCT R script into Python.

This pass mirrors only the R script's startup/configuration and image-loading
block so we can verify the loaded data before translating later sections.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, LSQUnivariateSpline, UnivariateSpline

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
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\behzad\Desktop\flat\output")
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
    volume = np.asarray(read_analyze(path))

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


def save_border_positions_overview_plot(
    series_specs: list[tuple[np.ndarray, str, float]],
    output_path: Path,
    title: str,
    ylim: tuple[float, float],
) -> None:
    """Save the translated border overview plot from the R script."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for values, color, linewidth in series_specs:
        x_values = np.arange(1.0, values.shape[0] + 1.0, 1.0)
        ax.plot(x_values, values, color=color, linewidth=linewidth)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("row")
    ax.set_ylabel("border position")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_border_refinement_plot(
    relative_positions: np.ndarray,
    spline_results: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save the translated plot(...) + matlines(...) refinement plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    split = relative_positions.shape[0]
    x_values = np.arange(1.0, split + 1.0, 1.0)
    ylim_source = relative_positions[:, : min(4, relative_positions.shape[1])]
    ax.plot(x_values, relative_positions[:, min(3, relative_positions.shape[1] - 1)], color="black", linewidth=1.0)
    for col in range(relative_positions.shape[1] - 1, -1, -1):
        ax.plot(x_values, relative_positions[:, col], color="black", linewidth=1.0)
    for col in range(spline_results.shape[1]):
        ax.plot(x_values, spline_results[:, col], color="red", linewidth=1.0)
    ax.set_ylim(float(np.nanmin(ylim_source)), float(np.nanmax(ylim_source)))
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("relative position")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_series_with_spline_line_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    spline_xy: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save simple source-vs-spline line plots from the later R sections."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(x_values, y_values, color="black", linewidth=1.0)
    ax.plot(spline_xy[:, 0], spline_xy[:, 1], color="red", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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
    # R uses seq(from, to, by=(to-from)/500), which should yield 501 points:
    # from + by * 0:500. Build that sequence directly so we avoid np.arange
    # endpoint drift and keep the same floor() inputs as closely as possible.
    dx = (end_x - start_x) / 500.0
    dy = (end_y - start_y) / 500.0
    offsets = np.arange(501, dtype=np.float64)
    line_x = start_x + (dx * offsets)
    line_y = start_y + (dy * offsets)
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


def slice_rows_1based(array: np.ndarray, start_pos: int, end_pos: int) -> np.ndarray:
    """Return an inclusive 1-based row slice like R's a[start:end, ]."""
    return array[(start_pos - 1):end_pos, ...]


def first_closest_zero_crossing(check: np.ndarray, sign_value: float) -> float:
    """Mirror the R fallback used in several half-height checks."""
    hits = np.where(check[:, 3] == sign_value)[0]
    if hits.size > 0:
        return float(check[hits[0], 0])
    closest = np.where(np.abs(check[:, 2]) == np.nanmin(np.abs(check[:, 2])))[0][0]
    return float(check[closest, 0])


def fit_line_coefficients(points: np.ndarray) -> np.ndarray:
    """Return R-style linear-model coefficients [intercept, slope]."""
    arr = np.asarray(points, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    design = np.column_stack((np.ones_like(x), x))
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeffs.astype(np.float64)


def fit_smooth_spline_like_r(x: np.ndarray, y: np.ndarray, df: int, degree: int = 3):
    """Approximate R smooth.spline(df=...) with a penalized B-spline smoother.

    This matches R more closely than a plain regression spline because it
    chooses the smoothing penalty by targeting effective degrees of freedom.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    unique_x, inverse = np.unique(x, return_inverse=True)
    y_sum = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse).astype(np.float64)
    unique_y = y_sum / counts

    if unique_x.size <= degree + 1:
        return UnivariateSpline(unique_x, unique_y, k=min(degree, unique_x.size - 1), s=0)

    if df >= unique_x.size - 1:
        return UnivariateSpline(unique_x, unique_y, k=degree, w=np.sqrt(counts), s=0)

    n_internal = int(min(max(df * 6, 24), max(8, unique_x.size - (degree + 1))))
    probs = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    interior = np.quantile(unique_x, probs)
    interior = np.unique(interior)

    eps = np.finfo(np.float64).eps * max(1.0, float(unique_x[-1] - unique_x[0]))
    interior = interior[(interior > unique_x[0] + eps) & (interior < unique_x[-1] - eps)]
    if interior.size == 0:
        return UnivariateSpline(unique_x, unique_y, k=degree, w=np.sqrt(counts), s=len(unique_x))

    knots = np.concatenate(
        (
            np.repeat(unique_x[0], degree + 1),
            interior,
            np.repeat(unique_x[-1], degree + 1),
        )
    )
    n_coeff = knots.size - degree - 1
    if n_coeff <= degree + 1:
        return UnivariateSpline(unique_x, unique_y, k=degree, w=np.sqrt(counts), s=len(unique_x))

    basis = np.empty((unique_x.size, n_coeff), dtype=np.float64)
    eye = np.eye(n_coeff, dtype=np.float64)
    for idx in range(n_coeff):
        basis[:, idx] = BSpline(knots, eye[idx], degree, extrapolate=True)(unique_x)

    weighted_basis = basis * counts[:, None]
    bt_w_b = basis.T @ weighted_basis
    rhs = basis.T @ (counts * unique_y)
    penalty = np.diff(np.eye(n_coeff, dtype=np.float64), n=2, axis=0)
    penalty = penalty.T @ penalty

    def solve_for_lambda(lam: float) -> tuple[np.ndarray, float]:
        system = bt_w_b + (lam * penalty)
        ridge = np.eye(system.shape[0], dtype=np.float64) * (1e-10 * max(1.0, np.trace(bt_w_b) / system.shape[0]))
        try:
            coeffs = np.linalg.solve(system + ridge, rhs)
            smoother = np.linalg.solve(system + ridge, bt_w_b)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(system + ridge, rhs, rcond=None)[0]
            smoother = np.linalg.lstsq(system + ridge, bt_w_b, rcond=None)[0]
        edf = float(np.trace(smoother))
        return coeffs, edf

    target_df = float(max(2.0, min(df, n_coeff)))
    coeffs_lo, edf_lo = solve_for_lambda(0.0)
    if target_df >= edf_lo:
        return BSpline(knots, coeffs_lo, degree, extrapolate=True)

    lam_lo = 0.0
    lam_hi = 1.0
    coeffs_hi, edf_hi = solve_for_lambda(lam_hi)
    while edf_hi > target_df and lam_hi < 1e12:
        lam_lo = lam_hi
        lam_hi *= 10.0
        coeffs_hi, edf_hi = solve_for_lambda(lam_hi)

    best_coeffs = coeffs_hi
    for _ in range(50):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        coeffs_mid, edf_mid = solve_for_lambda(lam_mid)
        best_coeffs = coeffs_mid
        if abs(edf_mid - target_df) < 1e-3:
            break
        if edf_mid > target_df:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return BSpline(knots, best_coeffs, degree, extrapolate=True)


def r_style_sd(values: np.ndarray) -> float:
    """Mirror R's sd() default sample standard deviation."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return np.nan
    return float(np.std(arr, ddof=1))


def spline_predict_from_series(values: np.ndarray, df: int, split: int, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Fit/predict a smooth spline for one 1D series over 1..split."""
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(1.0, split + 1.0, 1.0)
    if valid_mask is None:
        valid_mask = ~np.isnan(y)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & ~np.isnan(y)

    if valid_mask.sum() == 0:
        return np.full(split, np.nan, dtype=np.float64)
    if valid_mask.sum() == 1:
        out = np.full(split, np.nan, dtype=np.float64)
        out[:] = y[valid_mask][0]
        return out

    model = fit_smooth_spline_like_r(x[valid_mask], y[valid_mask], df=df)
    return np.asarray(model(x), dtype=np.float64)


def r_style_zscore(values: np.ndarray) -> np.ndarray:
    """Mirror the repeated R (x-mean(x))/sd(x) pattern with NA propagation."""
    arr = np.asarray(values, dtype=np.float64)
    out = arr.copy()
    mean_value = float(np.nanmean(arr))
    sd_value = r_style_sd(arr)
    if np.isnan(sd_value) or sd_value == 0.0:
        out[:] = np.nan
        return out
    out = (out - mean_value) / sd_value
    return out


def refine_border_position_pass(
    position_mats: dict[str, np.ndarray],
    order_names: list[str],
    target_name: str,
    z_threshold: float,
    output_dir: Path,
    plot_prefix: str,
) -> list[Path]:
    """Translate the repeated border-cleaning pass used from R lines 1555-2716."""
    split = int(position_mats[order_names[0]].shape[0])
    slice_count = int(position_mats[order_names[0]].shape[1])
    relative_count = len(order_names) - 1
    saved_paths: list[Path] = []

    for x in range(slice_count):
        left = np.column_stack([position_mats[name][:split, x] for name in order_names]).astype(np.float64)
        left_orig = left.copy()
        reference = left[:, -1].copy()
        for col in range(relative_count):
            left[:, col] = left[:, col] - reference

        initial_results = np.full((split, relative_count), np.nan, dtype=np.float64)
        z_deviations = np.full((split, relative_count), np.nan, dtype=np.float64)
        for col in range(relative_count):
            result = spline_predict_from_series(left[:, col], df=5, split=split)
            initial_results[:, col] = result
            z_deviations[:, col] = r_style_zscore(left[:, col] - result)

        overall_avg = np.abs(np.sum(z_deviations, axis=1)) / float(relative_count)
        flags = np.where(np.isnan(overall_avg), np.nan, np.where(overall_avg > z_threshold, 1.0, 0.0))

        refined_results = np.full((split, relative_count), np.nan, dtype=np.float64)
        givebacks = np.full((split, relative_count), np.nan, dtype=np.float64)
        keep_mask = flags == 0.0

        for col in range(relative_count):
            result = spline_predict_from_series(left[:, col], df=5, split=split, valid_mask=keep_mask)
            refined_results[:, col] = result
            replaced = left[:, col].copy()
            replaced[flags == 1.0] = result[flags == 1.0]
            replaced[np.isnan(flags)] = np.nan
            givebacks[:, col] = left_orig[:, col] - replaced

        target_values = np.sum(givebacks, axis=1) / float(relative_count)
        position_mats[target_name][:split, x] = target_values

        output_path = output_dir / f"{plot_prefix}_slice_{x + 1}.png"
        save_border_refinement_plot(
            left[:, :relative_count],
            initial_results,
            output_path,
            f"{target_name} refinement slice {x + 1}",
        )
        saved_paths.append(output_path)

    return saved_paths


def smooth_position_matrix(
    position_matrix: np.ndarray,
    x_values: np.ndarray,
    df: int,
    output_dir: Path,
    plot_prefix: str,
) -> tuple[np.ndarray, list[Path]]:
    """Smooth each slice of a position matrix and save the matching R plots."""
    revised = np.asarray(position_matrix, dtype=np.float64).copy()
    target_x = np.asarray(x_values, dtype=np.float64)
    saved_paths: list[Path] = []

    for z in range(revised.shape[1]):
        revise = np.column_stack((target_x, revised[:, z]))
        valid = ~np.isnan(revise[:, 1])
        revise_valid = revise[valid]
        if revise_valid.shape[0] == 0:
            spline_y = np.full_like(target_x, np.nan, dtype=np.float64)
        elif revise_valid.shape[0] == 1:
            spline_y = np.full_like(target_x, revise_valid[0, 1], dtype=np.float64)
        else:
            spline = fit_smooth_spline_like_r(revise_valid[:, 0], revise_valid[:, 1], df=df)
            spline_y = np.asarray(spline(target_x), dtype=np.float64)

        output_path = output_dir / f"{plot_prefix}_slice_{z + 1}.png"
        save_series_with_spline_line_plot(
            revise_valid[:, 0] if revise_valid.size else target_x,
            revise_valid[:, 1] if revise_valid.size else spline_y,
            np.column_stack((target_x, spline_y)),
            output_path,
            f"{plot_prefix} slice {z + 1}",
        )
        saved_paths.append(output_path)
        revised[:, z] = spline_y

    return revised, saved_paths


def fill_na_with_leading_non_na(values: np.ndarray) -> np.ndarray:
    """Mirror R's REVISE[is.na]=FILL behavior that reuses leading non-NA values."""
    arr = np.asarray(values, dtype=np.float64).copy()
    missing = np.isnan(arr)
    if not missing.any():
        return arr
    fill = arr[~missing]
    if fill.size == 0:
        return arr
    arr[missing] = fill[: np.count_nonzero(missing)]
    return arr


def nearest_depth_index(value: float, depthstrip: np.ndarray) -> int:
    """Translate WHICH.INDEX() from the R script."""
    return int(np.where(np.abs(depthstrip - value) == np.nanmin(np.abs(depthstrip - value)))[0][0] + 1)


def build_lookup_vector(values: np.ndarray, row_start: int, row_end: int, offset: float, depthstrip: np.ndarray) -> np.ndarray:
    """Build a 1-based lookup vector so Python indexing mirrors the R loops."""
    lookup = np.full(values.shape[0] + 1, np.nan, dtype=np.float64)
    for row in range(row_start, row_end + 1):
        value = float(values[row - 1]) + offset
        if not np.isnan(value):
            lookup[row] = float(nearest_depth_index(value, depthstrip))
    return lookup


def build_main_normalized_strip(
    harvest_stack: np.ndarray,
    rpe: np.ndarray,
    olm: np.ndarray,
    onl_opl: np.ndarray,
    inl_ipl: np.ndarray,
    rnfl_gcl: np.ndarray,
    vitreous: np.ndarray,
    row_start: int,
) -> np.ndarray:
    """Translate the main 90-column normalized strip construction."""
    normalized = np.zeros((harvest_stack.shape[0], 90, harvest_stack.shape[2]), dtype=np.float64)
    depthstrip = np.arange(1.0, harvest_stack.shape[1] + 1.0, 1.0)
    row_end = harvest_stack.shape[0]

    for z in range(harvest_stack.shape[2]):
        harvest = np.asarray(harvest_stack[:, :, z], dtype=np.float64)
        a = build_lookup_vector(rpe[:, z], row_start, row_end, 24.0, depthstrip)
        b = build_lookup_vector(rpe[:, z], row_start, row_end, 20.0, depthstrip)
        c = build_lookup_vector(rpe[:, z], row_start, row_end, 16.0, depthstrip)
        d = build_lookup_vector(rpe[:, z], row_start, row_end, 12.0, depthstrip)
        e = build_lookup_vector(rpe[:, z], row_start, row_end, 8.0, depthstrip)
        f = build_lookup_vector(rpe[:, z], row_start, row_end, 4.0, depthstrip)
        g = build_lookup_vector(vitreous[:, z], row_start, row_end, 0.0, depthstrip)
        h = build_lookup_vector(vitreous[:, z], row_start, row_end, -4.0, depthstrip)
        i = build_lookup_vector(vitreous[:, z], row_start, row_end, -8.0, depthstrip)
        j = build_lookup_vector(vitreous[:, z], row_start, row_end, -12.0, depthstrip)

        for x in range(row_start, row_end + 1):
            normalized[x - 1, 0, z] = harvest[x - 1, int(a[x]) - 1]
            normalized[x - 1, 1, z] = harvest[x - 1, int(b[x]) - 1]
            normalized[x - 1, 2, z] = harvest[x - 1, int(c[x]) - 1]
            normalized[x - 1, 3, z] = harvest[x - 1, int(d[x]) - 1]
            normalized[x - 1, 4, z] = harvest[x - 1, int(e[x]) - 1]
            normalized[x - 1, 5, z] = harvest[x - 1, int(f[x]) - 1]
            normalized[x - 1, 86, z] = harvest[x - 1, int(g[x]) - 1]
            normalized[x - 1, 87, z] = harvest[x - 1, int(h[x]) - 1]
            normalized[x - 1, 88, z] = harvest[x - 1, int(i[x]) - 1]
            normalized[x - 1, 89, z] = harvest[x - 1, int(j[x]) - 1]

        for x in range(row_start, row_end + 1):
            startpoint = float(rpe[x - 1, z])
            endpoint = float(olm[x - 1, z])
            values = startpoint + ((endpoint - startpoint) / 18.0) * np.arange(18, dtype=np.float64)
            normalized[x - 1, 6:24, z] = harvest[x - 1, [nearest_depth_index(v, depthstrip) - 1 for v in values]]

            startpoint = float(olm[x - 1, z])
            endpoint = float(onl_opl[x - 1, z])
            values = startpoint + ((endpoint - startpoint) / 16.0) * np.arange(16, dtype=np.float64)
            normalized[x - 1, 24:40, z] = harvest[x - 1, [nearest_depth_index(v, depthstrip) - 1 for v in values]]

            startpoint = float(onl_opl[x - 1, z])
            endpoint = float(inl_ipl[x - 1, z])
            values = startpoint + ((endpoint - startpoint) / 16.0) * np.arange(16, dtype=np.float64)
            normalized[x - 1, 40:56, z] = harvest[x - 1, [nearest_depth_index(v, depthstrip) - 1 for v in values]]

            startpoint = float(inl_ipl[x - 1, z])
            endpoint = float(rnfl_gcl[x - 1, z])
            values = startpoint + ((endpoint - startpoint) / 22.0) * np.arange(22, dtype=np.float64)
            normalized[x - 1, 56:78, z] = harvest[x - 1, [nearest_depth_index(v, depthstrip) - 1 for v in values]]

            startpoint = float(rnfl_gcl[x - 1, z])
            endpoint = float(vitreous[x - 1, z])
            values = startpoint + ((endpoint - startpoint) / 8.0) * np.arange(8, dtype=np.float64)
            normalized[x - 1, 78:86, z] = harvest[x - 1, [nearest_depth_index(v, depthstrip) - 1 for v in values]]

    normalized[np.isnan(normalized)] = 0.0
    return normalized[:, ::-1, :]


def build_fovea_normalized_strip(
    harvest_stack: np.ndarray,
    rpe_fovea: np.ndarray,
    olm_fovea: np.ndarray,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    """Translate the fovea-specific 90-column normalized strip construction."""
    normalized = np.zeros((harvest_stack.shape[0], 90, harvest_stack.shape[2]), dtype=np.float64)
    depthstrip = np.arange(1.0, harvest_stack.shape[1] + 1.0, 1.0)

    for z in range(harvest_stack.shape[2]):
        harvest = np.asarray(harvest_stack[:, :, z], dtype=np.float64)
        a = build_lookup_vector(rpe_fovea[:, z], row_start, row_end, 24.0, depthstrip)
        b = build_lookup_vector(rpe_fovea[:, z], row_start, row_end, 20.0, depthstrip)
        c = build_lookup_vector(rpe_fovea[:, z], row_start, row_end, 16.0, depthstrip)
        d = build_lookup_vector(rpe_fovea[:, z], row_start, row_end, 12.0, depthstrip)
        e = build_lookup_vector(rpe_fovea[:, z], row_start, row_end, 8.0, depthstrip)
        f = build_lookup_vector(rpe_fovea[:, z], row_start, row_end, 4.0, depthstrip)
        g = build_lookup_vector(olm_fovea[:, z], row_start, row_end, 0.0, depthstrip)
        h = build_lookup_vector(olm_fovea[:, z], row_start, row_end, -4.0, depthstrip)
        i = build_lookup_vector(olm_fovea[:, z], row_start, row_end, -8.0, depthstrip)
        j = build_lookup_vector(olm_fovea[:, z], row_start, row_end, -12.0, depthstrip)
        k = build_lookup_vector(olm_fovea[:, z], row_start, row_end, -16.0, depthstrip)
        l = build_lookup_vector(olm_fovea[:, z], row_start, row_end, -20.0, depthstrip)
        m = build_lookup_vector(olm_fovea[:, z], row_start, row_end, -24.0, depthstrip)
        n = build_lookup_vector(olm_fovea[:, z], row_start, row_end, -28.0, depthstrip)
        o = build_lookup_vector(olm_fovea[:, z], row_start, row_end, -32.0, depthstrip)
        p = build_lookup_vector(olm_fovea[:, z], row_start, row_end, -36.0, depthstrip)

        for x in range(row_start, row_end + 1):
            normalized[x - 1, 0, z] = harvest[x - 1, int(a[x]) - 1]
            normalized[x - 1, 1, z] = harvest[x - 1, int(b[x]) - 1]
            normalized[x - 1, 2, z] = harvest[x - 1, int(c[x]) - 1]
            normalized[x - 1, 3, z] = harvest[x - 1, int(d[x]) - 1]
            normalized[x - 1, 4, z] = harvest[x - 1, int(e[x]) - 1]
            normalized[x - 1, 5, z] = harvest[x - 1, int(f[x]) - 1]
            normalized[x - 1, 24, z] = harvest[x - 1, int(g[x]) - 1]
            normalized[x - 1, 25, z] = harvest[x - 1, int(h[x]) - 1]
            normalized[x - 1, 26, z] = harvest[x - 1, int(i[x]) - 1]
            normalized[x - 1, 27, z] = harvest[x - 1, int(j[x]) - 1]
            normalized[x - 1, 28, z] = harvest[x - 1, int(k[x]) - 1]
            normalized[x - 1, 29, z] = harvest[x - 1, int(l[x]) - 1]
            normalized[x - 1, 30, z] = harvest[x - 1, int(m[x]) - 1]
            normalized[x - 1, 31, z] = harvest[x - 1, int(n[x]) - 1]
            normalized[x - 1, 32, z] = harvest[x - 1, int(o[x]) - 1]
            normalized[x - 1, 33, z] = harvest[x - 1, int(p[x]) - 1]

        for x in range(row_start, row_end + 1):
            startpoint = float(rpe_fovea[x - 1, z])
            endpoint = float(olm_fovea[x - 1, z])
            values = startpoint + ((endpoint - startpoint) / 18.0) * np.arange(18, dtype=np.float64)
            normalized[x - 1, 6:24, z] = harvest[x - 1, [nearest_depth_index(v, depthstrip) - 1 for v in values]]

    return normalized[:, ::-1, :]


def build_profile_matrix(volume: np.ndarray, image_indices: list[int], row_slice: slice | None = None) -> np.ndarray:
    """Build the profile matrices used for the final text exports."""
    perc_depth = np.arange(-3.75, 107.5 + 1.25, 1.25, dtype=np.float64)
    profiles = np.column_stack((perc_depth, perc_depth))
    for _ in range(1, volume.shape[2]):
        profiles = np.column_stack((profiles, perc_depth))
    for z in range(volume.shape[2]):
        source = volume[:, :, z] if row_slice is None else volume[row_slice, :, z]
        for y in range(volume.shape[1]):
            profiles[y, z + 1] = float(np.nanmean(source[:, y]))
    return profiles


def format_export_cell(value: object) -> str:
    """Format mixed export-table values in a stable text form."""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "NA"
        rounded = round(float(value), 3)
        text = f"{rounded:.3f}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def write_object_table(table: np.ndarray, output_path: Path) -> None:
    """Write a mixed table in the same row orientation as R's write(t(EXPORT))."""
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in np.asarray(table, dtype=object).T:
            handle.write(" ".join(format_export_cell(cell) for cell in row))
            handle.write("\n")


def main() -> dict[str, np.ndarray]:
    parser = argparse.ArgumentParser(
        description="Step 1: load the four OCT Analyze volumes used by the R script."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing DARK/LIGHT and *_MARKED Analyze files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where plots, Analyze exports, text profiles, and NPZ files are written.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    dbg("identify-layers", "Translating the layer-identification block through line 1443")

    hand_borders = np.full((flattened_dark_retina_rrc.shape[0], 6), np.nan, dtype=np.float64)
    hand_borders[:, 5] = 431.0
    for x in range(flattened_markers_rrc.shape[0]):
        a = np.where(flattened_markers_rrc[x, :] == 254)[0]
        if a.size > 0:
            hand_borders[x, 4] = float(np.mean(a + 1))
        a = np.where(flattened_markers_rrc[x, :] == 253)[0]
        if a.size > 0:
            hand_borders[x, 3] = float(np.mean(a + 1))
        a = np.where(flattened_markers_rrc[x, :] == 252)[0]
        if a.size > 0:
            hand_borders[x, 2] = float(np.mean(a + 1))
        a = np.where(flattened_markers_rrc[x, :] == 250)[0]
        if a.size > 0:
            hand_borders[x, 1] = float(np.mean(a + 1))
        a = np.where(flattened_markers_rrc[x, :] == 249)[0]
        if a.size > 0:
            hand_borders[x, 0] = float(np.mean(a + 1))

    window_width_in_pixels = 40
    start_move = 21
    end_move = (flattened_dark_retina_rrc.shape[0] - start_move) - 1

    true_borders_dark = np.full((flattened_dark_retina_rrc.shape[0], 6, flattened_dark_retina_rrc.shape[2]), np.nan, dtype=np.float64)
    blank = np.full(6, np.nan, dtype=np.float64)
    blank[5] = 431.0

    for z in range(flattened_dark_retina_rrc.shape[2]):
        review = flattened_dark_retina_rrc[:, :, z]
        for x in range(start_move, end_move + 1):
            new_values = blank.copy()
            row_window = slice(x - 20, x + 20)
            profile = np.column_stack(
                (
                    np.arange(1.0, 462.0, 1.0),
                    np.nanmean(review[row_window, :], axis=0),
                )
            )
            segment = np.round(np.nanmean(hand_borders[row_window, :], axis=0))

            if not np.isnan(segment[0]):
                seg1 = int(segment[0])
                check = slice_rows_1based(profile, seg1 - 20, seg1 + 20)
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))[10:31, :]
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[0] = first_closest_zero_crossing(check, 1.0)

            if not np.isnan(segment[1]):
                seg2 = int(segment[1])
                movein = 20.0
                if not np.isnan(segment[2]):
                    movein_alt = np.ceil(segment[2] - segment[1]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = slice_rows_1based(profile, int(new_values[0]), int(seg2 + movein))
                if new_values[0] > seg2:
                    seg2 = int(new_values[0])
                check[:, 1] = np.log(check[:, 1])
                seg2_index = np.where(check[:, 0] == seg2)[0][0]
                m = np.nanmax(check[: (seg2_index + 1), 1])
                peak_start = np.where(check[: (seg2_index + 1), 1] == m)[0][0]
                check = check[peak_start:, :]
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))
                seg2_index = np.where(check[:, 0] == seg2)[0][0]
                if (seg2_index + 11) < check.shape[0]:
                    check = check[: (seg2_index + 11), :]
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[1] = first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[2]):
                seg3 = int(segment[2])
                movein = 20.0
                if not np.isnan(segment[3]):
                    movein_alt = np.ceil(segment[3] - segment[2]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = slice_rows_1based(profile, int(seg3 - movein), int(seg3 + movein))
                m = np.nanmax(check[: int(movein + 1), 1])
                peak_start = np.where(check[: int(movein + 1), 1] == m)[0][0]
                check = check[peak_start:, :]
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))
                seg3_index = np.where(check[:, 0] == seg3)[0][0]
                if (seg3_index + 11) < check.shape[0]:
                    check = check[: (seg3_index + 11), :]
                seg3_index = np.where(check[:, 0] == seg3)[0][0]
                if (seg3_index - 10) > 0:
                    check = check[(seg3_index - 10):, :]
                if check.size == 4:
                    check = np.vstack((check, check))
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[2] = first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[3]):
                seg4 = int(segment[3])
                movein = 20.0
                if not np.isnan(segment[4]):
                    movein_alt = np.ceil(segment[4] - segment[3]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = slice_rows_1based(profile, int(seg4 - movein), int(seg4 + movein))
                m = np.nanmax(check[: int(movein + 1), 1])
                peak_start = np.where(check[: int(movein + 1), 1] == m)[0][0]
                check = check[peak_start:, :]
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))
                seg4_index = np.where(check[:, 0] == seg4)[0][0]
                if (seg4_index + 11) < check.shape[0]:
                    check = check[: (seg4_index + 11), :]
                seg4_index = np.where(check[:, 0] == seg4)[0][0]
                if (seg4_index - 10) > 0:
                    check = check[(seg4_index - 10):, :]
                if check.size == 4:
                    check = np.vstack((check, check))
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[3] = first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[4]):
                seg5 = int(segment[4])
                check = slice_rows_1based(profile, seg5 - 5, seg5 + 5)
                m = np.nanmax(check[:, 1])
                localpeak = float(np.round(np.nanmean(check[np.where(check[:, 1] == m)[0], 0])))
                new_values[4] = localpeak
                low = float(check[2, 0])
                high = float(check[9, 0])
                if (localpeak > low) and (localpeak < high):
                    local_index = np.where(check[:, 0] == localpeak)[0][0]
                    check = check[(local_index - 2):(local_index + 3), :]
                    c_sp = fit_smooth_spline_like_r(check[:, 0], check[:, 1], df=2)
                    c_x = np.arange(check[0, 0], check[4, 0] + 0.1, 0.1)
                    cspline = np.column_stack((c_x, c_sp(c_x)))
                    vertex_local = float(np.mean(cspline[np.where(cspline[:, 1] == np.nanmax(cspline[:, 1]))[0], 0]))
                    if (vertex_local > low) and (vertex_local < high):
                        new_values[4] = float(np.round(vertex_local, 1))

            true_borders_dark[x - 1, :, z] = new_values

    true_borders_light = np.full((flattened_light_retina_rrc.shape[0], 6, flattened_light_retina_rrc.shape[2]), np.nan, dtype=np.float64)
    blank = np.full(6, np.nan, dtype=np.float64)
    blank[5] = 431.0

    for z in range(flattened_light_retina_rrc.shape[2]):
        review = flattened_light_retina_rrc[:, :, z]
        for x in range(start_move, end_move + 1):
            new_values = blank.copy()
            row_window = slice(x - 20, x + 20)
            profile = np.column_stack(
                (
                    np.arange(1.0, 462.0, 1.0),
                    np.nanmean(review[row_window, :], axis=0),
                )
            )
            segment = np.round(np.nanmean(hand_borders[row_window, :], axis=0))

            if not np.isnan(segment[0]):
                seg1 = int(segment[0])
                check = slice_rows_1based(profile, seg1 - 20, seg1 + 20)
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))[10:31, :]
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[0] = first_closest_zero_crossing(check, 1.0)

            if not np.isnan(segment[1]):
                seg2 = int(segment[1])
                movein = 20.0
                if not np.isnan(segment[2]):
                    movein_alt = np.ceil(segment[2] - segment[1]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = slice_rows_1based(profile, int(new_values[0]), int(seg2 + movein))
                if new_values[0] > seg2:
                    seg2 = int(new_values[0])
                check[:, 1] = np.log(check[:, 1])
                seg2_index = np.where(check[:, 0] == seg2)[0][0]
                m = np.nanmax(check[: (seg2_index + 1), 1])
                peak_start = np.where(check[: (seg2_index + 1), 1] == m)[0][0]
                check = check[peak_start:, :]
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))
                seg2_index = np.where(check[:, 0] == seg2)[0][0]
                if (seg2_index + 11) < check.shape[0]:
                    check = check[: (seg2_index + 11), :]
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[1] = first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[2]):
                seg3 = int(segment[2])
                movein = 20.0
                if not np.isnan(segment[3]):
                    movein_alt = np.ceil(segment[3] - segment[2]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = slice_rows_1based(profile, int(seg3 - movein), int(seg3 + movein))
                m = np.nanmax(check[: int(movein + 1), 1])
                peak_start = np.where(check[: int(movein + 1), 1] == m)[0][0]
                check = check[peak_start:, :]
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))
                seg3_index = np.where(check[:, 0] == seg3)[0][0]
                if (seg3_index + 11) < check.shape[0]:
                    check = check[: (seg3_index + 11), :]
                seg3_index = np.where(check[:, 0] == seg3)[0][0]
                if (seg3_index - 10) > 0:
                    check = check[(seg3_index - 10):, :]
                if check.size == 4:
                    check = np.vstack((check, check))
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[2] = first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[3]):
                seg4 = int(segment[3])
                movein = 20.0
                if not np.isnan(segment[4]):
                    movein_alt = np.ceil(segment[4] - segment[3]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = slice_rows_1based(profile, int(seg4 - movein), int(seg4 + movein))
                m = np.nanmax(check[: int(movein + 1), 1])
                peak_start = np.where(check[: int(movein + 1), 1] == m)[0][0]
                check = check[peak_start:, :]
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))
                seg4_index = np.where(check[:, 0] == seg4)[0][0]
                if (seg4_index + 11) < check.shape[0]:
                    check = check[: (seg4_index + 11), :]
                seg4_index = np.where(check[:, 0] == seg4)[0][0]
                if (seg4_index - 10) > 0:
                    check = check[(seg4_index - 10):, :]
                if check.size == 4:
                    check = np.vstack((check, check))
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[3] = first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[4]):
                seg5 = int(segment[4])
                check = slice_rows_1based(profile, seg5 - 5, seg5 + 5)
                m = np.nanmax(check[:, 1])
                localpeak = float(np.round(np.nanmean(check[np.where(check[:, 1] == m)[0], 0])))
                new_values[4] = localpeak
                low = float(check[2, 0])
                high = float(check[9, 0])
                if (localpeak > low) and (localpeak < high):
                    local_index = np.where(check[:, 0] == localpeak)[0][0]
                    check = check[(local_index - 2):(local_index + 3), :]
                    c_sp = fit_smooth_spline_like_r(check[:, 0], check[:, 1], df=2)
                    c_x = np.arange(check[0, 0], check[4, 0] + 0.1, 0.1)
                    cspline = np.column_stack((c_x, c_sp(c_x)))
                    vertex_local = float(np.mean(cspline[np.where(cspline[:, 1] == np.nanmax(cspline[:, 1]))[0], 0]))
                    if (vertex_local > low) and (vertex_local < high):
                        new_values[4] = float(np.round(vertex_local, 1))

            true_borders_light[x - 1, :, z] = new_values

    dbg("variable-stats", "Printing summary statistics for translated layer-identification variables")
    show_array_stats("HAND.BORDERS", hand_borders)
    show_scalar_stats("window.width.in.pixels.layers", window_width_in_pixels)
    show_scalar_stats("start.move.layers", start_move)
    show_scalar_stats("end.move.layers", end_move)
    show_array_stats("TRUE.BORDERS.DARK", true_borders_dark)
    show_array_stats("TRUE.BORDERS.LIGHT", true_borders_light)
    dbg("refine-borders", "Translating the border-refinement block through line 2716")

    dark_rpe_revision_peak = np.full(flattened_dark_retina_rrc.shape[2], np.nan, dtype=np.float64)
    for z in range(flattened_dark_retina_rrc.shape[2]):
        check = np.column_stack(
            (
                np.arange(1.0, 462.0, 1.0),
                np.full(461, np.nan, dtype=np.float64),
            )
        )
        for x in range(429, 434):
            check[x - 1, 1] = float(np.nanmean(flattened_dark_retina_rrc[599:, x - 1, z]))
        check = check[428:433, :]
        c_sp = fit_smooth_spline_like_r(check[:, 0], check[:, 1], df=3)
        cspline_x = np.arange(430.6, 431.4 + 0.001, 0.1)
        cspline = np.column_stack((cspline_x, c_sp(cspline_x)))
        dark_rpe_revision_peak[z] = float(np.mean(cspline[np.where(cspline[:, 1] == np.nanmax(cspline[:, 1]))[0], 0]))
        true_borders_dark[:, 5, z] = 431.0

    light_rpe_revision_peak = np.full(flattened_light_retina_rrc.shape[2], np.nan, dtype=np.float64)
    for z in range(flattened_light_retina_rrc.shape[2]):
        check = np.column_stack(
            (
                np.arange(1.0, 462.0, 1.0),
                np.full(461, np.nan, dtype=np.float64),
            )
        )
        for x in range(429, 434):
            check[x - 1, 1] = float(np.nanmean(flattened_light_retina_rrc[599:, x - 1, z]))
        check = check[428:433, :]
        c_sp = fit_smooth_spline_like_r(check[:, 0], check[:, 1], df=3)
        cspline_x = np.arange(430.6, 431.4 + 0.001, 0.1)
        cspline = np.column_stack((cspline_x, c_sp(cspline_x)))
        light_rpe_revision_peak[z] = float(np.mean(cspline[np.where(cspline[:, 1] == np.nanmax(cspline[:, 1]))[0], 0]))
        true_borders_light[:, 5, z] = 431.0

    vitreous_retina_position_dark = true_borders_dark[:, 0, :].copy()
    rnfl_gcl_position_dark = true_borders_dark[:, 1, :].copy()
    inl_ipl_position_dark = true_borders_dark[:, 2, :].copy()
    onl_opl_position_dark = true_borders_dark[:, 3, :].copy()
    olm_position_dark = true_borders_dark[:, 4, :].copy()
    rpe_position_dark = true_borders_dark[:, 5, :].copy()

    vitreous_retina_position_light = true_borders_light[:, 0, :].copy()
    rnfl_gcl_position_light = true_borders_light[:, 1, :].copy()
    inl_ipl_position_light = true_borders_light[:, 2, :].copy()
    onl_opl_position_light = true_borders_light[:, 3, :].copy()
    olm_position_light = true_borders_light[:, 4, :].copy()
    rpe_position_light = true_borders_light[:, 5, :].copy()

    border_overview_plot_path = plots_dir / "python_border_positions_overview_slice_1.png"
    save_border_positions_overview_plot(
        [
            (vitreous_retina_position_dark[:, 0], "black", 1.0),
            (vitreous_retina_position_light[:, 0], "black", 2.0),
            (rnfl_gcl_position_dark[:, 0], "red", 1.0),
            (rnfl_gcl_position_light[:, 0], "red", 2.0),
            (inl_ipl_position_dark[:, 0], "blue", 1.0),
            (inl_ipl_position_light[:, 0], "blue", 2.0),
            (onl_opl_position_dark[:, 0], "red", 1.0),
            (onl_opl_position_light[:, 0], "red", 2.0),
            (olm_position_dark[:, 0], "blue", 1.0),
            (olm_position_light[:, 0], "blue", 2.0),
            (rpe_position_dark[:, 0], "red", 1.0),
            (rpe_position_light[:, 0], "red", 2.0),
        ],
        border_overview_plot_path,
        "Borders so far",
        ylim=(0.0, 450.0),
    )

    vitreous_retina_position_dark = vitreous_retina_position_dark[599:, :]
    rnfl_gcl_position_dark = rnfl_gcl_position_dark[599:, :]
    inl_ipl_position_dark = inl_ipl_position_dark[599:, :]
    onl_opl_position_dark = onl_opl_position_dark[599:, :]
    olm_position_dark = olm_position_dark[599:, :]
    rpe_position_dark = rpe_position_dark[599:, :]

    vitreous_retina_position_light = vitreous_retina_position_light[599:, :]
    rnfl_gcl_position_light = rnfl_gcl_position_light[599:, :]
    inl_ipl_position_light = inl_ipl_position_light[599:, :]
    onl_opl_position_light = onl_opl_position_light[599:, :]
    olm_position_light = olm_position_light[599:, :]
    rpe_position_light = rpe_position_light[599:, :]

    position_mats_dark = {
        "VITREOUS.RETINA.POSITION.DARK": vitreous_retina_position_dark,
        "RNFL.GCL.POSITION.DARK": rnfl_gcl_position_dark,
        "INL.IPL.POSITION.DARK": inl_ipl_position_dark,
        "ONL.OPL.POSITION.DARK": onl_opl_position_dark,
        "OLM.POSITION.DARK": olm_position_dark,
        "RPE.POSITION.DARK": rpe_position_dark,
    }
    position_mats_light = {
        "VITREOUS.RETINA.POSITION.LIGHT": vitreous_retina_position_light,
        "RNFL.GCL.POSITION.LIGHT": rnfl_gcl_position_light,
        "INL.IPL.POSITION.LIGHT": inl_ipl_position_light,
        "ONL.OPL.POSITION.LIGHT": onl_opl_position_light,
        "OLM.POSITION.LIGHT": olm_position_light,
        "RPE.POSITION.LIGHT": rpe_position_light,
    }

    dark_inl_ipl_plot_paths = refine_border_position_pass(
        position_mats_dark,
        [
            "RPE.POSITION.DARK",
            "OLM.POSITION.DARK",
            "VITREOUS.RETINA.POSITION.DARK",
            "ONL.OPL.POSITION.DARK",
            "INL.IPL.POSITION.DARK",
        ],
        "INL.IPL.POSITION.DARK",
        z_threshold=2.0,
        output_dir=plots_dir,
        plot_prefix="python_dark_inl_ipl_refinement",
    )
    dark_onl_opl_plot_paths = refine_border_position_pass(
        position_mats_dark,
        [
            "INL.IPL.POSITION.DARK",
            "RPE.POSITION.DARK",
            "OLM.POSITION.DARK",
            "VITREOUS.RETINA.POSITION.DARK",
            "ONL.OPL.POSITION.DARK",
        ],
        "ONL.OPL.POSITION.DARK",
        z_threshold=2.0,
        output_dir=plots_dir,
        plot_prefix="python_dark_onl_opl_refinement",
    )
    dark_vitreous_plot_paths = refine_border_position_pass(
        position_mats_dark,
        [
            "ONL.OPL.POSITION.DARK",
            "INL.IPL.POSITION.DARK",
            "RPE.POSITION.DARK",
            "OLM.POSITION.DARK",
            "VITREOUS.RETINA.POSITION.DARK",
        ],
        "VITREOUS.RETINA.POSITION.DARK",
        z_threshold=2.0,
        output_dir=plots_dir,
        plot_prefix="python_dark_vitreous_retina_refinement",
    )
    dark_olm_plot_paths = refine_border_position_pass(
        position_mats_dark,
        [
            "VITREOUS.RETINA.POSITION.DARK",
            "ONL.OPL.POSITION.DARK",
            "INL.IPL.POSITION.DARK",
            "RPE.POSITION.DARK",
            "OLM.POSITION.DARK",
        ],
        "OLM.POSITION.DARK",
        z_threshold=3.0,
        output_dir=plots_dir,
        plot_prefix="python_dark_olm_refinement",
    )
    dark_rnfl_gcl_plot_paths = refine_border_position_pass(
        position_mats_dark,
        [
            "VITREOUS.RETINA.POSITION.DARK",
            "ONL.OPL.POSITION.DARK",
            "INL.IPL.POSITION.DARK",
            "RPE.POSITION.DARK",
            "OLM.POSITION.DARK",
            "RNFL.GCL.POSITION.DARK",
        ],
        "RNFL.GCL.POSITION.DARK",
        z_threshold=2.0,
        output_dir=plots_dir,
        plot_prefix="python_dark_rnfl_gcl_refinement",
    )

    light_inl_ipl_plot_paths = refine_border_position_pass(
        position_mats_light,
        [
            "RPE.POSITION.LIGHT",
            "OLM.POSITION.LIGHT",
            "VITREOUS.RETINA.POSITION.LIGHT",
            "ONL.OPL.POSITION.LIGHT",
            "INL.IPL.POSITION.LIGHT",
        ],
        "INL.IPL.POSITION.LIGHT",
        z_threshold=2.0,
        output_dir=plots_dir,
        plot_prefix="python_light_inl_ipl_refinement",
    )
    light_onl_opl_plot_paths = refine_border_position_pass(
        position_mats_light,
        [
            "INL.IPL.POSITION.LIGHT",
            "RPE.POSITION.LIGHT",
            "OLM.POSITION.LIGHT",
            "VITREOUS.RETINA.POSITION.LIGHT",
            "ONL.OPL.POSITION.LIGHT",
        ],
        "ONL.OPL.POSITION.LIGHT",
        z_threshold=2.0,
        output_dir=plots_dir,
        plot_prefix="python_light_onl_opl_refinement",
    )
    light_vitreous_plot_paths = refine_border_position_pass(
        position_mats_light,
        [
            "ONL.OPL.POSITION.LIGHT",
            "INL.IPL.POSITION.LIGHT",
            "RPE.POSITION.LIGHT",
            "OLM.POSITION.LIGHT",
            "VITREOUS.RETINA.POSITION.LIGHT",
        ],
        "VITREOUS.RETINA.POSITION.LIGHT",
        z_threshold=2.0,
        output_dir=plots_dir,
        plot_prefix="python_light_vitreous_retina_refinement",
    )
    light_olm_plot_paths = refine_border_position_pass(
        position_mats_light,
        [
            "VITREOUS.RETINA.POSITION.LIGHT",
            "ONL.OPL.POSITION.LIGHT",
            "INL.IPL.POSITION.LIGHT",
            "RPE.POSITION.LIGHT",
            "OLM.POSITION.LIGHT",
        ],
        "OLM.POSITION.LIGHT",
        z_threshold=3.0,
        output_dir=plots_dir,
        plot_prefix="python_light_olm_refinement",
    )
    light_rnfl_gcl_plot_paths = refine_border_position_pass(
        position_mats_light,
        [
            "VITREOUS.RETINA.POSITION.LIGHT",
            "ONL.OPL.POSITION.LIGHT",
            "INL.IPL.POSITION.LIGHT",
            "RPE.POSITION.LIGHT",
            "OLM.POSITION.LIGHT",
            "RNFL.GCL.POSITION.LIGHT",
        ],
        "RNFL.GCL.POSITION.LIGHT",
        z_threshold=2.0,
        output_dir=plots_dir,
        plot_prefix="python_light_rnfl_gcl_refinement",
    )

    dbg("variable-stats", "Printing summary statistics for translated border-refinement variables")
    show_array_stats("TRUE.BORDERS.DARK", true_borders_dark)
    show_array_stats("TRUE.BORDERS.LIGHT", true_borders_light)
    show_array_stats("RPE.REVISION.PEAK.DARK", dark_rpe_revision_peak)
    show_array_stats("RPE.REVISION.PEAK.LIGHT", light_rpe_revision_peak)
    show_array_stats("VITREOUS.RETINA.POSITION.DARK", vitreous_retina_position_dark)
    show_array_stats("RNFL.GCL.POSITION.DARK", rnfl_gcl_position_dark)
    show_array_stats("INL.IPL.POSITION.DARK", inl_ipl_position_dark)
    show_array_stats("ONL.OPL.POSITION.DARK", onl_opl_position_dark)
    show_array_stats("OLM.POSITION.DARK", olm_position_dark)
    show_array_stats("RPE.POSITION.DARK", rpe_position_dark)
    show_array_stats("VITREOUS.RETINA.POSITION.LIGHT", vitreous_retina_position_light)
    show_array_stats("RNFL.GCL.POSITION.LIGHT", rnfl_gcl_position_light)
    show_array_stats("INL.IPL.POSITION.LIGHT", inl_ipl_position_light)
    show_array_stats("ONL.OPL.POSITION.LIGHT", onl_opl_position_light)
    show_array_stats("OLM.POSITION.LIGHT", olm_position_light)
    show_array_stats("RPE.POSITION.LIGHT", rpe_position_light)
    show_scalar_stats("PLOT.BORDER.OVERVIEW", border_overview_plot_path)
    show_scalar_stats("PLOT.DARK.INL.IPL", ",".join(str(path) for path in dark_inl_ipl_plot_paths))
    show_scalar_stats("PLOT.DARK.ONL.OPL", ",".join(str(path) for path in dark_onl_opl_plot_paths))
    show_scalar_stats("PLOT.DARK.VITREOUS.RETINA", ",".join(str(path) for path in dark_vitreous_plot_paths))
    show_scalar_stats("PLOT.DARK.OLM", ",".join(str(path) for path in dark_olm_plot_paths))
    show_scalar_stats("PLOT.DARK.RNFL.GCL", ",".join(str(path) for path in dark_rnfl_gcl_plot_paths))
    show_scalar_stats("PLOT.LIGHT.INL.IPL", ",".join(str(path) for path in light_inl_ipl_plot_paths))
    show_scalar_stats("PLOT.LIGHT.ONL.OPL", ",".join(str(path) for path in light_onl_opl_plot_paths))
    show_scalar_stats("PLOT.LIGHT.VITREOUS.RETINA", ",".join(str(path) for path in light_vitreous_plot_paths))
    show_scalar_stats("PLOT.LIGHT.OLM", ",".join(str(path) for path in light_olm_plot_paths))
    show_scalar_stats("PLOT.LIGHT.RNFL.GCL", ",".join(str(path) for path in light_rnfl_gcl_plot_paths))
    dbg("final-sections", "Translating the remainder of the R script through the final exports")

    refined_border_overview_plot_path = plots_dir / "python_refined_border_positions_overview_slice_1.png"
    save_border_positions_overview_plot(
        [
            (vitreous_retina_position_dark[:, 0], "black", 1.0),
            (vitreous_retina_position_light[:, 0], "black", 2.0),
            (rnfl_gcl_position_dark[:, 0], "red", 1.0),
            (rnfl_gcl_position_light[:, 0], "red", 2.0),
            (inl_ipl_position_dark[:, 0], "blue", 1.0),
            (inl_ipl_position_light[:, 0], "blue", 2.0),
            (onl_opl_position_dark[:, 0], "red", 1.0),
            (onl_opl_position_light[:, 0], "red", 2.0),
            (olm_position_dark[:, 0], "blue", 1.0),
            (olm_position_light[:, 0], "blue", 2.0),
            (rpe_position_dark[:, 0], "red", 1.0),
            (rpe_position_light[:, 0], "red", 2.0),
        ],
        refined_border_overview_plot_path,
        "Refined borders",
        ylim=(0.0, 450.0),
    )

    x_499_to_2750 = np.arange(499.0, 2750.0 + 1.0, 1.0)
    r_vitreous_retina_position_dark, revised_vitreous_dark_plot_paths = smooth_position_matrix(
        vitreous_retina_position_dark,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_vitreous_retina_dark",
    )
    r_rnfl_gcl_position_dark, revised_rnfl_dark_plot_paths = smooth_position_matrix(
        rnfl_gcl_position_dark,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_rnfl_gcl_dark",
    )
    r_inl_ipl_position_dark, revised_inl_dark_plot_paths = smooth_position_matrix(
        inl_ipl_position_dark,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_inl_ipl_dark",
    )
    r_onl_opl_position_dark, revised_onl_dark_plot_paths = smooth_position_matrix(
        onl_opl_position_dark,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_onl_opl_dark",
    )
    r_olm_position_dark, revised_olm_dark_plot_paths = smooth_position_matrix(
        olm_position_dark,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_olm_dark",
    )
    r_rpe_position_dark = np.asarray(rpe_position_dark, dtype=np.float64).copy()
    for z in range(r_rpe_position_dark.shape[1]):
        r_rpe_position_dark[:, z] = fill_na_with_leading_non_na(r_rpe_position_dark[:, z])

    r_vitreous_retina_position_light, revised_vitreous_light_plot_paths = smooth_position_matrix(
        vitreous_retina_position_light,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_vitreous_retina_light",
    )
    r_rnfl_gcl_position_light, revised_rnfl_light_plot_paths = smooth_position_matrix(
        rnfl_gcl_position_light,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_rnfl_gcl_light",
    )
    r_inl_ipl_position_light, revised_inl_light_plot_paths = smooth_position_matrix(
        inl_ipl_position_light,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_inl_ipl_light",
    )
    r_onl_opl_position_light, revised_onl_light_plot_paths = smooth_position_matrix(
        onl_opl_position_light,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_onl_opl_light",
    )
    r_olm_position_light, revised_olm_light_plot_paths = smooth_position_matrix(
        olm_position_light,
        x_499_to_2750,
        df=11,
        output_dir=plots_dir,
        plot_prefix="python_revised_olm_light",
    )
    r_rpe_position_light = np.asarray(rpe_position_light, dtype=np.float64).copy()
    for z in range(r_rpe_position_light.shape[1]):
        r_rpe_position_light[:, z] = fill_na_with_leading_non_na(r_rpe_position_light[:, z])

    pad_dark = np.full((599, r_vitreous_retina_position_dark.shape[1]), np.nan, dtype=np.float64)
    r_vitreous_retina_position_dark = np.vstack((pad_dark, r_vitreous_retina_position_dark))
    r_rnfl_gcl_position_dark = np.vstack((pad_dark, r_rnfl_gcl_position_dark))
    r_inl_ipl_position_dark = np.vstack((pad_dark, r_inl_ipl_position_dark))
    r_onl_opl_position_dark = np.vstack((pad_dark, r_onl_opl_position_dark))
    r_olm_position_dark = np.vstack((pad_dark, r_olm_position_dark))
    r_rpe_position_dark = np.vstack((pad_dark, r_rpe_position_dark))

    pad_light = np.full((600, r_vitreous_retina_position_light.shape[1]), np.nan, dtype=np.float64)
    r_vitreous_retina_position_light = np.vstack((pad_light, r_vitreous_retina_position_light))
    r_rnfl_gcl_position_light = np.vstack((pad_light, r_rnfl_gcl_position_light))
    r_inl_ipl_position_light = np.vstack((pad_light, r_inl_ipl_position_light))
    r_onl_opl_position_light = np.vstack((pad_light, r_onl_opl_position_light))
    r_olm_position_light = np.vstack((pad_light, r_olm_position_light))
    r_rpe_position_light = np.vstack((pad_light, r_rpe_position_light))

    main_dark_outputs = np.column_stack(
        (
            np.asarray(IMAGE_INDEX_DARK, dtype=np.float64),
            np.asarray(IMAGE_INDEX_DARK, dtype=np.float64),
            np.asarray(IMAGE_INDEX_DARK, dtype=np.float64),
            np.asarray(IMAGE_INDEX_DARK, dtype=np.float64),
        )
    )
    for x in range(main_dark_outputs.shape[0]):
        main_dark_outputs[x, 1] = float(np.nanmean(r_rpe_position_dark[:, x] - r_vitreous_retina_position_dark[:, x]))
        main_dark_outputs[x, 2] = float(np.nanmean(r_rpe_position_dark[:, x] - r_olm_position_dark[:, x]))
        main_dark_outputs[x, 3] = float(np.nanmean(r_rnfl_gcl_position_dark[:, x] - r_vitreous_retina_position_dark[:, x]))

    main_light_outputs = np.column_stack(
        (
            np.asarray(IMAGE_INDEX_LIGHT, dtype=np.float64),
            np.asarray(IMAGE_INDEX_LIGHT, dtype=np.float64),
            np.asarray(IMAGE_INDEX_LIGHT, dtype=np.float64),
            np.asarray(IMAGE_INDEX_LIGHT, dtype=np.float64),
        )
    )
    for x in range(main_light_outputs.shape[0]):
        main_light_outputs[x, 1] = float(np.nanmean(r_rpe_position_light[:, x] - r_vitreous_retina_position_light[:, x]))
        main_light_outputs[x, 2] = float(np.nanmean(r_rpe_position_light[:, x] - r_olm_position_light[:, x]))
        main_light_outputs[x, 3] = float(np.nanmean(r_rnfl_gcl_position_light[:, x] - r_vitreous_retina_position_light[:, x]))

    flattened_dark_retina_rrc_n = build_main_normalized_strip(
        flattened_dark_retina_rrc,
        r_rpe_position_dark,
        r_olm_position_dark,
        r_onl_opl_position_dark,
        r_inl_ipl_position_dark,
        r_rnfl_gcl_position_dark,
        r_vitreous_retina_position_dark,
        row_start=601,
    )
    flattened_dark_retina_rrc_n_profiles = build_profile_matrix(flattened_dark_retina_rrc_n, IMAGE_INDEX_DARK)
    dark_main_profile_plot_path = plots_dir / "python_dark_main_normalized_profile_slice_1.png"
    save_profile_plot(
        flattened_dark_retina_rrc_n_profiles[:, [0, 1]],
        dark_main_profile_plot_path,
        "Dark Main Normalized Profile",
        verticals=(0.0, 10.0, 37.5, 57.5, 77.5, 100.0),
    )

    flattened_light_retina_rrc_n = build_main_normalized_strip(
        flattened_light_retina_rrc,
        r_rpe_position_light[: flattened_light_retina_rrc.shape[0], :],
        r_olm_position_light[: flattened_light_retina_rrc.shape[0], :],
        r_onl_opl_position_light[: flattened_light_retina_rrc.shape[0], :],
        r_inl_ipl_position_light[: flattened_light_retina_rrc.shape[0], :],
        r_rnfl_gcl_position_light[: flattened_light_retina_rrc.shape[0], :],
        r_vitreous_retina_position_light[: flattened_light_retina_rrc.shape[0], :],
        row_start=601,
    )
    flattened_light_retina_rrc_n_profiles = build_profile_matrix(flattened_light_retina_rrc_n, IMAGE_INDEX_LIGHT)
    light_main_profile_plot_path = plots_dir / "python_light_main_normalized_profile_slice_1.png"
    save_profile_plot(
        flattened_light_retina_rrc_n_profiles[:, [0, 1]],
        light_main_profile_plot_path,
        "Light Main Normalized Profile",
        verticals=(0.0, 10.0, 37.5, 57.5, 77.5, 100.0),
    )

    dark_norm_export = np.transpose(np.nan_to_num(flattened_dark_retina_rrc_n[:, ::-1, :], nan=0.0), (2, 1, 0)).astype(np.float32)
    light_norm_export = np.transpose(np.nan_to_num(flattened_light_retina_rrc_n[:, ::-1, :], nan=0.0), (2, 1, 0)).astype(np.float32)
    dark_norm_export_base = outdir / f"_flat-normed_{TO_PROCESS_DARK}"
    light_norm_export_base = outdir / f"_flat-normed_{TO_PROCESS_LIGHT}"
    write_analyze(str(dark_norm_export_base), dark_norm_export)
    write_analyze(str(light_norm_export_base), light_norm_export)

    r_rpe_position_dark_fovea = true_borders_dark[20:181, 5, :].copy()
    r_olm_position_dark_fovea_raw = true_borders_dark[20:181, 4, :].copy()
    x_neg80_to_80 = np.arange(-80.0, 80.0 + 1.0, 1.0)
    r_olm_position_dark_fovea_raw, dark_fovea_olm_plot_paths = smooth_position_matrix(
        r_olm_position_dark_fovea_raw,
        x_neg80_to_80,
        df=3,
        output_dir=plots_dir,
        plot_prefix="python_dark_fovea_olm",
    )
    r_olm_position_dark_fovea2 = r_olm_position_dark_fovea_raw[30:131, :].copy()
    r_rpe_position_dark_fovea2 = r_rpe_position_dark_fovea[30:131, :].copy()
    r_olm_position_dark_fovea = r_olm_position_dark.copy()
    r_olm_position_dark_fovea[50:151, :] = r_olm_position_dark_fovea2
    r_rpe_position_dark_fovea_full = r_rpe_position_dark.copy()
    r_rpe_position_dark_fovea_full[50:151, :] = r_rpe_position_dark_fovea2

    main_dark_outputs_fovea = main_dark_outputs[:, [0, 2]].copy()
    main_dark_outputs_fovea[:, 1] = np.nan
    for x in range(main_dark_outputs_fovea.shape[0]):
        main_dark_outputs_fovea[x, 1] = float(
            np.nanmean(r_rpe_position_dark_fovea_full[50:151, x] - r_olm_position_dark_fovea[50:151, x])
        )

    flattened_dark_retina_rrc_n_fovea = build_fovea_normalized_strip(
        flattened_dark_retina_rrc,
        r_rpe_position_dark_fovea_full,
        r_olm_position_dark_fovea,
        row_start=51,
        row_end=151,
    )
    flattened_dark_retina_rrc_n_fovea_profiles = build_profile_matrix(
        flattened_dark_retina_rrc_n_fovea,
        IMAGE_INDEX_DARK,
        row_slice=slice(50, 151),
    )
    dark_fovea_profile_plot_path = plots_dir / "python_dark_fovea_normalized_profile_slice_1.png"
    save_profile_plot(
        flattened_dark_retina_rrc_n_fovea_profiles[:, [0, 1]],
        dark_fovea_profile_plot_path,
        "Dark Fovea Normalized Profile",
        verticals=(0.0, 10.0, 37.5, 57.5, 77.5, 100.0),
    )
    flattened_dark_retina_rrc_n_fovea_profiles = flattened_dark_retina_rrc_n_fovea_profiles[56:90, :]
    flattened_dark_retina_rrc_n[49:152, :, :] = flattened_dark_retina_rrc_n_fovea[49:152, :, :]

    r_rpe_position_light_fovea = true_borders_light[20:181, 5, :].copy()
    r_olm_position_light_fovea_raw = true_borders_light[20:181, 4, :].copy()
    r_olm_position_light_fovea_raw, light_fovea_olm_plot_paths = smooth_position_matrix(
        r_olm_position_light_fovea_raw,
        x_neg80_to_80,
        df=3,
        output_dir=plots_dir,
        plot_prefix="python_light_fovea_olm",
    )
    r_olm_position_light_fovea2 = r_olm_position_light_fovea_raw[30:131, :].copy()
    r_rpe_position_light_fovea2 = r_rpe_position_light_fovea[30:131, :].copy()
    r_olm_position_light_fovea = r_olm_position_light.copy()
    r_olm_position_light_fovea[50:151, :] = r_olm_position_light_fovea2
    r_rpe_position_light_fovea_full = r_rpe_position_light.copy()
    r_rpe_position_light_fovea_full[50:151, :] = r_rpe_position_light_fovea2

    main_light_outputs_fovea = main_light_outputs[:, [0, 2]].copy()
    main_light_outputs_fovea[:, 1] = np.nan
    for x in range(main_light_outputs_fovea.shape[0]):
        main_light_outputs_fovea[x, 1] = float(
            np.nanmean(r_rpe_position_light_fovea_full[50:151, x] - r_olm_position_light_fovea[50:151, x])
        )

    flattened_light_retina_rrc_n_fovea = build_fovea_normalized_strip(
        flattened_light_retina_rrc,
        r_rpe_position_light_fovea_full[: flattened_light_retina_rrc.shape[0], :],
        r_olm_position_light_fovea[: flattened_light_retina_rrc.shape[0], :],
        row_start=51,
        row_end=151,
    )
    flattened_light_retina_rrc_n_fovea_profiles = build_profile_matrix(
        flattened_light_retina_rrc_n_fovea,
        IMAGE_INDEX_LIGHT,
        row_slice=slice(50, 151),
    )
    light_fovea_profile_plot_path = plots_dir / "python_light_fovea_normalized_profile_slice_1.png"
    save_profile_plot(
        flattened_light_retina_rrc_n_fovea_profiles[:, [0, 1]],
        light_fovea_profile_plot_path,
        "Light Fovea Normalized Profile",
        verticals=(0.0, 10.0, 37.5, 57.5, 77.5, 100.0),
    )
    flattened_light_retina_rrc_n_fovea_profiles = flattened_light_retina_rrc_n_fovea_profiles[56:90, :]
    flattened_light_retina_rrc_n[49:152, :, :] = flattened_light_retina_rrc_n_fovea[49:152, :, :]

    dark_norm_export = np.transpose(np.nan_to_num(flattened_dark_retina_rrc_n[:, ::-1, :], nan=0.0), (2, 1, 0)).astype(np.float32)
    light_norm_export = np.transpose(np.nan_to_num(flattened_light_retina_rrc_n[:, ::-1, :], nan=0.0), (2, 1, 0)).astype(np.float32)
    write_analyze(str(dark_norm_export_base), dark_norm_export)
    write_analyze(str(light_norm_export_base), light_norm_export)

    main_dark_outputs = np.column_stack(
        (
            main_dark_outputs[:, 0],
            apparent_angles_for_dark[:, 2],
            main_dark_outputs[:, 1:4],
        )
    )
    main_light_outputs = np.column_stack(
        (
            main_light_outputs[:, 0],
            apparent_angles_for_light[:, 2],
            main_light_outputs[:, 1:4],
        )
    )
    main_dark_outputs_fovea = np.column_stack(
        (
            main_dark_outputs_fovea[:, 0],
            apparent_angles_for_dark[:, 1],
            main_dark_outputs_fovea[:, 1],
        )
    )
    main_light_outputs_fovea = np.column_stack(
        (
            main_light_outputs_fovea[:, 0],
            apparent_angles_for_light[:, 1],
            main_light_outputs_fovea[:, 1],
        )
    )

    dark_profiles_table = np.round(
        np.column_stack((main_dark_outputs, flattened_dark_retina_rrc_n_profiles[:, 1:].T)),
        3,
    )
    dark_profiles_export = np.vstack((dark_profiles_table[0:1, :], dark_profiles_table)).astype(object)
    dark_profiles_export[0, :] = np.asarray(
        [TO_PROCESS_DARK, "angle", "whole", "RPEtoOLM", "RNFL", *np.arange(-3.75, 107.5 + 1.25, 1.25)],
        dtype=object,
    )
    dark_profiles_txt_path = outdir / f"_dark_profiles_{TO_PROCESS_DARK}.txt"
    write_object_table(dark_profiles_export, dark_profiles_txt_path)

    light_profiles_table = np.round(
        np.column_stack((main_light_outputs, flattened_light_retina_rrc_n_profiles[:, 1:].T)),
        3,
    )
    light_profiles_export = np.vstack((light_profiles_table[0:1, :], light_profiles_table)).astype(object)
    light_profiles_export[0, :] = np.asarray(
        [TO_PROCESS_LIGHT, "angle", "whole", "RPEtoOLM", "RNFL", *np.arange(-3.75, 107.5 + 1.25, 1.25)],
        dtype=object,
    )
    light_profiles_txt_path = outdir / f"_light_profiles_{TO_PROCESS_LIGHT}.txt"
    write_object_table(light_profiles_export, light_profiles_txt_path)

    dark_fovea_export_left = np.column_stack(
        (
            main_dark_outputs_fovea[:, 0],
            main_dark_outputs_fovea[:, 1],
            np.full(main_dark_outputs_fovea.shape[0], np.nan, dtype=np.float64),
            main_dark_outputs_fovea[:, 2],
            np.full(main_dark_outputs_fovea.shape[0], np.nan, dtype=np.float64),
        )
    )
    dark_fovea_export_right = flattened_dark_retina_rrc_n_fovea_profiles[:, 1:]
    dark_fovea_buffer = np.full((56, dark_fovea_export_right.shape[1]), np.nan, dtype=np.float64)
    dark_fovea_export_right = np.vstack((dark_fovea_buffer, dark_fovea_export_right)).T
    dark_fovea_table = np.round(np.column_stack((dark_fovea_export_left, dark_fovea_export_right)), 3)
    dark_fovea_export = np.vstack((dark_fovea_table[0:1, :], dark_fovea_table)).astype(object)
    dark_fovea_export[0, :] = np.asarray(
        [f"fovea{TO_PROCESS_DARK}", "angle", "whole", "RPEtoOLM", "RNFL", *np.arange(-3.75, 107.5 + 1.25, 1.25)],
        dtype=object,
    )
    dark_fovea_txt_path = outdir / f"_fovea_dark_profiles_{TO_PROCESS_DARK}.txt"
    write_object_table(dark_fovea_export, dark_fovea_txt_path)

    light_fovea_export_left = np.column_stack(
        (
            main_light_outputs_fovea[:, 0],
            main_light_outputs_fovea[:, 1],
            np.full(main_light_outputs_fovea.shape[0], np.nan, dtype=np.float64),
            main_light_outputs_fovea[:, 2],
            np.full(main_light_outputs_fovea.shape[0], np.nan, dtype=np.float64),
        )
    )
    light_fovea_export_right = flattened_light_retina_rrc_n_fovea_profiles[:, 1:]
    light_fovea_buffer = np.full((56, light_fovea_export_right.shape[1]), np.nan, dtype=np.float64)
    light_fovea_export_right = np.vstack((light_fovea_buffer, light_fovea_export_right)).T
    light_fovea_table = np.round(np.column_stack((light_fovea_export_left, light_fovea_export_right)), 3)
    light_fovea_export = np.vstack((light_fovea_table[0:1, :], light_fovea_table)).astype(object)
    light_fovea_export[0, :] = np.asarray(
        [f"fovea{TO_PROCESS_LIGHT}", "angle", "whole", "RPEtoOLM", "RNFL", *np.arange(-3.75, 107.5 + 1.25, 1.25)],
        dtype=object,
    )
    light_fovea_txt_path = outdir / f"_fovea_light_profiles_{TO_PROCESS_LIGHT}.txt"
    write_object_table(light_fovea_export, light_fovea_txt_path)

    final_save_path = outdir / f"_done_{TO_PROCESS_DARK}__and__{TO_PROCESS_LIGHT}.npz"
    np.savez_compressed(
        final_save_path,
        MAIN_DARK_OUTPUTS=main_dark_outputs,
        MAIN_LIGHT_OUTPUTS=main_light_outputs,
        MAIN_DARK_OUTPUTS_fovea=main_dark_outputs_fovea,
        MAIN_LIGHT_OUTPUTS_fovea=main_light_outputs_fovea,
        FLATTENED_DARK_RETINA_RRC_N=flattened_dark_retina_rrc_n,
        FLATTENED_LIGHT_RETINA_RRC_N=flattened_light_retina_rrc_n,
        FLATTENED_DARK_RETINA_RRC_N_profiles=flattened_dark_retina_rrc_n_profiles,
        FLATTENED_LIGHT_RETINA_RRC_N_profiles=flattened_light_retina_rrc_n_profiles,
        FLATTENED_DARK_RETINA_RRC_N_fovea_profiles=flattened_dark_retina_rrc_n_fovea_profiles,
        FLATTENED_LIGHT_RETINA_RRC_N_fovea_profiles=flattened_light_retina_rrc_n_fovea_profiles,
        R_RPE_POSITION_DARK=r_rpe_position_dark,
        R_RPE_POSITION_LIGHT=r_rpe_position_light,
        R_OLM_POSITION_DARK=r_olm_position_dark,
        R_OLM_POSITION_LIGHT=r_olm_position_light,
        R_ONL_OPL_POSITION_DARK=r_onl_opl_position_dark,
        R_ONL_OPL_POSITION_LIGHT=r_onl_opl_position_light,
        R_INL_IPL_POSITION_DARK=r_inl_ipl_position_dark,
        R_INL_IPL_POSITION_LIGHT=r_inl_ipl_position_light,
        R_RNFL_GCL_POSITION_DARK=r_rnfl_gcl_position_dark,
        R_RNFL_GCL_POSITION_LIGHT=r_rnfl_gcl_position_light,
        R_VITREOUS_RETINA_POSITION_DARK=r_vitreous_retina_position_dark,
        R_VITREOUS_RETINA_POSITION_LIGHT=r_vitreous_retina_position_light,
    )

    dbg("variable-stats", "Printing summary statistics for translated final-output variables")
    show_array_stats("R.RPE.POSITION.DARK", r_rpe_position_dark)
    show_array_stats("R.RPE.POSITION.LIGHT", r_rpe_position_light)
    show_array_stats("R.OLM.POSITION.DARK", r_olm_position_dark)
    show_array_stats("R.OLM.POSITION.LIGHT", r_olm_position_light)
    show_array_stats("FLATTENED.DARK.RETINA.RRC.N", flattened_dark_retina_rrc_n)
    show_array_stats("FLATTENED.LIGHT.RETINA.RRC.N", flattened_light_retina_rrc_n)
    show_array_stats("FLATTENED.DARK.RETINA.RRC.N.profiles", flattened_dark_retina_rrc_n_profiles)
    show_array_stats("FLATTENED.LIGHT.RETINA.RRC.N.profiles", flattened_light_retina_rrc_n_profiles)
    show_array_stats("MAIN.DARK.OUTPUTS", main_dark_outputs)
    show_array_stats("MAIN.LIGHT.OUTPUTS", main_light_outputs)
    show_array_stats("MAIN.DARK.OUTPUTS.fovea", main_dark_outputs_fovea)
    show_array_stats("MAIN.LIGHT.OUTPUTS.fovea", main_light_outputs_fovea)
    show_array_stats("FLATTENED.DARK.RETINA.RRC.N.fovea.profiles", flattened_dark_retina_rrc_n_fovea_profiles)
    show_array_stats("FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles", flattened_light_retina_rrc_n_fovea_profiles)
    show_scalar_stats("PLOT.REFINED.BORDER.OVERVIEW", refined_border_overview_plot_path)
    show_scalar_stats("PLOT.REVISED.VITREOUS.DARK", ",".join(str(path) for path in revised_vitreous_dark_plot_paths))
    show_scalar_stats("PLOT.REVISED.RNFL.DARK", ",".join(str(path) for path in revised_rnfl_dark_plot_paths))
    show_scalar_stats("PLOT.REVISED.INL.DARK", ",".join(str(path) for path in revised_inl_dark_plot_paths))
    show_scalar_stats("PLOT.REVISED.ONL.DARK", ",".join(str(path) for path in revised_onl_dark_plot_paths))
    show_scalar_stats("PLOT.REVISED.OLM.DARK", ",".join(str(path) for path in revised_olm_dark_plot_paths))
    show_scalar_stats("PLOT.REVISED.VITREOUS.LIGHT", ",".join(str(path) for path in revised_vitreous_light_plot_paths))
    show_scalar_stats("PLOT.REVISED.RNFL.LIGHT", ",".join(str(path) for path in revised_rnfl_light_plot_paths))
    show_scalar_stats("PLOT.REVISED.INL.LIGHT", ",".join(str(path) for path in revised_inl_light_plot_paths))
    show_scalar_stats("PLOT.REVISED.ONL.LIGHT", ",".join(str(path) for path in revised_onl_light_plot_paths))
    show_scalar_stats("PLOT.REVISED.OLM.LIGHT", ",".join(str(path) for path in revised_olm_light_plot_paths))
    show_scalar_stats("PLOT.MAIN.DARK.PROFILE", dark_main_profile_plot_path)
    show_scalar_stats("PLOT.MAIN.LIGHT.PROFILE", light_main_profile_plot_path)
    show_scalar_stats("PLOT.FOVEA.OLM.DARK", ",".join(str(path) for path in dark_fovea_olm_plot_paths))
    show_scalar_stats("PLOT.FOVEA.OLM.LIGHT", ",".join(str(path) for path in light_fovea_olm_plot_paths))
    show_scalar_stats("PLOT.FOVEA.DARK.PROFILE", dark_fovea_profile_plot_path)
    show_scalar_stats("PLOT.FOVEA.LIGHT.PROFILE", light_fovea_profile_plot_path)
    show_scalar_stats("EXPORT.NORM.DARK", dark_norm_export_base)
    show_scalar_stats("EXPORT.NORM.LIGHT", light_norm_export_base)
    show_scalar_stats("TXT.DARK.PROFILES", dark_profiles_txt_path)
    show_scalar_stats("TXT.LIGHT.PROFILES", light_profiles_txt_path)
    show_scalar_stats("TXT.FOVEA.DARK.PROFILES", dark_fovea_txt_path)
    show_scalar_stats("TXT.FOVEA.LIGHT.PROFILES", light_fovea_txt_path)
    show_scalar_stats("SAVE.FINAL", final_save_path)

    return volumes


if __name__ == "__main__":
    try:
        main()
    except StopTranslationBoundary:
        pass
    except Exception:
        print(f"ERROR: failure at step '{DEBUG_STEP}'")
        raise
