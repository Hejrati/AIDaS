"""Step 3 — OCT image flattening and spatial normalization.

This module converts the R script RAW_OCT_PROCESSING_2023_09SEP-05_WSU.R to Python.
It performs retinal flattening based on RPE markers and spatial alignment across images.

Core steps:
  1. Load Analyze format OCT images (DARK, LIGHT, DARK_MARKED, LIGHT_MARKED)
  2. Extract RPE and foveal center from marked images
  3. Sample retina perpendicular to RPE
  4. Flatten retina by shifting to align RPE
  5. Register images to common space
"""

from __future__ import annotations

import shutil
import subprocess
import re
import tempfile
import urllib.request
from datetime import datetime
import concurrent.futures
import numpy as np
from scipy.interpolate import BSpline, UnivariateSpline
from scipy.stats import pearsonr
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import os
import sys

from PIL import Image, ImageOps, ImageTk

try:
    import pyreadr
except Exception:
    pyreadr = None

from aidas.utils.io_utils import read_analyze, write_analyze
from aidas.utils.batch_ui import BatchTable
from aidas.utils.step3_image_utils import (
    make_comparison_preview_image as _make_comparison_preview_image,
    make_find_vertex_preview_image as _make_find_vertex_preview_image,
    make_main_results_summary_image as _make_main_results_summary_image,
    placeholder_image as _placeholder_image,
    save_profile_plot as _save_profile_plot,
)
from aidas.utils.ui_utils import SidebarStepFrame
from main import (
    build_fovea_normalized_strip as _main_build_fovea_normalized_strip,
    build_main_normalized_strip as _main_build_main_normalized_strip,
    build_profile_matrix as _main_build_profile_matrix,
    compute_retina_points_for_marked_slice as _main_compute_retina_points_for_marked_slice,
    fill_na_with_leading_non_na as _main_fill_na_with_leading_non_na,
    first_closest_zero_crossing as _main_first_closest_zero_crossing,
    refine_border_position_pass as _main_refine_border_position_pass,
    run_more_outputs_from_step3_npz as _main_run_more_outputs_from_step3_npz,
    slice_rows_1based as _main_slice_rows_1based,
    smooth_position_matrix as _main_smooth_position_matrix,
    write_object_table as _main_write_object_table,
)

MARKER_LAYER_VALUES = (
    ("RNFL-Vitreous", 249),
    ("GCL-RNFL", 250),
    ("INL-IPL", 252),
    ("ONL-OPL", 253),
    ("ELM", 254),
)


def _normalize_analyze_path(base_path):
    """Return the Analyze header path for a base path or .hdr path."""
    path = str(base_path)
    if path.lower().endswith(".hdr"):
        return Path(path)
    return Path(f"{path}.hdr")


def _load_analyze_volume_r_layout(path):
    """Load an Analyze volume using the same layout convention as main.py."""
    volume = np.asarray(read_analyze(_normalize_analyze_path(path)))

    # Match the R script's effective (y, x, slice) layout.
    if volume.ndim == 3:
        volume = np.transpose(volume, (2, 1, 0))[:, ::-1, :]

    if volume.ndim == 4:
        volume = volume[:, :, :, 0]

    return volume


def _build_coordinate_grids(image_2d):
    """Build the same 1-based X/Y grids used in main.py."""
    image_2d = np.asarray(image_2d, dtype=np.float64)
    xs = image_2d.copy()
    insert_x = np.arange(1, image_2d.shape[0] + 1, dtype=np.float64)
    for col in range(xs.shape[1]):
        xs[:, col] = insert_x

    ys = image_2d.copy()
    insert_y = np.arange(1, image_2d.shape[1] + 1, dtype=np.float64)
    for row in range(ys.shape[0]):
        ys[row, :] = insert_y

    return xs, ys


def _fit_smooth_spline_like_r(x, y, df, degree=3):
    """Approximate R smooth.spline(df=...) with a penalized B-spline smoother.

    This matches main.py more closely than a plain regression spline because it
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

    def solve_for_lambda(lam):
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


def _build_floor_sample_line(start_x, end_x, start_y, end_y):
    """Mirror the R/main.py seq(...); floor(...) sampling line construction."""
    # R uses seq(from, to, by=(to-from)/500), which should yield 501 points:
    # from + by * 0:500. Build that sequence directly so we avoid np.arange
    # endpoint drift and keep the same floor() inputs as closely as possible.
    dx = (end_x - start_x) / 500.0
    dy = (end_y - start_y) / 500.0
    offsets = np.arange(501, dtype=np.float64)
    line_x = start_x + (dx * offsets)
    line_y = start_y + (dy * offsets)
    return np.floor(np.column_stack((line_x, line_y)))


def _get_recon_value(unwrapped_recon, upper_x, upper_y, point):
    """Return a flattened Fortran-order sample, matching main.py."""
    col = float(point[0])
    row = float(point[1])
    if 1 <= col <= upper_x and 1 <= row <= upper_y:
        idx = int((col - 1) * upper_y + row - 1)
        return float(unwrapped_recon[idx])
    return np.nan


def _validate_marker_coverage(flattened_markers_rrc, start_move=21):
    """Return messages for marker layers missing from moving windows."""
    markers = np.asarray(flattened_markers_rrc)
    hand_borders = np.full((markers.shape[0], len(MARKER_LAYER_VALUES)), np.nan, dtype=np.float64)
    for col, (_name, marker_value) in enumerate(MARKER_LAYER_VALUES):
        for row in range(markers.shape[0]):
            idx = np.where(markers[row, :] == marker_value)[0]
            if idx.size:
                hand_borders[row, col] = float(np.mean(idx + 1))

    end_move = (markers.shape[0] - start_move) - 1
    messages = []
    for col, (name, marker_value) in enumerate(MARKER_LAYER_VALUES):
        bad_rows = []
        valid_rows = np.where(~np.isnan(hand_borders[:, col]))[0]
        for x in range(start_move, end_move + 1):
            row_window = slice(x - 20, x + 20)
            if np.all(np.isnan(hand_borders[row_window, col])):
                bad_rows.append(x)
        if bad_rows:
            valid_text = (
                f"{int(valid_rows.min()) + 1}..{int(valid_rows.max()) + 1}"
                if valid_rows.size
                else "none"
            )
            messages.append(
                f"{name} marker {marker_value} is missing from {len(bad_rows)} moving windows "
                f"(first row {bad_rows[0] + 1}, last row {bad_rows[-1] + 1}; valid marker rows: {valid_text})."
            )
    return messages



class OCTFlatteningProcessor:
    """Main processor for OCT image flattening and alignment."""
    
    def __init__(self, reference_dark_path, reference_light_path, dark_path, light_path,
                 image_index_dark, image_index_light, pixel_width=3.89):
        """Initialize OCT flattening processor.
        
        Args:
            reference_dark_path: Path to DARK_MARKED Analyze file (without extension)
            reference_light_path: Path to LIGHT_MARKED Analyze file (without extension)
            dark_path: Path to DARK Analyze file (without extension)
            light_path: Path to LIGHT Analyze file (without extension)
            image_index_dark: List of slice indices for DARK images
            image_index_light: List of slice indices for LIGHT images
            pixel_width: Microns per pixel (default 3.89)
        """
        self.reference_dark_path = reference_dark_path
        self.reference_light_path = reference_light_path
        self.dark_path = dark_path
        self.light_path = light_path
        self.image_index_dark = image_index_dark
        self.image_index_light = image_index_light
        self.pixel_width = pixel_width
        
        # Load images
        self.ref_dark = self._load_analyze(reference_dark_path)
        self.ref_light = self._load_analyze(reference_light_path)
        self.dark = self._load_analyze(dark_path)
        self.light = self._load_analyze(light_path)
    
    @staticmethod
    def _load_analyze(base_path):
        """Load Analyze data using the same axis convention as main.py."""
        return _load_analyze_volume_r_layout(base_path)
    
    def sample_perpendiculars(self, image_2d, retina_points, n_samples=500, progress_cb=None):
        """Sample image intensities using the same floor-based logic as main.py."""
        flattened = np.full((retina_points.shape[0], n_samples), np.nan, dtype=np.float64)
        upper_x = int(image_2d.shape[1])
        upper_y = int(image_2d.shape[0])
        unwrapped_recon = np.ravel(np.asarray(image_2d, dtype=np.float64), order="F")

        for x_idx in range(retina_points.shape[0]):
            line = _build_floor_sample_line(
                retina_points[x_idx, 4],
                retina_points[x_idx, 6],
                retina_points[x_idx, 5],
                retina_points[x_idx, 7],
            )
            values = np.array(
                [
                    _get_recon_value(unwrapped_recon, upper_x, upper_y, np.array([line[i, 1], line[i, 0]]))
                    for i in range(line.shape[0])
                ],
                dtype=np.float64,
            )
            flattened[x_idx, :] = values[1:]
            if progress_cb is not None and (x_idx % 100 == 0 or x_idx == retina_points.shape[0] - 1):
                progress_cb(x_idx + 1, retina_points.shape[0])
        
        return flattened
    
    def process_slice(self, slice_idx=0, reference_volume=None, raw_volume=None, progress_cb=None):
        """Process a single slice: extract RPE, fovea, and flatten.
        
        Returns:
            Dictionary with flattened retina, markers, and alignment info
        """
        # Extract RPE and fovea
        ref_volume = self.ref_dark if reference_volume is None else reference_volume
        img_volume = self.dark if raw_volume is None else raw_volume

        ref_2d = ref_volume[:, :, slice_idx]
        xs, ys = _build_coordinate_grids(ref_2d)
        (
            retina_points,
            rpe_info_2,
            slopey_neg100_to_100_deg,
            slopey_0_to_2750_deg,
        ) = _main_compute_retina_points_for_marked_slice(
            ref_2d,
            xs,
            ys,
        )

        image_2d = img_volume[:, :, slice_idx]

        # Sample flattened retina
        flattened_dark = self.sample_perpendiculars(
            image_2d,
            retina_points,
            progress_cb=(lambda done, total: progress_cb("image", done, total)) if progress_cb is not None else None,
        )
        flattened_markers = self.sample_perpendiculars(
            ref_2d,
            retina_points,
            progress_cb=(lambda done, total: progress_cb("markers", done, total)) if progress_cb is not None else None,
        )
        
        return {
            'flattened': flattened_dark,
            'markers': flattened_markers,
            'retina_points': retina_points,
            'rpe_info_2': rpe_info_2,
            'angle_fovea_deg': slopey_neg100_to_100_deg,
            'angle_main_deg': slopey_0_to_2750_deg,
        }

    @staticmethod
    def _resolve_slice_indices(index_list, depth):
        """Resolve configured image indices to zero-based slice indices."""
        if not index_list:
            return list(range(depth))

        # Support both 1-based (R style) and 0-based inputs.
        if min(index_list) >= 1:
            resolved = [int(i) - 1 for i in index_list]
        else:
            resolved = [int(i) for i in index_list]

        # Clamp to valid range and keep order without duplicates.
        seen = set()
        out = []
        for idx in resolved:
            idx = max(0, min(idx, depth - 1))
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        return out
    
    def process_all_slices(self, progress_cb=None):
        """Process all slices and return flattened volumes.
        
        Returns:
            Dictionary with flattened DARK and LIGHT volumes
        """
        dark_slices = self._resolve_slice_indices(self.image_index_dark, self.dark.shape[2])
        light_slices = self._resolve_slice_indices(self.image_index_light, self.light.shape[2])
        
        # Process all slices
        flattened_dark_raw = []
        flattened_light_raw = []
        flattened_markers_all = []
        apparent_angles_for_dark = np.column_stack(
            (
                np.asarray([idx + 1 for idx in dark_slices], dtype=np.float64),
                np.full(len(dark_slices), np.nan, dtype=np.float64),
                np.full(len(dark_slices), np.nan, dtype=np.float64),
            )
        )
        apparent_angles_for_light = np.column_stack(
            (
                np.asarray([idx + 1 for idx in light_slices], dtype=np.float64),
                np.full(len(light_slices), np.nan, dtype=np.float64),
                np.full(len(light_slices), np.nan, dtype=np.float64),
            )
        )
        
        total_jobs = (len(dark_slices) * 2) + (len(light_slices) * 2)
        completed_jobs = 0

        def slice_progress(label, modality, out_idx, z, done, total):
            if progress_cb is None:
                return
            local = 0.5 * (done / max(1, total))
            if label == "markers":
                local += 0.5
            overall = (completed_jobs + local) / max(1, total_jobs)
            progress_cb(overall, f"Flattening {modality} slice {z + 1} ({label})")

        for out_idx, z in enumerate(dark_slices):
            result = self.process_slice(
                z,
                reference_volume=self.ref_dark,
                raw_volume=self.dark,
                progress_cb=lambda label, done, total, out_idx=out_idx, z=z: slice_progress(
                    label, "DARK", out_idx, z, done, total
                ),
            )
            flattened_dark_raw.append(result['flattened'])
            flattened_markers_all.append(result['markers'])
            apparent_angles_for_dark[out_idx, 1] = result['angle_fovea_deg']
            apparent_angles_for_dark[out_idx, 2] = result['angle_main_deg']
            completed_jobs += 2
        
        for out_idx, z in enumerate(light_slices):
            result = self.process_slice(
                z,
                reference_volume=self.ref_light,
                raw_volume=self.light,
                progress_cb=lambda label, done, total, out_idx=out_idx, z=z: slice_progress(
                    label, "LIGHT", out_idx, z, done, total
                ),
            )
            flattened_light_raw.append(result['flattened'])
            apparent_angles_for_light[out_idx, 1] = result['angle_fovea_deg']
            apparent_angles_for_light[out_idx, 2] = result['angle_main_deg']
            completed_jobs += 2
        
        return {
            'flattened_dark': np.array(flattened_dark_raw),
            'flattened_light': np.array(flattened_light_raw),
            'markers': np.array(flattened_markers_all) if flattened_markers_all else None,
            'apparent_angles_for_dark': apparent_angles_for_dark,
            'apparent_angles_for_light': apparent_angles_for_light,
        }


def _emit_progress(progress_cb, percent, label):
    if progress_cb is not None:
        progress_cb(float(percent), str(label))


def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 3:
        return -np.inf
    try:
        return float(pearsonr(a[valid], b[valid])[0])
    except Exception:
        return -np.inf


def _to_linearized(raw):
    """Match R conversion from log-transformed intensity to linear scale."""
    out = np.array(raw, dtype=np.float64, copy=True)
    out[np.isnan(out)] = -32768
    out = out + 32768
    out[out < 0] = 0
    return np.power(2.0, out / 5000.0)


def _nanmean_axis0(arr):
    """Mean across axis 0 without RuntimeWarning for all-NaN columns."""
    arr = np.asarray(arr, dtype=np.float64)
    sums = np.nansum(arr, axis=0)
    counts = np.sum(np.isfinite(arr), axis=0)
    out = np.full(sums.shape, np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _save_flat_checkpoint(results, output_dir):
    """Save the Python equivalent of R's DARK__and__LIGHT__flat.RData."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    flattened_dark_retina_rrc = np.transpose(np.asarray(results["final_dark"], dtype=np.float64), (1, 2, 0))
    flattened_light_retina_rrc = np.transpose(np.asarray(results["final_light"], dtype=np.float64), (1, 2, 0))
    flattened_markers_rrc = np.asarray(results["markers"], dtype=np.float64)

    flat_save_path = outdir / "DARK__and__LIGHT__flat.npz"
    np.savez_compressed(
        flat_save_path,
        FLATTENED_DARK_RETINA_RRC=flattened_dark_retina_rrc,
        FLATTENED_LIGHT_RETINA_RRC=flattened_light_retina_rrc,
        FLATTENED_MARKERS_RRC=flattened_markers_rrc,
        FIRST_GRAND_MEAN=np.asarray(results["first_grand_mean"], dtype=np.float64),
        SECOND_GRAND_MEAN=np.asarray(results["second_grand_mean"], dtype=np.float64),
        FINAL_GRAND_MEAN=np.asarray(results["final_grand_mean"], dtype=np.float64),
        GRAND_PROFILE=np.asarray(results["grand_profile"], dtype=np.float64),
        APPARENT_ANGLES_FOR_DARK=np.asarray(results["apparent_angles_for_dark"], dtype=np.float64),
        APPARENT_ANGLES_FOR_LIGHT=np.asarray(results["apparent_angles_for_light"], dtype=np.float64),
        SHIFT_POSITION_DARK=np.asarray(results["shift_dark"], dtype=np.float64),
        SHIFT_POSITION_LIGHT=np.asarray(results["shift_light"], dtype=np.float64),
        SHIFT_POSITION_DARK_REFINED=np.asarray(results["shift_dark_refined"], dtype=np.float64),
        SHIFT_POSITION_LIGHT_REFINED=np.asarray(results["shift_light_refined"], dtype=np.float64),
        BEST_LAT_MOVE_DARK=np.asarray(results["best_lateral_dark"], dtype=np.float64),
        BEST_LAT_MOVE_LIGHT=np.asarray(results["best_lateral_light"], dtype=np.float64),
        VERTEX=np.asarray([results["vertex"]], dtype=np.float64),
    )
    return flat_save_path


def _save_main_style_exports(results, output_dir, progress_cb=None):
    """Generate the final text/npz outputs from main.py for Step 3."""
    outdir = Path(output_dir)
    plots_dir = outdir / "python_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_cb, 0, "Preparing profile exports")
    flat_checkpoint_path = _save_flat_checkpoint(results, outdir)

    # Match main.py's late-export stage: use the post-vertex cropped arrays
    # (461 columns), not the pre-crop 500-column RRC arrays.
    flattened_dark_retina_rrc = np.transpose(np.asarray(results["final_dark"], dtype=np.float64), (1, 2, 0))
    flattened_light_retina_rrc = np.transpose(np.asarray(results["final_light"], dtype=np.float64), (1, 2, 0))
    flattened_markers_rrc = np.asarray(results["markers"], dtype=np.float64)
    apparent_angles_for_dark = np.asarray(results["apparent_angles_for_dark"], dtype=np.float64)
    apparent_angles_for_light = np.asarray(results["apparent_angles_for_light"], dtype=np.float64)

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
        if progress_cb is not None and (x % 300 == 0 or x == flattened_markers_rrc.shape[0] - 1):
            _emit_progress(progress_cb, 5 * ((x + 1) / max(1, flattened_markers_rrc.shape[0])), "Reading hand-marked borders")

    start_move = 21
    end_move = (flattened_dark_retina_rrc.shape[0] - start_move) - 1
    coverage_messages = _validate_marker_coverage(flattened_markers_rrc, start_move=start_move)
    if coverage_messages:
        raise RuntimeError(
            "Step 3 cannot localize all retinal layers because the final cropped MARKED image "
            "has missing boundary markers:\n"
            + "\n".join(f"- {message}" for message in coverage_messages)
            + "\n\nRe-save Step 2 MARKED images after setting the fovea center, then reload Step 3."
        )

    blank = np.full(6, np.nan, dtype=np.float64)
    blank[5] = 431.0

    true_borders_dark = np.full((flattened_dark_retina_rrc.shape[0], 6, flattened_dark_retina_rrc.shape[2]), np.nan, dtype=np.float64)
    dark_total = max(1, flattened_dark_retina_rrc.shape[2] * (end_move - start_move + 1))
    dark_done = 0
    for z in range(flattened_dark_retina_rrc.shape[2]):
        review = flattened_dark_retina_rrc[:, :, z]
        profile_x = np.arange(1.0, review.shape[1] + 1.0, 1.0)
        for x in range(start_move, end_move + 1):
            new_values = blank.copy()
            row_window = slice(x - 20, x + 20)
            profile = np.column_stack((profile_x, np.nanmean(review[row_window, :], axis=0)))
            segment = np.round(np.nanmean(hand_borders[row_window, :], axis=0))

            if not np.isnan(segment[0]):
                seg1 = int(segment[0])
                check = _main_slice_rows_1based(profile, seg1 - 20, seg1 + 20)
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))[10:31, :]
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[0] = _main_first_closest_zero_crossing(check, 1.0)

            if not np.isnan(segment[1]):
                seg2 = int(segment[1])
                movein = 20.0
                if not np.isnan(segment[2]):
                    movein_alt = np.ceil(segment[2] - segment[1]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = _main_slice_rows_1based(profile, int(new_values[0]), int(seg2 + movein))
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
                new_values[1] = _main_first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[2]):
                seg3 = int(segment[2])
                movein = 20.0
                if not np.isnan(segment[3]):
                    movein_alt = np.ceil(segment[3] - segment[2]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = _main_slice_rows_1based(profile, int(seg3 - movein), int(seg3 + movein))
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
                new_values[2] = _main_first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[3]):
                seg4 = int(segment[3])
                movein = 20.0
                if not np.isnan(segment[4]):
                    movein_alt = np.ceil(segment[4] - segment[3]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = _main_slice_rows_1based(profile, int(seg4 - movein), int(seg4 + movein))
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
                new_values[3] = _main_first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[4]):
                seg5 = int(segment[4])
                check = _main_slice_rows_1based(profile, seg5 - 5, seg5 + 5)
                m = np.nanmax(check[:, 1])
                localpeak = float(np.round(np.nanmean(check[np.where(check[:, 1] == m)[0], 0])))
                new_values[4] = localpeak
                low = float(check[2, 0])
                high = float(check[9, 0])
                if (localpeak > low) and (localpeak < high):
                    local_index = np.where(check[:, 0] == localpeak)[0][0]
                    check = check[(local_index - 2):(local_index + 3), :]
                    c_sp = _fit_smooth_spline_like_r(check[:, 0], check[:, 1], df=2)
                    c_x = np.arange(check[0, 0], check[4, 0] + 0.1, 0.1)
                    cspline = np.column_stack((c_x, c_sp(c_x)))
                    vertex_local = float(np.mean(cspline[np.where(cspline[:, 1] == np.nanmax(cspline[:, 1]))[0], 0]))
                    if (vertex_local > low) and (vertex_local < high):
                        new_values[4] = float(np.round(vertex_local, 1))

            true_borders_dark[x - 1, :, z] = new_values
            dark_done += 1
            if progress_cb is not None and (dark_done % 150 == 0 or dark_done == dark_total):
                _emit_progress(
                    progress_cb,
                    5 + (20 * dark_done / dark_total),
                    f"Detecting DARK borders: slice {z + 1}",
                )

    true_borders_light = np.full((flattened_light_retina_rrc.shape[0], 6, flattened_light_retina_rrc.shape[2]), np.nan, dtype=np.float64)
    light_total = max(1, flattened_light_retina_rrc.shape[2] * (end_move - start_move + 1))
    light_done = 0
    for z in range(flattened_light_retina_rrc.shape[2]):
        review = flattened_light_retina_rrc[:, :, z]
        profile_x = np.arange(1.0, review.shape[1] + 1.0, 1.0)
        for x in range(start_move, end_move + 1):
            new_values = blank.copy()
            row_window = slice(x - 20, x + 20)
            profile = np.column_stack((profile_x, np.nanmean(review[row_window, :], axis=0)))
            segment = np.round(np.nanmean(hand_borders[row_window, :], axis=0))

            if not np.isnan(segment[0]):
                seg1 = int(segment[0])
                check = _main_slice_rows_1based(profile, seg1 - 20, seg1 + 20)
                half = (np.nanmin(check[:, 1]) + np.nanmax(check[:, 1])) / 2.0
                check = np.column_stack((check, check))[10:31, :]
                check[:, 2] = check[:, 1] - half
                check[:, 3] = np.sign(check[:, 2])
                new_values[0] = _main_first_closest_zero_crossing(check, 1.0)

            if not np.isnan(segment[1]):
                seg2 = int(segment[1])
                movein = 20.0
                if not np.isnan(segment[2]):
                    movein_alt = np.ceil(segment[2] - segment[1]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = _main_slice_rows_1based(profile, int(new_values[0]), int(seg2 + movein))
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
                new_values[1] = _main_first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[2]):
                seg3 = int(segment[2])
                movein = 20.0
                if not np.isnan(segment[3]):
                    movein_alt = np.ceil(segment[3] - segment[2]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = _main_slice_rows_1based(profile, int(seg3 - movein), int(seg3 + movein))
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
                new_values[2] = _main_first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[3]):
                seg4 = int(segment[3])
                movein = 20.0
                if not np.isnan(segment[4]):
                    movein_alt = np.ceil(segment[4] - segment[3]) / 2.0
                    if movein_alt < 20:
                        movein = movein_alt
                check = _main_slice_rows_1based(profile, int(seg4 - movein), int(seg4 + movein))
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
                new_values[3] = _main_first_closest_zero_crossing(check, -1.0)

            if not np.isnan(segment[4]):
                seg5 = int(segment[4])
                check = _main_slice_rows_1based(profile, seg5 - 5, seg5 + 5)
                m = np.nanmax(check[:, 1])
                localpeak = float(np.round(np.nanmean(check[np.where(check[:, 1] == m)[0], 0])))
                new_values[4] = localpeak
                low = float(check[2, 0])
                high = float(check[9, 0])
                if (localpeak > low) and (localpeak < high):
                    local_index = np.where(check[:, 0] == localpeak)[0][0]
                    check = check[(local_index - 2):(local_index + 3), :]
                    c_sp = _fit_smooth_spline_like_r(check[:, 0], check[:, 1], df=2)
                    c_x = np.arange(check[0, 0], check[4, 0] + 0.1, 0.1)
                    cspline = np.column_stack((c_x, c_sp(c_x)))
                    vertex_local = float(np.mean(cspline[np.where(cspline[:, 1] == np.nanmax(cspline[:, 1]))[0], 0]))
                    if (vertex_local > low) and (vertex_local < high):
                        new_values[4] = float(np.round(vertex_local, 1))

            true_borders_light[x - 1, :, z] = new_values
            light_done += 1
            if progress_cb is not None and (light_done % 150 == 0 or light_done == light_total):
                _emit_progress(
                    progress_cb,
                    25 + (20 * light_done / light_total),
                    f"Detecting LIGHT borders: slice {z + 1}",
                )

    for z in range(flattened_dark_retina_rrc.shape[2]):
        true_borders_dark[:, 5, z] = 431.0
    for z in range(flattened_light_retina_rrc.shape[2]):
        true_borders_light[:, 5, z] = 431.0

    vitreous_retina_position_dark = true_borders_dark[:, 0, :].copy()[599:, :]
    rnfl_gcl_position_dark = true_borders_dark[:, 1, :].copy()[599:, :]
    inl_ipl_position_dark = true_borders_dark[:, 2, :].copy()[599:, :]
    onl_opl_position_dark = true_borders_dark[:, 3, :].copy()[599:, :]
    olm_position_dark = true_borders_dark[:, 4, :].copy()[599:, :]
    rpe_position_dark = true_borders_dark[:, 5, :].copy()[599:, :]

    # Match main.py exactly here: both DARK and LIGHT smoothing inputs start at row 599.
    vitreous_retina_position_light = true_borders_light[:, 0, :].copy()[599:, :]
    rnfl_gcl_position_light = true_borders_light[:, 1, :].copy()[599:, :]
    inl_ipl_position_light = true_borders_light[:, 2, :].copy()[599:, :]
    onl_opl_position_light = true_borders_light[:, 3, :].copy()[599:, :]
    olm_position_light = true_borders_light[:, 4, :].copy()[599:, :]
    rpe_position_light = true_borders_light[:, 5, :].copy()[599:, :]

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

    _emit_progress(progress_cb, 48, "Refining border positions")
    _main_refine_border_position_pass(position_mats_dark, ["RPE.POSITION.DARK", "OLM.POSITION.DARK", "VITREOUS.RETINA.POSITION.DARK", "ONL.OPL.POSITION.DARK", "INL.IPL.POSITION.DARK"], "INL.IPL.POSITION.DARK", 2.0, plots_dir, "python_dark_inl_ipl_refinement")
    _main_refine_border_position_pass(position_mats_dark, ["INL.IPL.POSITION.DARK", "RPE.POSITION.DARK", "OLM.POSITION.DARK", "VITREOUS.RETINA.POSITION.DARK", "ONL.OPL.POSITION.DARK"], "ONL.OPL.POSITION.DARK", 2.0, plots_dir, "python_dark_onl_opl_refinement")
    _main_refine_border_position_pass(position_mats_dark, ["ONL.OPL.POSITION.DARK", "INL.IPL.POSITION.DARK", "RPE.POSITION.DARK", "OLM.POSITION.DARK", "VITREOUS.RETINA.POSITION.DARK"], "VITREOUS.RETINA.POSITION.DARK", 2.0, plots_dir, "python_dark_vitreous_retina_refinement")
    _main_refine_border_position_pass(position_mats_dark, ["VITREOUS.RETINA.POSITION.DARK", "ONL.OPL.POSITION.DARK", "INL.IPL.POSITION.DARK", "RPE.POSITION.DARK", "OLM.POSITION.DARK"], "OLM.POSITION.DARK", 3.0, plots_dir, "python_dark_olm_refinement")
    _main_refine_border_position_pass(position_mats_dark, ["VITREOUS.RETINA.POSITION.DARK", "ONL.OPL.POSITION.DARK", "INL.IPL.POSITION.DARK", "RPE.POSITION.DARK", "OLM.POSITION.DARK", "RNFL.GCL.POSITION.DARK"], "RNFL.GCL.POSITION.DARK", 2.0, plots_dir, "python_dark_rnfl_gcl_refinement")

    _main_refine_border_position_pass(position_mats_light, ["RPE.POSITION.LIGHT", "OLM.POSITION.LIGHT", "VITREOUS.RETINA.POSITION.LIGHT", "ONL.OPL.POSITION.LIGHT", "INL.IPL.POSITION.LIGHT"], "INL.IPL.POSITION.LIGHT", 2.0, plots_dir, "python_light_inl_ipl_refinement")
    _main_refine_border_position_pass(position_mats_light, ["INL.IPL.POSITION.LIGHT", "RPE.POSITION.LIGHT", "OLM.POSITION.LIGHT", "VITREOUS.RETINA.POSITION.LIGHT", "ONL.OPL.POSITION.LIGHT"], "ONL.OPL.POSITION.LIGHT", 2.0, plots_dir, "python_light_onl_opl_refinement")
    _main_refine_border_position_pass(position_mats_light, ["ONL.OPL.POSITION.LIGHT", "INL.IPL.POSITION.LIGHT", "RPE.POSITION.LIGHT", "OLM.POSITION.LIGHT", "VITREOUS.RETINA.POSITION.LIGHT"], "VITREOUS.RETINA.POSITION.LIGHT", 2.0, plots_dir, "python_light_vitreous_retina_refinement")
    _main_refine_border_position_pass(position_mats_light, ["VITREOUS.RETINA.POSITION.LIGHT", "ONL.OPL.POSITION.LIGHT", "INL.IPL.POSITION.LIGHT", "RPE.POSITION.LIGHT", "OLM.POSITION.LIGHT"], "OLM.POSITION.LIGHT", 3.0, plots_dir, "python_light_olm_refinement")
    _main_refine_border_position_pass(position_mats_light, ["VITREOUS.RETINA.POSITION.LIGHT", "ONL.OPL.POSITION.LIGHT", "INL.IPL.POSITION.LIGHT", "RPE.POSITION.LIGHT", "OLM.POSITION.LIGHT", "RNFL.GCL.POSITION.LIGHT"], "RNFL.GCL.POSITION.LIGHT", 2.0, plots_dir, "python_light_rnfl_gcl_refinement")

    _emit_progress(progress_cb, 58, "Smoothing DARK layer positions")
    x_dark = np.arange(2750.0 - vitreous_retina_position_dark.shape[0] + 1.0, 2750.0 + 1.0, 1.0)
    r_vitreous_retina_position_dark, _ = _main_smooth_position_matrix(vitreous_retina_position_dark, x_dark, 11, plots_dir, "python_revised_vitreous_retina_dark")
    r_rnfl_gcl_position_dark, _ = _main_smooth_position_matrix(rnfl_gcl_position_dark, x_dark, 11, plots_dir, "python_revised_rnfl_gcl_dark")
    r_inl_ipl_position_dark, _ = _main_smooth_position_matrix(inl_ipl_position_dark, x_dark, 11, plots_dir, "python_revised_inl_ipl_dark")
    r_onl_opl_position_dark, _ = _main_smooth_position_matrix(onl_opl_position_dark, x_dark, 11, plots_dir, "python_revised_onl_opl_dark")
    r_olm_position_dark, _ = _main_smooth_position_matrix(olm_position_dark, x_dark, 11, plots_dir, "python_revised_olm_dark")
    r_rpe_position_dark = np.asarray(rpe_position_dark, dtype=np.float64).copy()
    for z in range(r_rpe_position_dark.shape[1]):
        r_rpe_position_dark[:, z] = _main_fill_na_with_leading_non_na(r_rpe_position_dark[:, z])

    _emit_progress(progress_cb, 68, "Smoothing LIGHT layer positions")
    x_light = np.arange(2750.0 - vitreous_retina_position_light.shape[0] + 1.0, 2750.0 + 1.0, 1.0)
    r_vitreous_retina_position_light, _ = _main_smooth_position_matrix(vitreous_retina_position_light, x_light, 11, plots_dir, "python_revised_vitreous_retina_light")
    r_rnfl_gcl_position_light, _ = _main_smooth_position_matrix(rnfl_gcl_position_light, x_light, 11, plots_dir, "python_revised_rnfl_gcl_light")
    r_inl_ipl_position_light, _ = _main_smooth_position_matrix(inl_ipl_position_light, x_light, 11, plots_dir, "python_revised_inl_ipl_light")
    r_onl_opl_position_light, _ = _main_smooth_position_matrix(onl_opl_position_light, x_light, 11, plots_dir, "python_revised_onl_opl_light")
    r_olm_position_light, _ = _main_smooth_position_matrix(olm_position_light, x_light, 11, plots_dir, "python_revised_olm_light")
    r_rpe_position_light = np.asarray(rpe_position_light, dtype=np.float64).copy()
    for z in range(r_rpe_position_light.shape[1]):
        r_rpe_position_light[:, z] = _main_fill_na_with_leading_non_na(r_rpe_position_light[:, z])

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

    main_dark_outputs = np.column_stack((apparent_angles_for_dark[:, 0], apparent_angles_for_dark[:, 0], apparent_angles_for_dark[:, 0], apparent_angles_for_dark[:, 0]))
    for x in range(main_dark_outputs.shape[0]):
        main_dark_outputs[x, 1] = float(np.nanmean(r_rpe_position_dark[:, x] - r_vitreous_retina_position_dark[:, x]))
        main_dark_outputs[x, 2] = float(np.nanmean(r_rpe_position_dark[:, x] - r_olm_position_dark[:, x]))
        main_dark_outputs[x, 3] = float(np.nanmean(r_rnfl_gcl_position_dark[:, x] - r_vitreous_retina_position_dark[:, x]))

    main_light_outputs = np.column_stack((apparent_angles_for_light[:, 0], apparent_angles_for_light[:, 0], apparent_angles_for_light[:, 0], apparent_angles_for_light[:, 0]))
    for x in range(main_light_outputs.shape[0]):
        main_light_outputs[x, 1] = float(np.nanmean(r_rpe_position_light[:, x] - r_vitreous_retina_position_light[:, x]))
        main_light_outputs[x, 2] = float(np.nanmean(r_rpe_position_light[:, x] - r_olm_position_light[:, x]))
        main_light_outputs[x, 3] = float(np.nanmean(r_rnfl_gcl_position_light[:, x] - r_vitreous_retina_position_light[:, x]))

    _emit_progress(progress_cb, 76, "Building normalized retinal strips")
    flattened_dark_retina_rrc_n = _main_build_main_normalized_strip(flattened_dark_retina_rrc, r_rpe_position_dark, r_olm_position_dark, r_onl_opl_position_dark, r_inl_ipl_position_dark, r_rnfl_gcl_position_dark, r_vitreous_retina_position_dark, row_start=601)
    flattened_light_retina_rrc_n = _main_build_main_normalized_strip(flattened_light_retina_rrc, r_rpe_position_light[: flattened_light_retina_rrc.shape[0], :], r_olm_position_light[: flattened_light_retina_rrc.shape[0], :], r_onl_opl_position_light[: flattened_light_retina_rrc.shape[0], :], r_inl_ipl_position_light[: flattened_light_retina_rrc.shape[0], :], r_rnfl_gcl_position_light[: flattened_light_retina_rrc.shape[0], :], r_vitreous_retina_position_light[: flattened_light_retina_rrc.shape[0], :], row_start=601)
    flattened_dark_retina_rrc_n_profiles = _main_build_profile_matrix(flattened_dark_retina_rrc_n, list(range(1, flattened_dark_retina_rrc.shape[2] + 1)))
    flattened_light_retina_rrc_n_profiles = _main_build_profile_matrix(flattened_light_retina_rrc_n, list(range(1, flattened_light_retina_rrc.shape[2] + 1)))

    _emit_progress(progress_cb, 84, "Building fovea-normalized strips")
    r_rpe_position_dark_fovea = true_borders_dark[20:181, 5, :].copy()
    r_olm_position_dark_fovea_raw = true_borders_dark[20:181, 4, :].copy()
    x_neg80_to_80 = np.arange(-80.0, 80.0 + 1.0, 1.0)
    r_olm_position_dark_fovea_raw, _ = _main_smooth_position_matrix(r_olm_position_dark_fovea_raw, x_neg80_to_80, 3, plots_dir, "python_dark_fovea_olm")
    r_olm_position_dark_fovea2 = r_olm_position_dark_fovea_raw[30:131, :].copy()
    r_rpe_position_dark_fovea2 = r_rpe_position_dark_fovea[30:131, :].copy()
    r_olm_position_dark_fovea = r_olm_position_dark.copy()
    r_olm_position_dark_fovea[50:151, :] = r_olm_position_dark_fovea2
    r_rpe_position_dark_fovea_full = r_rpe_position_dark.copy()
    r_rpe_position_dark_fovea_full[50:151, :] = r_rpe_position_dark_fovea2

    main_dark_outputs_fovea = main_dark_outputs[:, [0, 2]].copy()
    main_dark_outputs_fovea[:, 1] = np.nan
    for x in range(main_dark_outputs_fovea.shape[0]):
        main_dark_outputs_fovea[x, 1] = float(np.nanmean(r_rpe_position_dark_fovea_full[50:151, x] - r_olm_position_dark_fovea[50:151, x]))

    flattened_dark_retina_rrc_n_fovea = _main_build_fovea_normalized_strip(flattened_dark_retina_rrc, r_rpe_position_dark_fovea_full, r_olm_position_dark_fovea, row_start=51, row_end=151)
    flattened_dark_retina_rrc_n_fovea_profiles = _main_build_profile_matrix(flattened_dark_retina_rrc_n_fovea, list(range(1, flattened_dark_retina_rrc.shape[2] + 1)), row_slice=slice(50, 151))
    flattened_dark_retina_rrc_n_fovea_profiles = flattened_dark_retina_rrc_n_fovea_profiles[56:90, :]
    flattened_dark_retina_rrc_n[49:152, :, :] = flattened_dark_retina_rrc_n_fovea[49:152, :, :]

    r_rpe_position_light_fovea = true_borders_light[20:181, 5, :].copy()
    r_olm_position_light_fovea_raw = true_borders_light[20:181, 4, :].copy()
    r_olm_position_light_fovea_raw, _ = _main_smooth_position_matrix(r_olm_position_light_fovea_raw, x_neg80_to_80, 3, plots_dir, "python_light_fovea_olm")
    r_olm_position_light_fovea2 = r_olm_position_light_fovea_raw[30:131, :].copy()
    r_rpe_position_light_fovea2 = r_rpe_position_light_fovea[30:131, :].copy()
    r_olm_position_light_fovea = r_olm_position_light.copy()
    r_olm_position_light_fovea[50:151, :] = r_olm_position_light_fovea2
    r_rpe_position_light_fovea_full = r_rpe_position_light.copy()
    r_rpe_position_light_fovea_full[50:151, :] = r_rpe_position_light_fovea2

    main_light_outputs_fovea = main_light_outputs[:, [0, 2]].copy()
    main_light_outputs_fovea[:, 1] = np.nan
    for x in range(main_light_outputs_fovea.shape[0]):
        main_light_outputs_fovea[x, 1] = float(np.nanmean(r_rpe_position_light_fovea_full[50:151, x] - r_olm_position_light_fovea[50:151, x]))

    flattened_light_retina_rrc_n_fovea = _main_build_fovea_normalized_strip(flattened_light_retina_rrc, r_rpe_position_light_fovea_full[: flattened_light_retina_rrc.shape[0], :], r_olm_position_light_fovea[: flattened_light_retina_rrc.shape[0], :], row_start=51, row_end=151)
    flattened_light_retina_rrc_n_fovea_profiles = _main_build_profile_matrix(flattened_light_retina_rrc_n_fovea, list(range(1, flattened_light_retina_rrc.shape[2] + 1)), row_slice=slice(50, 151))
    flattened_light_retina_rrc_n_fovea_profiles = flattened_light_retina_rrc_n_fovea_profiles[56:90, :]
    flattened_light_retina_rrc_n[49:152, :, :] = flattened_light_retina_rrc_n_fovea[49:152, :, :]

    main_dark_outputs = np.column_stack((main_dark_outputs[:, 0], apparent_angles_for_dark[:, 2], main_dark_outputs[:, 1:4]))
    main_light_outputs = np.column_stack((main_light_outputs[:, 0], apparent_angles_for_light[:, 2], main_light_outputs[:, 1:4]))
    main_dark_outputs_fovea = np.column_stack((main_dark_outputs_fovea[:, 0], apparent_angles_for_dark[:, 1], main_dark_outputs_fovea[:, 1]))
    main_light_outputs_fovea = np.column_stack((main_light_outputs_fovea[:, 0], apparent_angles_for_light[:, 1], main_light_outputs_fovea[:, 1]))

    _emit_progress(progress_cb, 90, "Writing normalized Analyze outputs")
    dark_norm_export = np.transpose(
        np.nan_to_num(flattened_dark_retina_rrc_n, nan=0.0),
        (2, 1, 0),
    ).astype(np.float32)
    light_norm_export = np.transpose(
        np.nan_to_num(flattened_light_retina_rrc_n, nan=0.0),
        (2, 1, 0),
    ).astype(np.float32)
    dark_norm_export_base = outdir / "_flat-normed_DARK"
    light_norm_export_base = outdir / "_flat-normed_LIGHT"
    write_analyze(str(dark_norm_export_base), dark_norm_export)
    write_analyze(str(light_norm_export_base), light_norm_export)

    _emit_progress(progress_cb, 92, "Writing profile tables")
    dark_profiles_table = np.round(np.column_stack((main_dark_outputs, flattened_dark_retina_rrc_n_profiles[:, 1:].T)), 3)
    dark_profiles_export = np.vstack((dark_profiles_table[0:1, :], dark_profiles_table)).astype(object)
    dark_profiles_export[0, :] = np.asarray(["DARK", "angle", "whole", "RPEtoOLM", "RNFL", *np.arange(-3.75, 107.5 + 1.25, 1.25)], dtype=object)
    dark_profiles_txt_path = outdir / "_dark_profiles_DARK.txt"
    _main_write_object_table(dark_profiles_export, dark_profiles_txt_path)

    light_profiles_table = np.round(np.column_stack((main_light_outputs, flattened_light_retina_rrc_n_profiles[:, 1:].T)), 3)
    light_profiles_export = np.vstack((light_profiles_table[0:1, :], light_profiles_table)).astype(object)
    light_profiles_export[0, :] = np.asarray(["LIGHT", "angle", "whole", "RPEtoOLM", "RNFL", *np.arange(-3.75, 107.5 + 1.25, 1.25)], dtype=object)
    light_profiles_txt_path = outdir / "_light_profiles_LIGHT.txt"
    _main_write_object_table(light_profiles_export, light_profiles_txt_path)

    dark_fovea_export_left = np.column_stack((main_dark_outputs_fovea[:, 0], main_dark_outputs_fovea[:, 1], np.full(main_dark_outputs_fovea.shape[0], np.nan, dtype=np.float64), main_dark_outputs_fovea[:, 2], np.full(main_dark_outputs_fovea.shape[0], np.nan, dtype=np.float64)))
    dark_fovea_export_right = flattened_dark_retina_rrc_n_fovea_profiles[:, 1:]
    dark_fovea_buffer = np.full((56, dark_fovea_export_right.shape[1]), np.nan, dtype=np.float64)
    dark_fovea_export_right = np.vstack((dark_fovea_buffer, dark_fovea_export_right)).T
    dark_fovea_table = np.round(np.column_stack((dark_fovea_export_left, dark_fovea_export_right)), 3)
    dark_fovea_export = np.vstack((dark_fovea_table[0:1, :], dark_fovea_table)).astype(object)
    dark_fovea_export[0, :] = np.asarray(["foveaDARK", "angle", "whole", "RPEtoOLM", "RNFL", *np.arange(-3.75, 107.5 + 1.25, 1.25)], dtype=object)
    dark_fovea_txt_path = outdir / "_fovea_dark_profiles_DARK.txt"
    _main_write_object_table(dark_fovea_export, dark_fovea_txt_path)

    light_fovea_export_left = np.column_stack((main_light_outputs_fovea[:, 0], main_light_outputs_fovea[:, 1], np.full(main_light_outputs_fovea.shape[0], np.nan, dtype=np.float64), main_light_outputs_fovea[:, 2], np.full(main_light_outputs_fovea.shape[0], np.nan, dtype=np.float64)))
    light_fovea_export_right = flattened_light_retina_rrc_n_fovea_profiles[:, 1:]
    light_fovea_buffer = np.full((56, light_fovea_export_right.shape[1]), np.nan, dtype=np.float64)
    light_fovea_export_right = np.vstack((light_fovea_buffer, light_fovea_export_right)).T
    light_fovea_table = np.round(np.column_stack((light_fovea_export_left, light_fovea_export_right)), 3)
    light_fovea_export = np.vstack((light_fovea_table[0:1, :], light_fovea_table)).astype(object)
    light_fovea_export[0, :] = np.asarray(["foveaLIGHT", "angle", "whole", "RPEtoOLM", "RNFL", *np.arange(-3.75, 107.5 + 1.25, 1.25)], dtype=object)
    light_fovea_txt_path = outdir / "_fovea_light_profiles_LIGHT.txt"
    _main_write_object_table(light_fovea_export, light_fovea_txt_path)

    final_save_path = outdir / "_done_DARK__and__LIGHT.npz"
    _emit_progress(progress_cb, 97, "Writing final Step 3 NPZ")
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

    _emit_progress(progress_cb, 100, "Profile exports complete")
    return {
        "dark_profiles_txt_path": dark_profiles_txt_path,
        "light_profiles_txt_path": light_profiles_txt_path,
        "dark_fovea_txt_path": dark_fovea_txt_path,
        "light_fovea_txt_path": light_fovea_txt_path,
        "dark_norm_export_base": dark_norm_export_base,
        "light_norm_export_base": light_norm_export_base,
        "flat_checkpoint_path": flat_checkpoint_path,
        "final_save_path": final_save_path,
    }


def _compute_shift_positions(volume_raw, first_grand_mean, look_to, pixel_width, progress_cb=None):
    """R-equivalent local vertical alignment scan (SHIFT.POSITION.*)."""
    n_x, n_y, n_z = volume_raw.shape
    dist_axis = np.arange(-200, -200 + n_x, 1)
    shift_pos = np.full((n_x, n_z + 1), np.nan, dtype=np.float64)
    shift_pos[:, 0] = dist_axis

    window_width = 400
    start_move = 201
    end_move = (n_x - start_move) - 1

    x_positions = list(range(start_move, max(start_move, end_move) + 1, 50))
    total_steps = max(1, n_z * len(x_positions))
    done_steps = 0

    for z in range(n_z):
        revise = volume_raw[:, :, z]
        for x_1b in x_positions:
            x0 = x_1b - 1
            x_start = x0 - 199
            x_end = x0 + 201
            if x_start < 0 or x_end > n_x:
                done_steps += 1
                continue

            profile = np.nanmean(revise[x_start:x_end, :], axis=0)
            comparison = np.nanmean(first_grand_mean[x_start:x_end, :], axis=0)

            look_block = look_to[x_start:x_end]
            if np.all(np.isnan(look_block)):
                top_range = 500
            else:
                top_range = int(np.nanmax(look_block))
            top_range = max(1, min(top_range, 500))

            i0 = top_range - 1
            i1 = 480
            if i1 <= i0:
                done_steps += 1
                continue

            check_p = profile[i0:i1]
            check_c = comparison[i0:i1]

            best_corr = -np.inf
            best_move = 0
            for move in range(-10, 11):
                if move == 0:
                    corr = _safe_corr(check_p, check_c)
                elif move < 0:
                    k = -move
                    corr = _safe_corr(check_p[k:], check_c[:-k]) if len(check_p) > k else -np.inf
                else:
                    k = move
                    corr = _safe_corr(check_p[:-k], check_c[k:]) if len(check_p) > k else -np.inf

                if corr > best_corr:
                    best_corr = corr
                    best_move = move

            shift_pos[x0, z + 1] = best_move
            done_steps += 1
            if progress_cb is not None and (done_steps % 5 == 0 or done_steps == total_steps):
                progress_cb(done_steps / total_steps, f"slice {z + 1}, row {x_1b}")

    # Convert from "move" to target border (R: 450 - move)
    shift_pos[:, 1:] = 450 - shift_pos[:, 1:]

    # Refine with smoothing spline over available x positions.
    refined = np.array(shift_pos, copy=True)
    x_idx = np.arange(n_x)
    for col in range(1, refined.shape[1]):
        valid = ~np.isnan(refined[:, col])
        if valid.sum() < 4:
            continue
        try:
            # Fit using the same distance axis as R (-200..3000) for stability.
            sp = UnivariateSpline(dist_axis[valid], refined[valid, col], k=3, s=10)
            refined[:, col] = sp(dist_axis)
        except Exception:
            # Fallback: linear interpolation over valid samples.
            xp = x_idx[valid]
            fp = refined[valid, col]
            if xp.size >= 2:
                refined[:, col] = np.interp(x_idx, xp, fp)

    # Additional regularization to suppress row-wise wobble that appears as
    # vertical distortion in flattened images.
    for col in range(1, refined.shape[1]):
        y = refined[:, col]
        if np.all(np.isnan(y)):
            continue

        valid = ~np.isnan(y)
        if valid.sum() < 2:
            continue

        # Fill missing values before filtering.
        y_filled = np.interp(x_idx, x_idx[valid], y[valid])

        # Median filter (window=9) to remove local outlier spikes.
        med_w = 9
        pad = med_w // 2
        y_pad = np.pad(y_filled, (pad, pad), mode='edge')
        y_med = np.empty_like(y_filled)
        for i in range(n_x):
            y_med[i] = np.median(y_pad[i:i + med_w])

        # Moving-average filter (window=11) for smooth trend.
        avg_w = 11
        k = np.ones(avg_w, dtype=np.float64) / avg_w
        y_smooth = np.convolve(np.pad(y_med, (avg_w // 2, avg_w // 2), mode='edge'), k, mode='valid')

        # Clamp abrupt row-to-row jumps (max 1 pixel/row).
        y_clamped = np.array(y_smooth, copy=True)
        for i in range(1, n_x):
            delta = y_clamped[i] - y_clamped[i - 1]
            if delta > 1.0:
                y_clamped[i] = y_clamped[i - 1] + 1.0
            elif delta < -1.0:
                y_clamped[i] = y_clamped[i - 1] - 1.0

        refined[:, col] = y_clamped

    refined[:, 1:] = np.round(refined[:, 1:], 0)
    if n_x > 200:
        refined[:199, 1:] = refined[199, 1:]
    return shift_pos, refined


def _apply_vertical_refinement(volume_raw, shift_refined):
    """Apply R-equivalent per-x border shift so border is aligned near 450."""
    n_x, n_y, n_z = volume_raw.shape
    out = np.zeros_like(volume_raw)

    for z in range(n_z):
        src = volume_raw[:, :, z]
        for x in range(n_x):
            border = shift_refined[x, z + 1]
            if np.isnan(border):
                continue
            border = int(border)
            if border < 450:
                length = 500 + (border - 449)
                src_start = (450 - border) - 1
                if length > 0 and src_start < n_y:
                    out[x, :length, z] = src[x, src_start:src_start + length]
            elif border == 450:
                out[x, :, z] = src[x, :]
            else:
                start_y = border - 450
                length = 500 - (border - 450)
                if start_y < n_y and length > 0:
                    out[x, start_y:start_y + length, z] = src[x, :length]
    return out


def _apply_vertical_refinement_markers(markers, shift_refined_col):
    n_x, n_y = markers.shape
    out = np.zeros_like(markers)
    for x in range(n_x):
        border = shift_refined_col[x]
        if np.isnan(border):
            continue
        border = int(border)
        if border < 450:
            length = 500 + (border - 449)
            src_start = (450 - border) - 1
            if length > 0 and src_start < n_y:
                out[x, :length] = markers[x, src_start:src_start + length]
        elif border == 450:
            out[x, :] = markers[x, :]
        else:
            start_y = border - 450
            length = 500 - (border - 450)
            if start_y < n_y and length > 0:
                out[x, start_y:start_y + length] = markers[x, :length]
    return out


def _best_lateral_moves(volume_refined, second_grand_mean, progress_cb=None):
    """R-equivalent lateral shift optimization (+/-39 rows)."""
    n_x, _, n_z = volume_refined.shape
    sgm = second_grand_mean[39:(n_x - 39), :]
    best = np.column_stack([np.arange(1, n_z + 1), np.zeros(n_z, dtype=np.int32)])

    for z in range(n_z):
        refine = volume_refined[:, :, z]
        best_corr = -np.inf
        best_move = 0
        for move in range(-39, 40):
            start = 39 + move
            end = (n_x - 39) + move
            if start < 0 or end > n_x:
                continue
            corr = _safe_corr(sgm.ravel(), refine[start:end, :].ravel())
            if corr > best_corr:
                best_corr = corr
                best_move = move
        best[z, 1] = best_move
        if progress_cb is not None:
            progress_cb((z + 1) / max(1, n_z), f"slice {z + 1}")
    return best


def _crop_rrc(volume_refined, best_lat_moves):
    """Crop to -100..2750 microns matching R script assignment semantics.

    The original R loop assigns the entire 3D array each iteration without [,,z],
    so the final result uses only the last lateral move across all slices.
    """
    _, n_y, n_z = volume_refined.shape
    out = np.full((2851, n_y, n_z), np.nan, dtype=np.float64)
    if len(best_lat_moves) == 0:
        return out

    move = int(best_lat_moves[-1, 1])
    start = (100 - move) - 1
    end = (2950 - move)

    src_start = max(0, start)
    src_end = min(volume_refined.shape[0], end)
    if src_start >= src_end:
        return out

    dst_start = max(0, src_start - start)
    dst_end = min(out.shape[0], dst_start + (src_end - src_start))
    width = dst_end - dst_start
    out[dst_start:dst_end, :, :] = volume_refined[src_start:src_start + width, :, :]
    return out


def _detect_vertex(final_grand_mean):
    """R-equivalent vertex detection around columns 434..466."""
    grand_profile = np.column_stack(
        (
            np.arange(1.0, final_grand_mean.shape[1] + 1.0, 1.0),
            _nanmean_axis0(final_grand_mean),
        )
    )

    gp = grand_profile[433:466, :]
    if gp.shape[0] == 0:
        return 431, grand_profile

    check_sp = _fit_smooth_spline_like_r(gp[:, 0], gp[:, 1], df=10)
    check_x = np.arange(434.0, 467.0, 1.0)
    check_spline = np.column_stack((check_x, check_sp(check_x), check_sp.derivative()(check_x)))
    threshold = float(np.quantile(check_spline[:, 1], 0.25))
    check_spline[:, 2] = np.where(check_spline[:, 1] < threshold, np.nan, check_spline[:, 2])

    positive = np.where(check_spline[:, 2] > 0)[0]
    if positive.size > 0:
        vertex = int(check_spline[positive[-1], 0] + 1.0)
    else:
        check_spline[:, 2] = check_spline[:, 2] - np.nanmedian(check_spline[:, 2])
        nonnegative = np.where(check_spline[:, 2] >= 0)[0]
        vertex = int(check_spline[nonnegative[-1], 0] + 1.0) if nonnegative.size else 431

    return vertex, grand_profile


def run_step3_pipeline(processor, progress_cb=None, diff_logger=None):
    """Run Step 3 using the R pipeline stages and return key outputs."""
    _emit_progress(progress_cb, 2, "Flattening slices from MARKED references")
    initial = processor.process_all_slices(
        progress_cb=lambda frac, label: _emit_progress(
            progress_cb,
            2 + (10 * frac),
            label,
        )
    )

    # Convert to R-style axis order: (x, y, z)
    dark_flat = np.transpose(initial['flattened_dark'], (1, 2, 0)).astype(np.float64)
    light_flat = np.transpose(initial['flattened_light'], (1, 2, 0)).astype(np.float64)
    markers = np.transpose(initial['markers'][0:1, :, :], (1, 2, 0)).astype(np.float64)[:, :, 0]
    apparent_angles_for_dark = np.asarray(initial["apparent_angles_for_dark"], dtype=np.float64)
    apparent_angles_for_light = np.asarray(initial["apparent_angles_for_light"], dtype=np.float64)
    if diff_logger is not None:
        diff_logger.capture(
            "01_flattened_input",
            {
                "dark_flat": dark_flat,
                "light_flat": light_flat,
                "markers": markers,
            },
        )

    _emit_progress(progress_cb, 12, "Converting log intensity to linear scale")
    dark_raw = _to_linearized(dark_flat)
    light_raw = _to_linearized(light_flat)
    if diff_logger is not None:
        diff_logger.capture("02_linearized", {"dark_raw": dark_raw, "light_raw": light_raw})

    _emit_progress(progress_cb, 20, "Computing first grand mean")
    # Mirror the R script exactly: initialize from DARK slice 1, then add z>=2 for
    # DARK and LIGHT. This intentionally omits LIGHT slice 1 in the accumulation.
    first_grand_mean = np.array(dark_raw[:, :, 0], copy=True)
    for z in range(1, dark_raw.shape[2]):
        first_grand_mean = first_grand_mean + dark_raw[:, :, z]
    for z in range(1, light_raw.shape[2]):
        first_grand_mean = first_grand_mean + light_raw[:, :, z]
    first_grand_mean = first_grand_mean / (dark_raw.shape[2] + light_raw.shape[2])
    if diff_logger is not None:
        diff_logger.capture("03_first_grand_mean", {"first_grand_mean": first_grand_mean})

    _emit_progress(progress_cb, 28, "Estimating vitreous/RPE lookup region")
    rough_vit = np.column_stack([np.arange(-200, -200 + markers.shape[0]), np.full(markers.shape[0], np.nan)])
    for x in range(markers.shape[0]):
        idx = np.where(markers[x, :] == 249)[0]
        if len(idx) > 0:
            rough_vit[x, 1] = idx[-1] + 1
    look_to = np.array(rough_vit, copy=True)
    look_to[:, 1] = np.round(250 - (0.7 * (250 - look_to[:, 1])), 0)
    if diff_logger is not None:
        diff_logger.capture("04_lookup_region", {"look_to": look_to, "rough_vit": rough_vit})

    _emit_progress(progress_cb, 40, "Computing vertical alignment for DARK")
    shift_dark, shift_dark_refined = _compute_shift_positions(
        dark_raw,
        first_grand_mean,
        look_to[:, 1],
        processor.pixel_width,
        progress_cb=lambda frac, label: _emit_progress(
            progress_cb,
            40 + (10 * frac),
            f"Computing vertical alignment for DARK ({label})",
        ),
    )

    _emit_progress(progress_cb, 50, "Computing vertical alignment for LIGHT")
    shift_light, shift_light_refined = _compute_shift_positions(
        light_raw,
        first_grand_mean,
        look_to[:, 1],
        processor.pixel_width,
        progress_cb=lambda frac, label: _emit_progress(
            progress_cb,
            50 + (10 * frac),
            f"Computing vertical alignment for LIGHT ({label})",
        ),
    )
    if diff_logger is not None:
        diff_logger.capture(
            "05_vertical_shifts",
            {
                "shift_dark": shift_dark,
                "shift_dark_refined": shift_dark_refined,
                "shift_light": shift_light,
                "shift_light_refined": shift_light_refined,
            },
        )

    _emit_progress(progress_cb, 60, "Applying refined vertical alignment")
    dark_refined = _apply_vertical_refinement(dark_raw, shift_dark_refined)
    light_refined = _apply_vertical_refinement(light_raw, shift_light_refined)
    markers_refined = _apply_vertical_refinement_markers(markers, shift_dark_refined[:, 1])
    if diff_logger is not None:
        diff_logger.capture(
            "06_vertical_refined",
            {
                "dark_refined": dark_refined,
                "light_refined": light_refined,
                "markers_refined": markers_refined,
            },
        )

    _emit_progress(progress_cb, 70, "Computing second grand mean")
    second_grand_mean = np.array(dark_refined[:, :, 0], copy=True)
    for z in range(1, dark_refined.shape[2]):
        second_grand_mean = second_grand_mean + dark_refined[:, :, z]
    for z in range(1, light_refined.shape[2]):
        second_grand_mean = second_grand_mean + light_refined[:, :, z]
    second_grand_mean = second_grand_mean / (dark_refined.shape[2] + light_refined.shape[2])
    if diff_logger is not None:
        diff_logger.capture("07_second_grand_mean", {"second_grand_mean": second_grand_mean})

    _emit_progress(progress_cb, 78, "Computing lateral alignment for DARK")
    best_lat_dark = _best_lateral_moves(
        dark_refined,
        second_grand_mean,
        progress_cb=lambda frac, label: _emit_progress(
            progress_cb,
            78 + (6 * frac),
            f"Computing lateral alignment for DARK ({label})",
        ),
    )

    _emit_progress(progress_cb, 84, "Computing lateral alignment for LIGHT")
    best_lat_light = _best_lateral_moves(
        light_refined,
        second_grand_mean,
        progress_cb=lambda frac, label: _emit_progress(
            progress_cb,
            84 + (6 * frac),
            f"Computing lateral alignment for LIGHT ({label})",
        ),
    )
    if diff_logger is not None:
        diff_logger.capture(
            "08_lateral_moves",
            {
                "best_lat_dark": best_lat_dark,
                "best_lat_light": best_lat_light,
            },
        )

    _emit_progress(progress_cb, 90, "Cropping to RRC (-100 to 2750 microns)")
    dark_rrc = _crop_rrc(dark_refined, best_lat_dark)
    light_rrc = _crop_rrc(light_refined, best_lat_light)
    first_dark_move = int(best_lat_dark[0, 1]) if len(best_lat_dark) > 0 else 0
    m_start = (100 - first_dark_move) - 1
    m_end = (2950 - first_dark_move)
    markers_rrc = markers_refined[m_start:m_end, :] if 0 <= m_start and m_end <= markers_refined.shape[0] else markers_refined
    if diff_logger is not None:
        diff_logger.capture(
            "09_rrc",
            {
                "dark_rrc": dark_rrc,
                "light_rrc": light_rrc,
                "markers_rrc": markers_rrc,
            },
        )

    _emit_progress(progress_cb, 95, "Detecting RPE vertex and final crop")
    final_grand_mean = np.array(dark_rrc[:, :, 0], copy=True)
    for z in range(1, dark_rrc.shape[2]):
        final_grand_mean = final_grand_mean + dark_rrc[:, :, z]
    for z in range(1, light_rrc.shape[2]):
        final_grand_mean = final_grand_mean + light_rrc[:, :, z]
    final_grand_mean = final_grand_mean / (dark_rrc.shape[2] + light_rrc.shape[2])
    vertex, grand_profile = _detect_vertex(final_grand_mean)

    c0 = max(0, (vertex - 430) - 1)
    c1 = min(dark_rrc.shape[1], vertex + 30)
    dark_final = dark_rrc[:, c0:c1, :]
    light_final = light_rrc[:, c0:c1, :]
    markers_final = markers_rrc[:, c0:c1]

    if diff_logger is not None:
        dark_export = np.transpose(np.nan_to_num(np.transpose(dark_final, (2, 0, 1)), nan=0.0), (0, 2, 1)).astype(np.float32)
        light_export = np.transpose(np.nan_to_num(np.transpose(light_final, (2, 0, 1)), nan=0.0), (0, 2, 1)).astype(np.float32)
        diff_logger.capture(
            "10_final",
            {
                "dark_final": dark_final,
                "light_final": light_final,
                "markers_final": markers_final,
                "final_grand_mean": final_grand_mean,
                "grand_profile": grand_profile,
                "dark_export": dark_export,
                "light_export": light_export,
            },
        )

    _emit_progress(progress_cb, 100, "Step 3 complete")

    # UI expects (slice, x, y) for quick rendering.
    return {
        'flattened_dark': np.transpose(dark_final, (2, 0, 1)),
        'flattened_light': np.transpose(light_final, (2, 0, 1)),
        'final_dark': np.transpose(dark_final, (2, 0, 1)),
        'final_light': np.transpose(light_final, (2, 0, 1)),
        'markers': markers_final,
        'first_grand_mean': first_grand_mean,
        'second_grand_mean': second_grand_mean,
        'final_grand_mean': final_grand_mean,
        'grand_profile': grand_profile,
        'vertex': vertex,
        'shift_dark': shift_dark,
        'shift_light': shift_light,
        'shift_dark_refined': shift_dark_refined,
        'shift_light_refined': shift_light_refined,
        'best_lateral_dark': best_lat_dark,
        'best_lateral_light': best_lat_light,
        'dark_rrc': dark_rrc,
        'light_rrc': light_rrc,
        'markers_rrc': markers_rrc,
        'apparent_angles_for_dark': apparent_angles_for_dark,
        'apparent_angles_for_light': apparent_angles_for_light,
    }


class RSetupWizard(ttk.Frame):
    """Guided R and R-package setup for the original Step 3 R script."""

    STEPS = (
        "Welcome",
        "R Program",
        "Download R",
        "Package Library",
        "Packages",
        "Finish",
    )

    def __init__(self, step_frame, parent, on_finish=None):
        super().__init__(parent)
        self.step_frame = step_frame
        self.on_finish = on_finish
        self.result = None
        self.cancelled = True
        self.current_step = 0
        self.busy = False
        self.rscript_path = step_frame._resolve_rscript_executable()
        self.installer_name = ""
        self.installer_url = ""
        self.installer_path = None
        self.package_status = {name: "pending" for name in step_frame.R_REQUIRED_PACKAGES}
        self.package_library_path = Path(
            getattr(step_frame, "r_package_library_path", None)
            or self._default_package_library()
        )
        self.log_path = self._package_log_path()

        self._build_styles()
        self._build_shell()
        self._render_step()
        self.focus_set()

    def _build_styles(self):
        self.style = ttk.Style(self)
        self.style.configure("WizardTitle.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("WizardSubtitle.TLabel", foreground="#555555")
        self.style.configure("WizardStep.TLabel", padding=(10, 7))
        self.style.configure("WizardStepActive.TLabel", padding=(10, 7), font=("Segoe UI", 9, "bold"))
        self.style.configure("WizardStepDone.TLabel", padding=(10, 7), foreground="#1b6e3c")
        self.style.configure("WizardAccent.TButton", padding=(10, 5))

    def _build_shell(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Install R for Step 3", style="WizardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="A guided setup for R, package libraries, and the packages required by this step.",
            style="WizardSubtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        middle = ttk.Frame(root)
        middle.pack(fill="both", expand=True)

        self.step_rail = ttk.Frame(middle, width=180)
        self.step_rail.pack(side="left", fill="y", padx=(0, 12))
        self.step_rail.pack_propagate(False)
        self.step_labels = []
        for label in self.STEPS:
            step_label = ttk.Label(self.step_rail, text=label, style="WizardStep.TLabel", anchor="w")
            step_label.pack(fill="x", pady=1)
            self.step_labels.append(step_label)

        right = ttk.Frame(middle)
        right.pack(side="left", fill="both", expand=True)

        self.content = ttk.Frame(right)
        self.content.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(right, text="Setup log")
        log_frame.pack(fill="both", expand=False, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(10, 0))
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.back_button = ttk.Button(footer, text="Back", command=self._back)
        self.back_button.pack(side="left", padx=(0, 4))
        self.next_button = ttk.Button(footer, text="Next", command=self._next)
        self.next_button.pack(side="left", padx=(0, 4))
        self.cancel_button = ttk.Button(footer, text="Cancel", command=self._cancel)
        self.cancel_button.pack(side="left")

        self._log("R setup wizard opened.")

    def _clear_content(self):
        for child in self.content.winfo_children():
            child.destroy()

    def _set_busy(self, busy, text=None, indeterminate=False):
        self.busy = bool(busy)
        for button in (self.back_button, self.next_button, self.cancel_button):
            button.configure(state="disabled" if busy else "normal")
        if indeterminate:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
        if text:
            self._log(text)
        if not busy:
            self._update_nav()

    def _set_progress(self, value):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=max(0, min(100, float(value))))

    def _update_nav(self):
        for idx, label in enumerate(self.step_labels):
            prefix = "[x] " if idx < self.current_step else ("[>] " if idx == self.current_step else "[ ] ")
            label.configure(text=prefix + self.STEPS[idx])
            if idx < self.current_step:
                label.configure(style="WizardStepDone.TLabel")
            elif idx == self.current_step:
                label.configure(style="WizardStepActive.TLabel")
            else:
                label.configure(style="WizardStep.TLabel")

        self.back_button.configure(state="disabled" if self.current_step == 0 else "normal")
        self.next_button.configure(text="Finish" if self.current_step == len(self.STEPS) - 1 else "Next")
        if self.current_step == 1 and self.rscript_path is None:
            self.next_button.configure(state="disabled")
        elif self.current_step == 2 and self.rscript_path is None:
            self.next_button.configure(state="disabled")
        elif self.current_step == 4 and not self._all_packages_ready():
            self.next_button.configure(state="disabled")
        else:
            self.next_button.configure(state="normal")

    def _render_step(self):
        self._clear_content()
        renderers = (
            self._render_welcome,
            self._render_r_program,
            self._render_download,
            self._render_library,
            self._render_packages,
            self._render_finish,
        )
        renderers[self.current_step]()
        self._set_progress((self.current_step / max(1, len(self.STEPS) - 1)) * 100)
        self._update_nav()

    def _section_title(self, title, subtitle):
        ttk.Label(self.content, text=title, style="WizardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            self.content,
            text=subtitle,
            style="WizardSubtitle.TLabel",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(4, 14))

    def _render_welcome(self):
        self._section_title(
            "Welcome",
            "This wizard installs the R runtime and the required R packages for Step 3.",
        )
        body = ttk.Frame(self.content)
        body.pack(fill="both", expand=True)
        items = (
            "Detect an existing Rscript executable.",
            "Download the official Windows R installer from CRAN if R is missing.",
            "Run the R installer and re-check the installed program.",
            "Choose a package-library folder that does not require administrator rights.",
            "Install AnalyzeFMRI and RNiftyReg with binary packages from CRAN.",
        )
        for item in items:
            ttk.Label(body, text=f"- {item}", wraplength=620, justify="left").pack(anchor="w", pady=3)
        ttk.Label(
            body,
            text=f"Full setup log:\n{self.log_path}",
            foreground="#555555",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(18, 0))

    def _render_r_program(self):
        self._section_title(
            "R Program",
            "AIDaS needs Rscript.exe to run the original Step 3 R workflow non-interactively.",
        )
        status = "Not found"
        if self.rscript_path is not None:
            status = str(self.rscript_path)
        self.r_status_var = tk.StringVar(value=status)

        form = ttk.LabelFrame(self.content, text="Detected Rscript")
        form.pack(fill="x")
        ttk.Label(form, textvariable=self.r_status_var, wraplength=620, justify="left").pack(
            anchor="w", padx=10, pady=10
        )

        actions = ttk.Frame(self.content)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Check Again", command=self._detect_rscript).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Locate Rscript...", command=self._locate_rscript).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Download R...", command=self._go_download).pack(side="left")

        ttk.Label(
            self.content,
            text="Choose Download R if R is not installed. Choose Locate Rscript if R is already installed but AIDaS cannot find it.",
            foreground="#555555",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _render_download(self):
        self._section_title(
            "Download And Install R",
            "Download the official Windows installer from CRAN, then run it. The R installer will ask where and how to install R.",
        )
        self.installer_path_var = tk.StringVar(value=str(self.installer_path or ""))
        self.installer_info_var = tk.StringVar(value=self.installer_name or "Installer has not been selected yet.")

        info = ttk.LabelFrame(self.content, text="Installer")
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.installer_info_var, wraplength=620, justify="left").pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        row = ttk.Frame(info)
        row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Entry(row, textvariable=self.installer_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Save As...", command=self._choose_installer_save_path).pack(side="left", padx=(6, 0))

        actions = ttk.Frame(self.content)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Find Latest Installer", command=self._find_latest_installer).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="Download Installer", command=self._download_selected_installer).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="Run Installer", command=self._run_downloaded_installer).pack(side="left")

        ttk.Label(
            self.content,
            text="After the installer finishes, this wizard checks again for Rscript.exe.",
            foreground="#555555",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _render_library(self):
        self._section_title(
            "Package Library",
            "R packages should be installed in a folder the current user can write to.",
        )
        self.library_var = tk.StringVar(value=str(self.package_library_path))
        frame = ttk.LabelFrame(self.content, text="R package-library folder")
        frame.pack(fill="x")
        row = ttk.Frame(frame)
        row.pack(fill="x", padx=10, pady=10)
        ttk.Entry(row, textvariable=self.library_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=self._browse_library).pack(side="left", padx=(6, 0))
        ttk.Label(
            self.content,
            text="Recommended: use the AIDaS folder under Local AppData. This avoids administrator permissions and keeps Step 3 packages separate from system R.",
            foreground="#555555",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _render_packages(self):
        self._section_title(
            "Required Packages",
            "Install the R packages used by the original Step 3 script.",
        )
        table = ttk.LabelFrame(self.content, text="Package status")
        table.pack(fill="x")
        self.package_status_vars = {}
        for package_name in self.step_frame.R_REQUIRED_PACKAGES:
            row = ttk.Frame(table)
            row.pack(fill="x", padx=10, pady=5)
            ttk.Label(row, text=package_name, width=18).pack(side="left")
            var = tk.StringVar(value=self.package_status.get(package_name, "pending"))
            self.package_status_vars[package_name] = var
            ttk.Label(row, textvariable=var).pack(side="left", fill="x", expand=True)

        actions = ttk.Frame(self.content)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Check Packages", command=self._check_packages).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Install Missing Packages", command=self._install_missing_packages).pack(side="left")

        ttk.Label(
            self.content,
            text=f"Packages will be installed to:\n{self.package_library_path}",
            foreground="#555555",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _render_finish(self):
        self._section_title(
            "Ready",
            "R and the Step 3 packages are ready. You can now run the original Step 3 R script.",
        )
        summary = (
            f"Rscript:\n{self.rscript_path}\n\n"
            f"Package library:\n{self.package_library_path}\n\n"
            f"Log:\n{self.log_path}"
        )
        ttk.Label(self.content, text=summary, wraplength=620, justify="left").pack(anchor="w")

    def _next(self):
        if self.busy:
            return
        if self.current_step == 3:
            if not self._save_library_choice():
                return
        if self.current_step == len(self.STEPS) - 1:
            self._finish()
            return
        if self.current_step == 1 and self.rscript_path is not None:
            self.current_step = 3
        else:
            self.current_step = min(len(self.STEPS) - 1, self.current_step + 1)
        if self.current_step == 2 and self.installer_url == "":
            self.after(100, self._find_latest_installer)
        if self.current_step == 4:
            self.after(100, self._check_packages)
        self._render_step()

    def _back(self):
        if self.busy:
            return
        self.current_step = max(0, self.current_step - 1)
        self._render_step()

    def _cancel(self):
        if self.busy:
            return
        self.cancelled = True
        self.result = None
        self.step_frame._close_r_setup_panel(render_previous=True)

    def _finish(self):
        self.cancelled = False
        self.result = Path(self.rscript_path) if self.rscript_path is not None else None
        if self.result is not None:
            self.step_frame.r_package_library_path = str(self.package_library_path)
            if self.step_frame.preferences is not None:
                self.step_frame.preferences.set("rscript_path", str(self.result))
                self.step_frame.preferences.set("r_package_library_path", str(self.package_library_path))
        callback = self.on_finish
        result = self.result
        self.step_frame._close_r_setup_panel(render_previous=callback is None)
        if callback is not None:
            self.step_frame.after(0, lambda: callback(result))

    def _go_download(self):
        self.current_step = 2
        self._render_step()
        if not self.installer_url:
            self._find_latest_installer()

    def _default_package_library(self):
        configured = getattr(self.step_frame, "r_package_library_path", None)
        if configured:
            return Path(configured)
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "AIDaS" / "R-packages"
        return Path.home() / "AIDaS_R_packages"

    def _package_log_path(self):
        output_dir = Path(self.step_frame.output_sdb_dir or self.step_frame.current_sdb_dir or os.getcwd())
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "step3_r_package_setup.log"

    def _log(self, message):
        text = f"{datetime.now().strftime('%H:%M:%S')}  {message}"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def _log_process_result(self, title, cmd, result):
        self._log(f"{title}: return code {result.returncode}")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write("Command:\n" + " ".join(str(part) for part in cmd) + "\n")
            handle.write("STDOUT:\n" + (result.stdout or "").rstrip() + "\n")
            handle.write("STDERR:\n" + (result.stderr or "").rstrip() + "\n\n")

    @staticmethod
    def _r_string(value):
        return "'" + str(value).replace("\\", "/").replace("'", "\\'") + "'"

    def _r_eval_command(self, expression):
        return self.step_frame._build_r_eval_command(self.rscript_path, expression)

    def _run_worker(self, title, worker, done):
        self._set_busy(True, title, indeterminate=True)

        def wrapped():
            try:
                value = worker()
                error = None
            except Exception as exc:
                value = None
                error = exc
            self.after(0, lambda: self._finish_worker(done, value, error))

        threading.Thread(target=wrapped, daemon=True).start()

    def _finish_worker(self, done, value, error):
        self._set_busy(False)
        done(value, error)

    def _detect_rscript(self):
        self.rscript_path = self.step_frame._resolve_rscript_executable()
        if hasattr(self, "r_status_var"):
            self.r_status_var.set(str(self.rscript_path) if self.rscript_path else "Not found")
        self._log(f"Rscript detection: {self.rscript_path or 'not found'}")
        self._update_nav()

    def _locate_rscript(self):
        selected = filedialog.askopenfilename(
            title="Locate Rscript executable",
            initialdir=r"C:\Program Files\R" if os.name == "nt" else (self.step_frame.current_sdb_dir or None),
            filetypes=[("Rscript executable", "Rscript*.exe"), ("All files", "*.*")],
        )
        if not selected:
            return
        rscript = self.step_frame._normalize_r_executable(Path(selected))
        if rscript is None:
            messagebox.showerror(
                "Locate Rscript executable",
                "Please select Rscript.exe, not R.exe, Rgui.exe, or RStudio.",
                parent=self,
            )
            return
        self.rscript_path = rscript
        if self.step_frame.preferences is not None:
            self.step_frame.preferences.set("rscript_path", str(rscript))
        self._detect_rscript()

    def _find_latest_installer(self):
        def worker():
            with urllib.request.urlopen(self.step_frame.R_DOWNLOAD_PAGE, timeout=30) as response:
                html = response.read().decode("utf-8", errors="replace")
            installers = sorted(
                set(re.findall(r'href=["\'](R-[0-9][^"\']+-win\.exe)["\']', html)),
                key=lambda name: [int(part) for part in re.findall(r"\d+", name)],
            )
            if not installers:
                raise RuntimeError(f"No Windows installer found at {self.step_frame.R_DOWNLOAD_PAGE}")
            name = installers[-1]
            return name, self.step_frame.R_DOWNLOAD_PAGE + name

        def done(value, error):
            if error:
                self._log(f"Could not find latest R installer: {error}")
                messagebox.showerror("R Setup", f"Could not find the latest R installer.\n{error}", parent=self)
                return
            self.installer_name, self.installer_url = value
            default_dir = Path(self.step_frame.current_sdb_dir or os.getcwd())
            self.installer_path = default_dir / self.installer_name
            if hasattr(self, "installer_info_var"):
                self.installer_info_var.set(f"{self.installer_name}\n{self.installer_url}")
            if hasattr(self, "installer_path_var"):
                self.installer_path_var.set(str(self.installer_path))
            self._log(f"Latest R installer: {self.installer_name}")

        self._run_worker("Finding latest R installer...", worker, done)

    def _choose_installer_save_path(self):
        initial_file = self.installer_name or "R-installer.exe"
        selected = filedialog.asksaveasfilename(
            title="Save R installer as",
            initialdir=self.step_frame.current_sdb_dir or None,
            initialfile=initial_file,
            defaultextension=".exe",
            filetypes=[("Windows installer", "*.exe"), ("All files", "*.*")],
            parent=self,
        )
        if selected:
            self.installer_path = Path(selected)
            self.installer_path_var.set(str(self.installer_path))
            self._log(f"Installer save path selected: {self.installer_path}")

    def _download_selected_installer(self):
        if not self.installer_url:
            messagebox.showwarning("R Setup", "Find the latest installer first.", parent=self)
            return
        path_text = self.installer_path_var.get().strip() if hasattr(self, "installer_path_var") else ""
        if not path_text:
            self._choose_installer_save_path()
            path_text = self.installer_path_var.get().strip()
        if not path_text:
            return
        self.installer_path = Path(path_text)
        if self.installer_path.exists():
            overwrite = messagebox.askyesno(
                "Overwrite Installer",
                f"This file already exists:\n{self.installer_path}\n\nOverwrite it?",
                parent=self,
            )
            if not overwrite:
                return

        def worker():
            self.installer_path.parent.mkdir(parents=True, exist_ok=True)

            def reporthook(block_count, block_size, total_size):
                if total_size > 0:
                    percent = min(100.0, (block_count * block_size / total_size) * 100.0)
                    self.after(0, lambda p=percent: self._set_progress(p))

            urllib.request.urlretrieve(self.installer_url, self.installer_path, reporthook=reporthook)
            return self.installer_path

        def done(value, error):
            if error:
                self._log(f"R installer download failed: {error}")
                messagebox.showerror("R Setup", f"Could not download R.\n{error}", parent=self)
                return
            self._log(f"Downloaded R installer: {value}")
            messagebox.showinfo("R Setup", "R installer downloaded. You can run it now.", parent=self)

        self._run_worker("Downloading R installer...", worker, done)

    def _run_downloaded_installer(self):
        path_text = self.installer_path_var.get().strip() if hasattr(self, "installer_path_var") else ""
        installer_path = Path(path_text) if path_text else self.installer_path
        if not installer_path or not installer_path.is_file():
            messagebox.showwarning("R Setup", "Download or select the R installer first.", parent=self)
            return

        def worker():
            result = subprocess.run([str(installer_path)], check=False)
            return result.returncode

        def done(value, error):
            if error:
                self._log(f"R installer could not be started: {error}")
                messagebox.showerror("R Setup", f"Could not run the R installer.\n{error}", parent=self)
                return
            self._log(f"R installer closed with return code {value}.")
            self._detect_rscript()
            if self.rscript_path is not None:
                messagebox.showinfo("R Setup", "Rscript was found. Continue to package setup.", parent=self)
                self.current_step = 3
                self._render_step()
            else:
                messagebox.showwarning(
                    "R Setup",
                    "AIDaS still cannot find Rscript. Finish the installer if it is still open, then click Check Again or Locate Rscript.",
                    parent=self,
                )

        self._run_worker("Running R installer. Complete the installer window to continue.", worker, done)

    def _browse_library(self):
        selected = filedialog.askdirectory(
            title="Select R package-library folder",
            initialdir=str(Path(self.library_var.get()).parent) if self.library_var.get() else None,
            parent=self,
        )
        if selected:
            self.library_var.set(selected)

    def _save_library_choice(self):
        library_path = Path(self.library_var.get().strip())
        if not str(library_path):
            messagebox.showwarning("Package Library", "Choose a package-library folder.", parent=self)
            return False
        try:
            library_path.mkdir(parents=True, exist_ok=True)
            test_path = library_path / ".aidas_write_test"
            test_path.write_text("ok", encoding="utf-8")
            test_path.unlink()
        except Exception as exc:
            messagebox.showerror(
                "Package Library",
                f"This folder is not writable:\n{library_path}\n\n{exc}",
                parent=self,
            )
            return False
        self.package_library_path = library_path.resolve()
        self.step_frame.r_package_library_path = str(self.package_library_path)
        if self.step_frame.preferences is not None:
            self.step_frame.preferences.set("r_package_library_path", str(self.package_library_path))
        self._log(f"Package library selected: {self.package_library_path}")
        return True

    def _package_check_expression(self, package_name):
        lib = self._r_string(self.package_library_path.resolve())
        return (
            f".libPaths(c({lib}, .libPaths())); "
            f"if (requireNamespace({self._r_string(package_name)}, quietly=TRUE)) "
            "quit(status=0) else quit(status=1)"
        )

    def _package_install_expression(self, package_name):
        lib = self._r_string(self.package_library_path.resolve())
        type_arg = ", type='binary'" if os.name == "nt" else ""
        pkg_type = "options(pkgType='win.binary'); " if os.name == "nt" else ""
        return "".join(
            (
                f".libPaths(c({lib}, .libPaths())); ",
                "options(repos=c(CRAN='https://cloud.r-project.org')); ",
                "options(install.packages.compile.from.source='never'); ",
                pkg_type,
                f"install.packages({self._r_string(package_name)}, ",
                "dependencies=c('Depends','Imports','LinkingTo'), ",
                f"lib={lib}{type_arg})",
            )
        )

    def _check_package_status_worker(self):
        if self.rscript_path is None:
            raise RuntimeError("Rscript is not selected.")
        self.package_library_path.mkdir(parents=True, exist_ok=True)
        statuses = {}
        for package_name in self.step_frame.R_REQUIRED_PACKAGES:
            expression = self._package_check_expression(package_name)
            cmd = self._r_eval_command(expression)
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._log_process_result(f"Check package {package_name}", cmd, result)
            statuses[package_name] = "installed" if result.returncode == 0 else "missing"
        return statuses

    def _check_packages(self):
        def done(value, error):
            if error:
                self._log(f"Package check failed: {error}")
                messagebox.showerror("R Packages", f"Could not check packages.\n{error}", parent=self)
                return
            self.package_status.update(value)
            for name, status in value.items():
                if hasattr(self, "package_status_vars"):
                    self.package_status_vars[name].set(status)
            self._update_nav()
            self._log("Package check completed.")

        self._run_worker("Checking R packages...", self._check_package_status_worker, done)

    def _install_missing_packages_worker(self):
        statuses = self._check_package_status_worker()
        env = os.environ.copy()
        env["R_LIBS_USER"] = str(self.package_library_path.resolve())
        env["R_INSTALL_STAGED"] = "false"
        for package_name, status in list(statuses.items()):
            if status == "installed":
                continue
            expression = self._package_install_expression(package_name)
            cmd = self._r_eval_command(expression)
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                input="n\n",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._log_process_result(f"Install package {package_name}", cmd, result)
            check_result = subprocess.run(
                self._r_eval_command(self._package_check_expression(package_name)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            statuses[package_name] = "installed" if result.returncode == 0 and check_result.returncode == 0 else "failed"
        return statuses

    def _install_missing_packages(self):
        def done(value, error):
            if error:
                self._log(f"Package installation failed: {error}")
                messagebox.showerror("R Packages", f"Could not install packages.\n{error}", parent=self)
                return
            self.package_status.update(value)
            for name, status in value.items():
                if hasattr(self, "package_status_vars"):
                    self.package_status_vars[name].set(status)
            self._update_nav()
            if self._all_packages_ready():
                self._log("All required R packages are installed.")
                messagebox.showinfo("R Packages", "All required packages are installed.", parent=self)
            else:
                self._log("Some R packages failed to install. See the setup log for details.")
                messagebox.showerror(
                    "R Packages",
                    f"Some packages failed to install.\n\nFull log:\n{self.log_path}",
                    parent=self,
                )

        self._run_worker("Installing missing R packages...", self._install_missing_packages_worker, done)

    def _all_packages_ready(self):
        return all(self.package_status.get(name) == "installed" for name in self.step_frame.R_REQUIRED_PACKAGES)


class RBatchSelectionPanel(ttk.Frame):
    """Embedded panel for selecting subfolders to run through the Step 3 R script."""

    TABLE_COLUMNS = (
        ("select", "", 42, "center"),
        ("folder", "Folder", 560, "w"),
        ("status", "Status", 380, "w"),
        ("inputs", "Inputs", 92, "center"),
    )
    COLUMN_MIN_WIDTHS = {
        "select": 42,
        "folder": 320,
        "status": 96,
        "inputs": 64,
    }
    COLUMN_MAX_WIDTHS = {
        "inputs": 92,
    }

    def __init__(self, step_frame, parent, root_dir):
        super().__init__(parent)
        self.step_frame = step_frame
        self.root_dir = Path(root_dir)
        self.rows = []

        self._build_ui()
        self._start_scan()

    def _build_ui(self):
        wrapper = ttk.Frame(self, padding=12)
        wrapper.pack(fill="both", expand=True)

        ttk.Label(wrapper, text="Batch Step 3 R Processing", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            wrapper,
            text=(
                "AIDaS will search the selected folder and subfolders for complete Step 3 inputs. "
                "Folders containing existing RData are shown as skipped and will not be processed."
            ),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        self.table = BatchTable(
            wrapper,
            columns=self.TABLE_COLUMNS,
            min_widths=self.COLUMN_MIN_WIDTHS,
            max_widths=self.COLUMN_MAX_WIDTHS,
            select_column="select",
            stretch_column="folder",
            empty_message="No folders with complete Step 3 inputs were found.",
        )
        self.table.pack(fill="both", expand=True)

        top = ttk.Frame(wrapper)
        top.pack(fill="x", pady=(0, 8))
        self.summary_var = tk.StringVar(value=f"Scanning: {self.root_dir}")
        ttk.Label(top, textvariable=self.summary_var, wraplength=760, justify="left").pack(
            side="left",
            fill="x",
            expand=True,
        )

        run_box = ttk.Frame(wrapper)
        run_box.pack(fill="x", pady=(10, 0))
        ttk.Button(run_box, text="Start", command=self._run_selected).pack(side="left")
        ttk.Label(run_box, text="Batch Size:").pack(side="left", padx=(12, 0))
        max_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
        self.workers_var = tk.IntVar(value=min(4, max_workers))
        self.workers_spin = ttk.Spinbox(
            run_box,
            from_=1,
            to=max_workers,
            textvariable=self.workers_var,
            width=5,
        )
        self.workers_spin.pack(side="left", padx=(6, 12))
        ttk.Button(run_box, text="Cancel", command=lambda: self.step_frame._render()).pack(side="right")

    def _start_scan(self):
        self.step_frame.status_var.set(f"Scanning subfolders under {self.root_dir}...")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        rows = []
        scanned = 0
        missing = 0
        try:
            folders = [self.root_dir]
            folders.extend(path for path in self.root_dir.rglob("*") if path.is_dir())
            for folder in folders:
                scanned += 1
                input_paths = self.step_frame._find_input_paths(folder)
                if any(input_paths.get(label) is None for label, *_rest in self.step_frame.REQUIRED_INPUTS):
                    missing += 1
                    continue
                has_rdata = self.step_frame._folder_has_r_data(folder)
                status = "Skipped: RData exists" if has_rdata else "Ready"
                try:
                    folder_text = str(folder.relative_to(self.root_dir))
                    if folder_text == ".":
                        folder_text = str(self.root_dir)
                except ValueError:
                    folder_text = str(folder)
                rows.append(
                    {
                        "folder": folder,
                        "include": not has_rdata,
                        "locked": has_rdata,
                        "status": status,
                        "values": {
                            "folder": folder_text,
                            "status": status,
                            "inputs": "4",
                        },
                    }
                )
        except Exception as exc:
            self.after(0, lambda: self._scan_failed(exc))
            return
        self.after(0, lambda: self._scan_done(rows, scanned, missing))

    def _scan_failed(self, exc):
        self.summary_var.set(f"Scan failed: {exc}")
        self.step_frame.status_var.set("Batch scan failed.")
        messagebox.showerror("Batch Step 3", f"Could not scan folders.\n{exc}", parent=self)

    def _scan_done(self, rows, scanned, missing):
        self.rows = rows
        self.table.set_rows(rows)
        ready = sum(1 for row in rows if not row["locked"])
        skipped = sum(1 for row in rows if row["locked"])
        self.summary_var.set(
            f"Scanned {scanned} folders. Found {ready} ready folder(s), {skipped} skipped folder(s) with RData. "
            f"{missing} folder(s) did not contain all four required inputs."
        )
        max_workers = max(1, min(ready or 1, (os.cpu_count() or 2) - 1, 8))
        self.workers_spin.configure(to=max_workers)
        self.workers_var.set(min(4, max_workers))
        self.step_frame.status_var.set("Batch scan complete. Select folders to process.")

    def _run_selected(self):
        folders = [row["folder"] for row in self.table.selected_rows()]
        if not folders:
            messagebox.showwarning("Batch Step 3", "Select at least one ready folder.", parent=self)
            return
        try:
            workers = max(1, int(self.workers_var.get()))
        except (TypeError, ValueError):
            workers = 1
        self.step_frame._start_batch_r_runs(folders, workers)


class RBatchRunPanel(ttk.Frame):
    """Embedded progress panel for concurrent folder-level R script runs."""

    TABLE_COLUMNS = (
        ("folder", "Folder", 560, "w"),
        ("status", "Status", 380, "w"),
        ("progress", "Progress", 92, "center"),
    )
    COLUMN_MIN_WIDTHS = {
        "folder": 320,
        "status": 160,
        "progress": 76,
    }
    COLUMN_MAX_WIDTHS = {
        "progress": 92,
    }

    def __init__(self, step_frame, parent, folders, workers):
        super().__init__(parent)
        self.step_frame = step_frame
        self.folders = [Path(folder) for folder in folders]
        self.workers = workers
        self.row_by_folder = {}
        self._build_ui()

    def _build_ui(self):
        wrapper = ttk.Frame(self, padding=12)
        wrapper.pack(fill="both", expand=True)
        ttk.Label(wrapper, text="Running Batch Step 3", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            wrapper,
            text="AIDaS is running the selected Step 3 R script folders. Progress and logs update as each folder finishes.",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        rows = []
        for folder in self.folders:
            row = {
                "folder": folder,
                "values": {
                    "folder": str(folder),
                    "status": "Queued",
                    "progress": "0%",
                },
            }
            rows.append(row)
            self.row_by_folder[str(folder)] = row

        self.table = BatchTable(
            wrapper,
            columns=self.TABLE_COLUMNS,
            min_widths=self.COLUMN_MIN_WIDTHS,
            max_widths=self.COLUMN_MAX_WIDTHS,
            stretch_column="folder",
            empty_message="No folders are queued.",
        )
        self.table.pack(fill="both", expand=True)
        self.table.set_rows(rows)

        self.summary_var = tk.StringVar(
            value=f"Running {len(self.folders)} folder(s) with up to {self.workers} parallel R process(es)."
        )
        ttk.Label(wrapper, textvariable=self.summary_var, wraplength=760, justify="left").pack(
            anchor="w", pady=(4, 10)
        )

        log_frame = ttk.LabelFrame(wrapper, text="Batch log")
        log_frame.pack(fill="both", expand=False, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def update_folder(self, folder, status=None, progress=None):
        row = self.row_by_folder.get(str(folder))
        if row is None:
            return
        values = dict(row.get("values") or {})
        if status is not None:
            values["status"] = status
        if progress is not None:
            values["progress"] = f"{int(max(0, min(100, float(progress))))}%"
        self.table.update_row(row, values=values)

    def set_summary(self, text):
        self.summary_var.set(text)

    def log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')}  {text}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


class Step3Frame(SidebarStepFrame):
    """Step 3 tab UI that runs this module's flattening pipeline inside the app."""
    TUTORIAL_IMAGE_NAME = "step3_tutorial.png"
    PIXEL_WIDTH_UM = 3.89
    MIN_NEGATIVE_UM = 200.0
    MIN_POSITIVE_UM = 3000.0
    MIN_DEPTH_OUTWARD_UM = 50.0
    MIN_DEPTH_INWARD_UM = 450.0
    CENTERED_FOVEA_GUARD_PX = 100
    REQUIRED_INPUTS = (
        ("Dark_MARKED", ("Dark_MARKED", "DARK_MARKED"), "Dark_MARKED.hdr/.img", 8),
        ("Light_MARKED", ("Light_MARKED", "LIGHT_MARKED"), "Light_MARKED.hdr/.img", 8),
        ("DARK", ("DARK", "Dark"), "DARK.hdr/.img", 16),
        ("LIGHT", ("LIGHT", "Light"), "LIGHT.hdr/.img", 16),
    )
    CORE_RESULT_FILES = (
        "_flat_DARK.hdr",
        "_flat_DARK.img",
        "_flat_LIGHT.hdr",
        "_flat_LIGHT.img",
        "DARK__and__LIGHT__flat.npz",
        "_done_DARK__and__LIGHT.npz",
    )
    R_SCRIPT_NAME = "RAW_OCT_PROCESSING_2023_09SEP-05_WSU.R"
    R_DOWNLOAD_PAGE = "https://cloud.r-project.org/bin/windows/base/"
    R_REQUIRED_PACKAGES = ("AnalyzeFMRI", "RNiftyReg")
    R_WORKSPACE_FILES = (
        "DARK__and__LIGHT__flat.RData",
        "_done_DARK__and__LIGHT.RData",
    )
    R_DISPLAY_FILES = (
        "DARK_MARKED_find_vertex.png",
        "DARK_MARKED_vertex.png",
    )
    R_ANALYZE_FILES = (
        "_flat_DARK.hdr",
        "_flat_DARK.img",
        "_flat_LIGHT.hdr",
        "_flat_LIGHT.img",
        "_flat-normed_DARK.hdr",
        "_flat-normed_DARK.img",
        "_flat-normed_LIGHT.hdr",
        "_flat-normed_LIGHT.img",
    )
    R_TABLE_FILES = (
        "_dark_profiles_DARK.txt",
        "_light_profiles_LIGHT.txt",
        "_fovea_dark_profiles_DARK.txt",
        "_fovea_light_profiles_LIGHT.txt",
    )
    R_OUTPUT_FILES = R_ANALYZE_FILES + R_WORKSPACE_FILES + R_DISPLAY_FILES + R_TABLE_FILES
    R_ARRAY_EXPORT_DIR = "step3_r_arrays"
    R_PROGRESS_BY_STEP = {
        "startup": (1, "Starting R script"),
        "input-config": (2, "Reading R input configuration"),
        "load-images": (5, "Loading Analyze volumes in R"),
        "fovea-center": (8, "Finding fovea center"),
        "rpe-line": (11, "Reading RPE line"),
        "rpe-spline": (14, "Fitting RPE spline"),
        "apparent-angle": (17, "Computing apparent angles"),
        "perpendiculars": (21, "Building perpendicular sampling lines"),
        "flattened-markers": (25, "Flattening marker image"),
        "dark-loop": (36, "Flattening DARK slices"),
        "light-loop": (47, "Flattening LIGHT slices"),
        "post-log-convert": (54, "Converting flattened data to raw scale"),
        "grand-mean": (59, "Building grand mean image"),
        "rough-vit-loop": (63, "Aligning retina profiles"),
        "python-export": (72, "Exporting R arrays for app loading"),
        "layer-borders": (78, "Identifying retinal layer borders"),
        "main-normalization": (86, "Spatially normalizing main retina"),
        "fovea-normalization": (92, "Spatially normalizing fovea"),
        "final-export": (97, "Writing final R outputs"),
        "done": (100, "R processing complete"),
    }

    def __init__(self, parent, preferences=None):
        super().__init__(parent)
        self.preferences = preferences

        self.current_sdb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.output_sdb_dir = self.current_sdb_dir
        self._output_dir_user_selected = False
        self.processor = None
        self.results = None
        self.figure = None
        self.canvas = None
        self._preview_photo = None
        self.r_setup_panel = None
        self.r_batch_panel = None
        self.r_batch_run_panel = None
        self.run_button_python = None
        self.run_button_r_script = None
        self.r_setup_button = None
        self.r_batch_button = None
        self.slice_combo = None
        self._busy = False
        self.last_diff_log_dir = None
        self.r_package_library_path = None if self.preferences is None else self.preferences.get("r_package_library_path")

        self.slice_var = tk.StringVar(value="0")
        self.view_var = tk.StringVar(value="Tutorial")
        self.status_var = tk.StringVar(value="Ready - checking Step 3 input files.")
        self.info_var = tk.StringVar(value="")
        self.progress_text_var = tk.StringVar(value="Idle")

        self._build_ui()
        self._refresh_input_status()

    def _build_ui(self):
        self.build_standard_layout(
            sidebar_width=self.SIDEBAR_WIDTH,
            sidebar_pack={"padx": (2, 6), "pady": 6},
            content_pack={"padx": 6, "pady": 6},
            status_var=self.status_var,
        )
        sources_section = self.add_sidebar_section("Input and Output Folders", padding=3, pady=(0, 5))
        sources = sources_section.body

        ttk.Button(sources, text="Select Input Folder", command=self._load_processor).pack(fill="x", pady=2)

        self.dir_var = tk.StringVar(value=f"Source: {self.current_sdb_dir}")
        ttk.Label(
            sources,
            textvariable=self.dir_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            foreground="gray",
            justify="left",
        ).pack(fill="x", pady=(2, 6))

        ttk.Button(sources, text="Select Output Folder", command=self._browse_output_folder).pack(fill="x", pady=2)

        self.output_dir_var = tk.StringVar(value=f"Output: {self.output_sdb_dir}")
        ttk.Label(
            sources,
            textvariable=self.output_dir_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            foreground="gray",
            justify="left",
        ).pack(fill="x", pady=(2, 8))

        process_section = self.add_sidebar_section("Process", padding=3, pady=(0, 5))
        process = process_section.body

        self.r_setup_button = ttk.Button(process, text="Setup R and Packages...", command=self._open_r_setup_wizard)
        self.r_setup_button.pack(fill="x", pady=2)

        self.run_button_r_script = ttk.Button(process, text="Run Step 3 (Original R Script)", command=self._run_r_script)
        self.run_button_r_script.pack(fill="x", pady=2)

        self.r_batch_button = ttk.Button(process, text="Batch Run R Script...", command=self._open_r_batch_scanner)
        self.r_batch_button.pack(fill="x", pady=2)

        self.run_button_python = ttk.Button(process, text="Run Step 3 (Python, experimental)", command=self._run_processing)
        self.run_button_python.pack(fill="x", pady=2)

        self.progress = ttk.Progressbar(process, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=2)
        progress_text_frame = ttk.Frame(process, height=44)
        progress_text_frame.pack(fill="x", pady=(0, 4))
        progress_text_frame.pack_propagate(False)
        ttk.Label(
            progress_text_frame,
            textvariable=self.progress_text_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            foreground="gray",
            justify="left",
        ).pack(fill="both", expand=True)

        view_results_section = self.add_sidebar_section("View Results", padding=3, pady=(0, 5))
        view_results = view_results_section.body

        ttk.Label(view_results, text="View").pack(anchor="w", pady=(0, 2))
        view_combo = ttk.Combobox(
            view_results,
            textvariable=self.view_var,
            values=["Tutorial", "Comparison", "DARK_MARKED_find_vertex", "DARK_MARKED_vertex"],
            state="readonly",
        )
        view_combo.pack(fill="x", pady=2)
        view_combo.bind("<<ComboboxSelected>>", lambda _: self._render())

        ttk.Label(view_results, text="Slice").pack(anchor="w", pady=(8, 2))
        self.slice_combo = ttk.Combobox(view_results, textvariable=self.slice_var, values=["0", "1"], state="readonly")
        self.slice_combo.pack(fill="x", pady=2)
        self.slice_combo.bind("<<ComboboxSelected>>", lambda _: self._render())

        stats_section = self.add_sidebar_section("Stats", padding=3, pady=(0, 5))
        stats = stats_section.body

        ttk.Label(
            stats,
            textvariable=self.info_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            justify="left",
        ).pack(fill="x")

        self.plot_holder = ttk.Frame(self.content)
        self.plot_holder.pack(fill="both", expand=True)
        self._render()

    def _set_process_buttons(self, state):
        if self.r_setup_button is not None:
            self.r_setup_button.configure(state=state)
        if self.r_batch_button is not None:
            self.r_batch_button.configure(state=state)
        if self.run_button_python is not None:
            self.run_button_python.configure(state=state)
        if self.run_button_r_script is not None:
            self.run_button_r_script.configure(state=state)

    @staticmethod
    def _script_path():
        return Path(__file__).resolve().parents[2] / Step3Frame.R_SCRIPT_NAME

    def _resolve_rscript_executable(self):
        configured = None if self.preferences is None else self.preferences.get("rscript_path")
        candidates = []
        if configured:
            candidates.append(Path(configured))

        env_override = os.environ.get("RSCRIPT_PATH") or os.environ.get("R_SCRIPT_PATH")
        if env_override:
            candidates.append(Path(env_override))

        for name in ("Rscript", "Rscript.exe"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))

        if os.name == "nt":
            for root in (Path(r"C:\Program Files\R"), Path(r"C:\Program Files (x86)\R")):
                if root.is_dir():
                    candidates.extend(root.glob("R*/bin/x64/Rscript.exe"))
                    candidates.extend(root.glob("R*/bin/Rscript.exe"))

        for candidate in candidates:
            candidate = self._normalize_r_executable(candidate)
            if candidate and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _normalize_r_executable(path):
        """Return a non-interactive R executable, preferring Rscript.exe.

        Users often select R.exe or Rgui.exe from the file dialog. Rgui/RStudio
        opens an interactive program and does not run this script as intended.
        If possible, convert those selections to the adjacent Rscript.exe.
        """
        if not path:
            return None
        path = Path(path)
        if not path.is_file():
            return None

        name = path.name.lower()
        if name in {"rscript.exe", "rscript"}:
            return path

        sibling_name = "Rscript.exe" if name.endswith(".exe") else "Rscript"
        sibling = path.with_name(sibling_name)
        if sibling.is_file():
            return sibling

        if name in {"r.exe", "rterm.exe", "r", "rterm"}:
            return path

        return None

    @staticmethod
    def _build_r_run_command(r_executable, script_path, script_args):
        name = Path(r_executable).name.lower()
        if name in {"rscript.exe", "rscript"}:
            return [str(r_executable), "--vanilla", str(script_path), *script_args]
        if name in {"r.exe", "rterm.exe", "r", "rterm"}:
            return [
                str(r_executable),
                "--vanilla",
                "--slave",
                f"--file={script_path}",
                "--args",
                *script_args,
            ]
        raise RuntimeError(
            "Select Rscript.exe, not the interactive R/RStudio program. "
            "Typical path: C:\\Program Files\\R\\R-x.x.x\\bin\\x64\\Rscript.exe"
        )

    @staticmethod
    def _build_r_eval_command(r_executable, expression):
        name = Path(r_executable).name.lower()
        if name in {"rscript.exe", "rscript"}:
            return [str(r_executable), "--vanilla", "-e", expression]
        if name in {"r.exe", "rterm.exe", "r", "rterm"}:
            return [str(r_executable), "--vanilla", "--slave", "-e", expression]
        raise RuntimeError("R package setup needs Rscript.exe or Rterm.exe.")

    def _download_r_installer(self):
        if os.name != "nt":
            raise RuntimeError("Automatic R download is currently supported on Windows only.")

        with urllib.request.urlopen(self.R_DOWNLOAD_PAGE, timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")

        installers = sorted(
            set(re.findall(r'href=["\'](R-[0-9][^"\']+-win\.exe)["\']', html)),
            key=lambda name: [int(part) for part in re.findall(r"\d+", name)],
        )
        if not installers:
            raise RuntimeError(f"Could not find a Windows R installer at {self.R_DOWNLOAD_PAGE}")

        installer_name = installers[-1]
        installer_url = self.R_DOWNLOAD_PAGE + installer_name
        download_dir = Path(tempfile.gettempdir()) / "AIDaS_R_setup"
        download_dir.mkdir(parents=True, exist_ok=True)
        installer_path = download_dir / installer_name

        self.status_var.set(f"Downloading {installer_name}...")
        self.progress_text_var.set("Downloading R installer...")
        self.update_idletasks()
        urllib.request.urlretrieve(installer_url, installer_path)
        return installer_path

    def _offer_r_install_or_locate(self):
        install_r = messagebox.askyesno(
            "R is required",
            "Step 3 can run the original R script, but Rscript was not found.\n\n"
            "Do you want AIDaS to download the official Windows R installer from CRAN or locate it manually? \n" \
            "Yes = Download and run installer\n " \
            "No = Select Rscript.exe manually",
        )

        if install_r:
            try:
                installer_path = self._download_r_installer()
            except Exception as exc:
                messagebox.showerror("Download R", f"Could not download R automatically.\n{exc}")
                self.status_var.set("R download failed. Select Rscript.exe manually to continue.")
            else:
                run_installer = messagebox.askyesno(
                    "Install R",
                    f"R installer downloaded to:\n{installer_path}\n\n"
                    "Do you want to run this installer now?",
                )
                if run_installer:
                    self.status_var.set("Running the R installer. Continue through the installer window.")
                    self.progress_text_var.set("Waiting for R installer...")
                    self.update_idletasks()
                    try:
                        subprocess.run([str(installer_path)], check=False)
                    except Exception as exc:
                        messagebox.showerror("Install R", f"Could not run the R installer.\n{exc}")
                    rscript = self._resolve_rscript_executable()
                    if rscript is not None:
                        if self.preferences is not None:
                            self.preferences.set("rscript_path", str(rscript))
                        return rscript
                    messagebox.showwarning(
                        "Install R",
                        "AIDaS still could not find Rscript after the installer closed.\n"
                        "If installation is still running, finish it first; otherwise select Rscript.exe manually.",
                    )

        selected = filedialog.askopenfilename(
            title="Locate Rscript executable",
            initialdir=r"C:\Program Files\R" if os.name == "nt" else (self.current_sdb_dir or None),
            filetypes=[("Rscript executable", "Rscript*.exe"), ("All files", "*.*")],
        )
        if not selected:
            self.status_var.set("Rscript was not found. Select an Rscript executable to continue.")
            return None
        rscript = self._normalize_r_executable(Path(selected))
        if rscript is None:
            messagebox.showerror(
                "Locate Rscript executable",
                "The selected program cannot run Step 3 non-interactively.\n\n"
                "Please select Rscript.exe, not R.exe, Rgui.exe, or RStudio.\n"
                "Typical path:\n"
                "C:\\Program Files\\R\\R-x.x.x\\bin\\x64\\Rscript.exe",
            )
            self.status_var.set("Select Rscript.exe to run the original Step 3 R script.")
            return None
        if self.preferences is not None:
            self.preferences.set("rscript_path", str(rscript))
        return rscript

    def _r_package_installed(self, rscript, package_name):
        expression = (
            f"if (requireNamespace({package_name!r}, quietly=TRUE)) "
            "quit(status=0) else quit(status=1)"
        )
        result = subprocess.run(
            self._build_r_eval_command(rscript, expression),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0

    def _install_r_package(self, rscript, package_name):
        expression = (
            "options(repos=c(CRAN='https://cloud.r-project.org')); "
            f"install.packages({package_name!r}, dependencies=TRUE)"
        )
        return subprocess.run(
            self._build_r_eval_command(rscript, expression),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _ensure_r_packages(self, rscript):
        for package_name in self.R_REQUIRED_PACKAGES:
            self.status_var.set(f"Checking R package {package_name}...")
            self.progress_text_var.set(f"Checking R package {package_name}...")
            self.update_idletasks()
            try:
                installed = self._r_package_installed(rscript, package_name)
            except Exception as exc:
                messagebox.showerror("R Package Check", f"Could not check R package {package_name}.\n{exc}")
                return False

            if installed:
                continue

            install = messagebox.askyesno(
                "Install R Package",
                f"The R package '{package_name}' is required for Step 3 and is not installed.\n\n"
                "Do you want AIDaS to install it from CRAN now?",
            )
            if not install:
                self.status_var.set(f"Missing R package: {package_name}.")
                return False

            self.status_var.set(f"Installing R package {package_name}...")
            self.progress_text_var.set(f"Installing R package {package_name}...")
            self.update_idletasks()
            result = self._install_r_package(rscript, package_name)
            if result.returncode != 0 or not self._r_package_installed(rscript, package_name):
                messagebox.showerror(
                    "Install R Package",
                    f"Could not install R package '{package_name}'.\n\n"
                    f"Last output:\n{self._short_process_text(result.stdout, max_lines=18)}",
                )
                self.status_var.set(f"R package installation failed: {package_name}.")
                return False

        self.status_var.set("Required R packages are installed.")
        return True

    @staticmethod
    def _analyze_base_name(base_path):
        return Path(str(base_path)).name

    @staticmethod
    def _r_index_string(slice_count):
        return ",".join(str(idx) for idx in range(1, int(slice_count) + 1))

    def _r_script_config_from_current_folder(self):
        input_paths, input_issues = self._refresh_input_status()
        if input_issues:
            raise RuntimeError(
                "Cannot run the R script because Step 3 inputs are incomplete:\n"
                + "\n".join(f"  - {item}" for item in input_issues)
            )

        input_info = self._read_input_stack_info(input_paths)
        self._validate_input_stack_shapes(input_info)
        output_dir = Path(self.output_sdb_dir or self.current_sdb_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "input_dir": str(Path(self.current_sdb_dir).resolve()),
            "output_dir": str(output_dir.resolve()),
            "reference_dark": self._analyze_base_name(input_paths["Dark_MARKED"]),
            "reference_light": self._analyze_base_name(input_paths["Light_MARKED"]),
            "to_process_dark": self._analyze_base_name(input_paths["DARK"]),
            "to_process_light": self._analyze_base_name(input_paths["LIGHT"]),
            "image_index_light": self._r_index_string(input_info["LIGHT"]["shape"][0]),
            "image_index_dark": self._r_index_string(input_info["DARK"]["shape"][0]),
            "pixel_width": str(self.PIXEL_WIDTH_UM),
        }

    @staticmethod
    def _short_process_text(text, max_lines=24):
        lines = [line for line in str(text or "").splitlines() if line.strip()]
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[-max_lines:])

    def _progress_from_r_line(self, line):
        match = re.search(r"DEBUG \[([^\]]+)\]\s*(.*)", str(line))
        if not match:
            return None
        step = match.group(1).strip()
        detail = match.group(2).strip()
        progress = self.R_PROGRESS_BY_STEP.get(step)
        if progress is None:
            return None
        percent, label = progress
        if detail and step in {"dark-loop", "light-loop"}:
            match_slice = re.search(r"Processing z=\s*(\d+)\s*of\s*(\d+)", detail)
            if match_slice:
                current = int(match_slice.group(1))
                total = max(1, int(match_slice.group(2)))
                span = 10.0
                percent = min(99.0, float(percent) + (span * (current - 1) / total))
                label = f"{label}: slice {current}/{total}"
        return percent, label

    def _handle_r_progress_line(self, line):
        progress = self._progress_from_r_line(line)
        if progress is None:
            return
        percent, label = progress
        self.after(0, lambda p=percent, text=label: self._update_progress(p, text))

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "values"):
            value = value.values
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        return np.asarray(value)

    def _load_r_workspace_results(self, output_dir):
        if pyreadr is None:
            raise RuntimeError("pyreadr is not installed, so R workspace files cannot be loaded.")

        output_dir = Path(output_dir)
        flat_rdata = output_dir / self.R_WORKSPACE_FILES[0]
        done_rdata = output_dir / self.R_WORKSPACE_FILES[1]
        if not flat_rdata.is_file() or not done_rdata.is_file():
            missing = [name for name in self.R_WORKSPACE_FILES if not (output_dir / name).is_file()]
            raise FileNotFoundError("Missing R workspace file(s): " + ", ".join(missing))

        flat_data = pyreadr.read_r(str(flat_rdata))
        done_data = pyreadr.read_r(str(done_rdata))

        def require(dataset, key):
            if key not in dataset:
                raise KeyError(f"R workspace file is missing required object: {key}")
            return self._to_numpy(dataset[key])

        flattened_dark = require(flat_data, "FLATTENED.DARK.RETINA.RRC")
        flattened_light = require(flat_data, "FLATTENED.LIGHT.RETINA.RRC")
        markers = require(flat_data, "FLATTENED.MARKERS.RRC")

        first_grand_mean = require(done_data, "FIRST.GRAND.MEAN") if "FIRST.GRAND.MEAN" in done_data else None
        second_grand_mean = require(done_data, "SECOND.GRAND.MEAN") if "SECOND.GRAND.MEAN" in done_data else None

        final_grand_mean = np.array(flattened_dark[:, :, 0], copy=True)
        for z in range(1, flattened_dark.shape[2]):
            final_grand_mean = final_grand_mean + flattened_dark[:, :, z]
        for z in range(1, flattened_light.shape[2]):
            final_grand_mean = final_grand_mean + flattened_light[:, :, z]
        final_grand_mean = final_grand_mean / (flattened_dark.shape[2] + flattened_light.shape[2])

        shift_dark = require(done_data, "SHIFT.POSITION.DARK") if "SHIFT.POSITION.DARK" in done_data else None
        shift_light = require(done_data, "SHIFT.POSITION.LIGHT") if "SHIFT.POSITION.LIGHT" in done_data else None
        shift_dark_refined = require(done_data, "SHIFT.POSITION.DARK.REFINED") if "SHIFT.POSITION.DARK.REFINED" in done_data else None
        shift_light_refined = require(done_data, "SHIFT.POSITION.LIGHT.REFINED") if "SHIFT.POSITION.LIGHT.REFINED" in done_data else None
        best_lateral_dark = require(done_data, "BEST.LAT.MOVE.DARK") if "BEST.LAT.MOVE.DARK" in done_data else None
        best_lateral_light = require(done_data, "BEST.LAT.MOVE.LIGHT") if "BEST.LAT.MOVE.LIGHT" in done_data else None
        if "APPARENT.ANGLES.FOR.DARK" in flat_data:
            apparent_angles_for_dark = require(flat_data, "APPARENT.ANGLES.FOR.DARK")
        else:
            dark_indices = np.arange(1, flattened_dark.shape[2] + 1, dtype=np.float64)
            apparent_angles_for_dark = np.column_stack((dark_indices, dark_indices, dark_indices))
        if "APPARENT.ANGLES.FOR.LIGHT" in flat_data:
            apparent_angles_for_light = require(flat_data, "APPARENT.ANGLES.FOR.LIGHT")
        else:
            light_indices = np.arange(1, flattened_light.shape[2] + 1, dtype=np.float64)
            apparent_angles_for_light = np.column_stack((light_indices, light_indices, light_indices))

        if "vertex" in done_data:
            vertex = int(np.ravel(self._to_numpy(done_data["vertex"]))[0])
        elif "vertex" in flat_data:
            vertex = int(np.ravel(self._to_numpy(flat_data["vertex"]))[0])
        else:
            vertex, _ = _detect_vertex(final_grand_mean)

        grand_profile = np.column_stack(
            (
                np.arange(1.0, final_grand_mean.shape[1] + 1.0, 1.0),
                np.nanmean(final_grand_mean, axis=0),
            )
        )

        results = {
            "flattened_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "flattened_light": np.transpose(flattened_light, (2, 0, 1)),
            "final_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "final_light": np.transpose(flattened_light, (2, 0, 1)),
            "markers": markers,
            "first_grand_mean": first_grand_mean,
            "second_grand_mean": second_grand_mean,
            "final_grand_mean": final_grand_mean,
            "grand_profile": grand_profile,
            "vertex": vertex,
            "shift_dark": shift_dark,
            "shift_light": shift_light,
            "shift_dark_refined": shift_dark_refined,
            "shift_light_refined": shift_light_refined,
            "best_lateral_dark": best_lateral_dark,
            "best_lateral_light": best_lateral_light,
            "apparent_angles_for_dark": apparent_angles_for_dark,
            "apparent_angles_for_light": apparent_angles_for_light,
            "dark_rrc": flattened_dark,
            "light_rrc": flattened_light,
            "markers_rrc": markers,
        }
        return results

    def _load_r_array_export(self, output_dir):
        export_dir = Path(output_dir) / self.R_ARRAY_EXPORT_DIR
        if not export_dir.is_dir():
            raise FileNotFoundError(f"Missing R array export folder: {export_dir}")

        def load_array(name, required=True):
            bin_path = export_dir / f"{name}.bin"
            shape_path = export_dir / f"{name}.shape"
            if not bin_path.is_file() or not shape_path.is_file():
                if required:
                    raise FileNotFoundError(f"Missing R array export: {name}")
                return None
            shape_text = shape_path.read_text(encoding="utf-8").strip()
            shape = tuple(int(part) for part in shape_text.split(",") if part.strip())
            data = np.fromfile(bin_path, dtype="<f8")
            expected = int(np.prod(shape)) if shape else 1
            if data.size != expected:
                raise ValueError(f"R array export {name} has {data.size} values; expected {expected}.")
            if not shape:
                return data
            return data.reshape(shape, order="F")

        flattened_dark = np.asarray(load_array("FLATTENED_DARK_RETINA_RRC"), dtype=np.float64)
        flattened_light = np.asarray(load_array("FLATTENED_LIGHT_RETINA_RRC"), dtype=np.float64)
        markers = np.asarray(load_array("FLATTENED_MARKERS_RRC"), dtype=np.float64)

        first_grand_mean = load_array("FIRST_GRAND_MEAN", required=False)
        second_grand_mean = load_array("SECOND_GRAND_MEAN", required=False)
        final_grand_mean = load_array("FINAL_GRAND_MEAN", required=False)
        if final_grand_mean is None:
            final_grand_mean = np.nanmean(np.concatenate((flattened_dark, flattened_light), axis=2), axis=2)
        if first_grand_mean is None:
            first_grand_mean = np.array(final_grand_mean, copy=True)
        if second_grand_mean is None:
            second_grand_mean = np.array(final_grand_mean, copy=True)

        grand_profile = load_array("GRAND_PROFILE", required=False)
        if grand_profile is None or grand_profile.shape[0] != final_grand_mean.shape[1]:
            grand_profile = np.column_stack(
                (
                    np.arange(1.0, final_grand_mean.shape[1] + 1.0, 1.0),
                    np.nanmean(final_grand_mean, axis=0),
                )
            )

        vertex_data = load_array("VERTEX", required=False)
        if vertex_data is not None and np.ravel(vertex_data).size:
            vertex = int(np.ravel(vertex_data)[0])
        else:
            vertex, _ = _detect_vertex(final_grand_mean)

        apparent_angles_for_dark = load_array("APPARENT_ANGLES_FOR_DARK", required=False)
        if apparent_angles_for_dark is None:
            dark_indices = np.arange(1, flattened_dark.shape[2] + 1, dtype=np.float64)
            apparent_angles_for_dark = np.column_stack((dark_indices, dark_indices, dark_indices))
        apparent_angles_for_light = load_array("APPARENT_ANGLES_FOR_LIGHT", required=False)
        if apparent_angles_for_light is None:
            light_indices = np.arange(1, flattened_light.shape[2] + 1, dtype=np.float64)
            apparent_angles_for_light = np.column_stack((light_indices, light_indices, light_indices))

        def optional_or_empty(name):
            value = load_array(name, required=False)
            return np.empty((0, 0), dtype=np.float64) if value is None else value

        return {
            "flattened_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "flattened_light": np.transpose(flattened_light, (2, 0, 1)),
            "final_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "final_light": np.transpose(flattened_light, (2, 0, 1)),
            "markers": markers,
            "first_grand_mean": first_grand_mean,
            "second_grand_mean": second_grand_mean,
            "final_grand_mean": final_grand_mean,
            "grand_profile": grand_profile,
            "vertex": vertex,
            "shift_dark": optional_or_empty("SHIFT_POSITION_DARK"),
            "shift_light": optional_or_empty("SHIFT_POSITION_LIGHT"),
            "shift_dark_refined": optional_or_empty("SHIFT_POSITION_DARK_REFINED"),
            "shift_light_refined": optional_or_empty("SHIFT_POSITION_LIGHT_REFINED"),
            "best_lateral_dark": optional_or_empty("BEST_LAT_MOVE_DARK"),
            "best_lateral_light": optional_or_empty("BEST_LAT_MOVE_LIGHT"),
            "apparent_angles_for_dark": apparent_angles_for_dark,
            "apparent_angles_for_light": apparent_angles_for_light,
            "dark_rrc": flattened_dark,
            "light_rrc": flattened_light,
            "markers_rrc": markers,
        }

    def _load_r_analyze_results(self, output_dir):
        output_dir = Path(output_dir)
        dark_base = output_dir / "_flat_DARK"
        light_base = output_dir / "_flat_LIGHT"
        if not (dark_base.with_suffix(".hdr").is_file() and light_base.with_suffix(".hdr").is_file()):
            raise FileNotFoundError("Missing R Analyze outputs _flat_DARK.hdr and _flat_LIGHT.hdr.")

        flattened_dark = _load_analyze_volume_r_layout(dark_base)
        flattened_light = _load_analyze_volume_r_layout(light_base)
        final_grand_mean = np.nanmean(np.concatenate((flattened_dark, flattened_light), axis=2), axis=2)
        grand_profile = np.column_stack(
            (
                np.arange(1.0, final_grand_mean.shape[1] + 1.0, 1.0),
                np.nanmean(final_grand_mean, axis=0),
            )
        )
        vertex, _ = _detect_vertex(final_grand_mean)
        dark_indices = np.arange(1, flattened_dark.shape[2] + 1, dtype=np.float64)
        light_indices = np.arange(1, flattened_light.shape[2] + 1, dtype=np.float64)

        return {
            "flattened_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "flattened_light": np.transpose(flattened_light, (2, 0, 1)),
            "final_dark": np.transpose(flattened_dark, (2, 0, 1)),
            "final_light": np.transpose(flattened_light, (2, 0, 1)),
            "markers": None,
            "first_grand_mean": final_grand_mean,
            "second_grand_mean": final_grand_mean,
            "final_grand_mean": final_grand_mean,
            "grand_profile": grand_profile,
            "vertex": vertex,
            "shift_dark": np.empty((0, 0), dtype=np.float64),
            "shift_light": np.empty((0, 0), dtype=np.float64),
            "shift_dark_refined": np.empty((0, 0), dtype=np.float64),
            "shift_light_refined": np.empty((0, 0), dtype=np.float64),
            "best_lateral_dark": np.empty((0, 0), dtype=np.float64),
            "best_lateral_light": np.empty((0, 0), dtype=np.float64),
            "apparent_angles_for_dark": np.column_stack((dark_indices, dark_indices, dark_indices)),
            "apparent_angles_for_light": np.column_stack((light_indices, light_indices, light_indices)),
            "dark_rrc": flattened_dark,
            "light_rrc": flattened_light,
            "markers_rrc": None,
        }

    def _load_r_results_with_fallbacks(self, output_dir):
        errors = []
        for loader in (self._load_r_workspace_results, self._load_r_array_export, self._load_r_analyze_results):
            try:
                results = loader(output_dir)
                return results, loader.__name__, errors
            except Exception as exc:
                errors.append(f"{loader.__name__}: {exc}")
        raise RuntimeError("Could not load R outputs using any supported method:\n" + "\n".join(errors))

    def _mirror_r_outputs(self, source_dir, destination_dir):
        source_dir = Path(source_dir)
        destination_dir = Path(destination_dir)
        if source_dir == destination_dir:
            return
        destination_dir.mkdir(parents=True, exist_ok=True)
        for name in self.R_OUTPUT_FILES:
            src = source_dir / name
            dst = destination_dir / name
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)

    def _load_r_script_results_into_ui(self, working_dir):
        results, loader_name, loader_errors = self._load_r_results_with_fallbacks(working_dir)
        self.results = results
        self._update_slice_options_from_results()
        self.progress.configure(value=100)
        self.progress_text_var.set("Completed")
        self.status_var.set(f"R script complete. Results loaded from {working_dir}.")
        self.info_var.set(
            f"flattened_dark: {results['flattened_dark'].shape}\n"
            f"flattened_light: {results['flattened_light'].shape}\n"
            f"final_grand_mean: {results['final_grand_mean'].shape}\n"
            f"vertex: {results['vertex']}\n"
            f"loaded via: {loader_name}"
        )
        if loader_errors:
            self.info_var.set(self.info_var.get() + "\n\nLoader fallbacks:\n" + "\n".join(loader_errors))
        self._render()
        return results

    def _update_slice_options_from_results(self):
        if self.slice_combo is None or self.results is None:
            return
        count = int(self.results["flattened_dark"].shape[0])
        values = [str(idx) for idx in range(max(1, count))]
        self.slice_combo.configure(values=values)
        if self.slice_var.get() not in values:
            self.slice_var.set(values[0])

    @staticmethod
    def _existing_basepath(folder, names):
        for name in names:
            base = os.path.join(folder, name)
            if os.path.isfile(base + ".hdr") and os.path.isfile(base + ".img"):
                return base
        return None

    @staticmethod
    def _analyze_stack_info(base_path):
        data = np.asarray(read_analyze(_normalize_analyze_path(base_path)))
        if data.ndim == 2:
            shape = (1, int(data.shape[0]), int(data.shape[1]))
        elif data.ndim == 3:
            shape = tuple(int(v) for v in data.shape)
        else:
            raise ValueError(f"Analyze file must be 2-D or 3-D, got shape {data.shape}.")
        return {
            "shape": shape,
            "dtype": str(data.dtype),
            "bits": int(data.dtype.itemsize * 8),
        }

    @classmethod
    def _analyze_stack_shape(cls, base_path):
        return cls._analyze_stack_info(base_path)["shape"]

    @staticmethod
    def _format_stack_info(label, info):
        bits_label = f"{info['bits']}-bit"
        return f"{label}: {info['shape']} | {bits_label} {info['dtype']}"

    @classmethod
    def _read_input_stack_info(cls, paths):
        return {label: cls._analyze_stack_info(path) for label, path in paths.items()}

    @classmethod
    def _validate_input_stack_shapes(cls, stack_info):
        shapes = {label: info["shape"] for label, info in stack_info.items()}
        expected = shapes["Dark_MARKED"]
        mismatched = {label: shape for label, shape in shapes.items() if shape != expected}
        if mismatched:
            lines = [f"Dark_MARKED: {expected}"]
            lines.extend(f"{label}: {shape}" for label, shape in mismatched.items())
            raise ValueError(
                "Step 3 inputs must all have the same Analyze stack shape "
                "(slices, height, width).\n" + "\n".join(lines)
            )
        return shapes

    def _find_input_paths(self, folder):
        return {
            label: self._existing_basepath(folder, names)
            for label, names, _display_name, _required_bits in self.REQUIRED_INPUTS
        }

    def _missing_input_names(self, input_paths):
        return [
            display_name
            for label, _names, display_name, _required_bits in self.REQUIRED_INPUTS
            if input_paths.get(label) is None
        ]

    def _input_requirement_issues(self, input_paths, input_info):
        issues = []
        for label, _names, display_name, required_bits in self.REQUIRED_INPUTS:
            if input_paths.get(label) is None:
                issues.append(display_name)
                continue
            info = input_info.get(label)
            if info is None:
                issues.append(f"{display_name} cannot be read")
                continue
            if info["bits"] != required_bits:
                issues.append(f"{display_name} must be {required_bits}-bit, found {info['bits']}-bit")
        return issues

    def _read_available_input_info(self, input_paths):
        input_info = {}
        read_errors = {}
        for label, path in input_paths.items():
            if path is None:
                continue
            try:
                input_info[label] = self._analyze_stack_info(path)
            except Exception as exc:
                read_errors[label] = str(exc)
        return input_info, read_errors

    def _format_input_checklist(self, input_paths, input_info=None, read_errors=None):
        input_info = {} if input_info is None else input_info
        read_errors = {} if read_errors is None else read_errors
        lines = []
        for label, _names, display_name, required_bits in self.REQUIRED_INPUTS:
            path = input_paths.get(label)
            if path is None:
                lines.append(f"❌ {display_name}")
            elif label in read_errors:
                lines.append(f"❌ {display_name} (cannot read)")
            else:
                info = input_info.get(label)
                if info is not None and info["bits"] == required_bits:
                    lines.append(f"✅ {display_name} ({required_bits}-bit)")
                elif info is not None:
                    lines.append(f"❌ {display_name} ({info['bits']}-bit, needs {required_bits}-bit)")
                else:
                    lines.append(f"❌ {display_name} (cannot read)")
        return "\n".join(lines)

    def _reset_to_tutorial_state(self):
        self.processor = None
        self.results = None
        self.view_var.set("Tutorial")
        if self.slice_combo is not None:
            self.slice_combo.configure(values=["0", "1"])
        self.slice_var.set("0")
        self.progress.configure(value=0)
        self.progress_text_var.set("Idle")
        self._render()

    def _refresh_input_status(self):
        if not self.current_sdb_dir:
            self._reset_to_tutorial_state()
            self.dir_var.set("Source: (no folder selected)")
            input_paths = {label: None for label, _names, _display_name, _required_bits in self.REQUIRED_INPUTS}
            self.info_var.set(
                "Step 3 input files:\n"
                + self._format_input_checklist(input_paths)
                + "\n\nSelect a folder containing MARKED and RAW Analyze files."
            )
            self.status_var.set("Missing Step 3 input folder.")
            return None, ["Step 3 input folder"]

        self.dir_var.set(f"Source: {self.current_sdb_dir}")
        self.output_dir_var.set(f"Output: {self.output_sdb_dir or '(no folder selected)'}")
        input_paths = self._find_input_paths(self.current_sdb_dir)
        input_info, read_errors = self._read_available_input_info(input_paths)
        issues = self._missing_input_names(input_paths)
        issues.extend(self._input_requirement_issues(input_paths, input_info))
        issues = list(dict.fromkeys(issues))

        if issues:
            self._reset_to_tutorial_state()
            self.info_var.set(
                "Step 3 input files:\n"
                + self._format_input_checklist(input_paths, input_info, read_errors)
            )
            self.status_var.set("Step 3 files are missing or do not meet bit-depth requirements.")
        else:
            self.info_var.set(
                "Step 3 is using these files:\n"
                + self._format_input_checklist(input_paths, input_info, read_errors)
            )
            self.status_var.set("All required Step 3 files found with correct bit depth. Ready to run.")

        return input_paths, issues

    def on_show(self):
        self._refresh_input_status()

    def set_input_folder(self, folder):
        if not folder:
            return
        self.current_sdb_dir = folder
        if not self._output_dir_user_selected:
            self.output_sdb_dir = folder
        self._refresh_input_status()

    def _browse_output_folder(self):
        folder = filedialog.askdirectory(
            title="Select output folder for Step 3 results",
            initialdir=self.output_sdb_dir or self.current_sdb_dir or None,
        )
        if folder:
            self.output_sdb_dir = folder
            self._output_dir_user_selected = True
            self.output_dir_var.set(f"Output: {self.output_sdb_dir}")
            if self._all_core_results_exist(self.output_sdb_dir):
                self._load_existing_results_from_output_folder(show_errors=True)
                return
            self.status_var.set(f"Step 3 output folder set to {self.output_sdb_dir}.")

    def _load_processor(self):
        selected_folder = filedialog.askdirectory(
            title="Select input folder containing Step 3 MARKED and RAW Analyze files",
            initialdir=self.current_sdb_dir or None,
        )
        if not selected_folder:
            return

        self.current_sdb_dir = selected_folder
        if not self._output_dir_user_selected:
            self.output_sdb_dir = selected_folder

        if self._all_core_results_exist(self.output_sdb_dir):
            self._load_existing_results_from_output_folder(show_errors=True)
            return

        self.status_var.set("Loading selected folder...")

        self._load_processor_from_current_folder(show_errors=True)

    def _all_core_results_exist(self, folder):
        if not folder:
            return False
        folder = Path(folder)
        npz_ready = all((folder / name).is_file() for name in self.CORE_RESULT_FILES)
        r_ready = all((folder / name).is_file() for name in self.R_WORKSPACE_FILES)
        return npz_ready or r_ready

    def _load_existing_results_from_output_folder(self, show_errors=False):
        if not self.output_sdb_dir:
            return False
        if not self._load_processor_from_current_folder(show_errors=show_errors):
            return False

        output_dir = Path(self.output_sdb_dir)
        try:
            flat_rdata = output_dir / self.R_WORKSPACE_FILES[0]
            done_rdata = output_dir / self.R_WORKSPACE_FILES[1]
            flat_npz = output_dir / "DARK__and__LIGHT__flat.npz"
            done_npz = output_dir / "_done_DARK__and__LIGHT.npz"
            if (
                (flat_rdata.is_file() and done_rdata.is_file())
                or (output_dir / self.R_ARRAY_EXPORT_DIR).is_dir()
                or (output_dir / "_flat_DARK.hdr").is_file()
            ):
                self.results, _loader_name, _loader_errors = self._load_r_results_with_fallbacks(output_dir)
            elif flat_npz.is_file() and done_npz.is_file():
                with np.load(flat_npz) as flat_data:
                    dark_rrc = np.asarray(flat_data["FLATTENED_DARK_RETINA_RRC"], dtype=np.float64)
                    light_rrc = np.asarray(flat_data["FLATTENED_LIGHT_RETINA_RRC"], dtype=np.float64)
                    markers = np.asarray(flat_data["FLATTENED_MARKERS_RRC"], dtype=np.float64)
                    vertex = int(np.ravel(flat_data["VERTEX"])[0])
                    self.results = {
                        "flattened_dark": np.transpose(dark_rrc, (2, 0, 1)),
                        "flattened_light": np.transpose(light_rrc, (2, 0, 1)),
                        "final_dark": np.transpose(dark_rrc, (2, 0, 1)),
                        "final_light": np.transpose(light_rrc, (2, 0, 1)),
                        "markers": markers,
                        "first_grand_mean": np.asarray(flat_data["FIRST_GRAND_MEAN"], dtype=np.float64),
                        "second_grand_mean": np.asarray(flat_data["SECOND_GRAND_MEAN"], dtype=np.float64),
                        "final_grand_mean": np.asarray(flat_data["FINAL_GRAND_MEAN"], dtype=np.float64),
                        "grand_profile": np.asarray(flat_data["GRAND_PROFILE"], dtype=np.float64),
                        "vertex": vertex,
                        "shift_dark": np.asarray(flat_data["SHIFT_POSITION_DARK"], dtype=np.float64),
                        "shift_light": np.asarray(flat_data["SHIFT_POSITION_LIGHT"], dtype=np.float64),
                        "shift_dark_refined": np.asarray(flat_data["SHIFT_POSITION_DARK_REFINED"], dtype=np.float64),
                        "shift_light_refined": np.asarray(flat_data["SHIFT_POSITION_LIGHT_REFINED"], dtype=np.float64),
                        "best_lateral_dark": np.asarray(flat_data["BEST_LAT_MOVE_DARK"], dtype=np.float64),
                        "best_lateral_light": np.asarray(flat_data["BEST_LAT_MOVE_LIGHT"], dtype=np.float64),
                        "dark_rrc": dark_rrc,
                        "light_rrc": light_rrc,
                        "markers_rrc": markers,
                    }
            else:
                raise FileNotFoundError("No usable Step 3 result files were found in the output folder.")
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Load Results", f"Could not load existing Step 3 results.\n{exc}")
            self.status_var.set("Could not load existing Step 3 results.")
            return False

        self.progress.configure(value=100)
        self.progress_text_var.set("Loaded existing results")
        self.status_var.set(f"Loaded existing Step 3 results from {output_dir}.")
        self._update_slice_options_from_results()
        self._render()
        return True

    def _load_processor_from_current_folder(self, show_errors=False):
        input_paths, input_issues = self._refresh_input_status()

        if input_issues:
            if show_errors:
                messagebox.showerror(
                    "Load",
                    "Cannot load Step 3 because these input requirements are not met:\n"
                    + "\n".join(f"  - {item}" for item in input_issues)
                    + "\n\nStep 3 requires 8-bit MARKED Analyze files and 16-bit raw DARK/LIGHT Analyze files.",
                )
            return False

        try:
            input_info = self._read_input_stack_info(input_paths)
            self._validate_input_stack_shapes(input_info)
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Load", f"Cannot load Step 3 inputs.\n{exc}")
            self.status_var.set("Step 3 input dimensions do not match.")
            return False

        self.processor = OCTFlatteningProcessor(
            reference_dark_path=input_paths["Dark_MARKED"],
            reference_light_path=input_paths["Light_MARKED"],
            dark_path=input_paths["DARK"],
            light_path=input_paths["LIGHT"],
            image_index_dark=[0, 1],
            image_index_light=[0, 1],
            pixel_width=3.89,
        )

        self.results = None
        if self.slice_combo is not None:
            slice_count = int(input_info["DARK"]["shape"][0])
            self.slice_combo.configure(values=[str(idx) for idx in range(max(1, slice_count))])
            self.slice_var.set("0")
        self.view_var.set("Comparison")
        self._render()
        self.info_var.set(
            f"{self._format_stack_info('Dark_MARKED ',input_info['Dark_MARKED'])}\n"
            f"{self._format_stack_info('Light_MARKED ', input_info['Light_MARKED'])}\n"
            f"{self._format_stack_info('DARK ', input_info['DARK'])}\n"
            f"{self._format_stack_info('LIGHT ', input_info['LIGHT'])}\n")
        self.status_var.set("Loaded Step 3 inputs with bit depth info. Ready to run.")
        return True

    def _run_processing(self):
        if self.processor is None:
            if not self._load_processor_from_current_folder(show_errors=True):
                return
        if self._busy:
            return
        if self._all_core_results_exist(self.output_sdb_dir):
            overwrite = messagebox.askyesno(
                "Overwrite Step 3 Results",
                "Step 3 results already exist in the output folder.\n\n"
                "Running Step 3 will overwrite those files. Are you sure?",
            )
            if not overwrite:
                return

        self._busy = True
        self._set_process_buttons("disabled")
        self.progress.configure(value=0)
        self.progress_text_var.set("Starting...")
        self.status_var.set("Running Step 3 pipeline and final outputs...")
        threading.Thread(target=self._process_worker, daemon=True).start()

    def _clear_plot_holder(self):
        if self.canvas is not None:
            try:
                widget = self.canvas.get_tk_widget() if hasattr(self.canvas, "get_tk_widget") else self.canvas
                widget.destroy()
            except Exception:
                pass
            self.canvas = None
        for child in self.plot_holder.winfo_children():
            child.destroy()
        self.figure = None
        self._preview_photo = None
        self.r_setup_panel = None
        self.r_batch_panel = None
        self.r_batch_run_panel = None

    def _open_r_setup_wizard(self, on_finish=None):
        self._clear_plot_holder()
        self.r_setup_panel = RSetupWizard(self, self.plot_holder, on_finish=on_finish)
        self.r_setup_panel.pack(fill="both", expand=True)
        self.status_var.set("Step 3 R setup is open in the preview area.")
        self.progress_text_var.set("R setup")
        return None

    def _close_r_setup_panel(self, *, render_previous):
        panel = self.r_setup_panel
        self.r_setup_panel = None
        if panel is not None:
            try:
                panel.destroy()
            except Exception:
                pass
        if render_previous:
            self._render()

    def _open_r_batch_scanner(self):
        if self._busy:
            return
        root_dir = filedialog.askdirectory(
            title="Select root folder for batch Step 3 R processing",
            initialdir=self.current_sdb_dir or None,
        )
        if not root_dir:
            return
        self._clear_plot_holder()
        self.r_batch_panel = RBatchSelectionPanel(self, self.plot_holder, Path(root_dir))
        self.r_batch_panel.pack(fill="both", expand=True)
        self.progress_text_var.set("Batch scan")
        self.status_var.set(f"Scanning batch root: {root_dir}")

    def _folder_has_r_data(self, folder):
        folder = Path(folder)
        if any((folder / name).is_file() for name in self.R_WORKSPACE_FILES):
            return True
        return any(path.is_file() for path in folder.glob("*.RData"))

    def _r_script_config_for_folder(self, folder):
        folder = Path(folder)
        input_paths = self._find_input_paths(folder)
        missing = self._missing_input_names(input_paths)
        if missing:
            raise RuntimeError("Missing Step 3 inputs: " + ", ".join(missing))
        input_info = self._read_input_stack_info(input_paths)
        requirement_issues = self._input_requirement_issues(input_paths, input_info)
        if requirement_issues:
            raise RuntimeError("Input requirement issue(s): " + "; ".join(requirement_issues))
        self._validate_input_stack_shapes(input_info)
        return {
            "input_dir": str(folder.resolve()),
            "output_dir": str(folder.resolve()),
            "reference_dark": self._analyze_base_name(input_paths["Dark_MARKED"]),
            "reference_light": self._analyze_base_name(input_paths["Light_MARKED"]),
            "to_process_dark": self._analyze_base_name(input_paths["DARK"]),
            "to_process_light": self._analyze_base_name(input_paths["LIGHT"]),
            "image_index_light": self._r_index_string(input_info["LIGHT"]["shape"][0]),
            "image_index_dark": self._r_index_string(input_info["DARK"]["shape"][0]),
            "pixel_width": str(self.PIXEL_WIDTH_UM),
        }

    def _start_batch_r_runs(self, folders, workers):
        folders = [Path(folder) for folder in folders]
        if not folders:
            messagebox.showwarning("Batch Step 3", "Select at least one folder to process.")
            return
        if self._busy:
            return
        script_path = self._script_path()
        if not script_path.is_file():
            messagebox.showerror("Batch Step 3", f"Could not find the R script:\n{script_path}")
            return
        rscript = self._ensure_r_ready_with_wizard()
        if rscript is None:
            self._open_r_setup_wizard(
                on_finish=lambda result: self._start_batch_r_runs(folders, workers) if result else None
            )
            return

        workers = max(1, min(int(workers), len(folders)))
        self._clear_plot_holder()
        self.r_batch_run_panel = RBatchRunPanel(self, self.plot_holder, folders, workers)
        self.r_batch_run_panel.pack(fill="both", expand=True)
        self._busy = True
        self._set_process_buttons("disabled")
        self.progress.configure(value=0)
        self.progress_text_var.set("Batch running")
        self.status_var.set(f"Running Step 3 R script for {len(folders)} folder(s).")
        threading.Thread(
            target=self._batch_r_worker,
            args=(Path(rscript), script_path, folders, workers),
            daemon=True,
        ).start()

    def _batch_panel_update(self, folder, status=None, progress=None, log=None):
        panel = self.r_batch_run_panel
        if panel is None:
            return
        if status is not None or progress is not None:
            panel.update_folder(folder, status=status, progress=progress)
        if log:
            panel.log(log)

    def _batch_r_worker(self, rscript_path, script_path, folders, workers):
        results = []
        completed = 0
        total = len(folders)

        def run_folder(folder):
            folder = Path(folder)
            self.after(0, lambda f=folder: self._batch_panel_update(f, status="Validating", progress=0))
            try:
                if self._folder_has_r_data(folder):
                    raise RuntimeError("Skipped because this folder contains RData.")
                r_config = self._r_script_config_for_folder(folder)
            except Exception as exc:
                return {"folder": folder, "returncode": 1, "stdout": "", "stderr": str(exc), "cmd": []}
            return self._run_r_script_for_config(rscript_path, script_path, r_config, batch_folder=folder)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(run_folder, folder): folder for folder in folders}
            for future in concurrent.futures.as_completed(future_map):
                folder = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"folder": folder, "returncode": 1, "stdout": "", "stderr": str(exc), "cmd": []}
                results.append(result)
                completed += 1
                overall = (completed / max(1, total)) * 100.0
                status = "Completed" if result["returncode"] == 0 else "Failed"
                self.after(
                    0,
                    lambda f=folder, s=status, o=overall: (
                        self._batch_panel_update(f, status=s, progress=100),
                        self.progress.configure(value=o),
                    ),
                )

        self.after(0, lambda: self._on_batch_r_done(results))

    def _run_r_script_for_config(self, rscript_path, script_path, r_config, batch_folder=None):
        input_dir = Path(r_config["input_dir"])
        output_dir = Path(r_config["output_dir"])
        script_args = [
            r_config["input_dir"],
            r_config["output_dir"],
            r_config["reference_dark"],
            r_config["reference_light"],
            r_config["to_process_dark"],
            r_config["to_process_light"],
            r_config["image_index_light"],
            r_config["image_index_dark"],
            r_config["pixel_width"],
        ]
        cmd = self._build_r_run_command(rscript_path, script_path, script_args)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = os.environ.copy()
        if self.r_package_library_path:
            env["R_LIBS_USER"] = str(self.r_package_library_path)
        process = subprocess.Popen(
            cmd,
            cwd=str(input_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            bufsize=1,
        )
        output_lines = []
        if batch_folder is not None:
            self.after(0, lambda f=batch_folder: self._batch_panel_update(f, status="Running", progress=1))
        if process.stdout is not None:
            for line in process.stdout:
                output_lines.append(line)
                if batch_folder is not None:
                    progress = self._progress_from_r_line(line)
                    if progress is not None:
                        percent, label = progress
                        self.after(
                            0,
                            lambda f=batch_folder, p=percent, text=label: self._batch_panel_update(
                                f, status=text, progress=p
                            ),
                        )
        returncode = process.wait()
        stdout = "".join(output_lines)
        log_path = self._write_r_run_log(output_dir, returncode, stdout, "", cmd)
        if batch_folder is not None:
            self.after(
                0,
                lambda f=batch_folder, rc=returncode, lp=log_path: self._batch_panel_update(
                    f,
                    log=f"{f}: {'completed' if rc == 0 else 'failed'}; log: {lp}",
                ),
            )
        return {
            "folder": input_dir,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": "",
            "cmd": cmd,
            "log_path": log_path,
        }

    def _on_batch_r_done(self, results):
        self._busy = False
        self._set_process_buttons("normal")
        success = sum(1 for result in results if result["returncode"] == 0)
        failed = len(results) - success
        self.progress.configure(value=100)
        self.progress_text_var.set("Batch completed")
        self.status_var.set(f"Batch Step 3 complete: {success} succeeded, {failed} failed.")
        if self.r_batch_run_panel is not None:
            self.r_batch_run_panel.set_summary(f"Batch complete: {success} succeeded, {failed} failed.")
            self.r_batch_run_panel.log(f"Batch complete: {success} succeeded, {failed} failed.")
        self.info_var.set(
            "Batch Step 3 R results:\n"
            + "\n".join(
                f"{'OK' if result['returncode'] == 0 else 'FAILED'}: {result['folder']}"
                for result in results
            )
        )

    def _ensure_r_ready_with_wizard(self):
        rscript = self._resolve_rscript_executable()
        if rscript is not None and self._packages_ready_for_rscript(rscript):
            return Path(rscript)
        self.status_var.set("Open R setup in the preview area to finish installation.")
        return None

    def _packages_ready_for_rscript(self, rscript):
        library_path = self.r_package_library_path
        if not library_path:
            return False
        library_path = Path(library_path)
        if not library_path.is_dir():
            return False
        lib_prefix = f".libPaths(c({RSetupWizard._r_string(library_path.resolve())}, .libPaths())); "
        for package_name in self.R_REQUIRED_PACKAGES:
            expression = (
                lib_prefix
                + f"if (requireNamespace({RSetupWizard._r_string(package_name)}, quietly=TRUE)) "
                "quit(status=0) else quit(status=1)"
            )
            try:
                result = subprocess.run(
                    self._build_r_eval_command(rscript, expression),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception:
                return False
            if result.returncode != 0:
                return False
        return True

    def _run_r_script(self):
        if self.processor is None:
            if not self._load_processor_from_current_folder(show_errors=True):
                return
        if self._busy:
            return

        script_path = self._script_path()
        if not script_path.is_file():
            messagebox.showerror(
                "Run Step 3 (R Script)",
                f"Could not find the R script:\n{script_path}",
            )
            return

        try:
            r_config = self._r_script_config_from_current_folder()
        except Exception as exc:
            messagebox.showerror("Run Step 3 (R Script)", str(exc))
            self.status_var.set("Step 3 R script inputs are not ready.")
            return

        rscript = self._ensure_r_ready_with_wizard()
        if rscript is None:
            self._open_r_setup_wizard(on_finish=lambda _result: self._run_r_script())
            return

        if self._all_core_results_exist(self.output_sdb_dir):
            overwrite = messagebox.askyesno(
                "Overwrite Step 3 Results",
                "Step 3 results already exist in the output folder.\n\n"
                "Running the R script will overwrite those files. Are you sure?",
            )
            if not overwrite:
                return

        self._busy = True
        self._set_process_buttons("disabled")
        self.progress.configure(value=0)
        self.progress_text_var.set("Launching Rscript...")
        self.status_var.set("Running Step 3 R script in the background...")
        threading.Thread(
            target=self._r_script_worker,
            args=(Path(rscript), script_path, r_config),
            daemon=True,
        ).start()

    def _r_script_worker(self, rscript_path, script_path, r_config):
        input_dir = Path(r_config["input_dir"])
        output_dir = Path(r_config["output_dir"])
        script_args = [
            r_config["input_dir"],
            r_config["output_dir"],
            r_config["reference_dark"],
            r_config["reference_light"],
            r_config["to_process_dark"],
            r_config["to_process_light"],
            r_config["image_index_light"],
            r_config["image_index_dark"],
            r_config["pixel_width"],
        ]
        try:
            cmd = self._build_r_run_command(rscript_path, script_path, script_args)
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            env = os.environ.copy()
            if self.r_package_library_path:
                env["R_LIBS_USER"] = str(self.r_package_library_path)
            process = subprocess.Popen(
                cmd,
                cwd=str(input_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
                bufsize=1,
            )
            output_lines = []
            if process.stdout is not None:
                for line in process.stdout:
                    output_lines.append(line)
                    self._handle_r_progress_line(line)
            returncode = process.wait()
            stdout = "".join(output_lines)
            self.after(
                0,
                lambda: self._on_r_script_done(
                    returncode,
                    stdout,
                    "",
                    input_dir,
                    output_dir,
                    cmd,
                ),
            )
        except Exception as exc:
            try:
                cmd
            except NameError:
                cmd = [str(rscript_path), str(script_path), *script_args]
            self.after(0, lambda: self._on_r_script_done(1, "", str(exc), input_dir, output_dir, cmd))

    def _write_r_run_log(self, output_dir, returncode, stdout, stderr, cmd):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "step3_rscript.log"
        log_path.write_text(
            "Command:\n"
            + " ".join(str(part) for part in cmd)
            + f"\n\nReturn code: {returncode}\n\nSTDOUT:\n{stdout or ''}\n\nSTDERR:\n{stderr or ''}\n",
            encoding="utf-8",
        )
        return log_path

    def _on_r_script_done(self, returncode, stdout, stderr, input_dir, output_dir, cmd):
        self._busy = False
        self._set_process_buttons("normal")

        output_dir = Path(output_dir)
        log_path = self._write_r_run_log(output_dir, returncode, stdout, stderr, cmd)

        if returncode != 0:
            self.progress_text_var.set("Failed")
            self.status_var.set("Step 3 R script failed.")
            message = ["The R script did not complete successfully."]
            if stdout.strip():
                message.append("\nSTDOUT (last lines):\n" + self._short_process_text(stdout))
            if stderr.strip():
                message.append("\nSTDERR (last lines):\n" + self._short_process_text(stderr))
            message.append(f"\nInput directory: {input_dir}")
            message.append(f"Output directory: {output_dir}")
            message.append(f"Full log saved to:\n{log_path}")
            messagebox.showerror("Step 3 (R Script)", "\n".join(message))
            return

        try:
            self._mirror_r_outputs(input_dir, output_dir)
            self._load_r_script_results_into_ui(output_dir)
        except Exception as exc:
            self.progress_text_var.set("Completed, but load failed")
            self.status_var.set(f"R script completed, but the results could not be loaded: {exc}")
            messagebox.showerror(
                "Step 3 (R Script)",
                f"The R script finished, but the app could not load its results.\n{exc}\n\nFull log saved to:\n{log_path}",
            )
            return

        if stdout.strip() or stderr.strip():
            log_text = []
            if stdout.strip():
                log_text.append(self._short_process_text(stdout, max_lines=12))
            if stderr.strip():
                log_text.append(self._short_process_text(stderr, max_lines=12))
            self.info_var.set(self.info_var.get() + "\n\nR output log:\n" + str(log_path) + "\n" + "\n".join(log_text))

    def _threadsafe_progress(self, percent, label):
        self.after(0, lambda: self._update_progress(percent, label))

    def _update_progress(self, percent, label):
        self.progress.configure(value=max(0, min(100, float(percent))))
        self.progress_text_var.set(label)
        self.status_var.set(f"Step 3: {label}")

    def _process_worker(self):
        diff_logger = None
        output_dir = self.output_sdb_dir or self.current_sdb_dir
        try:
            results = run_step3_pipeline(
                self.processor,
                progress_cb=self._threadsafe_progress,
                diff_logger=None,
            )

            summary_path = None
            save_error = self._save_generated_outputs(
                results=results,
                output_dir=output_dir,
                progress_cb=self._threadsafe_progress,
                ui_updates=False,
            )
            more_outputs = None
            more_error = None
            if save_error is None:
                try:
                    more_outputs = self._generate_more_outputs(
                        output_dir=output_dir,
                        results=results,
                        progress_cb=self._threadsafe_progress,
                    )
                except Exception as exc:
                    more_error = exc
            self.after(
                0,
                lambda: self._on_processing_done(
                    results,
                    None,
                    summary_path,
                    None,
                    save_error,
                    output_dir,
                    more_outputs,
                    more_error,
                ),
            )
        except Exception as exc:
            self.after(0, lambda error=exc: self._on_processing_done(None, error, None, None, None, output_dir))

    def _on_processing_done(
        self,
        results,
        error,
        summary_path=None,
        diff_logger=None,
        save_error=None,
        save_dir=None,
        more_outputs=None,
        more_error=None,
    ):
        self._busy = False
        self._set_process_buttons("normal")

        if error is not None:
            self.status_var.set(f"Step 3 failed: {error}")
            self.progress_text_var.set("Failed")
            messagebox.showerror("Step 3", f"Processing failed.\n{error}")
            return

        self.results = results
        self._update_slice_options_from_results()
        if summary_path is not None:
            self.last_diff_log_dir = str(Path(summary_path).parent)
        self.progress.configure(value=100)
        if save_error is not None:
            self.progress_text_var.set("Save failed")
            self.status_var.set(f"Step 3 complete, but auto-save failed: {save_error}")
        elif more_error is not None:
            self.progress_text_var.set("Final outputs failed")
            self.status_var.set(f"Step 3 core outputs saved, but final output generation failed: {more_error}")
        elif summary_path is not None:
            self.progress_text_var.set("Completed")
            msg = "Step 3 complete. Diff log saved."
            if diff_logger is not None and diff_logger.first_divergence is not None:
                first = diff_logger.first_divergence
                msg = (
                    f"Step 3 complete. First divergence: {first['stage']}::{first['array']} "
                    f"(see {summary_path})."
                )
            self.status_var.set(msg)
        else:
            self.progress_text_var.set("Completed")
            self.status_var.set(
                f"Step 3 complete. All outputs saved to {save_dir or self.output_sdb_dir or self.current_sdb_dir}"
            )
        self._render()
        if more_outputs:
            self.info_var.set(
                self.info_var.get()
                + "\n\nFinal outputs:\n"
                + "\n".join(str(path.name) for path in more_outputs.values())
            )
        if more_error is not None:
            messagebox.showerror(
                "Step 3",
                f"Step 3 core processing completed, but final output generation failed.\n{more_error}",
            )

    @staticmethod
    def _resource_path(relative_path):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        return base_dir / relative_path

    def _tutorial_asset_path(self):
        return self._resource_path(Path("assets") / self.TUTORIAL_IMAGE_NAME)

    def _display_preview_image(self, image, background="#ffffff"):
        label = tk.Label(self.plot_holder, bg=background, borderwidth=0, highlightthickness=0)
        label.pack(fill="both", expand=True)
        source = image.convert("RGB")

        def redraw(_event=None):
            try:
                if not label.winfo_exists():
                    return
                width = max(1, int(label.winfo_width()))
                height = max(1, int(label.winfo_height()))
            except tk.TclError:
                return
            if width <= 1 or height <= 1:
                return
            fitted = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), background)
            canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
            self._preview_photo = ImageTk.PhotoImage(canvas)
            try:
                label.configure(image=self._preview_photo)
            except tk.TclError:
                return

        label.bind("<Configure>", redraw, add="+")
        self.canvas = label
        self.after(0, redraw)

    def _render_tutorial(self):
        tutorial_path = self._tutorial_asset_path()
        if tutorial_path.is_file():
            with Image.open(tutorial_path) as img:
                image = img.copy()
        else:
            image = _placeholder_image(
                f"Missing Step 3 tutorial asset:\n{tutorial_path}",
                size=(1800, 1100),
                title="Step 3 Tutorial",
            )
            self.status_var.set(f"Step 3 tutorial image not found: {tutorial_path}")
        self.info_var.set(self._tutorial_info_text())
        if tutorial_path.is_file():
            self.status_var.set("Step 3 tutorial: using static asset image.")
        self._display_preview_image(image)

    def _result_info_text(self):
        return (
            f"flattened_dark: {self.results['flattened_dark'].shape}\n"
            f"flattened_light: {self.results['flattened_light'].shape}\n"
            f"final_grand_mean: {self.results['final_grand_mean'].shape}\n"
            f"vertex: {self.results['vertex']}"
        )

    def _render(self):
        view = self.view_var.get()
        self._clear_plot_holder()

        if view == "Tutorial":
            self._render_tutorial()
            return

        if self.results is None:
            self.status_var.set("Step 3 inputs loaded. Run Step 3 to view results.")
            return

        try:
            slice_idx = int(self.slice_var.get())
        except Exception:
            slice_idx = 0
        max_slice = self.results["flattened_dark"].shape[0] - 1
        slice_idx = max(0, min(slice_idx, max_slice))

        if view == "DARK_MARKED_find_vertex":
            image = _make_find_vertex_preview_image(self.results)
            self.status_var.set("Showing DARK_MARKED_find_vertex preview.")
        elif view == "DARK_MARKED_vertex":
            vertex_plot_path = Path(self.output_sdb_dir or self.current_sdb_dir) / "DARK_MARKED_vertex.png"
            if vertex_plot_path.is_file():
                with Image.open(vertex_plot_path) as img:
                    image = img.copy()
                self.status_var.set("Showing DARK_MARKED_vertex.png.")
            else:
                image = _placeholder_image(
                    f"DARK_MARKED_vertex.png not found in:\n{vertex_plot_path.parent}",
                    size=(1600, 1000),
                    title="DARK_MARKED_vertex",
                )
                self.status_var.set(f"DARK_MARKED_vertex.png not found in {vertex_plot_path.parent}.")
        else:
            if self.processor is None or getattr(self.processor, "light", None) is None:
                image = _placeholder_image(
                    "The original LIGHT volume is not loaded.\nRun Step 3 from an input folder to view the comparison.",
                    size=(1600, 1000),
                    title="Comparison",
                )
                self.status_var.set("Comparison preview needs the original LIGHT volume.")
            else:
                image = _make_comparison_preview_image(
                    self.processor.light,
                    self.results["flattened_light"],
                    slice_idx,
                )
                self.status_var.set(f"Showing Step 3 comparison for slice {slice_idx}.")
        self._display_preview_image(image)

        self.info_var.set(self._result_info_text())

    def _tutorial_info_text(self):
        left_px = int(np.ceil(self.MIN_NEGATIVE_UM / self.PIXEL_WIDTH_UM))
        right_px = int(np.ceil(self.MIN_POSITIVE_UM / self.PIXEL_WIDTH_UM))
        source_width_px = int(np.ceil((self.MIN_NEGATIVE_UM + self.MIN_POSITIVE_UM) / self.PIXEL_WIDTH_UM))
        outward_px = int(np.ceil(self.MIN_DEPTH_OUTWARD_UM / self.PIXEL_WIDTH_UM))
        inward_px = int(np.ceil(self.MIN_DEPTH_INWARD_UM / self.PIXEL_WIDTH_UM))
        safe_centered_side_px = right_px + self.CENTERED_FOVEA_GUARD_PX
        return (
            "Step 3 tutorial minimums:\n"
            f"Pixel width: {self.PIXEL_WIDTH_UM:g} um/input px\n"
            f"Fovea to near side: >= {left_px} px ({self.MIN_NEGATIVE_UM:g} um)\n"
            f"Fovea to far side: >= {right_px} px ({self.MIN_POSITIVE_UM:g} um)\n"
            f"Minimum RPE marker coverage: about {source_width_px} px\n"
            f"Centered fovea minimum: >= {right_px * 2} px\n"
            f"Centered fovea recommended: >= {safe_centered_side_px * 2} px "
            f"({safe_centered_side_px} px per side)\n"
            f"Height around RPE: >= {inward_px} px from top and >= {outward_px} px from bottom\n"
            f"Centered RPE height: >= {inward_px * 2} px"
        )

    def _generate_more_outputs(self, output_dir, results=None, progress_cb=None):
        output_dir = Path(output_dir)
        flat_npz = output_dir / "DARK__and__LIGHT__flat.npz"
        done_npz = output_dir / "_done_DARK__and__LIGHT.npz"

        if not flat_npz.exists() or not done_npz.exists():
            _emit_progress(progress_cb, 98, "Saving Step 3 checkpoints before final outputs")
            save_error = self._save_generated_outputs(
                results=results,
                output_dir=output_dir,
                progress_cb=progress_cb,
                ui_updates=False,
            )
            if save_error is not None:
                raise RuntimeError(save_error)

        _emit_progress(progress_cb, 99, "Generating tissue-border plots and R-format tables")
        outputs = _main_run_more_outputs_from_step3_npz(flat_npz, done_npz, output_dir)
        _emit_progress(progress_cb, 100, "Completed")
        return outputs

    @staticmethod
    def _prepare_export_volume(volume):
        return np.transpose(np.nan_to_num(volume, nan=0.0), (0, 2, 1)).astype(np.float32)

    def _save_generated_outputs(self, results=None, output_dir=None, progress_cb=None, ui_updates=True):
        results = self.results if results is None else results
        output_dir = self.output_sdb_dir if output_dir is None else output_dir
        if results is None or not output_dir:
            return None

        try:
            _emit_progress(progress_cb, 96, "Saving flattened Analyze outputs")
            dark_out = os.path.join(output_dir, "_flat_DARK")
            light_out = os.path.join(output_dir, "_flat_LIGHT")
            dark_export = self._prepare_export_volume(results['final_dark'])
            light_export = self._prepare_export_volume(results['final_light'])
            write_analyze(dark_out, dark_export)
            write_analyze(light_out, light_export)
            _save_main_style_exports(
                results,
                output_dir,
                progress_cb=lambda pct, label: _emit_progress(
                    progress_cb,
                    96 + (3 * pct / 100.0),
                    label,
                ),
            )

            # Save profile-style plots matching main.py: find_vertex and vertex
            _emit_progress(progress_cb, 99, "Saving profile plots")
            find_path = os.path.join(output_dir, "DARK_MARKED_find_vertex.png")
            _save_profile_plot(results['grand_profile'], find_path, "Find Vertex", verticals=(450.0, 434.0, 466.0))

            # Compute the local check_spline and save the vertex plot (matching main.py logic)
            gp = np.array(results['grand_profile'], copy=True)
            # Slice 433:466 (1-based indexing in main.py => 433:466 zero-based slice)
            gp_slice = gp[433:466, :]
            try:
                check_sp = _fit_smooth_spline_like_r(gp_slice[:, 0], gp_slice[:, 1], df=10)
                check_x = np.arange(434.0, 467.0, 1.0)
                check_spline = np.column_stack((check_x, check_sp(check_x), check_sp.derivative()(check_x)))
                threshold = float(np.quantile(check_spline[:, 1], 0.25))
                check_spline[:, 2] = np.where(check_spline[:, 1] < threshold, np.nan, check_spline[:, 2])
            except Exception:
                check_spline = None

            vertex_plot_path = os.path.join(output_dir, "DARK_MARKED_vertex.png")
            if gp_slice.size:
                _save_profile_plot(gp_slice, vertex_plot_path, "Vertex", verticals=(results['vertex'],), spline_xy=(check_spline[:, 0:2] if check_spline is not None else None))

            _emit_progress(progress_cb, 100, "Completed")
            if ui_updates:
                self.progress_text_var.set("Completed")
                if self.last_diff_log_dir:
                    self.info_var.set(self.info_var.get() + f"\n\ndiff_log: {self.last_diff_log_dir}")
                self.status_var.set(f"Step 3 complete. Outputs saved to {output_dir}")
            return None
        except Exception as exc:
            if ui_updates:
                self.status_var.set(f"Step 3 complete, but auto-save failed: {exc}")
            return str(exc)


def main():
    """Main processing function."""
    
    # Configuration (matching R script)
    REFERENCE_DARK = "DARK_MARKED"
    REFERENCE_LIGHT = "LIGHT_MARKED"
    TO_PROCESS_DARK = "DARK"
    TO_PROCESS_LIGHT = "LIGHT"
    PIXEL_WIDTH = 3.89  # microns per pixel
    
    IMAGE_INDEX_DARK = [1, 2]
    IMAGE_INDEX_LIGHT = [1, 2]
    
    # Initialize processor
    processor = OCTFlatteningProcessor(
        REFERENCE_DARK, REFERENCE_LIGHT,
        TO_PROCESS_DARK, TO_PROCESS_LIGHT,
        IMAGE_INDEX_DARK, IMAGE_INDEX_LIGHT,
        PIXEL_WIDTH
    )
    
    def cli_progress(pct, msg):
        print(f"[{int(pct):3d}%] {msg}")

    results = run_step3_pipeline(processor, progress_cb=cli_progress)

    print("✓ Processing complete")
    print(f"  Dark flattened shape: {results['flattened_dark'].shape}")
    print(f"  Light flattened shape: {results['flattened_light'].shape}")
    print(f"  Final grand mean shape: {results['final_grand_mean'].shape}")
    print(f"  Vertex: {results['vertex']}")

    _make_main_results_summary_image(results).save("oct_flattening_results.png")
    print("✓ Plots saved to 'oct_flattening_results.png'")
    return results


if __name__ == "__main__":
    main()
