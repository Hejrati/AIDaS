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

import numpy as np
from scipy.interpolate import BSpline, UnivariateSpline, interp1d
from scipy.stats import pearsonr
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from aidas.utils.io_utils import read_analyze, write_analyze
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


def _fit_line_coeffs(points):
    """Return R-style linear-model coefficients [intercept, slope]."""
    arr = np.asarray(points, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    design = np.column_stack((np.ones_like(x), x))
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeffs.astype(np.float64)


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
        
        # Spline parameters
        self.df_initial_spline = 10
        self.df_second_fit = 10
        
        # Load images
        self.ref_dark = self._load_analyze(reference_dark_path)
        self.ref_light = self._load_analyze(reference_light_path)
        self.dark = self._load_analyze(dark_path)
        self.light = self._load_analyze(light_path)
    
    @staticmethod
    def _load_analyze(base_path):
        """Load Analyze data using the same axis convention as main.py."""
        return _load_analyze_volume_r_layout(base_path)
    
    def find_rpe_and_fovea(self, slice_idx=0, reference_volume=None):
        """Extract RPE spline and fovea center using main.py logic."""
        ref_volume = self.ref_dark if reference_volume is None else reference_volume
        ref = ref_volume[:, :, slice_idx]
        xs, ys = _build_coordinate_grids(ref)

        fovea_mask = np.asarray(ref, dtype=np.float64).copy()
        fovea_mask[fovea_mask < 243] = np.nan
        fovea_mask[fovea_mask > 243] = np.nan
        fovea_mask[fovea_mask == 243] = 1.0

        xcoords = xs * fovea_mask
        ycoords = ys * fovea_mask
        mask = ~np.isnan(xcoords)
        fovea_line = np.column_stack((xcoords[mask], ycoords[mask]))
        if fovea_line.size:
            fovea_line = fovea_line[np.argsort(fovea_line[:, 0])]
        else:
            fovea_line = np.array([[1.0, 1.0], [1.1, 1.0]], dtype=np.float64)
        if fovea_line.shape[0] > 0 and np.unique(fovea_line[:, 0]).size == 1:
            fovea_line[0, 0] = fovea_line[0, 0] + 0.1

        rpe_mask = np.asarray(ref, dtype=np.float64).copy()
        rpe_mask[rpe_mask < 255] = np.nan
        rpe_mask[rpe_mask > 255] = np.nan
        rpe_mask[rpe_mask == 255] = 1.0

        xcoords = xs * rpe_mask
        ycoords = ys * rpe_mask
        mask = ~np.isnan(xcoords)
        rpe_line = np.column_stack((xcoords[mask], ycoords[mask]))
        if rpe_line.size:
            rpe_line = rpe_line[np.argsort(rpe_line[:, 0])]
        else:
            rpe_line = np.array([[1.0, 1.0], [1.1, 1.0]], dtype=np.float64)

        rpe_sp = _fit_smooth_spline_like_r(rpe_line[:, 0], rpe_line[:, 1], df=self.df_initial_spline)
        pred_x = np.arange(0.0, ref.shape[0] + 0.02, 0.02)
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

        fovea_coeffs = _fit_line_coeffs(fovea_line)
        compare_fovea_and_rpe = rpe_spline[:, [2, 3, 3, 3]].copy()
        compare_fovea_and_rpe[:, 2] = compare_fovea_and_rpe[:, 0] * fovea_coeffs[1] + fovea_coeffs[0]
        compare_fovea_and_rpe[:, 3] = np.abs(compare_fovea_and_rpe[:, 1] - compare_fovea_and_rpe[:, 2])
        center = int(np.where(compare_fovea_and_rpe[:, 3] == np.min(compare_fovea_and_rpe[:, 3]))[0][0] + 1)
        center_value = float(rpe_spline[center - 1, 0])

        return rpe_spline, fovea_coeffs, center_value
    
    def build_retina_points(self, rpe_spline, center_value):
        """Build the R-style perpendicular sample geometry from main.py."""
        rpe_info = rpe_spline[:, [2, 3, 4, 0]].copy()
        rpe_info[:, 3] = np.round((rpe_info[:, 3] - center_value) * self.pixel_width, 0)
        rpe_info = rpe_info[(rpe_info[:, 3] > -200.9) & (rpe_info[:, 3] < 3000.9)]

        unique_dist = np.unique(rpe_info[:, 3])
        rpe_info_2 = np.column_stack((unique_dist, unique_dist, unique_dist, unique_dist))
        for x in range(rpe_info_2.shape[0]):
            first = np.where(rpe_info[:, 3] == rpe_info_2[x, 3])[0][0]
            rpe_info_2[x, 0:3] = rpe_info[first, 0:3]

        rpe_info_2[:, 2] = (-1.0) / rpe_info_2[:, 2]

        deltas = rpe_info_2[:, [0, 1]].copy()
        deltas[:, 0] = np.cos(np.arctan(rpe_info_2[:, 2]))
        deltas[:, 1] = np.sin(np.arctan(rpe_info_2[:, 2]))
        pixel_move = round(500 / self.pixel_width, 1)
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

        return retina_points
    
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
    
    def compute_alignment_shift(self, flattened_volume, grand_mean, window_width=400):
        """Compute optimal vertical shift to align image to grand mean.
        
        Returns:
            Array of optimal shifts for each perpendicular
        """
        start_move = 200
        end_move = flattened_volume.shape[0] - start_move - 1
        
        shifts = np.full(flattened_volume.shape[0], np.nan)
        
        for x_idx in range(start_move, min(end_move, flattened_volume.shape[0] - window_width), 50):
            # Extract window
            x_start = max(0, x_idx - window_width // 2)
            x_end = min(flattened_volume.shape[0], x_idx + window_width // 2)
            
            profile = np.nanmean(flattened_volume[x_start:x_end, :], axis=0)
            comparison = np.nanmean(grand_mean[x_start:x_end, :], axis=0)
            
            # Try shifts from -10 to +10 microns
            best_corr = -np.inf
            best_shift = 0
            
            for shift in range(-10, 11):
                shift_px = int(np.round(shift * 10 / self.pixel_width))
                
                if shift_px == 0:
                    corr = pearsonr(profile[~np.isnan(profile)], 
                                   comparison[~np.isnan(comparison)])[0]
                elif shift_px > 0:
                    if len(profile[shift_px:]) > 10 and len(comparison[:-shift_px]) > 10:
                        valid = ~(np.isnan(profile[shift_px:]) | np.isnan(comparison[:-shift_px]))
                        if valid.sum() > 10:
                            corr = pearsonr(profile[shift_px:][valid], 
                                          comparison[:-shift_px][valid])[0]
                        else:
                            corr = -np.inf
                    else:
                        corr = -np.inf
                else:
                    shift_px = -shift_px
                    if len(profile[:-shift_px]) > 10 and len(comparison[shift_px:]) > 10:
                        valid = ~(np.isnan(profile[:-shift_px]) | np.isnan(comparison[shift_px:]))
                        if valid.sum() > 10:
                            corr = pearsonr(profile[:-shift_px][valid], 
                                          comparison[shift_px:][valid])[0]
                        else:
                            corr = -np.inf
                    else:
                        corr = -np.inf
                
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_shift = shift
            
            shifts[x_idx] = best_shift
        
        # Interpolate shifts
        valid_shifts = ~np.isnan(shifts)
        if valid_shifts.sum() > 1:
            shifts_interp = interp1d(np.where(valid_shifts)[0], 
                                     shifts[valid_shifts],
                                     kind='linear', 
                                     fill_value='extrapolate')
            shifts = shifts_interp(np.arange(shifts.shape[0]))
        
        return shifts
    
    def apply_shift_to_volume(self, volume, shifts, markers=None):
        """Apply computed shifts to flatten volume.
        
        Returns:
            Shifted volume
        """
        shifted = np.full_like(volume, np.nan)
        
        for x_idx in range(volume.shape[0]):
            if not np.isnan(shifts[x_idx]):
                shift_px = int(np.round(shifts[x_idx] * 10 / self.pixel_width))
                
                if shift_px == 0:
                    shifted[x_idx, :] = volume[x_idx, :]
                elif shift_px > 0:
                    shifted[x_idx, shift_px:] = volume[x_idx, :-shift_px]
                else:
                    shifted[x_idx, :shift_px] = volume[x_idx, -shift_px:]
        
        return shifted
    
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


def _save_profile_plot(profile_xy: np.ndarray, output_path: str, title: str, verticals=(), spline_xy: np.ndarray | None = None) -> None:
    """Save a simple profile plot matching `main.py`'s `save_profile_plot`.

    Args:
        profile_xy: Nx2 array of (x, mean-intensity)
        output_path: target PNG path
        title: plot title
        verticals: tuple of x positions to draw vertical lines
        spline_xy: optional Nx2 array to plot spline markers
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile_xy = np.asarray(profile_xy)
    if profile_xy.size == 0 or profile_xy.shape[0] == 0:
        return

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


class Step3Frame(ttk.Frame):
    """Step 3 tab UI that runs this module's flattening pipeline inside the app."""
    SIDEBAR_WIDTH = 280
    SIDEBAR_TEXT_WRAP = 250

    def __init__(self, parent, preferences=None, source_step=None):
        super().__init__(parent)
        self.preferences = preferences
        self.source_step = source_step

        self.current_sdb_dir = None
        self.processor = None
        self.results = None
        self.figure = None
        self.canvas = None
        self.more_outputs_button = None
        self._busy = False
        self.last_diff_log_dir = None

        self.slice_var = tk.StringVar(value="0")
        self.view_var = tk.StringVar(value="comparison")
        self.status_var = tk.StringVar(value="Ready — load Step 2 MARKED files or choose folder.")
        self.info_var = tk.StringVar(value="")
        self.progress_text_var = tk.StringVar(value="Idle")

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.configure(width=self.SIDEBAR_WIDTH)
        left.pack_propagate(False)

        ttk.Button(left, text="Use Step 2 Output", command=self._use_step2_output).pack(fill="x", pady=2)
        ttk.Button(left, text="Browse Output Folder", command=self._browse_output_folder).pack(fill="x", pady=2)
        ttk.Button(left, text="Load MARKED + RAW (Choose Folder)", command=self._load_processor).pack(fill="x", pady=2)

        self.dir_var = tk.StringVar(value="(no folder selected)")
        ttk.Label(
            left,
            textvariable=self.dir_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            foreground="gray",
            justify="left",
        ).pack(fill="x", pady=(2, 8))

        ttk.Button(left, text="Run Step 3", command=self._run_processing).pack(fill="x", pady=2)
        self.more_outputs_button = ttk.Button(
            left,
            text="More Process",
            command=self._run_more_outputs,
            state="disabled",
        )
        self.more_outputs_button.pack(fill="x", pady=2)

        self.progress = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=2)
        ttk.Label(
            left,
            textvariable=self.progress_text_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            foreground="gray",
            justify="left",
        ).pack(fill="x", pady=(0, 4))

        ttk.Label(left, text="View").pack(anchor="w", pady=(8, 2))
        view_combo = ttk.Combobox(
            left,
            textvariable=self.view_var,
            values=["comparison", "grand_mean"],
            state="readonly",
        )
        view_combo.pack(fill="x", pady=2)
        view_combo.bind("<<ComboboxSelected>>", lambda _: self._render())

        ttk.Label(left, text="Slice").pack(anchor="w", pady=(8, 2))
        slice_combo = ttk.Combobox(left, textvariable=self.slice_var, values=["0", "1"], state="readonly")
        slice_combo.pack(fill="x", pady=2)
        slice_combo.bind("<<ComboboxSelected>>", lambda _: self._render())

        ttk.Button(left, text="Export Flattened", command=self._export_results).pack(fill="x", pady=(10, 2))

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(
            left,
            textvariable=self.info_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            justify="left",
        ).pack(fill="x")

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.plot_holder = ttk.Frame(right)
        self.plot_holder.pack(fill="both", expand=True)

        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", padding=3).pack(
            side="bottom", fill="x"
        )

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

    def _save_step2_marked_outputs(self):
        if self.source_step is None or not hasattr(self.source_step, "_save_marked_images"):
            return
        saved = self.source_step._save_marked_images(require_complete=True)
        if not saved:
            raise RuntimeError("Step 2 MARKED outputs were not saved. Complete all Step 2 boundaries first.")

    def _use_step2_output(self):
        if self.source_step is None or not hasattr(self.source_step, "_marked_output_basepath"):
            messagebox.showwarning("Step 2 Output", "Step 2 output is not available. Use Browse Output Folder.")
            return

        try:
            self._save_step2_marked_outputs()
            dark_base = self.source_step._marked_output_basepath("Dark_MARKED")
            self.current_sdb_dir = os.path.dirname(dark_base)
            self.dir_var.set(self.current_sdb_dir)
            self.status_var.set("Using Step 2 output folder. MARKED files saved.")
        except Exception as exc:
            messagebox.showerror("Step 2 Output", f"Could not read Step 2 output path.\n{exc}")

    def _browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing MARKED and RAW Analyze files")
        if folder:
            self.current_sdb_dir = folder
            self.dir_var.set(folder)
            self.status_var.set("Output folder selected.")

    def _load_processor(self):
        selected_folder = filedialog.askdirectory(
            title="Select folder containing MARKED and RAW Analyze files",
            initialdir=self.current_sdb_dir or None,
        )
        if not selected_folder:
            return

        self.current_sdb_dir = selected_folder
        self.dir_var.set(selected_folder)
        self.status_var.set("Loading selected folder...")
        if self.more_outputs_button is not None:
            self.more_outputs_button.configure(state="disabled")

        dark_marked = self._existing_basepath(self.current_sdb_dir, ["Dark_MARKED", "DARK_MARKED"])
        light_marked = self._existing_basepath(self.current_sdb_dir, ["Light_MARKED", "LIGHT_MARKED"])
        dark_raw = self._existing_basepath(self.current_sdb_dir, ["DARK", "Dark"])
        light_raw = self._existing_basepath(self.current_sdb_dir, ["LIGHT", "Light"])

        missing = []
        if dark_marked is None:
            missing.append("Dark_MARKED.hdr/.img")
        if light_marked is None:
            missing.append("Light_MARKED.hdr/.img")
        if dark_raw is None:
            missing.append("DARK.hdr/.img")
        if light_raw is None:
            missing.append("LIGHT.hdr/.img")

        if missing:
            messagebox.showerror(
                "Load",
                "Cannot load Step 3 from this folder because these files are missing:\n"
                + "\n".join(f"  • {item}" for item in missing)
                + "\n\nStep 3 requires both marked references and both raw Analyze volumes.",
            )
            return

        input_paths = {
            "Dark_MARKED": dark_marked,
            "Light_MARKED": light_marked,
            "DARK": dark_raw,
            "LIGHT": light_raw,
        }
        try:
            input_info = self._read_input_stack_info(input_paths)
        except Exception as exc:
            messagebox.showerror("Load", f"Cannot load Step 3 inputs.\n{exc}")
            self.status_var.set("Step 3 input dimensions do not match.")
            return

        self.processor = OCTFlatteningProcessor(
            reference_dark_path=dark_marked,
            reference_light_path=light_marked,
            dark_path=dark_raw,
            light_path=light_raw,
            image_index_dark=[0, 1],
            image_index_light=[0, 1],
            pixel_width=3.89,
        )

        self.info_var.set(
            f"{self._format_stack_info('Dark_MARKED ',input_info['Dark_MARKED'])}\n"
            f"{self._format_stack_info('Light_MARKED ', input_info['Light_MARKED'])}\n"
            f"{self._format_stack_info('DARK ', input_info['DARK'])}\n"
            f"{self._format_stack_info('LIGHT ', input_info['LIGHT'])}\n")
        self.status_var.set("Loaded Step 3 inputs with bit depth info. Ready to run.")

    def _run_processing(self):
        if self.processor is None:
            messagebox.showwarning("Run", "Load Step 3 inputs first.")
            return
        if self._busy:
            return

        self._busy = True
        if self.more_outputs_button is not None:
            self.more_outputs_button.configure(state="disabled")
        self.progress.configure(value=0)
        self.progress_text_var.set("Starting...")
        self.status_var.set("Running Step 3 pipeline...")
        threading.Thread(target=self._process_worker, daemon=True).start()

    def _threadsafe_progress(self, percent, label):
        self.after(0, lambda: self._update_progress(percent, label))

    def _update_progress(self, percent, label):
        self.progress.configure(value=max(0, min(100, float(percent))))
        self.progress_text_var.set(label)
        self.status_var.set(f"Step 3: {label}")

    def _process_worker(self):
        diff_logger = None
        output_dir = self.current_sdb_dir
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
            self.after(0, lambda: self._on_processing_done(results, None, summary_path, None, save_error, output_dir))
        except Exception as exc:
            self.after(0, lambda: self._on_processing_done(None, exc, None, None, None, output_dir))

    def _on_processing_done(self, results, error, summary_path=None, diff_logger=None, save_error=None, save_dir=None):
        self._busy = False

        if error is not None:
            self.status_var.set(f"Step 3 failed: {error}")
            self.progress_text_var.set("Failed")
            if self.more_outputs_button is not None:
                self.more_outputs_button.configure(state="disabled")
            messagebox.showerror("Step 3", f"Processing failed.\n{error}")
            return

        self.results = results
        if self.more_outputs_button is not None:
            self.more_outputs_button.configure(state="normal")
        if summary_path is not None:
            self.last_diff_log_dir = str(Path(summary_path).parent)
        self.progress.configure(value=100)
        self.progress_text_var.set("Completed")
        if save_error is not None:
            self.status_var.set(f"Step 3 complete, but auto-save failed: {save_error}")
        elif summary_path is not None:
            msg = "Step 3 complete. Diff log saved."
            if diff_logger is not None and diff_logger.first_divergence is not None:
                first = diff_logger.first_divergence
                msg = (
                    f"Step 3 complete. First divergence: {first['stage']}::{first['array']} "
                    f"(see {summary_path})."
                )
            self.status_var.set(msg)
        else:
            self.status_var.set(f"Step 3 complete. Outputs saved to {save_dir or self.current_sdb_dir}")
        self._render()

    def _render(self):
        if self.results is None:
            return

        if Figure is None or FigureCanvasTkAgg is None:
            self.status_var.set("Matplotlib is unavailable in this environment; preview rendering is disabled.")
            return

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        view = self.view_var.get()
        slice_idx = int(self.slice_var.get())
        max_slice = self.results['flattened_dark'].shape[0] - 1
        slice_idx = max(0, min(slice_idx, max_slice))

        fig = Figure(figsize=(11, 7), dpi=100)

        if view == "grand_mean":
            # Row layout: profile on top, image on bottom
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            gm = self.results['final_grand_mean']

            # Top: intensity profile across columns
            profile = np.nanmean(gm, axis=0)
            xs = np.arange(1, profile.size + 1, dtype=np.float64)
            ax1.plot(xs, profile, color="black", linewidth=1.0)
            ax1.axvline(self.results['vertex'], color="r", linestyle="--", alpha=0.7)
            ax1.set_title("Intensity Profile")
            ax1.set_xlabel("column")
            ax1.set_ylabel("mean intensity")
            ax1.grid(True, alpha=0.3)

            # Bottom: grand mean image displayed horizontally (columns left->right)
            ax2.imshow(gm.T, cmap="gray", aspect="auto")
            # Draw horizontal line across the image at the detected vertex (row corresponding to column)
            ax2.axhline(self.results['vertex'], color="r", linestyle="--", alpha=0.7)
            ax2.set_title("Final Grand Mean")
            ax2.axis("off")
        else:
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            # Original volumes are (y, x, slice); rotate 90° counter-clockwise for preview
            ax1.imshow(np.rot90(self.processor.dark[:, :, slice_idx], k=1), cmap="gray", aspect="auto")
            ax1.set_title("Original DARK")
            ax1.axis("off")
            ax2.imshow(self.results['flattened_dark'][slice_idx].T, cmap="gray", aspect="auto")
            ax2.set_title("Flattened DARK")
            ax2.axis("off")
            ax3.imshow(np.rot90(self.processor.light[:, :, slice_idx], k=1), cmap="gray", aspect="auto")
            ax3.set_title("Original LIGHT")
            ax3.axis("off")
            ax4.imshow(self.results['flattened_light'][slice_idx].T, cmap="gray", aspect="auto")
            ax4.set_title("Flattened LIGHT")
            ax4.axis("off")

        fig.tight_layout()
        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_holder)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.info_var.set(
            f"flattened_dark: {self.results['flattened_dark'].shape}\n"
            f"flattened_light: {self.results['flattened_light'].shape}\n"
            f"final_grand_mean: {self.results['final_grand_mean'].shape}\n"
            f"vertex: {self.results['vertex']}"
        )

    def _export_results(self):
        if self.results is None:
            messagebox.showwarning("Export", "Run Step 3 first.")
            return

        out_dir = filedialog.askdirectory(title="Select export folder", initialdir=self.current_sdb_dir or None)
        if not out_dir:
            return

        try:
            # Convert UI/result arrays from (slice, x, y) to Analyze stack order
            # (slice, y, x). The Analyze writer already applies the repo's file
            # orientation convention, so we do not add an extra flip here.
            dark_export = self._prepare_export_volume(self.results['final_dark'])
            light_export = self._prepare_export_volume(self.results['final_light'])
            write_analyze(os.path.join(out_dir, "_flat_DARK"), dark_export)
            write_analyze(os.path.join(out_dir, "_flat_LIGHT"), light_export)
            _save_main_style_exports(self.results, out_dir)
            self.status_var.set("Flattened images exported.")
            messagebox.showinfo("Export", "Exported _flat_DARK and _flat_LIGHT.")
        except Exception as exc:
            messagebox.showerror("Export", f"Export failed.\n{exc}")

    def _run_more_outputs(self):
        if self.results is None:
            messagebox.showwarning("More Process", "Run Step 3 first.")
            return
        if not self.current_sdb_dir:
            messagebox.showwarning("More Process", "Choose an output folder first.")
            return

        try:
            output_dir = Path(self.current_sdb_dir)
            flat_npz = output_dir / "DARK__and__LIGHT__flat.npz"
            done_npz = output_dir / "_done_DARK__and__LIGHT.npz"
            if not flat_npz.exists() or not done_npz.exists():
                self.status_var.set("Saving Step 3 checkpoints before more process...")
                save_error = self._save_generated_outputs(
                    results=self.results,
                    output_dir=output_dir,
                    progress_cb=None,
                    ui_updates=False,
                )
                if save_error is not None:
                    raise RuntimeError(save_error)

            outputs = _main_run_more_outputs_from_step3_npz(flat_npz, done_npz, output_dir)
            self.status_var.set(f"More process complete. R-format outputs saved to {output_dir}")
            self.info_var.set(
                self.info_var.get()
                + "\n\nMore process outputs:\n"
                + "\n".join(str(path.name) for path in outputs.values())
            )
            messagebox.showinfo(
                "More Process",
                "Generated tissue-border plots and R-format thickness tables.",
            )
        except Exception as exc:
            self.status_var.set(f"More process failed: {exc}")
            messagebox.showerror("More Process", f"More process failed.\n{exc}")

    @staticmethod
    def _prepare_export_volume(volume):
        return np.transpose(np.nan_to_num(volume, nan=0.0), (0, 2, 1)).astype(np.float32)

    def _save_generated_outputs(self, results=None, output_dir=None, progress_cb=None, ui_updates=True):
        results = self.results if results is None else results
        output_dir = self.current_sdb_dir if output_dir is None else output_dir
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(results['first_grand_mean'], cmap='gray', aspect='auto')
    axes[0, 0].set_title('First Grand Mean')
    axes[0, 1].imshow(results['second_grand_mean'], cmap='gray', aspect='auto')
    axes[0, 1].set_title('Second Grand Mean')
    axes[1, 0].imshow(results['final_grand_mean'], cmap='gray', aspect='auto')
    axes[1, 0].axhline(results['vertex'], color='r', linestyle='--', alpha=0.6)
    axes[1, 0].set_title('Final Grand Mean')
    axes[1, 1].plot(results['grand_profile'][:, 0], results['grand_profile'][:, 1])
    axes[1, 1].axvline(results['vertex'], color='r', linestyle='--', alpha=0.6)
    axes[1, 1].set_title('Final Grand Profile')
    plt.tight_layout()
    plt.savefig('oct_flattening_results.png', dpi=150, bbox_inches='tight')
    print("✓ Plots saved to 'oct_flattening_results.png'")
    return results


if __name__ == "__main__":
    main()
