"""Step 4 - Analyze ISez.

Python translation of ``Rotated_Rescaled_Human_ISez6_8bit_JS.m`` with a
Tkinter/Matplotlib canvas that replaces MATLAB's ``ginput`` profile clicks.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from aidas.utils.io_utils import read_analyze, read_tiff
from aidas.utils.ui_utils import SidebarStepFrame


MATLAB_ROI_LOW = 300
MATLAB_ROI_HIGH = 450
MATLAB_ROI_TOP_LINE = 450
MATLAB_A_LIMIT = 1.0


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
    roi_image_path: Path
    profile_path: Path
    isez_path: Path


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
    """Apply the four MATLAB while-loop adjustments to the selected bounds."""

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
    """Rotate the IS band baseline and rescale the surrounding region to 0-100."""

    values = np.asarray(profile, dtype=np.float64)
    n_points = values.size
    s = _clamp_profile_index(start, n_points)
    e = _clamp_profile_index(end, n_points)
    if s > e:
        s, e = e, s
    if e <= s:
        raise ValueError("ISez end must be greater than start.")

    is_region = values[s - 1:e]
    x_rotation_point = (e + s) / 2.0
    y_rotation_point = (values[s - 1] + values[e - 1]) / 2.0
    adjacent_length = float(e - s)
    hypotenuse_length = float(np.sqrt(adjacent_length**2 + (is_region[-1] - is_region[0]) ** 2))
    if hypotenuse_length <= 0:
        theta_radians = 0.0
    else:
        ratio = max(-1.0, min(1.0, adjacent_length / hypotenuse_length))
        theta_radians = float(np.arccos(ratio))

    cos_t = np.cos(-theta_radians)
    sin_t = np.sin(-theta_radians)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

    x_points = np.linspace(s, e, is_region.size)
    shifted = np.vstack((x_points - x_rotation_point, is_region - y_rotation_point))
    rotated = rotation @ shifted
    new_y = rotated[1, :] + y_rotation_point
    max_index_zero = int(np.nanargmax(new_y))

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


def save_profile_plot(
    profile: np.ndarray,
    start_click: float,
    end_click: float,
    output_path: Path,
) -> None:
    """Save the MATLAB-style raw profile plot with the user's click positions."""

    values = np.asarray(profile, dtype=np.float64)
    xs = np.arange(1, values.size + 1, dtype=np.float64)
    ymin = float(np.nanmin(values))

    fig = Figure(figsize=(10.85, 3.28), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(xs, values, color="black", linewidth=1.2)
    ax.axvline(start_click, color="#1f77b4", linewidth=1.0)
    ax.axvline(end_click, color="#d62728", linewidth=1.0)
    ax.set_xlim(0, 140)
    ax.set_ylim(ymin, ymin + 80)
    fig.tight_layout()
    fig.savefig(output_path)


def save_isez_plot(result_data: dict[str, np.ndarray | float | int], center: float, output_path: Path) -> None:
    """Save the MATLAB-style rotated/rescaled ISez plot."""

    fig = Figure(figsize=(6.2, 3.28), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(result_data["x"], result_data["y"], color="black", linewidth=1.2)
    ax.plot(result_data["baseline_x"], result_data["baseline_y"], color="black", linewidth=1.0)
    ax.set_xlim(center - 40, center + 40)
    ax.set_ylim(-20, 120)
    fig.tight_layout()
    fig.savefig(output_path)


def analyze_and_save_roi(
    image: np.ndarray,
    roi: ISezROI,
    *,
    start_click: float,
    end_click: float,
    source_stem: str,
    output_dir: str | os.PathLike,
) -> ISezResult:
    """Run the MATLAB ISez analysis for one ROI and save the three outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

    roi_image_path = output_path / f"{source_stem}_ROI_{roi.suffix}.jpg"
    profile_path = output_path / f"{source_stem}_Profile_{roi.suffix}.png"
    isez_path = output_path / f"{source_stem}_ISez_{roi.suffix}.png"

    Image.fromarray(roi_overlay_image(image, roi)).save(roi_image_path, quality=95)
    save_profile_plot(profile, start_click, end_click, profile_path)
    save_isez_plot(rescaled, center, isez_path)

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
        roi_image_path=roi_image_path,
        profile_path=profile_path,
        isez_path=isez_path,
    )


def write_summary_csv(results: list[ISezResult], path: str | os.PathLike) -> None:
    """Write one compact summary row per completed ROI."""

    rows = []
    for result in results:
        rows.append(
            {
                "suffix": result.roi.suffix,
                "left": result.roi.left,
                "right": result.roi.right,
                "low": result.roi.low,
                "high": result.roi.high,
                "start": result.start,
                "end": result.end,
                "adjusted_start": result.adjusted_start,
                "adjusted_end": result.adjusted_end,
                "center": result.center,
                "slope": result.slope,
                "max_index": result.max_index,
                "min_intensity": result.min_intensity,
                "max_intensity": result.max_intensity,
                "roi_image": str(result.roi_image_path),
                "profile_image": str(result.profile_path),
                "isez_image": str(result.isez_path),
            }
        )

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["suffix"])
        writer.writeheader()
        writer.writerows(rows)


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
        self.profile_clicks: list[float] = []
        self.figure = None
        self.canvas = None
        self.ax_image = None
        self.ax_profile = None
        self.ax_isez = None
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

        source_section = self.add_sidebar_section("Input", padding=3, pady=(0, 5))
        source = source_section.body
        ttk.Button(source, text="Open OCT Image...", command=self._open_image).pack(fill="x", pady=2)
        ttk.Button(source, text="Load Step 3 _flat_LIGHT", command=self._load_step3_flat_light).pack(fill="x", pady=2)

        ttk.Label(source, text="Step 3 folder").pack(anchor="w", pady=(6, 0))
        folder_row = ttk.Frame(source)
        folder_row.pack(fill="x")
        ttk.Entry(folder_row, textvariable=self.input_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(folder_row, text="...", width=3, command=self._browse_input_folder).pack(side="right", padx=(2, 0))

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

        output_section = self.add_sidebar_section("Output", padding=3, pady=(0, 5))
        output = output_section.body
        out_row = ttk.Frame(output)
        out_row.pack(fill="x")
        ttk.Entry(out_row, textvariable=self.output_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="...", width=3, command=self._browse_output_folder).pack(side="right", padx=(2, 0))
        ttk.Button(output, text="Build Stacks", command=self._build_stack_outputs).pack(fill="x", pady=(6, 2))

        roi_section = self.add_sidebar_section("ROIs", padding=3, fill="both", expand=True, pady=(0, 5))
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

        start_row = ttk.Frame(roi_box)
        start_row.pack(fill="x", pady=1)
        ttk.Label(start_row, text="Start").pack(side="left")
        ttk.Entry(start_row, textvariable=self.start_var, width=8).pack(side="right")
        end_row = ttk.Frame(roi_box)
        end_row.pack(fill="x", pady=1)
        ttk.Label(end_row, text="End").pack(side="left")
        ttk.Entry(end_row, textvariable=self.end_var, width=8).pack(side="right")
        action_row = ttk.Frame(roi_box)
        action_row.pack(fill="x", pady=(6, 0))
        ttk.Button(action_row, text="Apply", command=self._apply_entry_clicks).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(action_row, text="Clear", command=self._clear_clicks).pack(side="right", expand=True, fill="x", padx=(2, 0))
        ttk.Label(
            roi_box,
            textvariable=self.profile_status_var,
            wraplength=self.SIDEBAR_TEXT_WRAP,
            foreground="gray",
            justify="left",
        ).pack(fill="x", pady=(6, 0))

        # stats_section = self.add_sidebar_section("Stats", padding=3, pady=(0, 5))
        # stats = stats_section.body
        # ttk.Label(stats, textvariable=self.stats_var, wraplength=self.SIDEBAR_TEXT_WRAP, justify="left").pack(fill="x")

        self.plot_holder = ttk.Frame(self.content)
        self.plot_holder.pack(fill="both", expand=True)

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

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select flattened OCT image for Step 4",
            initialdir=self.input_dir_var.get() or None,
            filetypes=[
                ("Supported images", "*.tif *.tiff *.hdr *.img *.png *.jpg *.jpeg"),
                ("TIFF", "*.tif *.tiff"),
                ("Analyze 7.5", "*.hdr *.img"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_path(path)

    def _browse_input_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Select folder containing Step 3 _flat_LIGHT output",
            initialdir=self.input_dir_var.get() or None,
        )
        if folder:
            self.input_dir_var.set(folder)
            self._input_dir_user_selected = True

    def _browse_output_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Select folder for Step 4 ISez outputs",
            initialdir=self.output_dir_var.get() or None,
        )
        if folder:
            self.output_dir_var.set(folder)
            self._output_dir_user_selected = True

    def _load_step3_flat_light(self) -> None:
        if not self._input_dir_user_selected:
            self.input_dir_var.set(self._default_input_folder())
        folder = Path(self.input_dir_var.get() or self._default_input_folder())
        candidates = [
            folder / "_flat_LIGHT.hdr",
            folder / "_flat_LIGHT.img",
            folder / "_flat_LIGHT.tif",
            folder / "_flat_LIGHT.tiff",
        ]
        for candidate in candidates:
            if candidate.is_file():
                self._load_path(candidate)
                return
        messagebox.showerror(
            "Load Step 3 Output",
            f"Could not find _flat_LIGHT in:\n{folder}\n\nExpected _flat_LIGHT.hdr/.img or _flat_LIGHT.tif.",
        )

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
        self._set_current_slice()
        self.completed.clear()
        self.current_roi_idx = 0
        self.profile_clicks.clear()
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
        self.start_var.set("")
        self.end_var.set("")
        self.completed.clear()
        self._refresh_roi_list()
        self._render_current_roi()

    def _on_roi_selected(self, _event=None) -> None:
        selection = self.roi_listbox.curselection()
        if not selection:
            return
        self.current_roi_idx = int(selection[0])
        self.profile_clicks.clear()
        self.start_var.set("")
        self.end_var.set("")
        self._render_current_roi()

    def _move_roi(self, delta: int) -> None:
        self.current_roi_idx = max(0, min(len(self.rois) - 1, self.current_roi_idx + delta))
        self.profile_clicks.clear()
        self.start_var.set("")
        self.end_var.set("")
        self._select_roi_in_list()
        self._render_current_roi()

    def _select_roi_in_list(self) -> None:
        self.roi_listbox.selection_clear(0, tk.END)
        self.roi_listbox.selection_set(self.current_roi_idx)
        self.roi_listbox.see(self.current_roi_idx)

    def _refresh_roi_list(self) -> None:
        self.roi_listbox.delete(0, tk.END)
        for roi in self.rois:
            mark = "done" if roi.suffix in self.completed else "open"
            self.roi_listbox.insert(tk.END, f"{roi.suffix:>5}   {mark}")

    def _render_empty_canvas(self) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.figure = Figure(figsize=(10, 7), dpi=100)
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Load a flattened OCT image", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_holder)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _render_current_roi(self) -> None:
        if self.image is None:
            self._render_empty_canvas()
            return

        roi = self.rois[self.current_roi_idx]
        try:
            profile = intensity_profile(self.image, roi)
            bounds = roi_bounds_for_image(roi, self.image.shape)
        except Exception as exc:
            self.status_var.set(str(exc))
            return

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.figure = Figure(figsize=(11, 7), dpi=100)
        grid = self.figure.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
        self.ax_image = self.figure.add_subplot(grid[0, :])
        self.ax_profile = self.figure.add_subplot(grid[1, 0])
        self.ax_isez = self.figure.add_subplot(grid[1, 1])

        display = to_uint8_display(self.image)
        left, right, low, high = bounds
        x_pad = max(30, (right - left) * 2)
        y_pad = 40
        x0 = max(0, left - x_pad)
        x1 = min(display.shape[1], right + x_pad)
        y0 = max(0, low - y_pad)
        y1 = min(display.shape[0], high + y_pad)
        crop = display[y0:y1, x0:x1]

        self.ax_image.imshow(crop, cmap="gray", aspect="auto", extent=(x0 + 1, x1, y1, y0 + 1))
        rect = Rectangle(
            (left + 1, low + 1),
            max(1, right - left),
            max(1, high - low),
            fill=False,
            edgecolor="yellow",
            linewidth=1.5,
            linestyle="--",
        )
        self.ax_image.add_patch(rect)
        self.ax_image.set_title(f"ROI {roi.suffix}: x={roi.left}:{roi.right}, y={roi.low}:{roi.high}")
        self.ax_image.set_xlabel("image x")
        self.ax_image.set_ylabel("image y")

        xs = np.arange(1, profile.size + 1, dtype=np.float64)
        ymin = float(np.nanmin(profile))
        self.ax_profile.plot(xs, profile, color="black", linewidth=1.2)
        self.ax_profile.set_xlim(0, 140)
        self.ax_profile.set_ylim(ymin, ymin + 80)
        self.ax_profile.set_title("Click start, then end")
        self.ax_profile.set_xlabel("profile row")
        self.ax_profile.set_ylabel("mean intensity / 3")
        for idx, click in enumerate(self.profile_clicks[:2]):
            color = "#1f77b4" if idx == 0 else "#d62728"
            self.ax_profile.axvline(click, color=color, linewidth=1.2)

        self._draw_isez_preview(profile)

        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_holder)
        self.canvas.mpl_connect("button_press_event", self._on_profile_click)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._update_profile_status(profile)

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

        self._render_current_roi()
        if len(self.profile_clicks) == 2: # and self.auto_advance_var.get():
            self.after(50, lambda: self._save_current_roi(auto_advance=True))

    def _current_profile_size(self) -> int:
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
        self._render_current_roi()
        self._save_current_roi(auto_advance=True)

    def _clear_clicks(self) -> None:
        self.profile_clicks.clear()
        self.start_var.set("")
        self.end_var.set("")
        self._render_current_roi()

    def _update_profile_status(self, profile: np.ndarray) -> None:
        roi = self.rois[self.current_roi_idx]
        if len(self.profile_clicks) < 2:
            self.profile_status_var.set(f"ROI {roi.suffix}: click start and end on the profile.")
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
            f"ROI {roi.suffix}: selected {start}-{end}; adjusted {adj_start}-{adj_end}."
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
        outdir = self.output_dir_var.get() or (str(self.current_path.parent) if self.current_path else os.getcwd())
        try:
            result = analyze_and_save_roi(
                self.image,
                roi,
                start_click=self.profile_clicks[0],
                end_click=self.profile_clicks[1],
                source_stem=self.current_stem,
                output_dir=outdir,
            )
            self.completed[roi.suffix] = result
            self._write_current_summary()
        except Exception as exc:
            messagebox.showerror("Step 4", f"Could not save ROI {roi.suffix}.\n{exc}")
            return

        self._refresh_roi_list()
        self.status_var.set(f"Saved ROI {roi.suffix}: {result.isez_path.name}")

        if auto_advance and self.current_roi_idx < len(self.rois) - 1:
            self.current_roi_idx += 1
            self.profile_clicks.clear()
            self.start_var.set("")
            self.end_var.set("")
            self._select_roi_in_list()
            self._render_current_roi()
        else:
            self._select_roi_in_list()

    def _write_current_summary(self) -> None:
        if not self.completed:
            return
        outdir = Path(self.output_dir_var.get() or ".")
        ordered = [self.completed[roi.suffix] for roi in self.rois if roi.suffix in self.completed]
        write_summary_csv(ordered, outdir / f"{self.current_stem}_ISez_summary.csv")

    def _build_stack_outputs(self) -> None:
        if not self.completed:
            messagebox.showwarning("Step 4", "Save at least one ROI before building stack outputs.")
            return

        ordered = [self.completed[roi.suffix] for roi in self.rois if roi.suffix in self.completed]
        outdir = Path(self.output_dir_var.get() or ".")
        try:
            isez_images = [Image.open(result.isez_path).convert("RGB") for result in ordered if result.isez_path.is_file()]
            if isez_images:
                stack_path = outdir / f"{self.current_stem}_ISez_stack.tif"
                isez_images[0].save(stack_path, save_all=True, append_images=isez_images[1:])

            roi_arrays = [
                np.asarray(Image.open(result.roi_image_path).convert("L"), dtype=np.uint8)
                for result in ordered
                if result.roi_image_path.is_file()
            ]
            if roi_arrays:
                roi_images = [Image.fromarray(arr) for arr in roi_arrays]
                roi_stack_path = outdir / f"{self.current_stem}_ROI_stack.tif"
                roi_images[0].save(roi_stack_path, save_all=True, append_images=roi_images[1:])
                max_projection = np.maximum.reduce(roi_arrays)
                Image.fromarray(max_projection).save(outdir / f"{self.current_stem}_ROI_max_projection.tif")
        except Exception as exc:
            messagebox.showerror("Step 4", f"Could not build stack outputs.\n{exc}")
            return

        self.status_var.set(f"Built stack outputs in {outdir}.")
        messagebox.showinfo("Step 4", "Built ISez stack, ROI stack, and ROI max projection.")


def main() -> None:
    root = tk.Tk()
    root.title("AIDaS Step 4 - Analyze ISez")
    root.geometry("1200x800")
    frame = Step4Frame(root)
    frame.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
