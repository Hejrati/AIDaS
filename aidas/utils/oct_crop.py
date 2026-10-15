"""Suggest editable OCT crop rectangles without changing image samples.

Matched ILM/BM contours give the retinal envelope. Without contours, intensity
changes in a middle stripe locate a compact layer bundle; nearby transitions
then track its curved edges across the scan. This module only analyzes detached
arrays; the manual import/display/save pipeline owns the pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter, label, median_filter


@dataclass(frozen=True)
class CropSuggestion:
    roi: tuple[int, int, int, int]
    method: str
    confident: bool


def _side_bounds(image: np.ndarray) -> tuple[int, int]:
    """Remove only constant background columns contiguous with the edges.

    Missing layer annotations and shadowed tissue do not establish an empty
    margin. Requiring constant background is intentionally conservative.
    """
    height, width = image.shape
    lows = np.min(image, axis=0)
    highs = np.max(image, axis=0)
    background = float(np.min(lows))
    blank = (lows == highs) & (lows == background)
    nonblank = np.flatnonzero(~blank)
    if not nonblank.size:
        return 0, width
    margin = max(2, int(math.ceil(width * 0.01)))
    left = max(0, int(nonblank[0]) - margin)
    right = min(width, int(nonblank[-1]) + margin + 1)
    return left, right


def _contour_envelope(contours, height: int, width: int):
    if contours is None:
        return None
    try:
        curves = np.array(contours, dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        return None
    if curves.ndim != 2 or curves.shape[0] < 2 or curves.shape[1] != width:
        return None
    valid = np.isfinite(curves) & (curves >= 0) & (curves <= height - 1)
    # The reader supplies ILM first and BM second. Both must span the scan;
    # isolated annotations are unsuitable for choosing a retinal crop.
    if np.any(np.mean(valid[:2], axis=1) < 0.80):
        return None
    joint = valid[0] & valid[1]
    if np.mean(joint) < 0.75:
        return None
    thickness = curves[1, joint] - curves[0, joint]
    if np.any(thickness <= 0) or np.median(thickness) < 3:
        return None
    for curve, mask in zip(curves[:2], valid[:2]):
        adjacent = mask[1:] & mask[:-1]
        if np.any(np.abs(np.diff(curve)[adjacent]) > max(16, height * 0.08)):
            return None
    # Include every valid point, including curved ends and local elevations.
    samples = curves[valid]
    padding = max(12, int(math.ceil(height * 0.04)),
                  int(math.ceil(float(np.median(thickness)) * 0.15)))
    return max(0, int(math.floor(samples.min())) - padding), min(
        height, int(math.ceil(samples.max())) + padding + 1
    )


def _block_medians(image: np.ndarray, block: int) -> np.ndarray:
    """Reduce speckle for analysis, including zero-filled CMDb background."""
    height, width = image.shape
    rows = math.ceil(height / block)
    columns = math.ceil(width / block)
    padded = np.full((rows * block, columns * block), np.nan, dtype=np.float32)
    padded[:height, :width] = image
    cells = padded.reshape(rows, block, columns, block).transpose(0, 2, 1, 3)
    cells = cells.reshape(rows, columns, block * block)
    # Zero-valued background mixed with unsigned tissue words otherwise
    # creates artificial large edges. Ignore sparse zeros only on this copy.
    nonzero = np.isfinite(cells) & (cells != 0)
    count = np.sum(nonzero, axis=2)
    masked = np.where(nonzero, cells, np.nan)
    # All-empty cells have a known zero background; supply a finite value to
    # avoid nanmedian's all-NaN warning without changing nonempty cells.
    masked[:, :, 0] = np.where(count == 0, 0, masked[:, :, 0])
    medians = np.nanmedian(masked, axis=2)
    medians[count < max(2, block * block // 8)] = 0
    return medians


def _has_interior_contrast(profile: np.ndarray, upper: int, lower: int) -> bool:
    """Require sustained tissue contrast with background on both sides.

    Opposite edge signs alone also describe the empty gap between a reflection
    and the retina. Sample shoulders well outside the candidate's edge so a
    narrow reflection at that edge cannot masquerade as its background.
    """
    span = lower - upper
    if span < 4:
        return False
    inside = profile[upper + span // 4:lower - span // 4 + 1]
    before = profile[max(0, upper - 3 * span // 4):max(0, upper - span // 4)]
    after = profile[min(len(profile), lower + span // 4):
                    min(len(profile), lower + 3 * span // 4 + 1)]
    if min(len(inside), len(before), len(after)) < 3:
        return False
    level = float(np.median(inside))
    before_level = float(np.median(before))
    after_level = float(np.median(after))
    before_delta = level - before_level
    after_delta = level - after_level
    noise = float(np.median(np.abs(np.diff(profile))))
    scale = float(np.percentile(profile, 95) - np.percentile(profile, 5))
    derivative = np.gradient(profile)
    weaker_edge = min(abs(float(derivative[upper])), abs(float(derivative[lower])))
    tolerance = max(noise * 3.0, scale * 0.01, weaker_edge * 0.30,
                    np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(profile)))))
    if before_delta * after_delta <= 0 or min(abs(before_delta), abs(after_delta)) <= tolerance:
        return False
    sign = 1 if before_delta > 0 else -1
    sustained = (sign * (inside - before_level) > tolerance) & (
        sign * (inside - after_level) > tolerance
    )
    # A reflection tail can dominate a few rows at the start of an empty
    # interval. Most of the core must carry contrast, rather than a bare
    # majority supplied by that tail.
    return float(np.mean(sustained)) >= 0.75


def _center_seed(smoothed: np.ndarray, block: int, height: int):
    """Locate a compact group of layer transitions in middle scan stripes."""
    # Signal processing is only needed when proposing an image-based crop.
    # Keep its substantial import cost out of application startup.
    from scipy.signal import find_peaks

    columns = smoothed.shape[1]
    for middle in (0.50, 0.30, 0.70):
        candidates = []
        left = max(0, int(columns * (middle - 0.10)))
        right = min(columns, max(left + 1, int(columns * (middle + 0.10))))
        profile = np.median(smoothed[:, left:right], axis=1)
        derivative = np.gradient(profile)
        energy = np.abs(derivative)
        threshold = max(float(np.median(energy)) * 4,
                        float(np.percentile(energy, 95)) * 0.12,
                        np.finfo(np.float32).eps)
        peaks, _ = find_peaks(energy, height=threshold, distance=2)
        groups = []
        group = []
        max_gap = max(4, int(math.ceil(height * 0.085 / block)))
        for peak in peaks:
            if group and peak - group[-1] > max_gap:
                groups.append(group)
                group = []
            group.append(int(peak))
        if group:
            groups.append(group)
        for group in groups:
            span = (group[-1] - group[0]) * block
            # Two edges of a single bright reflection do not establish a
            # retinal stack, regardless of their intensity or location.
            if len(group) < 3 or span < max(20, height * 0.06) or span > height * 0.55:
                continue
            signs = derivative[group]
            if not (np.any(signs > 0) and np.any(signs < 0)):
                continue
            if not _has_interior_contrast(profile, group[0], group[-1]):
                continue
            score = float(np.sum(energy[group])) * math.sqrt(span)
            candidates.append((score, (left + right) // 2, group[0], group[-1],
                               1 if derivative[group[0]] > 0 else -1,
                               1 if derivative[group[-1]] > 0 else -1))
        # The middle stripe defines the compact band when it contains enough
        # structure. Neighboring stripes are a fallback for a central shadow,
        # rather than a way for lateral reflections to replace a good seed.
        if candidates:
            return max(candidates, key=lambda item: item[0])[1:]
    return None


def _follow_transition(gradient: np.ndarray, column: int, row: int, sign: int):
    """Follow nearby layer edges outward, bridging short shadows by continuity."""
    rows, columns = gradient.shape
    window = 4
    near = np.arange(max(0, row - window), min(rows, row + window + 1))
    weight = np.exp(-0.5 * ((near - row) / 2.0) ** 2)
    row = int(near[np.argmax(np.maximum(0, sign * gradient[near, column]) * weight)])
    reference = max(float(sign * gradient[row, column]), np.finfo(np.float32).eps)
    positions = np.full(columns, float(row))
    supported = np.zeros(columns, dtype=bool)
    supported[column] = True
    for direction in (-1, 1):
        previous = float(row)
        slope = 0.0
        for x in range(column + direction, columns if direction > 0 else -1, direction):
            prediction = float(np.clip(previous + slope, 0, rows - 1))
            near = np.arange(max(0, int(round(prediction)) - window),
                             min(rows, int(round(prediction)) + window + 1))
            evidence = np.maximum(0, sign * gradient[near, x])
            score = evidence * np.exp(-0.5 * ((near - prediction) / 1.5) ** 2)
            chosen = int(np.argmax(score))
            if evidence[chosen] >= reference * 0.12:
                position = float(near[chosen])
                supported[x] = True
                slope = 0.7 * slope + 0.3 * float(np.clip(position - previous, -0.5, 0.5))
            else:
                position = prediction
            positions[x] = position
            previous = position
    return positions, supported


def _sparse_seed(smoothed: np.ndarray, gradient: np.ndarray, height: int, block: int):
    """Use separated coherent edges when shadows hide the center profile."""
    energy = np.abs(gradient)
    threshold = np.maximum(np.median(energy, axis=0) * 4,
                           np.percentile(energy, 95, axis=0) * 0.30)
    threshold = np.maximum(threshold, np.finfo(np.float32).eps)
    # An intense, narrow reflection must not raise the threshold enough to
    # hide the weaker edges of the actual retinal region.
    mask = energy > threshold[None, :]
    mask = median_filter(mask.astype(np.uint8), size=(1, 3), mode="nearest") > 0
    bands = []
    broad = max(8, int(math.ceil(gradient.shape[1] * 0.20)))
    short = max(3, int(math.ceil(gradient.shape[1] * 0.04)))
    for sign in (-1, 1):
        components, count = label(mask & (gradient * sign > 0),
                                  structure=np.ones((3, 3), dtype=bool))
        for component in range(1, count + 1):
            ys, xs = np.nonzero(components == component)
            span = np.unique(xs).size
            if span >= short:
                bands.append((float(np.median(ys)), span, ys, xs, sign))
    candidates = []
    for upper in bands:
        if upper[1] < broad:
            continue
        for lower in bands:
            if lower[4] == upper[4]:
                continue
            separation = (lower[0] - upper[0]) * block
            if not max(24, height * 0.06) <= separation <= height * 0.35:
                continue
            column = int(np.median(lower[3]))
            top = int(round(upper[0]))
            bottom = int(round(lower[0]))
            profile = np.median(smoothed[:, max(0, column - 2):column + 3], axis=1)
            if not _has_interior_contrast(profile, top, bottom):
                continue
            upper_energy = float(np.mean(energy[upper[2], upper[3]]))
            lower_energy = float(np.mean(energy[lower[2], lower[3]]))
            # Score the weaker edge of the pair. A distant brilliant
            # reflection must not dominate the selection of an unrelated
            # weak transition or the same-direction start of a plateau.
            score = min(upper_energy, lower_energy) * math.sqrt(separation)
            candidates.append((score, column, top, bottom,
                               upper[4], lower[4]))
    return max(candidates, key=lambda item: item[0])[1:] if candidates else None


def _image_envelope(image: np.ndarray):
    height, width = image.shape
    if height < 48 or width < 48:
        return None
    # A bounded analysis grid keeps opening even a large manual view quick.
    block = max(4, int(math.ceil(max(height, width) / 192)))
    reduced = _block_medians(image, block)
    if min(reduced.shape) < 8:
        return None
    smoothed = gaussian_filter(reduced, sigma=0.90, mode="nearest")
    gradient = np.gradient(smoothed, axis=0)
    seed = _center_seed(smoothed, block, height)
    if seed is None:
        seed = _sparse_seed(smoothed, gradient, height, block)
    if seed is None:
        return None
    column, upper, lower, top_sign, bottom_sign = seed
    tops, top_support = _follow_transition(gradient, column, upper, top_sign)
    bottoms, bottom_support = _follow_transition(gradient, column, lower, bottom_sign)
    if np.mean(top_support) < 0.35 or np.mean(bottom_support) < 0.20:
        return None
    # A local track must maintain distinct surfaces across the scan.
    thickness = (bottoms - tops) * block
    if np.median(thickness) < max(20, height * 0.05) or np.any(thickness <= 0):
        return None
    top = int(math.floor(float(tops.min()) * block))
    bottom = int(math.ceil(float(bottoms.max()) * block)) + block
    if bottom - top > height * 0.65:
        return None
    padding = max(16, block * 3)
    return max(0, top - padding), min(height, bottom + padding)


def detect_oct_crop(image, *, contours=None) -> CropSuggestion:
    """Return a source-coordinate box; uncertain images keep their full area.

    ``contours`` must contain matched native-coordinate ILM/BM rows, in that
    order. This is a suggestion, not a layer segmentation or an automatic save.
    """
    source = np.asarray(image)
    if source.ndim != 2 or not source.size:
        raise ValueError("OCT crop detection requires a nonempty 2-D image.")
    height, width = source.shape
    fallback = CropSuggestion((0, 0, width, height), "full image", False)
    if not np.issubdtype(source.dtype, np.number):
        return fallback
    analysis = np.array(source, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(analysis)):
        return fallback
    left, right = _side_bounds(analysis)
    bounds = _contour_envelope(contours, height, width)
    if bounds is not None:
        top, bottom = bounds
        return CropSuggestion((left, top, right - left, bottom - top), "EDB contours", True)
    if float(np.ptp(analysis)) == 0:
        return fallback
    # Analyze the useful width so exact padding does not drown out a scan.
    bounds = _image_envelope(analysis[:, left:right])
    if bounds is None:
        return fallback
    top, bottom = bounds
    return CropSuggestion((left, top, right - left, bottom - top), "image content", True)


def map_oct_crop_to_raw(roi, *, pixel_start: int, oct_width: int,
                        view_width: int, view_height: int):
    """Place an OCT rectangle in the unchanged full-file manual raw view.

    A native frame can begin part-way through a raw-view row. If its box wraps
    into the following row, select full width and include that extra row. A
    different row stride cannot display the OCT as a rectangle, so decline.
    """
    if oct_width != view_width or pixel_start < 0 or min(view_width, view_height) <= 0:
        return None
    x, y, width, height = (int(value) for value in roi)
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > oct_width:
        return None
    first = pixel_start + y * oct_width + x
    last = first + (height - 1) * oct_width + width - 1
    if last >= view_width * view_height:
        return None
    top, left = divmod(first, view_width)
    bottom = last // view_width + 1
    if left + width > view_width:
        left, width = 0, view_width
    return left, top, width, bottom - top
