"""Conservative histogram detection and masked repair of bright OCT pixels.

The input must use the intensity representation that will be displayed or
saved. In particular, Step 1's Analyze output uses signed 16-bit intensities.
No contrast normalization or filtering is applied to unmasked samples.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class BrightPixelDetection:
    """Suspect pixels separated from the main intensity population."""

    mask: np.ndarray
    threshold: float | None

    @property
    def count(self):
        return int(np.count_nonzero(self.mask))


def _image_array(image):
    data = np.asarray(image)
    if data.ndim != 2:
        raise ValueError("Bright-pixel processing expects a 2-D grayscale image.")
    if not (np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)):
        raise ValueError("Bright-pixel processing expects numeric intensities.")
    return data


def _protect_bright_bands(mask):
    """Retain long, continuous bright structures such as anatomical layers."""
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    height, width = mask.shape
    for label_id, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        region_height = region[0].stop - region[0].start
        region_width = region[1].stop - region[1].start
        if (region_width < max(16, width / 4)
                and region_height < max(16, height / 4)):
            continue
        component = labels[region] == label_id
        coordinates = np.column_stack(np.nonzero(component)).astype(np.float64)
        if len(coordinates) < 16:
            continue
        coordinates -= coordinates.mean(axis=0)
        minor, major = np.linalg.eigvalsh(coordinates.T @ coordinates / len(coordinates))
        if major >= 16 * (minor + 0.25):
            region_mask = mask[region]
            region_mask[component] = False


def _occupied_intensities(samples):
    """Count exact 8/16-bit values without sorting every image sample."""
    if np.issubdtype(samples.dtype, np.integer) and samples.dtype.itemsize <= 2:
        low = int(samples.min())
        counts = np.bincount(samples.astype(np.int64) - low)
        occupied = np.flatnonzero(counts)
        return (occupied + low).astype(samples.dtype), counts[occupied]
    return np.unique(samples, return_counts=True)


def detect_bright_pixels(image):
    """Find a small, clearly separated bright tail of the intensity histogram.

    A percentile alone never defines the mask: the upper population must have
    a pronounced empty intensity gap from the remaining pixels. A maximum of
    1% of finite samples may belong to that population. Normal intensity tails,
    zero padding, and long continuous bright bands are retained. This detects
    suspicious outliers conservatively; the caller still asks before repair.
    """
    data = _image_array(image)
    mask = np.zeros(data.shape, dtype=bool)
    finite = np.isfinite(data)
    samples = data[finite]
    threshold = None
    if samples.size >= 16:
        # Exact occupied-value frequencies avoid bin-boundary sensitivity for
        # signed OCT data and retain gaps between sparse upper-tail samples.
        values, frequencies = _occupied_intensities(samples)
        if len(values) > 1:
            values_float = values.astype(np.float64)
            # Missing SDB rows are padded with zeros. They must not dominate
            # the tissue statistics, especially in a tall crop. Keep them in
            # the histogram and total count so their gap and tail size remain
            # represented accurately.
            nonzero_samples = samples[samples != 0]
            bulk_samples = nonzero_samples if nonzero_samples.size >= 16 else samples
            q25, median, q75 = np.percentile(bulk_samples, (25, 50, 75))
            spread = float(q75 - q25)
            minimum_gap = 16.0 if np.issubdtype(data.dtype, np.integer) else (
                np.finfo(np.float64).eps * max(1.0, abs(float(median))) * 32
            )
            minimum_gap = max(minimum_gap, 2.0 * spread)
            upper_counts = np.cumsum(frequencies[::-1])[::-1][1:]
            tail_limit = max(1, int(samples.size * 0.01))
            with np.errstate(over="ignore"):
                candidates = np.flatnonzero(
                    (upper_counts <= tail_limit)
                    & (np.diff(values_float) >= minimum_gap)
                    & (values_float[1:] > float(median) + 3.0 * spread)
                )
            for index in candidates:
                # Halve before adding so opposite extreme float values do
                # not overflow while constructing their midpoint.
                cutoff = values_float[index] / 2.0 + values_float[index + 1] / 2.0
                candidate_mask = finite & (data > cutoff) & (data != 0)
                _protect_bright_bands(candidate_mask)
                if np.any(candidate_mask):
                    mask = candidate_mask
                    threshold = float(cutoff)
                    break
    mask.setflags(write=False)
    return BrightPixelDetection(mask=mask, threshold=threshold)


def detect_bright_pixels_in_bands(image, *, band_height=128):
    """Find source-image outliers using the whole image and overlapping bands.

    Mixed OCT backgrounds can hide tissue defects in a whole-image histogram.
    Full-width horizontal bands retain local tissue statistics independently
    of the crop selected later. Every band uses the same conservative detector
    and anatomical-band protection. Their masks are combined with the global
    mask, leaving the caller's intensities unchanged.
    """
    data = _image_array(image)
    if (isinstance(band_height, (bool, np.bool_))
            or not isinstance(band_height, (int, np.integer))
            or band_height <= 0):
        raise ValueError("band_height must be a positive integer.")
    band_height = int(band_height)
    global_detection = detect_bright_pixels(data)
    mask = np.array(global_detection.mask, copy=True)
    thresholds = []
    if global_detection.count:
        thresholds.append(global_detection.threshold)

    height = data.shape[0]
    if height > band_height:
        stride = max(1, band_height // 2)
        final_start = height - band_height
        starts = list(range(0, final_start + 1, stride))
        if starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            band_detection = detect_bright_pixels(data[start:start + band_height, :])
            if band_detection.count:
                mask[start:start + band_height, :] |= band_detection.mask
                thresholds.append(band_detection.threshold)

    mask.setflags(write=False)
    return BrightPixelDetection(
        mask=mask,
        threshold=float(min(thresholds)) if thresholds else None,
    )


def smooth_bright_pixels(image, mask):
    """Replace only masked pixels with local averages of usable neighbors.

    Original masked values never contribute to an average. Adjacent suspect
    pixels are repaired in simultaneous waves from the surrounding good data,
    so even a dense blob can be repaired without spreading its bright values.
    The returned copy retains the source dtype and every unmasked value exactly.
    """
    data = _image_array(image)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != data.shape:
        raise ValueError("The bright-pixel mask must match the image shape.")
    result = np.array(data, copy=True)
    if not np.any(selected):
        return result
    known = ~selected & np.isfinite(data)
    if not np.any(known):
        raise ValueError("Cannot smooth bright pixels without unmasked finite neighbors.")

    # Work in the affected bounding box and its surrounding good-pixel border.
    # This keeps sparse repairs cheap even when the full raw image is large.
    rows, columns = np.nonzero(selected)
    region = (
        slice(max(0, int(rows.min()) - 1), min(data.shape[0], int(rows.max()) + 2)),
        slice(max(0, int(columns.min()) - 1), min(data.shape[1], int(columns.max()) + 2)),
    )
    region_known = known[region].copy()
    region_selected = selected[region]
    pending = region_selected.copy()
    work = np.where(region_known, data[region], 0).astype(np.float64)
    kernel = np.ones((3, 3), dtype=np.float64)
    # Integer OCT neighborhood sums fit exactly in float64. Scaling large
    # floating values before summing prevents overflow for supported floats.
    mean_scale = 1.0 if np.issubdtype(data.dtype, np.integer) else 9.0
    while np.any(pending):
        neighbor_count = ndimage.convolve(
            region_known.astype(np.float64), kernel, mode="constant", cval=0
        )
        fill = pending & (neighbor_count > 0)
        if not np.any(fill):
            # Nonfinite unmasked barriers can disconnect a selected region.
            # Use original good data while leaving every barrier untouched.
            return _fill_disconnected(data, selected, result)
        neighbor_sum = ndimage.convolve(
            np.where(region_known, work / mean_scale, 0), kernel, mode="constant", cval=0
        )
        work[fill] = neighbor_sum[fill] / (neighbor_count[fill] / mean_scale)
        region_known[fill] = True
        pending[fill] = False

    replacements = work[region_selected]
    if np.issubdtype(data.dtype, np.integer):
        limits = np.iinfo(data.dtype)
        replacements = np.clip(np.rint(replacements), limits.min, limits.max)
    region_result = result[region]
    region_result[region_selected] = replacements.astype(data.dtype)
    return result


def _fill_disconnected(data, selected, result):
    """Use the nearest finite unmasked sample when invalid values form a wall."""
    known = ~selected & np.isfinite(data)
    indices = ndimage.distance_transform_edt(
        ~known, return_distances=False, return_indices=True
    )
    nearest = data[tuple(indices)]
    result[selected] = nearest[selected]
    return result
