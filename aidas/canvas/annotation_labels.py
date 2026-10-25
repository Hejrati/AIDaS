"""Viewport geometry for readable annotation labels, independent of Tk."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import isfinite
from typing import Hashable, Iterable, Sequence

import numpy as np


Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class LabelSpec:
    """The actual trace anchor and the measured outer size of its label."""

    key: Hashable
    anchor: Point
    width: float
    height: float


def _rectangle(rectangle: Sequence[float]) -> Box:
    left, top, right, bottom = map(float, rectangle)
    if not all(map(isfinite, (left, top, right, bottom))):
        raise ValueError("Viewport coordinates must be finite")
    return min(left, right), min(top, bottom), max(left, right), max(top, bottom)


def _clip_segment(start: Point, end: Point, bounds: Box) -> tuple[Point, Point] | None:
    """Clip a segment inclusively, including a segment with no interior vertex."""
    left, top, right, bottom = bounds
    x, y = start
    dx, dy = end[0] - x, end[1] - y
    first, last = 0.0, 1.0
    for direction, distance in (
        (-dx, x - left),
        (dx, right - x),
        (-dy, y - top),
        (dy, bottom - y),
    ):
        if direction == 0:
            if distance < 0:
                return None
            continue
        crossing = distance / direction
        if direction < 0:
            first = max(first, crossing)
        else:
            last = min(last, crossing)
        if first > last:
            return None
    return (x + first * dx, y + first * dy), (x + last * dx, y + last * dy)


def visible_polyline_anchor(points: Iterable[Sequence[float]], viewport: Sequence[float]) -> Point | None:
    """Return the leftmost visible point in the trace's input coordinates.

    Segment intersections are interpolated rather than snapping the anchor to
    the first stored vertex. Ties use the uppermost point, so reversed or
    nonmonotonic traces produce the same result. Nonfinite vertices break a
    trace instead of bridging an invalid portion of it.
    """
    bounds = _rectangle(viewport)
    left, top, right, bottom = bounds
    previous = None
    best = None
    for point in points:
        current = float(point[0]), float(point[1])
        x, y = current
        if not isfinite(x) or not isfinite(y):
            previous = None
            continue
        if left <= x <= right and top <= y <= bottom:
            if best is None or current < best:
                best = current
        if previous is not None:
            px, py = previous
            # Most dense retinal vertices are either beyond an anchor already
            # found or wholly outside one viewport edge. Avoid clipping those
            # segments, while still examining a trace that doubles back.
            could_improve = best is None or px <= best[0] or x <= best[0]
            same_outside_edge = (
                (px < left and x < left)
                or (px > right and x > right)
                or (py < top and y < top)
                or (py > bottom and y > bottom)
            )
            if could_improve and not same_outside_edge:
                clipped = _clip_segment(previous, current, bounds)
                if clipped is not None:
                    candidate = min(clipped)
                    best = candidate if best is None else min(best, candidate)
        previous = current
    return best


class PolylineAnchorIndex:
    """A reusable anchor lookup for a trace snapshot between geometry edits.

    Retinal traces normally run monotonically along x. Binary searches reduce
    those traces to the visible segment range, and vectorized y bounds reject
    invisible segments. An arbitrary manually drawn trace uses the general
    clipping helper so doubling back never attaches its label to a false point.
    """

    def __init__(self, points: Iterable[Sequence[float]]):
        self._points = tuple((float(point[0]), float(point[1])) for point in points)
        self._monotonic = False
        if not self._points:
            return
        coordinates = np.asarray(self._points)
        if not np.isfinite(coordinates).all():
            return
        xs = coordinates[:, 0]
        differences = np.diff(xs)
        if np.all(differences >= 0):
            self._monotonic = True
        elif np.all(differences <= 0):
            self._points = tuple(reversed(self._points))
            coordinates = coordinates[::-1]
            self._monotonic = True
        if self._monotonic:
            self._xs = tuple(point[0] for point in self._points)
            ys = coordinates[:, 1]
            self._minimum_y = float(ys.min())
            self._maximum_y = float(ys.max())
            self._segment_minimum_y = np.minimum(ys[:-1], ys[1:])
            self._segment_maximum_y = np.maximum(ys[:-1], ys[1:])

    def anchor(self, viewport: Sequence[float]) -> Point | None:
        """Return the same true visible anchor as ``visible_polyline_anchor``."""
        if not self._monotonic:
            return visible_polyline_anchor(self._points, viewport)
        bounds = _rectangle(viewport)
        left, top, right, bottom = bounds
        if bottom < self._minimum_y or top > self._maximum_y:
            return None
        if len(self._points) == 1:
            x, y = self._points[0]
            return (x, y) if left <= x <= right and top <= y <= bottom else None
        first = max(0, bisect_left(self._xs, left) - 1)
        last = min(len(self._points) - 1, bisect_right(self._xs, right))
        candidates = np.flatnonzero(
            (self._segment_minimum_y[first:last] <= bottom)
            & (self._segment_maximum_y[first:last] >= top)
        )
        best = None
        for offset in candidates:
            index = first + int(offset)
            start, end = self._points[index:index + 2]
            if best is not None and start[0] > best[0]:
                break
            clipped = _clip_segment(start, end, bounds)
            if clipped is not None:
                candidate = min(clipped)
                best = candidate if best is None else min(best, candidate)
                if best == (left, top):
                    break
        return best


def _overlaps(first: Box, second: Box, gap: float) -> bool:
    return (
        first[0] < second[2] + gap
        and first[2] + gap > second[0]
        and first[1] < second[3] + gap
        and first[3] + gap > second[1]
    )


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _column_tops(labels: list[LabelSpec], top: float, bottom: float, gap: float) -> list[float]:
    """Pack vertically in trace order, adjusting both edges of a close group."""
    positions = []
    next_top = top
    for label in labels:
        position = max(next_top, label.anchor[1] - label.height / 2)
        positions.append(position)
        next_top = position + label.height + gap
    next_bottom = bottom
    for index in range(len(labels) - 1, -1, -1):
        positions[index] = min(positions[index], next_bottom - labels[index].height)
        next_bottom = positions[index] - gap
    # Center a close group around its traces instead of pushing every label
    # below the first line. Uniform movement keeps the packed order intact.
    desired_shift = sum(label.anchor[1] - label.height / 2 - position for label, position in zip(labels, positions)) / len(labels)
    shift = _clamp(desired_shift, top - positions[0], bottom - positions[-1] - labels[-1].height)
    positions = [position + shift for position in positions]
    return positions


def layout_annotation_labels(
    labels: Iterable[LabelSpec],
    viewport: Sequence[float],
    *,
    obstacles: Iterable[Sequence[float]] = (),
    gap: float = 4.0,
    step: float = 12.0,
    margin: float = 4.0,
) -> dict[Hashable, Box]:
    """Place measured labels as a compact, nonoverlapping staircase.

    Labels follow vertical trace order, with a modest horizontal step. When
    the viewport is short, consecutive groups form additional columns. All
    returned boxes fit the viewport and avoid the supplied obstacles; a
    label is omitted only if there is insufficient room for its measured
    size. Coordinates and sizes use canvas pixels, not image pixels.
    """
    gap, step, margin = max(0.0, gap), max(0.0, step), max(0.0, margin)
    left, top, right, bottom = _rectangle(viewport)
    # Keep a margin whenever there is room, while allowing very narrow views.
    horizontal_margin = min(margin, max(0.0, (right - left) / 2))
    vertical_margin = min(margin, max(0.0, (bottom - top) / 2))
    left, right = left + horizontal_margin, right - horizontal_margin
    top, bottom = top + vertical_margin, bottom - vertical_margin
    width, height = right - left, bottom - top
    ordered = sorted(
        (
            label
            for label in labels
            if 0 < label.width <= width
            and 0 < label.height <= height
            and all(map(isfinite, (*label.anchor, label.width, label.height)))
        ),
        key=lambda label: (label.anchor[1], str(label.key)),
    )
    if not ordered:
        return {}
    blocked = [_rectangle(obstacle) for obstacle in obstacles]
    best = {}

    for column_count in range(1, len(ordered) + 1):
        # Balanced consecutive groups preserve order within every column.
        columns = [
            ordered[index * len(ordered) // column_count:(index + 1) * len(ordered) // column_count]
            for index in range(column_count)
        ]
        if any(sum(label.height for label in column) + gap * (len(column) - 1) > height for column in columns):
            continue
        minimum_widths = [max(label.width for label in column) for column in columns]
        spare_width = width - sum(minimum_widths) - gap * (column_count - 1)
        if spare_width < 0:
            continue

        placed = {}
        band_left = left
        for column, minimum_width in zip(columns, minimum_widths):
            band_right = band_left + minimum_width + spare_width / column_count
            tops = _column_tops(column, top, bottom, gap)
            next_top = top
            # Suffix sums avoid an O(n²) remaining-height computation.
            suffix_heights = [0.0] * (len(column) + 1)
            for index in range(len(column) - 1, -1, -1):
                suffix_heights[index] = suffix_heights[index + 1] + column[index].height
            for rank, (label, preferred_top) in enumerate(zip(column, tops)):
                remaining_height = suffix_heights[rank + 1] + gap * (len(column) - rank - 1)
                max_top = bottom - remaining_height - label.height
                preferred_top = _clamp(preferred_top, next_top, max_top)
                available_step = max(0.0, (band_right - band_left - minimum_width) / max(1, len(column) - 1))
                preferred_left = _clamp(
                    label.anchor[0] + gap + rank * min(step, available_step),
                    band_left,
                    band_right - label.width,
                )
                # Fast path: the preferred spot is usually free, so skip the
                # quadratic candidate search entirely in that case.
                preferred_box = (preferred_left, preferred_top, preferred_left + label.width, preferred_top + label.height)
                exclusions = [
                    exclusion for exclusion in (*blocked, *placed.values())
                    if exclusion[2] + gap > band_left and exclusion[0] < band_right + gap
                ]
                if not any(_overlaps(preferred_box, exclusion, gap) for exclusion in exclusions):
                    placed[label.key] = preferred_box
                    next_top = preferred_top + label.height + gap
                    continue
                x_candidates = {preferred_left, band_left, band_right - label.width}
                y_candidates = {preferred_top, next_top, max_top}
                for exclusion in exclusions:
                    x_candidates.update((
                        _clamp(exclusion[0] - gap - label.width, band_left, band_right - label.width),
                        _clamp(exclusion[2] + gap, band_left, band_right - label.width),
                    ))
                    y_candidates.update((
                        _clamp(exclusion[1] - gap - label.height, next_top, max_top),
                        _clamp(exclusion[3] + gap, next_top, max_top),
                    ))
                candidates = sorted(
                    ((x, y) for x in x_candidates for y in y_candidates),
                    key=lambda point: (
                        abs(point[0] - preferred_left) + 2 * abs(point[1] - preferred_top),
                        point[1], point[0],
                    ),
                )
                for x, y in candidates:
                    box = x, y, x + label.width, y + label.height
                    if not any(_overlaps(box, exclusion, gap) for exclusion in exclusions):
                        placed[label.key] = box
                        next_top = y + label.height + gap
                        break
            band_left = band_right + gap
        if len(placed) > len(best):
            best = placed
        if len(placed) == len(ordered):
            return placed
    return best
