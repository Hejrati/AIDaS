"""Deterministic synthetic retinal image fixtures."""

import numpy as np


def curved_retina(*, height=420, width=320, blank_sides=0, wrapped=False):
    """Layered curved retina above a long, gently fading choroid signal."""
    rng = np.random.default_rng(913)
    xx = np.arange(width, dtype=float)
    yy = np.arange(height, dtype=float)[:, None]
    top = 95 + 30 * ((xx - width / 2) / (width / 2)) ** 2
    top += 14 * np.exp(-((xx - width / 2) / (width / 10)) ** 2)
    bottom = top + 77 + 5 * np.sin(xx / 47)
    data = 120 + rng.normal(0, 4, (height, width))
    inside = (yy >= top) & (yy <= bottom)
    data += inside * 115
    for position, amplitude, sigma in (
        (top, 470, 1.8),
        (top + 18, 120, 2.0),
        (top + 39, 95, 2.0),
        (bottom - 8, 480, 2.2),
        (bottom, 145, 1.5),
        # A faint outside layer must survive a proposal driven by bright layers.
        (bottom + 12, 45, 1.5),
    ):
        data += amplitude * np.exp(-0.5 * ((yy - position) / sigma) ** 2)
    distance = yy - bottom - 16
    data += np.where(distance > 0, 130 * np.exp(-np.maximum(distance, 0) / 70), 0)
    data = np.clip(np.rint(data), 0, 65535).astype(np.uint16)
    if wrapped:
        # Native words above the signed range with zero-valued background
        # speckles resemble the supplied SDB without decoding its encoding.
        data = (47000 + data.astype(np.uint32) * 17).astype(np.uint16)
        background = (yy < top - 10) | (yy > bottom + 65)
        data[background & (rng.random(data.shape) < 0.38)] = 0
    if blank_sides:
        data[:, :blank_sides] = 0
        data[:, width - blank_sides:] = 0
    return data, top, bottom + 12
