"""Shared sizing, spacing, and split-layout rules for the AIDaS desktop UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutTokens:
    """One source of truth for application and workspace geometry."""

    design_width: int = 1280
    design_height: int = 820
    minimum_width: int = 1024
    minimum_height: int = 680
    screen_fraction: float = 0.75

    sidebar_width: int = 380
    sidebar_minimum: int = 320
    content_minimum: int = 560
    divider_width: int = 6

    space_xs: int = 4
    space_sm: int = 8
    space_md: int = 12
    space_lg: int = 16


@dataclass(frozen=True)
class ColorTokens:
    """Neutral scientific-workstation palette with one accessible accent."""

    application: str = "#e9eef3"
    surface: str = "#ffffff"
    surface_subtle: str = "#f5f7fa"
    border: str = "#c9d2dc"
    text: str = "#17212b"
    muted_text: str = "#5d6b78"
    accent: str = "#0b5f9e"
    accent_hover: str = "#084b7d"
    accent_soft: str = "#e5f1f9"


LAYOUT = LayoutTokens()
COLORS = ColorTokens()


def workspace_sidebar_width(
    total_width: int,
    *,
    divider_width: int = LAYOUT.divider_width,
    sidebar_width: int = LAYOUT.sidebar_width,
    sidebar_minimum: int = LAYOUT.sidebar_minimum,
    content_minimum: int = LAYOUT.content_minimum,
) -> int:
    """Return a compact sidebar width without overlapping the content pane.

    The preferred width stays fixed whenever both panes can satisfy their
    minimum usable widths. If an embedding window is unusually narrow, the
    sidebar contracts only as much as required to preserve the content pane.
    """

    available = max(0, int(total_width) - max(0, int(divider_width)))
    if available == 0:
        return 0

    desired = max(1, int(sidebar_width))
    sidebar_minimum = max(0, int(sidebar_minimum))
    content_minimum = max(0, int(content_minimum))

    if available >= sidebar_minimum + content_minimum:
        return min(max(desired, sidebar_minimum), available - content_minimum)

    return min(desired, max(1, available - 1))
