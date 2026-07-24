"""Interactive-display smoke test for the shared AIDaS workspace layout."""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidas.app import AIDaSApp
from aidas.utils.ui_layout import workspace_sidebar_width


WINDOW_SIZES = ((1280, 820), (1024, 680))


def main() -> int:
    app = AIDaSApp()
    results = []
    try:
        app.deiconify()
        steps = (app.step1, app.step2, app.step3, app.step4)
        for width, height in WINDOW_SIZES:
            app.geometry(f"{width}x{height}")
            app.update()
            for step_number, step in enumerate(steps, start=1):
                app.notebook.select(step_number - 1)
                app.update()

                workspace_width = step.workspace.winfo_width()
                sidebar_width = step.sidebar_shell.winfo_width()
                content_width = step.content_shell.winfo_width()
                sidebar_right = step.sidebar_shell.winfo_rootx() + sidebar_width
                content_left = step.content_shell.winfo_rootx()
                overlap = max(0, sidebar_right - content_left)
                expected_sidebar = workspace_sidebar_width(workspace_width)

                assert overlap == 0, (
                    f"Step {step_number} overlaps by {overlap}px at {width}x{height}."
                )
                assert abs(sidebar_width - expected_sidebar) <= 2, (
                    f"Step {step_number} split is {sidebar_width}px; expected "
                    f"{expected_sidebar}px at {width}x{height}."
                )
                results.append(
                    (
                        width,
                        height,
                        step_number,
                        workspace_width,
                        sidebar_width,
                        content_width,
                        overlap,
                    )
                )
    finally:
        app.destroy()

    print("UI_LAYOUT_OK")
    for result in results:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
