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


WINDOW_SIZES = ((1800, 1000), (1280, 820), (1024, 680))


def main() -> int:
    app = AIDaSApp()
    results = []
    try:
        app._finish_startup()
        app.update_idletasks()
        assert (app.winfo_width(), app.winfo_height()) == app._startup_window_size, (
            "The revealed main window does not match its adaptive startup size."
        )
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
                sash_x, sash_y = step.workspace.sash_coord(0)
                step.workspace.event_generate(
                    "<ButtonPress-1>", x=sash_x, y=sash_y + 10
                )
                step.workspace.event_generate(
                    "<B1-Motion>", x=sash_x + 80, y=sash_y + 10
                )
                step.workspace.event_generate(
                    "<ButtonRelease-1>", x=sash_x + 80, y=sash_y + 10
                )
                app.update_idletasks()
                assert step.sidebar_shell.winfo_width() == sidebar_width, (
                    f"Step {step_number} allowed its fixed sidebar divider to move."
                )
                if step_number == 1:
                    action_heights = {
                        step.crop_btn.winfo_reqheight(),
                        step.undo_crop_btn.winfo_reqheight(),
                        step.save_all_btn.winfo_reqheight(),
                    }
                    assert len(action_heights) == 1, (
                        "Step 1 Crop, Undo, and Save buttons do not have a "
                        "consistent requested height."
                    )
                    action_buttons = (
                        step.crop_btn,
                        step.undo_crop_btn,
                        step.save_all_btn,
                    )
                    action_gaps = {
                        later.winfo_rooty()
                        - (earlier.winfo_rooty() + earlier.winfo_height())
                        for earlier, later in zip(action_buttons, action_buttons[1:])
                    }
                    assert len(action_gaps) == 1, (
                        "Step 1 Crop, Undo, and Save buttons do not have "
                        "consistent vertical spacing."
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
