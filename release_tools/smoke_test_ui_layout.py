"""Interactive-display smoke test for the shared AIDaS workspace layout."""

# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aidas.app as app_module
from aidas.app import AIDaSApp
from aidas.core.config import Config
from aidas.ui.components import AppSplitButton
from aidas.ui.theme import COLOR_PAIRS, resolve_color
from aidas.utils.ui_layout import LAYOUT, workspace_sidebar_width


WINDOW_SIZES = ((1800, 1000), (1280, 820), (1024, 680))


class _SmokePreferences:
    """In-memory preferences keep the smoke independent of a user's settings."""

    def __init__(self, interface_mode):
        self.prefs = Config.DEFAULTS.copy()
        self.prefs["interface_mode"] = interface_mode

    def get(self, key, default=None):
        return self.prefs.get(key, default)

    def set(self, key, value):
        self.prefs[key] = value


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


def _sidebar_horizontal_overflow(step):
    left = step.sidebar.canvas.winfo_rootx()
    right = left + step.sidebar.canvas.winfo_width()
    overflow = []
    for widget in _descendants(step.ctrl):
        if not widget.winfo_ismapped() or widget.winfo_width() <= 1:
            continue
        widget_left = widget.winfo_rootx()
        widget_right = widget_left + widget.winfo_width()
        clipped_width = (
            isinstance(widget, ttk.Button)
            and widget.winfo_width() + 1 < widget.winfo_reqwidth()
        )
        if widget_left < left - 1 or widget_right > right + 1 or clipped_width:
            overflow.append(str(widget))
    return overflow


def _widget_center(widget):
    return (
        widget.winfo_rootx() + widget.winfo_width() / 2,
        widget.winfo_rooty() + widget.winfo_height() / 2,
    )


def _assert_about_presentation(app):
    """Verify About uses the active interface and a contained OK action."""

    app._show_about()
    app.update()
    dialog = app._about_dialog
    try:
        assert dialog is not None and dialog.winfo_exists()
        dialog.wait_visibility()
        app.update_idletasks()
        assert dialog._presentation_mode == app.interface_mode
        assert dialog.ok_button.cget("text") == "OK"
        assert dialog.ok_button.winfo_rootx() >= dialog.winfo_rootx()
        assert (
            dialog.ok_button.winfo_rootx() + dialog.ok_button.winfo_width()
            <= dialog.winfo_rootx() + dialog.winfo_width()
        )
        assert (
            dialog.ok_button.winfo_rooty() + dialog.ok_button.winfo_height()
            <= dialog.winfo_rooty() + dialog.winfo_height()
        )
        if app.interface_mode == "Classic":
            assert dialog._classic_about
            assert isinstance(dialog.classic_content, ttk.Frame)
            assert isinstance(dialog.classic_footer, ttk.Frame)
            assert isinstance(dialog.ok_button, ttk.Button)
            assert dialog.classic_content.winfo_y() == 0
            assert (
                dialog.classic_footer.winfo_y()
                + dialog.classic_footer.winfo_height()
                == dialog.winfo_height()
            ), "Classic About footer does not cover the bottom of the dialog."
            assert all(
                child.winfo_y() + child.winfo_height()
                <= dialog.classic_content.winfo_height()
                for child in dialog.classic_content.winfo_children()
            ), "Classic About content is clipped or escapes its opaque surface."
        else:
            assert not dialog._classic_about
            assert not isinstance(dialog.ok_button, ttk.Button)
    finally:
        try:
            dialog._close()
        except (AttributeError, tk.TclError):
            pass
        app._about_dialog = None
        app.update()


def _assert_settings_presentation(app):
    """Verify Settings matches the active shell and remains usable."""

    app._show_settings()
    app.update()
    dialog = app._settings_dialog
    try:
        assert dialog is not None and dialog.winfo_exists()
        assert dialog._presentation_mode == app.interface_mode
        if app.interface_mode == "Classic":
            assert isinstance(dialog.interface_menu, ttk.Combobox)
            assert isinstance(dialog.appearance_menu, ttk.Combobox)
            assert str(dialog.appearance_menu.cget("state")) == "disabled"

            scripts = dialog.script_menus["main"].master
            occupied = set()
            for widget in scripts.grid_slaves():
                info = widget.grid_info()
                for row in range(
                    int(info["row"]),
                    int(info["row"]) + int(info.get("rowspan", 1)),
                ):
                    for column in range(
                        int(info["column"]),
                        int(info["column"]) + int(info.get("columnspan", 1)),
                    ):
                        cell = (row, column)
                        assert cell not in occupied, (
                            "Classic Settings has overlapping Step 3 script controls."
                        )
                        occupied.add(cell)

            canvas = next(
                widget
                for widget in dialog.settings_panel.winfo_children()
                if isinstance(widget, tk.Canvas)
            )
            dialog.geometry("720x420")
            app.update()
            canvas.yview_moveto(1.0)
            app.update()
            before = canvas.yview()
            dialog.script_menus["output"].event_generate(
                "<MouseWheel>",
                delta=120,
            )
            app.update()
            assert canvas.yview() != before, (
                "Classic Settings does not scroll from descendant controls."
            )
        else:
            assert not isinstance(dialog.interface_menu, ttk.Combobox)
            assert not isinstance(dialog.appearance_menu, ttk.Combobox)
            assert dialog.appearance_menu.cget("state") == "normal"
    finally:
        try:
            dialog._close()
        except (AttributeError, tk.TclError):
            pass
        app._settings_dialog = None
        app.update()


def _assert_crop_split_button_geometry(step):
    """Guard the rendered split control's silhouette and arrow alignment."""

    split = step.crop_split_frame
    action = step.crop_btn
    options = step.crop_options_btn
    divider = step.crop_split_divider
    undo = step.undo_crop_btn
    split.update_idletasks()

    split_x = split.winfo_rootx()
    split_y = split.winfo_rooty()
    split_width = split.winfo_width()
    split_height = split.winfo_height()
    assert (split_width, split_height) == (
        AppSplitButton.DEFAULT_WIDTH,
        40,
    ), (
        "Crop & Scale must keep a clean 198x40 outer silhouette; "
        f"rendered size is {split_width}x{split_height}."
    )
    assert split_height == undo.winfo_height(), (
        "Crop & Scale and Undo do not have the same rendered height."
    )
    assert split_y == undo.winfo_rooty(), (
        "Crop & Scale and Undo are not vertically aligned."
    )
    assert action.cget("corner_radius") == undo.cget("corner_radius"), (
        "Crop & Scale does not use the same standard corner radius as Undo."
    )
    assert options.cget("corner_radius") == undo.cget("corner_radius"), (
        "The split-button segments do not share the standard corner radius."
    )
    assert split.cget("fg_color") == "transparent", (
        "The split button has a colored outer container that distorts its silhouette."
    )

    primary = COLOR_PAIRS["primary"]
    exterior = split.cget("bg_color")
    expected_action_corners = (
        (exterior,) * 4
        if action.cget("state") == "disabled"
        else (exterior, primary, primary, exterior)
    )
    expected_option_corners = (
        (exterior,) * 4
        if options.cget("state") == "disabled"
        else (primary, exterior, exterior, primary)
    )
    assert tuple(action.cget("background_corner_colors")) == expected_action_corners, (
        "The action segment does not preserve its enabled or disabled silhouette."
    )
    assert tuple(options.cget("background_corner_colors")) == expected_option_corners, (
        "The dropdown segment does not preserve its enabled or disabled silhouette."
    )

    assert options.winfo_width() == AppSplitButton.SEGMENT_WIDTH, (
        "The Crop & Scale dropdown target is not the standard square width."
    )
    for name, segment in (("action", action), ("dropdown", options)):
        assert segment.winfo_rooty() == split_y, (
            f"The split button {name} segment does not align with the outer edge."
        )
        assert segment.winfo_height() == split_height, (
            f"The split button {name} segment has an inconsistent height."
        )

    action_right = action.winfo_rootx() + action.winfo_width()
    divider_x = divider.winfo_rootx()
    options_x = options.winfo_rootx()
    options_right = options_x + options.winfo_width()
    assert action.winfo_rootx() == split_x
    assert action_right == options_x, (
        "The Crop & Scale segments overlap or leave a gap at their join."
    )
    assert divider.winfo_width() == 1, "The split-button divider must be exactly 1px."
    assert abs((divider_x + divider.winfo_width() / 2) - options_x) <= 1, (
        "The divider is not centered on the split-button join."
    )
    assert options_right == split_x + split_width, (
        "The dropdown segment does not terminate cleanly at the outer edge."
    )
    assert divider.winfo_height() >= split_height // 2, (
        "The split-button divider is too short to read as a segment boundary."
    )

    assert options.cget("text") == "\u25be", (
        "Crop & Scale must use the standard small down-pointing triangle."
    )
    assert options.cget("anchor") == "center"
    arrow_label = getattr(options, "_text_label", None)
    assert arrow_label is not None and arrow_label.winfo_ismapped(), (
        "The split-button arrow label is not rendered."
    )
    option_center = _widget_center(options)
    arrow_center = _widget_center(arrow_label)
    assert abs(option_center[0] - arrow_center[0]) <= 2, (
        "The split-button arrow is not horizontally centered in its segment."
    )
    assert abs(option_center[1] - arrow_center[1]) <= 2, (
        "The split-button arrow is not vertically centered in its segment."
    )


def _assert_step1_sidebar_actions_visible(step, width, height):
    """Guard the fixed Step 1 footer and its actions at each window size."""

    assert step.step_actions_footer.master is step.sidebar_shell, (
        "Step 1 workflow actions are not attached to the fixed sidebar shell."
    )
    assert step.step_actions_frame.master is step.step_actions_footer
    assert step.save_all_btn.master is step.step_actions_frame
    assert step.batch_segment_cropped_btn.master is step.step_actions_frame

    shell_left = step.sidebar_shell.winfo_rootx()
    shell_top = step.sidebar_shell.winfo_rooty()
    shell_right = shell_left + step.sidebar_shell.winfo_width()
    shell_bottom = shell_top + step.sidebar_shell.winfo_height()
    for name, widget in (
        ("footer", step.step_actions_footer),
        ("action row", step.step_actions_frame),
        ("Save", step.save_all_btn),
        ("Go to Step 2", step.batch_segment_cropped_btn),
    ):
        assert widget.winfo_ismapped(), (
            f"Step 1 {name} is hidden at {width}x{height}."
        )
        widget_left = widget.winfo_rootx()
        widget_top = widget.winfo_rooty()
        widget_right = widget_left + widget.winfo_width()
        widget_bottom = widget_top + widget.winfo_height()
        assert (
            widget_left >= shell_left - 1
            and widget_top >= shell_top - 1
            and widget_right <= shell_right + 1
            and widget_bottom <= shell_bottom + 1
        ), f"Step 1 {name} escapes the sidebar viewport at {width}x{height}."

    scroll_bottom = (
        step.sidebar.canvas.winfo_rooty() + step.sidebar.canvas.winfo_height()
    )
    assert scroll_bottom <= step.step_actions_footer.winfo_rooty(), (
        f"Step 1 scrolling controls overlap the fixed actions at {width}x{height}."
    )


def _assert_step4_build_stack_visible(step, width, height):
    """Guard the fixed Step 4 footer and Build stack action."""

    assert step.sidebar_footer.master is step.sidebar_shell, (
        "Step 4 Build stack footer is not attached to the fixed sidebar shell."
    )
    assert step.build_stacks_button.master is step.sidebar_footer

    shell_left = step.sidebar_shell.winfo_rootx()
    shell_top = step.sidebar_shell.winfo_rooty()
    shell_right = shell_left + step.sidebar_shell.winfo_width()
    shell_bottom = shell_top + step.sidebar_shell.winfo_height()
    for name, widget in (
        ("footer", step.sidebar_footer),
        ("Build stack", step.build_stacks_button),
    ):
        assert widget.winfo_ismapped(), (
            f"Step 4 {name} is hidden at {width}x{height}."
        )
        widget_left = widget.winfo_rootx()
        widget_top = widget.winfo_rooty()
        widget_right = widget_left + widget.winfo_width()
        widget_bottom = widget_top + widget.winfo_height()
        assert (
            widget_left >= shell_left - 1
            and widget_top >= shell_top - 1
            and widget_right <= shell_right + 1
            and widget_bottom <= shell_bottom + 1
        ), f"Step 4 {name} escapes the sidebar viewport at {width}x{height}."

    button = step.build_stacks_button
    assert button.winfo_height() + 1 >= button.winfo_reqheight(), (
        f"Step 4 Build stack is vertically clipped at {width}x{height}."
    )
    scroll_bottom = (
        step.sidebar.canvas.winfo_rooty() + step.sidebar.canvas.winfo_height()
    )
    assert scroll_bottom <= step.sidebar_footer.winfo_rooty() + 1, (
        f"Step 4 scrolling controls overlap Build stack at {width}x{height}."
    )


def main(interface_mode="Modern") -> int:
    preferences = _SmokePreferences(interface_mode)
    config_factory = mock.Mock(return_value=preferences)
    config_factory.peek.return_value = interface_mode
    with mock.patch.object(app_module, "Config", config_factory):
        app = AIDaSApp()
    results = []
    try:
        app._finish_startup()
        app.update_idletasks()
        assert (app.winfo_width(), app.winfo_height()) == app._startup_window_size, (
            "The revealed main window does not match its adaptive startup size."
        )
        if app.interface_mode == "Modern":
            for appearance_mode in ("Light", "Dark"):
                assert resolve_color(
                    COLOR_PAIRS["window_chrome"], appearance_mode
                ) != resolve_color(COLOR_PAIRS["menu_bar"], appearance_mode), (
                    f"The title and menu bars merge in {appearance_mode} mode."
                )

            navigation = app.header.navigation
            assert navigation.cget("border_width") == 0, (
                "The workflow step selector still draws a connecting outer boundary."
            )
            assert navigation.cget("fg_color") == "transparent", (
                "The workflow step selector still draws a connecting background track."
            )
            navigation_buttons = tuple(navigation._buttons_dict.values())
            assert len(navigation_buttons) == len(app.header.DEFAULT_STEPS)
            assert all(button.cget("border_width") >= 1 for button in navigation_buttons), (
                "One or more workflow steps has no individual boundary."
            )
            selected_label = navigation.get()
            for label, button in navigation._buttons_dict.items():
                selected = label == selected_label
                assert button.cget("fg_color") == (
                    COLOR_PAIRS["primary"]
                    if selected
                    else COLOR_PAIRS["button"]
                )
                assert button.cget("border_color") == (
                    COLOR_PAIRS["primary"]
                    if selected
                    else COLOR_PAIRS["border_strong"]
                )
                assert button.cget("text_color") == (
                    COLOR_PAIRS["on_primary"]
                    if selected
                    else COLOR_PAIRS["text"]
                )
            ordered_buttons = tuple(
                sorted(navigation_buttons, key=lambda button: button.winfo_rootx())
            )
            button_widths = {button.winfo_width() for button in ordered_buttons}
            assert max(button_widths) - min(button_widths) <= 1
            button_gaps = tuple(
                following.winfo_rootx()
                - (current.winfo_rootx() + current.winfo_width())
                for current, following in zip(ordered_buttons, ordered_buttons[1:])
            )
            assert button_gaps and min(button_gaps) > 0, (
                "Workflow buttons are still visually connected."
            )
            assert max(button_gaps) - min(button_gaps) <= 1, (
                "Workflow button spacing is not uniform."
            )
        else:
            assert app.header is None
            assert app.window_title_bar is None
            assert app.status_bar is None
            assert app.classic_menu.current_interface == "Classic"
            assert app.style.layout("AIDaS.TNotebook.Tab"), (
                "Classic mode must show the original four workflow tabs."
            )
            assert app.notebook.index("end") == 4

        _assert_about_presentation(app)
        _assert_settings_presentation(app)

        steps = (app.step1, app.step2, app.step3, app.step4)
        assert app.step2.ai_device_status_var.get().startswith("AI device:"), (
            "Step 2 does not expose its GPU/core execution status."
        )
        # The fallback-core selector is useful only when Step 2 cannot use a
        # compatible GPU. Exercise real ttk packing in both states.
        previous_gpu_compatible = app.step2._ai_gpu_compatible
        previous_runtime_gpu = app.step2._ai_runtime_using_gpu
        app.step2._open_step2_batch_segmentation_panel(
            PROJECT_ROOT,
            initial_rows=[],
        )
        device_panel = app.step2.batch_segmentation_panel
        try:
            app.step2._ai_gpu_compatible = True
            app.step2._ai_runtime_using_gpu = None
            device_panel._sync_device_controls()
            app.update_idletasks()
            assert not device_panel.core_row.winfo_manager(), (
                "Step 2 shows fallback-core controls for a compatible GPU."
            )

            app.step2._ai_runtime_using_gpu = False
            device_panel._sync_device_controls()
            app.update_idletasks()
            assert device_panel.core_row.winfo_manager() == "pack", (
                "Step 2 did not restore fallback-core controls after GPU fallback."
            )
        finally:
            app.step2._ai_gpu_compatible = previous_gpu_compatible
            app.step2._ai_runtime_using_gpu = previous_runtime_gpu
            app.step2._close_step2_batch_segmentation_panel(restore_previous=True)
        # Verify the live callbacks use one shared budget without launching
        # either long-running workflow. A three-core Step 3 reservation must
        # immediately reduce Step 2's fallback allowance by three.
        total_cores = app.step2._available_core_count()
        simulated_step3_cores = min(3, total_cores)
        app.step3._busy = True
        app.step3._active_r_core_allocation = simulated_step3_cores
        try:
            shared_total, step3_used, step2_free = app.step2._shared_core_budget()
            assert shared_total == total_cores
            assert step3_used == simulated_step3_cores
            assert step2_free == total_cores - simulated_step3_cores
        finally:
            app.step3._active_r_core_allocation = 0
            app.step3._busy = False
        # Exercise both live shell transitions before the layout matrix.  The
        # workflow objects and active page must survive; only presentation
        # chrome, menu, navigation, status, colors, and corner treatment may
        # change.
        original_mode = app.interface_mode
        workflow_ids = tuple(id(step) for step in steps)
        app.notebook.select(2)
        app.update()
        other_mode = "Classic" if original_mode == "Modern" else "Modern"
        for target_mode in (other_mode, original_mode, other_mode, original_mode):
            app._set_interface(target_mode)
            app.update_idletasks()
            app.update()
            assert app.interface_mode == target_mode
            assert app.requested_interface_mode == target_mode
            assert app.state() != "withdrawn" and app.winfo_viewable(), (
                "The main window was hidden while restoring interface chrome."
            )
            assert tuple(id(step) for step in steps) == workflow_ids, (
                "A live interface switch reconstructed a workflow page."
            )
            assert app.notebook.index(app.notebook.select()) == 2, (
                "A live interface switch changed the active workflow page."
            )
            if target_mode == "Classic":
                assert app.menu_bar is None
                assert app.classic_menu is not None
                assert not app.classic_menu.appearance_enabled
                assert app.header is None
                assert app.status_bar is None
                assert app.style.layout("AIDaS.TNotebook.Tab")
                assert app.step1.crop_btn.cget("corner_radius") == 0
            else:
                assert app.classic_menu is None
                assert app.menu_bar is not None
                assert app.menu_bar.appearance_enabled
                assert app.header is not None
                assert app.status_bar is not None
                modern_tab_layout = app.style.layout("AIDaS.TNotebook.Tab")
                assert modern_tab_layout and modern_tab_layout[0][0] == "null", (
                    "Modern mode did not hide the native workflow tabs."
                )
                assert app.step1.crop_btn.cget("corner_radius") > 0

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
                overflow = _sidebar_horizontal_overflow(step)
                assert not overflow, (
                    f"Step {step_number} has controls outside the sidebar viewport: "
                    + ", ".join(overflow[:5])
                )
                assert int(step.content_status_bar._label.cget("padx")) >= LAYOUT.space_sm, (
                    f"Step {step_number} status text touches its left border."
                )
                if (width, height) == WINDOW_SIZES[-1]:
                    step.sidebar.canvas.yview_moveto(1.0)
                    app.update_idletasks()
                    mapped_controls = [
                        widget
                        for widget in _descendants(step.ctrl)
                        if widget.winfo_ismapped() and widget.winfo_height() > 1
                    ]
                    lowest_control = max(
                        widget.winfo_rooty() + widget.winfo_height()
                        for widget in mapped_controls
                    )
                    viewport_bottom = (
                        step.sidebar.canvas.winfo_rooty()
                        + step.sidebar.canvas.winfo_height()
                    )
                    assert lowest_control <= viewport_bottom + 1, (
                        f"Step {step_number} cannot scroll to its lowest sidebar control."
                    )
                    step.sidebar.canvas.yview_moveto(0.0)
                if step_number == 1:
                    _assert_crop_split_button_geometry(step)
                    _assert_step1_sidebar_actions_visible(step, width, height)
                    assert (
                        step.content_status_bar.winfo_rootx()
                        == step.canvas_roi_toolbar.winfo_rootx()
                    ), "Step 1 status and bottom toolbar do not share a left edge."
                    assert (
                        step.content_status_bar.winfo_width()
                        == step.canvas_roi_toolbar.winfo_width()
                    ), "Step 1 status and bottom toolbar do not have the same width."
                    action_heights = {
                        step.crop_split_frame.winfo_reqheight(),
                        step.undo_crop_btn.winfo_reqheight(),
                    }
                    assert len(action_heights) == 1, (
                        "Step 1 Crop and Undo buttons do not have a "
                        "consistent requested height."
                    )
                    sidebar_action_heights = {
                        step.save_all_btn.winfo_reqheight(),
                        step.batch_segment_cropped_btn.winfo_reqheight(),
                    }
                    assert len(sidebar_action_heights) == 1, (
                        "Step 1 Save and Go to Step 2 buttons do not have a "
                        "consistent requested height."
                    )
                    assert step.crop_btn.winfo_reqheight() == step.crop_options_btn.winfo_reqheight(), (
                        "Step 1 Crop split-button segments have different heights."
                    )
                    action_buttons = (
                        step.crop_btn,
                        step.crop_options_btn,
                        step.undo_crop_btn,
                    )
                    assert step.crop_split_frame.master is step.canvas_toolbar
                    assert step.crop_btn.master is step.crop_split_frame
                    assert step.crop_options_btn.master is step.crop_split_frame
                    assert all(
                        button.master is step.canvas_toolbar
                        for button in action_buttons[2:]
                    ), "Step 1 processing actions are not in the top canvas toolbar."
                    aligned_controls = (
                        step.crop_split_frame,
                        step.undo_crop_btn,
                    )
                    assert len({control.winfo_rooty() for control in aligned_controls}) == 1, (
                        "Step 1 processing buttons are not aligned in one row."
                    )
                    sidebar_actions = (
                        step.save_all_btn,
                        step.batch_segment_cropped_btn,
                    )
                    assert len({control.winfo_rooty() for control in sidebar_actions}) == 1, (
                        "Step 1 sidebar actions are not aligned in one row."
                    )
                    roi_controls = tuple(
                        step.roi_entries + step.target_size_entries
                    )
                    assert all(
                        control.master.master is step.canvas_roi_toolbar
                        for control in roi_controls
                    ), "Step 1 ROI controls are not in the bottom canvas toolbar."
                    assert len({control.winfo_rooty() for control in roi_controls}) == 1, (
                        "Step 1 ROI controls are not aligned in one row."
                    )
                    view_buttons = (
                        step.source_view_radio,
                        step.target_view_radio,
                    )
                    assert all(
                        button.master.master.master is step.canvas_roi_toolbar
                        for button in view_buttons
                    ), "Step 1 view choices are not in the bottom canvas toolbar."
                if step_number == 4:
                    _assert_step4_build_stack_visible(step, width, height)
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

        app.tk.call("tk", "scaling", 2.0)
        app.geometry("1800x1000")
        app.update()
        dpi_scale = float(app.winfo_fpixels("1i")) / 96.0
        expected_scaled_sidebar = round(LAYOUT.sidebar_width * dpi_scale)
        for step_number, step in enumerate(steps, start=1):
            app.notebook.select(step_number - 1)
            step._sync_scaled_layout_values()
            step._apply_workspace_layout()
            app.update()
            assert abs(step.sidebar_shell.winfo_width() - expected_scaled_sidebar) <= 2, (
                f"Step {step_number} did not scale its sidebar for high DPI."
            )
            overflow = _sidebar_horizontal_overflow(step)
            assert not overflow, (
                f"Step {step_number} clips controls at high DPI: "
                + ", ".join(overflow[:5])
            )
    finally:
        app.destroy()

    print(f"UI_LAYOUT_OK {interface_mode}")
    for result in results:
        print(result)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        choices=("Modern", "Classic"),
        default="Modern",
    )
    options = parser.parse_args()
    raise SystemExit(main(options.interface))
