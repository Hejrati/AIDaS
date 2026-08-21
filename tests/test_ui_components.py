from __future__ import annotations

import inspect
from pathlib import Path
import unittest
from unittest import mock

from PIL import Image

from aidas.app import AboutDialog, SettingsDialog
from aidas.steps.step1_resize_raw import Step1Frame
from aidas.steps.step2_annotate import Step2BatchSegmentationSelectionPanel, Step2Frame
from aidas.steps.step3_flatten import (
    RBatchRunPanel,
    RBatchSelectionPanel,
    RBatchSelectionTable,
    RSetupWizard,
    Step3Frame,
)
from aidas.steps.step4_analyze_isez import Step4BatchROISelectionPanel, Step4Frame
from aidas.ui.components import AppButton, AppSplitButton, WorkflowHeader, WorkflowNavigation
from aidas.ui.theme import COLOR_PAIRS, CONTROLS, SHAPES
from aidas.utils.ui_utils import (
    ACTION_ICON_FILES,
    ACTION_ICON_SIZE,
    SidebarStepFrame,
    action_button,
    apply_app_icon_to,
    icon_action_button,
    load_action_icon,
)


PROJECT_ROOT = Path(__file__).parents[1]


class _StyleTarget:
    def __init__(self):
        self.options = {}

    def configure(self, **options):
        self.options.update(options)


class AppSplitButtonTests(unittest.TestCase):
    def test_dropdown_uses_the_standard_small_triangle_glyph(self):
        self.assertEqual(AppSplitButton.CHEVRON, "\u25be")
        self.assertEqual(AppSplitButton.CHEVRON_FONT_SIZE, 14)
        self.assertNotIn(AppSplitButton.CHEVRON, {"v", "V", "\u2304"})

    def test_split_button_uses_standard_large_control_height(self):
        signature = inspect.signature(AppSplitButton.__init__)
        self.assertEqual(signature.parameters["height"].default, CONTROLS.height_lg)
        self.assertEqual(AppSplitButton.DEFAULT_WIDTH, 198)
        self.assertEqual(AppSplitButton.SEGMENT_WIDTH, CONTROLS.height_lg)
        self.assertGreater(AppSplitButton.SEGMENT_WIDTH, CONTROLS.height_sm)

    def test_action_and_dropdown_keep_independent_commands(self):
        source = inspect.getsource(AppSplitButton.__init__)
        self.assertIn("command=command", source)
        self.assertIn("command=options_command", source)

    def test_segments_form_one_rounded_shape_with_a_square_join(self):
        source = inspect.getsource(AppSplitButton.__init__)
        self.assertEqual(source.count("background_corner_colors="), 2)
        self.assertIn('fg_color="transparent"', source)
        self.assertIn("self.divider.place(", source)
        self.assertNotIn("inner_radius", source)

    def test_step1_uses_shared_split_button_component(self):
        source = inspect.getsource(Step1Frame._build_controls)
        self.assertIn("AppSplitButton(", source)
        self.assertIn("self.crop_split_frame.action_button", source)
        self.assertIn("self.crop_split_frame.options_button", source)


class ActionButtonConventionTests(unittest.TestCase):
    def test_toplevel_icon_is_reapplied_after_ctk_initialization(self):
        source = inspect.getsource(apply_app_icon_to)

        self.assertIn('os.name == "nt"', source)
        self.assertLess(source.index("window.iconbitmap(ico)"), source.index("window.iconphoto(True, img)"))
        self.assertIn("window.after(250, apply_stored_icon)", source)

    def test_semantic_icons_share_one_color_icon_family(self):
        self.assertGreaterEqual(len(ACTION_ICON_FILES), 10)
        self.assertTrue(all(name.startswith("flat-color-icons--") for name in ACTION_ICON_FILES.values()))
        for filename in ACTION_ICON_FILES.values():
            with self.subTest(filename=filename):
                self.assertTrue((PROJECT_ROOT / "assets" / filename).is_file())

        self.assertEqual(
            ACTION_ICON_FILES["download"],
            "flat-color-icons--download.png",
        )

    def test_text_and_icon_only_helpers_share_one_icon_size(self):
        text_source = inspect.getsource(action_button)
        icon_source = inspect.getsource(icon_action_button)

        self.assertEqual(ACTION_ICON_SIZE, 20)
        self.assertIn('"compound": "left"', text_source)
        self.assertIn('"image": (icon, "disabled", icon)', text_source)
        self.assertIn("_aidas_disabled_action_icon", text_source)
        self.assertIn('"AIDaS.Icon.TButton"', icon_source)
        self.assertIn('setdefault("compound", "image")', icon_source)

    def test_close_and_cancel_actions_use_the_shared_x_badge(self):
        source = inspect.getsource(load_action_icon)

        self.assertIn('action in {"cancel", "close"}', source)
        self.assertIn("load_color_close_icon(owner, size=size)", source)

        settings_source = inspect.getsource(SettingsDialog.__init__)
        self.assertIn("load_color_close_ctk_icon(self)", settings_source)

    def test_ctk_buttons_default_to_left_compound_when_they_have_icons(self):
        source = inspect.getsource(AppButton.__init__)
        self.assertIn('kwargs.setdefault("compound", "left")', source)

    def test_disabled_ctk_buttons_do_not_keep_actionable_semantic_colors(self):
        constructor_source = inspect.getsource(AppButton.__init__)
        configure_source = inspect.getsource(AppButton.configure)

        self.assertIn('options.get("state") == "disabled"', constructor_source)
        self.assertIn("self._disabled_palette()", constructor_source)
        self.assertIn('target_state == "disabled"', configure_source)
        self.assertIn("self._enabled_palette", configure_source)
        disabled_source = inspect.getsource(AppButton._disabled_palette)
        self.assertIn("_DISABLED_PALETTE", disabled_source)
        self.assertIn('"background_corner_colors"', disabled_source)
        self.assertIn("disabled_fill if corner == enabled_fill", disabled_source)
        self.assertEqual(AppButton._DISABLED_PALETTE["border_width"], SHAPES.border_width)


class ResponsiveWorkflowPanelTests(unittest.TestCase):
    def test_about_dialog_has_no_scrollable_container(self):
        source = inspect.getsource(AboutDialog.__init__)

        self.assertIn("content = ctk.CTkFrame(", source)
        self.assertNotIn("CTkScrollableFrame", source)
        self.assertIn("self.resizable(False, False)", source)
        self.assertNotIn("scrollbar_button_color", source)

    def test_about_dialog_uses_ok_and_has_a_native_classic_surface(self):
        constructor_source = inspect.getsource(AboutDialog.__init__)
        classic_source = inspect.getsource(AboutDialog._build_classic_about)

        self.assertIn('text="OK"', constructor_source)
        self.assertNotIn('text="Close"', constructor_source)
        self.assertNotIn("load_color_close_ctk_icon", constructor_source)
        self.assertNotIn("CustomTkinter {ctk.__version__}", constructor_source)
        self.assertNotIn("Â", constructor_source)
        self.assertIn('f"{APP_SUBTITLE} - Version {__version__}"', constructor_source)
        self.assertIn('self._presentation_mode == "Classic"', constructor_source)
        self.assertIn("ttk.Frame(", classic_source)
        self.assertIn('text="OK"', classic_source)
        self.assertNotIn("AppButton(", classic_source)
        self.assertNotIn("CustomTkinter {ctk.__version__}", classic_source)
        self.assertNotIn("Â", classic_source)
        self.assertIn('f"{APP_SUBTITLE} - Version {__version__}"', classic_source)

    def test_settings_owns_sdb_r_setup_and_script_configuration(self):
        source = inspect.getsource(SettingsDialog.__init__)

        self.assertIn('"Default SDB image parameters"', source)
        self.assertIn('"R environment"', source)
        self.assertIn('"Step 3 R scripts"', source)
        self.assertIn("self._open_r_setup", source)
        self.assertIn("self._refresh_r_script_choices(role)", source)
        self.assertIn('text="Apply"', source)
        self.assertNotIn('text="Save SDB defaults"', source)
        apply_source = inspect.getsource(SettingsDialog._apply_changes)
        self.assertIn('self._preferences.set("sdb_raw_width"', apply_source)
        self.assertIn("self._step3.select_r_script", apply_source)
        self.assertIn("self._set_appearance_command", apply_source)

    def test_classic_settings_uses_native_property_sheet_controls(self):
        source = inspect.getsource(SettingsDialog._build_classic_settings)

        self.assertIn("ttk.LabelFrame(", source)
        self.assertIn("ttk.Combobox(", source)
        self.assertIn("ttk.Checkbutton(", source)
        self.assertIn("ttk.Entry(", source)
        self.assertIn("action_button(", source)
        self.assertNotIn("ctk.CTkOptionMenu(", source)
        self.assertNotIn("ctk.CTkSwitch(", source)
        self.assertIn("base_row = 1 + row * 3", source)

    def test_classic_settings_appearance_state_uses_native_readonly_mode(self):
        dialog = SettingsDialog.__new__(SettingsDialog)
        dialog._classic_settings = True
        dialog.interface_menu = mock.Mock()
        dialog.appearance_menu = mock.Mock()

        dialog._sync_interface_controls("Classic", "Dark")
        dialog.appearance_menu.configure.assert_called_with(state="disabled")

        dialog.appearance_menu.reset_mock()
        dialog._sync_interface_controls("Modern", "Dark")
        dialog.appearance_menu.configure.assert_called_with(state="readonly")

    def test_settings_compacts_long_r_script_names(self):
        filename = "RAW_OCT_PROCESSING_2023_09SEP-05_WSU_noHypoDenseBand_EA_edited.R"
        compact = SettingsDialog._compact_script_name(filename)

        self.assertLessEqual(len(compact), 48)
        self.assertIn("…", compact)
        self.assertTrue(compact.endswith("edited.R"))

    def test_step3_sidebar_restores_r_setup_without_script_configuration(self):
        source = inspect.getsource(Step3Frame._build_ui)

        self.assertNotIn('add_sidebar_section("R Scripts"', source)
        self.assertIn('"Set up R and packages…"', source)
        self.assertIn("self._open_r_setup_wizard", source)
        self.assertNotIn("_build_r_script_selector", source)
        selection_source = inspect.getsource(Step3Frame._selected_r_script_path)
        self.assertIn('"r_main_script_path"', selection_source)
        self.assertIn('"r_output_script_path"', selection_source)

    def test_step3_tutorial_has_light_and_dark_versions(self):
        render_source = inspect.getsource(Step3Frame._render_tutorial)
        conversion_source = inspect.getsource(Step3Frame._tutorial_image_for_appearance)
        refresh_source = inspect.getsource(Step3Frame.refresh_appearance)

        self.assertIn("_tutorial_image_for_appearance", render_source)
        self.assertIn('COLOR_PAIRS["surface"]', render_source)
        self.assertIn('resolve_color(COLOR_PAIRS["text"], "Dark")', conversion_source)
        self.assertIn("neutral_result", conversion_source)
        self.assertIn("pale_colored", conversion_source)
        self.assertIn("self._render()", refresh_source)

    def test_step3_view_changes_do_not_refresh_the_tutorial(self):
        source = inspect.getsource(Step3Frame._on_view_selected)

        self.assertIn("elif self.results is not None:", source)
        self.assertNotIn("else:\n            self._render()", source)

    def test_step1_reset_uses_configurable_sdb_defaults(self):
        source = inspect.getsource(Step1Frame._set_default_import_params)

        self.assertIn("self.default_raw_width", source)
        self.assertIn("self.default_raw_height", source)
        self.assertIn("self.default_raw_offset", source)

    def test_step1_exclude_saved_filter_is_above_the_directory_buttons(self):
        source = inspect.getsource(Step1Frame._build_controls)

        header_start = source.index("parent_dir_header = ttk.Frame(sdb)")
        parent_label = source.index('text="Parent dir:"')
        exclude_filter = source.index("self.exclude_saved_outputs_checkbox")
        directory_controls = source.index("dir_frame, _dir_entry, dir_buttons")
        header_source = source[header_start:directory_controls]
        self.assertLess(parent_label, exclude_filter)
        self.assertLess(exclude_filter, directory_controls)
        self.assertIn('ttk.Label(parent_dir_header, text="Parent dir:")', header_source)
        self.assertIn("parent_dir_header,\n            text=", header_source)
        self.assertIn('pack(side="left", padx=(4, 0))', header_source)

    def test_step1_view_selectors_blend_into_the_toolbar(self):
        source = inspect.getsource(Step1Frame._build_controls)
        self.assertEqual(source.count('style="AIDaS.ContentHeader.TRadiobutton"'), 2)

    def test_step1_undo_is_primary_when_enabled(self):
        source = inspect.getsource(Step1Frame._build_controls)
        undo_section = source[source.index("self.undo_crop_btn = AppButton(") :]
        self.assertIn('variant="primary"', undo_section.split("self.undo_crop_btn.pack", 1)[0])

    def test_workflow_handoff_buttons_share_the_success_variant(self):
        step1_source = inspect.getsource(Step1Frame._build_controls)
        step2_source = inspect.getsource(Step2Frame._build_controls)
        step3_source = inspect.getsource(Step3Frame._build_ui)

        step1_handoff = step1_source[step1_source.index("self.batch_segment_cropped_btn = AppButton(") :]
        step2_handoff = step2_source[step2_source.index("self.continue_to_step3_button = AppButton(") :]
        step3_handoff = step3_source[step3_source.index("self.continue_to_step4_button = AppButton(") :]
        self.assertIn('variant="success"', step1_handoff.split(".pack(", 1)[0])
        self.assertIn('variant="success"', step2_handoff.split(".grid(", 1)[0])
        self.assertIn('variant="success"', step3_handoff.split(".pack(", 1)[0])
        self.assertIn('"flat-color-icons--right.png"', step1_source)
        self.assertIn('"flat-color-icons--right.png"', step2_source)
        self.assertIn('"flat-color-icons--right.png"', step3_source)

    def test_fovea_prompt_uses_rounded_buttons_and_dpi_aware_icons(self):
        source = inspect.getsource(Step2Frame._collect_folder_fovea_lines)

        self.assertEqual(source.count("= AppButton("), 3)
        self.assertIn('variant="success"', source)
        self.assertIn('tint=COLOR_PAIRS["on_primary"]', source)
        self.assertEqual(source.count("load_ctk_image("), 2)
        self.assertIn("load_color_close_ctk_icon(self, size=20)", source)
        self.assertNotIn("btn_cancel = action_button(", source)
        self.assertNotIn("btn_skip = action_button(", source)
        self.assertNotIn("btn_set = action_button(", source)

    def test_fovea_prompt_reserves_buttons_before_the_long_path(self):
        source = inspect.getsource(Step2Frame._collect_folder_fovea_lines)

        actions_pack = source.index('actions_frame.pack(side="right", fill="y")')
        prompt_pack = source.index(
            'prompt_label.pack(side="left", fill="x", expand=True)'
        )
        self.assertLess(actions_pack, prompt_pack)
        for button_name in ("btn_cancel", "btn_skip", "btn_set"):
            button_start = source.index(f"{button_name} = AppButton(")
            button_pack = source.index(f"{button_name}.pack(", button_start)
            self.assertIn("actions_frame,", source[button_start:button_pack])
            self.assertLess(button_pack, prompt_pack)
        self.assertIn("width=1", source)
        self.assertIn('anchor="w"', source)
        self.assertIn("prompt_tooltip.text = msg", source)

    def test_step3_r_download_uses_the_download_action(self):
        source = inspect.getsource(RSetupWizard._render_page)
        start = source.index("self.r_auto_install_button = action_button(")
        block = source[start : source.index("self.r_auto_install_button.pack", start)]

        self.assertIn('"download"', block)
        self.assertNotIn('"package"', block)

    def test_step3_batch_limits_use_large_numeric_steppers_and_core_wording(self):
        source = inspect.getsource(RBatchSelectionPanel._build_ui)

        self.assertEqual(source.count("NativeNumericSpinbox("), 2)
        self.assertNotIn("ttk.Spinbox(", source)
        self.assertIn('text="Batch size:"', source)
        self.assertIn('text="Timeout per script (min):"', source)

    def test_step2_exposes_the_maximum_fallback_core_limit(self):
        source = inspect.getsource(Step2BatchSegmentationSelectionPanel._build_ui)

        self.assertIn("NativeNumericSpinbox(", source)
        self.assertIn("Maximum Step 2 cores for GPU fallback", source)
        self.assertIn("_shared_core_budget()", source)
        self.assertIn("ai_device_status_var", source)
        self.assertNotIn("CPU reserved", source)

    def test_batch_panels_reserve_the_footer_before_the_flexible_table(self):
        panel_footers = (
            (Step2BatchSegmentationSelectionPanel, "run_box"),
            (RBatchSelectionPanel, "run_box"),
            (RBatchRunPanel, "summary_row"),
            (Step4BatchROISelectionPanel, "run_box"),
        )
        for panel_class, footer_name in panel_footers:
            with self.subTest(panel=panel_class.__name__):
                source = inspect.getsource(panel_class._build_ui)
                self.assertIn(f'{footer_name}.pack(side="bottom", fill="x"', source)
                footer_assignment = f"self.action_footer = {footer_name}"
                self.assertIn(footer_assignment, source)
                self.assertLess(source.index(footer_assignment), source.index("self.table_host ="))

    def test_step3_run_footer_reserves_horizontal_space_for_actions(self):
        source = inspect.getsource(RBatchRunPanel._build_ui)

        self.assertLess(
            source.index('action_box.pack(side="right"'),
            source.index('self.summary_label.pack(side="left"'),
        )
        self.assertIn('summary_row.bind("<Configure>", self._resize_summary_footer', source)

    def test_step3_batch_table_only_shows_horizontal_scroll_for_overflow(self):
        table = RBatchSelectionTable.__new__(RBatchSelectionTable)
        table.xscroll = mock.Mock()
        table.tree = mock.Mock()
        table._xscroll_visible = False
        table._xscroll_after_id = None
        table._manual_column_widths = None
        table._horizontal_chrome_width = 4

        # Tk's fractional callback controls only the thumb and cannot reveal
        # the bar while automatic sizing is settling.
        table._on_xscroll("0.0", "1.0")
        table.xscroll.set.assert_called_once_with("0.0", "1.0")
        table.xscroll.grid.assert_not_called()
        table._on_xscroll("0.0", "0.75")
        table.xscroll.grid.assert_not_called()

        widths = {"#0": 40, "folder": 800, "status": 218, "inputs": 66}
        table.tree.column.side_effect = lambda column, option: widths[column]
        table.tree.winfo_width.return_value = 1000

        # Automatic path overflow is clipped inside Folder without a scrollbar.
        table._sync_xscroll_visibility()
        table.xscroll.grid.assert_not_called()

        # A real user-resize baseline enables the bar from integer width math.
        table._manual_column_widths = dict(widths)
        table._sync_xscroll_visibility()
        table.xscroll.grid.assert_called_once_with(row=1, column=0, sticky="ew")

        # Exact fit (996 usable pixels plus four pixels of chrome) hides it.
        widths["folder"] = 672
        table._sync_xscroll_visibility()
        table.xscroll.grid_remove.assert_called_once_with()
        table.tree.xview_moveto.assert_called_once_with(0.0)

    def test_step3_batch_table_caches_theme_chrome_before_manual_overflow(self):
        table = RBatchSelectionTable.__new__(RBatchSelectionTable)
        table.tree = mock.Mock()
        table._manual_column_widths = None
        table._horizontal_chrome_width = None
        widths = {"#0": 40, "folder": 672, "status": 218, "inputs": 66}
        table.tree.column.side_effect = lambda column, option: widths[column]
        table.tree.winfo_reqwidth.return_value = 1000

        self.assertEqual(table._tree_horizontal_chrome_width(), 4)

        # During overflow Tk may clamp reqwidth, but the calibrated value stays.
        table._manual_column_widths = dict(widths)
        widths["folder"] += 300
        self.assertEqual(table._tree_horizontal_chrome_width(), 4)

    def test_step3_batch_table_uses_viewport_not_folder_text_for_default_width(self):
        fit_source = inspect.getsource(RBatchSelectionTable._fit_columns_to_content)
        folder_source = inspect.getsource(RBatchSelectionTable._expand_folder_to_view)

        self.assertNotIn("_measure_text(folder)", fit_source)
        self.assertIn("_folder, status, inputs", fit_source)
        self.assertIn('self.tree.heading("status", "text")', fit_source)
        self.assertIn('self.tree.heading("inputs", "text")', fit_source)
        self.assertIn("self._status_width_values", fit_source)
        self.assertIn("self.MAX_PROGRESS_VALUE", fit_source)
        self.assertIn("padding=self.HEADING_WIDTH_PADDING", fit_source)
        self.assertGreaterEqual(RBatchSelectionTable.HEADING_WIDTH_PADDING, 30)
        self.assertIn("view_width - non_folder_width - chrome_width", folder_source)
        self.assertIn('self.tree.column("folder", width=desired_folder_width)', folder_source)

    def test_step3_batch_table_preserves_user_column_resizing(self):
        finish_source = inspect.getsource(RBatchSelectionTable._finish_column_resize)
        release_source = inspect.getsource(RBatchSelectionTable._on_tree_button_release)
        fit_source = inspect.getsource(RBatchSelectionTable._fit_columns_to_content)
        update_source = inspect.getsource(RBatchRunPanel.update_folder)

        self.assertIn("self.after_idle(self._finish_column_resize)", release_source)
        self.assertIn("current_widths != start_widths", finish_source)
        self.assertIn("self._manual_column_widths = current_widths", finish_source)
        self.assertIn("if self._manual_column_widths is not None:", fit_source)
        self.assertNotIn("_fit_columns_to_content", update_source)

    def test_step3_status_width_samples_cover_every_r_progress_label(self):
        progress_labels = {
            label for _percent, label in Step3Frame.R_PROGRESS_BY_STEP.values()
        }

        self.assertTrue(
            progress_labels.issubset(set(RBatchSelectionTable.STATUS_WIDTH_VALUES))
        )
        run_source = inspect.getsource(RBatchRunPanel._build_ui)
        self.assertIn("RBatchSelectionTable.RUN_STATUS_WIDTH_VALUES", run_source)

    def test_step3_run_header_and_action_glyphs_are_compact_and_legible(self):
        source = inspect.getsource(RBatchRunPanel._build_ui)

        self.assertNotIn('f"Timeout:', source)
        self.assertNotIn('"Progress and logs update', source)
        self.assertIn('text="\\u25a0  Stop"', source)
        self.assertIn("self.close_icon = load_color_close_icon(self)", source)
        self.assertIn("image=self.close_icon", source)
        self.assertIn('text="Close"', source)
        self.assertIn('style="AIDaS.DangerAction.TButton"', source)

    def test_step3_second_script_schedule_defaults_to_parallel(self):
        source = inspect.getsource(RBatchSelectionPanel._build_ui)

        self.assertIn(
            "tk.StringVar(value=self.step_frame.R_OUTPUT_MODE_PARALLEL)",
            source,
        )
        self.assertIn('text="Parallel (default)"', source)
        self.assertIn('text="Sequential"', source)

    def test_step3_run_summary_wraps_to_the_space_left_by_actions(self):
        panel = RBatchRunPanel.__new__(RBatchRunPanel)
        panel.action_box = mock.Mock()
        panel.action_box.winfo_reqwidth.return_value = 260
        panel.summary_label = mock.Mock()
        panel.summary_label.cget.return_value = 480

        panel._resize_summary_footer(type("Event", (), {"width": 520})())

        panel.summary_label.configure.assert_called_once_with(wraplength=244)

        panel.summary_label.reset_mock()
        panel._resize_summary_footer(type("Event", (), {"width": 300})())
        panel.summary_label.configure.assert_called_once_with(wraplength=120)

    def test_step2_reserves_status_space_through_the_shared_layout(self):
        source = inspect.getsource(Step2Frame.__init__)
        self.assertIn("status_var=self.status_var", source)
        self.assertIn("status_bar_content_margin=True", source)
        self.assertLess(source.index("self.status_var ="), source.index("self.build_standard_layout("))

    def test_step2_step3_and_step4_status_bars_align_to_their_content_margins(self):
        for builder in (Step2Frame.__init__, Step3Frame._build_ui, Step4Frame._build_ui):
            with self.subTest(builder=builder.__qualname__):
                source = inspect.getsource(builder)
                self.assertIn("status_bar_content_margin=True", source)

        layout_source = inspect.getsource(SidebarStepFrame.build_standard_layout)
        self.assertIn("(content_padding[0], content_padding[2])", layout_source)

    def test_workflow_status_text_has_an_internal_left_inset(self):
        source = inspect.getsource(SidebarStepFrame.add_status_bar)

        self.assertIn("padx=LAYOUT.space_sm", source)


class WorkflowNavigationTests(unittest.TestCase):
    def test_header_keeps_only_settings_and_help_shortcuts_at_top_right(self):
        source = inspect.getsource(WorkflowHeader.__init__)

        settings_index = source.index("self.settings_button = ctk.CTkButton")
        help_index = source.index("self.help_button = ctk.CTkButton")
        navigation_index = source.index("self.navigation = WorkflowNavigation")
        self.assertLess(settings_index, help_index)
        self.assertLess(help_index, navigation_index)
        self.assertNotIn("self.appearance_menu", source)
        self.assertNotIn('text="Appearance"', source)
        self.assertIn("image=self.settings_image", source)
        self.assertIn("image=self.help_image", source)
        settings_source = source[settings_index:help_index]
        help_source = source[help_index:navigation_index]
        self.assertEqual(settings_source.count('anchor="center"'), 1)
        self.assertEqual(help_source.count('anchor="center"'), 1)
        self.assertIn("size=(24, 24)", source)
        self.assertIn('text_color=COLOR_PAIRS["primary"]', source)
        self.assertGreaterEqual(source.count("width=36"), 2)
        self.assertGreaterEqual(source.count("height=36"), 2)

    def test_header_iconify_assets_are_high_resolution(self):
        filenames = (
            "iconify-fluent-color--settings-32.png",
            "iconify-fluent-color--question-circle-32.png",
        )
        for filename in filenames:
            icon_path = PROJECT_ROOT / "assets" / filename
            with self.subTest(filename=filename), Image.open(icon_path) as icon:
                self.assertEqual(icon.size, (256, 256))
                self.assertEqual(icon.mode, "RGBA")

    def test_header_uses_independent_navigation_buttons(self):
        source = inspect.getsource(WorkflowHeader.__init__)

        self.assertIn("WorkflowNavigation(", source)
        self.assertNotIn("CTkSegmentedButton(", source)
        self.assertGreater(WorkflowNavigation.GAP, 0)

    def test_navigation_container_has_no_connecting_background(self):
        source = inspect.getsource(WorkflowNavigation.__init__)

        self.assertIn('fg_color="transparent"', source)
        self.assertIn("border_width=0", source)
        self.assertIn("minsize=self.GAP", source)

    def test_selected_and_inactive_buttons_have_distinct_semantic_styles(self):
        selected = _StyleTarget()
        inactive = _StyleTarget()

        WorkflowNavigation._style_button(selected, selected=True)
        WorkflowNavigation._style_button(inactive, selected=False)

        self.assertEqual(selected.options["fg_color"], COLOR_PAIRS["primary"])
        self.assertEqual(selected.options["text_color"], COLOR_PAIRS["on_primary"])
        self.assertEqual(inactive.options["fg_color"], COLOR_PAIRS["button"])
        self.assertEqual(inactive.options["border_color"], COLOR_PAIRS["border_strong"])
        self.assertNotEqual(selected.options["fg_color"], inactive.options["fg_color"])


if __name__ == "__main__":
    unittest.main()
