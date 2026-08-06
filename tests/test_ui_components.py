from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from aidas.steps.step1_resize_raw import Step1Frame
from aidas.steps.step2_annotate import Step2BatchSegmentationSelectionPanel, Step2Frame
from aidas.steps.step3_flatten import RBatchSelectionPanel
from aidas.steps.step4_analyze_isez import Step4BatchROISelectionPanel
from aidas.ui.components import AppButton, AppSplitButton, WorkflowHeader, WorkflowNavigation
from aidas.ui.theme import COLOR_PAIRS, CONTROLS
from aidas.utils.ui_utils import ACTION_ICON_FILES, ACTION_ICON_SIZE, action_button, icon_action_button


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
    def test_semantic_icons_share_one_color_icon_family(self):
        self.assertGreaterEqual(len(ACTION_ICON_FILES), 10)
        self.assertTrue(all(name.startswith("flat-color-icons--") for name in ACTION_ICON_FILES.values()))
        for filename in ACTION_ICON_FILES.values():
            with self.subTest(filename=filename):
                self.assertTrue((PROJECT_ROOT / "assets" / filename).is_file())

    def test_text_and_icon_only_helpers_share_one_icon_size(self):
        text_source = inspect.getsource(action_button)
        icon_source = inspect.getsource(icon_action_button)

        self.assertEqual(ACTION_ICON_SIZE, 20)
        self.assertIn('"compound": "left"', text_source)
        self.assertIn('"AIDaS.Icon.TButton"', icon_source)

    def test_ctk_buttons_default_to_left_compound_when_they_have_icons(self):
        source = inspect.getsource(AppButton.__init__)
        self.assertIn('kwargs.setdefault("compound", "left")', source)


class ResponsiveWorkflowPanelTests(unittest.TestCase):
    def test_batch_panels_reserve_the_footer_before_the_flexible_table(self):
        panel_classes = (
            Step2BatchSegmentationSelectionPanel,
            RBatchSelectionPanel,
            Step4BatchROISelectionPanel,
        )
        for panel_class in panel_classes:
            with self.subTest(panel=panel_class.__name__):
                source = inspect.getsource(panel_class._build_ui)
                self.assertIn('run_box.pack(side="bottom", fill="x"', source)
                self.assertIn("self.action_footer = run_box", source)
                self.assertLess(source.index("self.action_footer = run_box"), source.index("self.table_host ="))

    def test_step2_reserves_status_space_through_the_shared_layout(self):
        source = inspect.getsource(Step2Frame.__init__)
        self.assertIn("self.build_standard_layout(status_var=self.status_var)", source)
        self.assertLess(source.index("self.status_var ="), source.index("self.build_standard_layout("))


class WorkflowNavigationTests(unittest.TestCase):
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
        self.assertEqual(inactive.options["fg_color"], COLOR_PAIRS["surface_subtle"])
        self.assertEqual(inactive.options["border_color"], COLOR_PAIRS["border_strong"])
        self.assertNotEqual(selected.options["fg_color"], inactive.options["fg_color"])


if __name__ == "__main__":
    unittest.main()
