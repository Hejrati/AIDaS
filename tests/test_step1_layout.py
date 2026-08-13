import inspect
import unittest

from aidas.steps.step1_resize_raw import Step1Frame


class Step1LayoutTests(unittest.TestCase):
    def test_save_and_handoff_actions_are_in_a_fixed_sidebar_footer(self):
        source = inspect.getsource(Step1Frame._build_controls)

        footer_start = source.index("self.step_actions_footer = ttk.Frame(")
        actions_start = source.index("self.step_actions_frame = ttk.Frame(")
        footer_source = source[footer_start:actions_start]
        actions_source = source[actions_start:]

        self.assertIn("self.sidebar_shell,", footer_source)
        self.assertIn('side="bottom"', footer_source)
        self.assertIn("before=self.sidebar", footer_source)
        self.assertIn("self.step_actions_footer,", actions_source)
        self.assertNotIn("self.step_actions_frame = ttk.Frame(self.ctrl)", source)


if __name__ == "__main__":
    unittest.main()
