"""Fresh interpreters verify startup imports rather than test-suite import order."""
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class StartupImportTests(unittest.TestCase):
    def run_isolated(self, script):
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(script)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_step_package_loads_only_the_requested_frame(self):
        self.run_isolated('''
            import sys
            import aidas.steps as steps
            assert not any(name.startswith('aidas.steps.step') for name in sys.modules)
            from aidas.steps import Step1Frame
            from aidas.steps.step1_resize_raw import Step1Frame as direct
            assert Step1Frame is direct
            assert steps.Step1Frame is direct
            assert 'aidas.steps.step3_flatten' not in sys.modules
            assert 'aidas.steps.step4_analyze_isez' not in sys.modules
            try:
                steps.nonexistent
            except AttributeError:
                pass
            else:
                raise AssertionError('Unknown package attributes must still fail.')
        ''')

    def test_empty_workflow_imports_do_not_load_analysis_libraries(self):
        self.run_isolated('''
            import sys
            import aidas.app
            from aidas.steps import Step1Frame, Step2Frame, Step3Frame, Step5Frame
            from aidas.steps.step4_analyze_isez import Step4Frame
            for name in ('pyreadr', 'pandas', 'matplotlib', 'scipy'):
                assert name not in sys.modules, name
        ''')

    def test_startup_builds_all_frames_and_releases_deferred_artwork(self):
        self.run_isolated('''
            import json
            import sys
            import tempfile
            from pathlib import Path
            from unittest.mock import patch
            import customtkinter as ctk
            from aidas.app import AIDaSApp
            from aidas.core.config import Config
            from aidas.steps.step2_annotate import Step2Frame

            original_root = ctk.CTk.__init__
            original_window = ctk.CTkToplevel.__init__
            def root_init(self, *args, **kwargs):
                original_root(self, *args, **kwargs)
                self.attributes('-alpha', 0)
            def window_init(self, *args, **kwargs):
                original_window(self, *args, **kwargs)
                self.attributes('-alpha', 0)

            with tempfile.TemporaryDirectory() as directory:
                Config.CONFIG_DIR = Path(directory)
                Config.CONFIG_FILE = Path(directory) / 'preferences.json'
                for mode in ('Modern', 'Classic'):
                    Config.CONFIG_FILE.write_text(json.dumps({
                        'interface_mode': mode, 'check_for_updates': False}))
                    with patch.object(ctk.CTk, '__init__', root_init), patch.object(
                            ctk.CTkToplevel, '__init__', window_init), patch.object(
                            Step2Frame, '_start_ai_device_probe'):
                        app = AIDaSApp()
                    try:
                        app._finish_startup()
                        app.update()
                        assert not app._aidas_defer_canvas_updates
                        assert not app._aidas_deferred_surfaces
                        assert len(app.notebook.tabs()) == 5
                        for index in range(5):
                            app.notebook.select(index)
                            app.update()
                            assert getattr(app, f'step{index + 1}').winfo_viewable()
                        for name in ('pyreadr', 'pandas', 'matplotlib', 'scipy'):
                            assert name not in sys.modules, name
                    finally:
                        app.destroy()
        ''')


if __name__ == '__main__':
    unittest.main()
