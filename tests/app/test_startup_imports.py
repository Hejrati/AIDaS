"""Fresh interpreters verify startup imports rather than test-suite import order."""

import subprocess
import sys
import textwrap
import unittest

from tests.support.paths import PROJECT_ROOT


class StartupImportTests(unittest.TestCase):
    def run_isolated(self, script):
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(script)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
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
            from aidas.steps.step3_flatten import Step3Frame
            from aidas.ui.splash import SplashWindow

            original_root = ctk.CTk.__init__
            original_window = ctk.CTkToplevel.__init__
            original_build = AIDaSApp._build_application
            original_progress = AIDaSApp._set_splash_progress
            original_tutorial = Step3Frame._render_tutorial
            stages = []
            exposed = []
            tutorials = []
            unmapped = []
            def tutorial(self):
                tutorials.append(self)
                original_tutorial(self)
            def root_init(self, *args, **kwargs):
                original_root(self, *args, **kwargs)
                self.attributes('-alpha', 0)
            def window_init(self, *args, **kwargs):
                original_window(self, *args, **kwargs)
                # Verify an opaque splash during construction. Hiding every
                # toplevel masked the regression in native window processing.
                if not isinstance(self, SplashWindow):
                    self.attributes('-alpha', 0)
                else:
                    self.bind('<Expose>', lambda event: exposed.append(True), add='+')
            def assert_visible_splash(app):
                splash = app._splash
                assert splash.winfo_viewable(), 'Splash is hidden during loading'
                assert splash.status_label.winfo_viewable(), 'Progress is hidden'
                assert float(splash.attributes('-alpha')) == 1.0
                assert not app.winfo_viewable(), 'Main window appeared too early'
            def build(self):
                assert_visible_splash(self)
                assert exposed, 'Splash received no paint events before loading'
                splash = self._splash
                def on_unmap(event):
                    if event.widget is splash:
                        unmapped.append(splash.percent_var.get())
                splash.bind('<Unmap>', on_unmap, add='+')
                original_build(self)
            def progress(self, value, message):
                dispatched = []
                # A real event callback must run before construction resumes;
                # idle layout/paint processing alone cannot guarantee this.
                self.after(0, lambda: dispatched.append(True))
                original_progress(self, value, message)
                assert dispatched, 'Startup starved native events and timers'
                assert_visible_splash(self)
                assert self._splash.status_var.get() == message
                assert self._splash.percent_var.get() == f'{value:.0f}%'
                stages.append(value)

            with tempfile.TemporaryDirectory() as directory:
                Config.CONFIG_DIR = Path(directory)
                Config.CONFIG_FILE = Path(directory) / 'preferences.json'
                for mode in ('Modern', 'Classic'):
                    stages.clear()
                    exposed.clear()
                    tutorials.clear()
                    unmapped.clear()
                    Config.CONFIG_FILE.write_text(json.dumps({
                        'interface_mode': mode, 'check_for_updates': False}))
                    with patch.object(ctk.CTk, '__init__', root_init), patch.object(
                            ctk.CTkToplevel, '__init__', window_init), patch.object(
                            Step2Frame, '_start_ai_device_probe'), patch.object(
                            AIDaSApp, '_build_application', build), patch.object(
                            AIDaSApp, '_set_splash_progress', progress), patch.object(
                            Step3Frame, '_render_tutorial', tutorial):
                        app = AIDaSApp()
                    try:
                        assert stages[0] == 3 and stages[-1] == 100
                        assert len(stages) > 5, 'Loading progress did not advance'
                        assert tutorials == [app.step3], 'Tutorial rendered repeatedly'
                        assert not unmapped, (mode, 'Splash blinked while loading', unmapped)
                        launch_checks = []
                        original_after = app.after
                        def record_after(delay, callback=None, *args):
                            if callback == app.update_controller.check_automatically:
                                assert app.winfo_viewable(), 'Update check queued before window reveal'
                                launch_checks.append(delay)
                            return original_after(delay, callback, *args)
                        with patch.object(app, 'after', side_effect=record_after):
                            app._finish_startup()
                            app._finish_startup()
                        assert launch_checks == [1500], (mode, launch_checks)
                        app.update()
                        assert not app._aidas_defer_canvas_updates
                        assert not app._aidas_deferred_surfaces
                        assert len(app.notebook.tabs()) == 5
                        for index in range(5):
                            app.notebook.select(index)
                            app.update()
                            assert getattr(app, f'step{index + 1}').winfo_viewable(), (mode, index)
                        for name in ('pyreadr', 'pandas', 'matplotlib', 'scipy'):
                            assert name not in sys.modules, name
                    finally:
                        app.destroy()
        ''')


if __name__ == '__main__':
    unittest.main()
