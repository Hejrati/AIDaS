from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "aidas"


class PackageStructureTests(unittest.TestCase):
    def test_runtime_modules_are_grouped_by_responsibility(self):
        modules = (
            "aidas.ai.inference",
            "aidas.ai.client",
            "aidas.ai.torch_model",
            "aidas.ai.worker",
            "aidas.canvas.image_canvas",
            "aidas.core.config",
            "aidas.core.single_instance",
            "aidas.services.update_service",
            "aidas.services.update_ui",
            "aidas.ui.classic",
            "aidas.ui.components",
            "aidas.ui.menu_bar",
            "aidas.ui.splash",
            "aidas.ui.tabs",
            "aidas.ui.theme",
            "aidas.ui.title_bar",
            "aidas.ui.windowing",
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.util.find_spec(module_name))

    def test_package_root_contains_only_composition_modules(self):
        root_modules = {path.name for path in PACKAGE_ROOT.glob("*.py")}
        self.assertEqual(root_modules, {"__init__.py", "app.py"})

    def test_pyinstaller_uses_the_new_ai_module_names(self):
        spec_text = (PROJECT_ROOT / "AIDaS.spec").read_text(encoding="utf-8")
        self.assertIn('"aidas.ai.worker"', spec_text)
        self.assertIn('"aidas.ai.inference"', spec_text)
        self.assertNotIn("aidas.ai_for_aidas_", spec_text)

    def test_pyinstaller_uses_onedir_to_avoid_runtime_extraction(self):
        spec_text = (PROJECT_ROOT / "AIDaS.spec").read_text(encoding="utf-8")
        self.assertIn("exclude_binaries=True", spec_text)
        self.assertIn("collect = COLLECT(", spec_text)

    def test_customtkinter_dependency_and_theme_data_are_packaged(self):
        spec_text = (PROJECT_ROOT / "AIDaS.spec").read_text(encoding="utf-8")
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("customtkinter>=6,<7", requirements.lower())
        self.assertIn("collect_data_files", spec_text)
        self.assertIn('collect_data_files("customtkinter")', spec_text)
        self.assertIn('"customtkinter",', spec_text)

    def test_application_chrome_supports_modern_and_classic_menu_bars(self):
        app_source = (PROJECT_ROOT / "aidas" / "app.py").read_text(encoding="utf-8")
        menu_source = (PROJECT_ROOT / "aidas" / "ui" / "menu_bar.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("ApplicationMenuBar", app_source)
        self.assertIn("build_classic_application_menu", app_source)
        self.assertNotIn("build_app_menu", app_source)
        self.assertIn("class _PopupMenu(tk.Toplevel)", menu_source)
        self.assertNotIn("class _PopupMenu(ctk.CTkToplevel)", menu_source)

    def test_main_window_uses_captionless_native_frame_component(self):
        app_source = (PROJECT_ROOT / "aidas" / "app.py").read_text(encoding="utf-8")
        title_source = (PROJECT_ROOT / "aidas" / "ui" / "title_bar.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("create_custom_windows_title_bar", app_source)
        self.assertIn("_deactivate_windows_window_header_manipulation = True", app_source)
        self.assertIn(
            "original_style & ~(_WS_CAPTION | _WS_THICKFRAME)",
            title_source,
        )
        self.assertIn("def resize_window(", title_source)
        self.assertIn("_WM_NCLBUTTONDOWN", title_source)
        self.assertNotIn("overrideredirect(", title_source)

    def test_visible_batch_tabs_use_the_shared_customtkinter_component(self):
        step_sources = [
            (PROJECT_ROOT / "aidas" / "steps" / filename).read_text(encoding="utf-8")
            for filename in (
                "step2_annotate.py",
                "step3_flatten.py",
                "step4_analyze_isez.py",
            )
        ]

        for source in step_sources:
            self.assertIn("ClosableTabView", source)
        self.assertNotIn("ttk.Notebook(self.canvas_area)", step_sources[0])
        self.assertNotIn("Step3Batch.TNotebook", step_sources[1])
        self.assertNotIn("Step4Batch.TNotebook", step_sources[2])

    def test_packaged_ai_has_one_nonredundant_runtime_requirements_file(self):
        spec_text = (PROJECT_ROOT / "AIDaS.spec").read_text(encoding="utf-8")
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertEqual(
            [path.name for path in PROJECT_ROOT.glob("requirements*.txt")],
            ["requirements.txt"],
        )
        self.assertIn("model_img.onnx", spec_text)
        self.assertNotIn("model_img.pth", spec_text)
        self.assertIn('"torch",', spec_text)
        self.assertIn('"pytest",', spec_text)
        self.assertIn('"TkAgg"', spec_text)
        self.assertIn("onnxruntime-directml", requirements)
        self.assertIn("truststore>=", requirements)
        self.assertIn("pyinstaller>=", requirements.lower())
        self.assertIn("pywin32-ctypes>=", requirements.lower())
        self.assertNotIn("torch>=", requirements)
        self.assertNotIn("onnx>=", requirements)
        self.assertNotIn("xarray", requirements)

        for runtime_package in (
            '"customtkinter"',
            '"numpy"',
            '"scipy"',
            '"PIL"',
            '"matplotlib"',
            '"pyreadr"',
            '"packaging"',
            '"truststore"',
            '"onnxruntime"',
        ):
            with self.subTest(package=runtime_package):
                self.assertIn(runtime_package, spec_text)

        for obsolete_collection in ('"xarray"', '"imgviz"', '"prettytable"', '"art"', '"cmap"'):
            with self.subTest(package=obsolete_collection):
                self.assertNotIn(obsolete_collection, spec_text)

    def test_startup_splash_is_dynamic_and_has_no_generated_bitmap(self):
        spec_text = (PROJECT_ROOT / "AIDaS.spec").read_text(encoding="utf-8")
        self.assertNotIn("startup_splash.png", spec_text)
        self.assertNotIn("PyInstallerSplash", spec_text)
        self.assertFalse((PROJECT_ROOT / "assets" / "startup_splash.png").exists())
        self.assertFalse((PROJECT_ROOT / "tools" / "render_startup_splash.py").exists())


if __name__ == "__main__":
    unittest.main()
