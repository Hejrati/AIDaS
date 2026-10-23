from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.steps.step3_flatten import RSetupWizard, Step3Frame
from aidas.ui.theme import apply_appearance_mode, get_interface_mode, set_interface_mode


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class RSetupLayoutTests(unittest.TestCase):
    def setUp(self):
        previous_interface = get_interface_mode()
        previous_appearance = (
            "System" if ctk.AppearanceModeTracker.appearance_mode_set_by == "system"
            else ctk.get_appearance_mode()
        )
        self.addCleanup(ctk.set_appearance_mode, previous_appearance)
        self.addCleanup(set_interface_mode, previous_interface, redraw=False)
        set_interface_mode("Modern", redraw=False)
        try:
            self.root = _QuietRoot()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        self.addCleanup(self.root.destroy)
        self.root.attributes("-alpha", 0)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        step = SimpleNamespace(
            R_REQUIRED_VERSION=Step3Frame.R_REQUIRED_VERSION,
            R_REQUIRED_PACKAGES=Step3Frame.R_REQUIRED_PACKAGES,
            R_LOCAL_PACKAGE_ORDER=Step3Frame.R_LOCAL_PACKAGE_ORDER,
            r_package_library_path=str(folder / "R-packages"),
            _resolve_rscript_executable=lambda: None,
        )
        with mock.patch.object(RSetupWizard, "_package_log_path", return_value=folder / "setup.log"):
            self.wizard = RSetupWizard(step, self.root)
        self.wizard.after_cancel(self.wizard._initial_check_after_id)
        self.wizard._initial_check_after_id = None
        self.wizard._detecting_r = False
        self.wizard.pack(fill="both", expand=True)

    def test_package_rows_and_actions_remain_visible_when_resizing_in_both_themes(self):
        wizard = self.wizard
        wizard.rscript_path = Path("C:/Program Files/R/R-3.3.1/bin/x64/Rscript.exe")
        wizard.package_status = {name: "installed" for name in wizard.setup_package_names}
        wizard._refresh_status_display()
        tree = wizard.package_tree
        self.assertEqual(len(tree.get_children()), len(Step3Frame.R_LOCAL_PACKAGE_ORDER))
        for mode in ("Light", "Dark"):
            apply_appearance_mode(mode, root=self.root, style=ttk.Style(self.root))
            for width, height in ((1050, 505), (900, 480), (1050, 650)):
                with self.subTest(mode=mode, size=(width, height)):
                    self.root.geometry(f"{width}x{height}+0+0")
                    self.root.update()
                    self.assertTrue(tree.winfo_viewable())
                    self.assertGreater(tree.winfo_height(), 48)
                    tree_bottom = tree.winfo_rooty() + tree.winfo_height()
                    self.assertLessEqual(tree_bottom, wizard.content.winfo_rooty() + wizard.content.winfo_height())
                    for button in (wizard.check_packages_button, wizard.close_button):
                        self.assertTrue(button.winfo_viewable())
                        self.assertLess(tree_bottom, button.winfo_rooty())
                        self.assertLessEqual(
                            button.winfo_rooty() + button.winfo_height(),
                            wizard.winfo_rooty() + wizard.winfo_height(),
                        )
                    rescan = wizard.check_packages_button
                    close = wizard.close_button
                    self.assertEqual(rescan.cget("text"), "Rescan")
                    self.assertEqual(close.cget("state"), "normal")
                    rescan_right = rescan.winfo_rootx() + rescan.winfo_width()
                    self.assertLessEqual(rescan_right, close.winfo_rootx())
                    self.assertLessEqual(close.winfo_rootx() - rescan_right, 20)
                    self.assertLessEqual(
                        abs((rescan.winfo_rooty() + rescan.winfo_height() / 2)
                            - (close.winfo_rooty() + close.winfo_height() / 2)),
                        2,
                    )
                    last = tree.get_children()[-1]
                    tree.see(last)
                    self.root.update_idletasks()
                    bounds = tree.bbox(last)
                    self.assertTrue(bounds, "The last package must be reachable by scrolling")
                    self.assertLessEqual(bounds[1] + bounds[3], tree.winfo_height())
                    self.assertEqual(tree.item(last, "values")[-1], "Installed")

    def test_setup_has_no_log_panel_and_logging_works_without_a_text_widget(self):
        pending = [self.wizard]
        while pending:
            widget = pending.pop()
            self.assertNotIsInstance(widget, tk.Text)
            if isinstance(widget, ttk.LabelFrame):
                self.assertNotEqual(widget.cget("text"), "Setup log")
            pending.extend(widget.winfo_children())
        self.wizard._log("Package check completed.")
        self.assertIn("Package check completed.", self.wizard.log_path.read_text(encoding="utf-8"))

    def test_setup_complete_label_is_absent_in_both_missing_and_ready_states(self):
        for ready in (False, True, False):
            with self.subTest(ready=ready):
                self.wizard.rscript_path = Path("Rscript.exe") if ready else None
                status = "installed" if ready else "missing"
                self.wizard.package_status = {name: status for name in self.wizard.setup_package_names}
                self.wizard._refresh_status_display()
                self.root.update()
                pending = [self.wizard]
                while pending:
                    widget = pending.pop()
                    if isinstance(widget, (tk.Label, ttk.Label, ctk.CTkLabel)):
                        self.assertNotEqual(widget.cget("text"), "Setup complete")
                    pending.extend(widget.winfo_children())


if __name__ == "__main__":
    unittest.main()
