from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from aidas.utils.batch_summary import format_batch_summary


class BatchSummaryTests(unittest.TestCase):
    def test_summary_uses_the_same_order_and_includes_zero_counts(self):
        self.assertEqual(
            format_batch_summary(
                scanned=505, ready=14, processed=4, skipped=485, unavailable=2,
            ),
            "Scanned: 505 · Ready: 14 · Processed: 4 · Skipped: 485 · "
            "Unavailable: 2 · Selected: 0",
        )

    def test_live_selection_counts_for_all_three_batch_panels(self):
        # Keep CustomTkinter's global Tk callbacks isolated from other tests.
        script = textwrap.dedent(
            r'''
            from pathlib import Path
            import tempfile
            import tkinter as tk
            from types import SimpleNamespace
            from unittest import mock

            import customtkinter as ctk

            from aidas.steps.step2_annotate import Step2BatchSegmentationSelectionPanel
            from aidas.steps.step3_flatten import RBatchSelectionPanel
            from aidas.steps.step4_analyze_isez import Step4BatchROISelectionPanel
            from aidas.utils.batch_summary import format_batch_summary

            root = ctk.CTk()
            root.withdraw()
            root.attributes("-alpha", 0)
            root.geometry("1024x680+0+0")
            root.deiconify()
            step = SimpleNamespace(
                status_var=tk.StringVar(master=root),
                _available_r_worker_limit=lambda: 8,
                _step2_core_allocation=lambda: 0,
                R_OUTPUT_MODE_PARALLEL="parallel",
                R_OUTPUT_MODE_SEQUENTIAL="sequential",
            )

            try:
                with tempfile.TemporaryDirectory() as directory:
                    for panel_type in (
                        Step2BatchSegmentationSelectionPanel,
                        RBatchSelectionPanel,
                        Step4BatchROISelectionPanel,
                    ):
                        with mock.patch.object(panel_type, "_start_scan"):
                            panel = panel_type(step, root, directory)
                        panel.pack(fill="both", expand=True)
                        root.update()
                        assert panel.summary_var.get() == "Scanning folders..."
                        rows = [
                            {
                                "folder": str(Path(directory) / name),
                                "include": include,
                                "locked": locked,
                                "status": "Processed" if locked else "Ready",
                                "outputs": "3/3" if locked else "0/3",
                                "image_paths": [str(Path(directory) / name / "Light.img")],
                                "values": {"folder": name, "status": "Ready", "inputs": "2"},
                            }
                            for name, include, locked in (
                                ("processed", True, True),
                                ("ready-a", True, False),
                                ("ready-b", False, False),
                            )
                        ]
                        panel._scan_done(rows, 505, 500, [(Path(directory) / "denied", "Access denied")])
                        root.update()
                        table = panel.table
                        run = (panel.process_button if panel_type is Step4BatchROISelectionPanel
                               else panel.next_button)

                        def assert_selected(count):
                            # Read immediately, without pumping Tk's event loop.
                            expected = format_batch_summary(
                                scanned=505, ready=2, processed=1, skipped=500,
                                unavailable=1, selected=count,
                            )
                            assert panel.summary_var.get() == expected, panel.summary_var.get()
                            assert len(table.selected_rows()) == count
                            assert ("disabled" in run.state()) == (count == 0)

                        def click(iid):
                            x, y, width, height = table.tree.bbox(iid, "#0")
                            table.tree.event_generate(
                                "<Button-1>", x=x + width // 2, y=y + height // 2,
                            )

                        assert_selected(1)
                        click("2")
                        assert_selected(2)
                        click("1")
                        assert_selected(1)
                        # Invoke the actual heading command, which selects every ready row.
                        root.tk.call(table.tree.heading("#0", "command"))
                        assert_selected(2)
                        root.tk.call(table.tree.heading("#0", "command"))
                        assert_selected(0)
                        root.tk.call(table.tree.heading("#0", "command"))
                        assert_selected(2)

                        panel.hide_processed_check.invoke()
                        assert_selected(2)
                        root.update()
                        click("0")
                        assert_selected(2)
                        assert rows[0]["include"] is True
                        panel.hide_processed_check.invoke()
                        assert_selected(2)

                        # Replacing table rows resets the live total immediately.
                        table.set_rows([rows[0], rows[1]])
                        assert_selected(1)
                        table.set_rows([rows[0]])
                        assert_selected(0)

                        # A new empty scan clears every previous statistic.
                        panel._scan_done([], 0, 0, [])
                        assert panel.summary_var.get() == format_batch_summary(
                            scanned=0, ready=0, processed=0, skipped=0,
                            unavailable=0, selected=0,
                        )
                        assert "disabled" in run.state()
                        panel._scan_failed(RuntimeError("scan unavailable"))
                        assert panel.summary_var.get() == "Scan failed."
                        assert panel.more_label.cget("text") == "More"
                        panel.destroy()
                        root.update()
                        print(panel_type.__name__ + ": passed")

                    # The Step 1 handoff uses the same totals as a folder scan.
                    with mock.patch.object(Step2BatchSegmentationSelectionPanel, "_start_scan"):
                        handoff = Step2BatchSegmentationSelectionPanel(
                            step, root, directory, initial_rows=rows,
                        )
                    assert handoff.summary_var.get() == format_batch_summary(
                        scanned=3, ready=2, processed=1, skipped=0,
                        unavailable=0, selected=2,
                    )
                    handoff.destroy()
            finally:
                root.destroy()
            '''
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count(": passed"), 3, result.stdout)


if __name__ == "__main__":
    unittest.main()
