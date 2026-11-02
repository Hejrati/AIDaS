from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from openpyxl import Workbook, load_workbook

from aidas.services import measurement_inventory, measurement_scan, step4_compiler
from aidas.services.measurement_inventory import InventoryCancelled, scan_measurement_directory
from aidas.services.measurement_scan import (
    MAX_MEASUREMENT_SUBFOLDERS,
    MeasurementScanLimitExceeded,
    validate_measurement_subfolder_limit,
)


class MeasurementScanLimitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def _empty_folders(self, parent, count):
        for number in range(count):
            (parent / f"folder_{number:03d}").mkdir()

    def _measurements(self, side):
        side.mkdir(parents=True)
        workbook = Workbook()
        workbook.active.append([None] * 6 + ["Round"])
        workbook.active.append([None] * 6 + [7.25])
        workbook.save(side / "Results.xlsx")
        workbook.close()
        raw = side / "_thickness_vs_distance_from_fovea_LIGHT.txt"
        raw.write_bytes(b"Distance\tWhole\tELM\tONL\r\n0\t0\t30.125\t40.25\r\n")
        return raw

    def test_exactly_500_subfolders_are_allowed_and_selected_root_does_not_count(self):
        self.assertEqual(MAX_MEASUREMENT_SUBFOLDERS, 500)
        self._empty_folders(self.root, 500)

        self.assertEqual(validate_measurement_subfolder_limit(self.root), 500)
        inventory = scan_measurement_directory(self.root)

        self.assertEqual(len(inventory.root.children), 500)
        self.assertTrue(all(child.kind == "folder" for child in inventory.root.children))

    def test_limit_plus_one_raises_before_inventory_file_classification(self):
        self._empty_folders(self.root, 501)
        (self.root / "Results.xlsx").touch()

        with mock.patch.object(measurement_inventory, "_file_kind") as classify:
            with self.assertRaises(MeasurementScanLimitExceeded) as raised:
                scan_measurement_directory(self.root)

        self.assertIsInstance(raised.exception, ValueError)
        self.assertEqual(str(raised.exception), (
            "The selected folder contains too many subfolders and exceeds the limit of 500. "
            "Please select another folder."
        ))
        classify.assert_not_called()

    def test_total_includes_nested_descendants_instead_of_only_immediate_children(self):
        for number in range(10):
            branch = self.root / f"branch_{number}"
            branch.mkdir()
            self._empty_folders(branch, 49)
        self.assertEqual(validate_measurement_subfolder_limit(self.root), 500)
        (self.root / "branch_0" / "folder_000" / "one_more_nested_folder").mkdir()

        with self.assertRaises(MeasurementScanLimitExceeded):
            validate_measurement_subfolder_limit(self.root)

    def test_rejection_stops_at_the_first_folder_over_budget(self):
        visited = []

        def walk(_root, *, onerror, **_options):
            yield str(self.root), [], []
            for number in range(10_000):
                visited.append(number)
                if number > 500:
                    self.fail("The scan continued after the first folder over its budget")
                yield str(self.root / f"folder_{number}"), [], []

        with mock.patch.object(measurement_scan, "walk_unique_directories", side_effect=walk):
            with self.assertRaises(MeasurementScanLimitExceeded):
                validate_measurement_subfolder_limit(self.root)

        self.assertEqual(len(visited), 501)

    def test_cancelled_inventory_keeps_cancellation_semantics_at_the_budget_boundary(self):
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls == 502

        def walk(_root, *, onerror, **_options):
            yield str(self.root), [], ["Results.xlsx"]
            for number in range(501):
                yield str(self.root / f"folder_{number}"), [], ["Results.xlsx"]

        with mock.patch.object(measurement_scan, "walk_unique_directories", side_effect=walk), \
                mock.patch.object(measurement_inventory, "_file_kind") as classify:
            with self.assertRaises(InventoryCancelled):
                scan_measurement_directory(self.root, cancelled=cancelled)

        self.assertEqual(calls, 502)
        classify.assert_not_called()

    def test_duplicate_directory_aliases_do_not_consume_multiple_folder_slots(self):
        physical = self.root / "physical"
        physical.mkdir()
        original_realpath = os.path.realpath

        def realpath(path, *args, **kwargs):
            if Path(path).name.startswith("alias_"):
                return str(physical)
            return original_realpath(path, *args, **kwargs)

        def walk(_root, *, onerror, **_options):
            branches = [f"alias_{number}" for number in range(501)]
            yield str(self.root), branches, []
            for branch in branches:
                yield str(self.root / branch), [], []

        with mock.patch("aidas.utils.filesystem.os.path.realpath", side_effect=realpath), \
                mock.patch("aidas.utils.filesystem.os.walk", side_effect=walk):
            self.assertEqual(validate_measurement_subfolder_limit(self.root), 1)

    def test_directory_symlink_descendants_cannot_bypass_inventory_or_compiler_limit(self):
        def walk(_root, *, onerror, followlinks=False, **_options):
            yield str(self.root), ["linked_samples"], []
            if followlinks:
                linked = self.root / "linked_samples"
                yield str(linked), [f"folder_{number}" for number in range(500)], []
                for number in range(500):
                    yield str(linked / f"folder_{number}"), [], []

        with mock.patch("aidas.utils.filesystem.os.walk", side_effect=walk), \
                mock.patch.object(measurement_inventory, "_file_kind") as classify, \
                mock.patch.object(step4_compiler, "find_subject_folders") as discover_subjects:
            with self.assertRaises(MeasurementScanLimitExceeded):
                scan_measurement_directory(self.root)
            with self.assertRaises(MeasurementScanLimitExceeded):
                step4_compiler.compile_step4_results(self.root, self.root / "compiled.xlsx")

        classify.assert_not_called()
        discover_subjects.assert_not_called()
        self.assertFalse((self.root / "compiled.xlsx").exists())

    def test_real_linked_subtree_counts_unique_reachable_folders(self):
        selected = self.root / "selected"
        shared = self.root / "shared"
        selected.mkdir()
        shared.mkdir()
        self._empty_folders(shared, 499)
        link = selected / "linked_samples"
        junction = False
        try:
            link.symlink_to(shared, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            if os.name != "nt":
                self.skipTest(f"Directory links are unavailable: {error}")
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(shared)],
                capture_output=True, text=True, check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if created.returncode:
                self.skipTest(f"Directory links are unavailable: {created.stderr.strip()}")
            junction = True
        try:
            self.assertEqual(validate_measurement_subfolder_limit(selected), 500)
            (shared / "one_more_reachable_folder").mkdir()

            with self.assertRaises(MeasurementScanLimitExceeded):
                scan_measurement_directory(selected)
            with self.assertRaises(MeasurementScanLimitExceeded):
                step4_compiler.compile_step4_results(selected, selected / "compiled.xlsx")

            self.assertTrue(shared.is_dir())
            self.assertEqual(len(list(shared.iterdir())), 500)
            self.assertFalse((selected / "compiled.xlsx").exists())
        finally:
            if junction:
                link.rmdir()
            else:
                link.unlink()

    def test_compiler_accepts_exact_limit_and_keeps_progress_callbacks(self):
        side = self.root / "Control" / "Subject_A" / "nasal"
        raw = self._measurements(side)
        original_raw = raw.read_bytes()
        self._empty_folders(self.root, 497)
        output = self.root / "compiled.xlsx"
        progress = []
        messages = []

        result = step4_compiler.compile_step4_results(
            self.root, output, include_fovea=False, log_callback=messages.append,
            progress_callback=lambda *event: progress.append(event),
        )

        self.assertEqual(result.group_ids, {"Control": ("Subject_A",)})
        self.assertEqual(progress[0], (0, 1, "Scanning input folders..."))
        self.assertEqual(progress[-1], (5, 5, "Saved compiled workbook."))
        self.assertTrue(messages[-1].endswith(str(output)))
        self.assertEqual(raw.read_bytes(), original_raw)
        workbook = load_workbook(output, data_only=True)
        try:
            self.assertEqual(workbook["rrMCP-AR"]["B3"].value, 7.25)
            self.assertEqual(workbook["ELM-RPE"]["B3"].value, 30.125)
            self.assertEqual(workbook["ONL"]["B3"].value, 40.25)
        finally:
            workbook.close()

    def test_compiler_rechecks_growth_below_pruned_side_before_any_extraction_or_writes(self):
        side = self.root / "Control" / "Subject_A" / "nasal"
        self._measurements(side)
        self._empty_folders(side, 497)
        output = self.root / "compiled.xlsx"
        previous = Workbook()
        previous.active["A1"] = "Keep the existing Excel workbook"
        previous.save(output)
        previous.close()
        (side / "_thickness_vs_distance_from_fovea_LIGHT2.txt").write_text("old cleanup", encoding="utf-8")
        (side / "_thickness_vs_distance_from_fovea_LIGHT2.xlsx").write_bytes(b"previous cleaned workbook")
        self.assertIsNotNone(scan_measurement_directory(self.root))
        (side / "one_more_after_inventory").mkdir()
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

        with mock.patch.object(step4_compiler, "Workbook") as create_workbook, \
                mock.patch.object(step4_compiler, "find_subject_folders") as discover_subjects, \
                mock.patch.object(step4_compiler, "extract_rrmcpar_from_subject") as extract_roi, \
                mock.patch.object(step4_compiler, "extract_elmrpe_onl_from_subject") as extract_thickness, \
                mock.patch.object(step4_compiler, "create_cleaned_light_thickness_workbook") as clean_thickness:
            with self.assertRaises(MeasurementScanLimitExceeded):
                step4_compiler.compile_step4_results(self.root, output)

        for operation in (create_workbook, discover_subjects, extract_roi, extract_thickness, clean_thickness):
            operation.assert_not_called()
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)

    def test_deep_legitimate_hierarchy_below_total_budget_still_compiles_and_displays(self):
        container = self.root.joinpath(*("d" for _ in range(12)))
        side = container / "Control" / "Subject_A" / "nasal"
        exports = side.joinpath(*("e" for _ in range(12)))
        raw = self._measurements(exports)
        original_raw = raw.read_bytes()
        output = self.root / "compiled.xlsx"

        result = step4_compiler.compile_step4_results(self.root, output, include_fovea=False)
        inventory = scan_measurement_directory(self.root, output_path=output)

        self.assertEqual(result.group_ids, {"Control": ("Subject_A",)})
        self.assertEqual((inventory.workbooks, inventory.thickness_files), (1, 2))
        self.assertEqual(validate_measurement_subfolder_limit(self.root), 27)
        self.assertEqual(raw.read_bytes(), original_raw)


if __name__ == "__main__":
    unittest.main()
