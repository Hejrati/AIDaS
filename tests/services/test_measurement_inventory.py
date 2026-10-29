from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aidas.services.measurement_inventory import InventoryCancelled, scan_measurement_directory


def _nodes(root):
    pending = [root]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(reversed(node.children))


class MeasurementInventoryTests(unittest.TestCase):
    def test_flat_step4_folder_shows_actual_workbook_and_measurement_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "R_TEST" / "flip human - Copy - Copy (12)"
            root.mkdir(parents=True)
            important = {
                "rr_MCPAR.xlsx": "excel",
                "_thickness_vs_distance_from_fovea_LIGHT.txt": "thickness",
                "_fovea_light_profiles_LIGHT.txt": "fovea",
                "_flat_LIGHT.hdr": "image",
                "_flat_LIGHT.img": "image",
                "ROI_to_move_stck.tif": "image",
                "MAX_Stack.tif": "image",
            }
            for name in (*important, "Light.img", "_dark_profiles_DARK.txt", "~$rr_MCPAR.xlsx"):
                (root / name).touch()
            arrays = root / "step3_r_arrays" / "nested"
            arrays.mkdir(parents=True)
            (arrays / "array.bin").touch()
            result = scan_measurement_directory(root)
            files = {node.name: node.kind for node in _nodes(result.root) if node.kind not in ("folder", "other")}
            self.assertEqual(files, important)
            self.assertEqual((result.workbooks, result.thickness_files, result.fovea_files), (1, 1, 1))
            self.assertEqual(result.unassigned_workbooks, 1)
            self.assertEqual(result.root.name, root.name)
            hidden = next(node for node in result.root.children if node.name == "step3_r_arrays")
            self.assertFalse(hidden.children)
            self.assertEqual(hidden.detail, "No recorded Excel results")
            self.assertEqual(result.root.total_files, 11)

    def test_nested_files_preserve_real_group_subject_and_side_relationships(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            side = root / "Control" / "Subject_A" / "Nasal" / "exports" / "final"
            side.mkdir(parents=True)
            workbook = side / "rr_MCPAR.xlsx"
            workbook.write_bytes(b"not opened by the visualization")
            original = workbook.read_bytes()
            result = scan_measurement_directory(root)
            by_name = {node.name: node for node in _nodes(result.root)}
            self.assertEqual(by_name["Control"].detail, "Group")
            self.assertEqual(by_name["Subject_A"].detail, "Subject ID")
            self.assertEqual(by_name["Nasal"].detail, "Nasal")
            self.assertEqual(by_name["rr_MCPAR.xlsx"].path, workbook.resolve())
            self.assertEqual(result.unassigned_workbooks, 0)
            self.assertEqual(workbook.read_bytes(), original)

    def test_compiled_output_is_shown_separately_from_input_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("custom.xlsx", "rrMCPAR_ELMRPE_ONL.xlsx", "rr_MCPAR.xlsx"):
                (root / name).touch()
            result = scan_measurement_directory(root, output_path=root / "custom.xlsx")
            self.assertEqual(result.workbooks, 1)
            self.assertEqual(result.unassigned_workbooks, 1)
            outputs = [node for node in result.root.children if node.kind == "output"]
            self.assertEqual({node.name for node in outputs}, {"custom.xlsx", "rrMCPAR_ELMRPE_ONL.xlsx"})

    def test_empty_folder_and_missing_folder_have_honest_empty_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = scan_measurement_directory(root)
            self.assertFalse(result.root.children)
            self.assertEqual(result.root.detail, "No recorded Excel results")
            result = scan_measurement_directory(root / "missing")
            self.assertEqual(result.root.detail, "Folder unavailable")
            self.assertTrue(result.errors)

    def test_cancelled_scan_stops_before_returning_an_outdated_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(InventoryCancelled):
                scan_measurement_directory(Path(temporary), cancelled=lambda: True)

    def test_unreadable_branch_does_not_hide_other_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()

            def walk(path, *, onerror):
                onerror(PermissionError("A subfolder is unavailable"))
                yield str(root), ["unavailable"], ["rr_MCPAR.xlsx"]

            with mock.patch("aidas.services.measurement_inventory.os.walk", side_effect=walk):
                result = scan_measurement_directory(root)
            self.assertEqual(result.workbooks, 1)
            self.assertTrue(result.errors)
            self.assertEqual(result.root.children[0].detail, "Folder unavailable")


if __name__ == "__main__":
    unittest.main()
