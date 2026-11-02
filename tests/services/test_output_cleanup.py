from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aidas.services.output_cleanup import remove_legacy_dark_outputs


class LegacyDarkOutputCleanupTests(unittest.TestCase):
    def _write(self, root: Path, name: str) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"saved data")
        return path

    def test_fresh_output_folder_has_nothing_to_remove(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self._write(root, "_thickness_vs_distance_from_fovea_LIGHT.txt")
            self.assertEqual(remove_legacy_dark_outputs(root), [])
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_removes_known_generated_outputs_and_keeps_light_originals_and_unrelated_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            obsolete = {
                "_flat_DARK.hdr", "_flat_DARK.img",
                "_flat-normed_DARK.hdr", "_flat-normed_DARK.img",
                "DARK_MARKED_find_vertex.png", "DARK_MARKED_vertex.png",
                "_tissueBorders__DARK.png", "_thickness_vs_distance_from_fovea_DARK.txt",
                "_dark_profiles_DARK.txt", "_fovea_dark_profiles_DARK.txt",
                "step3_r_arrays/FLATTENED_DARK_RETINA_RRC.bin",
                "step3_r_arrays/FLATTENED_DARK_RETINA_RRC.shape",
                "step3_r_arrays/APPARENT_ANGLES_FOR_DARK.bin",
                "step3_r_arrays/SHIFT_POSITION_DARK_REFINED.shape",
                "python_plots/python_dark_rpe_info_2_overlay_slice_1.png",
                "python_plots/python_shift_position_dark_3.png",
                "python_plots/python_revised_onl_opl_dark_slice_12.png",
                "python_plots/python_revised_vitreous_retina_dark_slice_1.png",
            }
            keep = {
                "DARK.hdr", "DARK.img", "DARK_MARKED.hdr", "DARK_MARKED.img",
                "_flat_LIGHT.img", "_thickness_vs_distance_from_fovea_LIGHT.txt",
                "_thickness_vs_distance_from_fovea_LIGHT2.xlsx",
                "dark_theme.png", "subject_dark.csv", "DARK.png",
                "step3_r_arrays/FLATTENED_LIGHT_RETINA_RRC.bin",
                "step3_r_arrays/my_dark_notes.txt",
                "python_plots/dark_participant.png",
                "python_plots/python_revised_unknown_dark_slice_1.png",
                "participant/_flat_DARK.img",
            }
            for name in obsolete | keep:
                self._write(root, name)
            removed = remove_legacy_dark_outputs(root)
            self.assertEqual({path.relative_to(root).as_posix() for path in removed}, obsolete)
            self.assertTrue(all((root / name).is_file() for name in keep))
            self.assertTrue(all((root / name).read_bytes() == b"saved data" for name in keep))
            self.assertEqual(remove_legacy_dark_outputs(root), [])

    def test_preserves_legacy_workspaces_until_both_light_replacements_of_same_format_exist(self):
        for suffix in ("RData", "npz"):
            for replacements in (0, 1, 2):
                with self.subTest(format=suffix, replacements=replacements), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    legacy = (f"DARK__and__LIGHT__flat.{suffix}", f"_done_DARK__and__LIGHT.{suffix}")
                    for name in legacy:
                        self._write(root, name)
                    current = (f"_flat_LIGHT.{suffix}", f"_done_LIGHT.{suffix}")
                    for name in current[:replacements]:
                        self._write(root, name)
                    # Replacements in a different format cannot retire these.
                    other = "npz" if suffix == "RData" else "RData"
                    self._write(root, f"_flat_LIGHT.{other}")
                    self._write(root, f"_done_LIGHT.{other}")
                    removed = remove_legacy_dark_outputs(root)
                    self.assertEqual({path.name for path in removed}, set(legacy) if replacements == 2 else set())
                    self.assertTrue(all((root / name).exists() == (replacements != 2) for name in legacy))
                    self.assertTrue(all((root / name).exists() for name in current[:replacements]))

    def test_skips_known_subdirectories_resolved_outside_the_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = Path(temporary).resolve()
            root, outside = sandbox / "output", sandbox / "other-participant"
            root.mkdir()
            protected = self._write(outside, "FLATTENED_DARK_RETINA_RRC.bin")
            protected_plot = self._write(outside, "python_dark_profile.png")
            original_resolve = Path.resolve

            def resolve(path, *args, **kwargs):
                if path.parent == root and path.name in ("step3_r_arrays", "python_plots"):
                    return original_resolve(outside)
                return original_resolve(path, *args, **kwargs)

            with patch.object(Path, "resolve", resolve):
                self.assertEqual(remove_legacy_dark_outputs(root), [])
            self.assertEqual(protected.read_bytes(), b"saved data")
            self.assertEqual(protected_plot.read_bytes(), b"saved data")

    def test_locked_obsolete_file_does_not_fail_a_successful_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            obsolete = self._write(root, "_flat_DARK.img")
            current = self._write(root, "_flat_LIGHT.img")
            with patch.object(Path, "unlink", side_effect=PermissionError("in use")):
                with self.assertLogs("aidas.services.output_cleanup", level="WARNING"):
                    self.assertEqual(remove_legacy_dark_outputs(root), [])
            self.assertTrue(obsolete.is_file())
            self.assertEqual(current.read_bytes(), b"saved data")


if __name__ == "__main__":
    unittest.main()
