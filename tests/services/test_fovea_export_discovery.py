"""Fovea exports follow each side's profile through supported folder layouts."""

from pathlib import Path
import tempfile
import unittest

from aidas.services.step4_compiler import extract_elmrpe_onl_from_subject, find_fovea_file


def _write_profile(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"foveaLIGHT 1 2\nRPEtoOLM {value} {value}\n", encoding="utf-8")


class FoveaExportDiscoveryTests(unittest.TestCase):
    def test_nested_profiles_supply_the_correct_sides_fovea_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            subject = Path(temporary)
            nasal = subject / "nasal" / "exports" / "final" / "_fovea_light_profiles_LIGHT.txt"
            temporal = subject / "temporal" / "exports" / "final" / "_fovea_light_profiles_LIGHT.txt"
            _write_profile(nasal, 85.918)
            _write_profile(temporal, 88.258)

            nasal_values, temporal_values, *_ = extract_elmrpe_onl_from_subject(
                subject, "ELM-RPE", True,
            )

            self.assertEqual(nasal_values[0], 85.918)
            self.assertEqual(temporal_values[0], 88.258)
            self.assertEqual(len(nasal_values), 24)
            self.assertEqual(len(temporal_values), 24)

    def test_mixed_case_side_folders_and_profile_names_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            subject = Path(temporary)
            _write_profile(subject / "NaSaL" / "_FOVEA_LIGHT_PROFILES_LIGHT.TXT", 12.345)
            _write_profile(subject / "TEMPORAL" / "_Fovea_Light_Profiles_Light.txt", 67.89)

            nasal_values, temporal_values, *_ = extract_elmrpe_onl_from_subject(
                subject, "ELM-RPE", True,
            )

            self.assertEqual(nasal_values[0], 12.345)
            self.assertEqual(temporal_values[0], 67.89)

    def test_temporal_only_profile_never_populates_the_nasal_fovea(self):
        with tempfile.TemporaryDirectory() as temporary:
            subject = Path(temporary)
            _write_profile(
                subject / "temporal" / "exports" / "_fovea_light_profiles_LIGHT.txt", 88.258,
            )

            nasal_values, temporal_values, *_ = extract_elmrpe_onl_from_subject(
                subject, "ELM-RPE", True,
            )

            self.assertIsNone(nasal_values[0])
            self.assertEqual(temporal_values[0], 88.258)

    def test_legacy_subject_root_and_rrmcpar_profiles_remain_nasal_fallbacks(self):
        for location in ("", "rrMCPAR"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as temporary:
                subject = Path(temporary)
                _write_profile(subject / location / "_fovea_light_profiles_LIGHT.txt", 55.77)
                _write_profile(
                    subject / "temporal" / "exports" / "_fovea_light_profiles_LIGHT.txt", 66.88,
                )

                nasal_values, temporal_values, *_ = extract_elmrpe_onl_from_subject(
                    subject, "ELM-RPE", True,
                )

                self.assertEqual(nasal_values[0], 55.77)
                self.assertEqual(temporal_values[0], 66.88)

    def test_direct_current_profile_keeps_priority_over_cleaned_or_nested_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            preferred = folder / "_fovea_light_profiles_LIGHT.txt"
            _write_profile(folder / "_fovea_light_profiles_LIGHT2.txt", 999)
            _write_profile(folder / "exports" / preferred.name, 888)
            _write_profile(preferred, 55.77)

            self.assertEqual(Path(find_fovea_file(folder)), preferred)


if __name__ == "__main__":
    unittest.main()
