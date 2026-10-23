import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.core.config import Config
from aidas.steps.step3_flatten import Step3Frame
from tests.support.r_runtime import r331_environment


class RRuntimeEnvironmentTests(unittest.TestCase):
    def test_explicit_library_override_is_preserved_without_reading_preferences(self):
        with mock.patch.dict(os.environ, {"R_LIBS_USER": "CI library", "OTHER": "kept"}, clear=True), mock.patch.object(
            Config, "peek",
        ) as peek:
            self.assertEqual(r331_environment(), {"R_LIBS_USER": "CI library", "OTHER": "kept"})
            peek.assert_not_called()

    def _assert_matches_application(self, home, configured, environment):
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            Path, "home", return_value=home,
        ), mock.patch.object(Config, "peek", return_value=configured):
            result = r331_environment()
            app_path = Step3Frame._default_r_package_library(
                SimpleNamespace(r_package_library_path=configured)
            )
            self.assertEqual(result["R_LIBS_USER"], str(app_path.resolve()))
            return result

    def test_saved_custom_library_matches_application_and_is_not_created(self):
        with TemporaryDirectory() as directory:
            home = Path(directory)
            custom = home / "custom library"
            self._assert_matches_application(home, str(custom), {"OTHER": "kept"})
            self.assertFalse(custom.exists())

    def test_default_library_matches_application_with_and_without_documents(self):
        with TemporaryDirectory() as directory:
            home = Path(directory)
            self._assert_matches_application(home, None, {})
            (home / "Documents").mkdir()
            result = self._assert_matches_application(home, None, {})
            self.assertEqual(result["R_LIBS_USER"], str(home / "Documents" / "AIDaS" / "R-packages"))
            self.assertFalse((home / "Documents" / "AIDaS").exists())

    def test_legacy_library_preference_follows_the_application_migration_policy(self):
        with TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "Documents").mkdir()
            app_data = home / "local app data"
            legacy = app_data / "AIDaS" / "R-packages"
            result = self._assert_matches_application(home, str(legacy), {"LOCALAPPDATA": str(app_data)})
            self.assertEqual(result["R_LIBS_USER"], str(home / "Documents" / "AIDaS" / "R-packages"))


if __name__ == "__main__":
    unittest.main()
