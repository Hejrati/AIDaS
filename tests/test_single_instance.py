from __future__ import annotations

import errno
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

from aidas.core.single_instance import SingleInstanceGuard


class SingleInstanceGuardTests(unittest.TestCase):
    def test_posix_lock_is_created_in_the_current_users_config_directory(self):
        fake_fcntl = SimpleNamespace(
            LOCK_EX=1,
            LOCK_NB=2,
            LOCK_UN=4,
            flock=mock.Mock(),
        )
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            sys.modules,
            {"fcntl": fake_fcntl},
        ), mock.patch.object(Path, "home", return_value=Path(temp_dir)):
            guard = SingleInstanceGuard()

            self.assertTrue(guard._acquire_posix_lock())
            self.assertTrue(
                (Path(temp_dir) / ".aidas" / guard.POSIX_LOCK_NAME).is_file()
            )
            guard.close()

    def test_posix_lock_permission_error_returns_false_instead_of_crashing(self):
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=4, flock=mock.Mock())
        permission_error = PermissionError(errno.EACCES, "permission denied")
        with mock.patch.dict(sys.modules, {"fcntl": fake_fcntl}), mock.patch.object(
            Path,
            "mkdir",
            side_effect=permission_error,
        ):
            guard = SingleInstanceGuard()

            self.assertFalse(guard._acquire_posix_lock())
            self.assertFalse(guard.acquired)


if __name__ == "__main__":
    unittest.main()
