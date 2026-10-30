"""Real R coverage for LIGHT-only output from all bundled main scripts."""

from __future__ import annotations

import os
import subprocess
import unittest

from tests.support.paths import PROJECT_ROOT
from tests.support.r_runtime import find_r331, r331_environment


class Step3LightOnlyMainRTests(unittest.TestCase):
    def test_bundled_main_scripts_generate_only_light_artifacts(self):
        rscript = find_r331()
        if rscript is None:
            message = "R 3.3.1 is required for the LIGHT-only main export regression."
            if os.environ.get("AIDAS_REQUIRE_R331_INTEGRATION") == "1":
                self.fail(message)
            self.skipTest(message)

        result = subprocess.run(
            [str(rscript), "--vanilla", str(PROJECT_ROOT / "tests/r/step3_light_only_main.R")],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=r331_environment(),
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("LIGHT-only main export regression checks", result.stdout)


if __name__ == "__main__":
    unittest.main()
