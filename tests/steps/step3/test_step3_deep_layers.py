"""Numerical regression for the separately bundled deep-layer R variant."""

from __future__ import annotations

import os
import subprocess
import unittest

from tests.support.paths import PROJECT_ROOT
from tests.support.r_runtime import find_r331, r331_environment


class Step3DeepLayersTests(unittest.TestCase):
    def test_full_and_deep_modes_preserve_their_numerical_contracts(self):
        rscript = find_r331()
        if rscript is None:
            message = "R 3.3.1 is required for the Step 3 deep-layer regression."
            if os.environ.get("AIDAS_REQUIRE_R331_INTEGRATION") == "1":
                self.fail(message)
            self.skipTest(message)

        result = subprocess.run(
            [str(rscript), "--vanilla", str(PROJECT_ROOT / "tests/r/step3_deep_layers.R")],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=r331_environment(),
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("deep-layer functional regression checks", result.stdout)


if __name__ == "__main__":
    unittest.main()
