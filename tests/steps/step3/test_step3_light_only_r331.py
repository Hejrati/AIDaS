"""Real R output variants retain LIGHT results while omitting DARK files."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from aidas.steps.step3_flatten import Step3Frame
from tests.support.paths import PROJECT_ROOT
from tests.support.r_runtime import find_r331, r331_environment


_OUTPUT_FIXTURE = r'''
args <- commandArgs(trailingOnly=TRUE)
setwd(args[[1L]])
TO.PROCESS.LIGHT <- "LIGHT"
deep_layers_only <- identical(args[[3L]], "1")
FLATTENED.LIGHT.RETINA.RRC <- array(0, dim=c(2851,461,1))
R.RPE.POSITION.LIGHT <- matrix(400, nrow=2851)
R.OLM.POSITION.LIGHT <- matrix(350, nrow=2851)
R.ONL.OPL.POSITION.LIGHT <- matrix(300, nrow=2851)
R.INL.IPL.POSITION.LIGHT <- matrix(250, nrow=2851)
R.RNFL.GCL.POSITION.LIGHT <- matrix(200, nrow=2851)
R.VITREOUS.RETINA.POSITION.LIGHT <- matrix(100, nrow=2851)
source(args[[2L]], chdir=FALSE, echo=FALSE)
'''


class Step3LightOnlyR331Tests(unittest.TestCase):
    def test_bundled_output_variants_preserve_light_files_without_generating_dark_files(self):
        rscript = find_r331()
        if rscript is None:
            message = "R 3.3.1 is required for the LIGHT-only output regression."
            if os.environ.get("AIDAS_REQUIRE_R331_INTEGRATION") == "1":
                self.fail(message)
            self.skipTest(message)

        variants = (
            (Step3Frame.R_OUTPUT_SCRIPT_NAME, False),
            (Step3Frame.R_OPTIMIZED_OUTPUT_SCRIPT_NAME, False),
            (Step3Frame.R_DEEP_LAYERS_OUTPUT_SCRIPT_NAME, True),
        )
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            fixture = root / "run_output_fixture.R"
            fixture.write_text(_OUTPUT_FIXTURE, encoding="utf-8")
            for index, (filename, deep) in enumerate(variants):
                with self.subTest(script=filename):
                    folders = []
                    for light_only in (False, True):
                        folder = root / f"variant-{index}-mode-{int(light_only)}"
                        folder.mkdir()
                        folders.append(folder)
                        environment = r331_environment()
                        environment["AIDAS_LIGHT_ONLY"] = "1" if light_only else "0"
                        result = subprocess.run(
                            [str(rscript), "--vanilla", fixture.as_posix(), folder.as_posix(),
                             (PROJECT_ROOT / filename).as_posix(), "1" if deep else "0"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            env=environment,
                            timeout=240,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                        self.assertEqual(result.returncode, 0, result.stdout)

                    paired, light = folders
                    for name in ("_tissueBorders__LIGHT.png",
                                 "_thickness_vs_distance_from_fovea_LIGHT.txt"):
                        self.assertTrue((light / name).is_file(), name)
                        self.assertEqual((light / name).read_bytes(), (paired / name).read_bytes(), name)
                    self.assertTrue((paired / "_tissueBorders__DARK.png").is_file())
                    self.assertTrue((paired / "_thickness_vs_distance_from_fovea_DARK.txt").is_file())
                    self.assertFalse(any("dark" in path.name.casefold() for path in light.rglob("*")))
                    self.assertFalse((light / "Rplots.pdf").exists())


if __name__ == "__main__":
    unittest.main()
