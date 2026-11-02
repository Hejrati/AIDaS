"""Real R exports preserve original standalone and optimized LIGHT policies."""

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
# Actual PNG devices and overlay code run here. Rasterizing the synthetic
# full-sized image keeps the environment-policy matrix inexpensive; the
# independent output-optimization regression checks default-renderer bytes.
image <- function(..., useRaster=TRUE) graphics::image(..., useRaster=useRaster)
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
# Optimized helpers also reject direct DARK requests before touching array
# arguments. Original scripts retain their untouched standalone behavior.
if(identical(args[[4L]], "1")) {
  save.tissue.border.output("_tissueBorders__DARK.png")
  if(exists("write_thickness_export")) write_thickness_export(NULL, "DARK")
}
'''


class Step3LightOnlyR331Tests(unittest.TestCase):
    def test_output_variants_preserve_light_values_and_their_export_policies(self):
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
                    for setting in (None, "0", "1"):
                        folder = root / f"variant-{index}-mode-{setting or 'unset'}"
                        folder.mkdir()
                        folders.append(folder)
                        environment = r331_environment()
                        if setting is None:
                            environment.pop("AIDAS_LIGHT_ONLY", None)
                        else:
                            environment["AIDAS_LIGHT_ONLY"] = setting
                        result = subprocess.run(
                            [str(rscript), "--vanilla", fixture.as_posix(), folder.as_posix(),
                             (PROJECT_ROOT / filename).as_posix(), "1" if deep else "0",
                             "1" if index else "0"],
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

                    light_names = {"_tissueBorders__LIGHT.png",
                                   "_thickness_vs_distance_from_fovea_LIGHT.txt"}
                    reference = folders[0]
                    for setting, folder in zip((None, "0", "1"), folders):
                        expected_names = set(light_names)
                        if index == 0 and setting != "1":
                            expected_names.update({"_tissueBorders__DARK.png",
                                                   "_thickness_vs_distance_from_fovea_DARK.txt"})
                        self.assertEqual({path.name for path in folder.iterdir()}, expected_names)
                        for name in expected_names:
                            self.assertGreater((folder / name).stat().st_size, 0, name)
                        for name in light_names:
                            self.assertEqual((folder / name).read_bytes(), (reference / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
