from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "RAW_OCT_PROCESSING_2023_09SEP-05_WSU.R"
OUTPUT_SCRIPT = (
    PROJECT_ROOT
    / "more_outputs_afterRAW_OCT_PROCESSING_2022_11NOV_27_WSU_noHypoDenseBand_EA edited.R"
)
REQUIRED_R_VERSION = "3.3.1"


def _r_version(executable: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\b(?:R|R scripting front-end)\s+version\s+(\d+\.\d+\.\d+)", result.stdout or "")
    return match.group(1) if match else None


def _find_r331() -> Path | None:
    candidates = []
    for environment_name in ("AIDAS_R331_PATH", "RSCRIPT_PATH", "R_SCRIPT_PATH"):
        configured = os.environ.get(environment_name)
        if configured:
            candidates.append(Path(configured))

    if os.name == "nt":
        candidates.extend(
            (
                Path(r"C:\Program Files\R\R-3.3.1\bin\x64\Rscript.exe"),
                Path(r"C:\Program Files\R\R-3.3.1\bin\Rscript.exe"),
            )
        )
    discovered = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if discovered:
        candidates.append(Path(discovered))

    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(candidate))
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        if _r_version(candidate) == REQUIRED_R_VERSION:
            return candidate
    return None


def _r_string(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "\\'") + "'"


class RealR331IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.rscript = _find_r331()
        if self.rscript is not None:
            return
        message = "R 3.3.1 is required for the real Step 3 R integration test."
        if os.environ.get("AIDAS_REQUIRE_R331_INTEGRATION") == "1":
            self.fail(message)
        self.skipTest(message)

    def _run_r(self, *arguments: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(self.rscript), "--vanilla", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def test_release_scripts_parse_with_r331(self):
        expression = (
            f"parse(file={_r_string(MAIN_SCRIPT)}); "
            f"parse(file={_r_string(OUTPUT_SCRIPT)}); "
            "cat('R331_PARSE_OK\\n')"
        )

        result = self._run_r("-e", expression)

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("R331_PARSE_OK", result.stdout)

    def test_production_packages_load_with_real_r331(self):
        result = self._run_r(
            "-e",
            "library(AnalyzeFMRI); library(RNiftyReg); cat('R331_PACKAGES_OK\\n')",
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("R331_PACKAGES_OK", result.stdout)

    def test_output_script_runs_headlessly_with_real_r331(self):
        fixture_source = r'''
args <- commandArgs(trailingOnly=TRUE)
workspace <- normalizePath(args[[1]], mustWork=TRUE)
output.script <- normalizePath(args[[2]], mustWork=TRUE)
setwd(workspace)

TO.PROCESS.LIGHT <- "LIGHT"
FLATTENED.LIGHT.RETINA.RRC <- array(0, dim=c(2851,461,1))
R.RPE.POSITION.LIGHT <- matrix(400, nrow=2851)
R.OLM.POSITION.LIGHT <- matrix(350, nrow=2851)
R.ONL.OPL.POSITION.LIGHT <- matrix(300, nrow=2851)
R.INL.IPL.POSITION.LIGHT <- matrix(250, nrow=2851)
R.RNFL.GCL.POSITION.LIGHT <- matrix(200, nrow=2851)
R.VITREOUS.RETINA.POSITION.LIGHT <- matrix(100, nrow=2851)
save.image(file.path(workspace,"_done_DARK__and__LIGHT.RData"), version=2)

rm(list=ls())
args <- commandArgs(trailingOnly=TRUE)
setwd(args[[1]])
load(file.path(args[[1]],"_done_DARK__and__LIGHT.RData"))
source(args[[2]], chdir=FALSE, echo=FALSE)
'''
        expected_files = (
            "_tissueBorders__DARK.png",
            "_tissueBorders__LIGHT.png",
            "_thickness_vs_distance_from_fovea_DARK.txt",
            "_thickness_vs_distance_from_fovea_LIGHT.txt",
        )

        # Keep the fixture under the checkout so old R builds can resolve it
        # consistently on locked-down Windows runners and managed workspaces.
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            workspace = Path(temporary_directory)
            fixture_script = workspace / "run_output_fixture.R"
            fixture_script.write_text(fixture_source, encoding="utf-8")

            result = self._run_r(
                fixture_script.as_posix(),
                workspace.as_posix(),
                OUTPUT_SCRIPT.as_posix(),
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            for filename in expected_files:
                output_path = workspace / filename
                self.assertTrue(output_path.is_file(), f"Missing {filename}\n{result.stdout}")
                self.assertGreater(output_path.stat().st_size, 0, filename)
            for filename in ("_tissueBorders__DARK.png", "_tissueBorders__LIGHT.png"):
                self.assertEqual((workspace / filename).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(
                (workspace / "_thickness_vs_distance_from_fovea_DARK.txt").read_text(),
                (workspace / "_thickness_vs_distance_from_fovea_LIGHT.txt").read_text(),
            )
            self.assertFalse((workspace / "Rplots.pdf").exists(), result.stdout)

    def test_analyzefmri_writer_uses_the_supported_r331_file_argument(self):
        package_check = self._run_r(
            "-e",
            "if(!requireNamespace('AnalyzeFMRI', quietly=TRUE)) quit(status=2)",
        )
        if package_check.returncode != 0:
            message = "The bundled AnalyzeFMRI package is required for its real R 3.3.1 integration test."
            if os.environ.get("AIDAS_REQUIRE_R331_INTEGRATION") == "1":
                self.fail(message + "\n" + package_check.stdout)
            self.skipTest(message)

        fixture_source = r'''
args <- commandArgs(trailingOnly=TRUE)
workspace <- normalizePath(args[[1]], mustWork=TRUE)
library(AnalyzeFMRI)
target <- file.path(workspace,"_flat_LIGHT")
f.write.analyze(array(as.numeric(1:24),dim=c(4,3,2)),target,size="float")
'''
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            workspace = Path(temporary_directory)
            fixture_script = workspace / "write_analyze_fixture.R"
            fixture_script.write_text(fixture_source, encoding="utf-8")

            result = self._run_r(fixture_script.as_posix(), workspace.as_posix())

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual((workspace / "_flat_LIGHT.hdr").stat().st_size, 348)
            self.assertEqual((workspace / "_flat_LIGHT.img").stat().st_size, 24 * 4)


class ReleaseRScriptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_source = MAIN_SCRIPT.read_text(encoding="utf-8")
        cls.output_source = OUTPUT_SCRIPT.read_text(encoding="utf-8")

    def test_app_light_channel_populates_the_legacy_dark_slots(self):
        expected_reads = (
            'REF.LIGHT<-f.read.analyze.volume(paste(REFERENCE.LIGHT,".hdr",sep=""))',
            'LIGHT<-f.read.analyze.volume(paste(TO.PROCESS.LIGHT,".hdr",sep=""))',
        )
        for statement in expected_reads:
            self.assertEqual(self.main_source.count(statement), 1, statement)
        self.assertNotIn('REF.DARK<-f.read.analyze.volume', self.main_source)
        self.assertNotIn('DARK<-f.read.analyze.volume', self.main_source)
        self.assertIn("REF.DARK=REF.LIGHT", self.main_source)
        self.assertIn("DARK=LIGHT", self.main_source)
        self.assertNotIn('arg.or.env(3, "AIDAS_REFERENCE_DARK"', self.main_source)
        self.assertNotIn('arg.or.env(8, "AIDAS_IMAGE_INDEX_DARK"', self.main_source)
        self.assertIn('if(!exists("FLATTENED.DARK.RETINA.RRC"))', self.output_source)

    def test_analyze_exports_use_the_r331_supported_file_argument(self):
        self.assertNotIn("path.out=OUTDIR", self.main_source)
        self.assertEqual(
            self.main_source.count("f.write.analyze(EXPORT[,dim(EXPORT)[2]:1,],file.path(OUTDIR"),
            4,
        )

    def test_expensive_diagnostics_are_disabled_by_default(self):
        self.assertIn('Sys.getenv("AIDAS_STEP3_DIAGNOSTICS", unset="")', self.main_source)
        self.assertIn("if(!.DIAGNOSTICS.ENABLED) return(invisible(NULL))", self.main_source)
        self.assertNotIn("stop.at.boundary", self.main_source)
        self.assertNotIn("dim(REF.DARK[,,z])", self.main_source)
        self.assertNotIn("dim(REF.LIGHT[,,z])", self.main_source)


if __name__ == "__main__":
    unittest.main()
