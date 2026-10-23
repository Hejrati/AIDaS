# Tests

Run commands from the repository root, using the same Python environment as the
application:

```powershell
python -m unittest discover -s tests -t . -v
```

The `-t .` option gives tests their full package names, such as
`tests.ui.test_ui_splash`, and keeps shared fixture imports consistent. The
release workflow uses the same command.

## Organization

| Folder | Coverage |
| --- | --- |
| `ai/` | Inference, worker commands, and worker integration |
| `app/` | Application composition, interface switching, startup, and status |
| `canvas/` | Image canvas zoom and redraw behavior |
| `core/` | Configuration, display geometry, filesystem, process cancellation, and instance locking |
| `data/` | Analyze export and SDB parsing/metadata |
| `processing/` | Bright-pixel detection and retinal cropping |
| `release/` | Packaging, installer, version, and release workflow contracts |
| `services/` | Measurement compilation, R package/script helpers, and updates |
| `steps/step1/` ... `steps/step5/` | Workflow behavior and handoffs for each step |
| `ui/` | Shared controls, menus, dialogs, tabs, themes, and rendering |
| `support/` | Shared synthetic data, headless fixtures, paths, and runtime discovery |
| `r/` | Standalone R numerical and output regression scripts |

Group by the behavior under test. Import reusable fixtures from `tests.support`,
not from another `test_*.py` module. `tests.support.paths.PROJECT_ROOT` locates
assets and subprocess working directories independently of test nesting.

## Focused runs

```powershell
python -m unittest tests.ui.test_ui_splash -v
python -m unittest discover -s tests/steps/step2 -t . -v
python -m unittest discover -s tests/processing -t . -v
```

Folders describe features, not execution requirements. Some modules combine
headless checks with GUI or integration tests. GUI cases require a working Tk
installation and desktop; Windows caption tests additionally require Windows.
Tests with unavailable optional runtimes may skip, as reported by unittest.

## R integration

The Python R integration checks use R 3.3.1. They respect an explicit
`R_LIBS_USER` override; otherwise they use AIDaS's configured managed package
library, following the same default and legacy-path policy as the application.
These checks do not install packages or change application preferences.
Set `AIDAS_R331_PATH` when R is not
installed in the usual Windows location or available through `PATH`:

```powershell
$env:AIDAS_R331_PATH = 'C:\Program Files\R\R-3.3.1\bin\x64\Rscript.exe'
python -m unittest tests.steps.step3.test_step3_r331_integration -v
```

The release workflow sets `AIDAS_REQUIRE_R331_INTEGRATION=1` after installing the
required R packages, so missing integration prerequisites fail the release check
instead of silently skipping it.

The following R scripts are separate regression checks and are not discovered
by Python's unittest runner. Run them from the repository root with the required
R packages installed:

```powershell
$env:AIDAS_R331_PATH = 'C:\Program Files\R\R-3.3.1\bin\x64\Rscript.exe'
& $env:AIDAS_R331_PATH --vanilla tests/r/step3_optimization_kernels.R
& $env:AIDAS_R331_PATH --vanilla tests/r/step3_lateral_registration.R
& $env:AIDAS_R331_PATH --vanilla tests/r/step3_output_optimization.R
```

Generated `__pycache__` directories and `.pyc` files are disposable. Optional or
platform-specific regression sources should be retained when they still cover
supported application behavior.
