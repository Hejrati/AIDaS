# Validated Windows releases

The **Validate and release AIDaS** workflow runs **Windows validation** for
every push to `main` and every pull request. The same job runs the complete
test suite, including the required R 3.3.1 integration checks, builds the
PyInstaller executable, exercises its ONNX worker, builds the Inno Setup
installer, and installs and smoke-tests that installer. Validated installers
are retained as workflow artifacts for seven days.
New pushes cancel superseded validations of the same branch or pull request;
release-tag builds continue independently.

The test runner collects cyclic garbage on the main thread before each test.
Automatic collection stays disabled during discovery and execution so an old
test's Tk objects cannot be finalized on a background worker. Reference-counted
cleanup and explicit collection assertions still run normally; the original
collection setting is restored afterward. Test discovery, assertions, and failure
exit status are unchanged.

Tagging runs those checks again. Before building a tagged release, the
workflow requires a successful validation run for the **exact source commit**
from a push to `main`. A check from another commit, a pull request, or a
failed, pending, or cancelled run cannot authorize release. Publication is
a separate job that receives only the installer from the successful
validation job and verifies that the tag still identifies the same commit.
Signing credentials are used only for tagged builds; pull requests never
publish releases.

## Create a release

1. Set `aidas/__init__.py` to the intended version and add feature-focused notes
   to `release_tools/notes/<tag>.md`. Commit and push the source to `main`.
2. Wait for **Windows validation** on that exact commit to pass. Inspect any
   failure and fix it on `main` before creating a release tag.
3. From a clean checkout of the validated commit, run the read-only preflight
   using the same Python environment as the application:

   ```powershell
   python release_tools/preflight.py --tag v3.4.5a1
   ```

   The proposed tag must exactly match the application version. The command
   refuses uncommitted source and verifies the latest main-push validation
   result for `HEAD` using GitHub Actions. It does not create or push a tag.
   If the tag already exists locally or on `origin`, it must identify that
   same commit. An existing release tag on older source is rejected even when
   the current application version still matches its name. Annotated and
   lightweight tags are both checked; unavailable remote lookups stop release.
   Python contacts the GitHub API directly using the GitHub repository in the
   `origin` remote; no GitHub CLI installation is required. Public repositories
   work without authentication. For private repositories or API rate limits,
   provide `GH_TOKEN` or `GITHUB_TOKEN` with Actions read permission through
   your environment. The CI workflow supplies its token automatically.
4. Only after preflight succeeds, create and push the annotated tag:

   ```powershell
   git tag -a v3.4.5a1 -m "Describe the release functionality and features"
   git push origin v3.4.5a1
   ```

The tag workflow enforces the same preflight even when someone pushes a tag
without running the local command. It then reruns the complete tests and
binary checks before signing or publishing. Checks reduce release risk;
external downloads, signing services, and other infrastructure can still
fail, in which case publication stops.

Do not move an existing release tag to different source. Fixes need a new
commit, a successful main validation, and a new version/tag. A failed
infrastructure attempt can be rerun against the same unchanged tag through
the Actions page, or with the workflow's **Run workflow** action and its
existing-tag input.

A workflow-only repair can be used for an existing, unchanged tag by running
the workflow from `main` with that tag as its input. The checked-out application,
tests, version, notes, and validated source SHA still come from the tag. This
does not authorize changing the tagged source or bypassing any validation step.

Repository administrators can additionally require the **Windows validation**
status check in the `main` branch ruleset. The workflow itself does not change
repository rules or create tags.
