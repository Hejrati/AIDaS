from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from release_tools.preflight import (
    check_release,
    fetch_validation_runs,
    github_repository,
    require_successful_validation,
    run_command,
)


COMMIT = "a" * 40


def validation_run(**overrides):
    values = {
        "head_sha": COMMIT,
        "head_branch": "main",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "run_number": 42,
        "run_attempt": 1,
        "html_url": "https://github.com/example/AIDaS/actions/runs/42",
    }
    return values | overrides


class ValidatedCommitTests(unittest.TestCase):
    def test_accepts_a_successful_main_validation_for_the_exact_commit(self):
        run = validation_run()
        self.assertIs(require_successful_validation({"workflow_runs": [run]}, COMMIT), run)

    def test_other_commit_branch_and_pull_request_checks_do_not_authorize_a_tag(self):
        for overrides in ({"head_sha": "b" * 40}, {"head_branch": "feature"}, {"event": "pull_request"}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "no Windows validation"):
                require_successful_validation({"workflow_runs": [validation_run(**overrides)]}, COMMIT)

    def test_missing_validation_does_not_authorize_a_tag(self):
        with self.assertRaisesRegex(ValueError, "wait for Windows validation"):
            require_successful_validation({"workflow_runs": []}, COMMIT)

    def test_failed_cancelled_or_pending_runs_do_not_authorize_a_tag(self):
        for state in ("failure", "cancelled", "timed_out", "skipped", None):
            status = "in_progress" if state is None else "completed"
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "not successful"):
                require_successful_validation(
                    {"workflow_runs": [validation_run(conclusion=state, status=status)]}, COMMIT,
                )

    def test_old_success_does_not_override_a_new_failed_validation(self):
        older = validation_run(run_number=41)
        latest = validation_run(conclusion="failure")
        with self.assertRaisesRegex(ValueError, "is failure"):
            require_successful_validation({"workflow_runs": [older, latest]}, COMMIT)

    def test_latest_rerun_attempt_determines_the_result(self):
        earlier = validation_run(conclusion="failure")
        latest = validation_run(run_attempt=2)
        self.assertIs(
            require_successful_validation({"workflow_runs": [earlier, latest]}, COMMIT), latest,
        )


class ReleasePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "aidas").mkdir()
        (self.root / "aidas" / "__init__.py").write_text('__version__ = "3.4.5a1"\n')

    def test_checks_clean_source_and_queries_the_exact_commit_without_mutating_git(self):
        payload = {"workflow_runs": [validation_run()]}
        with (
            patch("release_tools.preflight.run_command", side_effect=["", COMMIT, "https://github.com/example/AIDaS.git"]) as command,
            patch("release_tools.preflight.fetch_validation_runs", return_value=payload) as fetch,
        ):
            commit, run = check_release("v3.4.5a1", project_root=self.root)
        self.assertEqual(commit, COMMIT)
        self.assertEqual(run["conclusion"], "success")
        self.assertEqual(command.call_args_list[0].args[0], ["git", "status", "--porcelain"])
        self.assertEqual(command.call_args_list[1].args[0], ["git", "rev-parse", "HEAD"])
        self.assertEqual(command.call_args_list[2].args[0], ["git", "remote", "get-url", "origin"])
        fetch.assert_called_once_with("example/AIDaS", COMMIT)
        self.assertEqual(command.call_count, 3)

    def test_wrong_version_stops_before_git_or_github_checks(self):
        with patch("release_tools.preflight.run_command") as command:
            with self.assertRaisesRegex(ValueError, "must exactly match"):
                check_release("v3.4.4a2", project_root=self.root)
        command.assert_not_called()

    def test_uncommitted_changes_stop_before_github_checks(self):
        with patch("release_tools.preflight.run_command", return_value=" M aidas/ui/tutorial.py") as command:
            with self.assertRaisesRegex(ValueError, "uncommitted changes"):
                check_release("v3.4.5a1", project_root=self.root)
        self.assertEqual(command.call_count, 1)

    def test_unsupported_origin_stops_before_network_access(self):
        with (
            patch("release_tools.preflight.run_command", side_effect=["", COMMIT, "https://other.example/owner/repo.git"]),
            patch("release_tools.preflight.urlopen") as request,
        ):
            with self.assertRaisesRegex(ValueError, "github.com"):
                check_release("v3.4.5a1", project_root=self.root)
        request.assert_not_called()

    def test_unavailable_git_reports_the_required_command(self):
        with patch("release_tools.preflight.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, "Required command is not installed: git"):
                run_command(["git", "status"], cwd=self.root)

    def test_git_failure_is_reported_instead_of_approving_a_release(self):
        failure = subprocess.CalledProcessError(1, ["git", "remote"], stderr="origin is not configured")
        with patch("release_tools.preflight.subprocess.run", side_effect=failure):
            with self.assertRaisesRegex(ValueError, "git: origin is not configured"):
                run_command(["git", "remote"], cwd=self.root)


class GitHubRepositoryTests(unittest.TestCase):
    def test_https_and_ssh_origins_identify_the_same_repository(self):
        for origin in (
            "https://github.com/example/AIDaS.git",
            "https://github.com/example/AIDaS",
            "git@github.com:example/AIDaS.git",
            "ssh://git@github.com/example/AIDaS.git",
        ):
            with self.subTest(origin=origin):
                self.assertEqual(github_repository(origin), "example/AIDaS")

    def test_other_hosts_and_ambiguous_or_unsafe_repository_names_are_rejected(self):
        origins = (
            "http://github.com/example/AIDaS.git",
            "https://other.example/example/AIDaS.git",
            "https://github.com.other.example/example/AIDaS.git",
            "git@other.example:example/AIDaS.git",
            "https://user:password@github.com/example/AIDaS.git",
            "https://github.com:444/example/AIDaS.git",
            "https://github.com/example/AIDaS.git?other=repository",
            "https://github.com/example/AIDaS.git#fragment",
            "https://github.com/example/AIDaS/extra",
            "https://github.com/example/..",
            "https://github.com/example/%2e%2e",
            "https://github.com/example/repo name",
            "git@github.com:example/",
            "C:\\source\\AIDaS",
        )
        for origin in origins:
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                github_repository(origin)


class GitHubRequestTests(unittest.TestCase):
    def test_requests_exact_sha_on_main_with_timeout_without_cli_or_token(self):
        response = BytesIO(json.dumps({"workflow_runs": [validation_run()]}).encode())
        with (
            patch.dict("release_tools.preflight.os.environ", {}, clear=True),
            patch("release_tools.preflight.urlopen", return_value=response) as open_url,
        ):
            payload = fetch_validation_runs("example/AIDaS", COMMIT)
        self.assertEqual(payload["workflow_runs"][0]["head_sha"], COMMIT)
        request = open_url.call_args.args[0]
        self.assertTrue(request.full_url.startswith("https://api.github.com/repos/example/AIDaS/actions/workflows/release.yml/runs?"))
        self.assertIn(f"head_sha={COMMIT}", request.full_url)
        self.assertIn("branch=main&event=push", request.full_url)
        self.assertEqual(open_url.call_args.kwargs["timeout"], 30)
        self.assertIsNone(request.get_header("Authorization"))

    def test_optional_token_is_only_in_the_authorization_header(self):
        for values, expected in (
            ({"GH_TOKEN": "test-token"}, "test-token"),
            ({"GITHUB_TOKEN": "ci-token"}, "ci-token"),
            ({"GH_TOKEN": "preferred-token", "GITHUB_TOKEN": "other-token"}, "preferred-token"),
        ):
            with self.subTest(values=values):
                response = BytesIO(b'{"workflow_runs": []}')
                with (
                    patch.dict("release_tools.preflight.os.environ", values, clear=True),
                    patch("release_tools.preflight.urlopen", return_value=response) as open_url,
                ):
                    fetch_validation_runs("example/AIDaS", COMMIT)
                request = open_url.call_args.args[0]
                self.assertEqual(request.get_header("Authorization"), f"Bearer {expected}")
                self.assertNotIn(expected, request.full_url)

    def test_http_failure_reports_status_without_disclosing_token_or_response(self):
        failure = HTTPError("https://api.github.com/", 403, "secret response", {}, None)
        with (
            patch.dict("release_tools.preflight.os.environ", {"GH_TOKEN": "test-secret"}, clear=True),
            patch("release_tools.preflight.urlopen", side_effect=failure),
            self.assertRaisesRegex(ValueError, "HTTP 403") as caught,
        ):
            fetch_validation_runs("example/AIDaS", COMMIT)
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertNotIn("secret response", str(caught.exception))

    def test_network_failure_or_timeout_prevents_validation(self):
        for error in (URLError("offline"), TimeoutError("timed out"), OSError("network unavailable")):
            with self.subTest(error=error), patch("release_tools.preflight.urlopen", side_effect=error):
                with self.assertRaisesRegex(ValueError, "could not be reached"):
                    fetch_validation_runs("example/AIDaS", COMMIT)

    def test_invalid_github_response_fails_closed(self):
        for data in (b"not json", b"[]", b"{}", b'{"workflow_runs": null}'):
            with self.subTest(data=data), patch("release_tools.preflight.urlopen", return_value=BytesIO(data)):
                with self.assertRaisesRegex(ValueError, "valid workflow-run response"):
                    fetch_validation_runs("example/AIDaS", COMMIT)


if __name__ == "__main__":
    unittest.main()
