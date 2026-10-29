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
    require_matching_tag,
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
            patch("release_tools.preflight.run_command", side_effect=["", COMMIT, "https://github.com/example/AIDaS.git", "", ""]) as command,
            patch("release_tools.preflight.fetch_validation_runs", return_value=payload) as fetch,
        ):
            commit, run = check_release("v3.4.5a1", project_root=self.root)
        self.assertEqual(commit, COMMIT)
        self.assertEqual(run["conclusion"], "success")
        self.assertEqual(command.call_args_list[0].args[0], ["git", "status", "--porcelain"])
        self.assertEqual(command.call_args_list[1].args[0], ["git", "rev-parse", "HEAD"])
        self.assertEqual(command.call_args_list[2].args[0], ["git", "remote", "get-url", "origin"])
        fetch.assert_called_once_with("example/AIDaS", COMMIT)
        self.assertEqual(command.call_count, 5)

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

    def test_existing_remote_tag_on_other_source_stops_before_github_validation(self):
        old_tag = f"{'b' * 40}\trefs/tags/v3.4.5a1"
        with (
            patch("release_tools.preflight.run_command", side_effect=["", COMMIT, "https://github.com/example/AIDaS.git", "", old_tag]),
            patch("release_tools.preflight.fetch_validation_runs") as fetch,
        ):
            with self.assertRaisesRegex(ValueError, "Existing remote tag.*does not identify HEAD"):
                check_release("v3.4.5a1", project_root=self.root)
        fetch.assert_not_called()

    def test_git_commands_have_a_timeout(self):
        result = subprocess.CompletedProcess(["git", "status"], 0, stdout="", stderr="")
        with patch("release_tools.preflight.subprocess.run", return_value=result) as command:
            run_command(["git", "status"], cwd=self.root)
        self.assertEqual(command.call_args.kwargs["timeout"], 30)

    def test_remote_git_timeout_is_reported_without_approving_a_release(self):
        failure = subprocess.TimeoutExpired(["git", "ls-remote"], 30)
        with patch("release_tools.preflight.subprocess.run", side_effect=failure):
            with self.assertRaisesRegex(ValueError, "did not finish within 30 seconds"):
                run_command(["git", "ls-remote"], cwd=self.root)


class ExistingTagTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".")
        self.tag = "v3.4.5a1"
        self.reference = f"refs/tags/{self.tag}"

    def test_absent_local_and_remote_tags_allow_a_proposed_release(self):
        with patch("release_tools.preflight.run_command", side_effect=["", ""]) as command:
            require_matching_tag(self.tag, COMMIT, project_root=self.root)
        self.assertEqual(command.call_args_list[0].args[0], ["git", "tag", "--list", self.tag])
        self.assertEqual(command.call_args_list[1].args[0], [
            "git", "ls-remote", "origin", self.reference, f"{self.reference}^{{}}",
        ])

    def test_existing_local_and_remote_lightweight_tags_on_head_allow_retry(self):
        remote = f"{COMMIT}\t{self.reference}"
        with patch("release_tools.preflight.run_command", side_effect=[self.tag, COMMIT, remote]) as command:
            require_matching_tag(self.tag, COMMIT, project_root=self.root)
        self.assertEqual(command.call_args_list[1].args[0], [
            "git", "rev-parse", "--verify", f"{self.reference}^{{commit}}",
        ])

    def test_annotated_remote_tag_uses_peeled_commit_instead_of_tag_object(self):
        remote = f"{'b' * 40}\t{self.reference}\n{COMMIT}\t{self.reference}^{{}}"
        with patch("release_tools.preflight.run_command", side_effect=["", remote]):
            require_matching_tag(self.tag, COMMIT, project_root=self.root)

    def test_local_tag_on_other_source_is_rejected_before_remote_lookup(self):
        with patch("release_tools.preflight.run_command", side_effect=[self.tag, "b" * 40]) as command:
            with self.assertRaisesRegex(ValueError, "Existing local tag.*does not identify HEAD"):
                require_matching_tag(self.tag, COMMIT, project_root=self.root)
        self.assertEqual(command.call_count, 2)

    def test_remote_tag_on_other_source_is_rejected_when_local_tag_is_absent(self):
        for remote in (
            f"{'b' * 40}\t{self.reference}",
            f"{'c' * 40}\t{self.reference}\n{'b' * 40}\t{self.reference}^{{}}",
        ):
            with self.subTest(remote=remote), patch("release_tools.preflight.run_command", side_effect=["", remote]):
                with self.assertRaisesRegex(ValueError, "Existing remote tag.*does not identify HEAD"):
                    require_matching_tag(self.tag, COMMIT, project_root=self.root)

    def test_matching_local_tag_does_not_override_mismatched_remote_tag(self):
        remote = f"{'b' * 40}\t{self.reference}"
        with patch("release_tools.preflight.run_command", side_effect=[self.tag, COMMIT, remote]):
            with self.assertRaisesRegex(ValueError, "Existing remote tag"):
                require_matching_tag(self.tag, COMMIT, project_root=self.root)

    def test_failed_remote_lookup_is_not_treated_as_an_absent_tag(self):
        with patch("release_tools.preflight.run_command", side_effect=["", ValueError("git: remote unavailable")]):
            with self.assertRaisesRegex(ValueError, "remote unavailable"):
                require_matching_tag(self.tag, COMMIT, project_root=self.root)

    def test_malformed_remote_tag_response_fails_closed(self):
        for remote in ("bad response", f"not-a-sha\t{self.reference}", f"{COMMIT}\trefs/heads/main"):
            with self.subTest(remote=remote), patch("release_tools.preflight.run_command", side_effect=["", remote]):
                with self.assertRaisesRegex(ValueError, "valid remote tag references"):
                    require_matching_tag(self.tag, COMMIT, project_root=self.root)


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
