from __future__ import annotations

from contextlib import redirect_stderr
import io
import unittest
from unittest import mock

from aidas.ai.client import AIWorkerClient
from aidas.ai import worker
from aidas.ai.worker import build_parser


class _FakeListener:
    def getsockname(self):
        return "127.0.0.1", 43123


class AIWorkerClientCommandTests(unittest.TestCase):
    def _client_with_token(self, token):
        client = AIWorkerClient(
            ["aidas", "--aidas-ai-worker"],
            model_path="model.onnx",
        )
        client._listener = _FakeListener()
        client._connect_token = token
        return client

    def test_leading_hyphen_token_is_one_safe_argument(self):
        client = self._client_with_token("-token-that-looked-like-an-option")

        command = client.worker_command

        self.assertIn(
            "--connect-token=-token-that-looked-like-an-option",
            command,
        )
        parsed = build_parser().parse_args(command[2:])
        self.assertEqual(parsed.connect_token, "-token-that-looked-like-an-option")

    def test_command_diagnostics_redact_equals_form_token(self):
        token = "private-worker-token"
        client = self._client_with_token(token)

        command_line = client.command_line

        self.assertNotIn(token, command_line)
        self.assertIn("--connect-token=<redacted>", command_line)

    def test_generated_token_always_starts_with_safe_prefix(self):
        client = AIWorkerClient(["worker"], model_path="model.onnx")
        with mock.patch("aidas.ai.client.secrets.token_urlsafe", return_value="-unsafe"):
            client._open_listener()
        try:
            self.assertEqual(client._connect_token, "aidas_-unsafe")
        finally:
            client.close()

    def test_selected_core_limit_is_forwarded_to_the_worker(self):
        client = AIWorkerClient(
            ["worker"],
            model_path="model.onnx",
            core_limit=3,
        )
        client._listener = _FakeListener()
        client._connect_token = "aidas_token"

        command = client.worker_command

        self.assertEqual(command[command.index("--core-limit") + 1], "3")
        self.assertEqual(build_parser().parse_args(command[1:]).core_limit, 3)

    def test_worker_rejects_nonpositive_core_limit(self):
        parser = build_parser()

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--model", "model.onnx", "--core-limit", "0"])

    def test_client_rejects_nonpositive_core_limit(self):
        with self.assertRaisesRegex(ValueError, "one or greater"):
            AIWorkerClient(
                ["worker"],
                model_path="model.onnx",
                core_limit=0,
            )

    def test_device_probe_does_not_require_a_model(self):
        parsed = build_parser().parse_args(["--probe-device"])

        self.assertTrue(parsed.probe_device)
        self.assertIsNone(parsed.model)

    def test_device_probe_emits_machine_readable_compatibility(self):
        output = io.StringIO()
        with (
            mock.patch.object(
                worker,
                "probe_execution_devices",
                return_value={
                    "compatible_gpu": True,
                    "providers": ("DmlExecutionProvider", "CPUExecutionProvider"),
                },
            ),
            mock.patch.object(worker, "_OUTPUT_STREAM", output),
        ):
            self.assertEqual(worker._probe_device(), 0)

        self.assertIn('"compatible_gpu": true', output.getvalue())
        self.assertIn("DmlExecutionProvider", output.getvalue())


if __name__ == "__main__":
    unittest.main()
