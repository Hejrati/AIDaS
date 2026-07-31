from __future__ import annotations

import unittest
from unittest import mock

from aidas.ai.client import AIWorkerClient
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


if __name__ == "__main__":
    unittest.main()
