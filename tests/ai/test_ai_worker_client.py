from __future__ import annotations

from concurrent.futures import CancelledError
from contextlib import contextmanager, redirect_stderr
import io
import json
from pathlib import Path
import socket
import subprocess
import threading
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from aidas.ai.client import AIWorkerClient
from aidas.ai import client as worker_client, worker
from aidas.ai.worker import build_parser


class _FakeListener:
    def getsockname(self):
        return "127.0.0.1", 43123


class AIWorkerClientCommandTests(unittest.TestCase):
    def test_worker_prediction_archive_is_uncompressed_local_ipc(self):
        prediction = SimpleNamespace(
            boundaries=np.arange(12, dtype=np.float32).reshape(3, 4),
            fovea_x=2,
            device="CPU",
            execution_provider="CPUExecutionProvider",
            fallback_reason=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "prediction.npz"

            worker._write_prediction(output, prediction)

            with zipfile.ZipFile(output) as archive:
                self.assertTrue(archive.infolist())
                self.assertTrue(
                    all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
                )

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


class AIWorkerClientDeadlineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    @staticmethod
    def _message(kind, **fields):
        return (json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n").encode("utf-8")

    def _process(self, stop=None):
        process = mock.Mock()
        state = {"returncode": None}
        process.poll.side_effect = lambda: state["returncode"]

        def terminate():
            state["returncode"] = -15
            if stop is not None:
                stop.set()

        def wait(timeout=None):
            if state["returncode"] is None:
                raise subprocess.TimeoutExpired(["worker"], timeout)
            return state["returncode"]

        process.terminate.side_effect = terminate
        process.kill.side_effect = terminate
        process.wait.side_effect = wait
        return process, state

    def _client(self, chunks=(), *, cancel_event=None):
        client = AIWorkerClient(
            ["worker"], model_path="model.onnx", startup_timeout=1,
            prediction_timeout=1, cancel_event=cancel_event, temp_root=self.root,
        )
        clock = SimpleNamespace(now=0.0)
        connection = mock.Mock()
        connection.chunks = list(chunks)
        connection.makefile.return_value = io.StringIO()

        def receive(_size):
            delay, payload = connection.chunks.pop(0)
            clock.now += delay
            return payload

        connection.recv.side_effect = receive
        listener = mock.Mock()
        listener.getsockname.return_value = ("127.0.0.1", 43123)
        listener.accept.return_value = (connection, ("127.0.0.1", 12345))

        def open_listener():
            client._listener = listener
            client._connect_token = "aidas_test_token"

        client._open_listener = open_listener
        process, state = self._process()
        self.addCleanup(client.close)
        return client, connection, clock, process, state

    @contextmanager
    def _clocked(self, connection, clock):
        def wait(readers, _writers, _errors, timeout):
            if connection.chunks:
                return readers, [], []
            clock.now += timeout
            return [], [], []

        with mock.patch.object(worker_client.time, "monotonic", side_effect=lambda: clock.now), \
                mock.patch.object(worker_client.select, "select", side_effect=wait):
            yield

    def _prediction_client(self, chunks=(), *, cancel_event=None):
        client, connection, clock, process, state = self._client(chunks, cancel_event=cancel_event)
        client._open_listener()
        client._connection = connection
        client._writer = io.StringIO()
        client.process = process
        client.startup_result = {"type": "ready"}
        client.temp_dir = str(self.root / "ipc")
        Path(client.temp_dir).mkdir()
        return client, connection, clock, process, state

    def test_nonfinite_timeout_configuration_cannot_disable_the_deadlines(self):
        for option in ("startup_timeout", "prediction_timeout"):
            for value in (float("inf"), float("nan")):
                with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, "finite"):
                    AIWorkerClient(["worker"], model_path="model.onnx", **{option: value})

    def test_absolute_startup_deadline_is_not_reset_by_progress_junk_or_partial_lines(self):
        hello = self._message("hello", token="aidas_test_token")
        traffic = {
            "progress": self._message("progress", fraction=0.4, stage="loading_model"),
            "malformed": b"not JSON\n[]\n",
            "partial_line": b"x",
        }
        for kind, payload in traffic.items():
            with self.subTest(traffic=kind):
                client, connection, clock, process, _state = self._client(
                    [(0.2, hello), *[(0.2, payload)] * 20],
                )
                client.startup_progress_callback = mock.Mock()
                with self._clocked(connection, clock), \
                        mock.patch.object(worker_client.subprocess, "Popen", return_value=process):
                    with self.assertRaisesRegex(TimeoutError, "finish starting"):
                        client.start()
                self.assertLess(clock.now, 1.4)
                process.terminate.assert_called_once()
                self.assertIsNone(client.process)
                self.assertIsNone(client.temp_dir)

    def test_normal_fragmented_utf8_handshake_keeps_progress_and_authentication(self):
        progress = self._message("progress", fraction=0.4, stage="Préparation")
        split = progress.index(b"\xc3") + 1
        ready = {"type": "ready", "execution_provider": "CPUExecutionProvider"}
        client, connection, clock, process, _state = self._client([
            (0.1, self._message("hello", token="aidas_test_token")),
            (0.1, progress[:split]), (0.1, progress[split:]),
            (0.1, self._message("ready", execution_provider="CPUExecutionProvider")),
        ])
        callback = client.startup_progress_callback = mock.Mock()

        with self._clocked(connection, clock), \
                mock.patch.object(worker_client.subprocess, "Popen", return_value=process):
            self.assertEqual(client.start(), ready)
            self.assertEqual(client.start(), ready)

        callback.assert_called_once_with(0.4, "Préparation")
        self.assertNotIn("aidas_test_token", "\n".join(client._output_lines))
        self.assertIs(client.process, process)

    def test_real_connected_worker_stall_times_out_before_hello_or_ready(self):
        for authenticate in (False, True):
            with self.subTest(authenticate=authenticate):
                stop = threading.Event()
                process, _state = self._process(stop)
                peers = []

                def launch(command, **_kwargs):
                    host = command[command.index("--connect-host") + 1]
                    port = int(command[command.index("--connect-port") + 1])
                    token = next(part.split("=", 1)[1] for part in command if part.startswith("--connect-token="))

                    def peer():
                        with socket.create_connection((host, port), timeout=2) as connection:
                            if authenticate:
                                connection.sendall(self._message("hello", token=token))
                            stop.wait(3)

                    thread = threading.Thread(target=peer, daemon=True)
                    peers.append(thread)
                    thread.start()
                    return process

                client = AIWorkerClient(["worker"], model_path="model.onnx", startup_timeout=1, temp_root=self.root)
                try:
                    with mock.patch.object(worker_client.subprocess, "Popen", side_effect=launch):
                        with self.assertRaisesRegex(TimeoutError, "finish starting"):
                            client.start()
                    process.terminate.assert_called_once()
                    self.assertIsNone(client.process)
                    self.assertIsNone(client.temp_dir)
                finally:
                    stop.set()
                    client.close()
                    for peer in peers:
                        peer.join(timeout=3)
                        self.assertFalse(peer.is_alive())

    def test_invalid_wrong_request_or_repeated_progress_cannot_keep_prediction_alive(self):
        traffic = {
            "wrong_request": self._message("progress", request_id="other", fraction=0.5, stage="inference"),
            "invalid_fraction": self._message("progress", request_id="1", fraction=float("nan"), stage="inference"),
            "unrelated": self._message("heartbeat", request_id="1"),
            "malformed": b"not JSON\n",
            "repeated": self._message("progress", request_id="1", fraction=0.5, stage="inference"),
        }
        for kind, payload in traffic.items():
            with self.subTest(traffic=kind):
                client, connection, clock, _process, _state = self._prediction_client([(0.4, payload)] * 20)
                with self._clocked(connection, clock):
                    with self.assertRaisesRegex(TimeoutError, "prediction progress"):
                        client.predict(np.ones((2, 2), dtype=np.uint16))
                self.assertLess(clock.now, 2.0)
                self.assertEqual(list(Path(client.temp_dir).iterdir()), [])
                client.close()

    def test_advancing_matching_progress_allows_long_inference_and_preserves_prediction_values(self):
        chunks = [(0.6, self._message("progress", request_id="1", fraction=value, stage="inference"))
                  for value in (0.2, 0.4, 0.8)]
        chunks.append((0.6, self._message("result", request_id="1")))
        client, connection, clock, _process, _state = self._prediction_client(chunks)
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)
        output = Path(client.temp_dir) / "prediction_1.npz"
        np.savez(output, boundaries=expected, fovea_x=np.array([2]), device=np.array(["CPU"]),
                 execution_provider=np.array(["CPUExecutionProvider"]), fallback_reason=np.array([""]))
        image = np.arange(16, dtype=np.uint16).reshape(4, 4)
        before = image.tobytes()
        callback = mock.Mock()

        with self._clocked(connection, clock):
            result = client.predict(image, progress_callback=callback)

        self.assertGreater(clock.now, client.prediction_timeout)
        self.assertEqual(callback.call_count, 3)
        np.testing.assert_array_equal(result["boundaries"], expected)
        self.assertEqual(result["fovea_x"], 2)
        self.assertEqual(result["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(list(Path(client.temp_dir).iterdir()), [])

    def test_prediction_detects_dead_worker_without_waiting_for_inactivity_limit(self):
        client, connection, clock, _process, state = self._prediction_client()
        client.prediction_timeout = 1800
        state["returncode"] = 7

        with self._clocked(connection, clock):
            with self.assertRaisesRegex(RuntimeError, "Return code: 7"):
                client.predict(np.zeros((2, 2)))

        self.assertLess(clock.now, 1)
        self.assertEqual(list(Path(client.temp_dir).iterdir()), [])

    def test_cancelled_start_never_launches_and_waiting_prediction_cancels_promptly(self):
        cancelled = threading.Event()
        cancelled.set()
        client, _connection, _clock, process, _state = self._client(cancel_event=cancelled)
        with mock.patch.object(worker_client.subprocess, "Popen") as launch:
            with self.assertRaises(CancelledError):
                client.start()
        launch.assert_not_called()
        self.assertIsNone(client.temp_dir)

        cancelled.clear()
        client, connection, clock, process, _state = self._prediction_client(cancel_event=cancelled)

        def cancel_while_waiting(_readers, _writers, _errors, timeout):
            clock.now += timeout
            cancelled.set()
            return [], [], []

        with self._clocked(connection, clock), \
                mock.patch.object(worker_client.select, "select", side_effect=cancel_while_waiting):
            with self.assertRaises(CancelledError):
                client.predict(np.zeros((2, 2)))

        self.assertLess(clock.now, 1)
        self.assertEqual(list(Path(client.temp_dir).iterdir()), [])
        client.close()
        process.terminate.assert_called_once()

    def test_close_terminates_an_unconnected_worker_without_initialized_streams(self):
        client, _connection, _clock, process, state = self._client()
        client.process = process

        client.close()

        process.terminate.assert_called_once()
        self.assertEqual(state["returncode"], -15)
        self.assertIsNone(client.process)

    def test_close_kills_a_worker_that_does_not_finish_after_termination(self):
        client, _connection, _clock, _process, _state = self._client()
        process = client.process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(["worker"], 5), -9]

        client.close()

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=5), mock.call(timeout=5)])
        self.assertIsNone(client.process)


if __name__ == "__main__":
    unittest.main()
