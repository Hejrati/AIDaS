"""Persistent subprocess client for the isolated AIDaS ONNX worker."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time

import numpy as np


class AIWorkerClient:
    """Own one worker process and reuse its loaded model for a whole batch."""

    def __init__(
        self,
        command,
        *,
        model_path,
        provider_name="auto",
        device_id=0,
        env=None,
        popen_kwargs=None,
        startup_progress_callback=None,
        startup_timeout=120,
        temp_root=None,
    ):
        self.command = [os.fspath(part) for part in command]
        self.model_path = os.path.abspath(model_path)
        self.provider_name = str(provider_name)
        self.device_id = int(device_id)
        self.env = env
        self.popen_kwargs = dict(popen_kwargs or {})
        self.startup_progress_callback = startup_progress_callback
        self.startup_timeout = max(1.0, float(startup_timeout))
        self.temp_root = None if temp_root is None else os.path.abspath(temp_root)
        self.process = None
        self.temp_dir = None
        self.startup_result = None
        self._request_counter = 0
        self._output_lines = []
        self._listener = None
        self._connection = None
        self._reader = None
        self._writer = None
        self._connect_token = None
        self._worker_error_log = None

    @property
    def worker_command(self):
        if self._listener is None or self._connect_token is None:
            raise RuntimeError("ONNX worker IPC listener has not been initialized.")
        host, port = self._listener.getsockname()
        return self.command + [
            "--model",
            self.model_path,
            "--provider",
            self.provider_name,
            "--device-id",
            str(self.device_id),
            "--connect-host",
            str(host),
            "--connect-port",
            str(port),
            "--connect-token",
            self._connect_token,
        ]

    @property
    def command_line(self):
        command = self.worker_command
        redacted = list(command)
        try:
            token_index = redacted.index("--connect-token") + 1
            redacted[token_index] = "<redacted>"
        except (ValueError, IndexError):
            pass
        return subprocess.list2cmdline(redacted)

    def _open_listener(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(0.25)
        self._listener = listener
        self._connect_token = secrets.token_urlsafe(32)

    def start(self):
        if self.process is not None:
            return self.startup_result
        self.temp_dir = tempfile.mkdtemp(prefix="aidas_onnx_worker_", dir=self.temp_root)
        self._worker_error_log = os.path.join(self.temp_dir, "startup_error.log")
        try:
            self._open_listener()
        except Exception:
            self.close()
            raise
        worker_env = dict(os.environ if self.env is None else self.env)
        worker_env["AIDAS_AI_WORKER_ERROR_LOG"] = self._worker_error_log
        kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": worker_env,
            **self.popen_kwargs,
        }
        try:
            self.process = subprocess.Popen(self.worker_command, **kwargs)
        except Exception:
            self.close()
            raise

        try:
            deadline = time.monotonic() + self.startup_timeout
            while True:
                try:
                    self._connection, _address = self._listener.accept()
                    break
                except socket.timeout:
                    return_code = self.process.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            "ONNX worker exited before connecting to AIDaS.\n\n"
                            f"Command: {self.command_line}\n"
                            f"Return code: {return_code}\n\n"
                            f"Worker diagnostics:\n{self._read_worker_diagnostics()}"
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Timed out waiting for the ONNX worker to connect to AIDaS.\n\n"
                            f"Command: {self.command_line}\n"
                            f"Timeout: {self.startup_timeout:.0f} seconds\n\n"
                            f"Worker diagnostics:\n{self._read_worker_diagnostics()}"
                        )
            self._connection.settimeout(None)
            self._reader = self._connection.makefile("r", encoding="utf-8", newline="\n")
            self._writer = self._connection.makefile("w", encoding="utf-8", newline="\n")

            hello = self._read_message()
            if hello.get("type") != "hello" or not secrets.compare_digest(
                str(hello.get("token") or ""),
                self._connect_token,
            ):
                raise RuntimeError("Rejected an unauthenticated ONNX worker connection.")

            while True:
                message = self._read_message()
                message_type = message.get("type")
                if message_type == "progress" and self.startup_progress_callback is not None:
                    self.startup_progress_callback(
                        float(message.get("fraction", 0.0)),
                        str(message.get("stage") or ""),
                    )
                elif message_type == "ready":
                    self.startup_result = message
                    return message
                elif message_type in {"fatal", "error"}:
                    raise RuntimeError(
                        self._format_worker_error(message, "ONNX worker failed to start")
                    )
        except Exception:
            self.close()
            raise

    def _read_message(self):
        if self.process is None or self._reader is None:
            raise RuntimeError("ONNX worker process is not running.")
        while True:
            line = self._reader.readline()
            if line == "":
                return_code = self.process.poll()
                details = "\n".join(self._output_lines[-30:]) or "(worker produced no output)"
                raise RuntimeError(
                    "ONNX worker exited unexpectedly.\n\n"
                    f"Command: {self.command_line}\n"
                    f"Return code: {return_code}\n\n{details}"
                )
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                self._output_lines.append(text)
                continue
            if message.get("type") == "hello" and "token" in message:
                redacted = dict(message)
                redacted["token"] = "<redacted>"
                self._output_lines.append(json.dumps(redacted))
            else:
                self._output_lines.append(text)
            return message

    @staticmethod
    def _format_worker_error(message, heading):
        details = str(message.get("error") or "Unknown worker error")
        worker_traceback = str(message.get("traceback") or "").strip()
        if worker_traceback:
            details += f"\n\nWorker traceback:\n{worker_traceback}"
        return f"{heading}.\n\n{details}"

    def _read_worker_diagnostics(self):
        path = self._worker_error_log
        if not path or not os.path.isfile(path):
            return "(the worker did not create a diagnostic log)"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as stream:
                return stream.read().strip() or "(the worker diagnostic log is empty)"
        except OSError as exc:
            return f"(could not read the worker diagnostic log: {exc})"

    def predict(self, image, *, progress_callback=None):
        if self.process is None:
            self.start()
        if self.process is None or self._writer is None or self.temp_dir is None:
            raise RuntimeError("ONNX worker process is not available.")

        self._request_counter += 1
        request_id = str(self._request_counter)
        image_path = os.path.join(self.temp_dir, f"image_{request_id}.npy")
        output_path = os.path.join(self.temp_dir, f"prediction_{request_id}.npz")
        np.save(image_path, np.ascontiguousarray(image))

        request = {
            "type": "predict",
            "request_id": request_id,
            "image_npy": image_path,
            "output_npz": output_path,
        }
        self._writer.write(json.dumps(request) + "\n")
        self._writer.flush()

        try:
            while True:
                message = self._read_message()
                if str(message.get("request_id") or "") != request_id:
                    continue
                message_type = message.get("type")
                if message_type == "progress" and progress_callback is not None:
                    progress_callback(
                        float(message.get("fraction", 0.0)),
                        str(message.get("stage") or ""),
                    )
                elif message_type == "error":
                    raise RuntimeError(
                        self._format_worker_error(message, "AI_ForAIDAS ONNX prediction failed")
                    )
                elif message_type == "result":
                    if not os.path.isfile(output_path):
                        raise RuntimeError(
                            "ONNX worker reported success without writing prediction output.\n\n"
                            f"Command: {self.command_line}"
                        )
                    return self._load_prediction(output_path)
        finally:
            for path in (image_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

    def _load_prediction(self, output_path):
        with np.load(output_path) as data:
            fovea_arr = np.asarray(data["fovea_x"])
            fovea_x = int(fovea_arr[0]) if fovea_arr.size else -1
            device_arr = np.asarray(data["device"])
            device = str(device_arr[0]) if device_arr.size else "unknown"
            provider_arr = np.asarray(data.get("execution_provider", []))
            execution_provider = str(provider_arr[0]) if provider_arr.size else "unknown"
            fallback_arr = np.asarray(data.get("fallback_reason", []))
            fallback_reason = (
                str(fallback_arr[0])
                if fallback_arr.size and str(fallback_arr[0])
                else None
            )
            return {
                "boundaries": np.asarray(data["boundaries"], dtype=np.float32),
                "fovea_x": None if fovea_x < 0 else fovea_x,
                "device": device,
                "execution_provider": execution_provider,
                "fallback_reason": fallback_reason,
                "stdout": "\n".join(self._output_lines),
                "stderr": "",
                "command": self.command_line,
            }

    def close(self):
        process = self.process
        self.process = None
        if process is not None:
            if process.poll() is None and self._writer is not None:
                try:
                    self._writer.write(json.dumps({"type": "shutdown"}) + "\n")
                    self._writer.flush()
                    process.wait(timeout=5)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        for stream_name in ("_reader", "_writer"):
            stream = getattr(self, stream_name)
            setattr(self, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for socket_name in ("_connection", "_listener"):
            socket_object = getattr(self, socket_name)
            setattr(self, socket_name, None)
            if socket_object is not None:
                try:
                    socket_object.close()
                except OSError:
                    pass
        self._connect_token = None
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
        self._worker_error_log = None

    def __enter__(self):
        try:
            self.start()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()
        return False
