from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from aidas.ai.client import AIWorkerClient
from aidas.ai.worker import build_parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "OCT Segmenter" / "AI_ForAIDAS" / "model_img.onnx"
HAS_ONNX_RUNTIME = importlib.util.find_spec("onnxruntime") is not None


@unittest.skipUnless(HAS_ONNX_RUNTIME and MODEL_PATH.is_file(), "ONNX runtime model is unavailable")
class AIWorkerIntegrationTests(unittest.TestCase):
    def test_persistent_cpu_worker_reuses_one_process_for_dynamic_shapes(self):
        command = [sys.executable, "-m", "aidas.ai.worker"]
        with tempfile.TemporaryDirectory(prefix=".aidas_worker_test_", dir=PROJECT_ROOT) as temp_root:
            with AIWorkerClient(
                command,
                model_path=str(MODEL_PATH),
                provider_name="cpu",
                temp_root=temp_root,
            ) as client:
                process_id = client.process.pid
                first = client.predict(np.arange(65 * 127, dtype=np.uint16).reshape(65, 127))
                second = client.predict(np.arange(177 * 257, dtype=np.uint16).reshape(177, 257))

                self.assertEqual(client.process.pid, process_id)
                self.assertEqual(client._request_counter, 2)

        self.assertEqual(first["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(second["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(first["boundaries"].shape, (6, 127))
        self.assertEqual(second["boundaries"].shape, (6, 257))


class AIWorkerCommandLineTests(unittest.TestCase):
    def test_connect_token_can_start_with_dash(self):
        class _Listener:
            @staticmethod
            def getsockname():
                return ("127.0.0.1", 60551)

        client = AIWorkerClient(
            [sys.executable],
            model_path=str(MODEL_PATH),
            provider_name="cpu",
        )
        client._listener = _Listener()
        client._connect_token = "-token-starts-with-dash"
        args = build_parser().parse_args(client.worker_command[1:])
        self.assertEqual(args.connect_token, "-token-starts-with-dash")


if __name__ == "__main__":
    unittest.main()
