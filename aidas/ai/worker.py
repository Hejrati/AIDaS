"""Command-line bridge for isolated ONNX AI_ForAIDAS inference."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback

import numpy as np

from aidas.ai.inference import AIForAIDASPredictor


_OUTPUT_STREAM = sys.stdout


def _emit(message_type, **values):
    if _OUTPUT_STREAM is None:
        raise RuntimeError("The AI worker has no IPC output stream.")
    _OUTPUT_STREAM.write(json.dumps({"type": message_type, **values}) + "\n")
    _OUTPUT_STREAM.flush()


def _emit_progress(stage, fraction, *, request_id=None):
    message = {
        "stage": str(stage),
        "fraction": max(0.0, min(float(fraction), 1.0)),
    }
    if request_id is not None:
        message["request_id"] = str(request_id)
    _emit("progress", **message)


def _write_prediction(output_path, prediction):
    fovea_x = -1 if prediction.fovea_x is None else int(prediction.fovea_x)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        boundaries=prediction.boundaries,
        fovea_x=np.array([fovea_x], dtype=np.int64),
        device=np.array([prediction.device]),
        execution_provider=np.array([prediction.execution_provider]),
        fallback_reason=np.array([prediction.fallback_reason or ""]),
    )


def _result_message(prediction, output_path, *, request_id=None):
    message = {
        "boundaries_shape": list(prediction.boundaries.shape),
        "device": prediction.device,
        "execution_provider": prediction.execution_provider,
        "fallback_reason": prediction.fallback_reason,
        "fovea_x": None if prediction.fovea_x is None else int(prediction.fovea_x),
        "output_npz": os.path.abspath(output_path),
    }
    if request_id is not None:
        message["request_id"] = str(request_id)
    return message


def _build_predictor(args, progress_callback):
    return AIForAIDASPredictor(
        boundary_model_path=args.model,
        provider_name=args.provider,
        device_id=args.device_id,
        progress_callback=progress_callback,
    )


def _run_once(args):
    _emit_progress("loading_image", 0.04)
    image = np.load(args.image_npy)
    _emit_progress("image_loaded", 0.08)
    predictor = _build_predictor(args, _emit_progress)
    prediction = predictor.predict(image)
    _emit_progress("prediction_ready", 0.95)
    _emit_progress("writing_output", 0.97)
    _write_prediction(args.output_npz, prediction)
    _emit_progress("done", 1.0)
    _emit("result", **_result_message(prediction, args.output_npz))
    return 0


def _serve_requests(args, input_stream):
    """Keep one ONNX session alive and process newline-delimited requests."""
    active_request_id = None

    def report_progress(stage, fraction):
        _emit_progress(stage, fraction, request_id=active_request_id)

    try:
        predictor = _build_predictor(args, report_progress)
    except Exception as exc:
        _emit("fatal", error=str(exc), traceback=traceback.format_exc())
        return 1

    _emit(
        "ready",
        device=predictor.device,
        execution_provider=predictor.execution_provider,
        fallback_reason=predictor.fallback_reason,
    )

    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit("error", request_id=None, error=f"Invalid JSON request: {exc}")
            continue

        request_type = str(request.get("type") or "")
        if request_type == "shutdown":
            _emit("stopped")
            return 0
        if request_type != "predict":
            _emit(
                "error",
                request_id=request.get("request_id"),
                error=f"Unsupported worker request type: {request_type or '(missing)'}",
            )
            continue

        active_request_id = str(request.get("request_id") or "")
        try:
            image_path = os.fspath(request["image_npy"])
            output_path = os.fspath(request["output_npz"])
            _emit_progress("loading_image", 0.04, request_id=active_request_id)
            image = np.load(image_path)
            _emit_progress("image_loaded", 0.08, request_id=active_request_id)
            prediction = predictor.predict(image)
            _emit_progress("prediction_ready", 0.95, request_id=active_request_id)
            _emit_progress("writing_output", 0.97, request_id=active_request_id)
            _write_prediction(output_path, prediction)
            _emit_progress("done", 1.0, request_id=active_request_id)
            _emit(
                "result",
                **_result_message(prediction, output_path, request_id=active_request_id),
            )
        except Exception as exc:
            _emit(
                "error",
                request_id=active_request_id,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        finally:
            active_request_id = None

    return 0


def _serve_jsonl(args):
    return _serve_requests(args, sys.stdin)


def _serve_tcp(args):
    """Connect to the GUI's authenticated loopback IPC listener."""
    global _OUTPUT_STREAM
    try:
        connection = socket.create_connection(
            (args.connect_host, args.connect_port),
            timeout=30,
        )
        connection.settimeout(None)
        input_stream = connection.makefile("r", encoding="utf-8", newline="\n")
        output_stream = connection.makefile("w", encoding="utf-8", newline="\n")
        _OUTPUT_STREAM = output_stream
        _emit("hello", token=args.connect_token)
        return _serve_requests(args, input_stream)
    except Exception:
        return 1
    finally:
        for stream_name in ("input_stream", "output_stream"):
            stream = locals().get(stream_name)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        connection = locals().get("connection")
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass


def build_parser():
    parser = argparse.ArgumentParser(description="Run ONNX AI_ForAIDAS inference.")
    parser.add_argument("--image-npy", help="Input 2-D image saved with numpy.save")
    parser.add_argument("--model", required=True, help="AI_ForAIDAS boundary model .onnx")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=("auto", "dml", "cpu"),
        help="ONNX execution provider; auto prefers DirectML and falls back to CPU",
    )
    parser.add_argument(
        "--device-id",
        default=0,
        type=int,
        help="DirectML adapter index (0 is the Windows default adapter)",
    )
    parser.add_argument("--output-npz", help="Output .npz path for predicted arrays")
    parser.add_argument(
        "--serve-jsonl",
        action="store_true",
        help="Keep the model loaded and accept predict requests on standard input",
    )
    parser.add_argument("--connect-host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--connect-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--connect-token", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.connect_port is not None:
        if not args.connect_token:
            raise SystemExit("--connect-token is required with --connect-port")
        return _serve_tcp(args)
    if args.serve_jsonl:
        return _serve_jsonl(args)
    if not args.image_npy or not args.output_npz:
        raise SystemExit("--image-npy and --output-npz are required outside --serve-jsonl mode")
    return _run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
