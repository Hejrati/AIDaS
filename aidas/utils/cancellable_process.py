"""Run a read-only child process with prompt cancellation and pipe cleanup."""

from __future__ import annotations

from concurrent.futures import CancelledError
import subprocess
import time


_PIPE_DRAIN_TIMEOUT = 1.0
_PROCESS_REAP_TIMEOUT = 5.0


def run_cancellable_process(command, *, cancel_event, timeout=None, input=None, **popen_options):
    """Return a CompletedProcess, killing and reaping the child on cancellation.

    This is intended for checks, not package installers that may be writing
    files. Cleanup waits are bounded even if a descendant retains an inherited
    pipe after the direct child is killed. Preserve available timeout output.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError()
    if input is not None:
        if popen_options.get("stdin") is not None:
            raise ValueError("stdin and input arguments may not both be used")
        popen_options["stdin"] = subprocess.PIPE

    def stop(process, previous_output):
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                if process.poll() is None:
                    raise
        try:
            output = process.communicate(timeout=_PIPE_DRAIN_TIMEOUT)
        except subprocess.TimeoutExpired as error:
            output = (
                error.output if error.output is not None else previous_output[0],
                error.stderr if error.stderr is not None else previous_output[1],
            )
        except (OSError, ValueError):
            output = previous_output
        # A pipe drain timing out must not leave the direct child unreaped.
        try:
            process.wait(timeout=_PROCESS_REAP_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_PROCESS_REAP_TIMEOUT)
        return output

    def close_streams(process):
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name, None)
            reader = getattr(process, name + "_thread", None)
            if stream is None or (reader is not None and reader.is_alive()):
                # Windows communicate readers own their streams until EOF.
                # Closing here can wait forever for the reader's file lock;
                # the daemon reader closes its own stream when it finishes.
                continue
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    started = time.monotonic()
    process = subprocess.Popen(command, **popen_options)
    previous_output = (None, None)
    try:
        first_communication = True
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError()
                remaining = None if timeout is None else timeout - (time.monotonic() - started)
                if remaining is not None and remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        command, timeout, output=previous_output[0], stderr=previous_output[1],
                    )
                wait = 0.1 if remaining is None else min(0.1, remaining)
                try:
                    stdout, stderr = process.communicate(
                        input=input if first_communication else None, timeout=wait,
                    )
                except subprocess.TimeoutExpired as error:
                    previous_output = (
                        error.output if error.output is not None else previous_output[0],
                        error.stderr if error.stderr is not None else previous_output[1],
                    )
                    first_communication = False
                    continue
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError()
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except BaseException as error:
            stdout, stderr = stop(process, previous_output)
            if isinstance(error, subprocess.TimeoutExpired):
                error.output = stdout
                error.stderr = stderr
            raise
    finally:
        close_streams(process)
