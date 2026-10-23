"""Run a read-only child process with prompt cancellation and pipe cleanup."""

from __future__ import annotations

from concurrent.futures import CancelledError
import subprocess
import time


def run_cancellable_process(command, *, cancel_event, timeout=None, input=None, **popen_options):
    """Return a CompletedProcess, killing and reaping the child on cancellation.

    This is intended for checks, not package installers that may be writing
    files. Pipe output is drained with ``communicate`` even after a timeout or
    cancellation so the child cannot remain blocked or running in the background.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError()
    if input is not None:
        if popen_options.get("stdin") is not None:
            raise ValueError("stdin and input arguments may not both be used")
        popen_options["stdin"] = subprocess.PIPE

    def stop(process):
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                if process.poll() is None:
                    raise
        return process.communicate()

    started = time.monotonic()
    with subprocess.Popen(command, **popen_options) as process:
        first_communication = True
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError()
                remaining = None if timeout is None else timeout - (time.monotonic() - started)
                if remaining is not None and remaining <= 0:
                    stdout, stderr = stop(process)
                    raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
                wait = 0.1 if remaining is None else min(0.1, remaining)
                try:
                    stdout, stderr = process.communicate(
                        input=input if first_communication else None, timeout=wait,
                    )
                except subprocess.TimeoutExpired:
                    first_communication = False
                    continue
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError()
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except BaseException:
            stop(process)
            raise
