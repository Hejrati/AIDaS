from concurrent.futures import CancelledError
import subprocess
import sys
import threading
import unittest
from unittest import mock

from aidas.utils.cancellable_process import run_cancellable_process


class CancellableProcessTests(unittest.TestCase):
    def test_pre_cancelled_process_never_launches(self):
        cancellation = threading.Event()
        cancellation.set()
        with mock.patch("aidas.utils.cancellable_process.subprocess.Popen") as popen:
            with self.assertRaises(CancelledError):
                run_cancellable_process([sys.executable, "-c", "pass"], cancel_event=cancellation)
        popen.assert_not_called()

    def test_complete_process_preserves_arguments_status_and_captured_output(self):
        command = [sys.executable, "-c", "import sys; print('out'); print('err',file=sys.stderr); sys.exit(3)"]
        result = run_cancellable_process(
            command, cancel_event=threading.Event(), timeout=10,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(result.args, command)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout.strip(), "out")
        self.assertEqual(result.stderr.strip(), "err")

    def test_running_child_is_cancelled_and_reaped_without_blocked_pipes(self):
        cancellation = threading.Event()
        processes = []
        timers = []
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            timer = threading.Timer(0.25, cancellation.set)
            timers.append(timer)
            timer.start()
            return process

        try:
            with mock.patch("aidas.utils.cancellable_process.subprocess.Popen", side_effect=launch):
                with self.assertRaises(CancelledError):
                    run_cancellable_process(
                        [sys.executable, "-c", "import sys,time; sys.stdout.write('x'*200000); sys.stdout.flush(); sys.stderr.write('y'*200000); sys.stderr.flush(); time.sleep(30)"],
                        cancel_event=cancellation, timeout=10,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
            self.assertEqual(len(processes), 1)
            self.assertIsNotNone(processes[0].returncode)
            self.assertIsNotNone(processes[0].poll())
        finally:
            for timer in timers:
                timer.cancel()
                timer.join()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_timeout_kills_and_reaps_child_and_keeps_output(self):
        processes = []
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            self.addCleanup(self._reap, process)
            return process

        with mock.patch("aidas.utils.cancellable_process.subprocess.Popen", side_effect=launch):
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                run_cancellable_process(
                    [sys.executable, "-c", "import time; print('started',flush=True); time.sleep(30)"],
                    cancel_event=threading.Event(), timeout=2,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        self.assertIn("started", caught.exception.output)

    @staticmethod
    def _reap(process):
        if process.poll() is None:
            process.kill()
        process.communicate()

    def test_input_is_sent_once_across_repeated_communicate_waits(self):
        result = run_cancellable_process(
            [sys.executable, "-c", "import sys,time; data=sys.stdin.read(); time.sleep(.25); print(data.upper())"],
            cancel_event=threading.Event(), timeout=10, input="hello",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "HELLO")


if __name__ == "__main__":
    unittest.main()
