from concurrent.futures import CancelledError
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
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

    def test_inherited_pipe_cleanup_is_bounded_and_keeps_available_timeout_output(self):
        for reason in ("timeout", "cancel"):
            for latest_output in (None, b"complete partial stdout"):
                with self.subTest(reason=reason, latest_output=latest_output):
                    cancellation = threading.Event()
                    process = mock.MagicMock()
                    process.stdin = None
                    process.returncode = None
                    process.poll.side_effect = lambda: process.returncode
                    process.kill.side_effect = lambda: setattr(process, "returncode", -9)
                    process.wait.return_value = -9
                    for name in ("stdout", "stderr"):
                        stream = getattr(process, name)
                        stream.close.side_effect = AssertionError("Cannot close a pipe locked by its reader")
                        getattr(process, name + "_thread").is_alive.return_value = True
                    first = subprocess.TimeoutExpired(
                        ["worker"], 0.1, output=b"partial stdout", stderr=b"partial stderr",
                    )
                    final = subprocess.TimeoutExpired(
                        ["worker"], 1, output=latest_output, stderr=b"latest partial stderr",
                    )

                    def communicate(**_options):
                        if process.communicate.call_count == 1:
                            if reason == "cancel":
                                cancellation.set()
                            raise first
                        raise final

                    process.communicate.side_effect = communicate
                    expected = subprocess.TimeoutExpired if reason == "timeout" else CancelledError
                    with mock.patch("aidas.utils.cancellable_process.subprocess.Popen", return_value=process), \
                            mock.patch("aidas.utils.cancellable_process.time.monotonic", side_effect=[0, 0, 1]):
                        with self.assertRaises(expected) as raised:
                            run_cancellable_process(["worker"], cancel_event=cancellation, timeout=0.5)

                    if reason == "timeout":
                        self.assertEqual(raised.exception.output, latest_output or b"partial stdout")
                        self.assertEqual(raised.exception.stderr, b"latest partial stderr")
                    process.kill.assert_called_once_with()
                    self.assertEqual(process.communicate.call_count, 2)
                    self.assertTrue(all(call.kwargs["timeout"] > 0 for call in process.communicate.call_args_list))
                    process.wait.assert_called_once_with(timeout=mock.ANY)
                    self.assertGreater(process.wait.call_args.kwargs["timeout"], 0)
                    process.stdout.close.assert_not_called()
                    process.stderr.close.assert_not_called()
                    process.__exit__.assert_not_called()

    def test_cleanup_reaps_again_with_a_finite_wait_if_the_child_has_not_finished(self):
        process = mock.Mock(stdin=None, stdout=None, stderr=None)
        process.poll.return_value = None
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["worker"], 0.1),
            subprocess.TimeoutExpired(["worker"], 1),
        ]
        process.wait.side_effect = [subprocess.TimeoutExpired(["worker"], 5), -9]

        with mock.patch("aidas.utils.cancellable_process.subprocess.Popen", return_value=process), \
                mock.patch("aidas.utils.cancellable_process.time.monotonic", side_effect=[0, 0, 1]):
            with self.assertRaises(subprocess.TimeoutExpired):
                run_cancellable_process(["worker"], cancel_event=threading.Event(), timeout=0.5)

        self.assertEqual(process.kill.call_count, 2)
        self.assertEqual(process.wait.call_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] > 0 for call in process.wait.call_args_list))

    def test_cancellation_returns_while_a_real_descendant_still_holds_inherited_pipes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ready, stop, finished = (root / name for name in ("ready", "stop", "finished"))
            descendant = (
                "import pathlib,sys,time\n"
                "root=pathlib.Path(sys.argv[1])\n"
                "(root/'ready').write_text('ready')\n"
                "deadline=time.monotonic()+10\n"
                "while not (root/'stop').exists() and time.monotonic()<deadline:\n"
                " time.sleep(.02)\n"
                "(root/'finished').write_text('finished')\n"
            )
            parent = (
                "import subprocess,sys,time\n"
                f"subprocess.Popen([sys.executable,'-c',{descendant!r},sys.argv[1]],"
                "stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=sys.stderr,"
                "creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))\n"
                "print('parent output',flush=True)\n"
                "print('parent error',file=sys.stderr,flush=True)\n"
                "time.sleep(30)\n"
            )
            cancellation = threading.Event()
            processes = []
            real_popen = subprocess.Popen

            def launch(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                processes.append(process)
                return process

            def cancel_when_descendant_is_ready():
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                cancellation.set()

            canceller = threading.Thread(target=cancel_when_descendant_is_ready, daemon=True)
            canceller.start()
            try:
                with mock.patch("aidas.utils.cancellable_process.subprocess.Popen", side_effect=launch), \
                        mock.patch("aidas.utils.cancellable_process._PIPE_DRAIN_TIMEOUT", 0.1):
                    with self.assertRaises(CancelledError):
                        run_cancellable_process(
                            [sys.executable, "-c", parent, str(root)], cancel_event=cancellation, timeout=15,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                self.assertTrue(ready.exists())
                self.assertFalse(finished.exists(), "Cleanup waited for the surviving descendant's pipe EOF")
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].poll())
            finally:
                stop.touch()
                canceller.join(timeout=5)
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
