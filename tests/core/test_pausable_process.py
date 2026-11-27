"""Exercise the ownership and lifecycle of paused subprocesses."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from aidas.utils import pausable_process


class ProcessSuspensionTests(unittest.TestCase):
    def test_process_that_already_exited_never_touches_native_process_api(self):
        process = Mock()
        process.poll.return_value = 0
        with patch.object(pausable_process, "_pause_windows_tree") as windows:
            self.assertIsNone(pausable_process.pause_process_tree(process))
        windows.assert_not_called()

    def test_failed_resume_can_be_retried(self):
        resume = Mock(side_effect=[OSError("Could not resume"), None])
        discard = Mock()
        suspension = pausable_process.ProcessSuspension(resume, discard)
        with self.assertRaises(OSError):
            suspension.resume()
        discard.assert_not_called()
        suspension.resume()
        suspension.resume()
        self.assertEqual(resume.call_count, 2)
        discard.assert_called_once_with()

    def test_discard_after_termination_does_not_resume(self):
        resume = Mock()
        discard = Mock()
        suspension = pausable_process.ProcessSuspension(resume, discard)
        suspension.discard()
        suspension.discard()
        suspension.resume()
        resume.assert_not_called()
        discard.assert_called_once_with()

    @unittest.skipUnless(os.name == "nt", "uses Windows thread handles")
    def test_failed_pause_retains_only_threads_that_could_not_be_resumed(self):
        kernel = Mock()
        kernel.ResumeThread.side_effect = [1, 0xFFFFFFFF, 1]
        kernel.WaitForSingleObject.return_value = 0x102
        process = Mock(pid=1234)
        process.poll.return_value = None

        def fail_during_suspend(_kernel, _pid, _process_handle, threads):
            threads.update({11: 100, 12: 101})
            raise OSError("Could not suspend the remaining threads")

        with patch.object(pausable_process, "_kernel32", return_value=kernel), \
                patch.object(pausable_process, "_open_process", return_value=(99, 0)), \
                patch.object(pausable_process, "_suspend_windows_threads", side_effect=fail_during_suspend):
            with self.assertRaises(pausable_process.ProcessPauseError) as caught:
                pausable_process.pause_process_tree(process)

        self.assertEqual([call.args for call in kernel.CloseHandle.call_args_list], [(100,), (99,)])
        caught.exception.suspension.resume()
        self.assertEqual([call.args for call in kernel.CloseHandle.call_args_list], [(100,), (99,), (101,)])

    def test_pauses_and_resumes_a_live_child_and_its_worker(self):
        worker_code = """
from pathlib import Path
import sys
import time

def pulse(name):
    stop = Path(sys.argv[1]) / 'stop'
    heartbeat = Path(sys.argv[1]) / name
    deadline = time.monotonic() + 20
    while not stop.exists() and time.monotonic() < deadline:
        with heartbeat.open('ab') as stream:
            stream.write(b'.')
        time.sleep(0.04)
"""
        parent_code = worker_code + """
import subprocess

with subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]) as child:
    pulse('parent')
"""
        options = (
            {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt" else {"start_new_session": True}
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("parent", "worker")]
            process = subprocess.Popen(
                [sys.executable, "-c", parent_code, directory, worker_code + "\npulse('worker')\n"],
                **options,
            )
            suspension = None
            try:
                self._wait_for_pulses(paths, [0, 0])
                suspension = pausable_process.pause_process_tree(process)
                self.assertIsNotNone(suspension)
                # An already-started write can finish while the suspension
                # takes effect. Check a new sample once it has settled.
                time.sleep(0.1)
                paused = [path.stat().st_size for path in paths]
                time.sleep(0.2)
                self.assertEqual(paused, [path.stat().st_size for path in paths])
                suspension.resume()
                self._wait_for_pulses(paths, paused)
            finally:
                if suspension is not None:
                    suspension.resume()
                (root / "stop").touch()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    def _wait_for_pulses(self, paths, previous):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if all(path.exists() and path.stat().st_size > size for path, size in zip(paths, previous)):
                return
            time.sleep(0.025)
        self.fail("The subprocess and its worker did not advance.")
