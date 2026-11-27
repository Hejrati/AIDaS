"""Pause an application subprocess and its workers without restarting it."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from ctypes import wintypes
import os
import signal
import subprocess
import threading


class ProcessSuspension:
    """Own the resources needed to reverse one successful suspension."""

    def __init__(self, resume: Callable[[], None], discard: Callable[[], None]):
        self._lock = threading.RLock()
        self._resume = resume
        self._discard = discard

    def resume(self) -> None:
        with self._lock:
            if self._resume is not None:
                self._resume()
                self.discard()

    def discard(self) -> None:
        """Release resources after termination without resuming the process."""
        with self._lock:
            if self._discard is not None:
                self._discard()
                self._resume = self._discard = None


class ProcessPauseError(OSError):
    """A pause and its rollback failed, leaving resources for cleanup."""

    def __init__(self, error: BaseException, rollback_error: OSError, suspension: ProcessSuspension):
        super().__init__(f"{error}. Could not resume all suspended threads: {rollback_error}")
        self.suspension = suspension


def pause_process_tree(process: subprocess.Popen) -> ProcessSuspension | None:
    """Suspend a live child launched by the app in its own process group."""
    if process.poll() is not None:
        return None
    pid = process.pid
    if pid <= 1 or pid == os.getpid():
        raise OSError("The selected process is not an application child.")
    if os.name == "nt":
        return _pause_windows_tree(process)

    try:
        if os.getpgid(pid) != pid:
            raise OSError("The child process does not have its own process group.")
        os.killpg(pid, signal.SIGSTOP)
    except ProcessLookupError:
        return None

    def resume_group():
        try:
            os.killpg(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass

    return ProcessSuspension(resume_group, lambda: None)


# Windows Toolhelp flags, access rights, and return values.
_PROCESS_SNAPSHOT = 0x00000002
_THREAD_SNAPSHOT = 0x00000004
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_THREAD_QUERY_LIMITED_INFORMATION = 0x0800
_THREAD_SUSPEND_RESUME = 0x0002
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0
_NO_MORE_FILES = 18
_INVALID_PARAMETER = 87
_INVALID_SUSPEND_COUNT = 0xFFFFFFFF
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_MAX_SNAPSHOT_PASSES = 32


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


def _kernel32():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    functions = (
        ("CreateToolhelp32Snapshot", wintypes.HANDLE, (wintypes.DWORD, wintypes.DWORD)),
        ("Process32FirstW", wintypes.BOOL, (wintypes.HANDLE, ctypes.POINTER(_ProcessEntry))),
        ("Process32NextW", wintypes.BOOL, (wintypes.HANDLE, ctypes.POINTER(_ProcessEntry))),
        ("Thread32First", wintypes.BOOL, (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry))),
        ("Thread32Next", wintypes.BOOL, (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry))),
        ("OpenProcess", wintypes.HANDLE, (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)),
        ("OpenThread", wintypes.HANDLE, (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)),
        ("GetProcessTimes", wintypes.BOOL, (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4),
        ("GetProcessIdOfThread", wintypes.DWORD, (wintypes.HANDLE,)),
        ("SuspendThread", wintypes.DWORD, (wintypes.HANDLE,)),
        ("ResumeThread", wintypes.DWORD, (wintypes.HANDLE,)),
        ("WaitForSingleObject", wintypes.DWORD, (wintypes.HANDLE, wintypes.DWORD)),
        ("CloseHandle", wintypes.BOOL, (wintypes.HANDLE,)),
    )
    for name, result, arguments in functions:
        function = getattr(kernel, name)
        function.restype = result
        function.argtypes = arguments
    return kernel


def _snapshot(kernel, flag, entry, first, next_item, fields):
    handle = kernel.CreateToolhelp32Snapshot(flag, 0)
    if handle == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry.dwSize = ctypes.sizeof(entry)
        read = first
        values = []
        while read(handle, ctypes.byref(entry)):
            values.append(tuple(getattr(entry, field) for field in fields))
            entry.dwSize = ctypes.sizeof(entry)
            read = next_item
        error = ctypes.get_last_error()
        if error != _NO_MORE_FILES:
            raise ctypes.WinError(error)
        return values
    finally:
        kernel.CloseHandle(handle)


def _open_process(kernel, pid):
    handle = kernel.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == _INVALID_PARAMETER:
            return None
        raise ctypes.WinError(error)
    try:
        creation, exit_time, system, user = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *map(ctypes.byref, (creation, exit_time, system, user))):
            raise ctypes.WinError(ctypes.get_last_error())
        return handle, (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    except BaseException:
        kernel.CloseHandle(handle)
        raise


def _suspend_windows_threads(kernel, pid, process_handle, threads):
    rights = _THREAD_SUSPEND_RESUME | _THREAD_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE
    for _ in range(_MAX_SNAPSHOT_PASSES):
        entries = _snapshot(
            kernel, _THREAD_SNAPSHOT, _ThreadEntry(), kernel.Thread32First, kernel.Thread32Next,
            ("th32ThreadID", "th32OwnerProcessID"),
        )
        new_threads = [tid for tid, owner in entries if owner == pid and tid not in threads]
        if not new_threads:
            if threads or kernel.WaitForSingleObject(process_handle, 0) == _WAIT_OBJECT_0:
                return
            raise OSError(f"Could not find any threads in R process {pid}.")
        for tid in new_threads:
            handle = kernel.OpenThread(rights, False, tid)
            if not handle:
                error = ctypes.get_last_error()
                if error == _INVALID_PARAMETER:
                    continue
                raise ctypes.WinError(error)
            try:
                if kernel.GetProcessIdOfThread(handle) != pid:
                    continue
                if kernel.SuspendThread(handle) == _INVALID_SUSPEND_COUNT:
                    error = ctypes.get_last_error()
                    if kernel.WaitForSingleObject(handle, 0) == _WAIT_OBJECT_0:
                        continue
                    raise ctypes.WinError(error)
                threads[tid] = handle
            finally:
                if tid not in threads:
                    kernel.CloseHandle(handle)
    raise OSError(f"R process {pid} continued creating threads while being paused.")


def _pause_windows_tree(process):
    kernel = _kernel32()
    process_handles = {}
    suspended_threads = []

    def discard_threads():
        for threads in suspended_threads:
            for handle in threads.values():
                kernel.CloseHandle(handle)
            threads.clear()

    def resume_threads():
        error = None
        for threads in reversed(suspended_threads):
            for tid, handle in list(threads.items()):
                if kernel.ResumeThread(handle) == _INVALID_SUSPEND_COUNT:
                    resume_error = ctypes.get_last_error()
                    if kernel.WaitForSingleObject(handle, 0) != _WAIT_OBJECT_0:
                        error = resume_error
                        continue
                kernel.CloseHandle(handle)
                del threads[tid]
        if error is not None:
            raise ctypes.WinError(error)

    suspension = ProcessSuspension(resume_threads, discard_threads)

    def suspend(pid, entry):
        process_handles[pid] = entry
        threads = {}
        suspended_threads.append(threads)
        _suspend_windows_threads(kernel, pid, entry[0], threads)

    try:
        entry = _open_process(kernel, process.pid)
        if entry is None:
            return None
        process_handles[process.pid] = entry
        if process.poll() is not None:
            return None
        suspend(process.pid, entry)
        skipped = {0, os.getpid()}
        # Freeze each parent before enumerating children. Recheck the tree to
        # catch descendants created while another process was being suspended.
        for _ in range(_MAX_SNAPSHOT_PASSES):
            entries = _snapshot(
                kernel, _PROCESS_SNAPSHOT, _ProcessEntry(), kernel.Process32FirstW, kernel.Process32NextW,
                ("th32ProcessID", "th32ParentProcessID"),
            )
            found_child = False
            for pid, parent_pid in entries:
                if pid in process_handles or pid in skipped or parent_pid not in process_handles:
                    continue
                child = _open_process(kernel, pid)
                if child is None:
                    continue
                if child[1] < process_handles[parent_pid][1]:
                    # A reused parent ID can otherwise match an unrelated child.
                    kernel.CloseHandle(child[0])
                    skipped.add(pid)
                    continue
                suspend(pid, child)
                found_child = True
            if not found_child:
                return suspension if any(suspended_threads) else None
        raise OSError("The R process tree continued creating workers while being paused.")
    except BaseException as error:
        try:
            suspension.resume()
        except OSError as rollback_error:
            raise ProcessPauseError(error, rollback_error, suspension) from error
        raise
    finally:
        for handle, _ in process_handles.values():
            kernel.CloseHandle(handle)
