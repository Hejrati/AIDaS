"""Cross-platform process guard for the AIDaS desktop application."""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path
from typing import IO


class SingleInstanceGuard:
    """Keep at most one copy of the desktop application running.

    Windows uses a named kernel mutex, which is released automatically if the
    process terminates unexpectedly. POSIX systems use a non-blocking file
    lock; the file remains on disk, but the operating-system lock does not.
    """

    WINDOWS_MUTEX_NAME = r"Local\MVPRL.AIDaS.DesktopApplication"
    POSIX_LOCK_NAME = "mvprl-aidas-desktop.lock"

    def __init__(self) -> None:
        self._handle: int | None = None
        self._lock_file: IO[str] | None = None
        self._acquired = False

    @property
    def acquired(self) -> bool:
        return self._acquired

    def acquire(self) -> bool:
        """Return ``True`` only when this process owns the app guard."""
        if self._acquired:
            return True
        if os.name == "nt":
            return self._acquire_windows_mutex()
        return self._acquire_posix_lock()

    def _acquire_windows_mutex(self) -> bool:
        import ctypes
        from ctypes import wintypes

        error_already_exists = 183
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_mutex = kernel32.CreateMutexW
        create_mutex.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
        create_mutex.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL

        ctypes.set_last_error(0)
        handle = create_mutex(None, False, self.WINDOWS_MUTEX_NAME)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())

        if ctypes.get_last_error() == error_already_exists:
            close_handle(handle)
            return False

        self._handle = int(handle)
        self._acquired = True
        return True

    def _acquire_posix_lock(self) -> bool:
        import fcntl

        lock_path = Path(tempfile.gettempdir()) / self.POSIX_LOCK_NAME
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise

        self._lock_file = lock_file
        self._acquired = True
        return True

    def close(self) -> None:
        """Release this process's guard, if it owns one."""
        if self._handle is not None:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL
            close_handle(wintypes.HANDLE(self._handle))
            self._handle = None

        if self._lock_file is not None:
            import fcntl

            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_file.close()
                self._lock_file = None

        self._acquired = False

    def __enter__(self) -> "SingleInstanceGuard":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

