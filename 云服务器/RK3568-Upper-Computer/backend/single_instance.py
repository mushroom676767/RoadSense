"""Cross-platform single-process guard for the in-memory video hub."""

import atexit
import hashlib
import os


_lock_resource = None


def _instance_key(base_dir):
    normalized = os.path.normcase(os.path.abspath(base_dir))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def acquire_single_instance(base_dir):
    """Return False when another manager for this directory is already alive."""
    global _lock_resource
    if _lock_resource is not None:
        return True

    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_wchar_p,
        ]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        name = f"Local\\RK3568Manager-{_instance_key(base_dir)}"
        ctypes.set_last_error(0)
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
            kernel32.CloseHandle(handle)
            return False
        _lock_resource = ("windows", kernel32, handle)
    else:
        import fcntl

        runtime_dir = os.path.join(base_dir, "runtime")
        os.makedirs(runtime_dir, exist_ok=True)
        lock_file = open(
            os.path.join(runtime_dir, "manager.lock"),
            "a+",
            encoding="utf-8",
        )
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return False
        _lock_resource = ("posix", lock_file)

    atexit.register(release_single_instance)
    return True


def release_single_instance():
    global _lock_resource
    resource = _lock_resource
    _lock_resource = None
    if resource is None:
        return
    if resource[0] == "windows":
        _, kernel32, handle = resource
        kernel32.CloseHandle(handle)
    else:
        _, lock_file = resource
        lock_file.close()
