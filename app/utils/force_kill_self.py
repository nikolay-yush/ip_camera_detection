import os
import sys
import signal
import subprocess


def force_kill_self():
    """
    Forces immediate and hard termination of the process and all its child processes.
    Works reliably on both Windows and Linux/macOS.
    """
    print("[System] Executing hard process tree kill...", flush=True)
    pid = os.getpid()

    if sys.platform == "win32":
        # On Windows, use taskkill to recursively terminate the entire process tree (/T /F)
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
        # Fallback if taskkill fails
        os._exit(0)
    else:
        # On POSIX (Linux / macOS), kill the entire process group with SIGKILL
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            # Fallback to hard exit without running C++ static destructors
            os._exit(0)