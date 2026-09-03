import os
import signal


def force_kill_self():
    """Forces immediate termination of the current process group."""
    print("[System] Executing hard process kill...", flush=True)
    try:
        # Kill the entire process group (parent + all child processes)
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except Exception:
        # Fallback to single process kill
        os.kill(os.getpid(), signal.SIGKILL)