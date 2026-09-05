import time
from app.cleanup import run_cleanup_if_needed


def cleanup_worker(last_cleanup_time: float):

    while True:

        last_cleanup_time = run_cleanup_if_needed(
            last_cleanup_time
        )

        time.sleep(3600)
