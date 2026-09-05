import time
from app.cleanup import run_cleanup


def cleanup_worker():
    while True:
        run_cleanup()
        time.sleep(3600)
