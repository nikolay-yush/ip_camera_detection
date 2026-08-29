import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from app.settings import settings

STATE_FILE_NAME = ".cleanup_state.txt"

def _load_last_cleanup(state_file: Path) -> float:
    if state_file.exists():
        try:
            return float(state_file.read_text().strip())
        except Exception:
            return 0.0
    return 0.0


def _save_last_cleanup(state_file: Path, timestamp: float) -> None:
    state_file.write_text(str(timestamp))


def cleanup_old_events(base_dir: Path, days: int) -> None:
    cutoff = datetime.now() - timedelta(days=days)

    if not base_dir.exists():
        return

    for date_folder in base_dir.iterdir():
        if not date_folder.is_dir():
            continue

        try:
            folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
        except ValueError:
            continue

        if folder_date < cutoff:
            shutil.rmtree(date_folder, ignore_errors=True)
            print(f"[Cleanup] Deleted old folder: {date_folder}")


def run_cleanup_if_needed(last_cleanup_time: float, force: bool = False) -> float:
    now = time.time()
    cleanup_interval = settings.CLEANUP_DAYS * 86400
    state_file = settings.EVENTS_DIR / STATE_FILE_NAME

    if force or (now - last_cleanup_time > cleanup_interval):
        print("[Cleanup] Running cleanup check...")
        cleanup_old_events(settings.EVENTS_DIR, days=settings.CLEANUP_DAYS)
        _save_last_cleanup(state_file, now)
        return now

    return last_cleanup_time


def init_cleanup() -> float:
    settings.EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    state_file = settings.EVENTS_DIR / STATE_FILE_NAME
    
    last_cleanup = _load_last_cleanup(state_file)
    return run_cleanup_if_needed(last_cleanup)