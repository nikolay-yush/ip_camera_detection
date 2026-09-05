import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from app.settings import settings


def cleanup_old_events(
    base_dir: Path,
    days: int,
) -> None:
    
    cutoff = (
        datetime.now().date()
        - timedelta(days=days)
    )

    if not base_dir.exists():
        return

    for date_folder in base_dir.iterdir():

        if not date_folder.is_dir():
            continue

        try:
            folder_date = datetime.strptime(
                date_folder.name,
                "%Y-%m-%d",
            ).date()

        except ValueError:
            continue

        if folder_date < cutoff:

            shutil.rmtree(
                date_folder,
                ignore_errors=True,
            )

            print(
                f"[Cleanup] Deleted old folder: "
                f"{date_folder}"
            )


def run_cleanup() -> None:

    now = time.time()

    print("[Cleanup] Running cleanup check...")

    cleanup_old_events(
        settings.EVENTS_DIR,
        days=settings.CLEANUP_DAYS,
    )

    return now



def init_cleanup() -> None:

    settings.EVENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_cleanup()
