import logging
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path

from app.core.settings import settings

logger = logging.getLogger(__name__)


class EventsCleanupWorker:
    """Worker for periodic cleanup of outdated event directories."""

    def __init__(
        self,
        base_dir: Path | None = None,
        retention_days: int | None = None,
        check_interval_seconds: int | None = None,
    ) -> None:
        self.base_dir = base_dir or settings.EVENTS_DIR
        self.retention_days = retention_days or settings.CLEANUP_DAYS
        self.check_interval = check_interval_seconds or settings.CLEANUP_CHECK_INTERVAL_SECONDS

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def init_storage(self) -> None:
        """Ensure the base events directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run_cleanup(self) -> None:
        """Execute a single check and remove expired date folders."""
        if not self.base_dir.exists():
            return

        cutoff = datetime.now().date() - timedelta(days=self.retention_days)
        logger.info(
            "[Cleanup] Starting cleanup check (Cutoff: %s, Dir: %s)",
            cutoff,
            self.base_dir,
        )

        for date_folder in self.base_dir.iterdir():
            if not date_folder.is_dir():
                continue

            try:
                folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d").date()
            except ValueError:
                continue

            if folder_date < cutoff:
                shutil.rmtree(date_folder, ignore_errors=True)
                logger.info("[Cleanup] Deleted old folder: %s", date_folder)

    def _worker_loop(self) -> None:
        """Internal background thread loop."""
        self.init_storage()
        self.run_cleanup()

        # Interruptible wait instead of time.sleep
        while not self._stop_event.wait(timeout=self.check_interval):
            self.run_cleanup()

        logger.info("[Cleanup] Worker loop finished.")

    def start(self) -> None:
        """Start the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[Cleanup] Worker is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="EventCleanupWorkerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("[Cleanup] Worker thread successfully started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker thread gracefully."""
        if self._thread is None or not self._thread.is_alive():
            return

        logger.info("[Cleanup] Stopping worker...")
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("[Cleanup] Worker stopped.")