import logging
import os
import threading
import psutil

from app.core.settings import settings

logger = logging.getLogger(__name__)


class SystemMonitoringWorker:
    """Background worker for logging process resource utilization."""

    def __init__(self, interval: float | None = None) -> None:
        self.interval = interval or getattr(settings, "MONITOR_INTERVAL_SECONDS", 5.0)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _worker_loop(self) -> None:
        """Internal execution loop running inside the background thread."""
        process = psutil.Process(os.getpid())
        cpu_count = psutil.cpu_count(logical=True) or 1

        # Warm-up call for CPU percent calculation
        process.cpu_percent(interval=None)

        # Main loop with interruptible sleep
        while not self._stop_event.wait(timeout=self.interval):
            try:
                # Raw sum across all cores (e.g., 370.0%)
                raw_cpu = process.cpu_percent(interval=None)

                # Normalized overall CPU usage (0-100%)
                system_cpu_pct = raw_cpu / cpu_count

                # Resident Set Size (RAM in MB)
                ram_mb = process.memory_info().rss / (1024 * 1024)

                # Total native threads
                num_threads = process.num_threads()

                logger.info(
                    "[MONITOR] CPU (System): %5.1f%% | CPU (Cores): %5.1f%% (%d cores) | RAM: %6.1f MB | Threads: %d (App + C++ Pool)",
                    system_cpu_pct,
                    raw_cpu,
                    cpu_count,
                    ram_mb,
                    num_threads,
                )
            except Exception as exc:
                logger.error("[MONITOR] Metric extraction error: %s", exc)

        logger.info("[MONITOR] System monitor loop finished.")

    def start(self) -> None:
        """Start the background system monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[MONITOR] Worker is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="SystemMonitorWorkerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("[MONITOR] Background system monitor started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the monitor worker to stop and wait for completion."""
        if self._thread is None or not self._thread.is_alive():
            return

        logger.info("[MONITOR] Stopping system monitor worker...")
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("[MONITOR] System monitor worker stopped.")