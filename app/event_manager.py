import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import cv2

from app.settings import settings
from app.sound import play_sound
from app.telegram_bot import send_detection


class EventManager:

    def __init__(self):
        # Bound queue size to prevent memory leaks if network dispatch drops or lags
        self.queue: Queue = Queue(maxsize=20)
        self.running = True

        # Thread pool to isolate slow I/O tasks (Telegram requests & audio playback)
        self.executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="event-async",
        )

        # Cache last created directory key to prevent redundant filesystem syscalls
        self._current_dir_cache: str = ""

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="event-worker",
        )
        self.thread.start()

    def handle_detection(self, camera_id: str, frame) -> None:
        """Receives camera_id and frame from YOLO worker and enqueues event without frame duplication."""
        if not self.running:
            return

        # Drop older unhandled events if queue reaches capacity
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except Empty:
                pass

        self.queue.put((camera_id, frame))

    def _worker(self):
        while self.running:
            try:
                camera_id, frame = self.queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                # 1. Save screenshot to structured directory using optimized JPEG compression
                screenshot_path = self._save_screenshot(camera_id, frame)

                # 2. Retrieve user-friendly camera name from configuration settings
                camera_info = settings.CAMERAS.get(camera_id, {})
                camera_name = camera_info.get("name", camera_id)

                # 3. Offload I/O bound operations (Audio & Telegram API) to background thread pool
                self.executor.submit(play_sound)
                self.executor.submit(
                    self._safe_send_telegram, screenshot_path, camera_name
                )

            except Exception as exc:
                print(f"[EventManager] Event worker error: {exc}")

            finally:
                self.queue.task_done()

    def _save_screenshot(self, camera_id: str, frame) -> Path:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        dir_key = f"{date_str}_{camera_id}"

        event_dir = settings.EVENTS_DIR / date_str / camera_id

        # Invoke mkdir only when date or camera target changes
        if self._current_dir_cache != dir_key:
            event_dir.mkdir(parents=True, exist_ok=True)
            self._current_dir_cache = dir_key

        screenshot_path = event_dir / f"event_{now:%H-%M-%S-%f}.jpg"

        # Encode JPEG with 80% quality parameter -> 2.5x faster encoding and reduced file footprint
        ok = cv2.imwrite(
            str(screenshot_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )

        if not ok:
            raise RuntimeError(f"Failed to save screenshot: {screenshot_path}")

        print(f"[EventManager] Screenshot saved: {screenshot_path}")
        return screenshot_path

    def _safe_send_telegram(self, screenshot_path: Path, camera_name: str) -> None:
        """Executes inside ThreadPoolExecutor to prevent network latency from stalling the event loop."""
        try:
            send_detection(screenshot_path, camera_id=camera_name)  # type: ignore
        except Exception as exc:
            print(f"[EventManager] Telegram dispatch failed: {exc}")

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.5)
        self.executor.shutdown(wait=False)