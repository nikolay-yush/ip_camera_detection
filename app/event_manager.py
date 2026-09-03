import threading
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import cv2

from app.settings import settings
from app.sound import play_sound
from app.telegram_bot import send_detection


class EventManager:

    def __init__(self):
        self.queue = Queue()
        self.running = True

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="event-worker",
        )
        self.thread.start()

    def handle_detection(self, camera_id: str, frame) -> None:
        """Receives camera_id directly from main loop and enqueues the frame."""
        self.queue.put((camera_id, frame.copy()))

    def _worker(self):
        while self.running:
            try:
                camera_id, frame = self.queue.get(timeout=1)
            except Empty:
                continue

            try:
                # 1. Save screenshot to structured directory
                screenshot_path = self._save_screenshot(camera_id, frame)
                
                # 2. Local audio notification
                play_sound()

                # 3. Get user-friendly camera name if present in settings, fallback to ID
                camera_info = settings.CAMERAS.get(camera_id, {})
                camera_name = camera_info.get("name", camera_id)

                # 4. Dispatch Telegram notification
                send_detection(screenshot_path, camera_id=camera_name) # type: ignore

            except Exception as exc:
                print(f"[EventManager] Event error: {exc}")

            finally:
                self.queue.task_done()

    def _save_screenshot(self, camera_id: str, frame) -> Path:
        now = datetime.now()

        # Structure: EVENTS_DIR / YYYY-MM-DD / camera_id /
        event_dir = (
            settings.EVENTS_DIR
            / now.strftime("%Y-%m-%d")
            / camera_id
        )

        event_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        screenshot_path = (
            event_dir
            / f"event_{now:%H-%M-%S-%f}.jpg"
        )

        ok = cv2.imwrite(
            str(screenshot_path),
            frame,
        )

        if not ok:
            raise RuntimeError(f"Failed to save screenshot: {screenshot_path}")

        print(f"[EventManager] Screenshot saved: {screenshot_path}")

        return screenshot_path

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)