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

    def handle_detection(self, frame) -> None:

        self.queue.put(
            frame.copy()
        )

    def _worker(self):

        while self.running:

            try:

                frame = self.queue.get(
                    timeout=1
                )

            except Empty:

                continue

            try:
                # events are handled here
                screenshot_path = (self._save_screenshot(frame))
                play_sound()
                send_detection(screenshot_path)

            except Exception as exc:

                print(
                    f"[EventManager] "
                    f"Event error: {exc}"
                )

            finally:

                self.queue.task_done()

    def _save_screenshot(self, frame) -> Path:

        now = datetime.now()

        event_dir = (
            settings.EVENTS_DIR
            / now.strftime("%Y-%m-%d")
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

            raise RuntimeError(
                f"Failed to save screenshot: "
                f"{screenshot_path}"
            )

        print(
            f"[EventManager] "
            f"Screenshot saved: "
            f"{screenshot_path}"
        )

        return screenshot_path

    def stop(self):

        self.running = False

        self.thread.join(
            timeout=2
        )