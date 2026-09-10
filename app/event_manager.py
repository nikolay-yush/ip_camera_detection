import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from app.bots.telegram_bot import send_detection
from app.core.settings import settings
from app.utils.play_alarm_sound import play_alarm_sound

logger = logging.getLogger("event_manager")


class EventManager:
    """Lightweight event manager: asynchronously saves frames, plays alerts, and dispatches to Telegram."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="event-async",
        )

    def handle_detection(self, camera, frame) -> None:
        """Accepts Camera instance and clean HD frame, immediately offloading tasks to the thread pool."""
        if frame is None or camera is None:
            return

        # Deep copy clean frame to prevent corruption from ongoing RTSP stream writes
        frame_copy = frame.copy()
        
        cam_id = str(camera.camera_id)
        camera_name = (
            getattr(camera, "camera_win_name", None)
            or settings.CAMERAS.get(cam_id, {}).get("location")
            or f"Camera {cam_id}"
        )

        # Offload heavy operations (disk I/O, audio, network) to background thread pool
        self.executor.submit(self._process_event, cam_id, camera_name, frame_copy)

    def _process_event(self, camera_id: str, camera_name: str, frame) -> None:
        """Background task handling snapshot saving, audio alarm, and Telegram notification."""
        try:
            # 1. Save clean screenshot to disk
            screenshot_path = self._save_screenshot(camera_id, frame)

            # 2. Play alarm sound and send Telegram notification asynchronously
            self.executor.submit(self._safe_play_sound)
            self.executor.submit(self._safe_send_telegram, screenshot_path, camera_name)

        except Exception as exc:
            logger.error("[EventManager] Event processing error: %s", exc, exc_info=True)

    def _save_screenshot(self, camera_id: str, frame) -> Path:
        """Persists clean frame as JPEG image."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        event_dir = settings.EVENTS_DIR / date_str / camera_id
        event_dir.mkdir(parents=True, exist_ok=True)

        screenshot_path = event_dir / f"event_{now:%H-%M-%S-%f}.jpg"

        ok = cv2.imwrite(
            str(screenshot_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

        if not ok:
            raise RuntimeError(f"Failed to save screenshot: {screenshot_path}")

        logger.info("[EventManager] Screenshot saved: %s", screenshot_path)
        return screenshot_path

    def _safe_play_sound(self) -> None:
        """Safely executes audio playback."""
        try:
            play_alarm_sound()
        except Exception as exc:
            logger.error("[EventManager] Audio playback failed: %s", exc)

    def _safe_send_telegram(self, screenshot_path: Path, camera_name: str) -> None:
        """Safely dispatches notification to Telegram."""
        try:
            send_detection(screenshot_path, camera_id=camera_name)
        except Exception as exc:
            logger.error("[EventManager] Telegram dispatch failed: %s", exc)

    def stop(self) -> None:
        """Gracefully shuts down thread pool executor."""
        self.executor.shutdown(wait=False, cancel_futures=True)