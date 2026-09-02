import threading
import time
from datetime import datetime

import cv2

from app.settings import settings


class Recorder:

    def __init__(self, camera, target_fps: float = 12.0):

        self.camera = camera

        self.running = True
        self.enabled = False

        self.writer = None

        self.current_date = None
        self.current_hour = None

        # Frame rate limit matching source camera stream (12 FPS)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0833
        self.last_write_time = 0.0

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="video-recorder",
        )

    def start(self):

        self.thread.start()

    def set_enabled(self, enabled: bool):

        self.enabled = enabled

    def _get_event_dir(self, current_date):

        event_dir = (
            settings.EVENTS_DIR
            / current_date.strftime("%Y-%m-%d")
        )

        event_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return event_dir

    def _create_writer(self, frame, now):

        event_dir = self._get_event_dir(
            now.date()
        )

        filename = f"{now.strftime('%H')}-00.mkv"

        video_path = event_dir / filename

        height, width = frame.shape[:2]

        # Universal system codecs for Linux/OpenCV compatibility
        codecs_to_try = ["avc1", "H264", "XVID", "mp4v"]
        self.writer = None

        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    self.target_fps,
                    (width, height),
                )

                if writer.isOpened():
                    self.writer = writer
                    print(
                        f"[Recorder] Recording started ({self.target_fps} FPS, codec: {codec}): "
                        f"{video_path}"
                    )
                    break
            except Exception:
                continue

        if self.writer is None:
            print(
                f"[Recorder] Failed to open video writer for: "
                f"{video_path}"
            )
            return

        self.current_date = now.date()
        self.current_hour = now.hour

    def _release_writer(self):

        if self.writer is not None:
            try:
                self.writer.release()
                print("[Recorder] Video writer released safely.")
            except Exception as exc:
                print(f"[Recorder] Error releasing writer: {exc}")
            finally:
                self.writer = None

            self.current_date = None
            self.current_hour = None

    def _run(self):

        while self.running:

            # =============================================
            # Recording enabled?
            # =============================================

            if not self.enabled:

                self._release_writer()

                time.sleep(1)

                continue

            # =============================================
            # Get latest camera frame
            # =============================================

            frame = self.camera.get_frame()

            if frame is None:

                time.sleep(0.01)

                continue

            now_mono = time.monotonic()

            # Enforce exact frame interval
            if now_mono - self.last_write_time < self.frame_interval:
                time.sleep(0.005)
                continue

            self.last_write_time = now_mono

            now = datetime.now()

            # =============================================
            # New hour or new day
            # =============================================

            if (
                self.writer is None
                or self.current_date != now.date()
                or self.current_hour != now.hour
            ):

                self._release_writer()

                self._create_writer(
                    frame,
                    now,
                )

                # Pause briefly if failed to create writer to prevent log spam
                if self.writer is None:
                    time.sleep(2.0)
                    continue

            # =============================================
            # Write frame
            # =============================================

            if self.writer is not None:
                try:
                    self.writer.write(frame)
                except Exception as exc:
                    print(f"[Recorder] Error writing frame: {exc}")

        self._release_writer()

    def stop(self):

        self.running = False

        self.thread.join(
            timeout=2
        )

        self._release_writer()