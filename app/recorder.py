import threading
import time
from datetime import datetime

import cv2

from app.settings import settings


class Recorder:

    def __init__(self, camera):

        self.camera = camera

        self.running = True
        self.enabled = False

        self.writer = None

        self.current_date = None
        self.current_hour = None

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

        filename = f"{now.strftime('%H')}-00.mp4"

        video_path = event_dir / filename

        height, width = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(  # type: ignore
            *"mp4v"
        )

        self.writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            20.0,
            (width, height),
        )

        if not self.writer.isOpened():

            self.writer = None

            print(
                f"[Recorder] Failed to open: "
                f"{video_path}"
            )

            return

        self.current_date = now.date()
        self.current_hour = now.hour

        print(
            f"[Recorder] Recording: "
            f"{video_path}"
        )

    def _release_writer(self):

        if self.writer is not None:

            self.writer.release()

            self.writer = None

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

            # =============================================
            # Write frame
            # =============================================

            if self.writer is not None:

                self.writer.write(frame)

            time.sleep(0.01)

        self._release_writer()

    def stop(self):

        self.running = False

        self.thread.join(
            timeout=2
        )

        self._release_writer()

