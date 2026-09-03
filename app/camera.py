import threading
import time
import cv2


class Camera:

    def __init__(self, camera_id: str, rtsp: str, camera_win_name: str = ""):
        self.camera_id = camera_id
        self.rtsp = rtsp
        self.running = True

        self.lock = threading.Lock()
        self.frame = None
        self.camera_win_name = camera_win_name

        print(f"[Camera:{self.camera_id}] Connecting to {self.rtsp}...")

        self.cap = cv2.VideoCapture(
            self.rtsp,
            cv2.CAP_FFMPEG,
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        self.thread = threading.Thread(
            target=self._read,
            daemon=True,
            name=f"rtsp-reader-{self.camera_id}",
        )

        self.thread.start()

    def _read(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ok, frame = self.cap.read()  # type: ignore

            if not self.running:
                break

            if not ok:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        print(f"[Camera:{self.camera_id}] Stopping reader...")
        self.running = False

        self.thread.join(timeout=0.5)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print(f"[Camera:{self.camera_id}] Stopped")