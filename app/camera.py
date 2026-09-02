import threading
import cv2


class Camera:

    def __init__(self, url: str):

        self.running = True

        self.lock = threading.Lock()
        self.frame = None

        self.cap = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG,
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        self.thread = threading.Thread(
            target=self._read,
            daemon=True,
            name="rtsp-reader",
        )

        self.thread.start()

    def _read(self):

        while self.running:

            ok, frame = self.cap.read() # type: ignore

            if not self.running:
                break

            if not ok:
                continue

            with self.lock:
                self.frame = frame

    def get_frame(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):

        self.running = False

        self.thread.join(timeout=0.5)

        if not self.thread.is_alive():

            if self.cap is not None:
                self.cap.release()

            self.cap = None
