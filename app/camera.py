import logging
import os
import threading
import time
import cv2

logger = logging.getLogger(__name__)

# Low-latency FFmpeg capture configuration for OpenCV
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
)


class Camera:
    """Threaded RTSP camera reader with non-blocking initialization and health status flags."""

    def __init__(self, camera_id: str, rtsp: str, camera_win_name: str = ""):
        self.camera_id = camera_id
        self.rtsp = rtsp
        self.camera_win_name = camera_win_name

        self.running = True

        # State flags (accessible for status checks in main loop)
        self.is_initializing = True  # True while performing the first connection attempt
        self.initial_success = False  # True if the initial stream yielded a valid frame
        self.is_connected = False  # True if stream capture is currently active

        self.lock = threading.Lock()
        self.frame = None
        self.cap: cv2.VideoCapture | None = None
        self._last_frame_time = 0.0

        # Start worker thread without blocking the main thread during initialization
        self.thread = threading.Thread(
            target=self._read,
            daemon=True,
            name=f"rtsp-reader-{self.camera_id}",
        )
        self.thread.start()

    def _connect(self) -> bool:
        """Attempt initial RTSP connection and verify video frame receipt."""
        logger.info("[Camera:%s] Connecting to %s...", self.camera_id, self.rtsp)

        cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.warning("[Camera:%s] Cannot open RTSP stream URL.", self.camera_id)
            cap.release()
            return False

        # Read initial frame to validate stream content
        ok, first_frame = cap.read()  # type: ignore
        if not ok or first_frame is None:
            logger.error("[Camera:%s] Stream opened, but no frame payload received.", self.camera_id)
            cap.release()
            return False

        self.cap = cap
        self.is_connected = True
        self._last_frame_time = time.time()

        with self.lock:
            self.frame = first_frame

        logger.info("[Camera:%s] First connection successful with valid frame.", self.camera_id)
        return True

    def _read(self):
        """Worker loop for continuous background frame capture."""
        # 1. Perform initial connection within the worker thread
        success = self._connect()
        self.initial_success = success
        self.is_initializing = False  # Initial check phase completed

        if not success:
            self.is_connected = False
            return

        # 2. Continuous frame reading loop
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.is_connected = False
                time.sleep(0.1)
                continue

            ok, frame = self.cap.read()  # type: ignore

            if not self.running:
                break

            if not ok or frame is None:
                self.is_connected = False
                time.sleep(0.01)
                continue

            # Frame received successfully — update timestamp and frame buffer
            self.is_connected = True
            self._last_frame_time = time.time()

            with self.lock:
                self.frame = frame

        self._cleanup()

    def _cleanup(self):
        """Release underlying OpenCV VideoCapture resources."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    @property
    def no_frame_duration(self) -> float:
        """Returns elapsed time in seconds since the last valid frame was captured."""
        if self._last_frame_time == 0.0:
            return float("inf")
        return time.time() - self._last_frame_time

    def get_frame(self):
        """Safely retrieve a copy of the latest captured frame."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        """Stop background worker thread and clean up stream resources."""
        self.running = False

        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

        self._cleanup()
        logger.info("[Camera:%s] Stopped.", self.camera_id)