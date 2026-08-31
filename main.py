import os
import sys

from app.settings import settings
from app.cleanup import init_cleanup, run_cleanup_if_needed
from app.recorder import Recorder
from app.event_manager import EventManager


# ============================================================
# RTSP / FFmpeg
# ============================================================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0"
)

os.environ["OPENCV_LOG_LEVEL"] = "OFF"


# ============================================================
# Silence native stderr
# Must be before importing cv2
# ============================================================

try:
    devnull = os.open(os.devnull, os.O_WRONLY)

    os.dup2(
        devnull,
        sys.stderr.fileno(),
    )

    os.close(devnull)

except Exception:
    pass


import threading
import time

import cv2
import numpy as np

from ultralytics import YOLO

from app.settings import settings



# ============================================================
# Utils
# ============================================================

def is_recording_time() -> bool:

    hour = time.localtime().tm_hour

    return settings.RECORDING_START_HOUR <= hour < settings.RECORDING_END_HOUR


# ============================================================
# Cleanup worker
# ============================================================

def cleanup_worker(last_cleanup_time: float):

    while True:

        last_cleanup_time = run_cleanup_if_needed(
            last_cleanup_time
        )

        time.sleep(3600)


# ============================================================
# Camera
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():
    last_event_time = 0.0

    # ========================================================
    # Cleanup
    # ========================================================

    print("Init cleanup...")

    last_cleanup_time = init_cleanup()

    cleanup_thread = threading.Thread(
        target=cleanup_worker,
        args=(last_cleanup_time,),
        daemon=True,
        name="cleanup-worker",
    )

    cleanup_thread.start()

    # ========================================================
    # YOLO
    # ========================================================

    print("Loading YOLOv8n...")

    model = YOLO(settings.YOLO_MODEL_PATH)

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print("Warming up...")

    dummy = np.zeros(
        (settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE, 3),
        dtype=np.uint8,
    )

    model.predict(
        dummy,
        classes=settings.YOLO_PERSON_CLASS_IDS,
        conf=settings.YOLO_CONFIDENCE,
        imgsz=settings.YOLO_FRAME_SIZE,
        device="cpu",
        verbose=False,
    )

    print("YOLO ready")

    # ========================================================
    # Window
    # ========================================================

    window_name = (
        "YCC365 YOLO"
        f" | {settings.RTSP_URL}"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    # ========================================================
    # Camera
    # ========================================================

    print("Connecting camera...")

    camera = Camera(
        settings.RTSP_URL
    )

    print("Camera started")

    # ========================================================
    # Event manager
    # ========================================================
    event_manager = EventManager()

    # ========================================================
    # Recorder
    # ========================================================

    recorder = Recorder(
        camera
    )

    recorder.start()

    recorder.set_enabled(
        is_recording_time()
    )

    print("Recorder started")

    print(
        f"Recording schedule: "
        "{settings.RECORDING_START_HOUR}:00 - {settings.RECORDING_END_HOUR}:00"
    )

    print(
        "Press Q / ESC or close the window to exit"
    )

    # ========================================================
    # YOLO state
    # ========================================================

    last_inference = 0.0
    last_result = None

    try:

        while True:

            # =================================================
            # Always get latest frame
            # =================================================

            frame = camera.get_frame()

            if frame is None:
                continue

            # =================================================
            # Recorder schedule
            # =================================================

            recorder.set_enabled(
                is_recording_time()
            )

            now = time.monotonic()

            # =================================================
            # YOLO
            # =================================================

            if now - last_inference >= settings.YOLO_FRAME_INTERVAL:

                last_inference = now

                last_result = model.predict(
                    frame,
                    classes=settings.YOLO_PERSON_CLASS_IDS,
                    conf=settings.YOLO_CONFIDENCE,
                    imgsz=settings.YOLO_FRAME_SIZE,
                    device="cpu",
                    verbose=False,
                )[0]

                # Person detected
                if len(last_result.boxes) > 0: # type: ignore

                    now = time.monotonic()

                    if now - last_event_time >= settings.EVENT_COOLDOWN:

                        last_event_time = now

                        event_manager.handle_detection(
                            frame
                        )

            # =================================================
            # Draw latest YOLO result
            # on CURRENT camera frame
            # =================================================

            if last_result is not None:

                display = last_result.plot(
                    img=frame,
                )

            else:

                display = frame

            # =================================================
            # Display
            # =================================================

            cv2.imshow(
                window_name,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            # =================================================
            # Window close
            # =================================================

            try:

                visible = cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE,
                )

                if visible < 1:
                    break

            except cv2.error:

                break

    except KeyboardInterrupt:

        pass

    finally:

        print("Stopping...")

        # ----------------------------------------------------
        # Stop recorder first
        # ----------------------------------------------------

        try:
            recorder.stop()
        except Exception:
            pass

        # ----------------------------------------------------
        # Stop event manager
        # ----------------------------------------------------

        try:
            event_manager.stop()
        except Exception:
            pass

        # ----------------------------------------------------
        # Stop camera
        # ----------------------------------------------------

        try:
            camera.stop()
        except Exception:
            pass

        # ----------------------------------------------------
        # Close GUI
        # ----------------------------------------------------

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("Stopped")

        os._exit(0)


if __name__ == "__main__":
    main()
