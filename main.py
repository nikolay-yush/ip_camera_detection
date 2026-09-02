import os
import sys
import threading
import time
import cv2
import numpy as np

from ultralytics import YOLO
from app.settings import settings
from app.camera import Camera
from app.cleanup import init_cleanup
from app.recorder import Recorder
from app.event_manager import EventManager

from app.utils.is_recording_time import is_recording_time
from app.utils.cleanup_worker import cleanup_worker
from app.utils.system_monitor_worker import system_monitor_worker


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


# ============================================================
# Main
# ============================================================

def main():
    # ========================================================
    # Start System Resource Monitor
    # ========================================================
    # monitor_thread = threading.Thread(
    #     target=system_monitor_worker,
    #     args=(5.0,),  # 5s interval
    #     daemon=True,
    #     name="system-monitor",
    # )
    # monitor_thread.start()

    # print("System resource monitor started (5s interval)")

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
        f"Recording schedule: {settings.RECORDING_START_HOUR}:00 - {settings.RECORDING_END_HOUR}:00"
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

                        annotated_frame = last_result.plot(img=frame.copy())

                        event_manager.handle_detection(
                            frame=annotated_frame
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
