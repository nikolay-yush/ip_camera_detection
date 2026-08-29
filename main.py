import cv2
import time
import logging

from ultralytics import YOLO

from app.settings import settings
from app.cleanup import init_cleanup, run_cleanup_if_needed
from app.event_shot_manager import EventShotManager


# Suppress YOLO verbose output
logging.getLogger("ultralytics").setLevel(logging.ERROR)

WINDOW_NAME = "IP Camera YOLO"

# Reconnection settings
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_SECONDS = 60.0

model = YOLO("yolov8n.pt")
TARGET_CLASSES = [0]


def open_camera_stream(url: str) -> cv2.VideoCapture:
    """Helper function to create a VideoCapture object."""
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


# Initial video stream connection
cap = open_camera_stream(settings.RTSP_URL)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Timers and counters
last_event_time = 0.0
frame_counter = 0
reconnect_attempts = 0

# Event and snapshot manager instance
shot_mgr = EventShotManager()

# Run initial directory cleanup before starting the loop
last_cleanup_time = init_cleanup()


# =========================
# MAIN LOOP
# =========================
try:
    while True:
        # Check if user closed the display window via (X) button
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        ret, frame = cap.read()

        # 🚨 Handle stream failure / disconnect with limited retries
        if not ret:
            reconnect_attempts += 1
            print(
                f"[Camera] Stream lost! Attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}."
            )
            cap.release()

            if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                print("[Camera] Maximum reconnect attempts reached. Exiting application...")
                break

            print(f"[Camera] Waiting {int(RECONNECT_DELAY_SECONDS)}s before reconnect attempt...")
            time.sleep(RECONNECT_DELAY_SECONDS)

            cap = open_camera_stream(settings.RTSP_URL)
            continue

        # Reset retry counter on successful frame capture
        reconnect_attempts = 0
        frame_counter += 1

        # Run YOLO inference on every 2nd frame to reduce latency
        if frame_counter % 2 != 0:
            continue

        frame = cv2.resize(frame, (960, 540))

        results = model(
            frame,
            classes=TARGET_CLASSES,
            conf=0.4,
            verbose=False
        )

        annotated = results[0].plot()
        detected = len(results[0].boxes) > 0
        now = time.time()

        # Handle active cooldown period between detection events
        if now - last_event_time < settings.COOLDOWN_TIME:
            cv2.imshow(WINDOW_NAME, annotated)

            # ESC key to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # Periodically check and clean up old event snapshots
        last_cleanup_time = run_cleanup_if_needed(last_cleanup_time)

        # Process detection lifecycle and handle image saving
        event_just_ended = shot_mgr.process_event(detected, annotated)
        if event_just_ended:
            last_event_time = now

        cv2.imshow(WINDOW_NAME, annotated)

        # ESC key to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Guaranteed cleanup of resources on application exit
    print("Releasing camera stream and destroying windows...")
    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")