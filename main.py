import logging
import multiprocessing as mp
import os
import signal
import time
import cv2

from app.stream_recorders.ffmpeg_recorder import FFmpegRecorder
from app.utils.silence_err_c import silence_c_libraries_safe
from app.workers.events_cleanup_worker import EventsCleanupWorker

silence_c_libraries_safe()

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0"
)

from app.camera import Camera
from app.core.settings import settings
from app.core.setup_logging import setup_logging
from app.event_detectors.yolo_detector import yolo_detection
from app.utils.force_kill_all_processes import force_kill_all_processes
from app.event_manager import EventManager

setup_logging()
logger = logging.getLogger("main")

# Limit: Process up to 5 cameras per worker instance
MAX_CAMERAS = 5


def main() -> None:
    logger.info("Camera management system initializing...")

    # 0. Start background thread to purge expired event recordings
    events_cleanup_worker = EventsCleanupWorker(
        base_dir=settings.EVENTS_DIR,
        retention_days=settings.CLEANUP_DAYS,
        check_interval_seconds=settings.CLEANUP_CHECK_INTERVAL_SECONDS,
    )
    events_cleanup_worker.start()

    # 1. Fetch maximum of 5 cameras from active configuration
    camera_items = list(settings.CAMERAS.items())[:MAX_CAMERAS]

    raw_cameras: dict[str, Camera] = {}
    for cam_id, cam_config in camera_items:
        raw_cameras[str(cam_id)] = Camera(
            camera_id=str(cam_id),
            rtsp=cam_config.get("rtsp", ""),
            camera_win_name=cam_config.get("window_name", ""),
        )

    logger.info("Waiting for RTSP connection (%d cameras)...", len(raw_cameras))

    # 2. Wait for camera threads initialization & launch recorders
    active_cameras: dict[str, Camera] = {}
    active_recorders: dict[str, FFmpegRecorder] = {}

    for cam_id, camera in raw_cameras.items():
        while camera.is_initializing:
            time.sleep(0.05)

        if camera.initial_success:
            active_cameras[cam_id] = camera

            # Initialize and launch stream dump recorder
            recorder = FFmpegRecorder(camera_id=cam_id, rtsp_url=camera.rtsp)
            recorder.start_recording()
            active_recorders[cam_id] = recorder

            logger.info("[Main] Camera '%s' connected and recording started.", cam_id)

            # Initialize resizable OpenCV window
            win_name = camera.camera_win_name or f"Camera {cam_id}"
            cv2.namedWindow(
                win_name, 
                cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO
            )
            cv2.resizeWindow(win_name, 640, 360)
        else:
            logger.error("[Main] Camera '%s' failed initial connection.", cam_id)
            camera.stop()

    if not active_cameras:
        logger.warning("No active cameras available. Exiting...")
        events_cleanup_worker.stop()
        force_kill_all_processes()
        return

    num_active = len(active_cameras)
    logger.info("Successfully initialized %d cameras.", num_active)

    # 3. Spawn a single isolated YOLO inference worker process
    input_queue = mp.Queue(maxsize=num_active * 2)
    output_queue = mp.Queue(maxsize=num_active * 2)
    yolo_stop_event = mp.Event()

    yolo_process = mp.Process(
        target=yolo_detection,
        args=(input_queue, output_queue, yolo_stop_event),
        daemon=True,
        name="yolo-worker",
    )
    yolo_process.start()
    logger.info("Started single YOLO process for %d cameras.", num_active)

    # 4. Bind system OS signals for graceful shutdown handling
    stop_event = False

    def handle_shutdown(signum, frame):
        nonlocal stop_event
        logger.info("Shutdown signal received...")
        stop_event = True

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Event manager instances for each camera
    event_managers: dict[str, EventManager] = {
        cam_id: EventManager() for cam_id in active_cameras
    }

    # Last event times for each camera
    last_event_times: dict[str, float] = {
        cam_id: 0.0 for cam_id in active_cameras
    }

    # Frame and detection metadata caches
    latest_frames: dict[str, object] = {}
    latest_detections: dict[str, list] = {}

    # 5. Main processing loop
    try:
        while not stop_event:
            # A. Fetch raw frames from active camera threads
            for cam_id, camera in list(active_cameras.items()):
                if camera.no_frame_duration > 10.0:
                    logger.warning(
                        "[Main] Camera '%s' stream lost! No frames for %.1f sec.",
                        cam_id,
                        camera.no_frame_duration,
                    )

                win_name = camera.camera_win_name or f"Camera {cam_id}"

                # Check if user closed the window using the OS 'X' button
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("[Main] Window for camera '%s' closed by user. Stopping...", cam_id)

                    cv2.destroyWindow(win_name)

                    if cam_id in active_recorders:
                        active_recorders[cam_id].stop_recording()
                        del active_recorders[cam_id]

                    if cam_id in event_managers:
                        event_managers[cam_id].stop()
                        del event_managers[cam_id]
                    last_event_times.pop(cam_id, None)

                    camera.stop()
                    del active_cameras[cam_id]
                    latest_frames.pop(cam_id, None)
                    latest_detections.pop(cam_id, None)
                    continue

                frame = camera.get_frame()
                if frame is not None:
                    latest_frames[cam_id] = frame

                    try:
                        input_queue.put_nowait((cam_id, frame))
                    except Exception:
                        pass  # Skip frame submission if YOLO queue is congested

            # Exit early if all windows were closed manually by the user
            if not active_cameras:
                logger.info("[Main] All camera windows have been closed. Exiting...")
                break

            # B. Consume lightweight metadata (detections) from YOLO worker process
            while True:
                try:
                    res_cam_id, detections = output_queue.get_nowait()
                    if res_cam_id in active_cameras:
                        latest_detections[res_cam_id] = detections
                except Exception:
                    break

            # C. Render cached frames with lightweight overlays onto OpenCV GUI windows
            for cam_id, display_frame in list(latest_frames.items()):
                if cam_id in active_cameras:
                    win_name = active_cameras[cam_id].camera_win_name or f"Camera {cam_id}"
                    detections = latest_detections.get(cam_id, [])

                    if detections:
                        now = time.monotonic()
                        cooldown = getattr(settings, "EVENT_COOLDOWN", 5.0)

                        if now - last_event_times.get(cam_id, 0.0) >= cooldown:
                            last_event_times[cam_id] = now
                            event_managers[cam_id].handle_detection(
                                active_cameras[cam_id], display_frame
                            )

                        # Copy frame array only when drawing overlays
                        render_frame = display_frame.copy()  # type: ignore
                        h, w = render_frame.shape[:2]

                        # Fast pixel denormalization and drawing loop
                        for det in detections:
                            box_norm = det.get("box_norm", [0, 0, 0, 0])
                            x1 = int(box_norm[0] * w)
                            y1 = int(box_norm[1] * h)
                            x2 = int(box_norm[2] * w)
                            y2 = int(box_norm[3] * h)

                            label_name = det.get("label", det.get("class_name", "object"))
                            conf = det.get("conf", 0.0)
                            label_text = f"{label_name} {conf:.2f}"

                            # 1. Bounding box
                            cv2.rectangle(render_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # 2. Text background badge
                            (text_w, text_h), baseline = cv2.getTextSize(
                                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )
                            text_y1 = max(y1 - text_h - 4, 0)

                            cv2.rectangle(render_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            cv2.putText(
                                render_frame,
                                label_text,
                                (x1, max(y1 - 7, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                        cv2.imshow(win_name, render_frame)  # type: ignore

                    else:
                        # Direct display without copying memory
                        cv2.imshow(win_name, display_frame)  # type: ignore

            # D. Ensure stream recorders are alive and perform hourly rotation
            for recorder in active_recorders.values():
                if not recorder.is_recording():
                    logger.warning("[Main] Recorder '%s' died, restarting...", recorder.camera_id)
                    recorder.start_recording()
                else:
                    recorder.rotate_if_new_hour()

            # E. Process GUI event loop and key press actions
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                logger.info("Quit key pressed ('q' or ESC). Exiting main loop...")
                stop_event = True
                break

    except Exception as exc:
        logger.exception("Unexpected error in main loop: %s", exc)

    finally:
        logger.info("Closing OpenCV GUI windows...")
        cv2.destroyAllWindows()

        logger.info("Stopping stream recorders...")
        for recorder in active_recorders.values():
            recorder.stop_recording()

        logger.info("Stopping event managers...")
        for manager in event_managers.values():
            manager.stop()

        logger.info("Stopping background cleanup worker...")
        events_cleanup_worker.stop()

        logger.info("Stopping YOLO process...")
        if yolo_stop_event is not None:
            yolo_stop_event.set()

        if yolo_process is not None and yolo_process.is_alive():
            yolo_process.join(timeout=1.5)

            if yolo_process.is_alive():
                logger.warning("YOLO process did not exit gracefully. Terminating...")
                yolo_process.terminate()
                yolo_process.join(timeout=1.0)

        # Release IPC queues to avoid feeder thread deadlocks
        if input_queue is not None:
            input_queue.close()
            input_queue.cancel_join_thread()

        if output_queue is not None:
            output_queue.close()
            output_queue.cancel_join_thread()

        logger.info("Stopping camera worker threads...")
        for camera in active_cameras.values():
            camera.stop()

        logger.info("All camera components stopped clean.")
        force_kill_all_processes()


if __name__ == "__main__":
    main()