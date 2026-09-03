import time
import multiprocessing as mp
import numpy as np
import torch
from ultralytics import YOLO

from app.event_manager import EventManager
from app.settings import settings


def yolo_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    stop_event: mp.Event,  # type: ignore
) -> None:
    print("[YOLO Worker] Process started. Checking target compute device...")
    
    # 1. Automatic compute accelerator selection (CUDA if available, fallback to CPU)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    if settings.YOLO_DEVICE:
        target_device = settings.YOLO_DEVICE.lower()

    print(f"[YOLO Worker] Selected target device: '{target_device.upper()}'")

    print("[YOLO Worker] Loading model...")
    model = YOLO(settings.YOLO_MODEL_PATH)

    # 2. Warm up model to allocate engine resources
    print("[YOLO Worker] Warming up model...")
    dummy = np.zeros(
        (settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE, 3),
        dtype=np.uint8,
    )
    model.predict(
        dummy,
        classes=settings.YOLO_PERSON_CLASS_IDS,
        conf=settings.YOLO_CONFIDENCE,
        imgsz=settings.YOLO_FRAME_SIZE,
        device=target_device,
        verbose=False,
    )
    print("[YOLO Worker] Model warmup complete. Ready for frames.")

    event_manager = EventManager()
    last_event_times: dict[str, float] = {}
    
    # Track inference timestamps per camera to throttle FPS (e.g., max ~12 FPS per camera)
    last_infer_times: dict[str, float] = {}
    infer_interval = getattr(settings, "YOLO_INFER_INTERVAL", 0.08)  # ~12 FPS throttle per stream

    while not stop_event.is_set():
        try:
            # 3. Blocking queue fetch with timeout to eliminate idle CPU spin cycles
            try:
                camera_id, frame = input_queue.get(timeout=0.2)
            except Exception:
                continue

            now = time.monotonic()
            last_infer = last_infer_times.get(camera_id, 0.0)

            # 4. Skip inference if time elapsed since last detection is below interval threshold
            if now - last_infer < infer_interval:
                # Forward raw unannotated frame to prevent video pipeline latency
                if output_queue.full():
                    try:
                        output_queue.get_nowait()
                    except Exception:
                        pass
                output_queue.put((camera_id, frame))
                continue

            last_infer_times[camera_id] = now

            # 5. Run object detection on target compute device
            results = model.predict(
                frame,
                classes=settings.YOLO_PERSON_CLASS_IDS,
                conf=settings.YOLO_CONFIDENCE,
                imgsz=settings.YOLO_FRAME_SIZE,
                device=target_device,
                verbose=False,
            )[0]

            has_person = len(results.boxes) > 0  # type: ignore
            annotated_frame = None

            if has_person:
                # Draw bounding boxes directly onto image buffer without extra allocation
                annotated_frame = results.plot(img=frame)

                last_cam_event = last_event_times.get(camera_id, 0.0)
                if now - last_cam_event >= settings.EVENT_COOLDOWN:
                    last_event_times[camera_id] = now
                    event_manager.handle_detection(
                        camera_id=camera_id,
                        frame=annotated_frame,
                    )

            # 6. Evict stale frames from output queue to maintain real-time sync
            if output_queue.full():
                try:
                    output_queue.get_nowait()
                except Exception:
                    pass

            out_frame = annotated_frame if annotated_frame is not None else frame
            output_queue.put((camera_id, out_frame))

        except Exception as exc:
            if not stop_event.is_set():
                print(f"[YOLO Worker] Error during frame processing: {exc}")

    try:
        event_manager.stop()
    except Exception:
        pass

    print("[YOLO Worker] Stopped successfully.")