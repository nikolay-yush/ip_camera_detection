import os

# Restrict core C-libraries (OpenMP/MKL) to 6 threads to prevent CPU oversubscription
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

import time
import traceback
import multiprocessing as mp
import numpy as np
import torch
from ultralytics import YOLO

from app.event_manager import EventManager
from app.settings import settings
from app.utils.shm_ring_buffer import SharedMemoryRingBuffer


def yolo_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    stop_event: mp.Event,  # type: ignore
) -> None:
    print("[YOLO Worker] Process started. Initializing Shared Memory access...", flush=True)

    # Restrict PyTorch thread pool for intra/inter-op parallelism
    torch.set_num_threads(6)
    torch.set_num_interop_threads(6)

    # Attach to the existing Shared Memory ring buffer initialized by the main process
    shm_slots = getattr(settings, "SHM_SLOTS", max(12, len(settings.CAMERAS) * 4))
    shm_buffer = SharedMemoryRingBuffer(
        name_prefix="yolo_frames",
        num_slots=shm_slots,
        shape=(settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE, 3),
        create=False,
    )

    # Explicitly pass task="detect" to suppress ambiguity warnings across backends
    model_path = getattr(settings, "YOLO_MODEL_PATH", "yolov8n_openvino_model")
    print(f"[YOLO Worker] Loading model from {model_path}...", flush=True)
    model = YOLO(model_path, task="detect")

    # Safe retrieval of configuration variables
    conf_threshold = getattr(settings, "YOLO_CONF", getattr(settings, "YOLO_CONFIDENCE", 0.35))
    person_ids = getattr(settings, "YOLO_PERSON_CLASS_IDS", [0])

    # Warm up model runtime using a dummy zero-array
    dummy = np.zeros(
        (settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE, 3),
        dtype=np.uint8,
    )
    model.predict(
        dummy,
        classes=person_ids,
        conf=conf_threshold,
        imgsz=settings.YOLO_FRAME_SIZE,
        device="cpu",
        verbose=False,
    )

    event_manager = EventManager()
    last_event_times: dict[str, float] = {}
    last_infer_times: dict[str, float] = {}
    infer_interval = getattr(settings, "YOLO_INFER_INTERVAL", 0.08)

    print("[YOLO Worker] Model loaded and ready for inference loop.", flush=True)

    try:
        while not stop_event.is_set():
            try:
                # Receive lightweight metadata task item with a short timeout
                task = input_queue.get(timeout=0.01)
            except Exception:
                continue

            # Safe tuple unpacking to prevent "list index out of range"
            if isinstance(task, (tuple, list)) and len(task) >= 2:
                camera_id = task[0]
                slot_idx = task[1]
            else:
                continue

            now = time.monotonic()
            last_infer = last_infer_times.get(camera_id, 0.0)

            # Throttle inference frequency based on configured interval
            if now - last_infer < infer_interval:
                continue

            last_infer_times[camera_id] = now

            # Fetch frame array slice from Shared Memory using correct get_array method
            try:
                raw_frame = shm_buffer.get_array(slot_idx)
                if raw_frame is None:
                    continue
                frame = np.ascontiguousarray(raw_frame)
            except Exception:
                continue

            # Execute forward inference pass
            results = model.predict(
                frame,
                classes=person_ids,
                conf=conf_threshold,
                imgsz=settings.YOLO_FRAME_SIZE,
                device="cpu",
                verbose=False,
            )

            # Check if inference returned valid results
            if results and len(results) > 0:
                det = results[0]
                has_person = len(det.boxes) > 0  # type: ignore

                annotated_frame = det.plot()

                if has_person:
                    last_cam_event = last_event_times.get(camera_id, 0.0)
                    cooldown = getattr(settings, "EVENT_COOLDOWN", 5.0)
                    if now - last_cam_event >= cooldown:
                        last_event_times[camera_id] = now
                        event_manager.handle_detection(
                            camera_id=camera_id,
                            frame=annotated_frame,
                        )

                # Send (camera_id, annotated_frame) back to match main.py expectation
                _push_to_output_queue(output_queue, (camera_id, annotated_frame))

    except Exception as exc:
        print(f"[YOLO Worker] Error: {exc}", flush=True)
        traceback.print_exc()

    finally:
        print("[YOLO Worker] Cleaning up resources...", flush=True)
        try:
            shm_buffer.close()
        except Exception:
            pass

        try:
            # Correct stop method for EventManager
            if hasattr(event_manager, "stop"):
                event_manager.stop()
        except Exception:
            pass


def _push_to_output_queue(output_queue: mp.Queue, item: tuple) -> None:
    """Helper function to safely put items in queue without blocking."""
    if output_queue.full():
        try:
            output_queue.get_nowait()
        except Exception:
            pass
    try:
        output_queue.put_nowait(item)
    except Exception:
        pass