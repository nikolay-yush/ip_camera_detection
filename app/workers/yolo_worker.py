import os

# Restrict core C-libraries (OpenMP/MKL) to 6 threads to prevent CPU oversubscription
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

import time
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
    shm_buffer = SharedMemoryRingBuffer(
        name_prefix="yolo_frames",
        num_slots=getattr(settings, "SHM_SLOTS", 6),
        shape=(settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE, 3),
        create=False,
    )

    # Explicitly pass task="detect" to suppress ambiguity warnings across backends
    print(f"[YOLO Worker] Loading model from {settings.YOLO_MODEL_PATH}...", flush=True)
    model = YOLO(settings.YOLO_MODEL_PATH, task="detect")

    # Warm up model runtime using a dummy zero-array
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

    event_manager = EventManager()
    last_event_times: dict[str, float] = {}
    last_infer_times: dict[str, float] = {}
    infer_interval = getattr(settings, "YOLO_INFER_INTERVAL", 0.08)

    print("[YOLO Worker] Model loaded and ready for inference loop.", flush=True)

    try:
        while not stop_event.is_set():
            try:
                # Receive lightweight metadata tuple: camera_id & target slot_idx
                camera_id, slot_idx = input_queue.get(timeout=0.1)
            except Exception:
                continue

            now = time.monotonic()
            last_infer = last_infer_times.get(camera_id, 0.0)

            # Throttle inference frequency based on configured interval
            if now - last_infer < infer_interval:
                continue

            last_infer_times[camera_id] = now

            # Fetch frame array slice from Shared Memory and guarantee contiguous C-layout
            raw_frame = shm_buffer.get_array(slot_idx)
            frame = np.ascontiguousarray(raw_frame)

            # Execute forward inference pass
            results = model.predict(
                frame,
                classes=settings.YOLO_PERSON_CLASS_IDS,
                conf=settings.YOLO_CONFIDENCE,
                imgsz=settings.YOLO_FRAME_SIZE,
                device="cpu",
                verbose=False,
            )[0]

            has_person = len(results.boxes) > 0  # type: ignore

            if has_person:
                annotated_frame = results.plot()

                last_cam_event = last_event_times.get(camera_id, 0.0)
                if now - last_cam_event >= settings.EVENT_COOLDOWN:
                    last_event_times[camera_id] = now
                    event_manager.handle_detection(
                        camera_id=camera_id,
                        frame=annotated_frame,
                    )

                # Send detection notification and slot index back to main process
                _push_to_output_queue(output_queue, (camera_id, slot_idx, True))

    except Exception as exc:
        print(f"[YOLO Worker] Error: {exc}", flush=True)

    finally:
        print("[YOLO Worker] Cleaning up resources...", flush=True)
        shm_buffer.close()
        try:
            event_manager.stop()
        except Exception:
            pass


def _push_to_output_queue(output_queue: mp.Queue, item: tuple) -> None:
    if output_queue.full():
        try:
            output_queue.get_nowait()
        except Exception:
            pass
    try:
        output_queue.put_nowait(item)
    except Exception:
        pass