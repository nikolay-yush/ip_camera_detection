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
    print("[YOLO Worker] Process started. Initializing Shared Memory access...")
    
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

    # Attach to existing shared memory ring buffer created by main process
    shm_buffer = SharedMemoryRingBuffer(
        name_prefix="yolo_frames",
        num_slots=getattr(settings, "SHM_SLOTS", 6),
        shape=(settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE, 3),
        create=False,
    )

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    if settings.YOLO_DEVICE:
        target_device = settings.YOLO_DEVICE.lower()

    use_half = target_device == "cuda"
    print(f"[YOLO Worker] Compute target: {target_device.upper()} | FP16: {use_half}")

    model = YOLO(settings.YOLO_MODEL_PATH)

    # Warmup model
    dummy = np.zeros(
        (settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE, 3),
        dtype=np.uint8,
    )
    with torch.inference_mode():
        model.predict(
            dummy,
            classes=settings.YOLO_PERSON_CLASS_IDS,
            conf=settings.YOLO_CONFIDENCE,
            imgsz=settings.YOLO_FRAME_SIZE,
            device=target_device,
            half=use_half,
            verbose=False,
        )

    event_manager = EventManager()
    last_event_times: dict[str, float] = {}
    last_infer_times: dict[str, float] = {}
    infer_interval = getattr(settings, "YOLO_INFER_INTERVAL", 0.08)

    try:
        while not stop_event.is_set():
            try:
                # Receive lightweight tuple: camera_id & slot_idx (Only ~30 bytes IPC bandwidth)
                camera_id, slot_idx = input_queue.get(timeout=0.1)
            except Exception:
                continue

            # Zero-copy access to the frame array in Shared RAM
            frame = shm_buffer.get_array(slot_idx)

            now = time.monotonic()
            last_infer = last_infer_times.get(camera_id, 0.0)

            if now - last_infer < infer_interval:
                continue

            last_infer_times[camera_id] = now

            with torch.inference_mode():
                results = model.predict(
                    frame,
                    classes=settings.YOLO_PERSON_CLASS_IDS,
                    conf=settings.YOLO_CONFIDENCE,
                    imgsz=settings.YOLO_FRAME_SIZE,
                    device=target_device,
                    half=use_half,
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

                # Send detection notice/slot back to main process if needed
                _push_to_output_queue(output_queue, (camera_id, slot_idx, True))

    except Exception as exc:
        print(f"[YOLO Worker] Error: {exc}")

    finally:
        print("[YOLO Worker] Cleaning up resources...")
        shm_buffer.close()
        try:
            event_manager.stop()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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