import logging
import multiprocessing as mp
import os
import time
import numpy as np

logger = logging.getLogger("yolo_worker")


def yolo_detection(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    stop_event: mp.Event,  # type: ignore
) -> None:
    """Independent multiprocessing worker for YOLO object detection.
    
    Returns normalized bounding box coordinates [0.0 - 1.0] and class labels
    for maximum IPC throughput and fast rendering.
    """
    # Restrict OpenMP / BLAS threads inside child process to prevent CPU starvation
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass

    # Imports inside child process to prevent shared IPC/socket state corruption
    from app.core.settings import settings

    logger.info("[YOLO] Worker process starting...")

    # Configuration extraction
    model_path = str(settings.YOLO_MODEL_PATH)
    imgsz = settings.YOLO_FRAME_SIZE
    conf_threshold = settings.YOLO_CONFIDENCE
    person_ids = settings.YOLO_PERSON_CLASS_IDS
    device = settings.YOLO_DEVICE
    infer_interval = settings.YOLO_FRAME_INTERVAL

    # Model Initialization
    logger.info("[YOLO] Loading model from: %s", model_path)
    try:
        model = YOLO(model_path, task="detect")
    except Exception as exc:
        logger.critical("[YOLO] Failed to load model: %s", exc)
        return

    # Warm-up pass
    try:
        warmup_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(
            warmup_frame,
            imgsz=imgsz,
            classes=person_ids,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )
        logger.info("[YOLO] Model warm-up completed successfully.")
    except Exception as exc:
        logger.warning("[YOLO] Model warm-up failed: %s", exc)

    # Class names map extraction (e.g. {0: 'person', 1: 'bicycle', ...})
    model_names = getattr(model, "names", {})

    last_infer_times: dict[str, float] = {}

    logger.info("[YOLO] Worker loop ready. Target FPS: %d", getattr(settings, "YOLO_FPS", 15))

    # Main Inference Loop
    try:
        while not stop_event.is_set():
            try:
                task = input_queue.get(timeout=0.02)
            except Exception:
                continue

            if not isinstance(task, (tuple, list)) or len(task) < 2:
                continue

            camera_id, frame = task[0], task[1]
            if camera_id is None or frame is None:
                continue

            # Per-camera FPS limit check
            now = time.monotonic()
            last_infer = last_infer_times.get(camera_id, 0.0)
            if now - last_infer < infer_interval:
                continue

            # Inference execution
            try:
                results = model.predict(
                    frame,
                    imgsz=imgsz,
                    classes=person_ids,
                    conf=conf_threshold,
                    device=device,
                    verbose=False,
                )
            except Exception as exc:
                logger.error("[YOLO:%s] Inference error: %s", camera_id, exc)
                continue

            last_infer_times[camera_id] = time.monotonic()

            detections = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                # Single bulk copy from GPU/CPU tensor to NumPy arrays
                coords_norm = np.round(boxes.xyxyn.cpu().numpy(), 4).tolist()  # type: ignore
                confs = np.round(boxes.conf.cpu().numpy(), 2).tolist()  # type: ignore
                clss = boxes.cls.cpu().numpy().astype(int).tolist()  # type: ignore

                # Construct payload with explicit string labels
                for i in range(len(coords_norm)):
                    class_id = clss[i]
                    # Fetch class string name with safety fallback
                    label_name = model_names.get(class_id, f"cls_{class_id}")
                    
                    detections.append({
                        "box_norm": coords_norm[i],
                        "conf": confs[i],
                        "label": str(label_name),
                        "class_id": class_id,
                    })

            # Dispatch payload containing lightweight detection metadata
            _push_to_output_queue(
                output_queue,
                (camera_id, detections),
            )

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception("[YOLO] Unhandled worker exception: %s", exc)
    finally:
        logger.info("[YOLO] Shutting down worker process...")


def _push_to_output_queue(output_queue: mp.Queue, item: tuple) -> None:
    """Non-blocking queue push. Drops older detection frame if queue is full."""
    try:
        output_queue.put_nowait(item)
        return
    except Exception:
        pass

    # Purge stale detection to keep queue size low and avoid frame lag
    try:
        output_queue.get_nowait()
    except Exception:
        return

    try:
        output_queue.put_nowait(item)
    except Exception:
        pass