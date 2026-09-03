import os
import signal
import sys
import threading
import time
import multiprocessing as mp
import cv2
import numpy as np

from app.camera import Camera
from app.cleanup import init_cleanup
from app.ffmpeg_recorder import NativeFFmpegRecorder
from app.settings import settings
from app.utils.silence_err_c import silence_c_libraries
from app.workers.yolo_worker import yolo_worker
from app.workers.cleanup_worker import cleanup_worker
from app.utils.is_recording_time import is_recording_time
from app.utils.force_kill_self import force_kill_self
from app.utils.shm_ring_buffer import SharedMemoryRingBuffer


# ============================================================
# Environment Setup (Low Latency RTSP & OpenCV Config)
# ============================================================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"  # Enforce TCP transport for stream stability
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0"
)

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

silence_c_libraries()  # Suppress C-level stderr logs from FFmpeg/OpenCV


# ============================================================
# Main Application
# ============================================================

def main():
    print("[System] Starting Camera Detection YOLO Application...")

    cameras: dict[str, Camera] = {}
    recorders: dict[str, NativeFFmpegRecorder] = {}

    # Cache latest YOLO annotated frames for display rendering
    latest_frames: dict[str, np.ndarray] = {}

    # Timers for background health checks
    last_schedule_check = 0.0

    # Placeholders for multiprocessing components
    yolo_process: mp.Process | None = None
    yolo_input_queue: mp.Queue | None = None
    yolo_output_queue: mp.Queue | None = None
    yolo_stop_event: mp.Event | None = None  # type: ignore
    shm_buffer: SharedMemoryRingBuffer | None = None

    # --------------------------------------------------------
    # 0. Shutdown Signal Handler
    # --------------------------------------------------------
    def shutdown(sig, frame):
        print("\n[System] Shutdown signal received. Triggering exit...", flush=True)
        raise KeyboardInterrupt  

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        # --------------------------------------------------------
        # 1. Load Camera Configurations & Recorders
        # --------------------------------------------------------
        if not settings.CAMERAS:
            print("[System] CRITICAL: No cameras defined in CAMERAS configuration!")
            sys.exit(1)

        print(f"[System] Loaded {len(settings.CAMERAS)} camera(s) from settings:")
        for cid, cinfo in settings.CAMERAS.items():
            cam_name = cinfo.get("location", cinfo.get("model", cid))
            rtsp_url = cinfo.get("rtsp")

            if not rtsp_url:
                print(f"[System] WARNING: Missing RTSP URL for camera '{cid}'. Skipping...")
                continue

            print(f"Camera: {cid} ({cam_name}) | RTSP: {rtsp_url}")

            if cinfo.get("is_recording", False):
                print(f"[System] [{cid}] Initializing Native FFmpeg Recorder...")
                recorders[cid] = NativeFFmpegRecorder(cid, rtsp_url)

        # --------------------------------------------------------
        # 2. Storage Cleanup Worker
        # --------------------------------------------------------
        print("\n[System] Initializing Automated Cleanup Worker...")
        last_cleanup_time = init_cleanup()
        cleanup_thread = threading.Thread(
            target=cleanup_worker,
            args=(last_cleanup_time,),
            daemon=True,
            name="cleanup-worker",
        )
        cleanup_thread.start()

        # --------------------------------------------------------
        # 3. Multiprocessing & Shared Memory Setup for YOLO
        # --------------------------------------------------------
        print("\n[System] Initializing Shared Memory Ring Buffer...")
        shm_slots = getattr(settings, "SHM_SLOTS", 6)
        shm_buffer = SharedMemoryRingBuffer(
            name_prefix="yolo_frames",
            num_slots=shm_slots,
            shape=(settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE, 3),
            dtype=np.uint8,
            create=True,
        )

        print("[System] Spawning Isolated YOLO Process...")
        queue_size = max(4, len(settings.CAMERAS) * 2)
        yolo_input_queue = mp.Queue(maxsize=queue_size)
        yolo_output_queue = mp.Queue(maxsize=queue_size)
        yolo_stop_event = mp.Event()

        yolo_process = mp.Process(
            target=yolo_worker,
            args=(yolo_input_queue, yolo_output_queue, yolo_stop_event),
            name="yolo-worker-process",
            daemon=True,
        )
        yolo_process.start()

        if not yolo_process.is_alive():
            print("[System] CRITICAL: YOLO process failed to start. Exiting...")
            raise RuntimeError("YOLO process failed to start!")

        # --------------------------------------------------------
        # 4. Connect ALL Camera Streams & Windows Setup
        # --------------------------------------------------------
        print("\n[System] Connecting to Camera Streams...")
        for cid, cinfo in settings.CAMERAS.items():
            rtsp = cinfo.get("rtsp")

            if not rtsp:
                if cid in recorders:
                    if recorders[cid].is_recording():
                        recorders[cid].stop_recording()
                    del recorders[cid]
                continue

            camera_win_name = f"Camera: {cinfo.get('location', 'Unnamed')} | ({cinfo.get('model', 'Unnamed')}) | [{cid}]"

            cameras[cid] = Camera(camera_id=cid, rtsp=rtsp, camera_win_name=camera_win_name)
            cv2.namedWindow(camera_win_name, cv2.WINDOW_NORMAL)

        if not cameras:
            print("[System] CRITICAL: No valid cameras could be connected. Exiting...")
            raise RuntimeError("No cameras connected.")

        print(f"[System] All {len(cameras)} cameras connected. Starting main processing loop...\n")

        # --------------------------------------------------------
        # Processing & Event Loop
        # --------------------------------------------------------
        while True:
            now = time.time()

            # 1. Scheduled recording maintenance & Hourly file rotation (every 5s)
            if now - last_schedule_check >= 5.0:
                last_schedule_check = now
                should_record = is_recording_time()

                for cid, rec in recorders.items():
                    if cid in cameras:
                        if should_record:
                            if not rec.is_recording():
                                print(f"[Recorder:{cid}] Starting scheduled recording...", flush=True)
                                rec.start_recording()
                            else:
                                # Ensure hour transition rotation is executed
                                rec.check_hour_rotation()
                        elif not should_record and rec.is_recording():
                            print(f"[Recorder:{cid}] Stopping scheduled recording...", flush=True)
                            rec.stop_recording()

            # 2. Retrieve processed metadata or frames from YOLO worker
            while not yolo_output_queue.empty():
                try:
                    res = yolo_output_queue.get_nowait()
                    if len(res) == 2:
                        out_cid, processed_frame = res
                        latest_frames[out_cid] = processed_frame
                except Exception:
                    break

            # 3. Read camera frames, write to SHM, and queue index
            for cid, cam in list(cameras.items()):
                frame = cam.get_frame()

                if frame is None:
                    continue

                # Prepare frame matching YOLO input resolution
                if frame.shape[:2] != (settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE):
                    yolo_frame = cv2.resize(
                        frame, (settings.YOLO_FRAME_SIZE, settings.YOLO_FRAME_SIZE)
                    )
                else:
                    yolo_frame = frame

                # Write directly to Shared Memory and get slot index
                slot_idx, _ = shm_buffer.write_next(yolo_frame)

                # Push slot reference to worker queue (Zero-Copy)
                if not yolo_input_queue.full():
                    try:
                        yolo_input_queue.put_nowait((cid, slot_idx))
                    except Exception:
                        pass

                # Display latest annotated frame or fallback to raw stream
                display_frame = latest_frames.get(cid, frame)
                cv2.imshow(cam.camera_win_name, display_frame)

            # 4. Handle OpenCV UI events (1ms delay keeps UI fluid)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[System] Exit requested via key press ('q' or ESC). Exiting main loop...", flush=True)
                break

            # 5. Check for GUI window closure events
            closed_cids = []
            for cid, cam in cameras.items():
                prop_visible = cv2.getWindowProperty(cam.camera_win_name, cv2.WND_PROP_VISIBLE)
                if prop_visible < 1:
                    closed_cids.append(cid)

            # 6. Safely release resources for closed camera windows
            for cid in closed_cids:
                cam = cameras.pop(cid)
                print(f"\n[System] Window '{cam.camera_win_name}' closed. Releasing camera [{cid}]...", flush=True)

                if cid in recorders:
                    try:
                        if recorders[cid].is_recording():
                            recorders[cid].stop_recording()
                    except Exception as e:
                        print(f"[System] Error stopping recorder [{cid}]: {e}", flush=True)
                    del recorders[cid]

                try:
                    cam.stop()
                except Exception as e:
                    print(f"[System] Error stopping camera [{cid}]: {e}", flush=True)

                try:
                    cv2.destroyWindow(cam.camera_win_name)
                except Exception:
                    pass

                latest_frames.pop(cid, None)

            # Break loop if all windows were closed
            if not cameras:
                print("[System] All camera windows closed. Exiting main loop...", flush=True)
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user or signal.", flush=True)

    except Exception as e:
        print(f"\n[System] CRITICAL Exception: {e}", flush=True)

    finally:
        print("\n[System] Cleaning up resources...", flush=True)

        # 1. Safely stop all active native FFmpeg recorders
        for cid, rec in recorders.items():
            try:
                if rec.is_recording():
                    print(f"[System] [{cid}] Stopping recorder...", flush=True)
                    rec.stop_recording()
            except Exception as e:
                print(f"[System] Error stopping recorder {cid}: {e}", flush=True)

        # 2. Stop camera video capture threads
        for cid, cam in cameras.items():
            try:
                cam.stop()
            except Exception as e:
                print(f"[System] Error stopping camera {cid}: {e}", flush=True)

        # 3. Gracefully terminate YOLO subprocess and drain IPC queues
        if yolo_stop_event is not None:
            yolo_stop_event.set()

        if yolo_input_queue is not None:
            yolo_input_queue.close()
            yolo_input_queue.cancel_join_thread()

        if yolo_output_queue is not None:
            yolo_output_queue.close()
            yolo_output_queue.cancel_join_thread()

        if yolo_process is not None and yolo_process.is_alive():
            print("[System] Terminating YOLO process...", flush=True)
            yolo_process.join(timeout=1.0)
            if yolo_process.is_alive():
                yolo_process.terminate()

        # 4. Release Shared Memory Block from OS Kernel
        if shm_buffer is not None:
            print("[System] Unlinking Shared Memory allocations...", flush=True)
            shm_buffer.close()
            shm_buffer.unlink()

        # 5. Destroy active OpenCV GUI windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[System] Cleanup complete. Terminating process tree.", flush=True)
        force_kill_self()


if __name__ == "__main__":
    print("[System] Setting multiprocessing start method to 'spawn'...")
    mp.set_start_method("spawn", force=True)

    main()