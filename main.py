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
from app.workers.yolo_worker import yolo_worker
from app.workers.cleanup_worker import cleanup_worker
from app.utils.is_recording_time import is_recording_time
from app.utils.force_kill_self import force_kill_self


# ============================================================
# Environment Setup (Silence & Low Latency Streaming)
# ============================================================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0"
)

os.environ["OPENCV_LOG_LEVEL"] = "OFF"


try:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)
except Exception:
    pass

# ============================================================





def main():
    print("[System] Starting Camera Detection YOLO Application...")

    cameras: dict[str, Camera] = {}
    recorders: dict[str, NativeFFmpegRecorder] = {}

    # Cache latest YOLO annotated frames for rendering
    latest_frames: dict[str, np.ndarray] = {}

    # Check for scheduled recording start/stop events every 5 seconds
    last_schedule_check = time.time()

    # Placeholders for multiprocessing objects
    yolo_process = None
    yolo_input_queue = None
    yolo_output_queue = None
    yolo_stop_event = None

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
            print("[System] CRITICAL: No cameras defined in CAMERAS environment JSON!")
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
        # 3. Multiprocessing Setup for YOLO Inference
        # --------------------------------------------------------
        print("\n[System] Spawning Isolated YOLO Process...")
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

            camera_win_name = f"Camera: {cinfo.get('location', 'Unnamed')} ({cinfo.get('model', 'Unnamed')})"

            cameras[cid] = Camera(camera_id=cid, rtsp=rtsp, camera_win_name=camera_win_name)

            cv2.namedWindow(camera_win_name, cv2.WINDOW_NORMAL)

        if not cameras:
            print("[System] CRITICAL: No valid cameras could be connected. Exiting...")
            raise RuntimeError("No cameras connected.")

        # Initial schedule check on start
        if is_recording_time():
            for cid, rec in recorders.items():
                if cid in cameras and not rec.is_recording():
                    print(f"[Recorder:{cid}] Starting initial recording...", flush=True)
                    rec.start_recording()

        print(f"[System] All {len(cameras)} cameras connected. Starting main processing loop...\n")

        # --------------------------------------------------------
        #  Processing & Event Loop
        # --------------------------------------------------------
        while True:
            now = time.time()

            # Check for scheduled recording start/stop events
            if now - last_schedule_check >= 5.0:
                last_schedule_check = now
                should_record = is_recording_time()

                for cid, rec in list(recorders.items()):
                    if cid in cameras:
                        if should_record and not rec.is_recording():
                            print(f"[Recorder:{cid}] Starting scheduled recording...", flush=True)
                            rec.start_recording()
                        elif not should_record and rec.is_recording():
                            print(f"[Recorder:{cid}] Stopping scheduled recording...", flush=True)
                            rec.stop_recording()

            # Retrieve processed frames from YOLO output queue
            while not yolo_output_queue.empty():
                try:
                    out_cid, processed_frame = yolo_output_queue.get_nowait()
                    latest_frames[out_cid] = processed_frame
                except Exception:
                    break
            
            # Read streams from all active cameras and display them
            for cid, cam in list(cameras.items()):
                frame = cam.get_frame()

                if frame is None:
                    continue

                if not yolo_input_queue.full():
                    try:
                        yolo_input_queue.put_nowait((cid, frame))
                    except Exception:
                        pass

                display_frame = latest_frames.get(cid, frame)
                cv2.imshow(cam.camera_win_name, display_frame)

            # Process GUI events (Required by OpenCV HighGUI)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[System] Exit requested via key press ('q' or ESC). Exiting main loop...", flush=True)
                break

            # Check for closed windows
            closed_cids = []
            for cid, cam in cameras.items():
                prop_visible = cv2.getWindowProperty(cam.camera_win_name, cv2.WND_PROP_VISIBLE)
                prop_autosize = cv2.getWindowProperty(cam.camera_win_name, cv2.WND_PROP_AUTOSIZE)

                if prop_visible < 1 or prop_autosize < 0:
                    closed_cids.append(cid)

            # Safely release resources for closed windows
            for cid in closed_cids:
                cam = cameras[cid]
                print(f"\n[System] Window '{cam.camera_win_name}' closed. Releasing camera [{cid}]...", flush=True)

                if cid in recorders:
                    try:
                        if recorders[cid].is_recording():
                            recorders[cid].stop_recording()
                            print(f"[System] [{cid}] Recorder stopped.", flush=True)
                    except Exception as e:
                        print(f"[System] Error stopping recorder [{cid}]: {e}", flush=True)
                    del recorders[cid]

                try:
                    cam.stop()
                    print(f"[System] [{cid}] Stream capture stopped.", flush=True)
                except Exception as e:
                    print(f"[System] Error stopping camera [{cid}]: {e}", flush=True)

                try:
                    cv2.destroyWindow(cam.camera_win_name)
                except Exception:
                    pass

                del cameras[cid]
                latest_frames.pop(cid, None)

            # Exit if all camera display windows are closed
            if not cameras:
                print("[System] All camera windows closed. Exiting main loop...", flush=True)
                break

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user or signal.", flush=True)
        
    except Exception as e:
        print(f"\n[System] CRITICAL Exception: {e}", flush=True)

    finally:
        print("\n[System] Cleaning up resources...", flush=True)

        # 1. Stop all active recorders
        for cid, rec in list(recorders.items()):
            try:
                if rec.is_recording():
                    print(f"[System] [{cid}] Stopping recorder...", flush=True)
                    rec.stop_recording()
            except Exception as e:
                print(f"[System] Error stopping recorder {cid}: {e}", flush=True)

        # 2. Stop camera stream capture threads
        for cid, cam in list(cameras.items()):
            try:
                cam.stop()
            except Exception as e:
                print(f"[System] Error stopping camera {cid}: {e}", flush=True)

        # 3. Kill YOLO subprocess
        if yolo_process is not None and yolo_process.is_alive():
            try:
                print("[System] Terminating YOLO process...", flush=True)
                yolo_process.terminate()
                yolo_process.kill()
            except Exception:
                pass

        # 4. Destroy OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[System] Cleanup done. Force killing process group now.", flush=True)
        
        # Hard terminate all processes in this group immediately
        force_kill_self()


if __name__ == "__main__":
    print("[System] Setting multiprocessing start method to 'spawn'...")
    mp.set_start_method("spawn", force=True)
    
    # Create new process group so force_kill_self() can kill all sub-processes at once
    try:
        os.setpgrp()
    except Exception:
        pass

    main()