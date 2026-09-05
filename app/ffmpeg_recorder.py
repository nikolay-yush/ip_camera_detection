import os
import subprocess
from datetime import datetime
from pathlib import Path
from app.settings import settings


class NativeFFmpegRecorder:

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.process: subprocess.Popen | None = None
        self._current_file_hour: int | None = None

    def start_recording(self) -> bool:
        """Starts raw RTSP stream dumping to MP4 file organized by date and camera folders."""
        if self.is_recording():
            return False

        now = datetime.now()
        self._current_file_hour = now.hour

        # Target folder structure: EVENTS_DIR / YYYY-MM-DD / camera_id
        date_str = now.strftime("%Y-%m-%d")
        cam_dir = settings.EVENTS_DIR / date_str / self.camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        # File name is just the starting hour (e.g., 19.mkv or 19-00.mkv)
        output_file = str(cam_dir / f"{now:%Y-%m-%d %H-00}.mkv")

        # Stable raw stream copy command via UDP
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-rtsp_transport", "udp",
            "-buffer_size", "10240000",             # 10MB UDP socket buffer
            "-i", self.rtsp_url,
            "-c", "copy",                           # Direct stream copy without decoding
            "-y",                                   # Overwrite if file exists
            output_file,
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[Recorder:{self.camera_id}] Raw recording started: {output_file}", flush=True)
            return True
        except Exception as exc:
            print(f"[Recorder:{self.camera_id}] Failed to start FFmpeg: {exc}", flush=True)
            self.process = None
            self._current_file_hour = None
            return False

    def stop_recording(self) -> None:
        """Guarantees process tree termination across platforms."""
        if self.process is None:
            return

        try:
            pid = self.process.pid

            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                self.process.kill()

            self.process.wait(timeout=2)
            print(f"[Recorder:{self.camera_id}] Recording stopped.", flush=True)
        except Exception as exc:
            print(f"[Recorder:{self.camera_id}] Error stopping process: {exc}", flush=True)
        finally:
            self.process = None
            self._current_file_hour = None

    def is_recording(self) -> bool:
        """Checks if FFmpeg process is currently running without side-effects."""
        if self.process is None:
            return False

        return_code = self.process.poll()
        if return_code is not None:
            print(f"[Recorder:{self.camera_id}] FFmpeg exited with code {return_code}", flush=True)
            self.process = None
            self._current_file_hour = None
            return False

        return True

    def rotate_if_new_hour(self) -> None:
        """Rotates the recording file when transitioning to a new hour."""
        if not self.is_recording():
            return

        now_hour = datetime.now().hour
        if self._current_file_hour is not None and now_hour != self._current_file_hour:
            print(f"[Recorder:{self.camera_id}] Rotating file for new hour ({now_hour}:00)...", flush=True)
            self.stop_recording()
            self.start_recording()