import os
import signal
import sys
import subprocess
from datetime import datetime
from pathlib import Path

from app.settings import settings


class NativeFFmpegRecorder:

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.process: subprocess.Popen | None = None
        self.current_hour: int | None = None

    def _get_output_path(self, now: datetime) -> Path:
        """Constructs target output file path and ensures parent directories exist."""
        event_dir = settings.EVENTS_DIR / now.strftime("%Y-%m-%d") / self.camera_id
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir / f"{now.strftime('%H')}-00.mkv"

    def start_recording(self) -> None:
        """Launches native FFmpeg stream copy process with RTSP connection timeouts."""
        if self.is_recording():
            return

        now = datetime.now()
        self.current_hour = now.hour
        video_path = self._get_output_path(now)

        # Correct universal timeout option for FFmpeg RTSP demuxer (-timeout in microseconds)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-timeout", "5000000",   # 5-second RTSP socket timeout (microseconds)
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c:v", "copy",          # Direct stream copy without CPU re-encoding
            "-an",                   # Mute audio tracks
            "-f", "matroska",
            "-y",
            str(video_path),
        ]

        try:
            # Platform-specific flags for safe graceful signal handling (SIGINT)
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            print(f"[Recorder:{self.camera_id}] Started recording: {video_path}", flush=True)

        except Exception as exc:
            print(f"[Recorder:{self.camera_id}] Failed to start FFmpeg process: {exc}", flush=True)
            self.process = None

    def stop_recording(self) -> None:
        """Safely stops FFmpeg process to guarantee MKV index container finalization."""
        if self.process is None:
            return

        try:
            # Graceful shutdown handling across Windows and Linux/macOS
            if sys.platform == "win32":
                self.process.send_signal(signal.CTRL_C_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)

            self.process.wait(timeout=3)
            print(f"[Recorder:{self.camera_id}] Stopped safely.", flush=True)

        except (subprocess.TimeoutExpired, Exception):
            print(f"[Recorder:{self.camera_id}] Force killing hung FFmpeg process...", flush=True)
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            self.process = None
            self.current_hour = None

    def is_recording(self) -> bool:
        """Checks if FFmpeg subprocess is currently active."""
        return self.process is not None and self.process.poll() is None

    def check_hour_rotation(self) -> None:
        """Rotates recording file when hour boundary changes or revives dropped stream."""
        now = datetime.now()

        # If recording dropped (e.g. RTSP timeout or crash), restart it
        if not self.is_recording():
            self.start_recording()
            return

        # Rotate recording file only when the clock transitions to a new hour
        if self.current_hour is not None and now.hour != self.current_hour:
            print(
                f"[Recorder:{self.camera_id}] Hour changed ({self.current_hour} -> {now.hour}). Rotating file...",
                flush=True,
            )
            self.stop_recording()
            self.start_recording()