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

    def start_recording(self) -> None:
        """Starts a single FFmpeg process that handles hourly rotation internally."""
        if self.is_recording():
            return

        # 1. Explicitly create today's subfolder to prevent FFmpeg "No such file or directory" error
        now = datetime.now()
        today_dir = settings.EVENTS_DIR / now.strftime("%Y-%m-%d") / self.camera_id
        today_dir.mkdir(parents=True, exist_ok=True)

        # 2. Path template for dynamic folder creation (%Y-%m-%d) and hourly files (%Y-%m-%d_%H-00.mkv)
        output_template = str(
            settings.EVENTS_DIR / "%Y-%m-%d" / self.camera_id / "%Y-%m-%d_%H-00.mkv"
        )

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-timeout", "5000000",        # RTSP socket connection timeout (microseconds)
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c:v", "copy",
            "-an",
            # Enable automatic stream segmentation by FFmpeg:
            "-f", "segment",
            "-segment_time", "3600",      # Split recording every 3600 seconds (1 hour)
            "-reset_timestamps", "1",
            "-strftime", "1",             # Enable strftime patterns in output path
            output_template,
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[Recorder:{self.camera_id}] Process started.", flush=True)
        except Exception as exc:
            print(f"[Recorder:{self.camera_id}] Failed to start FFmpeg: {exc}", flush=True)
            self.process = None

    def stop_recording(self) -> None:
        """Kills the active FFmpeg recording process."""
        if self.process is None:
            return

        try:
            self.process.kill()
            self.process.wait(timeout=2)
            print(f"[Recorder:{self.camera_id}] Process killed.", flush=True)
        except Exception:
            pass
        finally:
            self.process = None

    def is_recording(self) -> bool:
        """Checks if the FFmpeg subprocess is currently running."""
        return self.process is not None and self.process.poll() is None

    def check_health(self) -> None:
        """Periodically checks stream status and restarts FFmpeg only if crashed or disconnected."""
        if not self.is_recording():
            print(f"[Recorder:{self.camera_id}] Stream dead or stopped. Restarting...", flush=True)
            self.stop_recording()  # Cleanup leftover process references if any
            self.start_recording()