import sys
import subprocess
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

        # Create output directory for the camera (FFmpeg formats date and hour in the filename)
        out_dir = settings.EVENTS_DIR / self.camera_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Output path template using strftime formatting: .../camera_id/2026-03-30_14-00.mkv
        output_template = str(out_dir / "%Y-%m-%d_%H-00.mkv")

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
            "-strftime", "1",             # Enable strftime patterns (%Y-%m-%d_%H-00) in output path
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