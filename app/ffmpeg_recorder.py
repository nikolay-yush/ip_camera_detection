# app/recorder.py
import subprocess
import time
from datetime import datetime
from pathlib import Path
from app.settings import settings


class NativeFFmpegRecorder:
    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.process = None

    def _get_output_path(self, now: datetime) -> Path:
            # Structure: EVENTS_DIR / YYYY-MM-DD / camera_id / HH-00.mkv
            event_dir = settings.EVENTS_DIR / now.strftime("%Y-%m-%d") / self.camera_id
            event_dir.mkdir(parents=True, exist_ok=True)
            return event_dir / f"{now.strftime('%H')}-00.mkv"

    def start_recording(self):
        if self.is_recording():
            return

        now = datetime.now()
        video_path = self._get_output_path(now)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c:v", "copy",
            "-an",
            "-f", "matroska",
            "-y",
            str(video_path),
        ]

        try:
            self.process = subprocess.Popen(cmd)
            print(f"[Recorder:{self.camera_id}] Started recording: {video_path}")
        except Exception as exc:
            print(f"[Recorder:{self.camera_id}] Failed to start FFmpeg: {exc}")

    def stop_recording(self):
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
                print(f"[Recorder:{self.camera_id}] Stopped safely.")
            except Exception:
                self.process.kill()
            finally:
                self.process = None

    def is_recording(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def check_hour_rotation(self):
        if self.is_recording():
            self.stop_recording()
            self.start_recording()