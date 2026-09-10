import logging
import os
import signal
import subprocess
from datetime import datetime
from app.core.settings import settings

logger = logging.getLogger("ffmpeg_recorder")


class FFmpegRecorder:

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.process: subprocess.Popen | None = None
        self._current_file_hour: int | None = None

    def start_recording(self) -> bool:
        """Starts raw RTSP stream dumping to MKV file organized by date and camera folders."""
        if self.is_recording():
            return False

        now = datetime.now()
        self._current_file_hour = now.hour

        # Target folder structure: EVENTS_DIR / YYYY-MM-DD / camera_id
        date_str = now.strftime("%Y-%m-%d")
        cam_dir = settings.EVENTS_DIR / date_str / self.camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)

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
            logger.info("[Recorder:%s] Raw recording started: %s", self.camera_id, output_file)
            return True
        except Exception as exc:
            logger.error("[Recorder:%s] Failed to start FFmpeg process: %s", self.camera_id, exc, exc_info=True)
            self.process = None
            self._current_file_hour = None
            return False

    def stop_recording(self) -> None:
        """Guarantees clean process termination and flushes video containers."""
        if self.process is None:
            return

        logger.info("[Recorder:%s] Stopping FFmpeg process (PID %d)...", self.camera_id, self.process.pid)

        try:
            pid = self.process.pid

            if os.name == "nt":
                # Windows taskkill allows processes to close clean with /T
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                # POSIX: Send SIGINT first so FFmpeg can write file trailers cleanly
                self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    logger.warning("[Recorder:%s] FFmpeg did not stop on SIGINT, sending SIGKILL...", self.camera_id)
                    self.process.kill()
                    self.process.wait(timeout=1.0)

            logger.info("[Recorder:%s] Recording stopped cleanly.", self.camera_id)
        except Exception as exc:
            logger.error("[Recorder:%s] Error stopping FFmpeg process: %s", self.camera_id, exc)
        finally:
            self.process = None
            self._current_file_hour = None

    def is_recording(self) -> bool:
        """Checks if FFmpeg process is currently running without side-effects."""
        if self.process is None:
            return False

        return_code = self.process.poll()
        if return_code is not None:
            logger.warning("[Recorder:%s] FFmpeg process exited unexpectedly with code %d", self.camera_id, return_code)
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
            logger.info("[Recorder:%s] Rotating recording file for new hour (%02d:00)...", self.camera_id, now_hour)
            self.stop_recording()
            self.start_recording()