import os
from pathlib import Path
import sys
from pydantic_settings import BaseSettings, SettingsConfigDict

# 🚀 Stable UDP transferring
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|"
    "buffer_size;10240000|"
    "max_delay;500000|"
    "stimeout;5000000|"
    "timeout;5000000"
)

# 🔇 Disable sys spam Qt/FFmpeg/OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"


# 🔇 Suppress low-level C-library stderr spam (FFmpeg HEVC frame drops)
def _suppress_c_stderr() -> None:
    try:
        stderr_fd = sys.stderr.fileno()
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, stderr_fd)
        os.close(null_fd)
    except Exception:
        pass

# Activate C-level stderr suppression
_suppress_c_stderr()


class Settings(BaseSettings):
    RTSP_URL: str = ""
    MAX_SCREEN_SHOTS: int = 5
    COOLDOWN_TIME: float = 10.0
    SHOT_INTERVAL: float = 0.3
    CLEANUP_DAYS: int = 2
    EVENTS_DIR: Path = Path("events")
    BOT_TOKEN = ""
    CHAT_ID = ""

    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()