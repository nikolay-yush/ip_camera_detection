import json
from pathlib import Path
from typing import Any, Dict

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration management using environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 📹 IP Cameras  
    CAMERAS: Dict[str, Dict[str, Any]] = {}

    # 🎬 Recording 
    RECORDING_START_HOUR: int = 4
    RECORDING_END_HOUR: int = 23

    # 🧠 YOLO Detection 
    YOLO_MODEL_PATH: Path = Path("detection_models/yolov8n.pt")
    YOLO_DEVICE: str = "cpu"
    YOLO_PERSON_CLASS_IDS: list[int] = [0]
    YOLO_CONFIDENCE: float = 0.40
    YOLO_FRAME_SIZE: int = 640
    YOLO_FPS: int = 12

    # 🚨 Events Detection
    EVENTS_DIR: Path = Path("events")
    EVENT_COOLDOWN: int = 60

    # 🤖 Telegram Bot 
    BOT_TOKEN: str = ""
    CHAT_IDS: list[str] = []

    # 🔊 Sounds Directory
    SOUNDS_DIR: Path = Path("sounds")

    # 🧹 Cleanup events folders
    CLEANUP_DAYS: int = 2
    CLEANUP_CHECK_INTERVAL_SECONDS: int = 3600

    # 📝 Logging
    LOGS_DIR: Path = Path("logs")
    LOG_RETENTION_DAYS: int = 14
    LOG_MAX_BYTES: int = 10485760
    LOG_RETENTION_COUNT: int = 10  # Maximum number of log files to keep

    @property
    def YOLO_FRAME_INTERVAL(self) -> float:
        """Calculate delay between processed frames based on target FPS."""
        return 1.0 / self.YOLO_FPS if self.YOLO_FPS > 0 else 0.0

    @field_validator("CAMERAS", mode="before")
    @classmethod
    def parse_cameras_json(cls, value: Any) -> Dict[str, Dict[str, Any]]:
        """Parse CAMERAS JSON string from .env into a Python dictionary and populate window_name."""
        cameras_data: Dict[str, Dict[str, Any]] = {}

        if isinstance(value, str):
            try:
                cameras_data = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON string in CAMERAS environment variable: {exc}"
                )
        elif isinstance(value, dict):
            cameras_data = value

        for cam_id, cam_info in cameras_data.items():
            model = cam_info.get("model", "Unknown")
            location = cam_info.get("location", "Unknown")

            # window_name: "YCC365 - Front yard (cam_1_front)"
            cam_info["window_name"] = f"{model} - {location} ({cam_id})"

        return cameras_data


settings = Settings()