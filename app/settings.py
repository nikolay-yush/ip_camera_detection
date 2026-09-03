import json
from pathlib import Path
from typing import Any, Dict
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    CAMERAS: Dict[str, Dict[str, Any]] = {}
    
    CLEANUP_DAYS: int = 2

    BOT_TOKEN: str = ""
    CHAT_IDS: list[str] = []

    YOLO_MODEL_PATH: Path = Path("yolov8n.pt")
    YOLO_DEVICE: str = "cpu"
    YOLO_PERSON_CLASS_IDS: list[int] = [0]
    YOLO_CONFIDENCE: float = 0.5
    YOLO_FRAME_SIZE: int = 640
    YOLO_FPS: int = 8

    EVENTS_DIR: Path = Path("events")
    EVENT_COOLDOWN: int = 60

    RECORDING_START_HOUR: int = 5
    RECORDING_END_HOUR: int = 22

    @property
    def YOLO_FRAME_INTERVAL(self) -> float:
        return 1.0 / self.YOLO_FPS if self.YOLO_FPS > 0 else 0.0

    @field_validator("CAMERAS", mode="before")
    @classmethod
    def parse_cameras_json(cls, value: Any) -> Dict[str, Dict[str, Any]]:
        """Parses the CAMERAS JSON string from .env into a Python dictionary."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON string in CAMERAS environment variable: {exc}"
                )
        return value

settings = Settings()