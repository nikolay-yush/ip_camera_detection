from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    RTSP_URL: str = ""
    
    CLEANUP_DAYS: int = 2

    BOT_TOKEN: str = ""
    CHAT_IDS: list[str] = []

    YOLO_MODEL_PATH: Path = Path("yolov8n.pt")
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


settings = Settings()