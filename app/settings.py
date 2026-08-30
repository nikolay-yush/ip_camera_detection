from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    RTSP_URL: str = ""
    MAX_SCREEN_SHOTS: int = 5
    SHOT_INTERVAL: float = 0.3
    CLEANUP_DAYS: int = 2
    EVENTS_DIR: Path = Path("events")
    BOT_TOKEN: str = ""
    CHAT_IDS: list[str] = []


settings = Settings()