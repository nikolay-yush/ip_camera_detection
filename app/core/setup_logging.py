import logging
import sys
from logging.handlers import RotatingFileHandler

from app.core.settings import settings


def setup_logging() -> None:
    """Configure global logging settings with size-based file rotation."""
    log_dir = settings.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Max size: 10 MB per log file, keeps up to LOG_RETENTION_DAYS (or count) old files
    max_bytes = getattr(settings, "LOG_MAX_BYTES", 10 * 1024 * 1024)
    backup_count = getattr(settings, "LOG_RETENTION_COUNT", settings.LOG_RETENTION_DAYS)

    file_handler = RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True,  # Prevent duplicate handlers
    )