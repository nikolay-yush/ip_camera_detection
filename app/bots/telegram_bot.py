import logging
import time
from datetime import datetime
from pathlib import Path

import requests

from app.core.settings import settings

logger = logging.getLogger(__name__)


def send_photo(
    photo_path: Path,
    caption: str = "",
) -> bool:
    """Dispatches a photo with caption to all Telegram chat IDs specified in settings.CHAT_IDS."""
    if not photo_path.is_file():
        logger.error("[Telegram] Photo file not found: %s", photo_path)
        return False

    all_success = True

    for chat_id in settings.CHAT_IDS:
        try:
            with open(photo_path, "rb") as photo:
                response = requests.post(
                    f"https://api.telegram.org/bot{settings.BOT_TOKEN}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": caption,
                        "parse_mode": "Markdown",
                    },
                    files={
                        "photo": photo,
                    },
                    timeout=20,
                )

            response.raise_for_status()
            logger.info("[Telegram] Successfully sent photo to chat_id=%s", chat_id)

        except Exception as exc:
            all_success = False
            logger.error("[Telegram] Failed to send photo to chat_id=%s: %s", chat_id, exc)

        time.sleep(0.5)

    return all_success


def send_detection(
    photo_path: Path,
    camera_id: str = "Unknown",
) -> bool:
    """Formats detection alert caption and triggers photo dispatch."""
    now = datetime.now()

    caption = (
        "🚨 **Human detected**\n\n"
        f"📹 **Camera:** {camera_id}\n"
        f"📅 **Date:** {now:%Y-%m-%d}\n"
        f"🕒 **Time:** {now:%H:%M:%S}"
    )

    return send_photo(
        photo_path=photo_path,
        caption=caption,
    )