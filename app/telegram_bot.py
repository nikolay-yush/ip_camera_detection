from pathlib import Path
from datetime import datetime

from app.settings import settings

import requests


def send_message(text: str) -> bool:
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{settings.BOT_TOKEN}/sendMessage",
            json={
                "chat_id": settings.CHAT_ID,
                "text": text,
            },
            timeout=10,
        )

        return response.ok

    except Exception as e:
        print("Telegram error:", e)
        return False


def send_photo(photo_path: Path, caption: str = "") -> bool:
    try:
        with open(photo_path, "rb") as photo:
            response = requests.post(
                f"https://api.telegram.org/bot{settings.BOT_TOKEN}/sendPhoto",
                data={
                    "chat_id": settings.CHAT_ID,
                    "caption": caption,
                },
                files={
                    "photo": photo,
                },
                timeout=20,
            )

        return response.ok

    except Exception as e:
        print("Telegram error:", e)
        return False


def send_detection(photo_path: Path):
    now = datetime.now()

    caption = (
        "🚨 Human detected\n\n"
        f"📅 {now:%Y-%m-%d}\n"
        f"🕒 {now:%H:%M:%S}"
    )

    send_photo(photo_path, caption)


