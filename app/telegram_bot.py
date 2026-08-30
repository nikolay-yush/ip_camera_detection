from pathlib import Path
from datetime import datetime

import requests

from app.settings import settings


def send_photo(
    photo_path: Path,
    caption: str = "",
) -> bool:

    for chat_id in settings.CHAT_IDS:

        try:

            with open(photo_path, "rb") as photo:

                response = requests.post(
                    f"https://api.telegram.org/"
                    f"bot{settings.BOT_TOKEN}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": caption,
                    },
                    files={
                        "photo": photo,
                    },
                    timeout=20,
                )

            response.raise_for_status()

        except Exception as exc:

            print(
                f"[Telegram] "
                f"Failed to send photo "
                f"to {chat_id}: {exc}"
            )

    return True


def send_detection(
    photo_path: Path,
) -> bool:

    now = datetime.now()

    caption = (
        "🚨 Human detected\n\n"
        f"📅 {now:%Y-%m-%d}\n"
        f"🕒 {now:%H:%M:%S}"
    )

    return send_photo(
        photo_path,
        caption,
    )