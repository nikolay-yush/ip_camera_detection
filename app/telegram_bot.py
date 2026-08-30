from pathlib import Path
from datetime import datetime, time
from time import sleep

# from app.settings import settings

import requests

"""
def send_message(text: str) -> bool:
    CHAT_IDS = [
        settings.CHAT_ID,
        "772220383",
    ]
    
    for chat_id in CHAT_IDS:
        try:
            requests.post(
                f"https://telegram.org{settings.BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                },
                timeout=10,
            )
        except Exception as e:
            print(f"Telegram error for {chat_id}: {e}")
            
        time.sleep(0.1) # type: ignore
        
    return True


def send_photo(photo_path: Path, caption: str = "") -> bool:
    CHAT_IDS = [
        settings.CHAT_ID,
        "772220383",
    ]
    
    for chat_id in CHAT_IDS:
        try:
            with open(photo_path, "rb") as photo:
                requests.post(
                    f"https://api.telegram.org/bot{settings.BOT_TOKEN}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": caption,
                    },
                    files={
                        "photo": photo,
                    },
                    timeout=20,
                )
        except Exception as e:
            print(f"Telegram error for {chat_id}: {e}")
            
        time.sleep(0.1) # type: ignore
        
    return True



def send_detection(photo_path: Path):
    now = datetime.now()

    caption = (
        "🚨 Human detected\n\n"
        f"📅 {now:%Y-%m-%d}\n"
        f"🕒 {now:%H:%M:%S}"
    )

    send_photo(photo_path, caption)

"""

def send_m(text: str) -> bool:
    CHAT_IDS = [
        "7451622920",
        "772220383",
    ]
    
    for chat_id in CHAT_IDS:
        try:
            requests.post(
                f"https://api.telegram.org/bot{"8428877325:AAHLJO92u1luwwcGL6aQ2lvpogUx12e0b6g"}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                },
                timeout=10,
            )
        except Exception as e:
            print(f"Telegram error for {chat_id}: {e}")
            
        sleep(0.1) # type: ignore
        
    return True

send_m("Bot started")