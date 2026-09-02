import time

from app.settings import settings


def is_recording_time() -> bool:

    hour = time.localtime().tm_hour

    return settings.RECORDING_START_HOUR <= hour < settings.RECORDING_END_HOUR

