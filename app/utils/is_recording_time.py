import time
from app.settings import settings


def is_recording_time() -> bool:
    """
    Checks if the current system hour falls within the allowed recording window.
    Handles both daytime intervals (e.g., 08:00 to 22:00) and overnight schedules
    spanning past midnight (e.g., 22:00 to 06:00).
    """
    hour = time.localtime().tm_hour
    start = settings.RECORDING_START_HOUR
    end = settings.RECORDING_END_HOUR

    # 24/7 continuous recording when start and end hours match
    if start == end:
        return True

    # Regular daytime schedule (e.g., 8 to 22)
    if start < end:
        return start <= hour < end

    # Overnight schedule crossing midnight (e.g., 22 to 6)
    return hour >= start or hour < end