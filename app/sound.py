import winsound
from pathlib import Path


SOUND_FILE = Path("alarm.wav")


def play_sound() -> None:
    """Play custom alarm sound asynchronously."""

    try:

        if not SOUND_FILE.exists():

            print(
                f"[Sound] File not found: "
                f"{SOUND_FILE}"
            )

            print("\a", end="", flush=True)

            return

        winsound.PlaySound( # type: ignore
            str(SOUND_FILE),
            winsound.SND_FILENAME # type: ignore
            | winsound.SND_ASYNC, # type: ignore
        )

    except Exception as exc:

        print(
            f"[Sound] Failed to play sound: "
            f"{exc}"
        )

        print(
            "\a",
            end="",
            flush=True,
        )