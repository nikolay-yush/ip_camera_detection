import os
import sys
import subprocess
from pathlib import Path

SOUND_FILE = Path("alarm.wav")


def play_sound() -> None:
    """Play custom alarm sound asynchronously across Windows and Linux."""
    try:
        if not SOUND_FILE.exists():
            print(f"[Sound] File not found: {SOUND_FILE}")
            print("\a", end="", flush=True)
            return

        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound( # type: ignore
                str(SOUND_FILE),
                winsound.SND_FILENAME | winsound.SND_ASYNC, # type: ignore
            )

        else:
            players = ["paplay", "aplay", "ffplay", "cvlc"]
            player_bin = None

            for player in players:
                if os.system(f"which {player} > /dev/null 2>&1") == 0:
                    player_bin = player
                    break

            if player_bin:
                cmd = [player_bin, str(SOUND_FILE)]
                if player_bin == "ffplay":
                    cmd.extend(["-nodisp", "-autoexit", "-loglevel", "quiet"])

                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                print("\a", end="", flush=True)

    except Exception as exc:
        print(f"[Sound] Failed to play sound: {exc}")
        print("\a", end="", flush=True)