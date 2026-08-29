import time
import sys
import subprocess
import threading
import cv2
from pathlib import Path
from datetime import datetime

from app.telegram_bot import send_detection
from app.settings import settings


def _play_sound() -> None:
    """Plays a system alert sound cross-platform (Windows, Linux, macOS)."""
    try:
        if sys.platform.startswith("win"):
            import winsound
            # Plays standard Windows system error/alert sound
            winsound.Beep(1000, 300) # type: ignore
        elif sys.platform.startswith("darwin"):
            # macOS system sound
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
        else:
            # Linux system sound
            subprocess.run(
                ["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
    except Exception:
        # Fallback terminal bell for any platform
        print("\a", end="", flush=True)


def _show_notification(title: str, message: str) -> None:
    """Displays a native OS desktop notification."""
    try:
        if sys.platform.startswith("win"):
            try:
                from plyer import notification
                notification.notify(
                    title=title,
                    message=message,
                    app_name="YOLO Detection",
                    timeout=5
                ) # type: ignore
                
            except ImportError:
                # Built-in PowerShell Toast notification fallback for Windows
                ps_script = f"""
                [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
                $template = "<toast><visual><binding template='ToastText02'><text id='1'>{title}</text><text id='2'>{message}</text></binding></visual></toast>"
                $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
                $xml.LoadXml($template)
                $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
                [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("YOLO Detection").Show($toast)
                """
                subprocess.run(["powershell", "-Command", ps_script], capture_output=True)

        elif sys.platform.startswith("darwin"):
            # macOS AppleScript notification
            subprocess.run([
                "osascript", "-e",
                f'display notification "{message}" with title "{title}" sound name "default"'
            ])

        else:
            # Linux notify-send
            subprocess.run([
                "notify-send",
                "-u", "critical",
                "-i", "dialog-warning",
                title,
                message
            ], check=False)

    except Exception as e:
        print(f"[Alert Error] Failed to send notification: {e}")


def _trigger_system_alert() -> None:
    """Triggers sound and desktop notification asynchronously."""
    _play_sound()
    _show_notification("🚨 PERSON DETECTED!", "Activity detected on camera stream!")


class EventShotManager:
    def __init__(self):
        self.event_active: bool = False
        self.shot_count: int = 0
        self.last_shot_time: float = 0.0
        self.event_folder: Path | None = None
        self.notification_sent: bool = False

    def process_event(self, detected: bool, frame) -> bool:
        """
        Handles detection lifecycle: starts/stops events, saves snapshots, and fires alerts.
        Returns True when an event has just ended (triggering cooldown in main loop).
        """
        now = time.time()
        event_just_ended = False

        # Event start
        if detected and not self.event_active:
            self.event_active = True
            self.shot_count = 0
            self.notification_sent = False

            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H-%M-%S")

            self.event_folder = (
                settings.EVENTS_DIR
                / date_str
                / f"event_{time_str}"
            )
            self.event_folder.mkdir(parents=True, exist_ok=True)
            print("EVENT START")

            # Trigger non-blocking system alert thread
            threading.Thread(target=_trigger_system_alert, daemon=True).start()

        # Event end
        if not detected and self.event_active:
            self.event_active = False
            self.event_folder = None
            event_just_ended = True
            print("EVENT END - cooldown started")

        # Save snapshots at set intervals
        if (
            self.event_active
            and self.shot_count < settings.MAX_SCREEN_SHOTS
            and self.event_folder is not None
        ):
            if now - self.last_shot_time > settings.SHOT_INTERVAL:
                filename = (
                    self.event_folder
                    / f"shot_{int(now)}_{self.shot_count}.jpg"
                )

                cv2.imwrite(str(filename), frame)
                print("Saved:", filename)

                self.shot_count += 1
                self.last_shot_time = now

                # Send first screenshot to Telegram
                if not self.notification_sent:
                    send_detection(filename)
                    self.notification_sent = True

        return event_just_ended