# Multi-Camera VMS + YOLO (OpenVINO CPU Accelerated)

Python-based video management and analytics system optimized for low CPU utilization using **Intel OpenVINO C++ Runtime**, **Shared Memory (IPC)**, and **FFmpeg**.

---

## 1. Prerequisites

* **Python**: Version 3.10+ (`uv` recommended).
* **FFmpeg**: Installed and available in system `PATH`.
* **Build Tools**: C++ compiler toolchain (`build-essential` on Linux).

---

## 2. Installation & Model Export

```bash
# Clone repository
git clone [https://github.com/nikolay-yush/ip_camera_detection.git](https://github.com/nikolay-yush/ip_camera_detection.git)
cd ip_camera_detection

# Install dependencies
uv sync

# Export PyTorch model to OpenVINO format (640x640)
uv run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino')"

3. Configuration & Run
Step 1: Create Environment File (.env)
Create a .env file in the project root with your data:

# 🌐 RTSP IP-Cameras (JSON format)
CAMERAS='{
  "cam_1_front": {
    "rtsp": "rtsp://user:password@192.168.0.100:554/stream",
    "location": "Front yard",
    "model": "YCC365",
    "is_recording": true
  },
  "cam_2_front": {
    "rtsp": "rtsp://admin:123456@192.168.0.101:554/stream",
    "location": "Front yard 1",
    "model": "YCC365",
    "is_recording": true
  }
}'

# 🤖 Telegram Bot Notifications
BOT_TOKEN="your_telegram_bot_token"
CHAT_IDS=["123456789"]

# 🧠 YOLO Settings
YOLO_MODEL_PATH="yolov8n_openvino_model"
YOLO_DEVICE="cpu"
YOLO_PERSON_CLASS_IDS=[0]
YOLO_CONFIDENCE=0.42
YOLO_FRAME_SIZE=640
YOLO_FPS=8

# 🚨 Event & Storage Settings
EVENTS_DIR="events"
EVENT_COOLDOWN=600
CLEANUP_DAYS=3

# 📹 Recording Hours (24h format)
RECORDING_START_HOUR=2
RECORDING_END_HOUR=23

Step 2: Create Storage Directory & Launch

# Create directory for recordings and snapshots
mkdir -p events

# Run VMS Core
uv run python main.py