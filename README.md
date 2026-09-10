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

# HREF YOLO model:
https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

# Download free alarm sound in format .wav
....search sound.wav

3. Configuration & Run
Step 1: Create Environment File (.env)
Create a .env file in the project root with your data:

# 📹 IP Cameras
CAMERAS='{
  "cam_1_front": {
    "rtsp": "rtsp://<USER>:<PAASWORD>@<HOST>:<PORT>/stream",
    "location": "Front yard",
    "model": "****",
    "is_recording": true
  }
}'

# 🎬 Recording 
RECORDING_START_HOUR = 4
RECORDING_END_HOUR = 23

# 🧠 YOLO AI-model
YOLO_MODEL_PATH = "detection_models/yolov8n.pt"
YOLO_DEVICE = "cpu"
YOLO_PERSON_CLASS_IDS = [0]
YOLO_CONFIDENCE = 0.42
YOLO_FRAME_SIZE = 640
YOLO_FPS = 12

# 🚨 Events Detection
EVENTS_DIR = "events"
EVENT_COOLDOWN = 60

# 🤖 Telgram Bot
BOT_TOKEN = "****"
CHAT_IDS = ["****"]

# 🔊 Sounds Directory
SOUNDS_DIR = "sounds"

# 🧹 Cleanup events folders
CLEANUP_DAYS = 3
CLEANUP_CHECK_INTERVAL_SECONDS = 3600

# 📝 Logging
LOGS_DIR = "logs"
LOG_RETENTION_DAYS = 14
LOG_MAX_BYTES = 10485760
LOG_RETENTION_COUNT = 10


# Create directory for recordings and snapshots
mkdir -p events

# Run VMS Core
uv run python main.py