# Multi-Camera VMS + YOLO (OpenVINO CPU Accelerated)
Python-based video management and analytics system optimized for low CPU utilization using **Intel OpenVINO C++ Runtime**, **Shared Memory (IPC)**, and **FFmpeg**.

---

## 1. Prerequisites

Ensure the following system dependencies are installed before getting started:
* **Python**: Version 3.10+ (`uv` package manager recommended).
* **FFmpeg**: Installed and configured in the system `PATH` (used for background RTSP recording).
* **Build Tools**: Standard C++ build environment (e.g., `build-essential` on Linux).

---

## 2. Installation & Model Export

Run the following commands in your terminal:

```bash
# 1. Clone the repository and navigate into the project directory
git clone <REPOSITORY_URL>
cd <PROJECT_DIRECTORY>

# 2. Install dependencies using uv
uv sync

# 3. Export the PyTorch model (.pt) to OpenVINO C++ IR format (640x640)
uv run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino')"