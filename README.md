# Real-Time Edge Computer Vision & Telemetry Pipeline

An optimized, multi-threaded computer vision pipeline utilizing the YOLO11 architecture for highly accurate object tracking, spatial-temporal filtering, and automated analytics logging.

## 🚀 Key Features

- **Multi-Threaded Video Ingestion:** Decouples frame reading from processing loops to maximize frame-rate stability and minimize input latency.
- **Temporal Smoothing:** Implements a strict 3-frame persistence verification algorithm to eliminate false-positive detections and bounding-box flickering.
- **Asynchronous Telemetry:** Exfiltrates real-time time-series data frame-by-frame (`occupancy_log.csv`) and captures precise object lifetime metrics (`dwell_time_log.csv`).

## 📁 System Architecture

- `detector.py`: Main execution layer initializing the multi-threaded capture, YOLO tracking inference, and log handling.
- `occupancy_log.csv`: Continuous time-series spreadsheet tracking live frame occupancy timestamps.
- `dwell_time_log.csv`: Event-driven spreadsheet logging individual Track IDs, precise arrival/departure clocks, and total duration metrics.

## 🛠️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nyalaman2085/realtime-video-telemetry.git
cd realtime-video-telemetry
   ```
