# Smoothed Person Detection Pipeline using YOLO11 & OpenCV

An optimized, production-ready computer vision pipeline designed to solve a common machine learning engineering challenge: erratic bounding box flicker and false-positive ambient noise in real-time streams.

## 🚀 Features & Engineering Solutions

- **State-of-the-Art Core:** Upgraded from legacy MobileNet architectures to the ultra-fast, high-accuracy **Ultralytics YOLO11n** framework.
- **Temporal Smoothing Filter:** Leverages deep byte-tracking identification numbers combined with a custom frame persistence validation mechanism. Bounding boxes are only rendered once an object successfully persists across **3 consecutive frames**, removing high-frequency background flicker.
- **Multi-Stage Spatial Filtering:** Implements rigid minimum dimension bounds (width < 80px, height < 120px) alongside an adaptive area mask requiring target frames to consume at least **3% of the total screen space**, successfully neutralizing distant background noise.
- **Gaussian Noise Mitigation:** Integrates localized pre-processing frame blurs to stabilize variance factors prior to model forwarding layers.

## 🛠️ Installation & Setup

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/smoothed-yolo11-person-detector.git](https://github.com/YOUR_USERNAME/smoothed-yolo11-person-detector.git)
   cd smoothed-yolo11-person-detector
   ```
