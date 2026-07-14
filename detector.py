import cv2
import sys
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# --- Configuration Parameters ---
MODEL_PATH = "yolo11n.pt"
THRESHOLD = 0.65
PERSISTENCE_FRAMES = 3  # Frame persistence requirement for temporal smoothing

def run_detector(source_path=0, output_path="output_smoothed.mp4"):
    """
    Runs the filtered and temporally smoothed person tracking pipeline.
    source_path: 0 for webcam, or a string path to a video file.
    """
    print(f"Loading model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        sys.exit(f"Error: Could not open video source {source_path}")

    # Extract source properties
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w0, h0))

    # Track frame state for temporal smoothing
    track_history = defaultdict(int)
    frame_count = 0

    print("Processing stream. Press 'q' to exit early...")
    while True:
        ret, frame_proc = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 1. Pre-processing: Apply Gaussian Blur to reduce high-frequency frame noise
        frame_blurred = cv2.GaussianBlur(frame_proc, (5, 5), 0)

        # 2. Inference: Run YOLO11 tracking loop (Built-in ByteTrack handles IDs)
        results = model.track(source=frame_blurred, conf=THRESHOLD, persist=True, verbose=False)
        current_frame_tracked_ids = set()

        if results and results[0].boxes is not None:
            boxes_data = results[0].boxes

            for box in boxes_data:
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_label = model.names[cls_id]

                # Class Filter: Exclusively track people
                if score >= THRESHOLD and raw_label.lower() == "person":
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)

                    box_width = x2 - x1
                    box_height = y2 - y1
                    area = box_width * box_height

                    # Spatial Filters: Ignore detections below absolute limits
                    if box_width < 80 or box_height < 120:
                        continue

                    # Area Filter: Box must occupy at least 3% of total frame real estate
                    if area < 0.03 * w0 * h0:
                        continue

                    # Persistent Temporal Tracking
                    track_id = int(box.id[0]) if box.id is not None else 0
                    current_frame_tracked_ids.add(track_id)
                    track_history[track_id] += 1

                    # Rendering Rule: Draw only if stable for >= 3 consecutive frames
                    if track_history[track_id] >= PERSISTENCE_FRAMES:
                        display_text = f"Person: {score*100:.0f}%"
                        
                        cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame_proc, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
                        cv2.putText(frame_proc, display_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Drop tracks not present in the current frame to prevent memory leaks
        for old_id in list(track_history.keys()):
            if old_id not in current_frame_tracked_ids:
                del track_history[old_id]

        # Write frame and display live preview
        out.write(frame_proc)
        cv2.imshow("Smoothed YOLO11 Person Detection", frame_proc)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Saved to: {output_path}")

if __name__ == "__main__":
    # 0 runs local webcam. Switch to a file path string (e.g., "video.mp4") to process files.
    run_detector(source_path=0)