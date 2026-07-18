import cv2
import time
import csv
from datetime import datetime
from collections import defaultdict
from threading import Thread
from ultralytics import YOLO

# Global configurations
MODEL_PATH = "yolo11n.pt"
THRESHOLD = 0.65
PERSISTENCE_FRAMES = 3  # Frame persistence requirement for temporal smoothing

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                self.stop()

    def read(self):
        return self.frame

    def stop(self):
        if not self.stopped:
            self.stopped = True
            self.stream.release()

def run_detector(source_path=0, output_path="output_smoothed.mp4"):
    print(f"Loading model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    is_live = isinstance(source_path, int) or source_path == 0

    if is_live:
        vs = WebcamVideoStream(src=source_path).start()
        time.sleep(1.0)
        frame_init = vs.read()
        if frame_init is None:
            print("Error: Could not open live video source.")
            return
        h0, w0 = frame_init.shape[:2]
        fps = 30
    else:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source_path}")
            return
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FRAME_FPS) or 30

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w0, h0))

    # Tracking & Analytics States
    track_history = defaultdict(int)
    track_timestamps = {}  # Stores custom structural timestamps: {track_id: entry_datetime_object}
    
    # Initialize CSV Logs and write headers
    occ_csv = open("occupancy_log.csv", mode="w", newline="")
    dwell_csv = open("dwell_time_log.csv", mode="w", newline="")
    
    occ_writer = csv.writer(occ_csv)
    dwell_writer = csv.writer(dwell_csv)
    
    occ_writer.writerow(["Timestamp", "Peak_Occupancy"])
    dwell_writer.writerow(["Track_ID", "Entry_Time", "Exit_Time", "Dwell_Duration_Sec"])

    print("Processing stream and logging telemetry. Press 'q' to exit early...")
    
    try:
        while True:
            if is_live:
                frame_proc = vs.read()
            else:
                ret, frame_proc = cap.read()
                if not ret:
                    break

            if frame_proc is None:
                continue

            current_frame_tracked_ids = set()
            stable_person_count = 0
            timestamp_now = datetime.now()

            # Run YOLO11 Tracking Layer
            results = model.track(frame_proc, persist=True, verbose=False)

            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                cls = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, conf, class_idx in zip(boxes, ids, confs, cls):
                    if class_idx != 0 or conf < THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    area = box_width * box_height

                    # Spatial Filters
                    if box_width < 80 or box_height < 120:
                        continue
                    if area < 0.03 * w0 * h0:
                        continue

                    # Persistent Temporal Tracking
                    current_frame_tracked_ids.add(track_id)
                    track_history[track_id] += 1

                    # Track entry metrics upon hitting baseline target stability
                    if track_history[track_id] == PERSISTENCE_FRAMES:
                        track_timestamps[track_id] = datetime.now()

                    # Rendering Layer
                    if track_history[track_id] >= PERSISTENCE_FRAMES:
                        stable_person_count += 1
                        display_text = f"Person: {conf * 100:.0f}%"
                        
                        cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame_proc, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
                        cv2.putText(frame_proc, display_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Log live frame occupancy to time-series data log
            occ_writer.writerow([timestamp_now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], stable_person_count])

            # Drop dead tracks and calculate dwell times
            for old_id in list(track_history.keys()):
                if old_id not in current_frame_tracked_ids:
                    # If it was officially validated as a stable person, calculate dwell duration
                    if old_id in track_timestamps:
                        entry_time = track_timestamps[old_id]
                        exit_time = datetime.now()
                        duration = (exit_time - entry_time).total_seconds()
                        
                        # Export data record directly to dwell time sheet
                        dwell_writer.writerow([
                            old_id, 
                            entry_time.strftime("%H:%M:%S"), 
                            exit_time.strftime("%H:%M:%S"), 
                            f"{duration:.2f}"
                        ])
                        del track_timestamps[old_id]
                        
                    del track_history[old_id]

            # Write frame pipeline and preview window
            out.write(frame_proc)
            cv2.imshow("Smoothed YOLO11 Person Detection", frame_proc)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Close out log handles cleanly
        occ_csv.close()
        dwell_csv.close()
        
        if is_live:
            vs.stop()
        else:
            cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Saved logs and video.")

if __name__ == "__main__":
    run_detector(source_path=0, output_path="webcam_output.mp4")