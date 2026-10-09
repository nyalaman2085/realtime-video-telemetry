"""Person tracking and telemetry with bounded recovery and deterministic cleanup."""

import csv
import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.65
PERSISTENCE_FRAMES = 3
MAX_MISSING_FRAMES = 8
MIN_BOX_WIDTH = 80
MIN_BOX_HEIGHT = 120
MIN_BOX_AREA_RATIO = 0.03


class WebcamVideoStream:
    """Read webcam frames on a background thread and expose safe snapshots."""

    def __init__(self, source=0):
        self.stream = cv2.VideoCapture(source)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.lock = Lock()
        self.frame = None
        self.stopped = not self.stream.isOpened()
        self.thread = None

    def start(self):
        if not self.stopped:
            self.thread = Thread(target=self._update, daemon=True)
            self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed or frame is None:
                self.stopped = True
                break
            with self.lock:
                self.frame = frame
        self.stream.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def is_stopped(self):
        return self.stopped

    def stop(self):
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.stream.isOpened():
            self.stream.release()


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def run_detector(source_path=0, output_path="webcam_output.mp4"):
    """Run person tracking and write annotated video plus two CSV telemetry logs."""
    model = YOLO(MODEL_PATH)
    is_live = isinstance(source_path, int)

    webcam = None
    capture = None
    writer = None
    occupancy_file = None
    dwell_file = None

    track_seen = {}
    track_started = {}
    track_last_seen = {}
    track_entry_clock = {}
    track_missing = {}
    frame_index = 0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    occupancy_path = output.with_name("occupancy_log.csv")
    dwell_path = output.with_name("dwell_time_log.csv")

    try:
        if is_live:
            webcam = WebcamVideoStream(source_path).start()
            deadline = time.monotonic() + 3.0
            first_frame = None
            while time.monotonic() < deadline and first_frame is None:
                first_frame = webcam.read()
                if first_frame is None:
                    time.sleep(0.05)
            if first_frame is None:
                raise RuntimeError("Could not read a frame from the webcam.")
            frame_height, frame_width = first_frame.shape[:2]
            fps = 30.0
        else:
            capture = cv2.VideoCapture(source_path)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video source: {source_path}")
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            if frame_width <= 0 or frame_height <= 0:
                raise RuntimeError("The source video has invalid dimensions.")

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output}")

        occupancy_file = occupancy_path.open("w", newline="", encoding="utf-8")
        dwell_file = dwell_path.open("w", newline="", encoding="utf-8")
        occupancy_writer = csv.writer(occupancy_file)
        dwell_writer = csv.writer(dwell_file)
        occupancy_writer.writerow(["timestamp", "stable_person_count"])
        dwell_writer.writerow(
            ["track_id", "entry_timestamp", "exit_timestamp", "dwell_duration_seconds"]
        )

        print(f"Model: {MODEL_PATH}")
        print(f"Output video: {output}")
        print("Press q in the preview window to stop.")

        while True:
            if is_live:
                frame = webcam.read()
                if frame is None:
                    if webcam.is_stopped():
                        break
                    time.sleep(0.01)
                    continue
            else:
                grabbed, frame = capture.read()
                if not grabbed or frame is None:
                    break

            frame_index += 1
            now_monotonic = time.monotonic()
            now_timestamp = _timestamp()
            height, width = frame.shape[:2]
            current_ids = set()

            results = model.track(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=[0],
                persist=True,
                verbose=False,
            )
            boxes = results[0].boxes if results else None

            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                for coordinates_box, track_id, confidence in zip(coordinates, ids, confidences):
                    x1, y1, x2, y2 = map(int, coordinates_box)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    area = max(0, box_width) * max(0, box_height)

                    if box_width < MIN_BOX_WIDTH or box_height < MIN_BOX_HEIGHT:
                        continue
                    if area < MIN_BOX_AREA_RATIO * width * height:
                        continue

                    current_ids.add(track_id)
                    track_seen[track_id] = track_seen.get(track_id, 0) + 1
                    track_missing[track_id] = 0
                    track_last_seen[track_id] = now_monotonic

                    if track_seen[track_id] == PERSISTENCE_FRAMES:
                        track_started[track_id] = now_monotonic
                        track_entry_clock[track_id] = now_timestamp

                    if track_seen[track_id] >= PERSISTENCE_FRAMES:
                        label = f"Person {confidence:.0%} | ID {track_id}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                        label_y = max(20, y1 - 8)
                        cv2.putText(
                            frame, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                        )
                        cv2.putText(
                            frame, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1,
                        )

            # Allow brief detector/tracker misses before closing a track.
            for track_id in list(track_seen):
                if track_id not in current_ids:
                    track_missing[track_id] = track_missing.get(track_id, 0) + 1
                    if track_missing[track_id] > MAX_MISSING_FRAMES:
                        if track_id in track_started:
                            duration = max(0.0, now_monotonic - track_started[track_id])
                            dwell_writer.writerow([
                                track_id,
                                track_entry_clock[track_id],
                                now_timestamp,
                                f"{duration:.2f}",
                            ])
                        track_seen.pop(track_id, None)
                        track_started.pop(track_id, None)
                        track_last_seen.pop(track_id, None)
                        track_entry_clock.pop(track_id, None)
                        track_missing.pop(track_id, None)

            stable_count = sum(
                1 for track_id, count in track_seen.items()
                if count >= PERSISTENCE_FRAMES and track_missing.get(track_id, 0) == 0
            )
            occupancy_writer.writerow([now_timestamp, stable_count])
            writer.write(frame)
            cv2.imshow("Real-Time Video Telemetry", frame)

            if frame_index % 30 == 0:
                occupancy_file.flush()
                dwell_file.flush()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Close dwell intervals for tracks still present when the user stops.
        end_time = time.monotonic()
        end_timestamp = _timestamp()
        for track_id, start_time in track_started.items():
            duration = max(0.0, end_time - start_time)
            dwell_writer.writerow([
                track_id,
                track_entry_clock[track_id],
                end_timestamp,
                f"{duration:.2f}",
            ])

        print(f"Saved annotated video: {output}")
        print(f"Saved occupancy log: {occupancy_path}")
        print(f"Saved dwell-time log: {dwell_path}")

    finally:
        if occupancy_file:
            occupancy_file.close()
        if dwell_file:
            dwell_file.close()
        if writer:
            writer.release()
        if webcam:
            webcam.stop()
        if capture:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detector()
