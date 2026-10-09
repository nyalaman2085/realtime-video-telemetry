"""Threaded OpenCV video capture helper with safe frame snapshots."""

from threading import Lock, Thread

import cv2


class WebcamVideoStream:
    """Read webcam frames in a background thread.

    The returned frame is copied while holding a lock so callers cannot observe
    a partially updated frame. Call stop() when finished.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.lock = Lock()
        self.frame = None
        self.stopped = not self.stream.isOpened()
        self.thread = None

    def start(self):
        if not self.stopped and (self.thread is None or not self.thread.is_alive()):
            self.thread = Thread(target=self.update, daemon=True)
            self.thread.start()
        return self

    def update(self):
        try:
            while not self.stopped:
                grabbed, frame = self.stream.read()
                if not grabbed or frame is None:
                    self.stopped = True
                    break
                with self.lock:
                    self.frame = frame
        finally:
            self.stream.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.stream.isOpened():
            self.stream.release()
