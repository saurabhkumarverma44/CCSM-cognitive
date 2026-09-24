# core/camera.py

import cv2

class Camera:
    """
    Simple webcam interface using OpenCV.
    """
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {self.device_index}")

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None