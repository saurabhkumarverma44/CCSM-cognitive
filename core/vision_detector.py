# core/vision_detector.py

from ultralytics import YOLO
from pathlib import Path

class VisionDetector:
    """
    Wraps a pretrained YOLO model for simple image and video prediction.
    """
    def __init__(self, model_name: str = "yolov8n.pt"):
        # Load a small, fast YOLOv8 model by default.
        # You can switch to 'yolo11n.pt' later if available.
        self.model = YOLO(model_name)

    def detect_image(self, image_path: str, output_dir: str = "runs/detect_ccsm"):
        """
        Run object detection on a single image and save annotated result.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = self.model.predict(
            source=image_path,
            project=output_dir,
            name="image_test",
            save=True,
            verbose=True,
        )
        return results