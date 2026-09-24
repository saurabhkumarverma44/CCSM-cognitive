
# core/tracker_wrapper.py

import numpy as np
from ultralytics import YOLO


class TrackerWrapper:
    """
    Wraps Ultralytics YOLO tracking to return a simple Python list
    of tracked objects with persistent IDs.

    YOLO inference is configured to use the NVIDIA GPU when available.
    """

    def __init__(
        self,
        model_name: str = "yolo12l.pt",
        tracker_config: str = "bytetrack.yaml",
        device: str = "cuda:0",
    ):
        # Load YOLO model
        self.model = YOLO(model_name)

        # GPU / CPU device
        self.device = device

        # Choose tracker backend (ByteTrack or BoT-SORT)
        self.tracker_config = tracker_config

        # Persist=True keeps tracker state across frames
        self.persist = True

    def track_frame(self, frame):
        """
        Run tracking on a single frame and return:

        tracked_objects:
        [
            {
                "track_id": int,
                "class_name": str,
                "class_id": int,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
            },
            ...
        ]

        Also returns the complete Ultralytics result object.
        """

        results = self.model.track(
            source=frame,
            tracker=self.tracker_config,
            persist=self.persist,
            device=self.device,
            verbose=False,
        )[0]

        boxes = results.boxes

        # No tracking IDs available
        if boxes.id is None:
            return [], results

        tracked_objects = []

        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.int().cpu().tolist()

        for obj_id, bbox, conf, cls_id in zip(
            ids, xyxy, confs, clses
        ):
            x1, y1, x2, y2 = bbox.tolist()

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            class_name = results.names.get(
                cls_id,
                str(cls_id)
            )

            tracked_objects.append({
                "track_id": int(obj_id),
                "class_name": class_name,
                "class_id": int(cls_id),
                "confidence": float(conf),
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],
                "center": (
                    float(cx),
                    float(cy),
                ),
            })

        return tracked_objects, results

