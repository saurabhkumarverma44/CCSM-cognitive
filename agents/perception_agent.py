
# agents/perception_agent.py
"""CCSM perception pipeline with memory, event logging, and Phase 8 event capture."""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from config.settings import CONFIG
from core.camera import Camera
from core.event_logger import EventLogger
from core.event_manager import EventManager
from core.memory_manager import MemoryManager
from core.memory_models import TrackedObjectMemory
from core.tracker_wrapper import TrackerWrapper

logger = logging.getLogger(__name__)


class PerceptionAgent:
    """
    Orchestrates the perception pipeline:

        Camera
          -> YOLO detection
          -> ByteTrack
          -> MemoryManager
          -> EventLogger + EventManager

    EventLogger handles terminal output and dashboard updates.
    EventManager records structured lifecycle events for Phase 8.

    LLM reasoning is deliberately not executed inside this class's
    frame-processing path, so API latency cannot block camera processing.
    """

    def __init__(
        self,
        camera_index: int = CONFIG.perception.camera_index,
        model_name: str = CONFIG.perception.model_name,
        tracker_config: str = CONFIG.perception.tracker_config,
        device: str = CONFIG.perception.device,
    ) -> None:
        self.camera = Camera(device_index=camera_index)

        self.tracker = TrackerWrapper(
            model_name=model_name,
            tracker_config=tracker_config,
            device=device,
        )

        self.memory = MemoryManager()
        self.event_logger = EventLogger()
        self.event_manager = EventManager()

        # Register each callback exactly once.
        # MemoryManager remains the single source of lifecycle transitions.
        self.memory.register_state_callback(
            self.event_logger.on_state_change
        )
        self.memory.register_purge_callback(
            self.event_logger.on_purge
        )

        # Phase 8: structured event capture.
        self.memory.register_state_callback(
            self.event_manager.on_state_change
        )
        self.memory.register_purge_callback(
            self.event_manager.on_purge
        )

        self._frame_index: int = 0
        self._opened: bool = False

        logger.info(
            "PerceptionAgent initialized | model=%s tracker=%s device=%s",
            model_name,
            tracker_config,
            device,
        )

    @property
    def frame_index(self) -> int:
        """Return the index of the next frame to process."""
        return self._frame_index

    def open(self) -> None:
        """Open the configured camera."""
        if self._opened:
            logger.debug("Camera is already open.")
            return

        self.camera.open()
        self._opened = True
        logger.info("Perception camera opened.")

    def close(self) -> None:
        """Release the camera and close OpenCV windows."""
        try:
            if self._opened:
                self.camera.release()
                self._opened = False
        finally:
            cv2.destroyAllWindows()
            logger.info("PerceptionAgent closed.")

    def process_frame(
        self,
    ) -> Tuple[List[dict], np.ndarray]:
        """
        Backward-compatible processing method.

        Returns:
            A tuple of tracked-object dictionaries and the annotated frame.
        """
        tracked_objects, _, annotated_frame, _ = (
            self.process_frame_with_memory()
        )
        return tracked_objects, annotated_frame

    def process_frame_with_memory(
        self,
    ) -> Tuple[
        List[dict],
        List[TrackedObjectMemory],
        np.ndarray,
        np.ndarray,
    ]:
        """
        Process one camera frame and update temporal memory.

        Returns:
            tracked_objects:
                Detection/tracking dictionaries from TrackerWrapper.

            memory_snapshot:
                Current tracked-object memory snapshot.

            annotated_frame:
                Frame annotated by the detection model.

            raw_frame:
                Original frame from the camera.

        EventLogger and EventManager are notified through MemoryManager
        callbacks when meaningful lifecycle transitions occur.
        """
        if not self._opened:
            raise RuntimeError(
                "Camera is not open. Call agent.open() before processing frames."
            )

        raw_frame = self.camera.read_frame()

        tracked_objects, results = self.tracker.track_frame(raw_frame)

        current_frame = self._frame_index

        # Update memory first. Lifecycle callbacks fire during this call.
        self.memory.update(tracked_objects, current_frame)

        # Take a snapshot after memory has been updated.
        memory_snapshot = self.memory.all_objects()

        # Preserve existing terminal debug and dashboard behavior.
        self.event_logger.tick(memory_snapshot, current_frame)

        # Advance the frame counter only after the frame is processed.
        self._frame_index += 1

        annotated_frame = results.plot()

        return (
            tracked_objects,
            memory_snapshot,
            annotated_frame,
            raw_frame,
        )
