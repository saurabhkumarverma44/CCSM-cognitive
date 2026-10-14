# agents/perception_agent.py
# Phase 5.1 - Perception Agent with integrated Memory + EventLogger
import logging
import cv2
from typing import List, Tuple

from config.settings import CONFIG
from core.camera import Camera
from core.event_logger import EventLogger
from core.memory_manager import MemoryManager
from core.memory_models import TrackedObjectMemory
from core.tracker_wrapper import TrackerWrapper

logger = logging.getLogger(__name__)


class PerceptionAgent:
    """
    Orchestrates the perception pipeline:
        Camera -> YOLO12l (GPU) -> ByteTrack -> MemoryManager -> EventLogger

    Public methods
    --------------
    open()                      Open the camera.
    close()                     Release camera and destroy windows.
    process_frame()             Backward-compatible (tracked, annotated_frame).
    process_frame_with_memory() Full result including memory snapshot.
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

        # Wire state-change + purge callbacks
        self.memory.register_state_callback(self.event_logger.on_state_change)
        self.memory.register_purge_callback(self.event_logger.on_purge)

        self._frame_index: int = 0

        logger.info("PerceptionAgent | model=%s tracker=%s device=%s",
                    model_name, tracker_config, device)

    def open(self) -> None:
        """Open the camera device."""
        self.camera.open()

    def close(self) -> None:
        """Release camera and destroy all OpenCV windows."""
        self.camera.release()
        cv2.destroyAllWindows()

    def process_frame(self) -> Tuple[List[dict], "np.ndarray"]:
        """Backward-compatible wrapper. Returns (tracked_objects, annotated_frame)."""
        tracked, _, annotated, _ = self.process_frame_with_memory()
        return tracked, annotated

    def process_frame_with_memory(
        self,
    ) -> Tuple[List[dict], List[TrackedObjectMemory], "np.ndarray", "np.ndarray"]:
        """
        Full perception step for one frame.

        Returns
        -------
        tracked_objects  : list of dicts from TrackerWrapper
        memory_snapshot  : list of TrackedObjectMemory from MemoryManager
        annotated_frame  : YOLO-annotated frame (results.plot())
        raw_frame        : unmodified camera frame (for additional rendering)
        """
        raw_frame = self.camera.read_frame()
        tracked_objects, results = self.tracker.track_frame(raw_frame)
        self.memory.update(tracked_objects, self._frame_index)

        # Tick event logger for dashboard / debug output
        memory_snapshot = self.memory.all_objects()
        self.event_logger.tick(memory_snapshot, self._frame_index)

        self._frame_index += 1
        annotated = results.plot()
        return tracked_objects, memory_snapshot, annotated, raw_frame
