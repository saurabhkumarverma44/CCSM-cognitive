# agents/perception_agent.py
import logging
import cv2
from config.settings import CONFIG
from core.camera import Camera
from core.memory_manager import MemoryManager
from core.tracker_wrapper import TrackerWrapper

logger = logging.getLogger(__name__)

class PerceptionAgent:
    def __init__(
        self,
        camera_index: int = CONFIG.perception.camera_index,
        model_name: str = CONFIG.perception.model_name,
        tracker_config: str = CONFIG.perception.tracker_config,
        device: str = CONFIG.perception.device,
    ):
        self.camera = Camera(device_index=camera_index)
        self.tracker = TrackerWrapper(
            model_name=model_name, tracker_config=tracker_config, device=device)
        self.memory = MemoryManager()
        self._frame_index: int = 0
        logger.info("PerceptionAgent | model=%s tracker=%s device=%s",
                    model_name, tracker_config, device)

    def open(self) -> None:
        self.camera.open()

    def close(self) -> None:
        self.camera.release()
        cv2.destroyAllWindows()

    def process_frame(self):
        tracked, _, frame = self.process_frame_with_memory()
        return tracked, frame

    def process_frame_with_memory(self):
        frame = self.camera.read_frame()
        tracked_objects, results = self.tracker.track_frame(frame)
        self.memory.update(tracked_objects, self._frame_index)
        self._frame_index += 1
        annotated = results.plot()
        return tracked_objects, self.memory.all_objects(), annotated
