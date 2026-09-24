# config/settings.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PerceptionConfig:
    model_name: str = "yolo12l.pt"
    tracker_config: str = "bytetrack.yaml"
    device: str = "cuda:0"
    camera_index: int = 0
    confidence_threshold: float = 0.30

@dataclass(frozen=True)
class MemoryConfig:
    occlusion_threshold: int = 3
    lost_timeout: int = 60
    velocity_ema_alpha: float = 0.4
    max_trajectory_length: int = 300
    max_event_length: int = 50
    occlusion_iou_threshold: float = 0.30

@dataclass(frozen=True)
class LoggingConfig:
    log_dir: str = "logs"
    level: str = "INFO"

@dataclass(frozen=True)
class CCSMConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

CONFIG = CCSMConfig()
