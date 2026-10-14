# config/settings.py
# CCSM Phase 5.2 & Phase 6 - Stable Lifecycle & Spatial-Temporal Recovery
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
    # Frames without detection before OCCLUDED state is considered
    occlusion_threshold: int = 3
    # Frames without detection before LOST state is declared
    lost_timeout: int = 90
    # EMA smoothing factor for velocity (0 < alpha <= 1)
    velocity_ema_alpha: float = 0.4
    # Max stored trajectory points per object
    max_trajectory_length: int = 300
    # Max stored events per object
    max_event_length: int = 50
    # Minimum IoU between predicted bbox and an occluder bbox to confirm occlusion
    occlusion_iou_threshold: float = 0.20
    # Maximum pixels an object can move per frame (prediction clamp)
    max_prediction_displacement: float = 80.0
    # Velocity decay factor per missing frame (1.0 = no decay)
    velocity_decay: float = 0.92
    # Maximum frames to keep predicting position
    max_prediction_frames: int = 90
    # [Phase 5.2] Consecutive detections needed to confirm NEW -> VISIBLE
    min_confirmation_frames: int = 2
    # [Phase 5.2] Auto-purge LOST objects this many frames after they become LOST
    purge_lost_after_frames: int = 300

@dataclass(frozen=True)
class IdentityRecoveryConfig:
    # Enable spatial-temporal identity re-association
    enabled: bool = True
    # Maximum distance (pixels) between new detection center and candidate predicted position
    max_recovery_distance: float = 140.0
    # Minimum matching score (0.0 to 1.0) to declare a match
    min_recovery_score: float = 0.55
    # Maximum missing frames for an occluded/lost object to remain a candidate
    max_candidate_missing_frames: int = 150
    # Scoring weights
    w_distance: float = 0.45
    w_motion: float = 0.25
    w_scale: float = 0.20
    w_temporal: float = 0.10

@dataclass(frozen=True)
class OptimizerConfig:
    # Max waypoint points to retain per trajectory after compression
    max_retained_waypoints: int = 50
    # Minimum pixel movement to record a new waypoint
    min_waypoint_distance: float = 12.0
    # Douglas-Peucker epsilon for trajectory decimation
    compression_epsilon: float = 4.0

@dataclass(frozen=True)
class VisualizationConfig:
    show_hud: bool = True
    show_ghost: bool = True
    ghost_dash_count: int = 12
    colour_visible: tuple = (0, 220, 0)
    colour_occluded: tuple = (0, 140, 255)
    colour_reappeared: tuple = (0, 255, 220)
    colour_lost: tuple = (60, 60, 200)
    colour_unconfirmed: tuple = (180, 180, 60)

@dataclass(frozen=True)
class LoggingConfig:
    log_dir: str = "logs"
    level: str = "INFO"
    dashboard_interval: int = 30
    debug_mode: bool = False

@dataclass(frozen=True)
class CCSMConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    recovery: IdentityRecoveryConfig = field(default_factory=IdentityRecoveryConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

CONFIG = CCSMConfig()
