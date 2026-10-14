# core/memory_models.py
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, Tuple

class ObjectState(str, Enum):
    NEW = "NEW"
    VISIBLE = "VISIBLE"
    PARTIALLY_OCCLUDED = "PARTIALLY_OCCLUDED"
    OCCLUDED = "OCCLUDED"
    REAPPEARED = "REAPPEARED"
    LOST = "LOST"

@dataclass(frozen=True)
class TrajectoryPoint:
    frame: int
    timestamp: float
    cx: float
    cy: float
    bbox: Tuple[float, float, float, float]
    confidence: float

@dataclass(frozen=True)
class MemoryEvent:
    frame: int
    timestamp: float
    event_type: str
    detail: str = ""

@dataclass(frozen=True)
class Velocity:
    vx: float = 0.0
    vy: float = 0.0

    @property
    def speed(self) -> float:
        return (self.vx ** 2 + self.vy ** 2) ** 0.5

    def __repr__(self) -> str:
        return f"Velocity(vx={self.vx:.2f}, vy={self.vy:.2f}, spd={self.speed:.2f})"

@dataclass
class TrackedObjectMemory:
    track_id: int
    class_id: int
    class_name: str
    first_seen_frame: int
    first_seen_time: float = field(default_factory=time.monotonic)
    last_seen_frame: int = 0
    last_seen_time: float = field(default_factory=time.monotonic)
    current_frame: int = 0
    state: ObjectState = ObjectState.NEW
    missing_frames: int = 0
    occlusion_start_frame: Optional[int] = None
    reappearance_count: int = 0
    last_position: Optional[Tuple[float, float]] = None
    previous_position: Optional[Tuple[float, float]] = None
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    last_confidence: float = 0.0
    velocity: Velocity = field(default_factory=Velocity)
    trajectory: Deque[TrajectoryPoint] = field(default_factory=deque)
    events: Deque[MemoryEvent] = field(default_factory=deque)
    predicted_position: Optional[Tuple[float, float]] = None
    occluder_id: Optional[int] = None

    # --- Phase 5.2 Lifecycle fields ---
    # Frames the object was missing before its last reappearance (fixes "0 frames" display bug)
    last_occlusion_duration: int = 0
    # Cumulative frames spent in VISIBLE state
    total_visible_frames: int = 0
    # Cumulative frames spent in OCCLUDED state
    total_occluded_frames: int = 0
    # Consecutive detections without a miss (reset on each gap)
    consecutive_detections: int = 0
    # True once the object has been confirmed (N consecutive detections)
    confirmed: bool = False
    # Frame number when state transitioned to LOST (None if not yet LOST)
    lost_at_frame: Optional[int] = None

    @property
    def age_frames(self) -> int:
        return self.current_frame - self.first_seen_frame

    @property
    def is_visible(self) -> bool:
        return self.state in (
            ObjectState.VISIBLE, ObjectState.NEW,
            ObjectState.REAPPEARED, ObjectState.PARTIALLY_OCCLUDED,
        )

    @property
    def is_missing(self) -> bool:
        return self.state in (ObjectState.OCCLUDED, ObjectState.LOST)

    @property
    def visibility_ratio(self) -> float:
        """Fraction of active life spent visible (0.0 – 1.0)."""
        total = self.total_visible_frames + self.total_occluded_frames
        return self.total_visible_frames / total if total > 0 else 0.0

    @property
    def stability_score(self) -> float:
        """
        0.0 – 1.0 composite reliability metric.
        Rewards: high detection confidence, high visibility ratio, age > 60 frames.
        Penalises: unconfirmed status.
        """
        conf   = self.last_confidence
        vis    = self.visibility_ratio
        age_b  = min(1.0, self.age_frames / 60.0)
        conf_b = 0.0 if not self.confirmed else 0.05
        return min(1.0, conf * 0.35 + vis * 0.40 + age_b * 0.20 + conf_b)

    @property
    def time_since_last_seen(self) -> float:
        return time.monotonic() - self.last_seen_time

    def to_summary_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "state": self.state.value,
            "age_frames": self.age_frames,
            "missing_frames": self.missing_frames,
            "last_position": self.last_position,
            "predicted_position": self.predicted_position,
            "velocity": {
                "vx": round(self.velocity.vx, 3),
                "vy": round(self.velocity.vy, 3),
                "speed": round(self.velocity.speed, 3),
            },
            "last_confidence": round(self.last_confidence, 3),
            "reappearance_count": self.reappearance_count,
            "occluder_id": self.occluder_id,
            "last_seen_frame": self.last_seen_frame,
            # Phase 5.2 lifecycle
            "confirmed": self.confirmed,
            "total_visible_frames": self.total_visible_frames,
            "total_occluded_frames": self.total_occluded_frames,
            "visibility_ratio": round(self.visibility_ratio, 3),
            "stability_score": round(self.stability_score, 3),
        }

    def __repr__(self) -> str:
        return (
            f"TrackedObjectMemory("
            f"id={self.track_id}, class={self.class_name}, "
            f"state={self.state.value}, pos={self.last_position}, "
            f"missing={self.missing_frames})"
        )
