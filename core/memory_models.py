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
        }

    def __repr__(self) -> str:
        return (
            f"TrackedObjectMemory("
            f"id={self.track_id}, class={self.class_name}, "
            f"state={self.state.value}, pos={self.last_position}, "
            f"missing={self.missing_frames})"
        )
