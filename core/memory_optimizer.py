# core/memory_optimizer.py
# Phase 6 - Trajectory Memory Optimization & Importance Scoring
"""
Implements intelligent memory compression and observation importance scoring
to prevent unbounded storage growth while preserving critical inflection points,
velocity shifts, and occlusion waypoints.

Reference Research Goals:
  - Section 24: Memory Optimization
  - Section 25: Memory Importance Scoring
  - Section 26: Memory Compression
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config.settings import CONFIG, OptimizerConfig
from core.memory_models import TrackedObjectMemory, TrajectoryPoint


@dataclass(frozen=True)
class TrajectoryWaypoint:
    frame: int
    timestamp: float
    cx: float
    cy: float
    speed: float
    importance: float
    reason: str


@dataclass(frozen=True)
class MemoryCompressionSummary:
    total_objects: int
    raw_trajectory_points: int
    compressed_waypoints: int
    compression_ratio: float
    retention_percentage: float


class MemoryOptimizer:
    """
    Analyzes trajectory point streams and compresses them into key waypoints
    based on spatial deviation, speed shifts, and directional turning angles.
    """

    def __init__(self, cfg: Optional[OptimizerConfig] = None) -> None:
        self._cfg: OptimizerConfig = cfg or CONFIG.optimizer

    def compute_point_importance(
        self,
        prev_pt: Optional[TrajectoryPoint],
        curr_pt: TrajectoryPoint,
        next_pt: Optional[TrajectoryPoint],
    ) -> Tuple[float, str]:
        """
        Calculates an importance score (0.0 to 1.0) and primary reason for a trajectory point.
        """
        # Endpoints are always maximally important
        if prev_pt is None or next_pt is None:
            return 1.0, "endpoint"

        # Vector 1: prev -> curr
        v1x, v1y = curr_pt.cx - prev_pt.cx, curr_pt.cy - prev_pt.cy
        len1 = (v1x ** 2 + v1y ** 2) ** 0.5

        # Vector 2: curr -> next
        v2x, v2y = next_pt.cx - curr_pt.cx, next_pt.cy - curr_pt.cy
        len2 = (v2x ** 2 + v2y ** 2) ** 0.5

        # Stationary or very small movement
        if len1 < 2.0 and len2 < 2.0:
            return 0.1, "stationary"

        # Turning angle calculation (cosine between v1 and v2)
        turn_score = 0.0
        if len1 > 1.0 and len2 > 1.0:
            dot = v1x * v2x + v1y * v2y
            cos_theta = max(-1.0, min(1.0, dot / (len1 * len2)))
            # theta in [0, pi]; sharp turn (180 deg) has cos_theta = -1
            turn_score = (1.0 - cos_theta) / 2.0  # 0.0 for straight, 1.0 for u-turn

        # Speed delta
        speed_delta = abs(len2 - len1)
        speed_score = min(1.0, speed_delta / 15.0)

        # Composite score
        importance = turn_score * 0.60 + speed_score * 0.40

        reason = "path"
        if turn_score > 0.45:
            reason = "turn"
        elif speed_score > 0.40:
            reason = "acceleration"

        return min(1.0, max(0.05, importance)), reason

    def compress_trajectory(
        self, trajectory: Sequence[TrajectoryPoint]
    ) -> List[TrajectoryWaypoint]:
        """
        Compresses a sequence of TrajectoryPoints using Douglas-Peucker decimation
        augmented by turning angle and speed importance scores.
        """
        n = len(trajectory)
        if n == 0:
            return []
        if n <= 2:
            return [
                TrajectoryWaypoint(
                    frame=p.frame, timestamp=p.timestamp, cx=p.cx, cy=p.cy,
                    speed=0.0, importance=1.0, reason="endpoint",
                )
                for p in trajectory
            ]

        # Use Douglas-Peucker to find structural geometric vertices
        selected_indices = set()
        self._douglas_peucker(trajectory, 0, n - 1, self._cfg.compression_epsilon, selected_indices)
        selected_indices.add(0)
        selected_indices.add(n - 1)

        # Also score and add any major directional turns
        waypoints: List[TrajectoryWaypoint] = []
        for i in range(n):
            prev_p = trajectory[i - 1] if i > 0 else None
            curr_p = trajectory[i]
            next_p = trajectory[i + 1] if i < n - 1 else None

            imp, reason = self.compute_point_importance(prev_p, curr_p, next_p)

            # Keep if Douglas-Peucker kept it OR high turning importance
            if i in selected_indices or imp >= 0.55:
                speed = 0.0
                if prev_p is not None:
                    speed = ((curr_p.cx - prev_p.cx) ** 2 + (curr_p.cy - prev_p.cy) ** 2) ** 0.5
                waypoints.append(
                    TrajectoryWaypoint(
                        frame=curr_p.frame,
                        timestamp=curr_p.timestamp,
                        cx=curr_p.cx,
                        cy=curr_p.cy,
                        speed=round(speed, 2),
                        importance=round(imp, 3),
                        reason=reason,
                    )
                )

        # Cap at max_retained_waypoints if necessary
        if len(waypoints) > self._cfg.max_retained_waypoints:
            # Sort intermediate points by importance and keep top ones + endpoints
            first, last = waypoints[0], waypoints[-1]
            middle = sorted(waypoints[1:-1], key=lambda w: w.importance, reverse=True)
            kept = sorted(
                [first, last] + middle[: self._cfg.max_retained_waypoints - 2],
                key=lambda w: w.frame,
            )
            return kept

        return waypoints

    def _douglas_peucker(
        self,
        pts: Sequence[TrajectoryPoint],
        start_idx: int,
        end_idx: int,
        epsilon: float,
        out_indices: set,
    ) -> None:
        """Standard recursive Douglas-Peucker line simplification."""
        if end_idx <= start_idx + 1:
            return

        p1 = pts[start_idx]
        p2 = pts[end_idx]

        max_dist = 0.0
        max_idx = start_idx

        dx = p2.cx - p1.cx
        dy = p2.cy - p1.cy
        line_len = (dx ** 2 + dy ** 2) ** 0.5

        for i in range(start_idx + 1, end_idx):
            p = pts[i]
            if line_len == 0.0:
                dist = ((p.cx - p1.cx) ** 2 + (p.cy - p1.cy) ** 2) ** 0.5
            else:
                # Perpendicular distance formula
                dist = abs(dy * p.cx - dx * p.cy + p2.cx * p1.cy - p2.cy * p1.cx) / line_len

            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            out_indices.add(max_idx)
            self._douglas_peucker(pts, start_idx, max_idx, epsilon, out_indices)
            self._douglas_peucker(pts, max_idx, end_idx, epsilon, out_indices)

    def summarize_compression(
        self, objects: Sequence[TrackedObjectMemory]
    ) -> MemoryCompressionSummary:
        """Computes aggregate memory reduction statistics across all tracked objects."""
        total_raw = sum(len(m.trajectory) for m in objects)
        total_comp = sum(len(self.compress_trajectory(m.trajectory)) for m in objects)

        ratio = (total_raw / total_comp) if total_comp > 0 else 1.0
        ret_pct = (total_comp / total_raw * 100.0) if total_raw > 0 else 100.0

        return MemoryCompressionSummary(
            total_objects=len(objects),
            raw_trajectory_points=total_raw,
            compressed_waypoints=total_comp,
            compression_ratio=round(ratio, 2),
            retention_percentage=round(ret_pct, 1),
        )
