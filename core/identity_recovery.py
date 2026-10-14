# core/identity_recovery.py
# Phase 6 - Spatial-Temporal Identity Recovery (Re-Association / Re-ID)
"""
Matches newly detected objects against currently OCCLUDED or recently LOST
memory objects to re-associate identities when the tracker (ByteTrack)
generates a new track ID after an occlusion gap.

Evaluation signals:
  1. Class identity (hard requirement)
  2. Spatial distance to predicted/last position
  3. Directional motion consistency
  4. Bounding box scale & aspect ratio similarity
  5. Temporal recency (penalty for long absence)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.settings import CONFIG, IdentityRecoveryConfig
from core.memory_models import ObjectState, TrackedObjectMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecoveryMatch:
    recovered_id: int
    score: float
    distance_px: float
    details: Dict[str, float]


class IdentityRecoveryEngine:
    """
    Evaluates new tracker detections against missing memory objects to
    recover original persistent track IDs.
    """

    def __init__(self, cfg: Optional[IdentityRecoveryConfig] = None) -> None:
        self._cfg: IdentityRecoveryConfig = cfg or CONFIG.recovery

    def find_match(
        self,
        new_obj: dict,
        candidates: List[TrackedObjectMemory],
        claimed_ids: Optional[set] = None,
    ) -> Optional[RecoveryMatch]:
        """
        Evaluate candidate occluded/lost objects against `new_obj`.

        Parameters
        ----------
        new_obj : dict
            Detection dict from tracker with track_id, class_id, class_name,
            bbox, center, confidence.
        candidates : List[TrackedObjectMemory]
            Active memory objects currently in OCCLUDED or LOST state.
        claimed_ids : set, optional
            Set of candidate track_ids already matched this frame.

        Returns
        -------
        Optional[RecoveryMatch]
            The best matching candidate if score >= min_recovery_score.
        """
        if not self._cfg.enabled or not candidates:
            return None

        claimed = claimed_ids or set()
        det_cid = new_obj.get("class_id")
        det_cname = new_obj.get("class_name")
        det_cx, det_cy = new_obj["center"]
        det_bbox = new_obj["bbox"]
        dw = max(1.0, det_bbox[2] - det_bbox[0])
        dh = max(1.0, det_bbox[3] - det_bbox[1])
        det_area = dw * dh
        det_ar = dw / dh

        best_match: Optional[RecoveryMatch] = None
        best_score: float = -1.0

        for cand in candidates:
            if cand.track_id in claimed:
                continue

            # 1. Hard class match
            if cand.class_id != det_cid and cand.class_name != det_cname:
                continue

            # Candidate must be missing
            if cand.state not in (ObjectState.OCCLUDED, ObjectState.LOST):
                continue

            if cand.missing_frames > self._cfg.max_candidate_missing_frames:
                continue

            # Position check
            ref_pos = cand.predicted_position or cand.last_position
            if ref_pos is None:
                continue

            px, py = ref_pos
            dist = ((det_cx - px) ** 2 + (det_cy - py) ** 2) ** 0.5
            if dist > self._cfg.max_recovery_distance:
                continue

            # 2. Distance score (0.0 to 1.0)
            s_dist = max(0.0, 1.0 - (dist / self._cfg.max_recovery_distance))

            # 3. Motion vector consistency
            s_motion = self._score_motion(cand, det_cx, det_cy)

            # 4. Scale & aspect ratio similarity
            s_scale = self._score_geometry(cand, det_area, det_ar)

            # 5. Temporal recency
            s_temporal = max(
                0.0,
                1.0 - (cand.missing_frames / self._cfg.max_candidate_missing_frames),
            )

            # Composite weighted score
            total_score = (
                self._cfg.w_distance * s_dist
                + self._cfg.w_motion * s_motion
                + self._cfg.w_scale * s_scale
                + self._cfg.w_temporal * s_temporal
            )

            if total_score >= self._cfg.min_recovery_score and total_score > best_score:
                best_score = total_score
                best_match = RecoveryMatch(
                    recovered_id=cand.track_id,
                    score=round(total_score, 3),
                    distance_px=round(dist, 1),
                    details={
                        "s_dist": round(s_dist, 3),
                        "s_motion": round(s_motion, 3),
                        "s_scale": round(s_scale, 3),
                        "s_temporal": round(s_temporal, 3),
                    },
                )

        return best_match

    def _score_motion(
        self, cand: TrackedObjectMemory, det_cx: float, det_cy: float
    ) -> float:
        """Evaluate directional consistency between candidate velocity and displacement."""
        if cand.last_position is None:
            return 0.5

        lx, ly = cand.last_position
        dx = det_cx - lx
        dy = det_cy - ly
        disp_mag = (dx ** 2 + dy ** 2) ** 0.5
        vel_mag = cand.velocity.speed

        if vel_mag < 1.0 or disp_mag < 3.0:
            # Low motion: stationary consistency
            return 0.85 if disp_mag < 40.0 else max(0.0, 1.0 - disp_mag / 100.0)

        # Cosine similarity of velocity vector and total displacement vector
        dot = cand.velocity.vx * dx + cand.velocity.vy * dy
        cos_sim = dot / (vel_mag * disp_mag)
        # Normalize from [-1.0, 1.0] to [0.0, 1.0]
        return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))

    def _score_geometry(
        self, cand: TrackedObjectMemory, det_area: float, det_ar: float
    ) -> float:
        """Evaluate similarity of bounding box area and aspect ratio."""
        if cand.last_bbox is None:
            return 0.5

        bx1, by1, bx2, by2 = cand.last_bbox
        cw = max(1.0, bx2 - bx1)
        ch = max(1.0, by2 - by1)
        c_area = cw * ch
        c_ar = cw / ch

        area_sim = min(c_area, det_area) / max(c_area, det_area)
        ar_sim = min(c_ar, det_ar) / max(c_ar, det_ar)
        return area_sim * 0.5 + ar_sim * 0.5
