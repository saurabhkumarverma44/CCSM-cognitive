# core/memory_manager.py  Phase 5.2 & Phase 6 - Stable Lifecycle & Identity Recovery
from __future__ import annotations
import logging
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

from config.settings import CONFIG, MemoryConfig, IdentityRecoveryConfig
from core.identity_recovery import IdentityRecoveryEngine, RecoveryMatch
from core.memory_models import (
    MemoryEvent, ObjectState, TrackedObjectMemory, TrajectoryPoint, Velocity,
)

logger = logging.getLogger(__name__)


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - inter
    return inter / union if union > 0 else 0.0


def _smooth_velocity(old: Velocity, rvx: float, rvy: float, alpha: float) -> Velocity:
    return Velocity(vx=alpha * rvx + (1 - alpha) * old.vx, vy=alpha * rvy + (1 - alpha) * old.vy)


def _predict_position(last: Tuple[float, float], v: Velocity,
                      n: int, max_disp: float) -> Tuple[float, float]:
    raw_dx, raw_dy = v.vx * n, v.vy * n
    dist = (raw_dx ** 2 + raw_dy ** 2) ** 0.5
    max_total = max_disp * n
    if dist > max_total and dist > 0:
        scale = max_total / dist
        raw_dx *= scale; raw_dy *= scale
    return (last[0] + raw_dx, last[1] + raw_dy)


def _decay_velocity(v: Velocity, decay: float) -> Velocity:
    return Velocity(vx=v.vx * decay, vy=v.vy * decay)


StateChangeCallback = Callable[[TrackedObjectMemory, ObjectState, ObjectState, int], None]
PurgeCallback       = Callable[[TrackedObjectMemory, int], None]


class MemoryManager:
    """
    Maintains a persistent registry of TrackedObjectMemory records keyed by track_id.

    Phase 5.2 features:
      - Object confirmation gate (min_confirmation_frames before NEW -> VISIBLE)
      - Lifecycle counters (total_visible_frames, total_occluded_frames, consecutive_detections)
      - last_occlusion_duration accurately captured prior to reset
      - Auto-purge of stale LOST objects after purge_lost_after_frames
      - PurgeCallback for lifecycle summary logging

    Phase 6 features:
      - Spatial-temporal identity recovery (re-associating broken ByteTrack tracks)
      - Tracker ID to Persistent Memory ID re-mapping
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        recovery_config: Optional[IdentityRecoveryConfig] = None,
    ) -> None:
        self._cfg: MemoryConfig = config or CONFIG.memory
        self._rec_cfg: IdentityRecoveryConfig = recovery_config or CONFIG.recovery
        self._recovery_engine = IdentityRecoveryEngine(self._rec_cfg)
        self._tracker_to_memory_id: Dict[int, int] = {}
        self._registry: Dict[int, TrackedObjectMemory] = {}
        self._state_callbacks: List[StateChangeCallback] = []
        self._purge_callbacks: List[PurgeCallback] = []
        logger.info(
            "MemoryManager init occ=%d lost=%d confirm=%d purge_after=%d recovery=%s",
            self._cfg.occlusion_threshold, self._cfg.lost_timeout,
            self._cfg.min_confirmation_frames, self._cfg.purge_lost_after_frames,
            self._rec_cfg.enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_state_callback(self, cb: StateChangeCallback) -> None:
        self._state_callbacks.append(cb)

    def register_purge_callback(self, cb: PurgeCallback) -> None:
        """Register a callback fired just before a LOST object is auto-purged."""
        self._purge_callbacks.append(cb)

    def update(self, tracked_objects: List[dict], frame_index: int) -> None:
        now = time.monotonic()

        # Step 1: Remap any known tracker IDs that were previously re-associated
        processed_objects: List[dict] = []
        for obj in tracked_objects:
            raw_tid = obj["track_id"]
            if raw_tid in self._tracker_to_memory_id:
                mapped_tid = self._tracker_to_memory_id[raw_tid]
                c = dict(obj)
                c["track_id"] = mapped_tid
                processed_objects.append(c)
            else:
                processed_objects.append(obj)

        # Step 2: Attempt identity recovery for new detections not yet in _registry
        missing_candidates = [
            m for m in self._registry.values()
            if m.state in (ObjectState.OCCLUDED, ObjectState.LOST)
        ]
        claimed_ids = set()

        final_objects: List[dict] = []
        for obj in processed_objects:
            tid = obj["track_id"]
            if tid not in self._registry and self._rec_cfg.enabled and missing_candidates:
                match = self._recovery_engine.find_match(obj, missing_candidates, claimed_ids)
                if match:
                    recovered_id = match.recovered_id
                    claimed_ids.add(recovered_id)
                    raw_tid = obj["track_id"]
                    self._tracker_to_memory_id[raw_tid] = recovered_id

                    c = dict(obj)
                    c["track_id"] = recovered_id
                    final_objects.append(c)

                    logger.info(
                        "Identity recovered: ByteTrack ID=%d -> persistent ID=%d (score=%.2f, dist=%.1fpx)",
                        raw_tid, recovered_id, match.score, match.distance_px,
                    )
                    continue

            final_objects.append(obj)

        detected_ids = {o["track_id"] for o in final_objects}

        for obj in final_objects:
            tid = obj["track_id"]
            if tid in self._registry:
                self._update_existing(obj, frame_index, now)
            else:
                self._create_new(obj, frame_index, now)

        for tid, mem in list(self._registry.items()):
            if tid not in detected_ids:
                self._handle_missing(mem, frame_index, now)
            else:
                mem.current_frame = frame_index

        self._apply_occluder_logic(frame_index)
        self._auto_purge_lost(frame_index)

        logger.debug("Frame %d vis=%d occ=%d lost=%d", frame_index,
                     len(self.visible_objects()), len(self.occluded_objects()),
                     len(self.lost_objects()))

    def all_objects(self) -> List[TrackedObjectMemory]:
        return list(self._registry.values())

    def visible_objects(self) -> List[TrackedObjectMemory]:
        return [m for m in self._registry.values() if m.is_visible]

    def occluded_objects(self) -> List[TrackedObjectMemory]:
        return [m for m in self._registry.values()
                if m.state == ObjectState.OCCLUDED]

    def lost_objects(self) -> List[TrackedObjectMemory]:
        return [m for m in self._registry.values()
                if m.state == ObjectState.LOST]

    def get(self, track_id: int) -> Optional[TrackedObjectMemory]:
        return self._registry.get(track_id)

    def purge_lost(self) -> int:
        """Manually remove all LOST objects. Returns count removed."""
        lost = [t for t, m in self._registry.items() if m.state == ObjectState.LOST]
        for t in lost:
            mem = self._registry.pop(t)
            self._tracker_to_memory_id = {k: v for k, v in self._tracker_to_memory_id.items() if v != t}
            self._fire_purge(mem, -1)
        if lost:
            logger.info("Manual purge: removed %d LOST: %s", len(lost), lost)
        return len(lost)

    def summary(self) -> List[dict]:
        return [m.to_summary_dict() for m in self._registry.values()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fire_state_change(self, mem: TrackedObjectMemory,
                           old: ObjectState, new: ObjectState, frame: int) -> None:
        for cb in self._state_callbacks:
            try:
                cb(mem, old, new, frame)
            except Exception:
                logger.exception("State callback error")

    def _fire_purge(self, mem: TrackedObjectMemory, frame: int) -> None:
        for cb in self._purge_callbacks:
            try:
                cb(mem, frame)
            except Exception:
                logger.exception("Purge callback error")

    def _create_new(self, obj: dict, frame_index: int, now: float) -> None:
        tid = obj["track_id"]
        cx, cy = obj["center"]
        bbox = tuple(obj["bbox"])
        mem = TrackedObjectMemory(
            track_id=tid, class_id=obj["class_id"], class_name=obj["class_name"],
            first_seen_frame=frame_index, first_seen_time=now,
            last_seen_frame=frame_index, last_seen_time=now,
            current_frame=frame_index, state=ObjectState.NEW,
            last_position=(cx, cy), last_bbox=bbox,
            last_confidence=obj["confidence"],
            trajectory=deque(maxlen=self._cfg.max_trajectory_length),
            events=deque(maxlen=self._cfg.max_event_length),
            consecutive_detections=1,
            total_visible_frames=1,  # first detection counts
        )
        mem.trajectory.append(TrajectoryPoint(
            frame=frame_index, timestamp=now, cx=cx, cy=cy,
            bbox=bbox, confidence=obj["confidence"]))
        mem.events.append(MemoryEvent(
            frame=frame_index, timestamp=now, event_type="NEW",
            detail=f"First detection: {obj['class_name']} id={tid}"))
        self._registry[tid] = mem
        self._fire_state_change(mem, ObjectState.NEW, ObjectState.NEW, frame_index)

    def _update_existing(self, obj: dict, frame_index: int, now: float) -> None:
        tid = obj["track_id"]
        mem = self._registry[tid]
        cx, cy = obj["center"]
        bbox = tuple(obj["bbox"])
        conf = obj["confidence"]

        if mem.last_position is not None:
            rvx = cx - mem.last_position[0]
            rvy = cy - mem.last_position[1]
            mem.velocity = _smooth_velocity(mem.velocity, rvx, rvy,
                                            self._cfg.velocity_ema_alpha)

        mem.previous_position = mem.last_position
        mem.last_position = (cx, cy)
        mem.last_bbox = bbox
        mem.last_confidence = conf
        mem.current_frame = frame_index
        mem.last_seen_frame = frame_index
        mem.last_seen_time = now

        # Lifecycle counters
        mem.consecutive_detections += 1
        mem.total_visible_frames += 1
        if not mem.confirmed and mem.consecutive_detections >= self._cfg.min_confirmation_frames:
            mem.confirmed = True

        prev = mem.state

        if mem.state in (ObjectState.OCCLUDED, ObjectState.LOST):
            # KEY FIX: capture missing count BEFORE resetting
            mem.last_occlusion_duration = mem.missing_frames
            mem.state = ObjectState.REAPPEARED
            mem.reappearance_count += 1
            mem.occluder_id = None
            detail = (f"Reappeared after {mem.last_occlusion_duration} frames "
                      f"(#{mem.reappearance_count})")
            mem.events.append(MemoryEvent(
                frame=frame_index, timestamp=now,
                event_type="REAPPEARED", detail=detail))

        elif mem.state in (ObjectState.NEW, ObjectState.REAPPEARED):
            if mem.confirmed:
                mem.state = ObjectState.VISIBLE
            # else: stay in current state until confirmed

        else:
            mem.state = ObjectState.VISIBLE

        # Reset missing state after recording duration
        mem.missing_frames = 0
        mem.occlusion_start_frame = None
        mem.predicted_position = None

        mem.trajectory.append(TrajectoryPoint(
            frame=frame_index, timestamp=now, cx=cx, cy=cy,
            bbox=bbox, confidence=conf))

        if prev != mem.state:
            self._fire_state_change(mem, prev, mem.state, frame_index)

    def _handle_missing(self, mem: TrackedObjectMemory,
                        frame_index: int, now: float) -> None:
        if mem.state == ObjectState.LOST:
            mem.current_frame = frame_index
            return

        mem.missing_frames += 1
        mem.current_frame = frame_index
        mem.consecutive_detections = 0

        mem.velocity = _decay_velocity(mem.velocity, self._cfg.velocity_decay)

        if (mem.last_position is not None
                and mem.missing_frames <= self._cfg.max_prediction_frames):
            mem.predicted_position = _predict_position(
                mem.last_position, mem.velocity, mem.missing_frames,
                self._cfg.max_prediction_displacement)

        prev = mem.state

        if mem.missing_frames <= self._cfg.occlusion_threshold:
            pass  # occluder logic may still promote

        elif mem.missing_frames <= self._cfg.lost_timeout:
            if mem.state != ObjectState.OCCLUDED:
                mem.state = ObjectState.OCCLUDED
                if mem.occlusion_start_frame is None:
                    mem.occlusion_start_frame = frame_index
                mem.events.append(MemoryEvent(
                    frame=frame_index, timestamp=now, event_type="OCCLUDED",
                    detail=f"Occluded at frame {frame_index}. Last seen {mem.last_seen_frame}."))

        else:
            if mem.state != ObjectState.LOST:
                mem.state = ObjectState.LOST
                mem.lost_at_frame = frame_index
                mem.events.append(MemoryEvent(
                    frame=frame_index, timestamp=now, event_type="LOST",
                    detail=f"LOST after {mem.missing_frames} frames."))

        if mem.state == ObjectState.OCCLUDED:
            mem.total_occluded_frames += 1

        if prev != mem.state:
            self._fire_state_change(mem, prev, mem.state, frame_index)

    def _apply_occluder_logic(self, frame_index: int) -> None:
        short_miss = [
            m for m in self._registry.values()
            if 0 < m.missing_frames <= self._cfg.occlusion_threshold
            and m.state not in (ObjectState.LOST, ObjectState.OCCLUDED)
        ]
        visible_with_bbox = [
            m for m in self._registry.values()
            if m.is_visible and m.last_bbox is not None
        ]
        for mm in short_miss:
            if mm.predicted_position is None or mm.last_bbox is None:
                continue
            lx1, ly1, lx2, ly2 = mm.last_bbox
            w, h = lx2 - lx1, ly2 - ly1
            px, py = mm.predicted_position
            pred_bbox = (px - w / 2, py - h / 2, px + w / 2, py + h / 2)
            best_iou, best_id = 0.0, None
            for vm in visible_with_bbox:
                if vm.track_id == mm.track_id:
                    continue
                iou = _iou(pred_bbox, vm.last_bbox)
                if iou > best_iou:
                    best_iou = iou; best_id = vm.track_id
            if best_iou >= self._cfg.occlusion_iou_threshold and best_id is not None:
                prev = mm.state
                mm.state = ObjectState.OCCLUDED
                mm.occluder_id = best_id
                mm.total_occluded_frames += 1
                if mm.occlusion_start_frame is None:
                    mm.occlusion_start_frame = frame_index
                if prev != ObjectState.OCCLUDED:
                    n = time.monotonic()
                    mm.events.append(MemoryEvent(
                        frame=frame_index, timestamp=n, event_type="OCCLUDED",
                        detail=f"Occluder-aware: pred bbox overlaps id={best_id} IoU={best_iou:.2f}"))
                    self._fire_state_change(mm, prev, ObjectState.OCCLUDED, frame_index)

    def _auto_purge_lost(self, frame_index: int) -> None:
        """Remove LOST objects that have been stale for purge_lost_after_frames."""
        to_purge = [
            tid for tid, mem in self._registry.items()
            if (mem.state == ObjectState.LOST
                and mem.lost_at_frame is not None
                and frame_index - mem.lost_at_frame >= self._cfg.purge_lost_after_frames)
        ]
        for tid in to_purge:
            mem = self._registry.pop(tid)
            self._tracker_to_memory_id = {k: v for k, v in self._tracker_to_memory_id.items() if v != tid}
            self._fire_purge(mem, frame_index)
            logger.info("Auto-purged id=%d cls=%s age=%df vis=%df occ=%df",
                        tid, mem.class_name, mem.age_frames,
                        mem.total_visible_frames, mem.total_occluded_frames)
