# core/memory_manager.py
from __future__ import annotations
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from config.settings import CONFIG, MemoryConfig
from core.memory_models import (
    MemoryEvent, ObjectState, TrackedObjectMemory, TrajectoryPoint, Velocity,
)

logger = logging.getLogger(__name__)


def _iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    union = max(0.0,(ax2-ax1)*(ay2-ay1)) + max(0.0,(bx2-bx1)*(by2-by1)) - inter
    return inter/union if union > 0 else 0.0


def _smooth_velocity(old: Velocity, rvx: float, rvy: float, alpha: float) -> Velocity:
    return Velocity(vx=alpha*rvx+(1-alpha)*old.vx, vy=alpha*rvy+(1-alpha)*old.vy)


def _predict_position(
    last: Tuple[float,float], v: Velocity, n: int
) -> Tuple[float,float]:
    return (last[0]+v.vx*n, last[1]+v.vy*n)


class MemoryManager:
    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self._cfg: MemoryConfig = config or CONFIG.memory
        self._registry: Dict[int, TrackedObjectMemory] = {}
        logger.info("MemoryManager init | occ=%d lost=%d",
                    self._cfg.occlusion_threshold, self._cfg.lost_timeout)

    def update(self, tracked_objects: List[dict], frame_index: int) -> None:
        now = time.monotonic()
        detected_ids = {o["track_id"] for o in tracked_objects}
        for obj in tracked_objects:
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
        logger.debug("Frame %d | vis=%d occ=%d lost=%d", frame_index,
                     len(self.visible_objects()), len(self.occluded_objects()),
                     len(self.lost_objects()))

    def all_objects(self) -> List[TrackedObjectMemory]:
        return list(self._registry.values())

    def visible_objects(self) -> List[TrackedObjectMemory]:
        return [m for m in self._registry.values() if m.is_visible]

    def occluded_objects(self) -> List[TrackedObjectMemory]:
        return [m for m in self._registry.values() if m.state == ObjectState.OCCLUDED]

    def lost_objects(self) -> List[TrackedObjectMemory]:
        return [m for m in self._registry.values() if m.state == ObjectState.LOST]

    def get(self, track_id: int) -> Optional[TrackedObjectMemory]:
        return self._registry.get(track_id)

    def purge_lost(self) -> int:
        lost = [t for t,m in self._registry.items() if m.state == ObjectState.LOST]
        for t in lost:
            del self._registry[t]
        if lost:
            logger.info("Purged %d LOST: %s", len(lost), lost)
        return len(lost)

    def summary(self) -> List[dict]:
        return [m.to_summary_dict() for m in self._registry.values()]

    def _create_new(self, obj: dict, frame_index: int, now: float) -> None:
        tid = obj["track_id"]
        cx, cy = obj["center"]
        bbox = tuple(obj["bbox"])
        mem = TrackedObjectMemory(
            track_id=tid, class_id=obj["class_id"], class_name=obj["class_name"],
            first_seen_frame=frame_index, first_seen_time=now,
            last_seen_frame=frame_index, last_seen_time=now, current_frame=frame_index,
            state=ObjectState.NEW, last_position=(cx,cy), last_bbox=bbox,
            last_confidence=obj["confidence"],
            trajectory=deque(maxlen=self._cfg.max_trajectory_length),
            events=deque(maxlen=self._cfg.max_event_length),
        )
        mem.trajectory.append(TrajectoryPoint(
            frame=frame_index, timestamp=now, cx=cx, cy=cy,
            bbox=bbox, confidence=obj["confidence"]))
        mem.events.append(MemoryEvent(
            frame=frame_index, timestamp=now, event_type="NEW",
            detail=f"First: {obj['class_name']} id={tid}"))
        self._registry[tid] = mem
        logger.info("NEW id=%d cls=%s pos=(%.1f,%.1f)", tid, obj["class_name"], cx, cy)

    def _update_existing(self, obj: dict, frame_index: int, now: float) -> None:
        tid = obj["track_id"]
        mem = self._registry[tid]
        cx, cy = obj["center"]
        bbox = tuple(obj["bbox"])
        conf = obj["confidence"]
        if mem.last_position is not None:
            rvx = cx - mem.last_position[0]
            rvy = cy - mem.last_position[1]
            mem.velocity = _smooth_velocity(mem.velocity, rvx, rvy, self._cfg.velocity_ema_alpha)
        mem.previous_position = mem.last_position
        mem.last_position = (cx, cy)
        mem.last_bbox = bbox
        mem.last_confidence = conf
        mem.current_frame = frame_index
        mem.last_seen_frame = frame_index
        mem.last_seen_time = now
        prev = mem.state
        if mem.state in (ObjectState.OCCLUDED, ObjectState.LOST):
            mem.state = ObjectState.REAPPEARED
            mem.reappearance_count += 1
            mem.occluder_id = None
            mem.events.append(MemoryEvent(
                frame=frame_index, timestamp=now, event_type="REAPPEARED",
                detail=f"Reappeared after {mem.missing_frames} frames (#{mem.reappearance_count})"))
            logger.info("REAPPEARED id=%d after %d frames", tid, mem.missing_frames)
        elif mem.state in (ObjectState.NEW, ObjectState.REAPPEARED):
            mem.state = ObjectState.VISIBLE
        else:
            mem.state = ObjectState.VISIBLE
        mem.missing_frames = 0
        mem.occlusion_start_frame = None
        mem.predicted_position = None
        mem.trajectory.append(TrajectoryPoint(
            frame=frame_index, timestamp=now, cx=cx, cy=cy, bbox=bbox, confidence=conf))
        if prev != mem.state:
            logger.debug("State %s->%s id=%d", prev.value, mem.state.value, tid)

    def _handle_missing(self, mem: TrackedObjectMemory, frame_index: int, now: float) -> None:
        if mem.state == ObjectState.LOST:
            mem.current_frame = frame_index
            return
        mem.missing_frames += 1
        mem.current_frame = frame_index
        if mem.last_position is not None:
            mem.predicted_position = _predict_position(
                mem.last_position, mem.velocity, mem.missing_frames)
        if mem.missing_frames <= self._cfg.occlusion_threshold:
            pass
        elif mem.missing_frames <= self._cfg.lost_timeout:
            if mem.state != ObjectState.OCCLUDED:
                mem.state = ObjectState.OCCLUDED
                if mem.occlusion_start_frame is None:
                    mem.occlusion_start_frame = frame_index
                mem.events.append(MemoryEvent(
                    frame=frame_index, timestamp=now, event_type="OCCLUDED",
                    detail=f"Occluded at {frame_index}. Last seen {mem.last_seen_frame}."))
                logger.info("OCCLUDED id=%d cls=%s miss=%d pred=%s",
                            mem.track_id, mem.class_name,
                            mem.missing_frames, mem.predicted_position)
        else:
            if mem.state != ObjectState.LOST:
                mem.state = ObjectState.LOST
                mem.events.append(MemoryEvent(
                    frame=frame_index, timestamp=now, event_type="LOST",
                    detail=f"LOST after {mem.missing_frames} frames."))
                logger.info("LOST id=%d cls=%s miss=%d",
                            mem.track_id, mem.class_name, mem.missing_frames)

    def _apply_occluder_logic(self, frame_index: int) -> None:
        short_miss = [
            m for m in self._registry.values()
            if 0 < m.missing_frames <= self._cfg.occlusion_threshold
            and m.state not in (ObjectState.LOST, ObjectState.OCCLUDED)
        ]
        vis = [m for m in self._registry.values()
               if m.is_visible and m.last_bbox is not None]
        for mm in short_miss:
            if mm.predicted_position is None or mm.last_bbox is None:
                continue
            lx1,ly1,lx2,ly2 = mm.last_bbox
            w, h = lx2-lx1, ly2-ly1
            px, py = mm.predicted_position
            pb = (px-w/2, py-h/2, px+w/2, py+h/2)
            best_iou, best_id = 0.0, None
            for vm in vis:
                if vm.track_id == mm.track_id:
                    continue
                iou = _iou(pb, vm.last_bbox)
                if iou > best_iou:
                    best_iou, best_id = iou, vm.track_id
            if best_iou >= self._cfg.occlusion_iou_threshold and best_id is not None:
                prev = mm.state
                mm.state = ObjectState.OCCLUDED
                mm.occluder_id = best_id
                if mm.occlusion_start_frame is None:
                    mm.occlusion_start_frame = frame_index
                if prev != ObjectState.OCCLUDED:
                    now = time.monotonic()
                    mm.events.append(MemoryEvent(
                        frame=frame_index, timestamp=now, event_type="OCCLUDED",
                        detail=f"Occluder-aware: overlaps id={best_id} IoU={best_iou:.2f}"))
                    logger.info("OCCLUDED(occluder) id=%d behind=%d IoU=%.2f",
                                mm.track_id, best_id, best_iou)
