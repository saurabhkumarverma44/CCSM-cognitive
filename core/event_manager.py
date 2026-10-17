
"""CCSM Phase 9: structured lifecycle events with listener support."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Deque, Optional

from core.memory_models import ObjectState, TrackedObjectMemory

logger = logging.getLogger(__name__)

EventListener = Callable[["SceneEvent"], None]


@dataclass(frozen=True)
class SceneEvent:
    """Immutable record of a tracked-object lifecycle event."""

    event_id: int
    timestamp: str
    frame: int
    track_id: int
    class_name: str
    old_state: str
    new_state: str
    missing_frames: int
    occluder_id: Optional[int]
    stability_score: float
    reason: str


class EventManager:
    """
    Capture important memory transitions and notify subscribers.

    Event history is retained independently of listener notifications.
    Listener callbacks run outside the history lock, so they cannot
    hold up event-history operations while performing their work.
    """

    IMPORTANT_STATES = {
        ObjectState.NEW,
        ObjectState.OCCLUDED,
        ObjectState.REAPPEARED,
        ObjectState.LOST,
    }

    def __init__(self, max_history: int = 1000) -> None:
        if max_history < 1:
            raise ValueError("max_history must be at least 1.")

        self._events: Deque[SceneEvent] = deque(maxlen=max_history)
        self._listeners: list[EventListener] = []
        self._lock = threading.Lock()
        self._next_id = 1

    def register_listener(self, callback: EventListener) -> None:
        """Register a listener, avoiding duplicate registrations."""
        if not callable(callback):
            raise TypeError("callback must be callable.")

        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def unregister_listener(self, callback: EventListener) -> None:
        """Remove a previously registered listener."""
        with self._lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

    def _create_event(
        self,
        *,
        frame: int,
        track_id: int,
        class_name: str,
        old_state: str,
        new_state: str,
        missing_frames: int,
        occluder_id: Optional[int],
        stability_score: float,
        reason: str,
    ) -> SceneEvent:
        """Assign an event ID and store the event atomically."""
        event = SceneEvent(
            event_id=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame=int(frame),
            track_id=int(track_id),
            class_name=str(class_name),
            old_state=str(old_state),
            new_state=str(new_state),
            missing_frames=int(missing_frames),
            occluder_id=occluder_id,
            stability_score=round(float(stability_score), 3),
            reason=str(reason),
        )

        with self._lock:
            event = SceneEvent(
                event_id=self._next_id,
                timestamp=event.timestamp,
                frame=event.frame,
                track_id=event.track_id,
                class_name=event.class_name,
                old_state=event.old_state,
                new_state=event.new_state,
                missing_frames=event.missing_frames,
                occluder_id=event.occluder_id,
                stability_score=event.stability_score,
                reason=event.reason,
            )
            self._next_id += 1
            self._events.append(event)

        self._notify_listeners(event)
        return event

    def _notify_listeners(self, event: SceneEvent) -> None:
        """Notify registered listeners without holding the history lock."""
        with self._lock:
            listeners = tuple(self._listeners)

        for callback in listeners:
            try:
                callback(event)
            except Exception:
                # A subscriber must not break the perception pipeline.
                logger.exception(
                    "Scene event listener failed for event_id=%d.",
                    event.event_id,
                )

    def on_state_change(
        self,
        mem: TrackedObjectMemory,
        old_state: ObjectState,
        new_state: ObjectState,
        frame: int,
    ) -> None:
        """Callback compatible with MemoryManager state notifications."""
        if new_state not in self.IMPORTANT_STATES:
            return

        # Ignore redundant notifications, except initial NEW events.
        if old_state == new_state and new_state != ObjectState.NEW:
            return

        self._create_event(
            frame=frame,
            track_id=mem.track_id,
            class_name=mem.class_name,
            old_state=old_state.value,
            new_state=new_state.value,
            missing_frames=mem.missing_frames,
            occluder_id=mem.occluder_id,
            stability_score=mem.stability_score,
            reason=self._describe_transition(old_state, new_state, mem),
        )

    def on_purge(self, mem: TrackedObjectMemory, frame: int) -> None:
        """Record object-memory removal as a PURGED event."""
        self._create_event(
            frame=frame,
            track_id=mem.track_id,
            class_name=mem.class_name,
            old_state=mem.state.value,
            new_state="PURGED",
            missing_frames=mem.missing_frames,
            occluder_id=mem.occluder_id,
            stability_score=mem.stability_score,
            reason="Object memory was purged.",
        )

    def get_events(self, limit: Optional[int] = None) -> list[SceneEvent]:
        """Return an oldest-first snapshot without modifying history."""
        if limit is not None and limit < 0:
            raise ValueError("limit must not be negative.")

        with self._lock:
            events = list(self._events)

        if limit is not None:
            return events[-limit:] if limit else []

        return events

    def drain_events(self) -> list[SceneEvent]:
        """Return and clear the current history buffer."""
        with self._lock:
            events = list(self._events)
            self._events.clear()

        return events

    @staticmethod
    def _describe_transition(
        old_state: ObjectState,
        new_state: ObjectState,
        mem: TrackedObjectMemory,
    ) -> str:
        """Describe a lifecycle transition using available memory data."""
        if new_state == ObjectState.NEW:
            return "A new object entered memory."

        if new_state == ObjectState.OCCLUDED:
            if mem.occluder_id is not None:
                return (
                    "Object became occluded; possible occluder ID "
                    f"{mem.occluder_id}."
                )
            return "Object became occluded."

        if new_state == ObjectState.REAPPEARED:
            return (
                "Object reappeared after "
                f"{mem.last_occlusion_duration} missing frame(s)."
            )

        if new_state == ObjectState.LOST:
            return (
                "Object was marked lost after "
                f"{mem.missing_frames} missing frame(s)."
            )

        return (
            f"State changed from {old_state.value} "
            f"to {new_state.value}."
        )
