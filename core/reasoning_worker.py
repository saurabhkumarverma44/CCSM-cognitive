
"""CCSM asynchronous event reasoning with request pacing and bounded shutdown."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from core.event_manager import SceneEvent
from core.llm_client import GroqSceneAssistant

logger = logging.getLogger(__name__)

ContextProvider = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class ReasoningResult:
    """Result of processing one batch of scene events."""

    event_ids: tuple[int, ...]
    timestamp: str
    answer: Optional[str]
    error: Optional[str]


class EventReasoningWorker:
    """Process scene events on a background thread."""

    def __init__(
        self,
        context_provider: ContextProvider,
        assistant: Any = None,
        queue_capacity: int = 100,
        batch_size: int = 3,
        batch_wait_seconds: float = 0.5,
        max_results: int = 200,
        min_request_interval_seconds: float = 30.0,
        failure_cooldown_seconds: float = 90.0,
    ) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be at least 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if batch_wait_seconds < 0:
            raise ValueError("batch_wait_seconds cannot be negative.")
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")
        if min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds cannot be negative.")
        if failure_cooldown_seconds < 0:
            raise ValueError("failure_cooldown_seconds cannot be negative.")
        if not callable(context_provider):
            raise TypeError("context_provider must be callable.")

        self._context_provider = context_provider
        self._assistant = assistant
        self._queue: queue.Queue[SceneEvent] = queue.Queue(
            maxsize=queue_capacity
        )

        self._batch_size = batch_size
        self._batch_wait_seconds = batch_wait_seconds
        self._min_request_interval = min_request_interval_seconds
        self._failure_cooldown = failure_cooldown_seconds

        self._results: deque[ReasoningResult] = deque(maxlen=max_results)
        self._results_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._shutdown_deadline: Optional[float] = None
        self._last_request_started: Optional[float] = None
        self._cooldown_until = 0.0

        self.accepted_events = 0
        self.dropped_events = 0
        self.processed_batches = 0

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None:
                raise RuntimeError(
                    "Worker has already been started or stopped."
                )

            self._thread = threading.Thread(
                target=self._run,
                name="ccsm-event-reasoner",
                daemon=True,
            )
            self._thread.start()

        logger.info("CCSM event reasoning worker started.")

    def submit_event(self, event: SceneEvent) -> bool:
        with self._lifecycle_lock:
            if (
                self._thread is None
                or not self._thread.is_alive()
                or self._stop_event.is_set()
            ):
                return False

            try:
                self._queue.put_nowait(event)
            except queue.Full:
                with self._counter_lock:
                    self.dropped_events += 1
                logger.warning(
                    "Reasoning queue full; dropping event_id=%s.",
                    event.event_id,
                )
                return False

            with self._counter_lock:
                self.accepted_events += 1

            return True

    def get_results(
        self,
        limit: Optional[int] = None,
    ) -> list[ReasoningResult]:
        if limit is not None and limit < 0:
            raise ValueError("limit must not be negative.")

        with self._results_lock:
            results = list(self._results)

        if limit is None:
            return results

        return results[-limit:] if limit else []

    def stop(self, timeout: Optional[float] = 30.0) -> None:
        """Stop accepting events and wait up to timeout seconds.

        A running HTTP call cannot be forcibly cancelled by this worker.
        If the deadline expires, queued events are discarded and counted.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must not be negative.")

        with self._lifecycle_lock:
            thread = self._thread

            if thread is None:
                self._stop_event.set()
                return

            if thread is threading.current_thread():
                logger.warning("Worker cannot join itself.")
                return

            if not self._stop_event.is_set():
                self._shutdown_deadline = (
                    None if timeout is None
                    else time.monotonic() + timeout
                )
                self._stop_event.set()

            deadline = self._shutdown_deadline

        if deadline is None:
            thread.join()
        else:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

        if thread.is_alive():
            logger.warning(
                "Reasoning worker is still inside an active operation. "
                "The daemon thread may finish after main.py exits."
            )
        else:
            logger.info("CCSM event reasoning worker stopped.")

    def _shutdown_expired(self) -> bool:
        if not self._stop_event.is_set():
            return False

        deadline = self._shutdown_deadline
        return deadline is not None and time.monotonic() >= deadline

    def _discard_pending_events(self) -> None:
        discarded = 0

        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
                discarded += 1

        if discarded:
            with self._counter_lock:
                self.dropped_events += discarded

            logger.warning(
                "Discarded %d queued event(s) during shutdown.",
                discarded,
            )

    def _get_assistant(self) -> Any:
        if self._assistant is None:
            self._assistant = GroqSceneAssistant()
        return self._assistant

    def _wait_for_request_slot(self) -> bool:
        while True:
            now = time.monotonic()

            if self._shutdown_expired():
                return False

            allowed_at = self._cooldown_until

            if self._last_request_started is not None:
                allowed_at = max(
                    allowed_at,
                    self._last_request_started
                    + self._min_request_interval,
                )

            remaining = allowed_at - now
            if remaining <= 0:
                return not self._shutdown_expired()

            wait_seconds = min(remaining, 0.25)

            deadline = self._shutdown_deadline
            if self._stop_event.is_set() and deadline is not None:
                wait_seconds = min(
                    wait_seconds,
                    max(0.0, deadline - now),
                )

            if wait_seconds <= 0:
                return False

            # This wakes immediately when stop() is called.
            self._stop_event.wait(timeout=wait_seconds)

            # If shutdown has started, only proceed if a request slot is
            # already available. Do not wait through cooldown on shutdown.
            if self._stop_event.is_set():
                now = time.monotonic()
                if now < allowed_at:
                    return False

    def _run(self) -> None:
        try:
            while not self._shutdown_expired():
                if self._stop_event.is_set():
                    try:
                        first = self._queue.get_nowait()
                    except queue.Empty:
                        break
                else:
                    try:
                        first = self._queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                batch = [first]

                try:
                    batch_deadline = (
                        time.monotonic() + self._batch_wait_seconds
                    )

                    while len(batch) < self._batch_size:
                        if self._shutdown_expired():
                            break

                        if self._stop_event.is_set():
                            try:
                                event = self._queue.get_nowait()
                            except queue.Empty:
                                break
                        else:
                            remaining = batch_deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            try:
                                event = self._queue.get(timeout=remaining)
                            except queue.Empty:
                                break

                        batch.append(event)

                    if self._shutdown_expired():
                        with self._counter_lock:
                            self.dropped_events += len(batch)
                        continue

                    if not self._wait_for_request_slot():
                        with self._counter_lock:
                            self.dropped_events += len(batch)
                        continue

                    if self._shutdown_expired():
                        with self._counter_lock:
                            self.dropped_events += len(batch)
                        continue

                    self._last_request_started = time.monotonic()
                    self._process_batch(batch)

                finally:
                    for _ in batch:
                        self._queue.task_done()

        finally:
            self._discard_pending_events()
            logger.debug("Reasoning worker loop exited.")

    def _process_batch(self, events: Sequence[SceneEvent]) -> None:
        event_ids = tuple(event.event_id for event in events)

        try:
            context = self._context_provider()
            if not isinstance(context, Mapping):
                raise TypeError(
                    "context_provider must return a mapping."
                )

            answer = self._get_assistant().answer(
                self._build_question(events),
                context,
            )

            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Assistant returned an empty answer.")

            result = ReasoningResult(
                event_ids=event_ids,
                timestamp=datetime.now(timezone.utc).isoformat(),
                answer=answer.strip(),
                error=None,
            )

        except Exception as exc:
            logger.warning(
                "Reasoning failed for event IDs %s (%s).",
                event_ids,
                type(exc).__name__,
            )
            self._cooldown_until = (
                time.monotonic() + self._failure_cooldown
            )

            result = ReasoningResult(
                event_ids=event_ids,
                timestamp=datetime.now(timezone.utc).isoformat(),
                answer=None,
                error=type(exc).__name__,
            )

        with self._results_lock:
            self._results.append(result)

        with self._counter_lock:
            self.processed_batches += 1

    @staticmethod
    def _build_question(events: Sequence[SceneEvent]) -> str:
        lines = [
            "Analyze this CCSM lifecycle event batch using the supplied context.",
            "The context may include current scene data and persistent history.",
            "Treat the event batch as historical observations at their recorded frames.",
            "Use the current snapshot to decide what is visible now.",
            "Use episodic history to describe previously recorded transitions.",
            "Do not infer that an old track ID is the same physical object "
            "across different sessions.",
            "Do not invent events, identities, or unobserved actions.",
            "If evidence is insufficient, say so.",
            "",
            "EVENT BATCH:",
        ]

        for event in events:
            lines.append(
                f"event_id={event.event_id}; "
                f"frame={event.frame}; "
                f"track_id={event.track_id}; "
                f"class={event.class_name}; "
                f"transition={event.old_state}->{event.new_state}; "
                f"missing_frames={event.missing_frames}; "
                f"occluder_id={event.occluder_id}; "
                f"stability_score={event.stability_score:.3f}; "
                f"reason={event.reason}"
            )

        lines.extend(
            [
                "",
                "Summarize only transitions supported by the records.",
                "Then explain what the current snapshot supports.",
                "Keep historical facts separate from current visibility.",
            ]
        )

        return "\n".join(lines)
