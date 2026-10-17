
import threading
import time

import pytest

from core.event_manager import SceneEvent
from core.reasoning_worker import EventReasoningWorker


class FakeAssistant:
    def __init__(self, answer="Mock reasoning result"):
        self.answer_text = answer
        self.calls = []
        self.lock = threading.Lock()

    def answer(self, question, scene_context):
        with self.lock:
            self.calls.append((question, scene_context))
        return self.answer_text


def make_event(event_id=1, new_state="OCCLUDED"):
    return SceneEvent(
        event_id=event_id,
        timestamp="2026-10-17T10:00:00+00:00",
        frame=10 + event_id,
        track_id=event_id,
        class_name="person",
        old_state="VISIBLE",
        new_state=new_state,
        missing_frames=3,
        occluder_id=None,
        stability_score=0.9,
        reason="Test lifecycle event.",
    )


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_worker_processes_event_without_real_api():
    assistant = FakeAssistant()
    worker = EventReasoningWorker(
        context_provider=lambda: {"frame": 12, "entities": []},
        assistant=assistant,
        batch_size=1,
        batch_wait_seconds=0,
    )
    worker.start()
    try:
        assert worker.submit_event(make_event())
        assert wait_until(lambda: len(worker.get_results()) == 1)

        result = worker.get_results()[0]
        assert result.answer == "Mock reasoning result"
        assert result.error is None
        assert result.event_ids == (1,)
        assert len(assistant.calls) == 1
    finally:
        worker.stop()


def test_worker_batches_events():
    assistant = FakeAssistant()
    worker = EventReasoningWorker(
        context_provider=lambda: {"frame": 20},
        assistant=assistant,
        batch_size=2,
        batch_wait_seconds=0.25,
    )
    worker.start()
    try:
        assert worker.submit_event(make_event(1))
        assert worker.submit_event(make_event(2))
        assert wait_until(lambda: len(worker.get_results()) == 1)

        result = worker.get_results()[0]
        assert result.event_ids == (1, 2)
        assert len(assistant.calls) == 1
    finally:
        worker.stop()


def test_worker_records_assistant_failure():
    class BrokenAssistant:
        def answer(self, question, scene_context):
            raise RuntimeError("mock failure")

    worker = EventReasoningWorker(
        context_provider=lambda: {"frame": 1},
        assistant=BrokenAssistant(),
        batch_size=1,
        batch_wait_seconds=0,
    )
    worker.start()
    try:
        assert worker.submit_event(make_event())
        assert wait_until(lambda: len(worker.get_results()) == 1)

        result = worker.get_results()[0]
        assert result.answer is None
        assert result.error == "RuntimeError"
    finally:
        worker.stop()


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        EventReasoningWorker(
            context_provider=lambda: {},
            queue_capacity=0,
        )


def test_worker_rejects_events_before_start():
    worker = EventReasoningWorker(
        context_provider=lambda: {},
        assistant=FakeAssistant(),
    )
    assert worker.submit_event(make_event()) is False

def test_worker_drains_queued_events_on_shutdown():
    assistant = FakeAssistant()
    worker = EventReasoningWorker(
        context_provider=lambda: {"frame": 30},
        assistant=assistant,
        batch_size=2,
        batch_wait_seconds=0.25,
        min_request_interval_seconds=0,
        failure_cooldown_seconds=0,
    )

    worker.start()
    assert worker.submit_event(make_event(1))
    assert worker.submit_event(make_event(2))

    worker.stop(timeout=3.0)

    assert not worker.is_running
    assert len(worker.get_results()) == 1
    assert worker.get_results()[0].event_ids == (1, 2)
    assert worker.dropped_events == 0


def test_worker_rejects_events_after_shutdown_begins():
    worker = EventReasoningWorker(
        context_provider=lambda: {"frame": 1},
        assistant=FakeAssistant(),
        batch_size=1,
        batch_wait_seconds=0,
        min_request_interval_seconds=0,
    )

    worker.start()
    worker.stop(timeout=2.0)

    assert worker.submit_event(make_event(99)) is False
