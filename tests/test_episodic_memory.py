
import json
from dataclasses import dataclass
from pathlib import Path

from core.episodic_memory import EpisodicMemory


@dataclass
class FakeEvent:
    event_id: str
    timestamp: str
    frame: int
    track_id: int
    class_name: str
    old_state: str
    new_state: str
    missing_frames: int = 0
    occluder_id: int | None = None
    stability_score: float = 0.95
    reason: str = "test event"


def make_event(
    event_id: str = "event-1",
    track_id: int = 1,
    class_name: str = "person",
    new_state: str = "OCCLUDED",
    frame: int = 10,
) -> FakeEvent:
    return FakeEvent(
        event_id=event_id,
        timestamp=f"2026-10-17T10:00:{frame:02d}+00:00",
        frame=frame,
        track_id=track_id,
        class_name=class_name,
        old_state="VISIBLE",
        new_state=new_state,
    )


def test_record_and_retrieve_event(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite3"
    with EpisodicMemory(db, session_id="session-a") as memory:
        assert memory.record_event(make_event()) is True
        assert memory.event_count() == 1

        events = memory.recent_events()
        assert len(events) == 1
        assert events[0]["event_id"] == "event-1"
        assert events[0]["track_id"] == 1
        assert events[0]["new_state"] == "OCCLUDED"


def test_duplicate_event_is_not_inserted_twice(tmp_path: Path) -> None:
    with EpisodicMemory(
        tmp_path / "memory.sqlite3",
        session_id="session-a",
    ) as memory:
        assert memory.record_event(make_event("same-id")) is True
        assert memory.record_event(make_event("same-id")) is False
        assert memory.event_count() == 1


def test_events_survive_reopening_database(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite3"

    with EpisodicMemory(db, session_id="persistent-session") as memory:
        memory.record_event(make_event("persisted"))

    with EpisodicMemory(db, session_id="persistent-session") as memory:
        events = memory.recent_events()
        assert len(events) == 1
        assert events[0]["event_id"] == "persisted"


def test_track_timeline_is_filtered_and_chronological(
    tmp_path: Path,
) -> None:
    with EpisodicMemory(
        tmp_path / "memory.sqlite3",
        session_id="session-a",
    ) as memory:
        memory.record_event(make_event("e1", track_id=7, frame=10))
        memory.record_event(make_event("e2", track_id=8, frame=11))
        memory.record_event(make_event("e3", track_id=7, frame=12))

        timeline = memory.track_timeline(track_id=7)

        assert [event["event_id"] for event in timeline] == ["e1", "e3"]


def test_search_by_class_and_state(tmp_path: Path) -> None:
    with EpisodicMemory(
        tmp_path / "memory.sqlite3",
        session_id="session-a",
    ) as memory:
        memory.record_event(
            make_event("person-hidden", class_name="person", new_state="OCCLUDED")
        )
        memory.record_event(
            make_event("car-visible", class_name="car", new_state="VISIBLE")
        )

        results = memory.search_events(
            class_name="person",
            new_state="OCCLUDED",
        )

        assert len(results) == 1
        assert results[0]["event_id"] == "person-hidden"


def test_history_context_includes_recent_events_and_timelines(
    tmp_path: Path,
) -> None:
    with EpisodicMemory(
        tmp_path / "memory.sqlite3",
        session_id="session-a",
    ) as memory:
        memory.record_event(make_event("track-1-a", track_id=1))
        memory.record_event(make_event("track-2-a", track_id=2))

        context = memory.build_history_context(
            current_frame=100,
            recent_limit=10,
            timeline_track_ids=[1],
        )

        assert context["source"] == "persistent_episodic_memory"
        assert context["current_frame"] == 100
        assert len(context["recent_events"]) == 2
        assert "1" in context["track_timelines"]
        assert len(context["track_timelines"]["1"]) == 1
        assert "not proof" in context["warning"]


def test_session_isolation_and_cross_session_retrieval(
    tmp_path: Path,
) -> None:
    db = tmp_path / "memory.sqlite3"

    with EpisodicMemory(db, session_id="session-a") as memory:
        memory.record_event(make_event("event-a"))

    with EpisodicMemory(db, session_id="session-b") as memory:
        memory.record_event(make_event("event-b"))

        assert memory.event_count(current_session_only=True) == 1
        assert memory.event_count(current_session_only=False) == 2
        assert [e["event_id"] for e in memory.recent_events()] == ["event-b"]

        all_events = memory.recent_events(current_session_only=False)
        assert {e["event_id"] for e in all_events} == {"event-a", "event-b"}


def test_json_export(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite3"
    output = tmp_path / "exports" / "events.json"

    with EpisodicMemory(db, session_id="session-a") as memory:
        memory.record_event(make_event("export-me"))
        exported_path = memory.export_json(output)

    assert exported_path.exists()
    payload = json.loads(exported_path.read_text(encoding="utf-8"))
    assert payload["event_count_returned"] == 1
    assert payload["events"][0]["event_id"] == "export-me"


def test_invalid_event_is_rejected_safely(tmp_path: Path) -> None:
    with EpisodicMemory(
        tmp_path / "memory.sqlite3",
        session_id="session-a",
    ) as memory:
        assert memory.record_event_safely({"track_id": 1}) is False
        assert memory.event_count() == 0


def test_closed_memory_rejects_operations(tmp_path: Path) -> None:
    memory = EpisodicMemory(
        tmp_path / "memory.sqlite3",
        session_id="session-a",
    )
    memory.close()
    memory.close()

    try:
        memory.event_count()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError after closing memory")
