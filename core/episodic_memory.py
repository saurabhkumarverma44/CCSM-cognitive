
"""Persistent episodic scene memory backed by SQLite."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

_LOG = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert common Python values to JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _event_dict(event: Any) -> dict[str, Any]:
    """Normalize a SceneEvent-like object to a dictionary."""
    if isinstance(event, dict):
        result = dict(event)
    elif is_dataclass(event):
        result = asdict(event)
    elif hasattr(event, "__dict__"):
        result = vars(event).copy()
    else:
        raise TypeError("Event must be a dictionary or an object with fields.")

    return {str(key): _json_safe(value) for key, value in result.items()}


class EpisodicMemory:
    """Store and retrieve scene events across CCSM sessions.

    Each CCSM run gets a session ID by default. Reusing a session ID when
    reopening the database allows that session's events to be queried again.
    Event IDs are unique within a session.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        database_path: str | Path = "data/ccsm_memory.sqlite3",
        session_id: str | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())
        self._lock = threading.RLock()
        self._closed = False

        self._connection = sqlite3.connect(
            str(self.database_path),
            timeout=10.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row

        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 10000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize_schema()

        _LOG.info(
            "Episodic memory ready: database=%s session=%s",
            self.database_path,
            self.session_id,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("EpisodicMemory is closed.")

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    timestamp TEXT,
                    frame INTEGER,
                    track_id INTEGER,
                    class_name TEXT,
                    old_state TEXT,
                    new_state TEXT,
                    missing_frames INTEGER,
                    occluder_id INTEGER,
                    stability_score REAL,
                    reason TEXT,
                    event_json TEXT NOT NULL,
                    UNIQUE(session_id, event_id)
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scene_events_session_time
                ON scene_events(session_id, timestamp, id)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scene_events_track
                ON scene_events(session_id, track_id, id)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scene_events_class_state
                ON scene_events(session_id, class_name, new_state, id)
                """
            )
            self._connection.execute(
                """
                INSERT INTO schema_metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(self.SCHEMA_VERSION),),
            )

    @staticmethod
    def _normalize_event(event: Any) -> dict[str, Any]:
        data = _event_dict(event)

        event_id = data.get("event_id")
        if event_id is None or str(event_id).strip() == "":
            raise ValueError("Event must contain a non-empty event_id.")

        data["event_id"] = str(event_id)

        for key in ("track_id", "occluder_id", "frame", "missing_frames"):
            if data.get(key) is not None:
                data[key] = int(data[key])

        if data.get("stability_score") is not None:
            data["stability_score"] = float(data["stability_score"])

        return data

    def record_event(self, event: Any) -> bool:
        """Insert an event; return False if its session/event ID already exists."""
        data = self._normalize_event(event)

        values = (
            self.session_id,
            data["event_id"],
            data.get("timestamp"),
            data.get("frame"),
            data.get("track_id"),
            data.get("class_name"),
            data.get("old_state"),
            data.get("new_state"),
            data.get("missing_frames"),
            data.get("occluder_id"),
            data.get("stability_score"),
            data.get("reason"),
            json.dumps(data, ensure_ascii=False, sort_keys=True),
        )

        with self._lock:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO scene_events (
                    session_id, event_id, timestamp, frame, track_id,
                    class_name, old_state, new_state, missing_frames,
                    occluder_id, stability_score, reason, event_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            self._connection.commit()
            inserted = cursor.rowcount == 1

        if inserted:
            _LOG.debug(
                "Stored event %s for track %s",
                data["event_id"],
                data.get("track_id"),
            )
        return inserted

    def record_event_safely(self, event: Any) -> bool:
        """Listener-safe wrapper: log a bad event instead of crashing perception."""
        try:
            return self.record_event(event)
        except (sqlite3.Error, TypeError, ValueError, OverflowError, RuntimeError):
            _LOG.exception("Could not persist scene event")
            return False

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
        try:
            data = json.loads(row["event_json"])
        except (TypeError, json.JSONDecodeError):
            data = {}

        data["_database_id"] = row["id"]
        data["_session_id"] = row["session_id"]
        return data

    def recent_events(
        self,
        limit: int = 20,
        current_session_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Return recent events in chronological order (oldest first)."""
        limit = max(1, min(int(limit), 1000))

        with self._lock:
            self._ensure_open()
            if current_session_only:
                rows = self._connection.execute(
                    """
                    SELECT * FROM scene_events
                    WHERE session_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (self.session_id, limit),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """
                    SELECT * FROM scene_events
                    ORDER BY id DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [self._row_to_event(row) for row in reversed(rows)]

    def track_timeline(
        self,
        track_id: int,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return one track's event timeline in chronological order."""
        limit = max(1, min(int(limit), 1000))
        selected_session = session_id or self.session_id

        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT * FROM scene_events
                WHERE session_id = ? AND track_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (selected_session, int(track_id), limit),
            ).fetchall()

        return [self._row_to_event(row) for row in reversed(rows)]

    def search_events(
        self,
        track_id: int | None = None,
        class_name: str | None = None,
        new_state: str | None = None,
        limit: int = 20,
        current_session_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Search events using optional track, class and state filters."""
        limit = max(1, min(int(limit), 1000))
        clauses: list[str] = []
        parameters: list[Any] = []

        if current_session_only:
            clauses.append("session_id = ?")
            parameters.append(self.session_id)
        if track_id is not None:
            clauses.append("track_id = ?")
            parameters.append(int(track_id))
        if class_name is not None:
            clauses.append("class_name = ?")
            parameters.append(str(class_name))
        if new_state is not None:
            clauses.append("new_state = ?")
            parameters.append(str(new_state))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)

        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                f"""
                SELECT * FROM scene_events
                {where_sql}
                ORDER BY id DESC LIMIT ?
                """,
                parameters,
            ).fetchall()

        return [self._row_to_event(row) for row in reversed(rows)]

    def build_history_context(
        self,
        current_frame: int | None = None,
        recent_limit: int = 12,
        timeline_track_ids: Iterable[int] | None = None,
        timeline_limit: int = 10,
    ) -> dict[str, Any]:
        """Build a compact history object, explicitly separate from live state."""
        recent = self.recent_events(
            limit=recent_limit,
            current_session_only=True,
        )

        timelines: dict[str, list[dict[str, Any]]] = {}
        for track_id in dict.fromkeys(
            int(value) for value in (timeline_track_ids or [])
        ):
            timelines[str(track_id)] = self.track_timeline(
                track_id=track_id,
                limit=timeline_limit,
            )

        return {
            "source": "persistent_episodic_memory",
            "session_id": self.session_id,
            "current_frame": current_frame,
            "recent_events": recent,
            "track_timelines": timelines,
            "warning": (
                "These are historical event records, not proof that an object "
                "is currently visible. Use the current scene snapshot for "
                "present visibility."
            ),
        }

    def event_count(self, current_session_only: bool = True) -> int:
        with self._lock:
            self._ensure_open()
            if current_session_only:
                row = self._connection.execute(
                    "SELECT COUNT(*) FROM scene_events WHERE session_id = ?",
                    (self.session_id,),
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT COUNT(*) FROM scene_events"
                ).fetchone()
        return int(row[0])

    def export_json(
        self,
        output_path: str | Path,
        current_session_only: bool = True,
    ) -> Path:
        """Export stored events as JSON and return the output path."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        events = self.recent_events(
            limit=1000,
            current_session_only=current_session_only,
        )

        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "current_session_only": current_session_only,
            "event_count_returned": len(events),
            "events": events,
        }

        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return destination

    def close(self) -> None:
        """Close the SQLite connection; safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "EpisodicMemory":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
