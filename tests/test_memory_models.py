# tests/test_memory_models.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from collections import deque
import pytest
from core.memory_models import (
    MemoryEvent, ObjectState, TrackedObjectMemory, TrajectoryPoint, Velocity,
)

class TestVelocity:
    def test_speed_zero(self):
        assert Velocity(0.0, 0.0).speed == pytest.approx(0.0)
    def test_speed_345(self):
        assert Velocity(3.0, 4.0).speed == pytest.approx(5.0)
    def test_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            Velocity(1.0, 2.0).vx = 99.0

class TestObjectState:
    def test_all_states(self):
        vals = {s.value for s in ObjectState}
        for s in ("NEW","VISIBLE","PARTIALLY_OCCLUDED","OCCLUDED","REAPPEARED","LOST"):
            assert s in vals
    def test_str_value(self):
        assert ObjectState.OCCLUDED == "OCCLUDED"

class TestTrajectoryPoint:
    def test_creation(self):
        tp = TrajectoryPoint(1, 1.0, 100.0, 200.0, (50.0,150.0,150.0,250.0), 0.92)
        assert tp.cx == 100.0
        assert tp.confidence == pytest.approx(0.92)
    def test_immutable(self):
        tp = TrajectoryPoint(1, 1.0, 10.0, 20.0, (0,0,20,40), 0.9)
        with pytest.raises((AttributeError, TypeError)):
            tp.cx = 999.0

def _mem(track_id=1, cls="person", frame=0):
    return TrackedObjectMemory(
        track_id=track_id, class_id=0, class_name=cls,
        first_seen_frame=frame, last_seen_frame=frame, current_frame=frame,
        state=ObjectState.NEW,
        trajectory=deque(maxlen=300), events=deque(maxlen=50),
    )

class TestTrackedObjectMemory:
    def test_initial(self):
        mem = _mem()
        assert mem.state == ObjectState.NEW
        assert mem.missing_frames == 0
        assert mem.velocity.speed == pytest.approx(0.0)
    def test_is_visible_new(self):
        assert _mem().is_visible is True
    def test_is_missing_occluded(self):
        mem = _mem(); mem.state = ObjectState.OCCLUDED
        assert mem.is_missing is True
    def test_age_frames(self):
        mem = _mem(frame=10); mem.current_frame = 25
        assert mem.age_frames == 15
    def test_summary_keys(self):
        mem = _mem(); mem.last_position = (100.0, 200.0)
        keys = mem.to_summary_dict().keys()
        for k in ("track_id","class_name","state","missing_frames","velocity"):
            assert k in keys
    def test_summary_vals(self):
        mem = _mem(track_id=7, cls="car")
        mem.state = ObjectState.VISIBLE
        s = mem.to_summary_dict()
        assert s["track_id"] == 7 and s["state"] == "VISIBLE"
    def test_repr(self):
        r = repr(_mem(track_id=3))
        assert "id=3" in r and "NEW" in r

class TestMemoryEvent:
    def test_creation(self):
        ev = MemoryEvent(5, 1.5, "OCCLUDED", "test")
        assert ev.event_type == "OCCLUDED"
    def test_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            MemoryEvent(1, 1.0, "NEW").event_type = "LOST"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
