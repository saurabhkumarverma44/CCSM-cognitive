# tests/test_optimizer_and_context.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.memory_models import ObjectState, TrackedObjectMemory, TrajectoryPoint, Velocity
from core.memory_optimizer import MemoryOptimizer, TrajectoryWaypoint
from core.context_builder import ContextBuilder, _direction_str

def _make_dummy_memory(tid=1, cname="person", cx=200.0, cy=300.0, vx=4.0, vy=0.0):
    mem = TrackedObjectMemory(
        track_id=tid, class_id=0, class_name=cname,
        first_seen_frame=0, last_seen_frame=60, current_frame=60,
        state=ObjectState.VISIBLE,
        last_position=(cx, cy),
        last_confidence=0.91,
        velocity=Velocity(vx=vx, vy=vy),
        confirmed=True,
    )
    # Add a linear trajectory
    for f in range(30):
        mem.trajectory.append(TrajectoryPoint(
            frame=f, timestamp=f * 0.033,
            cx=cx + f * vx, cy=cy + f * vy,
            bbox=(cx-30, cy-60, cx+30, cy+60), confidence=0.9,
        ))
    return mem


class TestMemoryOptimizer:

    def test_compress_linear_trajectory(self):
        optimizer = MemoryOptimizer()
        mem = _make_dummy_memory()
        compressed = optimizer.compress_trajectory(mem.trajectory)
        # Linear points should be heavily decimated, keeping endpoints
        assert len(compressed) < len(mem.trajectory)
        assert len(compressed) >= 2
        assert compressed[0].frame == 0
        assert compressed[-1].frame == 29

    def test_sharp_turn_preserved(self):
        optimizer = MemoryOptimizer()
        # Trajectory moving right then making a 90 degree turn down at frame 15
        pts = []
        for f in range(15):
            pts.append(TrajectoryPoint(frame=f, timestamp=f*0.03, cx=100.0 + f * 5.0, cy=200.0,
                                       bbox=(0,0,10,10), confidence=0.9))
        for f in range(15, 30):
            pts.append(TrajectoryPoint(frame=f, timestamp=f*0.03, cx=175.0, cy=200.0 + (f-15) * 5.0,
                                       bbox=(0,0,10,10), confidence=0.9))

        compressed = optimizer.compress_trajectory(pts)
        frames = [w.frame for w in compressed]
        # Frame 14 or 15 or 16 must be retained as the corner vertex
        assert any(f in (14, 15, 16) for f in frames)

    def test_summarize_compression(self):
        optimizer = MemoryOptimizer()
        m1 = _make_dummy_memory(tid=1)
        m2 = _make_dummy_memory(tid=2)
        summary = optimizer.summarize_compression([m1, m2])
        assert summary.total_objects == 2
        assert summary.raw_trajectory_points == 60
        assert summary.compressed_waypoints < 60
        assert summary.compression_ratio > 1.0


class TestContextBuilder:

    def test_direction_strings(self):
        assert _direction_str(0.0, 0.0) == "stationary"
        assert "right" in _direction_str(5.0, 0.0)
        assert "left" in _direction_str(-5.0, 0.0)
        assert "down" in _direction_str(0.0, 5.0)
        assert "up" in _direction_str(0.0, -5.0)
        assert "down-right" in _direction_str(5.0, 5.0)

    def test_build_object_narrative(self):
        cb = ContextBuilder()
        mem = _make_dummy_memory(tid=7, cname="backpack", cx=150.0, cy=250.0)
        mem.reappearance_count = 1
        mem.last_occlusion_duration = 12
        narrative = cb.build_object_narrative(mem)
        assert "Backpack (ID 7):" in narrative
        assert "VISIBLE" in narrative
        assert "Reappeared 1 time(s)" in narrative
        assert "12 frames" in narrative

    def test_build_scene_narrative_and_payload(self):
        cb = ContextBuilder()
        m1 = _make_dummy_memory(tid=1, cname="person")
        m2 = _make_dummy_memory(tid=2, cname="bottle")
        m2.state = ObjectState.OCCLUDED
        m2.occluder_id = 1
        m2.missing_frames = 8

        payload = cb.build_llm_payload([m1, m2], frame_index=100)
        assert payload["frame"] == 100
        assert payload["counts"]["visible"] == 1
        assert payload["counts"]["occluded"] == 1
        assert "Bottle (ID 2) is occluded behind ID 1" in payload["prompt_text"]
        assert len(payload["entities"]) == 2
