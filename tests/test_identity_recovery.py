# tests/test_identity_recovery.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from config.settings import MemoryConfig, IdentityRecoveryConfig
from core.identity_recovery import IdentityRecoveryEngine, RecoveryMatch
from core.memory_manager import MemoryManager
from core.memory_models import ObjectState, TrackedObjectMemory, Velocity

def _make_cand(tid: int, cid: int, cname: str, cx: float, cy: float,
               vx: float = 0.0, vy: float = 0.0, missing: int = 5,
               state: ObjectState = ObjectState.OCCLUDED) -> TrackedObjectMemory:
    m = TrackedObjectMemory(
        track_id=tid, class_id=cid, class_name=cname,
        first_seen_frame=0, last_seen_frame=10, current_frame=10 + missing,
        state=state, missing_frames=missing,
        last_position=(cx, cy),
        predicted_position=(cx + vx * missing, cy + vy * missing),
        last_bbox=(cx - 30, cy - 60, cx + 30, cy + 60),
        last_confidence=0.90,
        velocity=Velocity(vx=vx, vy=vy),
        confirmed=True,
    )
    return m

def _make_det(tid: int, cid: int, cname: str, cx: float, cy: float) -> dict:
    return {
        "track_id": tid, "class_id": cid, "class_name": cname,
        "confidence": 0.88,
        "bbox": [cx - 30, cy - 60, cx + 30, cy + 60],
        "center": (cx, cy),
    }


class TestIdentityRecoveryEngine:

    def test_class_mismatch_fails(self):
        engine = IdentityRecoveryEngine()
        cand = _make_cand(tid=1, cid=0, cname="person", cx=200, cy=300)
        det = _make_det(tid=9, cid=1, cname="bicycle", cx=200, cy=300)
        match = engine.find_match(det, [cand])
        assert match is None

    def test_distance_too_large_fails(self):
        cfg = IdentityRecoveryConfig(max_recovery_distance=100.0)
        engine = IdentityRecoveryEngine(cfg)
        cand = _make_cand(tid=1, cid=0, cname="person", cx=200, cy=300)
        det = _make_det(tid=9, cid=0, cname="person", cx=400, cy=300)  # dist=200
        match = engine.find_match(det, [cand])
        assert match is None

    def test_stationary_match_succeeds(self):
        engine = IdentityRecoveryEngine()
        cand = _make_cand(tid=1, cid=0, cname="person", cx=200, cy=300)
        det = _make_det(tid=9, cid=0, cname="person", cx=210, cy=302)  # dist ~10px
        match = engine.find_match(det, [cand])
        assert match is not None
        assert match.recovered_id == 1
        assert match.score >= 0.70

    def test_moving_object_matches_predicted_position(self):
        engine = IdentityRecoveryEngine()
        # Cand moving at vx=5.0, missing 6 frames -> predicted at cx=230
        cand = _make_cand(tid=2, cid=0, cname="person", cx=200, cy=300, vx=5.0, vy=0.0, missing=6)
        det = _make_det(tid=88, cid=0, cname="person", cx=232, cy=300)
        match = engine.find_match(det, [cand])
        assert match is not None
        assert match.recovered_id == 2
        assert match.distance_px <= 5.0

    def test_already_claimed_id_skipped(self):
        engine = IdentityRecoveryEngine()
        cand = _make_cand(tid=1, cid=0, cname="person", cx=200, cy=300)
        det = _make_det(tid=9, cid=0, cname="person", cx=205, cy=300)
        match = engine.find_match(det, [cand], claimed_ids={1})
        assert match is None


class TestMemoryManagerRecoveryIntegration:

    def test_memory_manager_recovers_reappearing_track_with_new_id(self):
        mm = MemoryManager()
        # Frame 0-5: Person B (tid=2) tracked
        for f in range(6):
            mm.update([
                {"track_id": 1, "class_id": 0, "class_name": "person", "confidence": 0.9,
                 "bbox": [100, 200, 160, 320], "center": (130.0, 260.0)},
                {"track_id": 2, "class_id": 0, "class_name": "person", "confidence": 0.9,
                 "bbox": [200, 200, 260, 320], "center": (230.0, 260.0)},
            ], frame_index=f)

        assert mm.get(2) is not None
        assert mm.get(2).state == ObjectState.VISIBLE

        # Frame 6-12: Person B missing for 7 frames (becomes OCCLUDED)
        for f in range(6, 13):
            mm.update([
                {"track_id": 1, "class_id": 0, "class_name": "person", "confidence": 0.9,
                 "bbox": [100, 200, 160, 320], "center": (130.0, 260.0)},
            ], frame_index=f)

        assert mm.get(2).state == ObjectState.OCCLUDED
        assert mm.get(2).missing_frames == 7

        # Frame 13: Person B reappears, but ByteTrack assigned NEW ID 99 near position (235, 260)
        mm.update([
            {"track_id": 1, "class_id": 0, "class_name": "person", "confidence": 0.9,
             "bbox": [100, 200, 160, 320], "center": (130.0, 260.0)},
            {"track_id": 99, "class_id": 0, "class_name": "person", "confidence": 0.92,
             "bbox": [205, 200, 265, 320], "center": (235.0, 260.0)},
        ], frame_index=13)

        # Identity 2 should have been recovered!
        mem2 = mm.get(2)
        assert mem2 is not None
        assert mem2.state == ObjectState.REAPPEARED
        assert mem2.reappearance_count == 1
        assert mem2.last_occlusion_duration == 7
        # There should NOT be an active new object 99 in registry
        assert mm.get(99) is None

        # Frame 14: Next frame continues with ByteTrack ID 99 -> mapped to 2
        mm.update([
            {"track_id": 1, "class_id": 0, "class_name": "person", "confidence": 0.9,
             "bbox": [100, 200, 160, 320], "center": (130.0, 260.0)},
            {"track_id": 99, "class_id": 0, "class_name": "person", "confidence": 0.92,
             "bbox": [208, 200, 268, 320], "center": (238.0, 260.0)},
        ], frame_index=14)

        assert mem2.state == ObjectState.VISIBLE
        assert mem2.last_position == (238.0, 260.0)
        assert mm.get(99) is None
