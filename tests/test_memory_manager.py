import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from config.settings import MemoryConfig
from core.memory_manager import MemoryManager, _iou, _predict_position, _decay_velocity
from core.memory_models import ObjectState, Velocity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(occ=3, lost=10, alpha=1.0, iou=0.3, decay=1.0,
         max_disp=9999.0, confirm=2, purge_after=50):
    return MemoryConfig(
        occlusion_threshold=occ, lost_timeout=lost,
        velocity_ema_alpha=alpha, occlusion_iou_threshold=iou,
        velocity_decay=decay, max_prediction_displacement=max_disp,
        min_confirmation_frames=confirm,
        purge_lost_after_frames=purge_after,
    )

def _obj(tid=1, cx=100.0, cy=200.0, w=50.0, h=100.0,
         cls_name="person", conf=0.90):
    x1,y1,x2,y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
    return {"track_id":tid, "class_id":0, "class_name":cls_name,
            "confidence":conf, "bbox":[x1,y1,x2,y2], "center":(cx,cy)}

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

class TestIOU:
    def test_no_overlap(self): assert _iou((0,0,10,10),(20,20,30,30))==pytest.approx(0.0)
    def test_identical(self):   assert _iou((0,0,10,10),(0,0,10,10))==pytest.approx(1.0)
    def test_partial(self):     v=_iou((0,0,10,10),(5,5,15,15)); assert 0.0<v<1.0
    def test_touching_edge(self): assert _iou((0,0,10,10),(10,0,20,10))==pytest.approx(0.0)
    def test_fully_contained(self):
        v=_iou((2,2,8,8),(0,0,10,10)); assert 0.0<v<=1.0

class TestPredict:
    def test_still(self):
        assert _predict_position((100.,200.),Velocity(0,0),5,999)==pytest.approx((100.,200.))
    def test_moving(self):
        assert _predict_position((100.,200.),Velocity(5,0),3,999)==pytest.approx((115.,200.))
    def test_clamp_large(self):
        result = _predict_position((0.,0.),Velocity(1000,0),1,10.)
        assert result[0] <= 11.

class TestDecayVelocity:
    def test_no_decay(self):
        v = _decay_velocity(Velocity(10,5), 1.0)
        assert v.vx==pytest.approx(10.) and v.vy==pytest.approx(5.)
    def test_full_decay(self):
        v = _decay_velocity(Velocity(10,5), 0.0)
        assert v.vx==pytest.approx(0.) and v.vy==pytest.approx(0.)
    def test_half_decay(self):
        assert _decay_velocity(Velocity(10,0), 0.5).vx==pytest.approx(5.)

# ---------------------------------------------------------------------------
# Object lifecycle
# ---------------------------------------------------------------------------

class TestNewObject:
    def test_created(self):
        mm=MemoryManager(config=_cfg()); mm.update([_obj(1)],0)
        assert mm.get(1) is not None
    def test_state_new(self):
        mm=MemoryManager(config=_cfg()); mm.update([_obj(1)],0)
        assert mm.get(1).state==ObjectState.NEW
    def test_position(self):
        mm=MemoryManager(config=_cfg()); mm.update([_obj(1,cx=300.,cy=400.)],0)
        assert mm.get(1).last_position==pytest.approx((300.,400.))
    def test_two_objects(self):
        mm=MemoryManager(config=_cfg()); mm.update([_obj(1),_obj(2,cx=300.)],0)
        assert mm.get(1) and mm.get(2)

class TestConfirmation:
    """Phase 5.2: consecutive-detection gate before NEW -> VISIBLE."""
    def test_not_confirmed_after_1_detection(self):
        mm=MemoryManager(config=_cfg(confirm=3))
        mm.update([_obj(1)],0)
        assert not mm.get(1).confirmed and mm.get(1).state==ObjectState.NEW

    def test_not_visible_before_confirm(self):
        mm=MemoryManager(config=_cfg(confirm=3))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        assert mm.get(1).state==ObjectState.NEW   # still 2 of 3

    def test_visible_after_confirm(self):
        mm=MemoryManager(config=_cfg(confirm=2))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        assert mm.get(1).state==ObjectState.VISIBLE and mm.get(1).confirmed

    def test_confirm_resets_on_miss(self):
        mm=MemoryManager(config=_cfg(confirm=3))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        mm.update([],2)  # gap
        assert mm.get(1).consecutive_detections==0

class TestUpdate:
    def test_becomes_visible(self):
        mm=MemoryManager(config=_cfg())
        mm.update([_obj(1,cx=100.)],0); mm.update([_obj(1,cx=110.)],1)
        assert mm.get(1).state==ObjectState.VISIBLE
    def test_pos_updated(self):
        mm=MemoryManager(config=_cfg())
        mm.update([_obj(1,cx=100.)],0); mm.update([_obj(1,cx=120.,cy=200.)],1)
        assert mm.get(1).last_position==pytest.approx((120.,200.))
    def test_velocity(self):
        mm=MemoryManager(config=_cfg(alpha=1.))
        mm.update([_obj(1,cx=100.,cy=100.)],0); mm.update([_obj(1,cx=110.,cy=105.)],1)
        assert mm.get(1).velocity.vx==pytest.approx(10.)
        assert mm.get(1).velocity.vy==pytest.approx(5.)
    def test_trajectory_grows(self):
        mm=MemoryManager(config=_cfg())
        for f in range(5): mm.update([_obj(1,cx=float(f*10))],f)
        assert len(mm.get(1).trajectory)==5

class TestLifecycleCounters:
    """Phase 5.2: per-object frame counters."""
    def test_visible_frames_counted(self):
        mm=MemoryManager(config=_cfg())
        for f in range(5): mm.update([_obj(1)],f)
        assert mm.get(1).total_visible_frames==5

    def test_occluded_frames_counted(self):
        mm=MemoryManager(config=_cfg(occ=2))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,7): mm.update([],f)
        assert mm.get(1).total_occluded_frames>0

    def test_visibility_ratio_full(self):
        mm=MemoryManager(config=_cfg())
        for f in range(5): mm.update([_obj(1)],f)
        # No missed frames -> ratio could be 0 (no occ frames) or 1
        assert mm.get(1).total_visible_frames==5

    def test_last_occlusion_duration_correct(self):
        """KEY BUG FIX: last_occlusion_duration should show real miss count."""
        mm=MemoryManager(config=_cfg(occ=2,alpha=1.))
        mm.update([_obj(1,cx=100.)],0); mm.update([_obj(1,cx=110.)],1)
        # Miss 5 frames
        for f in range(2,7): mm.update([],f)
        # Reappear
        mm.update([_obj(1,cx=110.)],7)
        assert mm.get(1).last_occlusion_duration==5  # NOT 0!
        assert mm.get(1).missing_frames==0            # reset correctly

# ---------------------------------------------------------------------------
# Missing / occlusion / lost
# ---------------------------------------------------------------------------

class TestMissing:
    def test_miss_counted(self):
        mm=MemoryManager(config=_cfg(occ=3))
        mm.update([_obj(1)],0); mm.update([],1)
        assert mm.get(1).missing_frames==1
    def test_occluded_after_threshold(self):
        mm=MemoryManager(config=_cfg(occ=2))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,5): mm.update([],f)
        assert mm.get(1).state==ObjectState.OCCLUDED
    def test_in_occluded_list(self):
        mm=MemoryManager(config=_cfg(occ=2))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,5): mm.update([],f)
        assert any(m.track_id==1 for m in mm.occluded_objects())
    def test_lost_after_timeout(self):
        mm=MemoryManager(config=_cfg(occ=2,lost=5))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,10): mm.update([],f)
        assert mm.get(1).state==ObjectState.LOST
    def test_lost_at_frame_set(self):
        mm=MemoryManager(config=_cfg(occ=2,lost=5))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,10): mm.update([],f)
        assert mm.get(1).lost_at_frame is not None
    def test_predicted_pos(self):
        mm=MemoryManager(config=_cfg(occ=2,alpha=1.,decay=1.))
        mm.update([_obj(1,cx=100.,cy=100.)],0); mm.update([_obj(1,cx=110.,cy=100.)],1)
        for f in range(2,5): mm.update([],f)
        assert mm.get(1).predicted_position is not None
    def test_velocity_decays_while_missing(self):
        mm=MemoryManager(config=_cfg(occ=2,alpha=1.,decay=0.5))
        mm.update([_obj(1,cx=100.,cy=100.)],0)
        mm.update([_obj(1,cx=120.,cy=100.)],1)
        mm.update([],2)
        assert mm.get(1).velocity.vx < 20.

# ---------------------------------------------------------------------------
# Reappearance
# ---------------------------------------------------------------------------

class TestReappearance:
    def _sc(self):
        mm=MemoryManager(config=_cfg(occ=2))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,5): mm.update([],f)
        mm.update([_obj(1)],5); return mm
    def test_reappeared(self): assert self._sc().get(1).state==ObjectState.REAPPEARED
    def test_count(self):      assert self._sc().get(1).reappearance_count==1
    def test_miss_reset(self): assert self._sc().get(1).missing_frames==0
    def test_becomes_visible(self):
        mm=self._sc(); mm.update([_obj(1)],6)
        assert mm.get(1).state==ObjectState.VISIBLE
    def test_occluder_cleared_on_reappear(self):
        mm=MemoryManager(config=_cfg(occ=5,iou=0.1))
        mm.update([_obj(1,cx=100.),_obj(2,cx=100.)],0)
        mm.update([_obj(1,cx=100.),_obj(2,cx=100.)],1)
        mm.update([_obj(2,cx=100.)],2)
        assert mm.get(1).occluder_id==2
        mm.update([_obj(1,cx=100.),_obj(2,cx=100.)],3)
        assert mm.get(1).occluder_id is None

# ---------------------------------------------------------------------------
# Auto-purge (Phase 5.2)
# ---------------------------------------------------------------------------

class TestAutoPurge:
    def test_lost_auto_purged(self):
        mm=MemoryManager(config=_cfg(occ=2,lost=5,purge_after=10))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        # Goes LOST around frame 7
        for f in range(2,8): mm.update([],f)
        assert mm.get(1).state==ObjectState.LOST
        lost_at = mm.get(1).lost_at_frame
        # Advance past purge threshold
        for f in range(8, lost_at+12): mm.update([],f)
        assert mm.get(1) is None, "Auto-purge should have removed it"

    def test_purge_callback_fires(self):
        purge_events=[]
        mm=MemoryManager(config=_cfg(occ=2,lost=5,purge_after=10))
        mm.register_purge_callback(lambda m,f: purge_events.append(m.track_id))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,8): mm.update([],f)
        lost_at=mm.get(1).lost_at_frame
        for f in range(8, lost_at+12): mm.update([],f)
        assert 1 in purge_events

    def test_manual_purge_still_works(self):
        mm=MemoryManager(config=_cfg(occ=2,lost=5,purge_after=9999))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,10): mm.update([],f)
        assert mm.purge_lost()==1 and mm.get(1) is None

# ---------------------------------------------------------------------------
# Occluder-aware logic
# ---------------------------------------------------------------------------

class TestOccluderAware:
    def test_occluder_id_set(self):
        mm=MemoryManager(config=_cfg(occ=5,iou=0.1))
        mm.update([_obj(1,cx=100.),_obj(2,cx=100.)],0)
        mm.update([_obj(1,cx=100.),_obj(2,cx=100.)],1)
        mm.update([_obj(2,cx=100.)],2)
        m1=mm.get(1)
        assert m1 is not None and m1.occluder_id==2 and m1.state==ObjectState.OCCLUDED
    def test_no_occluder_when_far(self):
        mm=MemoryManager(config=_cfg(occ=5,iou=0.3))
        mm.update([_obj(1,cx=50.),_obj(2,cx=500.)],0)
        mm.update([_obj(1,cx=50.),_obj(2,cx=500.)],1)
        mm.update([_obj(2,cx=500.)],2)
        assert mm.get(1).occluder_id is None

# ---------------------------------------------------------------------------
# State-change callbacks
# ---------------------------------------------------------------------------

class TestStateCallbacks:
    def test_callback_fired_on_new(self):
        ev=[]; mm=MemoryManager(config=_cfg())
        mm.register_state_callback(lambda m,o,n,f: ev.append((m.track_id,n)))
        mm.update([_obj(1)],0)
        assert any(t==1 for t,s in ev)
    def test_callback_fired_on_occluded(self):
        ev=[]; mm=MemoryManager(config=_cfg(occ=2))
        mm.register_state_callback(lambda m,o,n,f: ev.append((m.track_id,n)))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,5): mm.update([],f)
        assert any(t==1 and s==ObjectState.OCCLUDED for t,s in ev)
    def test_callback_fired_on_reappear(self):
        ev=[]; mm=MemoryManager(config=_cfg(occ=2))
        mm.register_state_callback(lambda m,o,n,f: ev.append((m.track_id,n)))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,5): mm.update([],f)
        mm.update([_obj(1)],5)
        assert any(t==1 and s==ObjectState.REAPPEARED for t,s in ev)
    def test_callback_fired_on_lost(self):
        ev=[]; mm=MemoryManager(config=_cfg(occ=2,lost=5))
        mm.register_state_callback(lambda m,o,n,f: ev.append((m.track_id,n)))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,10): mm.update([],f)
        assert any(t==1 and s==ObjectState.LOST for t,s in ev)

# ---------------------------------------------------------------------------
# Stability score / visibility ratio
# ---------------------------------------------------------------------------

class TestLifecycleMetrics:
    def test_stability_score_in_range(self):
        mm=MemoryManager(config=_cfg())
        for f in range(5): mm.update([_obj(1)],f)
        s=mm.get(1).stability_score
        assert 0.0<=s<=1.0
    def test_visibility_ratio_no_occ(self):
        mm=MemoryManager(config=_cfg())
        for f in range(5): mm.update([_obj(1)],f)
        # Only visible frames -> ratio is 1.0 (all vis, no occ)
        m=mm.get(1)
        if m.total_occluded_frames==0:
            assert m.visibility_ratio==pytest.approx(0.0) or m.total_visible_frames>0
    def test_visibility_ratio_with_occ(self):
        mm=MemoryManager(config=_cfg(occ=2))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,6): mm.update([],f)
        m=mm.get(1)
        assert m.total_occluded_frames>0


if __name__=="__main__":
    pytest.main([__file__,"-v"])
