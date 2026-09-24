import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from config.settings import MemoryConfig
from core.memory_manager import MemoryManager, _iou, _predict_position
from core.memory_models import ObjectState, Velocity

def _cfg(occ=3, lost=10, alpha=1.0, iou=0.3):
    return MemoryConfig(occlusion_threshold=occ, lost_timeout=lost,
                        velocity_ema_alpha=alpha, occlusion_iou_threshold=iou)

def _obj(tid=1, cx=100.0, cy=200.0, w=50.0, h=100.0, cls_name="person", conf=0.90):
    x1,y1,x2,y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
    return {"track_id":tid, "class_id":0, "class_name":cls_name,
            "confidence":conf, "bbox":[x1,y1,x2,y2], "center":(cx,cy)}

class TestHelpers:
    def test_iou_none(self): assert _iou((0,0,10,10),(20,20,30,30))==pytest.approx(0.0)
    def test_iou_same(self): assert _iou((0,0,10,10),(0,0,10,10))==pytest.approx(1.0)
    def test_iou_partial(self): v=_iou((0,0,10,10),(5,5,15,15)); assert 0.0<v<1.0
    def test_predict_still(self):
        assert _predict_position((100.0,200.0),Velocity(0,0),5)==pytest.approx((100.0,200.0))
    def test_predict_moving(self):
        assert _predict_position((100.0,200.0),Velocity(5,0),3)==pytest.approx((115.0,200.0))

class TestNewObject:
    def test_created(self):
        mm=MemoryManager(config=_cfg()); mm.update([_obj(1)],0)
        assert mm.get(1) is not None
    def test_state_new(self):
        mm=MemoryManager(config=_cfg()); mm.update([_obj(1)],0)
        assert mm.get(1).state==ObjectState.NEW
    def test_position(self):
        mm=MemoryManager(config=_cfg()); mm.update([_obj(1,cx=300.0,cy=400.0)],0)
        assert mm.get(1).last_position==pytest.approx((300.0,400.0))
    def test_two_objects(self):
        mm=MemoryManager(config=_cfg()); mm.update([_obj(1),_obj(2,cx=300.0)],0)
        assert mm.get(1) and mm.get(2)

class TestUpdate:
    def test_becomes_visible(self):
        mm=MemoryManager(config=_cfg())
        mm.update([_obj(1,cx=100.0)],0); mm.update([_obj(1,cx=110.0)],1)
        assert mm.get(1).state==ObjectState.VISIBLE
    def test_pos_updated(self):
        mm=MemoryManager(config=_cfg())
        mm.update([_obj(1,cx=100.0)],0); mm.update([_obj(1,cx=120.0,cy=200.0)],1)
        assert mm.get(1).last_position==pytest.approx((120.0,200.0))
    def test_velocity(self):
        mm=MemoryManager(config=_cfg(alpha=1.0))
        mm.update([_obj(1,cx=100.0,cy=100.0)],0); mm.update([_obj(1,cx=110.0,cy=105.0)],1)
        assert mm.get(1).velocity.vx==pytest.approx(10.0)
        assert mm.get(1).velocity.vy==pytest.approx(5.0)
    def test_trajectory_grows(self):
        mm=MemoryManager(config=_cfg())
        for f in range(5): mm.update([_obj(1,cx=float(f*10))],f)
        assert len(mm.get(1).trajectory)==5

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
    def test_predicted_pos(self):
        mm=MemoryManager(config=_cfg(occ=2,alpha=1.0))
        mm.update([_obj(1,cx=100.0,cy=100.0)],0); mm.update([_obj(1,cx=110.0,cy=100.0)],1)
        for f in range(2,5): mm.update([],f)
        pred=mm.get(1).predicted_position
        assert pred is not None and pred[0]==pytest.approx(140.0,abs=1.0)

class TestReappearance:
    def _sc(self):
        mm=MemoryManager(config=_cfg(occ=2))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,5): mm.update([],f)
        mm.update([_obj(1)],5); return mm
    def test_reappeared(self): assert self._sc().get(1).state==ObjectState.REAPPEARED
    def test_count(self): assert self._sc().get(1).reappearance_count==1
    def test_miss_reset(self): assert self._sc().get(1).missing_frames==0
    def test_becomes_visible(self):
        mm=self._sc(); mm.update([_obj(1)],6)
        assert mm.get(1).state==ObjectState.VISIBLE

class TestPurge:
    def test_removes_lost(self):
        mm=MemoryManager(config=_cfg(occ=2,lost=5))
        mm.update([_obj(1)],0); mm.update([_obj(1)],1)
        for f in range(2,10): mm.update([],f)
        assert mm.purge_lost()==1 and mm.get(1) is None
    def test_zero_when_none(self):
        mm=MemoryManager(config=_cfg()); mm.update([_obj(1)],0)
        assert mm.purge_lost()==0

class TestOccluderAware:
    def test_occluder_id_set(self):
        mm=MemoryManager(config=_cfg(occ=5,iou=0.1))
        mm.update([_obj(1,cx=100.0),_obj(2,cx=100.0)],0)
        mm.update([_obj(1,cx=100.0),_obj(2,cx=100.0)],1)
        mm.update([_obj(2,cx=100.0)],2)
        m1=mm.get(1)
        assert m1 is not None and m1.occluder_id==2 and m1.state==ObjectState.OCCLUDED

if __name__=="__main__":
    pytest.main([__file__,"-v"])
