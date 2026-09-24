import sys, os, json
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import MemoryConfig
from core.memory_manager import MemoryManager
from core.memory_models import ObjectState

LOG_PATH = Path("logs/occlusion_experiment.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

CFG = MemoryConfig(
    occlusion_threshold=3, lost_timeout=30,
    velocity_ema_alpha=0.5, occlusion_iou_threshold=0.15,
)

def _bbox(cx, cy, w=60, h=120):
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

def _obj(tid, cx, cy, conf=0.88):
    return {"track_id":tid, "class_id":0, "class_name":"person",
            "confidence":conf, "bbox":_bbox(cx,cy), "center":(float(cx),float(cy))}

def build_scenario():
    frames = []
    for f in range(40):
        det = []
        ax = 200.0 + f * 1.5
        det.append(_obj(1, ax, 300.0))
        if f < 10:
            bx = 200.0 + f * 1.5
            det.append(_obj(2, bx, 300.0))
        elif f >= 20:
            bx = 200.0 + (20*1.5) + (f-20)*2.0
            det.append(_obj(2, bx, 300.0))
        frames.append((f, det))
    return frames

def run():
    mm = MemoryManager(config=CFG)
    log_entries = []
    print("\n=== CCSM Occlusion Experiment (synthetic) ===\n")
    print(f"{'Frame':<6} {'ID':<4} {'State':<22} {'Pos':<18} {'Pred':<18} {'Miss':<6}")
    print("-" * 80)
    for frame_idx, detections in build_scenario():
        mm.update(detections, frame_index=frame_idx)
        for mem in sorted(mm.all_objects(), key=lambda m: m.track_id):
            pos = mem.last_position or mem.predicted_position
            ps = f"({pos[0]:.0f},{pos[1]:.0f})" if pos else "?"
            pr = (f"({mem.predicted_position[0]:.0f},{mem.predicted_position[1]:.0f})"
                  if mem.predicted_position else "-")
            print(f"{frame_idx:<6} {mem.track_id:<4} {mem.state.value:<22} {ps:<18} {pr:<18} {mem.missing_frames:<6}")
            log_entries.append({"frame":frame_idx, **mem.to_summary_dict()})
    print("\n--- Assertions ---")
    mem2 = mm.get(2)
    assert mem2 is not None, "Person B memory record missing!"
    occ = [e for e in mem2.events if e.event_type=="OCCLUDED"]
    assert len(occ)>=1, "Person B never OCCLUDED!"
    print("[PASS] Person B was marked OCCLUDED")
    reapp = [e for e in mem2.events if e.event_type=="REAPPEARED"]
    assert len(reapp)>=1, "Person B never REAPPEARED!"
    print("[PASS] Person B reappeared - identity recovered")
    assert mem2.state != ObjectState.LOST, "Person B incorrectly LOST!"
    print("[PASS] Person B is NOT LOST")
    assert mem2.reappearance_count >= 1
    print(f"[PASS] reappearance_count={mem2.reappearance_count}")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2, default=str)
    print(f"\n[LOG] Saved to {LOG_PATH.resolve()}")
    print("\n=== Experiment PASSED ===\n")

if __name__=="__main__":
    run()
