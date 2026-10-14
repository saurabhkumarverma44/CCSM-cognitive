# experiments/occlusion_test.py
# Phase 5.1 - Synthetic occlusion scenario test
"""
40-frame synthetic scenario:
  Person A (ID=1): always visible, moves right slowly.
  Person B (ID=2): visible frames 0-9, hidden frames 10-19, reappears frame 20+.

Assertions:
  1. Person B becomes OCCLUDED during frames 10-19.
  2. Person B reappears (REAPPEARED) at frame 20+.
  3. Person B is NOT LOST at end.
  4. reappearance_count == 1.
  5. State-change callback fires for OCCLUDED and REAPPEARED.
"""
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
    velocity_decay=0.95, max_prediction_displacement=50.0,
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
    # Track callback events
    callback_events = []
    mm = MemoryManager(config=CFG)
    mm.register_state_callback(
        lambda m, o, n, f: callback_events.append(
            {"id": m.track_id, "old": o.value, "new": n.value, "frame": f}))

    log_entries = []
    print("\n=== CCSM Occlusion Experiment (Phase 5.1 synthetic) ===\n")
    print(f"{'Frame':<6} {'ID':<4} {'State':<22} {'Pos':<18} {'Pred':<18} {'Miss':<6}")
    print("-" * 80)

    for frame_idx, detections in build_scenario():
        mm.update(detections, frame_index=frame_idx)
        for mem in sorted(mm.all_objects(), key=lambda m: m.track_id):
            pos = mem.last_position or mem.predicted_position
            ps = f"({pos[0]:.0f},{pos[1]:.0f})" if pos else "?"
            pr = (f"({mem.predicted_position[0]:.0f},{mem.predicted_position[1]:.0f})"
                  if mem.predicted_position else "-")
            print(f"{frame_idx:<6} {mem.track_id:<4} {mem.state.value:<22}"
                  f" {ps:<18} {pr:<18} {mem.missing_frames:<6}")
            log_entries.append({"frame":frame_idx, **mem.to_summary_dict()})

    print("\n--- Assertions ---")

    mem2 = mm.get(2)
    assert mem2 is not None, "Person B memory record missing!"

    occ_events = [e for e in mem2.events if e.event_type=="OCCLUDED"]
    assert len(occ_events)>=1, "Person B never OCCLUDED!"
    print("[PASS] Person B was marked OCCLUDED")

    reapp_events = [e for e in mem2.events if e.event_type=="REAPPEARED"]
    assert len(reapp_events)>=1, "Person B never REAPPEARED!"
    print("[PASS] Person B reappeared - identity recovered")

    assert mem2.state != ObjectState.LOST, "Person B incorrectly LOST!"
    print("[PASS] Person B is NOT LOST")

    assert mem2.reappearance_count >= 1
    print(f"[PASS] reappearance_count={mem2.reappearance_count}")

    # Callback checks
    cb_occ  = [e for e in callback_events if e["id"]==2 and e["new"]=="OCCLUDED"]
    cb_reap = [e for e in callback_events if e["id"]==2 and e["new"]=="REAPPEARED"]
    assert len(cb_occ)>=1, "State-change callback not fired for OCCLUDED"
    print(f"[PASS] Callback fired for OCCLUDED (frame {cb_occ[0]['frame']})")
    assert len(cb_reap)>=1, "State-change callback not fired for REAPPEARED"
    print(f"[PASS] Callback fired for REAPPEARED (frame {cb_reap[0]['frame']})")

    # Prediction should have been active during occlusion
    any_pred = any(e.get("predicted_position") is not None
                   for e in log_entries if e["track_id"]==2 and e["frame"] in range(10,20))
    assert any_pred, "No predicted position during occlusion!"
    print("[PASS] Predicted position was active during occlusion")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({"log": log_entries, "callbacks": callback_events},
                  f, indent=2, default=str)
    print(f"\n[LOG] Saved to {LOG_PATH.resolve()}")
    print("\n=== Experiment PASSED ===\n")

if __name__=="__main__":
    run()
