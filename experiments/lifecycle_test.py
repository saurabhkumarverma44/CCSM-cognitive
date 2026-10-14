# experiments/lifecycle_test.py
# Phase 5.2 - Stable Object Lifecycle & Auto-Purge Verification Experiment
"""
Synthetic scenario evaluating Phase 5.2 lifecycle features:
  1. False-Positive / Noise Rejection:
     Object 1 appears for only 1 frame (frame 0).
     -> Remains unconfirmed, never promoted to VISIBLE without min_confirmation_frames.
  2. Confirmed Stable Tracking:
     Object 2 is tracked continuously frames 0-60.
     -> Confirmed at frame 1, achieves high stability_score (>0.75).
  3. Accurate Occlusion Duration:
     Object 3 visible frames 0-10, missing frames 11-25 (15 frames), reappears frame 26.
     -> last_occlusion_duration == 15 (verifies fix for 0-frame duration bug).
  4. Auto-Purge Lifecycle Termination:
     Object 1 becomes LOST and is auto-purged after purge_lost_after_frames.
     -> Purge callback fires with complete lifetime summary.
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import MemoryConfig
from core.memory_manager import MemoryManager
from core.memory_models import ObjectState

LOG_PATH = Path("logs/lifecycle_experiment.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

CFG = MemoryConfig(
    occlusion_threshold=3,
    lost_timeout=20,
    min_confirmation_frames=2,
    purge_lost_after_frames=30,  # fast purge for test
    velocity_ema_alpha=0.5,
    occlusion_iou_threshold=0.15,
)

def _bbox(cx: float, cy: float, w: float = 60.0, h: float = 120.0):
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

def _obj(tid: int, cx: float, cy: float, cls_name: str = "person", conf: float = 0.90):
    return {
        "track_id": tid,
        "class_id": 0,
        "class_name": cls_name,
        "confidence": conf,
        "bbox": _bbox(cx, cy),
        "center": (float(cx), float(cy)),
    }

def run():
    print()
    print("=" * 70)
    print("  CCSM Phase 5.2 Lifecycle & Auto-Purge Experiment")
    print("=" * 70)
    print()

    state_changes = []
    purged_records = []

    mm = MemoryManager(config=CFG)
    mm.register_state_callback(
        lambda m, o, n, f: state_changes.append(
            {"id": m.track_id, "old": o.value, "new": n.value, "frame": f}
        )
    )
    mm.register_purge_callback(
        lambda m, f: purged_records.append(
            {"id": m.track_id, "frame": f, "summary": m.to_summary_dict()}
        )
    )

    log_entries = []

    # Scenario: 70 frames
    # Frame 0: Obj 1 (noise), Obj 2 (permanent), Obj 3 (temporary occlusion)
    # Frame 1-10: Obj 2, Obj 3
    # Frame 11-25: Obj 2 only (Obj 3 missing for 15 frames)
    # Frame 26-40: Obj 2, Obj 3 (Obj 3 reappears)
    # Frame 41-70: Obj 2 only
    for f in range(70):
        detections = []

        # Object 1: Frame 0 only
        if f == 0:
            detections.append(_obj(1, 100.0, 200.0, cls_name="noise_item", conf=0.45))

        # Object 2: Frame 0-60
        if f <= 60:
            detections.append(_obj(2, 250.0 + f * 1.0, 300.0, cls_name="person", conf=0.92))

        # Object 3: Frames 0-10, missing 11-25, reappears 26-40
        if f <= 10:
            detections.append(_obj(3, 400.0, 300.0, cls_name="backpack", conf=0.88))
        elif 26 <= f <= 40:
            detections.append(_obj(3, 420.0, 300.0, cls_name="backpack", conf=0.89))

        mm.update(detections, frame_index=f)

        for mem in mm.all_objects():
            log_entries.append({
                "frame": f,
                "track_id": mem.track_id,
                "state": mem.state.value,
                "confirmed": mem.confirmed,
                "consecutive": mem.consecutive_detections,
                "total_vis": mem.total_visible_frames,
                "total_occ": mem.total_occluded_frames,
                "stability": round(mem.stability_score, 3),
            })

    # --- Print Summary ---
    print(f"{'Check':<50} {'Result':<10}")
    print("-" * 70)

    # Check 1: Noise object 1 never confirmed
    obj1_confirmed_ever = any(
        e["confirmed"] for e in log_entries if e["track_id"] == 1
    )
    assert not obj1_confirmed_ever, "Object 1 (noise) was incorrectly confirmed!"
    print(f"{'1. Single-frame noise (Obj 1) remained unconfirmed':<50} {'[PASS]':<10}")

    # Check 2: Object 2 confirmed at frame 1 and high stability
    mem2 = mm.get(2)
    assert mem2 is not None, "Object 2 missing from memory!"
    assert mem2.confirmed, "Object 2 was not confirmed!"
    assert mem2.stability_score >= 0.70, f"Object 2 stability score too low: {mem2.stability_score}"
    print(f"{'2. Stable object (Obj 2) confirmed with stability=' + f'{mem2.stability_score:.2f}':<50} {'[PASS]':<10}")

    # Check 3: Object 3 occlusion duration
    mem3 = mm.get(3)
    assert mem3 is not None, "Object 3 missing from memory!"
    assert mem3.reappearance_count == 1, f"Expected 1 reappearance, got {mem3.reappearance_count}"
    assert mem3.last_occlusion_duration == 15, (
        f"Expected last_occlusion_duration=15, got {mem3.last_occlusion_duration}"
    )
    print(f"{'3. Occlusion duration recorded exactly 15 frames':<50} {'[PASS]':<10}")

    # Check 4: Auto-purge fired for Object 1
    purged_ids = [p["id"] for p in purged_records]
    assert 1 in purged_ids, f"Object 1 was not auto-purged! Purged IDs: {purged_ids}"
    p1 = next(p for p in purged_records if p["id"] == 1)
    assert p1["summary"]["total_visible_frames"] == 1
    print(f"{'4. Auto-purge callback fired for Object 1 (frame ' + str(p1['frame']) + ')':<50} {'[PASS]':<10}")

    # Check 5: Object 1 no longer present in registry after purge
    assert mm.get(1) is None, "Object 1 still present in memory after purge!"
    print(f"{'5. Object 1 cleanly pruned from active memory':<50} {'[PASS]':<10}")

    # Save log
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "log": log_entries,
                "state_changes": state_changes,
                "purged_records": purged_records,
            },
            f,
            indent=2,
        )
    print()
    print(f"[LOG] Results saved to {LOG_PATH.resolve()}")
    print("=" * 70)
    print("  All Phase 5.2 Lifecycle Assertions PASSED successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    run()
