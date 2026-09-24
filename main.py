# main.py
# CCSM Phase 5: Camera -> YOLO12l (GPU) -> ByteTrack -> MemoryManager
# Controls:  q = quit,  m = toggle HUD
import logging, sys
from pathlib import Path
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs") / "ccsm.log", encoding="utf-8"),
    ],
)

from agents.perception_agent import PerceptionAgent
from core.memory_models import ObjectState

STATE_COLOURS = {
    ObjectState.NEW:                (0, 255, 0),
    ObjectState.VISIBLE:            (0, 255, 0),
    ObjectState.PARTIALLY_OCCLUDED: (0, 200, 255),
    ObjectState.OCCLUDED:           (0, 120, 255),
    ObjectState.REAPPEARED:         (255, 200, 0),
    ObjectState.LOST:               (0, 0, 255),
}

def draw_memory_hud(frame, memory_snapshot, frame_index):
    h, w = frame.shape[:2]
    y = 20
    cv2.putText(frame, f"Frame: {frame_index}", (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    y += 22
    cv2.putText(frame, f"Objects in memory: {len(memory_snapshot)}", (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,180,180), 1)
    y += 22
    for mem in sorted(memory_snapshot, key=lambda m: m.track_id):
        colour = STATE_COLOURS.get(mem.state, (255,255,255))
        pos = (f"({mem.last_position[0]:.0f},{mem.last_position[1]:.0f})"
               if mem.last_position else "?")
        vel = f"v=({mem.velocity.vx:.1f},{mem.velocity.vy:.1f})"
        miss = f" miss={mem.missing_frames}" if mem.missing_frames else ""
        line = f"ID={mem.track_id} {mem.class_name[:6]} [{mem.state.value}] {pos} {vel}{miss}"
        cv2.putText(frame, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        y += 18
        if mem.state == ObjectState.OCCLUDED and mem.predicted_position:
            px, py = int(mem.predicted_position[0]), int(mem.predicted_position[1])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px,py), 12, (0,120,255), 2)
                cv2.putText(frame, f"?{mem.track_id}", (px+14,py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,120,255), 1)
    return frame

def print_memory_console(memory_snapshot, frame_index):
    print(f"--- Frame {frame_index} | {len(memory_snapshot)} object(s) ---")
    for mem in sorted(memory_snapshot, key=lambda m: m.track_id):
        pos = mem.last_position or mem.predicted_position
        ps = f"({pos[0]:.0f},{pos[1]:.0f})" if pos else "unknown"
        pred = ""
        if mem.state == ObjectState.OCCLUDED and mem.predicted_position:
            p = mem.predicted_position
            pred = f" | pred=({p[0]:.0f},{p[1]:.0f})"
        occ = f" | occ_by={mem.occluder_id}" if mem.occluder_id else ""
        print(f"  ID={mem.track_id:<3} {mem.class_name:<10} [{mem.state.value:<20}] "
              f"pos={ps:<14} vel=({mem.velocity.vx:+.1f},{mem.velocity.vy:+.1f}) "
              f"miss={mem.missing_frames}{pred}{occ}")

def main():
    agent = PerceptionAgent()
    agent.open()
    show_hud, fi = True, 0
    print("CCSM Phase 5 running.  q=quit  m=toggle HUD")
    try:
        while True:
            tracked, memory_snapshot, frame = agent.process_frame_with_memory()
            fi += 1
            if show_hud:
                draw_memory_hud(frame, memory_snapshot, fi)
            cv2.imshow("CCSM Phase 5 - Memory Layer", frame)
            if fi % 15 == 0:
                print_memory_console(memory_snapshot, fi)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                show_hud = not show_hud
                print(f"HUD: {'ON' if show_hud else 'OFF'}")
    finally:
        agent.close()
        print("CCSM stopped.")

if __name__ == "__main__":
    main()
