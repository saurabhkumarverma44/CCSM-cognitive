# main.py
# CCSM Phase 6 - Spatial-Temporal Memory, Identity Recovery & LLM Context
# Controls:
#   q = quit
#   m = toggle HUD
#   d = toggle debug mode
#   c = print LLM semantic context narrative
#   p = show memory compression stats & purge
import logging
import sys
from pathlib import Path
import cv2

# ------------------------------------------------------------------
# Logging setup MUST happen before any project imports
# ------------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs") / "ccsm.log", encoding="utf-8"),
    ],
)

from agents.perception_agent import PerceptionAgent
from config.settings import CONFIG
from core.context_builder import ContextBuilder
from core.memory_optimizer import MemoryOptimizer
from core.visualizer import annotate_frame
from core.memory_models import ObjectState

_LOG = logging.getLogger("ccsm.main")


def main() -> None:
    agent = PerceptionAgent()
    agent.open()

    context_builder = ContextBuilder(fps=30.0)
    optimizer = MemoryOptimizer()

    show_hud = True
    fi = 0

    print()
    print("=" * 66)
    print("  CCSM Phase 6 — Spatial-Temporal Memory & Identity Recovery")
    print("  q=quit  m=HUD  d=debug  c=LLM context  p=memory stats")
    print("=" * 66)
    print()

    try:
        while True:
            tracked, memory_snapshot, annotated, raw_frame = (
                agent.process_frame_with_memory()
            )
            fi += 1

            # Build the display frame:
            # 1. Start from YOLO-annotated frame (has visible bounding boxes)
            # 2. Add ghost outlines and memory HUD on top
            display = annotate_frame(
                annotated,
                memory_snapshot,
                fi,
                cfg=CONFIG.visualization if show_hud else
                    CONFIG.visualization.__class__(show_hud=False,
                                                   show_ghost=True),
            )

            cv2.imshow("CCSM Phase 6 - Memory + Re-ID + Occlusion", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                show_hud = not show_hud
                print(f"[HUD] {'ON' if show_hud else 'OFF'}")
            elif key == ord("d"):
                agent.event_logger._cfg = (
                    agent.event_logger._cfg.__class__(
                        **{
                            **{
                                f.name: getattr(agent.event_logger._cfg, f.name)
                                for f in agent.event_logger._cfg.__dataclass_fields__.values()
                            },
                            "debug_mode": not agent.event_logger._cfg.debug_mode,
                        }
                    )
                )
                dm = agent.event_logger._cfg.debug_mode
                print(f"[DEBUG] {'ON' if dm else 'OFF'}")
            elif key == ord("c"):
                narrative = context_builder.build_scene_narrative(memory_snapshot, fi)
                print()
                print(narrative)
                print()
            elif key == ord("p"):
                summary = optimizer.summarize_compression(memory_snapshot)
                purged_count = agent.memory.purge_lost()
                print()
                print("-" * 50)
                print("  CCSM MEMORY COMPRESSION & PURGE STATS")
                print("-" * 50)
                print(f"  Tracked Objects in memory : {summary.total_objects}")
                print(f"  Raw Trajectory Points     : {summary.raw_trajectory_points}")
                print(f"  Compressed Waypoints      : {summary.compressed_waypoints}")
                print(f"  Compression Ratio         : {summary.compression_ratio:.2f}x ({summary.retention_percentage:.1f}% retained)")
                print(f"  Manual Purge              : {purged_count} LOST object(s) removed")
                print("-" * 50)
                print()

    finally:
        agent.close()
        print()
        print("[CCSM] Stopped.")


if __name__ == "__main__":
    main()
