# core/context_builder.py
# Phase 6 - Semantic Context Builder for LLM Reasoning
"""
Transforms temporal memory representations into compact, structured
natural-language narratives and JSON context blocks for LLM reasoning.

Reference Research Goals:
  - Section 27: Context Builder
  - Section 28: LLM Reasoning Context Layer
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional, Sequence

from core.memory_models import ObjectState, TrackedObjectMemory


def _direction_str(vx: float, vy: float, threshold: float = 0.5) -> str:
    """Converts a 2D velocity vector into a human-readable cardinal direction."""
    speed = (vx ** 2 + vy ** 2) ** 0.5
    if speed < threshold:
        return "stationary"
    h = "right" if vx > threshold else ("left" if vx < -threshold else "")
    v = "down" if vy > threshold else ("up" if vy < -threshold else "")
    if h and v:
        return f"moving {v}-{h} at {speed:.1f} px/f"
    elif h:
        return f"moving {h} at {speed:.1f} px/f"
    elif v:
        return f"moving {v} at {speed:.1f} px/f"
    return "slow drift"


class ContextBuilder:
    """
    Synthesizes episodic memory snapshots into semantic, structured context
    suitable for LLM prompts and robotic decision-making.
    """

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps

    def build_object_narrative(self, mem: TrackedObjectMemory) -> str:
        """Generates a concise chronological narrative for an individual object."""
        tid = mem.track_id
        cls = mem.class_name
        age_s = mem.age_frames / self.fps
        pos_str = f"({mem.last_position[0]:.0f}, {mem.last_position[1]:.0f})" if mem.last_position else "unknown"
        dir_str = _direction_str(mem.velocity.vx, mem.velocity.vy)

        lines = [
            f"- {cls.title()} (ID {tid}):",
            f"  • Status: {mem.state.value} (stability: {mem.stability_score * 100:.0f}%, confidence: {mem.last_confidence:.2f})",
            f"  • Lifespan: {mem.age_frames} frames (~{age_s:.1f}s), first seen frame {mem.first_seen_frame}",
            f"  • Position & Motion: Currently at {pos_str}, {dir_str}",
        ]

        if mem.reappearance_count > 0:
            last_dur_s = mem.last_occlusion_duration / self.fps
            lines.append(
                f"  • Occlusion History: Reappeared {mem.reappearance_count} time(s); "
                f"last occlusion lasted {mem.last_occlusion_duration} frames (~{last_dur_s:.2f}s)"
            )
        elif mem.state == ObjectState.OCCLUDED:
            by_str = f" behind {mem.occluder_id}" if mem.occluder_id else ""
            lines.append(
                f"  • Occlusion: Currently hidden{by_str} for {mem.missing_frames} frames"
            )
        elif mem.state == ObjectState.LOST:
            lines.append(
                f"  • Lost: Missing for {mem.missing_frames} frames"
            )

        return "\n".join(lines)

    def build_scene_narrative(
        self,
        memory_snapshot: Sequence[TrackedObjectMemory],
        frame_index: int,
    ) -> str:
        """Generates an executive scene narrative across all tracked objects."""
        visible = [m for m in memory_snapshot if m.is_visible]
        occluded = [m for m in memory_snapshot if m.state == ObjectState.OCCLUDED]
        lost = [m for m in memory_snapshot if m.state == ObjectState.LOST]

        lines = [
            f"=== CCSM SCENE CONTEXT | Frame {frame_index} (~{frame_index / self.fps:.1f}s) ===",
            f"Summary: {len(visible)} visible, {len(occluded)} occluded, {len(lost)} lost.",
        ]

        # Key interactions
        interactions = []
        for occ in occluded:
            if occ.occluder_id:
                interactions.append(
                    f"{occ.class_name.title()} (ID {occ.track_id}) is occluded behind ID {occ.occluder_id}"
                )
        if interactions:
            lines.append("Active Interactions: " + "; ".join(interactions) + ".")

        lines.append("")
        lines.append("Tracked Entities:")
        for m in sorted(memory_snapshot, key=lambda x: x.track_id):
            lines.append(self.build_object_narrative(m))

        return "\n".join(lines)

    def build_llm_payload(
        self,
        memory_snapshot: Sequence[TrackedObjectMemory],
        frame_index: int,
    ) -> dict:
        """
        Creates a structured dictionary formatted for direct injection
        into an LLM chat/completion API call.
        """
        visible = [m for m in memory_snapshot if m.is_visible]
        occluded = [m for m in memory_snapshot if m.state == ObjectState.OCCLUDED]
        lost = [m for m in memory_snapshot if m.state == ObjectState.LOST]

        entities = []
        for m in sorted(memory_snapshot, key=lambda x: x.track_id):
            entities.append({
                "id": m.track_id,
                "class": m.class_name,
                "state": m.state.value,
                "confirmed": m.confirmed,
                "stability": round(m.stability_score, 3),
                "position": m.last_position,
                "motion": {
                    "vx": round(m.velocity.vx, 2),
                    "vy": round(m.velocity.vy, 2),
                    "direction": _direction_str(m.velocity.vx, m.velocity.vy),
                },
                "age_frames": m.age_frames,
                "missing_frames": m.missing_frames,
                "reappearance_count": m.reappearance_count,
                "last_occlusion_duration": m.last_occlusion_duration,
                "occluded_by": m.occluder_id,
            })

        return {
            "frame": frame_index,
            "timestamp_seconds": round(frame_index / self.fps, 2),
            "counts": {
                "visible": len(visible),
                "occluded": len(occluded),
                "lost": len(lost),
                "total": len(memory_snapshot),
            },
            "entities": entities,
            "prompt_text": self.build_scene_narrative(memory_snapshot, frame_index),
        }
