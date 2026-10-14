# core/visualizer.py  Phase 5.2 - Enhanced visualization
"""
Draws memory-aware annotations on OpenCV frames.

Phase 5.2 enhancements:
  - HUD shows object age (frames) and stability score
  - Unconfirmed objects shown with amber colour and '?' indicator
  - Ghost label shows miss count and occluder information
  - REAPPEARED flash: cyan highlight for one frame
  - Trajectory ghost trail: dots along last N known positions
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import CONFIG, VisualizationConfig
from core.memory_models import ObjectState, TrackedObjectMemory

# BGR colours
_C_VISIBLE       = (0,  220,  0)
_C_UNCONFIRMED   = (50, 180, 220)   # amber-cyan for NEW not yet confirmed
_C_OCCLUDED      = (0,  140, 255)
_C_REAPPEARED    = (0,  255, 220)
_C_LOST          = (80,  80, 200)
_C_HUD_HEAD      = (235, 235, 235)
_C_HUD_DIM       = (160, 160, 160)


def _state_colour(state: ObjectState, confirmed: bool = True) -> Tuple[int,int,int]:
    if state == ObjectState.NEW and not confirmed:
        return _C_UNCONFIRMED
    return {
        ObjectState.NEW:                _C_VISIBLE,
        ObjectState.VISIBLE:            _C_VISIBLE,
        ObjectState.PARTIALLY_OCCLUDED: _C_OCCLUDED,
        ObjectState.OCCLUDED:           _C_OCCLUDED,
        ObjectState.REAPPEARED:         _C_REAPPEARED,
        ObjectState.LOST:               _C_LOST,
    }.get(state, (200, 200, 200))


def _draw_dashed_rect(frame: np.ndarray,
                      x1: int, y1: int, x2: int, y2: int,
                      colour: Tuple[int,int,int],
                      thickness: int = 2,
                      dash_count: int = 12) -> None:
    """Draw a dashed rectangle (works on all platforms without extra libs)."""
    for (sx, sy), (ex, ey) in [
        ((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
        ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1)),
    ]:
        dx, dy = ex-sx, ey-sy
        segs = max(dash_count, 2)
        for i in range(segs):
            if i % 2 == 0:
                t0, t1 = i/segs, (i+1)/segs
                cv2.line(frame,
                         (int(sx+dx*t0), int(sy+dy*t0)),
                         (int(sx+dx*t1), int(sy+dy*t1)),
                         colour, thickness)


def _draw_text_block(frame: np.ndarray,
                     lines: List[str],
                     anchor_x: int, anchor_y: int,
                     colour: Tuple[int,int,int]) -> None:
    """Draw a semi-transparent label block above (anchor_x, anchor_y)."""
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.44
    thick  = 1
    pad    = 3
    line_h = 15

    text_w = max((cv2.getTextSize(l, font, fscale, thick)[0][0] for l in lines), default=40)
    bg_x1 = anchor_x
    bg_y1 = max(0, anchor_y - len(lines) * line_h - pad * 2)
    bg_x2 = anchor_x + text_w + pad * 2
    bg_y2 = anchor_y

    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        ty = bg_y1 + pad + (i + 1) * line_h
        cv2.putText(frame, line, (bg_x1 + pad, ty), font, fscale, colour, thick)


def _draw_ghost(frame: np.ndarray,
                mem: TrackedObjectMemory,
                cfg: VisualizationConfig) -> None:
    """Dashed ghost outline at predicted (or last-known) position."""
    h_frame, w_frame = frame.shape[:2]

    if mem.predicted_position is not None:
        cx, cy = mem.predicted_position
    elif mem.last_position is not None:
        cx, cy = mem.last_position
    else:
        return

    if mem.last_bbox is not None:
        lx1, ly1, lx2, ly2 = mem.last_bbox
        bw, bh = lx2-lx1, ly2-ly1
    else:
        bw, bh = 60.0, 120.0

    x1 = max(0, min(int(cx - bw/2), w_frame-2))
    y1 = max(0, min(int(cy - bh/2), h_frame-2))
    x2 = max(x1+1, min(int(cx + bw/2), w_frame-1))
    y2 = max(y1+1, min(int(cy + bh/2), h_frame-1))

    colour = _state_colour(mem.state, mem.confirmed)

    if cfg.show_ghost:
        _draw_dashed_rect(frame, x1, y1, x2, y2, colour,
                          thickness=2, dash_count=cfg.ghost_dash_count)

    # Draw trajectory trail (last 5 known positions as small dots)
    if len(mem.trajectory) >= 2:
        pts = list(mem.trajectory)[-6:]
        for tp in pts:
            tx, ty = int(tp.cx), int(tp.cy)
            if 0 <= tx < w_frame and 0 <= ty < h_frame:
                cv2.circle(frame, (tx, ty), 2, colour, -1)

    # Label lines
    lines = [f"ID:{mem.track_id} {mem.class_name}"]
    if mem.state == ObjectState.OCCLUDED:
        lines.append(f"OCCLUDED {mem.missing_frames}f")
        if mem.occluder_id is not None:
            lines.append(f"BY ID:{mem.occluder_id}")
    elif mem.state == ObjectState.REAPPEARED:
        lines.append("REAPPEARED")
    elif mem.state == ObjectState.LOST:
        lines.append("LOST")

    _draw_text_block(frame, lines, x1, y1, colour)


def annotate_frame(
    frame: np.ndarray,
    memory_snapshot: List[TrackedObjectMemory],
    frame_index: int,
    cfg: Optional[VisualizationConfig] = None,
) -> np.ndarray:
    """
    Main visualization entry point. Called after results.plot() has drawn
    YOLO boxes. Adds the memory/ghost/HUD layer on top.
    """
    if cfg is None:
        cfg = CONFIG.visualization

    h, w = frame.shape[:2]

    for mem in memory_snapshot:
        if mem.state in (ObjectState.OCCLUDED, ObjectState.PARTIALLY_OCCLUDED):
            _draw_ghost(frame, mem, cfg)

        elif mem.state == ObjectState.REAPPEARED:
            # Bright solid flash for one frame
            if mem.last_bbox is not None:
                x1, y1, x2, y2 = (int(v) for v in mem.last_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), _C_REAPPEARED, 3)
                _draw_text_block(frame,
                                 [f"ID:{mem.track_id} {mem.class_name}", "RECOVERED"],
                                 x1, y1, _C_REAPPEARED)

        elif mem.state == ObjectState.NEW and not mem.confirmed:
            # Unconfirmed: draw a thin amber outline over the YOLO box
            if mem.last_bbox is not None:
                x1, y1, x2, y2 = (int(v) for v in mem.last_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), _C_UNCONFIRMED, 1)
                cv2.putText(frame, f"? ID:{mem.track_id} {mem.class_name}",
                            (x1, max(y1-4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, _C_UNCONFIRMED, 1)

    if cfg.show_hud:
        _draw_hud(frame, memory_snapshot, frame_index)

    return frame


def _draw_hud(frame: np.ndarray,
              memory_snapshot: List[TrackedObjectMemory],
              frame_index: int) -> None:
    """Compact HUD panel: top-left, semi-transparent."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x    = 8
    y    = 18
    ls   = 17

    visible  = [m for m in memory_snapshot if m.is_visible]
    occluded = [m for m in memory_snapshot if m.state == ObjectState.OCCLUDED]
    lost     = [m for m in memory_snapshot if m.state == ObjectState.LOST]

    n_lines  = 3 + len(visible) + len(occluded) + min(len(lost), 4) + 3
    panel_h  = n_lines * ls + 8
    panel_w  = 330
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    def put(text, colour=_C_HUD_DIM):
        nonlocal y
        cv2.putText(frame, text, (x, y), font, 0.40, colour, 1)
        y += ls

    put(f"Frame:{frame_index}  Mem:{len(memory_snapshot)}", _C_HUD_HEAD)
    put(f"VIS:{len(visible)}  OCC:{len(occluded)}  LOST:{len(lost)}", _C_HUD_HEAD)
    put("---", (70, 70, 70))

    for mem in sorted(visible, key=lambda m: m.track_id):
        pos = mem.last_position
        ps  = f"({pos[0]:.0f},{pos[1]:.0f})" if pos else "?"
        chk = "" if mem.confirmed else "?"
        stb = f"{mem.stability_score:.2f}"
        col = _state_colour(mem.state, mem.confirmed)
        put(f"[VIS{chk}] {mem.track_id} {mem.class_name[:9]:<9} {ps}  s={stb}", col)

    for mem in sorted(occluded, key=lambda m: m.track_id):
        by  = f"->by:{mem.occluder_id}" if mem.occluder_id else ""
        put(f"[OCC]  {mem.track_id} {mem.class_name[:9]:<9} {mem.missing_frames}f {by}",
            _C_OCCLUDED)

    if lost:
        put(f"[LOST x{len(lost)}] (auto-purged after {CONFIG.memory.purge_lost_after_frames}f)",
            _C_LOST)
        for mem in sorted(lost, key=lambda m: m.track_id)[:4]:
            put(f"  {mem.track_id} {mem.class_name[:9]}", _C_LOST)
        if len(lost) > 4:
            put(f"  ... +{len(lost)-4} more", _C_LOST)
