# core/event_logger.py  Phase 5.2 - Event-driven terminal logging
"""
Listens to MemoryManager state-change callbacks and prints
only meaningful events to the terminal.

Fixes vs Phase 5.1:
  - Uses mem.last_occlusion_duration (not mem.missing_frames) for RECOVERED
    -> correctly shows how many frames the object was missing
  - Suppresses redundant REAPPEARED -> VISIBLE transition line
  - Prints lifecycle summary when a LOST object is auto-purged
  - Stability score shown in dashboard
"""
from __future__ import annotations
import logging
import os
import sys
from typing import Dict, Optional

from config.settings import CONFIG, LoggingConfig
from core.memory_models import ObjectState, TrackedObjectMemory

logger = logging.getLogger(__name__)

_TAG = {
    "NEW":        ("[NEW]      ", "\033[92m"),
    "VISIBLE":    ("[VISIBLE]  ", "\033[92m"),
    "OCCLUDED":   ("[OCCLUDED] ", "\033[93m"),
    "REAPPEARED": ("[RECOVERED]", "\033[96m"),
    "LOST":       ("[LOST]     ", "\033[91m"),
    "PURGED":     ("[PURGED]   ", "\033[35m"),
    "STATE":      ("[STATE]    ", "\033[97m"),
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"

_ANSI_SUPPORTED = (sys.platform != "win32") or (
    "ANSICON" in os.environ or "WT_SESSION" in os.environ
    or "ConEmuANSI" in os.environ
    or os.environ.get("TERM_PROGRAM") == "vscode"
)
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(  # type: ignore
            ctypes.windll.kernel32.GetStdHandle(-11), 7)  # type: ignore
        _ANSI_SUPPORTED = True
    except Exception:
        pass


def _c(tag_key: str, text: str) -> str:
    tag_str, colour = _TAG.get(tag_key, ("", ""))
    if _ANSI_SUPPORTED:
        return f"{colour}{_BOLD}{tag_str}{_RESET} {colour}{text}{_RESET}"
    return f"{tag_str} {text}"


def _plain(text: str) -> str:
    return f"\033[0m{text}\033[0m" if _ANSI_SUPPORTED else text


def _stability_bar(score: float, width: int = 8) -> str:
    """Visual bar: e.g. [######..] 0.82"""
    filled = round(score * width)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {score:.2f}"


class EventLogger:
    """
    Subscribes to MemoryManager state-change + purge callbacks.
    Prints only when something meaningful changes.
    """

    def __init__(self, cfg: Optional[LoggingConfig] = None) -> None:
        self._cfg: LoggingConfig = cfg or CONFIG.logging
        # Track last printed state per object to avoid duplicate messages
        self._last_logged_state: Dict[int, ObjectState] = {}

    # ------------------------------------------------------------------
    # Callbacks registered with MemoryManager
    # ------------------------------------------------------------------

    def on_state_change(self,
                        mem: TrackedObjectMemory,
                        old_state: ObjectState,
                        new_state: ObjectState,
                        frame: int) -> None:
        """Fires on every state transition from MemoryManager."""
        tid = mem.track_id
        cls = mem.class_name

        # Suppress REAPPEARED -> VISIBLE: we already printed [RECOVERED]
        # when old_state was OCCLUDED -> REAPPEARED; the follow-up
        # REAPPEARED -> VISIBLE is redundant noise.
        if old_state == ObjectState.REAPPEARED and new_state == ObjectState.VISIBLE:
            self._last_logged_state[tid] = new_state
            return

        # De-duplicate: only print if state actually changed from what we last logged
        if self._last_logged_state.get(tid) == new_state:
            return
        self._last_logged_state[tid] = new_state

        # --- Print event ---
        if new_state == ObjectState.NEW:
            print(_c("NEW", f"ID={tid} class={cls}  (frame {frame})"))

        elif old_state == ObjectState.NEW and new_state == ObjectState.VISIBLE:
            conf_str = f"  conf={mem.last_confidence:.2f}" if self._cfg.debug_mode else ""
            print(_c("VISIBLE", f"ID={tid} {cls}  confirmed after {mem.consecutive_detections}f{conf_str}"))

        elif new_state == ObjectState.OCCLUDED:
            occ = f"  occluded_by=ID:{mem.occluder_id}" if mem.occluder_id else ""
            print(_c("OCCLUDED", f"ID={tid} {cls}  {old_state.value} -> OCCLUDED{occ}"))

        elif new_state == ObjectState.REAPPEARED:
            # FIX: use last_occlusion_duration (captured before missing_frames reset)
            dur = mem.last_occlusion_duration
            stab = _stability_bar(mem.stability_score)
            print(_c("REAPPEARED",
                     f"ID={tid} {cls}  back after {dur} frames  "
                     f"(reapp #{mem.reappearance_count})  stab={stab}"))

        elif new_state == ObjectState.VISIBLE and old_state not in (
                ObjectState.NEW, ObjectState.REAPPEARED):
            print(_c("STATE", f"ID={tid} {cls}  {old_state.value} -> VISIBLE"))

        elif new_state == ObjectState.LOST:
            stab = _stability_bar(mem.stability_score)
            print(_c("LOST",
                     f"ID={tid} {cls}  after {mem.missing_frames} missing frames  "
                     f"age={mem.age_frames}f  stab={stab}"))

        elif self._cfg.debug_mode:
            print(_c("STATE", f"ID={tid} {cls}  {old_state.value} -> {new_state.value}"))

    def on_purge(self, mem: TrackedObjectMemory, frame: int) -> None:
        """
        Fires when a LOST object is auto-purged from memory.
        Prints a full lifecycle summary.
        """
        tid = mem.track_id
        vis_pct = f"{mem.visibility_ratio*100:.0f}%"
        stab = _stability_bar(mem.stability_score)
        lifecycle = (
            f"age={mem.age_frames}f  "
            f"vis={mem.total_visible_frames}f ({vis_pct})  "
            f"occ={mem.total_occluded_frames}f  "
            f"reapps={mem.reappearance_count}  "
            f"stab={stab}"
        )
        print(_c("PURGED", f"ID={tid} {mem.class_name}  [{lifecycle}]"))
        self._last_logged_state.pop(tid, None)

    # ------------------------------------------------------------------
    # Per-frame tick (dashboard)
    # ------------------------------------------------------------------

    def tick(self, memory_snapshot, frame_index: int) -> None:
        if self._cfg.debug_mode:
            self._print_debug(memory_snapshot, frame_index)
        interval = self._cfg.dashboard_interval
        if interval > 0 and frame_index > 0 and frame_index % interval == 0:
            self._print_dashboard(memory_snapshot, frame_index)

    def _print_debug(self, memory_snapshot, frame_index: int) -> None:
        for mem in sorted(memory_snapshot, key=lambda m: m.track_id):
            pos = mem.last_position or mem.predicted_position
            ps = f"({pos[0]:.0f},{pos[1]:.0f})" if pos else "?"
            pred = (f" pred=({mem.predicted_position[0]:.0f},{mem.predicted_position[1]:.0f})"
                    if mem.state == ObjectState.OCCLUDED and mem.predicted_position else "")
            occ = f" occ_by={mem.occluder_id}" if mem.occluder_id else ""
            logger.debug("F%d ID=%d %s [%s] pos=%s vel=(%.1f,%.1f) miss=%d%s%s",
                         frame_index, mem.track_id, mem.class_name,
                         mem.state.value, ps, mem.velocity.vx, mem.velocity.vy,
                         mem.missing_frames, pred, occ)

    def _print_dashboard(self, memory_snapshot, frame_index: int) -> None:
        visible  = sorted([m for m in memory_snapshot if m.is_visible],
                          key=lambda m: m.track_id)
        occluded = sorted([m for m in memory_snapshot
                           if m.state == ObjectState.OCCLUDED],
                          key=lambda m: m.track_id)
        lost     = sorted([m for m in memory_snapshot
                           if m.state == ObjectState.LOST],
                          key=lambda m: m.track_id)
        sep  = "=" * 66
        thin = "-" * 66

        lines = [
            "",
            _plain(sep),
            _plain(f"  CCSM MEMORY MONITOR  |  Frame: {frame_index}  "
                   f"|  Tracked: {len(memory_snapshot)}"),
            _plain(sep),
            _plain(f"  VIS:{len(visible)}  OCC:{len(occluded)}  "
                   f"LOST:{len(lost)}  (auto-purge after {CONFIG.memory.purge_lost_after_frames}f)"),
            _plain(thin),
        ]

        if visible:
            lines.append(_plain("  VISIBLE:"))
            for m in visible:
                pos = m.last_position
                ps = f"({pos[0]:.0f},{pos[1]:.0f})" if pos else "?"
                conf_ind = "?" if not m.confirmed else ""
                stab = f"  {_stability_bar(m.stability_score)}" if self._cfg.debug_mode else ""
                lines.append(_c("VISIBLE",
                                 f"    ID {m.track_id:<3} {m.class_name:<12} "
                                 f"{ps:<16} age:{m.age_frames}f {conf_ind}{stab}"))

        if occluded:
            lines.append(_plain("  OCCLUDED:"))
            for m in occluded:
                by = f"  -> by ID:{m.occluder_id}" if m.occluder_id else ""
                lines.append(_c("OCCLUDED",
                                 f"    ID {m.track_id:<3} {m.class_name:<12} "
                                 f"miss:{m.missing_frames}f{by}"))

        if lost:
            lines.append(_plain("  LOST  (pending purge):"))
            for m in lost:
                wait = (CONFIG.memory.purge_lost_after_frames
                        - (frame_index - (m.lost_at_frame or frame_index)))
                wait_str = f"  purge_in:{max(0,wait)}f"
                lines.append(_c("LOST",
                                 f"    ID {m.track_id:<3} {m.class_name:<12}{wait_str}"))

        lines.append(_plain(sep))
        print("\n".join(lines), flush=True)
