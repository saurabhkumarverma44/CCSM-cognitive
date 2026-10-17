
"""CCSM: scene memory, persistent episodic history, and LLM reasoning."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2

# Configure logging before importing project modules.
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path("logs") / "ccsm.log",
            encoding="utf-8",
        ),
    ],
)

from agents.perception_agent import PerceptionAgent
from config.settings import CONFIG
from core.context_builder import ContextBuilder
from core.episodic_memory import EpisodicMemory
from core.llm_client import (
    GroqSceneAssistant,
    LLMConfigurationError,
    LLMRequestError,
)
from core.memory_optimizer import MemoryOptimizer
from core.reasoning_worker import EventReasoningWorker
from core.visualizer import annotate_frame

_LOG = logging.getLogger("ccsm.main")


def main() -> None:
    """Run perception, persistent event memory, and LLM reasoning."""
    agent = PerceptionAgent()
    context_builder = ContextBuilder(fps=30.0)
    optimizer = MemoryOptimizer()

    episodic_memory = EpisodicMemory(
        database_path="data/ccsm_memory.sqlite3",
    )

    reasoning_worker = EventReasoningWorker(
        context_provider=lambda: build_reasoning_context(
            agent=agent,
            context_builder=context_builder,
            episodic_memory=episodic_memory,
        ),
        queue_capacity=100,
        batch_size=3,
        batch_wait_seconds=0.5,
        max_results=200,
        min_request_interval_seconds=30.0,
        failure_cooldown_seconds=90.0,
    )

    # Register both independent listeners.
    agent.event_manager.register_listener(
        episodic_memory.record_event_safely
    )
    agent.event_manager.register_listener(
        reasoning_worker.submit_event
    )

    show_hud = True
    frame_index = 0
    llm_assistant: GroqSceneAssistant | None = None
    displayed_reasoning_timestamps: set[str] = set()

    print()
    print("=" * 66)
    print("  CCSM - Scene Memory + Persistent Episodic Reasoning")
    print(
        "  q=quit  m=HUD  d=debug  c=print context  "
        "g=ask Groq LLM  p=memory stats"
    )
    print("=" * 66)
    print()
    print(f"[CCSM] Episodic database: {episodic_memory.database_path}")
    print(f"[CCSM] Session ID: {episodic_memory.session_id}")

    try:
        agent.open()
        reasoning_worker.start()

        while True:
            (
                tracked_objects,
                memory_snapshot,
                annotated_frame,
                raw_frame,
            ) = agent.process_frame_with_memory()

            frame_index += 1

            # Display completed automatic reasoning results.
            results = reasoning_worker.get_results()

            for result in results:
                if result.timestamp in displayed_reasoning_timestamps:
                    continue

                displayed_reasoning_timestamps.add(result.timestamp)

                print()
                print("-" * 66)
                print("CCSM AUTOMATIC EVENT REASONING")
                print(f"Event IDs: {result.event_ids}")

                if result.error:
                    print(f"Reasoning failed: {result.error}")
                elif result.answer:
                    print(result.answer)
                else:
                    print("(No answer returned.)")

                print("-" * 66)
                print()

            # Bound display bookkeeping during long-running sessions.
            if len(displayed_reasoning_timestamps) > 400:
                displayed_reasoning_timestamps = {
                    result.timestamp for result in results
                }

            display_config = (
                CONFIG.visualization
                if show_hud
                else CONFIG.visualization.__class__(
                    show_hud=False,
                    show_ghost=True,
                )
            )

            display_frame = annotate_frame(
                annotated_frame,
                memory_snapshot,
                frame_index,
                cfg=display_config,
            )

            cv2.imshow(
                "CCSM - Scene Memory + Automatic Reasoning",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("m"):
                show_hud = not show_hud
                print(f"[HUD] {'ON' if show_hud else 'OFF'}")

            elif key == ord("d"):
                cfg = agent.event_logger._cfg

                updated_values = {
                    field.name: getattr(cfg, field.name)
                    for field in cfg.__dataclass_fields__.values()
                }
                updated_values["debug_mode"] = not cfg.debug_mode

                agent.event_logger._cfg = cfg.__class__(
                    **updated_values
                )

                print(
                    "[DEBUG] "
                    f"{'ON' if agent.event_logger._cfg.debug_mode else 'OFF'}"
                )

            elif key == ord("c"):
                payload = build_reasoning_context(
                    agent=agent,
                    context_builder=context_builder,
                    episodic_memory=episodic_memory,
                    memory_snapshot=memory_snapshot,
                    frame_index=frame_index,
                )

                print()
                print("=" * 66)
                print("CCSM CURRENT SCENE")
                print("=" * 66)
                print(payload.get("prompt_text", "(No scene narrative available.)"))

                history = payload.get("episodic_memory", {})
                recent = history.get("recent_events", [])
                prior = history.get("prior_session_events", [])

                print()
                print(
                    f"[Episodic Memory] Current-session events: "
                    f"{len(recent)}"
                )
                print(
                    f"[Episodic Memory] Previous-session records: "
                    f"{len(prior)}"
                )
                print()

            elif key == ord("g"):
                question = input(
                    "\n[CCSM LLM] Ask about the current scene or its "
                    "recorded history (blank cancels): "
                ).strip()

                if not question:
                    print("[CCSM LLM] Cancelled.")
                    continue

                try:
                    if llm_assistant is None:
                        llm_assistant = GroqSceneAssistant()

                    payload = build_reasoning_context(
                        agent=agent,
                        context_builder=context_builder,
                        episodic_memory=episodic_memory,
                        memory_snapshot=memory_snapshot,
                        frame_index=frame_index,
                    )

                    answer = llm_assistant.answer(question, payload)

                    print()
                    print("-" * 66)
                    print("CCSM GROQ ANSWER")
                    print("-" * 66)
                    print(answer)
                    print("-" * 66)
                    print()

                except (
                    LLMConfigurationError,
                    LLMRequestError,
                    ValueError,
                ) as exc:
                    print(f"[CCSM LLM] {exc}")

                except Exception:
                    _LOG.exception(
                        "Unexpected error while answering a scene question."
                    )
                    print(
                        "[CCSM LLM] Unexpected error. "
                        "Check logs/ccsm.log."
                    )

            elif key == ord("p"):
                summary = optimizer.summarize_compression(memory_snapshot)
                purged_count = agent.memory.purge_lost()

                print()
                print("-" * 54)
                print("  CCSM MEMORY AND EPISODIC HISTORY")
                print("-" * 54)
                print(f"  Tracked objects            : {summary.total_objects}")
                print(
                    f"  Raw trajectory points      : "
                    f"{summary.raw_trajectory_points}"
                )
                print(
                    f"  Compressed waypoints       : "
                    f"{summary.compressed_waypoints}"
                )
                print(
                    f"  Compression ratio          : "
                    f"{summary.compression_ratio:.2f}x"
                )
                print(
                    f"  Retention                  : "
                    f"{summary.retention_percentage:.1f}%"
                )
                print(f"  Manual lost-object purge   : {purged_count}")
                print(
                    f"  Current-session events     : "
                    f"{episodic_memory.event_count()}"
                )
                print(
                    f"  All persisted events       : "
                    f"{episodic_memory.event_count(current_session_only=False)}"
                )
                print("-" * 54)
                print()

    except KeyboardInterrupt:
        print("\n[CCSM] Keyboard interrupt received.")

    finally:
        # Stop event submission before closing the database.
        try:
            agent.event_manager.unregister_listener(
                reasoning_worker.submit_event
            )
        except Exception:
            _LOG.exception("Could not unregister reasoning listener.")

        try:
            agent.event_manager.unregister_listener(
                episodic_memory.record_event_safely
            )
        except Exception:
            _LOG.exception("Could not unregister episodic-memory listener.")

        try:
            reasoning_worker.stop(timeout=30.0)
        except Exception:
            _LOG.exception("Error while stopping the reasoning worker.")
        finally:
            try:
                agent.close()
            except Exception:
                _LOG.exception("Error while closing the perception agent.")
            finally:
                try:
                    episodic_memory.close()
                except Exception:
                    _LOG.exception("Error while closing episodic memory.")

                cv2.destroyAllWindows()

        print()
        print("[CCSM] Stopped.")


def build_reasoning_context(
    *,
    agent: PerceptionAgent,
    context_builder: ContextBuilder,
    episodic_memory: EpisodicMemory,
    memory_snapshot: Any = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    """Combine current perception with retrieved persistent event history."""
    if memory_snapshot is None:
        memory_snapshot = agent.memory.all_objects()

    if frame_index is None:
        frame_index = agent.frame_index

    payload = context_builder.build_llm_payload(
        memory_snapshot,
        frame_index,
    )

    recent_current = episodic_memory.recent_events(
        limit=12,
        current_session_only=True,
    )

    track_ids = list(
        dict.fromkeys(
            int(event["track_id"])
            for event in recent_current
            if event.get("track_id") is not None
        )
    )[:5]

    history = episodic_memory.build_history_context(
        current_frame=frame_index,
        recent_limit=12,
        timeline_track_ids=track_ids,
        timeline_limit=10,
    )

    all_recent = episodic_memory.recent_events(
        limit=1000,
        current_session_only=False,
    )

    history["prior_session_events"] = [
        event
        for event in all_recent
        if event.get("_session_id") != episodic_memory.session_id
    ][-12:]

    history["cross_session_identity_warning"] = (
        "Track IDs may be reused across sessions. Never infer that objects "
        "from different sessions are the same physical object based on their "
        "track IDs alone."
    )

    payload["episodic_memory"] = history
    payload["historical_context_warning"] = (
        "Use current scene data for current visibility and historical events "
        "for previous state transitions. Do not confuse the two."
    )

    return payload


if __name__ == "__main__":
    main()
