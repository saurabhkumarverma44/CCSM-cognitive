# CCSM Architecture — Phase 5.2 & Phase 6

## Pipeline Overview

```
Camera (OpenCV)
   │
   ▼
YOLO12l (GPU / CUDA:0)
   │
   ▼
ByteTrack (Persistent IDs)
   │
   ▼
IdentityRecovery (Re-association of occluded tracks)
   │
   ▼
MemoryManager (State Machine + Lifecycle Counters + Purge)
   │
   ├── visualizer.py   --> Annotated OpenCV display frame (HUD + Ghosts)
   ├── event_logger.py --> ANSI event terminal logging + dashboard
   ├── memory_optimizer.py --> Trajectory compression & importance ranking
   └── context_builder.py  --> Structured semantic context for LLM reasoning
```

---

## Object State Machine (Phase 5.2)

```
        ┌─────────────┐
        │     NEW     │ (Detection 1)
        └──────┬──────┘
               │ consecutive_detections >= min_confirmation_frames
               ▼
        ┌─────────────┐
        │   VISIBLE   │◄───────────────────────┐
        └──────┬──────┘                        │
               │                               │
       ┌───────┴───────────────┐               │
       │ missing <= occ_thresh │ missing > occ │
       │ + occluder IoU hit    │               │
       ▼                       ▼               │
┌──────────────┐       ┌──────────────┐        │
│ PARTIALLY_   │       │   OCCLUDED   │────────┘
│ OCCLUDED     │       └──────┬───────┘ (Re-detected or Re-associated)
└──────────────┘              │ missing > lost_timeout
                              ▼
                       ┌──────────────┐
                       │     LOST     │ (Retained for re-ID)
                       └──────┬───────┘
                              │ elapsed >= purge_lost_after_frames
                              ▼
                       ┌──────────────┐
                       │    PURGED    │ (Pruned with lifetime summary log)
                       └──────────────┘
```

---

## State Definitions & Lifecycle Fields

- **NEW**: First detection this session. Unconfirmed objects render with amber `? ID:N` badge on the HUD until confirmed.
- **VISIBLE**: Active confirmed detection (`consecutive_detections >= min_confirmation_frames`).
- **OCCLUDED**: Missing from tracker, but motion is predicted and position overlaps another visible object or duration <= `lost_timeout`.
- **REAPPEARED**: Instantaneous transition when an occluded or lost object is re-detected or re-associated. Triggers a bright cyan flash.
- **LOST**: Missing longer than `lost_timeout`. Ghost outline is removed, but record is retained for re-identification.
- **PURGED**: Stale lost object automatically removed from memory after `purge_lost_after_frames` to prevent memory leaks. Fires `on_purge` callback with full lifetime summary.

### Lifecycle Metrics & Properties
- `total_visible_frames`: Cumulative count of frames object was actively detected.
- `total_occluded_frames`: Cumulative count of frames object was occluded.
- `last_occlusion_duration`: Exact number of missing frames prior to recovery (captured before reset).
- `visibility_ratio`: `total_visible_frames / (total_visible_frames + total_occluded_frames)`.
- `stability_score`: Composite reliability metric (0.0–1.0) weighting confidence (35%), visibility ratio (40%), track age (20%), and confirmation bonus (5%).

---

## Phase 6 Research Components

### 1. Spatial-Temporal Identity Recovery (`core/identity_recovery.py`)
In real tracking scenarios, when an object is occluded for several seconds, ByteTrack often allocates a brand-new ID upon reappearance.
Identity Recovery evaluates candidate new detections against occluded / lost memory records using:
1. **Class Match**: Must share the same object class.
2. **Spatial Proximity**: Distance between new detection centroid and the object's `predicted_position`.
3. **Motion Vector Alignment**: Directional alignment (cosine similarity) with velocity.
4. **Bounding Box Geometry**: Width, height, and aspect ratio similarity.
If the matching score exceeds `min_recovery_score`, the original persistent identity is recovered, preserving continuity and firing `[RECOVERED]`.

### 2. Memory Optimizer & Importance Scoring (`core/memory_optimizer.py`)
Decimates redundant linear trajectory points while preserving key inflection waypoints:
- Calculates observation importance based on velocity shifts, state transitions, and turning points.
- Compresses memory footprint while retaining maximum semantic accuracy.

### 3. Semantic Context Builder (`core/context_builder.py`)
Translates low-level frame data and memory snapshots into compact natural language narratives and structured JSON for LLM reasoning.

---

## File Structure

```text
D:\Projects\CCSM
├── agents/
│   └── perception_agent.py      # Orchestration (Camera + Tracker + Memory + Logger)
├── config/
│   └── settings.py              # Configuration dataclasses (Perception, Memory, Viz, Recovery)
├── core/
│   ├── camera.py                # OpenCV webcam capture
│   ├── tracker_wrapper.py       # YOLO12l GPU + ByteTrack wrapper
│   ├── memory_models.py         # TrackedObjectMemory, ObjectState, Velocity, Trajectory
│   ├── memory_manager.py        # Central state machine & lifecycle management
│   ├── identity_recovery.py     # Spatial-temporal re-association / re-ID
│   ├── memory_optimizer.py      # Trajectory compression & importance scoring
│   ├── context_builder.py       # Semantic scene narrative generator for LLMs
│   ├── visualizer.py            # OpenCV HUD overlay & ghost rendering
│   └── event_logger.py          # ANSI terminal logging & dashboard
├── docs/
│   └── ARCHITECTURE.md          # Architecture & pipeline documentation
├── experiments/
│   ├── occlusion_test.py        # Phase 5.1 synthetic occlusion scenario
│   └── lifecycle_test.py        # Phase 5.2 lifecycle & auto-purge verification
├── tests/
│   ├── test_memory_models.py    # 16 unit tests for data models & metrics
│   ├── test_memory_manager.py   # 51 unit tests for MemoryManager
│   └── test_identity_recovery.py# Re-ID matching tests
└── main.py                      # Main executable loop with interactive hotkeys
```

---

## Phase Roadmap

- Phase 5: Memory Layer Core `[DONE]`
- Phase 5.1: Robust Occlusion + Ghost Visualization `[DONE]`
- Phase 5.2: Stable Object Lifecycle Management & Auto-Purge `[DONE]`
- Phase 6: Persistent Spatial-Temporal Memory & Identity Recovery `[IN PROGRESS]`
- Phase 7: Scene Understanding & Spatial Relationships
- Phase 8: LLM Reasoning / Context Layer
- Phase 9: Robotics / Agent Decision Layer
- Phase 10: Evaluation & Real-World Demonstration
