
from config.settings import MemoryConfig
from core.event_manager import EventManager
from core.memory_manager import MemoryManager
from core.memory_models import ObjectState


def _config():
    return MemoryConfig(
        occlusion_threshold=2,
        lost_timeout=5,
        velocity_ema_alpha=1.0,
        occlusion_iou_threshold=0.3,
        velocity_decay=1.0,
        max_prediction_displacement=9999.0,
        min_confirmation_frames=2,
        purge_lost_after_frames=50,
    )


def _person(track_id=1):
    return {
        "track_id": track_id,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "bbox": [75, 150, 125, 250],
        "center": (100, 200),
    }


def test_new_object_event_is_recorded():
    manager = MemoryManager(config=_config())
    events = EventManager()
    manager.register_state_callback(events.on_state_change)

    manager.update([_person()], 0)

    recorded = events.get_events()
    assert len(recorded) == 1
    assert recorded[0].new_state == ObjectState.NEW.value
    assert recorded[0].track_id == 1


def test_occlusion_and_reappearance_are_recorded():
    manager = MemoryManager(config=_config())
    events = EventManager()
    manager.register_state_callback(events.on_state_change)

    manager.update([_person()], 0)
    manager.update([_person()], 1)
    for frame in range(2, 5):
        manager.update([], frame)
    manager.update([_person()], 5)

    states = [event.new_state for event in events.get_events()]
    assert ObjectState.OCCLUDED.value in states
    assert ObjectState.REAPPEARED.value in states


def test_history_is_bounded():
    events = EventManager(max_history=2)
    manager = MemoryManager(config=_config())
    manager.register_state_callback(events.on_state_change)

    manager.update([_person(1)], 0)
    manager.update([_person(2)], 0)
    manager.update([_person(3)], 0)

    assert len(events.get_events()) == 2


def test_drain_events_clears_history():
    events = EventManager()
    manager = MemoryManager(config=_config())
    manager.register_state_callback(events.on_state_change)

    manager.update([_person()], 0)

    drained = events.drain_events()
    assert len(drained) == 1
    assert events.get_events() == []
