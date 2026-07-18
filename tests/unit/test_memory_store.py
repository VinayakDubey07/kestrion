import pytest
from datetime import datetime

from kestrion.core.types import Event, EventType, Checkpoint, AgentState, RunStatus
from kestrion.store.memory_store import MemoryCheckpointStore

@pytest.mark.asyncio
async def test_memory_store_events():
    store = MemoryCheckpointStore()
    run_id = "test_run"
    
    evt1 = Event.create(run_id, EventType.RUN_STARTED, payload={"foo": "bar"}, node="start")
    seq1 = await store.append_event(evt1)
    
    evt2 = Event.create(run_id, EventType.STATE_TRANSITION, payload={}, node="next")
    seq2 = await store.append_event(evt2)
    
    assert seq1 == 1
    assert seq2 == 2
    
    events = await store.events_since(run_id, 0)
    assert len(events) == 2
    assert events[0].type == EventType.RUN_STARTED
    assert events[1].type == EventType.STATE_TRANSITION
    
    events_since_1 = await store.events_since(run_id, 1)
    assert len(events_since_1) == 1
    assert events_since_1[0].type == EventType.STATE_TRANSITION

@pytest.mark.asyncio
async def test_memory_store_checkpoints():
    store = MemoryCheckpointStore()
    run_id = "test_run"
    
    state = AgentState(run_id=run_id, status=RunStatus.RUNNING, current_node="node1", scratch={"x": 1})
    ckpt = Checkpoint(
        checkpoint_id="ckpt_1",
        run_id=run_id,
        state=state,
        created_at=datetime.utcnow(),
        event_seq=1,
    )
    
    await store.save(ckpt)
    
    latest = await store.latest(run_id)
    assert latest is not None
    assert latest.checkpoint_id == "ckpt_1"
    assert latest.state.scratch["x"] == 1
    
    # Verify deep copy isolation
    state.scratch["x"] = 2
    latest_again = await store.latest(run_id)
    assert latest_again.state.scratch["x"] == 1  # Should not be affected by the mutation above
