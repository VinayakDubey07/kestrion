import os
from datetime import datetime, timezone

import pytest

from kestrion.core.types import AgentState, Checkpoint, Event, EventType
from kestrion.store.postgres_store import PostgresCheckpointStore


@pytest.fixture
async def pg_store():
    # Only run these tests if POSTGRES_DSN is provided in the environment
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        pytest.skip("POSTGRES_DSN not set. Skipping Postgres tests.")

    store = PostgresCheckpointStore(dsn)
    await store.setup()
    
    # Clean up tables before tests to ensure a clean state
    pool = store._get_pool()
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE events, checkpoints RESTART IDENTITY CASCADE;")
        
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_append_and_fetch_events(pg_store: PostgresCheckpointStore):
    run_id = "run_test_123"
    
    e1 = Event(
        event_id="evt_1",
        run_id=run_id,
        type=EventType.STATE_TRANSITION,
        timestamp=datetime.now(timezone.utc),
        node="node_a",
        payload={"input": "hello"},
    )
    
    e2 = Event(
        event_id="evt_2",
        run_id=run_id,
        type=EventType.TOOL_CALL_COMPLETED,
        timestamp=datetime.now(timezone.utc),
        node="node_a",
        payload={"output": "world"},
        tokens_in=10,
        tokens_out=20,
        cost_usd=0.001,
    )
    
    seq1 = await pg_store.append_event(e1)
    seq2 = await pg_store.append_event(e2)
    
    assert seq1 > 0
    assert seq2 == seq1 + 1
    
    events = await pg_store.events_since(run_id, 0)
    assert len(events) == 2
    
    assert events[0].event_id == "evt_1"
    assert events[0].payload == {"input": "hello"}
    
    assert events[1].event_id == "evt_2"
    assert events[1].tokens_in == 10
    assert events[1].cost_usd == 0.001


@pytest.mark.asyncio
async def test_save_and_latest_checkpoint(pg_store: PostgresCheckpointStore):
    run_id = "run_test_chk"
    
    state = AgentState(
        run_id=run_id,
        history=[{"role": "user", "content": "hi"}],
        scratch={"key": "value", "count": 1},
    )
    
    chk = Checkpoint(
        checkpoint_id="chk_1",
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        event_seq=1,
        state=state,
    )
    
    # Test saving a checkpoint
    await pg_store.save(chk)
    
    # Fetch it back
    latest = await pg_store.latest(run_id)
    assert latest is not None
    assert latest.checkpoint_id == "chk_1"
    assert latest.event_seq == 1
    assert latest.state.scratch["count"] == 1
    
    # Save a newer checkpoint
    state2 = AgentState(
        run_id=run_id,
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        scratch={"key": "value", "count": 2},
    )
    chk2 = Checkpoint(
        checkpoint_id="chk_2",
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        event_seq=2,
        state=state2,
    )
    await pg_store.save(chk2)
    
    # Should fetch the newer one
    latest = await pg_store.latest(run_id)
    assert latest is not None
    assert latest.checkpoint_id == "chk_2"
    assert latest.event_seq == 2


@pytest.mark.asyncio
async def test_latest_no_checkpoints(pg_store: PostgresCheckpointStore):
    latest = await pg_store.latest("run_non_existent")
    assert latest is None


@pytest.mark.asyncio
async def test_save_upsert(pg_store: PostgresCheckpointStore):
    """Saving a checkpoint with the same ID should upsert/overwrite gracefully."""
    run_id = "run_upsert"
    
    state = AgentState(run_id=run_id, history=[])
    chk = Checkpoint(
        checkpoint_id="chk_dup",
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        event_seq=5,
        state=state,
    )
    
    await pg_store.save(chk)
    
    # Save again with same ID
    chk_new = Checkpoint(
        checkpoint_id="chk_dup",
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        event_seq=6,
        state=state,
    )
    await pg_store.save(chk_new)
    
    latest = await pg_store.latest(run_id)
    assert latest is not None
    assert latest.event_seq == 6
