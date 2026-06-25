"""
Tests the store in isolation from the engine. If these fail, any failure
in test_engine.py involving resume/crash-recovery is ambiguous — it could
be the engine or the store. Keeping these separate makes failures easier
to diagnose.
"""

from datetime import datetime, timezone

import pytest

from kestrion.core.types import AgentState, Checkpoint, Event, EventType, RunStatus, new_id


async def test_append_event_returns_increasing_sequence_numbers(tmp_store):
    evt1 = Event.create(run_id="run_1", type=EventType.RUN_STARTED)
    evt2 = Event.create(run_id="run_1", type=EventType.RUN_COMPLETED)

    seq1 = await tmp_store.append_event(evt1)
    seq2 = await tmp_store.append_event(evt2)

    assert seq2 > seq1


async def test_events_since_returns_events_in_order(tmp_store):
    for i in range(5):
        await tmp_store.append_event(
            Event.create(run_id="run_1", type=EventType.STATE_TRANSITION, payload={"i": i})
        )

    events = await tmp_store.events_since("run_1", 0)

    assert len(events) == 5
    assert [e.payload["i"] for e in events] == [0, 1, 2, 3, 4]


async def test_events_since_only_returns_events_after_given_seq(tmp_store):
    seqs = []
    for i in range(5):
        seq = await tmp_store.append_event(
            Event.create(run_id="run_1", type=EventType.STATE_TRANSITION, payload={"i": i})
        )
        seqs.append(seq)

    events = await tmp_store.events_since("run_1", seqs[2])

    assert len(events) == 2
    assert [e.payload["i"] for e in events] == [3, 4]


async def test_events_since_filters_by_run_id(tmp_store):
    await tmp_store.append_event(Event.create(run_id="run_A", type=EventType.RUN_STARTED))
    await tmp_store.append_event(Event.create(run_id="run_B", type=EventType.RUN_STARTED))
    await tmp_store.append_event(Event.create(run_id="run_A", type=EventType.RUN_COMPLETED))

    events_a = await tmp_store.events_since("run_A", 0)
    events_b = await tmp_store.events_since("run_B", 0)

    assert len(events_a) == 2
    assert len(events_b) == 1


async def test_event_round_trip_preserves_all_fields(tmp_store):
    original = Event.create(
        run_id="run_1",
        type=EventType.TOOL_CALL_COMPLETED,
        payload={"tool": "get_cluster_state", "output": {"replicas": 2}},
        node="inspect_cluster",
        tokens_in=42,
        tokens_out=17,
        cost_usd=0.0015,
    )
    await tmp_store.append_event(original)

    [retrieved] = await tmp_store.events_since("run_1", 0)

    assert retrieved.event_id == original.event_id
    assert retrieved.run_id == original.run_id
    assert retrieved.type == original.type
    assert retrieved.payload == original.payload
    assert retrieved.node == original.node
    assert retrieved.tokens_in == original.tokens_in
    assert retrieved.tokens_out == original.tokens_out
    assert retrieved.cost_usd == original.cost_usd


async def test_latest_returns_none_when_no_checkpoint_exists(tmp_store):
    result = await tmp_store.latest("run_nonexistent")
    assert result is None


async def test_checkpoint_save_and_latest_round_trip(tmp_store):
    state = AgentState(
        run_id="run_1",
        status=RunStatus.RUNNING,
        current_node="propose_change",
        scratch={"current_replicas": 2},
        last_event_seq=3,
    )
    checkpoint = Checkpoint(
        checkpoint_id=new_id("ckpt"),
        run_id="run_1",
        state=state,
        created_at=datetime.now(timezone.utc),
        event_seq=3,
    )

    await tmp_store.save(checkpoint)
    retrieved = await tmp_store.latest("run_1")

    assert retrieved is not None
    assert retrieved.checkpoint_id == checkpoint.checkpoint_id
    assert retrieved.event_seq == 3
    assert retrieved.state.run_id == "run_1"
    assert retrieved.state.status == RunStatus.RUNNING
    assert retrieved.state.current_node == "propose_change"
    assert retrieved.state.scratch == {"current_replicas": 2}


async def test_latest_returns_most_recent_checkpoint_when_multiple_exist(tmp_store):
    for seq in [1, 2, 3]:
        state = AgentState(run_id="run_1", current_node=f"node_{seq}", last_event_seq=seq)
        await tmp_store.save(
            Checkpoint(
                checkpoint_id=new_id("ckpt"),
                run_id="run_1",
                state=state,
                created_at=datetime.now(timezone.utc),
                event_seq=seq,
            )
        )

    latest = await tmp_store.latest("run_1")

    assert latest.event_seq == 3
    assert latest.state.current_node == "node_3"


async def test_save_rejects_non_json_serializable_scratch(tmp_store):
    """
    Regression test for the pickle -> JSON swap: a non-serializable value
    in scratch must fail loudly, not silently corrupt or drop data.
    """
    class NotSerializable:
        pass

    state = AgentState(run_id="run_1", scratch={"bad": NotSerializable()})
    checkpoint = Checkpoint(
        checkpoint_id=new_id("ckpt"),
        run_id="run_1",
        state=state,
        created_at=datetime.now(timezone.utc),
        event_seq=0,
    )

    with pytest.raises(ValueError, match="non-JSON-serializable"):
        await tmp_store.save(checkpoint)
