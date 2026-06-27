"""
Cheap, fast checks on the data types themselves, independent of the
engine or any storage. If these fail, nothing else in the test suite
is worth trusting.
"""

from datetime import datetime

from kestrion.core.types import (
    AgentState,
    Event,
    EventType,
    NodeResult,
    RunStatus,
)


def test_event_create_generates_unique_id_and_timestamp():
    evt1 = Event.create(run_id="run_123", type=EventType.RUN_STARTED)
    evt2 = Event.create(run_id="run_123", type=EventType.RUN_STARTED)

    assert evt1.event_id != evt2.event_id
    assert evt1.event_id.startswith("evt_")
    assert evt1.run_id == "run_123"
    assert evt1.type == EventType.RUN_STARTED
    assert isinstance(evt1.timestamp, datetime)
    assert evt1.timestamp.tzinfo is not None  # must be timezone-aware


def test_event_create_carries_payload_and_cost_fields():
    evt = Event.create(
        run_id="run_123",
        type=EventType.LLM_CALL_COMPLETED,
        payload={"model": "claude-sonnet-4-6"},
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.002,
    )
    assert evt.payload == {"model": "claude-sonnet-4-6"}
    assert evt.tokens_in == 100
    assert evt.tokens_out == 50
    assert evt.cost_usd == 0.002


def test_node_result_defaults_to_empty_event_list():
    result = NodeResult(next_node="next_step", state_updates={"foo": "bar"})
    assert result.events == []
    assert result.next_node == "next_step"
    assert result.state_updates == {"foo": "bar"}


def test_node_result_next_node_none_means_run_complete():
    result = NodeResult(next_node=None, state_updates={})
    assert result.next_node is None


def test_agent_state_defaults():
    state = AgentState(run_id="run_abc")
    assert state.status == RunStatus.PENDING
    assert state.current_node is None
    assert state.scratch == {}
    assert state.history == []
    assert state.total_tokens == 0
    assert state.last_event_seq == 0


def test_agent_state_round_trips_through_to_dict_and_from_dict():
    """
    This is the regression test for the pickle -> JSON serialization
    change: to_dict/from_dict must be lossless for everything a real
    run accumulates in scratch and history.
    """
    state = AgentState(
        run_id="run_xyz",
        status=RunStatus.WAITING_ON_HUMAN,
        current_node="apply_change",
        scratch={"proposed_yaml": "replicas: 3", "nested": {"a": [1, 2, 3]}},
        history=[{"type": "tool_result", "tool": "get_cluster_state"}],
        total_tokens=150,
        total_cost_usd=0.01,
        last_event_seq=7,
    )

    restored = AgentState.from_dict(state.to_dict())

    assert restored.run_id == state.run_id
    assert restored.status == state.status
    assert restored.current_node == state.current_node
    assert restored.scratch == state.scratch
    assert restored.history == state.history
    assert restored.total_tokens == state.total_tokens
    assert restored.total_cost_usd == state.total_cost_usd
    assert restored.last_event_seq == state.last_event_seq