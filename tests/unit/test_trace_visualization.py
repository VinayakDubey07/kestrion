import pytest
from datetime import datetime
from kestrion.core.types import Event, EventType
from kestrion.cli.main import generate_mermaid_flowchart

def test_generate_mermaid_flowchart_standard():
    # Setup some test events
    events = [
        Event(
            event_id="evt_1",
            run_id="run_123",
            type=EventType.RUN_STARTED,
            timestamp=datetime.now(),
            node="engine"
        ),
        Event(
            event_id="evt_2",
            run_id="run_123",
            type=EventType.STATE_TRANSITION,
            timestamp=datetime.now(),
            node="engine",
            payload={"from": "START", "to": "node_a"}
        ),
        Event(
            event_id="evt_3",
            run_id="run_123",
            type=EventType.LLM_CALL_COMPLETED,
            timestamp=datetime.now(),
            node="node_a",
            tokens_in=10,
            tokens_out=20
        ),
        Event(
            event_id="evt_4",
            run_id="run_123",
            type=EventType.TOOL_CALL_STARTED,
            timestamp=datetime.now(),
            node="node_a",
            payload={"tool": "get_weather"}
        ),
        Event(
            event_id="evt_5",
            run_id="run_123",
            type=EventType.STATE_TRANSITION,
            timestamp=datetime.now(),
            node="engine",
            payload={"from": "node_a", "to": "node_b"}
        ),
        Event(
            event_id="evt_6",
            run_id="run_123",
            type=EventType.HUMAN_INTERVENTION,
            timestamp=datetime.now(),
            node="node_b",
            payload={"tool": "deploy_prod"}
        ),
        Event(
            event_id="evt_7",
            run_id="run_123",
            type=EventType.RUN_COMPLETED,
            timestamp=datetime.now(),
            node="node_b"
        )
    ]

    graph = generate_mermaid_flowchart(events, "run_123")

    # Assert layout and comments
    assert "flowchart TD" in graph
    assert "%% Mermaid trace for run run_123" in graph

    # Assert nodes are created
    assert "START([START])" in graph
    assert "END([END])" in graph
    assert 'node_a["Node: node_a"]' in graph
    assert 'node_b["Node: node_b"]' in graph

    # Assert sub-elements are created
    assert 'tool_2(["Tool: get_weather"])' in graph
    assert 'llm_1{"LLM Call<br/>10 in / 20 out"}' in graph
    assert 'human_3[/"Human Gate: deploy_prod"/]' in graph

    # Assert transitions exist
    assert "START --> node_a" in graph
    assert "node_a --> tool_2" in graph
    assert "node_a --> llm_1" in graph
    assert "node_a --> node_b" in graph
    assert "node_b --> human_3" in graph
    assert "node_b --> END" in graph

    # Assert styles are applied
    assert "style START fill:#10b981" in graph
    assert "style END fill:#10b981" in graph
