import asyncio
from unittest.mock import MagicMock
import pytest

from kestrion.core.engine import Engine
from kestrion.core.types import Event, EventType
from kestrion.store.memory_store import MemoryCheckpointStore
from kestrion.telemetry.otel import OpenTelemetryProvider

@pytest.mark.asyncio
async def test_engine_emits_to_telemetry_provider():
    # Arrange
    store = MemoryCheckpointStore()
    mock_telemetry = MagicMock()
    
    # Needs to be async function for the mock since Engine awaits/tasks it
    async def mock_on_event(event):
        mock_telemetry(event)
        
    class MockTelemetryProvider:
        async def on_event(self, event: Event):
            await mock_on_event(event)

    provider = MockTelemetryProvider()
    engine = Engine(
        nodes={},
        tools={},
        store=store,
        entry_node="start",
        telemetry=provider
    )
    
    # Act
    # Calling start should emit RUN_STARTED
    # It will crash immediately since node "start" is missing, but that's fine,
    # the RUN_STARTED event should be emitted before driving the graph.
    try:
        await engine.start(run_id="test_run_123")
    except KeyError:
        pass
        
    # Wait briefly for the asyncio.create_task to execute
    await asyncio.sleep(0.01)

    # Assert
    assert mock_telemetry.call_count == 1
    event = mock_telemetry.call_args[0][0]
    assert event.type == EventType.RUN_STARTED
    assert event.run_id == "test_run_123"

@pytest.mark.asyncio
async def test_otel_provider_handles_events():
    # Integration test for OpenTelemetryProvider itself
    # We won't assert the actual OTel SDK spans because setting up the SDK exporter is complex,
    # but we will verify it handles events without crashing.
    provider = OpenTelemetryProvider(tracer_name="test_tracer")
    
    run_id = "test_run_456"
    
    # RUN_STARTED
    await provider.on_event(Event.create(run_id, EventType.RUN_STARTED, payload={"entry_node": "agent_loop"}))
    assert run_id in provider._active_runs
    
    # LLM_CALL_STARTED
    await provider.on_event(Event.create(run_id, EventType.LLM_CALL_STARTED, payload={}))
    assert run_id in provider._active_llms
    
    # LLM_CALL_COMPLETED
    await provider.on_event(Event.create(run_id, EventType.LLM_CALL_COMPLETED, payload={"stop_reason": "end_turn"}, tokens_in=10, tokens_out=5, cost_usd=0.01))
    assert run_id not in provider._active_llms
    
    # TOOL_CALL_STARTED
    await provider.on_event(Event.create(run_id, EventType.TOOL_CALL_STARTED, payload={"tool": "search"}))
    assert (run_id, "search") in provider._active_tools
    
    # TOOL_CALL_COMPLETED
    await provider.on_event(Event.create(run_id, EventType.TOOL_CALL_COMPLETED, payload={"tool": "search", "output": "results"}))
    assert (run_id, "search") not in provider._active_tools
    
    # RUN_COMPLETED
    await provider.on_event(Event.create(run_id, EventType.RUN_COMPLETED, payload={}))
    assert run_id not in provider._active_runs


def test_detect_genai_system():
    from kestrion.telemetry.otel import _detect_genai_system

    assert _detect_genai_system("claude-3-5-sonnet") == "anthropic"
    assert _detect_genai_system("gpt-4o") == "openai"
    assert _detect_genai_system("o1-mini") == "openai"
    assert _detect_genai_system("llama-3-8b") == "ollama"
    assert _detect_genai_system("custom-model") == "kestrion"
    assert _detect_genai_system(None) == "kestrion"


@pytest.mark.asyncio
async def test_otel_provider_genai_attributes_and_failures():
    provider = OpenTelemetryProvider(tracer_name="test_genai_tracer")
    run_id = "test_run_genai_999"

    # Start run
    await provider.on_event(
        Event.create(run_id, EventType.RUN_STARTED, payload={"entry_node": "agent_loop"})
    )
    run_span = provider._active_runs[run_id]
    assert run_span is not None

    # LLM started with model and temperature
    await provider.on_event(
        Event.create(
            run_id,
            EventType.LLM_CALL_STARTED,
            payload={"model": "claude-3-5-sonnet", "temperature": 0.7, "top_p": 0.95},
        )
    )
    assert run_id in provider._active_llms

    # LLM completed
    await provider.on_event(
        Event.create(
            run_id,
            EventType.LLM_CALL_COMPLETED,
            payload={"model": "claude-3-5-sonnet", "stop_reason": "end_turn"},
            tokens_in=50,
            tokens_out=25,
            cost_usd=0.001,
        )
    )
    assert run_id not in provider._active_llms

    # Tool started and failed
    await provider.on_event(
        Event.create(
            run_id,
            EventType.TOOL_CALL_STARTED,
            payload={"tool": "sql_query", "tool_call_id": "call_abc"},
        )
    )
    await provider.on_event(
        Event.create(
            run_id,
            EventType.TOOL_CALL_FAILED,
            payload={"tool": "sql_query", "error": "syntax error"},
        )
    )
    assert (run_id, "sql_query") not in provider._active_tools

    # Run failed
    await provider.on_event(
        Event.create(run_id, EventType.RUN_FAILED, payload={"error": "Fatal run error"})
    )
    assert run_id not in provider._active_runs


@pytest.mark.asyncio
async def test_agent_accepts_list_of_telemetry_providers():
    from kestrion.agent.agent import Agent
    from kestrion.observability.logging import CompositeTelemetryProvider

    p1 = MagicMock()
    p2 = MagicMock()

    mock_llm = MagicMock()
    agent = Agent(provider=mock_llm, telemetry=[p1, p2])

    assert isinstance(agent._engine.telemetry, CompositeTelemetryProvider)
    assert p1 in agent._engine.telemetry.providers
    assert p2 in agent._engine.telemetry.providers

