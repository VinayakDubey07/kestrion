"""
Unit tests for kestrion.adapters.vercel (Vercel AI SDK Data Stream Protocol adapter).
"""

import json
from unittest.mock import AsyncMock, MagicMock

from kestrion.adapters.vercel import _format_vercel_events, stream_to_vercel
from kestrion.core.types import Event, EventType


def test_format_vercel_events_text_chunk():
    ev = Event.create(
        run_id="test_run",
        type=EventType.LLM_CALL_COMPLETED,
        payload={"text": "Hello world!"},
    )
    chunks = _format_vercel_events([ev])
    assert len(chunks) == 1
    assert chunks[0] == f'0:{json.dumps("Hello world!")}\n'


def test_format_vercel_events_tool_call():
    ev = Event.create(
        run_id="test_run",
        type=EventType.LLM_CALL_COMPLETED,
        payload={
            "text": "",
            "tool_calls": [
                {"id": "call_123", "name": "weather", "arguments": {"city": "Paris"}}
            ],
        },
    )
    chunks = _format_vercel_events([ev])
    assert len(chunks) == 2
    assert chunks[0].startswith("9:")
    assert "weather" in chunks[0]
    assert chunks[1].startswith("d:")
    assert "tool_call_ui" in chunks[1]


def test_format_vercel_events_tool_result():
    ev = Event.create(
        run_id="test_run",
        type=EventType.TOOL_CALL_COMPLETED,
        payload={"tool_call_id": "call_123", "output": "Sunny, 22C"},
    )
    chunks = _format_vercel_events([ev])
    assert len(chunks) == 1
    assert chunks[0].startswith("8:")
    data = json.loads(chunks[0][2:])
    assert data[0]["toolCallId"] == "call_123"
    assert data[0]["result"] == "Sunny, 22C"


def test_format_vercel_events_run_failed():
    ev = Event.create(
        run_id="test_run",
        type=EventType.RUN_FAILED,
        payload={"error": "Rate limit exceeded"},
    )
    chunks = _format_vercel_events([ev])
    assert len(chunks) == 1
    assert chunks[0].startswith("e:")
    assert "Rate limit exceeded" in chunks[0]


async def test_stream_to_vercel_success():
    ev = Event.create(
        run_id="run_vercel_1",
        type=EventType.LLM_CALL_COMPLETED,
        payload={"text": "Streaming output"},
    )

    mock_agent = MagicMock()
    mock_agent._engine = MagicMock()
    mock_agent._engine.store = MagicMock()
    mock_agent._engine.store.events_since = AsyncMock(return_value=[ev])

    async def mock_run(prompt, run_id=None):
        return "Streaming output"

    mock_agent.run = mock_run

    chunks = []
    async for chunk in stream_to_vercel(mock_agent, "test prompt", run_id="run_vercel_1"):
        chunks.append(chunk)

    assert any(chunk.startswith("0:") and "Streaming output" in chunk for chunk in chunks)


async def test_stream_to_vercel_error_handling():
    mock_agent = MagicMock()
    mock_agent._engine = MagicMock()
    mock_agent._engine.store = MagicMock()
    mock_agent._engine.store.events_since = AsyncMock(return_value=[])

    async def mock_run_fail(prompt, run_id=None):
        raise RuntimeError("Agent crashed hard")

    mock_agent.run = mock_run_fail

    chunks = []
    async for chunk in stream_to_vercel(mock_agent, "test prompt", run_id="run_vercel_err"):
        chunks.append(chunk)

    assert any(chunk.startswith("e:") and "Agent crashed hard" in chunk for chunk in chunks)
