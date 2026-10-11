"""
Unit tests for kestrion.observability.logging (JSONLogFormatter, StructuredLoggerProvider, CompositeTelemetryProvider).
"""

import io
import json
import logging
from unittest.mock import AsyncMock

import pytest

from kestrion.core.types import Event, EventType
from kestrion.observability.logging import (
    CompositeTelemetryProvider,
    JSONLogFormatter,
    StructuredLoggerProvider,
    configure_structured_logging,
)


def test_json_log_formatter_basic():
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Agent completed task",
        args=(),
        exc_info=None,
    )
    record.run_id = "run_123"  # type: ignore[attr-defined]
    record.tokens_in = 150  # type: ignore[attr-defined]

    output = formatter.format(record)
    data = json.loads(output)

    assert data["message"] == "Agent completed task"
    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["run_id"] == "run_123"
    assert data["tokens_in"] == 150
    assert "timestamp" in data


def test_json_log_formatter_with_exception():
    formatter = JSONLogFormatter()
    try:
        raise ValueError("Something exploded")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="error_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=20,
        msg="Error occurred",
        args=(),
        exc_info=exc_info,
    )

    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "ERROR"
    assert "exception" in data
    assert "ValueError: Something exploded" in data["exception"]


@pytest.mark.asyncio
async def test_structured_logger_provider_lifecycle():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONLogFormatter())

    logger = logging.getLogger("test_structured_events")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    provider = StructuredLoggerProvider(logger=logger, include_payload=True)
    run_id = "run_life_1"

    # 1. RUN_STARTED
    await provider.on_event(
        Event.create(run_id, EventType.RUN_STARTED, payload={"entry_node": "agent"})
    )

    # 2. LLM_CALL_STARTED & COMPLETED
    await provider.on_event(Event.create(run_id, EventType.LLM_CALL_STARTED))
    await provider.on_event(
        Event.create(
            run_id,
            EventType.LLM_CALL_COMPLETED,
            payload={"stop_reason": "end_turn"},
            tokens_in=100,
            tokens_out=40,
            cost_usd=0.003,
        )
    )

    # 3. TOOL_CALL_STARTED & COMPLETED
    await provider.on_event(
        Event.create(run_id, EventType.TOOL_CALL_STARTED, payload={"tool": "calc"})
    )
    await provider.on_event(
        Event.create(
            run_id, EventType.TOOL_CALL_COMPLETED, payload={"tool": "calc", "output": "42"}
        )
    )

    # 4. RUN_COMPLETED
    await provider.on_event(
        Event.create(run_id, EventType.RUN_COMPLETED, payload={"result": "done"})
    )

    # Parse logged JSON lines
    lines = [json.loads(line) for line in stream.getvalue().strip().split("\n") if line]
    assert len(lines) == 6

    # Verify run started
    assert lines[0]["event_type"] == EventType.RUN_STARTED.value
    assert lines[0]["run_id"] == run_id

    # Verify LLM call completed has tokens, cost, duration
    llm_comp = lines[2]
    assert llm_comp["event_type"] == EventType.LLM_CALL_COMPLETED.value
    assert llm_comp["tokens_in"] == 100
    assert llm_comp["tokens_out"] == 40
    assert llm_comp["cost_usd"] == 0.003
    assert "duration_ms" in llm_comp

    # Verify tool call completed has duration
    tool_comp = lines[4]
    assert tool_comp["event_type"] == EventType.TOOL_CALL_COMPLETED.value
    assert "duration_ms" in tool_comp

    # Verify run completed has duration
    run_comp = lines[5]
    assert run_comp["event_type"] == EventType.RUN_COMPLETED.value
    assert "duration_ms" in run_comp


@pytest.mark.asyncio
async def test_structured_logger_provider_error_level():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONLogFormatter())

    logger = logging.getLogger("test_error_events")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    provider = StructuredLoggerProvider(logger=logger)
    run_id = "run_err_1"

    # TOOL_CALL_FAILED should log with ERROR level
    await provider.on_event(
        Event.create(
            run_id,
            EventType.TOOL_CALL_FAILED,
            payload={"tool": "broken_tool", "error": "Disk full"},
        )
    )

    # RUN_FAILED should log with ERROR level
    await provider.on_event(
        Event.create(run_id, EventType.RUN_FAILED, payload={"error": "Agent crashed"})
    )

    lines = [json.loads(line) for line in stream.getvalue().strip().split("\n") if line]
    assert len(lines) == 2
    assert lines[0]["level"] == "ERROR"
    assert lines[0]["error"] == "Disk full"
    assert lines[1]["level"] == "ERROR"
    assert lines[1]["error"] == "Agent crashed"


@pytest.mark.asyncio
async def test_composite_telemetry_provider():
    p1 = AsyncMock()
    p2 = AsyncMock()

    composite = CompositeTelemetryProvider([p1, p2])
    ev = Event.create("run_comp_1", EventType.RUN_STARTED)

    await composite.on_event(ev)

    p1.on_event.assert_awaited_once_with(ev)
    p2.on_event.assert_awaited_once_with(ev)


@pytest.mark.asyncio
async def test_composite_telemetry_provider_fault_tolerant():
    p_failing = AsyncMock()
    p_failing.on_event.side_effect = RuntimeError("Telemetry failed")
    p_working = AsyncMock()

    composite = CompositeTelemetryProvider([p_failing, p_working])
    ev = Event.create("run_comp_2", EventType.RUN_STARTED)

    # Should not raise exception
    await composite.on_event(ev)
    p_working.on_event.assert_awaited_once_with(ev)


def test_configure_structured_logging():
    stream = io.StringIO()
    handler = configure_structured_logging(
        level=logging.DEBUG, stream=stream, logger_name="test_config_log"
    )
    assert isinstance(handler, logging.StreamHandler)

    log = logging.getLogger("test_config_log")
    log.info("Hello structured world")

    raw = stream.getvalue().strip()
    data = json.loads(raw)
    assert data["message"] == "Hello structured world"
    assert data["level"] == "INFO"
