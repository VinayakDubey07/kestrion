"""
Enterprise Structured JSON Logging for Kestrion.

Provides machine-readable NDJSON (Newline Delimited JSON) logs for modern
observability platforms (Datadog, AWS CloudWatch, Google Cloud Logging, Splunk,
Elasticsearch, and Grafana Loki).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

from kestrion.core.types import Event, EventType, TelemetryProvider


class JSONLogFormatter(logging.Formatter):
    """
    Formats standard library LogRecords into structured single-line JSON.

    Standard keys:
    - timestamp: ISO-8601 UTC timestamp
    - level: Log level string (INFO, WARNING, ERROR, etc.)
    - logger: Logger hierarchy name
    - message: Log message
    - Any custom keys passed in the LogRecord (run_id, event_type, duration_ms, etc.)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom event metadata if present
        standard_record_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message",
        }
        for key, val in record.__dict__.items():
            if key not in standard_record_attrs and not key.startswith("_"):
                data[key] = val

        # Include exception details if present
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            data["exception"] = record.exc_text

        return json.dumps(data, default=str)


class StructuredLoggerProvider(TelemetryProvider):
    """
    Telemetry provider that emits structured JSON log records for every engine event.

    Features:
    - Zero external dependencies (uses standard library logging + json).
    - Automatically measures and records latency (duration_ms) for runs,
      LLM invocations, and tool executions.
    - Privacy-safe by default: payload inspection can be enabled via `include_payload=True`.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        include_payload: bool = False,
    ):
        self.logger = logger or logging.getLogger("kestrion.events")
        self.level = level
        self.include_payload = include_payload

        # In-flight timers to calculate elapsed duration_ms
        self._run_start_times: dict[str, float] = {}
        self._llm_start_times: dict[str, float] = {}
        self._tool_start_times: dict[tuple[str, str], float] = {}

    async def on_event(self, event: Event) -> None:
        """Called whenever an engine event is durably recorded."""
        now = time.perf_counter()
        duration_ms: float | None = None

        # Track and compute durations
        if event.type == EventType.RUN_STARTED:
            self._run_start_times[event.run_id] = now
        elif event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED, EventType.RUN_EXPIRED):
            start = self._run_start_times.pop(event.run_id, None)
            if start is not None:
                duration_ms = round((now - start) * 1000, 2)

        elif event.type == EventType.LLM_CALL_STARTED:
            self._llm_start_times[event.run_id] = now
        elif event.type == EventType.LLM_CALL_COMPLETED:
            start = self._llm_start_times.pop(event.run_id, None)
            if start is not None:
                duration_ms = round((now - start) * 1000, 2)

        elif event.type == EventType.TOOL_CALL_STARTED:
            tool_name = str(event.payload.get("tool", "unknown"))
            self._tool_start_times[(event.run_id, tool_name)] = now
        elif event.type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED):
            tool_name = str(event.payload.get("tool", "unknown"))
            start = self._tool_start_times.pop((event.run_id, tool_name), None)
            if start is not None:
                duration_ms = round((now - start) * 1000, 2)

        # Build log record extra attributes
        extra: dict[str, Any] = {
            "run_id": event.run_id,
            "event_id": event.event_id,
            "event_type": event.type.value,
            "node": event.node,
        }

        if event.tokens_in > 0:
            extra["tokens_in"] = event.tokens_in
        if event.tokens_out > 0:
            extra["tokens_out"] = event.tokens_out
        if event.cost_usd > 0.0:
            extra["cost_usd"] = event.cost_usd
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms

        if self.include_payload and event.payload:
            extra["payload"] = event.payload

        # Craft human-readable message while metadata sits in JSON fields
        msg = f"Event {event.type.value} on run {event.run_id}"
        if event.node:
            msg += f" (node: {event.node})"

        # Determine log level (warn/error on failures)
        lvl = self.level
        if event.type in (EventType.RUN_FAILED, EventType.TOOL_CALL_FAILED):
            lvl = logging.ERROR
            if "error" in event.payload:
                extra["error"] = str(event.payload["error"])
        elif event.type == EventType.RUN_EXPIRED:
            lvl = logging.WARNING

        self.logger.log(lvl, msg, extra=extra)


class CompositeTelemetryProvider(TelemetryProvider):
    """
    Combines multiple telemetry providers into one.
    Dispatches each event sequentially to all registered providers.
    """

    def __init__(self, providers: list[TelemetryProvider]):
        self.providers = list(providers)

    async def on_event(self, event: Event) -> None:
        for provider in self.providers:
            try:
                await provider.on_event(event)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Error in telemetry provider %r handling event %s", provider, event.type.value
                )


def configure_structured_logging(
    level: int = logging.INFO,
    stream: Any = None,
    logger_name: str = "kestrion",
) -> logging.StreamHandler:
    """
    Convenience helper to configure stdout structured JSON logging for a logger.

    Usage::

        from kestrion.observability import configure_structured_logging
        configure_structured_logging(level=logging.INFO)
    """
    target_logger = logging.getLogger(logger_name)
    target_logger.setLevel(level)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JSONLogFormatter())

    target_logger.addHandler(handler)
    return handler
