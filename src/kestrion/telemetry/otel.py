"""
OpenTelemetry provider for Kestrion.

Provides observability by converting the immutable event stream into
standard OTel traces and spans.
"""
import logging
from typing import Dict, Tuple

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span
except ImportError as exc:
    raise ImportError(
        "The opentelemetry-api package is required to use OpenTelemetryProvider. "
        "Install it with: pip install opentelemetry-api"
    ) from exc

from kestrion.core.types import Event, EventType

logger = logging.getLogger(__name__)


class OpenTelemetryProvider:
    """
    Translates Kestrion events into OpenTelemetry traces.
    
    Maintains in-flight span state in memory to correlate START and COMPLETE
    events without polluting the durable event log with tracing context.
    """

    def __init__(self, tracer_name: str = "kestrion"):
        self.tracer = trace.get_tracer(tracer_name)
        
        # State tracking for in-flight spans
        self._active_runs: Dict[str, Span] = {}
        self._active_llms: Dict[str, Span] = {}
        self._active_tools: Dict[Tuple[str, str], Span] = {}

    async def on_event(self, event: Event) -> None:
        """Called by the Engine immediately when an event is durably recorded."""
        try:
            self._process_event(event)
        except Exception:
            logger.exception("Failed to process Kestrion event for telemetry")

    def _process_event(self, event: Event) -> None:
        parent_span = self._active_runs.get(event.run_id)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span else None

        if event.type == EventType.RUN_STARTED:
            entry_node = event.payload.get("entry_node", "agent")
            span = self.tracer.start_span(f"kestrion.run.{entry_node}")
            span.set_attribute("kestrion.run_id", event.run_id)
            self._active_runs[event.run_id] = span

        elif event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED, EventType.RUN_EXPIRED):
            run_span = self._active_runs.get(event.run_id)
            if run_span:
                del self._active_runs[event.run_id]
                run_span.set_attribute("kestrion.status", event.type.value)
                run_span.end()

        # LLM calls
        elif event.type == EventType.LLM_CALL_STARTED:
            span = self.tracer.start_span("kestrion.llm.call", context=parent_ctx)
            span.set_attribute("kestrion.run_id", event.run_id)
            self._active_llms[event.run_id] = span
            
        elif event.type == EventType.LLM_CALL_COMPLETED:
            llm_span = self._active_llms.get(event.run_id)
            if llm_span:
                del self._active_llms[event.run_id]
                llm_span.set_attribute("llm.usage.prompt_tokens", event.tokens_in)
                llm_span.set_attribute("llm.usage.completion_tokens", event.tokens_out)
                llm_span.set_attribute("kestrion.cost_usd", event.cost_usd)
                if stop_reason := event.payload.get("stop_reason"):
                    llm_span.set_attribute("llm.stop_reason", stop_reason)
                llm_span.end()

        # Tool calls
        elif event.type == EventType.TOOL_CALL_STARTED:
            tool_name = event.payload.get("tool", "unknown")
            span = self.tracer.start_span(f"kestrion.tool.{tool_name}", context=parent_ctx)
            span.set_attribute("kestrion.run_id", event.run_id)
            span.set_attribute("kestrion.tool.name", tool_name)
            self._active_tools[(event.run_id, tool_name)] = span

        elif event.type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED):
            tool_name = event.payload.get("tool", "unknown")
            tool_span = self._active_tools.get((event.run_id, tool_name))
            if tool_span:
                del self._active_tools[(event.run_id, tool_name)]
                if event.type == EventType.TOOL_CALL_FAILED:
                    tool_span.set_status(trace.Status(trace.StatusCode.ERROR))
                    tool_span.set_attribute("kestrion.error_msg", str(event.payload.get("error", "")))
                tool_span.end()
