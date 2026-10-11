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


def _detect_genai_system(model: str | None) -> str:
    """Infer the gen_ai.system attribute based on model name or default to kestrion."""
    if not model:
        return "kestrion"
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "o1" in m or "o3" in m or "text-embedding" in m or "openai" in m:
        return "openai"
    if "llama" in m or "mistral" in m or "qwen" in m or "ollama" in m:
        return "ollama"
    return "kestrion"


class OpenTelemetryProvider:
    """
    Translates Kestrion events into OpenTelemetry traces conforming to both
    legacy Kestrion span attributes and the official OpenTelemetry GenAI Semantic Conventions.
    
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
            # OTel GenAI Workflow conventions
            span.set_attribute("gen_ai.operation.name", "workflow")
            span.set_attribute("gen_ai.workflow.name", entry_node)
            span.set_attribute("gen_ai.workflow.run_id", event.run_id)
            self._active_runs[event.run_id] = span

        elif event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED, EventType.RUN_EXPIRED):
            run_span = self._active_runs.get(event.run_id)
            if run_span:
                del self._active_runs[event.run_id]
                run_span.set_attribute("kestrion.status", event.type.value)
                run_span.set_attribute("gen_ai.workflow.status", event.type.value)
                if event.type == EventType.RUN_FAILED:
                    err_msg = str(event.payload.get("error", "Run failed"))
                    run_span.set_status(trace.Status(trace.StatusCode.ERROR, err_msg))
                    run_span.set_attribute("error.type", "RunExecutionError")
                run_span.end()

        # LLM calls
        elif event.type == EventType.LLM_CALL_STARTED:
            span = self.tracer.start_span("kestrion.llm.call", context=parent_ctx)
            span.set_attribute("kestrion.run_id", event.run_id)
            # OTel GenAI Chat conventions
            model = event.payload.get("model")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", _detect_genai_system(model))
            if model:
                span.set_attribute("gen_ai.request.model", str(model))
            if "temperature" in event.payload:
                span.set_attribute("gen_ai.request.temperature", float(event.payload["temperature"]))
            if "top_p" in event.payload:
                span.set_attribute("gen_ai.request.top_p", float(event.payload["top_p"]))
            self._active_llms[event.run_id] = span
            
        elif event.type == EventType.LLM_CALL_COMPLETED:
            llm_span = self._active_llms.get(event.run_id)
            if llm_span:
                del self._active_llms[event.run_id]
                # Backwards compatible attributes
                llm_span.set_attribute("llm.usage.prompt_tokens", event.tokens_in)
                llm_span.set_attribute("llm.usage.completion_tokens", event.tokens_out)
                llm_span.set_attribute("kestrion.cost_usd", event.cost_usd)
                if stop_reason := event.payload.get("stop_reason"):
                    llm_span.set_attribute("llm.stop_reason", str(stop_reason))
                    # OTel GenAI convention: sequence of finish reasons
                    llm_span.set_attribute("gen_ai.response.finish_reasons", [str(stop_reason)])
                # Official OTel GenAI token usage conventions
                llm_span.set_attribute("gen_ai.usage.input_tokens", event.tokens_in)
                llm_span.set_attribute("gen_ai.usage.output_tokens", event.tokens_out)
                llm_span.set_attribute("gen_ai.usage.total_tokens", event.tokens_in + event.tokens_out)
                llm_span.set_attribute("gen_ai.output.cost", event.cost_usd)
                if model := event.payload.get("model"):
                    llm_span.set_attribute("gen_ai.response.model", str(model))
                llm_span.end()

        # Tool calls
        elif event.type == EventType.TOOL_CALL_STARTED:
            tool_name = event.payload.get("tool", "unknown")
            span = self.tracer.start_span(f"kestrion.tool.{tool_name}", context=parent_ctx)
            span.set_attribute("kestrion.run_id", event.run_id)
            span.set_attribute("kestrion.tool.name", tool_name)
            # OTel GenAI Tool conventions
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", tool_name)
            call_id = event.payload.get("tool_call_id") or event.payload.get("call_id")
            if call_id:
                span.set_attribute("gen_ai.tool.call.id", str(call_id))
            self._active_tools[(event.run_id, tool_name)] = span

        elif event.type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED):
            tool_name = event.payload.get("tool", "unknown")
            tool_span = self._active_tools.get((event.run_id, tool_name))
            if tool_span:
                del self._active_tools[(event.run_id, tool_name)]
                if event.type == EventType.TOOL_CALL_FAILED:
                    err_msg = str(event.payload.get("error", "Tool execution failed"))
                    tool_span.set_status(trace.Status(trace.StatusCode.ERROR, err_msg))
                    tool_span.set_attribute("kestrion.error_msg", err_msg)
                    tool_span.set_attribute("error.type", "ToolExecutionError")
                tool_span.end()
