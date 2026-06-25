"""
Anthropic provider. Requires the `anthropic` package — install via
`pip install kestrion[anthropic]`, not bundled into the core install,
since someone running Ollama-only shouldn't need this SDK at all.
"""

from __future__ import annotations

from kestrion.core.types import ToolSpec
from kestrion.llm.base import LLMResponse, Message, ToolCallRequest

try:
    import anthropic
except ImportError as exc:
    raise ImportError(
        "The anthropic package is required to use AnthropicProvider. "
        "Install it with: pip install kestrion[anthropic]"
    ) from exc


# Rough per-model pricing for cost tracking. Kept here rather than
# hardcoded inline so it's the one place to update as pricing changes.
# Values are USD per million tokens (input, output).
_PRICING_PER_MTOK = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = _PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out


class AnthropicProvider:
    """Implements the LLMProvider protocol structurally — no inheritance needed."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key)  # falls back to ANTHROPIC_API_KEY env var

    def _to_anthropic_messages(self, messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            if m.role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content or "",
                    }],
                })
            elif m.role == "assistant" and m.tool_calls:
                blocks = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    def _to_anthropic_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        response = await self._client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, arguments=block.input))

        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens

        return LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=_estimate_cost(self.model, tokens_in, tokens_out),
            stop_reason=response.stop_reason,
        )