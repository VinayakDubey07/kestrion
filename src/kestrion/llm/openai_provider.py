"""
OpenAI provider. Requires the `openai` package — install via
`pip install kestrion[openai]`.

Also usable for ANY OpenAI-compatible API (many hosted providers mimic
this wire format) by passing a custom base_url.
"""

from __future__ import annotations

import json

from kestrion.core.types import ToolSpec
from kestrion.llm.base import LLMResponse, Message, ToolCallRequest

try:
    import openai
except ImportError as exc:
    raise ImportError(
        "The openai package is required to use OpenAIProvider. "
        "Install it with: pip install kestrion[openai]"
    ) from exc


_PRICING_PER_MTOK = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = _PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out


class OpenAIProvider:
    """Implements the LLMProvider protocol structurally."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _to_openai_messages(self, messages: list[Message], system: str | None) -> list[dict]:
        out = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""})
            elif m.role == "assistant" and m.tool_calls:
                out.append({
                    "role": "assistant",
                    "content": m.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ],
                })
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    def _to_openai_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
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
            "messages": self._to_openai_messages(messages, system),
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCallRequest(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        return LLMResponse(
            text=choice.message.content,
            tool_calls=tool_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=_estimate_cost(self.model, tokens_in, tokens_out),
            stop_reason=choice.finish_reason,
        )