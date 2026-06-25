"""
The provider boundary. Agent never imports anthropic/openai/ollama
directly — it only ever talks to something satisfying LLMProvider. This
is the same Protocol pattern as CheckpointStore: swap the provider, zero
changes to agent.py or engine.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from kestrion.core.types import ToolSpec


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool the model wants to call, as decided by the provider."""
    id: str               # provider-assigned call id, needed to return results correctly
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """
    Normalized response shape across providers. Agent code reads this,
    never a raw Anthropic/OpenAI/Ollama response object — that raw shape
    differs per provider and leaking it upward would defeat the point of
    having a provider boundary at all.
    """
    text: str | None              # any plain-text content the model produced
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    stop_reason: str | None = None   # "end_turn", "tool_use", "max_tokens", etc.


@dataclass(frozen=True)
class Message:
    """
    A single turn in the conversation sent to the provider. Kept
    deliberately minimal — role + content, plus optional tool-result
    linkage. Providers translate this into their own wire format.
    """
    role: str                      # "user" | "assistant" | "tool"
    content: str | None = None
    tool_call_id: str | None = None      # set when role == "tool" (a tool result)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)  # set when role == "assistant" and it called tools


@runtime_checkable
class LLMProvider(Protocol):
    """
    Every provider (Anthropic, OpenAI, Ollama, ...) implements this.
    One method: given conversation history and the tools available,
    get the model's next move.
    """

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
    ) -> LLMResponse: ...