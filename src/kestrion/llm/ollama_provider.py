"""
Ollama provider — local models, zero cost, zero API key. Talks to a
local Ollama server over its HTTP API directly (no SDK dependency at
all, just httpx), since Ollama's wire format is simple enough not to
need one and this avoids adding yet another required package.

Note: tool-calling support varies significantly by which local model is
loaded. Not every model Ollama can run supports function calling well —
that's a model capability question, not something this provider can
paper over.
"""

from __future__ import annotations

import json

from kestrion.core.types import ToolSpec
from kestrion.llm.base import LLMResponse, Message, ToolCallRequest

try:
    import httpx
except ImportError as exc:
    raise ImportError(
        "The httpx package is required to use OllamaProvider. "
        "Install it with: pip install kestrion[ollama]"
    ) from exc


class OllamaProvider:
    """Implements the LLMProvider protocol structurally. No cost — local inference."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _to_ollama_messages(self, messages: list[Message], system: str | None) -> list[dict]:
        out = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "tool":
                # Ollama's chat API expects tool results as role="tool" content.
                out.append({"role": "tool", "content": m.content or ""})
            elif m.role == "assistant" and m.tool_calls:
                out.append({
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {"function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in m.tool_calls
                    ],
                })
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    def _to_ollama_tools(self, tools: list[ToolSpec]) -> list[dict]:
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
        payload = {
            "model": self.model,
            "messages": self._to_ollama_messages(messages, system),
            "stream": False,
        }
        if tools:
            payload["tools"] = self._to_ollama_tools(tools)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        tool_calls = []
        malformed_calls: list[str] = []
        for i, tc in enumerate(message.get("tool_calls", []) or []):
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", {})
            fn_name = fn.get("name", "")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    # Small local models sometimes emit malformed JSON
                    # arguments. Surfacing this as a synthetic tool call
                    # with an error lets the agent loop recover (the LLM
                    # sees the failure and can retry) instead of crashing
                    # the whole run with an unhandled exception.
                    malformed_calls.append(fn_name or f"call_{i}")
                    args = {}
            else:
                args = raw_args
            tool_calls.append(
                ToolCallRequest(id=f"ollama_call_{i}", name=fn_name, arguments=args)
            )

        text = message.get("content") or None
        if malformed_calls:
            note = (
                f"[kestrion: malformed tool-call arguments from model for: "
                f"{', '.join(malformed_calls)} — treated as empty arguments]"
            )
            text = f"{text}\n{note}" if text else note

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            cost_usd=0.0,  # local inference — no per-token billing
            stop_reason="tool_use" if tool_calls else "end_turn",
        )