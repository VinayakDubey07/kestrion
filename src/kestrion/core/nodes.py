"""
Core Nodes for Kestrion graphs.
Provides reusable node implementations for generic workflows, distinct from
the higher-level `Agent` wrapper.
"""

from __future__ import annotations


from kestrion.core.types import AgentState, Event, EventType, NodeResult
from kestrion.llm.base import LLMProvider

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Accurate token estimation if tiktoken is available, else fallback to chars // 4."""
    if HAS_TIKTOKEN:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            # Fallback for models tiktoken doesn't recognize explicitly
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    return len(text) // 4


class SummarizationNode:
    """
    A standalone node that monitors the conversation history length and
    compacts it using an LLM when it exceeds configured thresholds.
    Can be used by raw Engine users in any graph.
    """

    name = "summarization"

    def __init__(
        self,
        provider: LLMProvider,
        max_history_turns: int | None = None,
        max_history_tokens: int | None = None,
        keep_turns: int = 4,
        system_prompt: str | None = None,
        next_node: str | None = None,
    ):
        self.provider = provider
        self.max_history_turns = max_history_turns
        self.max_history_tokens = max_history_tokens
        self.keep_turns = keep_turns
        self.system_prompt = system_prompt
        self.next_node = next_node

    async def run(self, state: AgentState) -> NodeResult:
        messages_dicts = state.scratch.get("_messages", [])
        
        # If no configuration triggers compaction, just pass through
        if not self.max_history_turns and not self.max_history_tokens:
            return NodeResult(next_node=self.next_node, state_updates={})

        should_compact = False
        if self.max_history_turns is not None and len(messages_dicts) > self.max_history_turns:
            should_compact = True

        if not should_compact and self.max_history_tokens is not None:
            # Reconstruct content strings for counting
            full_text = " ".join(m.get("content", "") for m in messages_dicts)
            est_tokens = estimate_tokens(full_text)
            if est_tokens > self.max_history_tokens:
                should_compact = True

        if should_compact and len(messages_dicts) > self.keep_turns:
            keep = self.keep_turns
            if keep % 2 != 0:
                keep += 1

            to_compact = messages_dicts[:-keep]
            to_keep = messages_dicts[-keep:]

            from kestrion.llm.base import Message
            # Reconstruct messages for LLM
            summary_prompt = [
                Message(role=m.get("role", "user"), content=m.get("content", "")) for m in to_compact
            ]
            summary_prompt.append(
                Message(role="user", content="Summarize the preceding conversation turns concisely, retaining all key decisions, tasks, state, and context.")
            )

            # Query LLM
            summary_response = await self.provider.complete(
                messages=summary_prompt,
                tools=[],
                system=self.system_prompt,
            )

            summary_text = summary_response.text or ""

            if to_keep and to_keep[0].get("role") in ("user", "tool"):
                compacted_messages = [
                    {"role": "user", "content": f"[System Context: Summary of preceding conversation:\n{summary_text}]"},
                    {"role": "assistant", "content": "Understood. I will continue the conversation using this context."},
                    *to_keep
                ]
            else:
                compacted_messages = [
                    {"role": "user", "content": f"[System Context: Summary of preceding conversation:\n{summary_text}]"},
                    *to_keep
                ]

            # Emit compaction event directly so the engine folds it
            compact_event = Event.create(
                run_id=state.run_id,
                type=EventType.CONTEXT_COMPACTED,
                payload={
                    "original_turns": len(messages_dicts),
                    "compacted_turns": len(compacted_messages),
                    "summary": summary_text
                },
                node=self.name,
                tokens_in=summary_response.tokens_in,
                tokens_out=summary_response.tokens_out,
                cost_usd=summary_response.cost_usd,
            )
            # The engine will append this event and call _fold on it
            return NodeResult(next_node=self.next_node, state_updates={"_messages": compacted_messages}, events=[compact_event])

        return NodeResult(next_node=self.next_node, state_updates={})
