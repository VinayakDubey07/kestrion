"""
Agent is the ergonomic, user-facing API. Underneath, it's a single
implicit Engine node that runs a "call the LLM, execute whatever tools
it asks for, feed results back, repeat" loop until the model produces
a final answer with no more tool calls.

Design choice worth being explicit about: this loop lives INSIDE one
Node's run() call, not spread across multiple graph nodes. That means
from the Engine's perspective, an entire multi-tool-call conversation
turn is one "step" — checkpointed once it's done, not after every
individual tool call within the turn. Tradeoff: simpler mental model
and fewer checkpoint writes, at the cost of slightly coarser resume
granularity (if the process dies mid-turn, you replay that turn, not
resume from the middle of it). For approval-gated tools this doesn't
matter — ApprovalRequired still propagates out and pauses the run
exactly as before, since it's a real exception, not swallowed by the
loop.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass

from kestrion.core.engine import Engine
from kestrion.core.types import AgentState, Event, EventType, NodeResult, RunStatus, Tool
from kestrion.llm.base import LLMProvider, LLMResponse, Message, ToolCallRequest
from kestrion.store.sqlite_store import SQLiteCheckpointStore


def _message_to_dict(m: Message) -> dict:
    """
    dataclasses.asdict() handles the nested ToolCallRequest list
    correctly (plain __dict__ does not — it only goes one level deep,
    leaving ToolCallRequest objects unconverted and therefore not
    JSON-serializable when the engine checkpoints state.scratch).
    """
    return asdict(m)


def _message_from_dict(d: dict) -> Message:
    tool_calls = [ToolCallRequest(**tc) for tc in d.get("tool_calls", [])]
    return Message(
        role=d["role"],
        content=d.get("content"),
        tool_call_id=d.get("tool_call_id"),
        tool_calls=tool_calls,
    )


@dataclass
class RunResult:
    """What Agent.run()/.resume() hands back — a thin, friendly view over AgentState."""
    run_id: str
    status: RunStatus
    output: str | None          # the model's final text answer, if the run completed
    state: AgentState           # full state, for anyone who needs scratch/history/cost


def _store_from_url(url: str):
    """
    Parses a store= string into a concrete CheckpointStore. Only sqlite
    is implemented today; this function is the seam where postgres://
    gets added later without changing any Agent call sites.
    """
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        return SQLiteCheckpointStore(path=path)
    raise ValueError(f"Unsupported store URL scheme: {url!r}. Supported: sqlite:///path/to/file.db")


class _AgentLoopNode:
    """
    The single implicit node every Agent run uses. Holds a reference to
    the owning Agent so it can reach the LLM provider and tool registry.
    """

    name = "agent_loop"

    def __init__(self, agent: "Agent"):
        self._agent = agent

    async def run(self, state: AgentState) -> NodeResult:
        agent = self._agent
        # Conversation history is rebuilt from state.scratch each turn —
        # state is the only thing that survives a resume, so it must be
        # the sole source of truth for what's been said so far.
        messages: list[Message] = [
            _message_from_dict(m) for m in state.scratch.get("_messages", [])
        ]

        while True:
            response: LLMResponse = await agent._provider.complete(
                messages=messages,
                tools=[t.spec for t in agent._tools.values()],
                system=agent.system_prompt,
            )

            # Emitted immediately, not batched into the eventual NodeResult.
            # Reason: if a gated tool call later in this same turn raises
            # ApprovalRequired, node.run() exits via exception and never
            # returns a NodeResult at all — anything batched would be
            # silently lost. engine.record_llm_call durably appends the
            # event before that can happen.
            llm_event = Event.create(
                run_id=state.run_id,
                type=EventType.LLM_CALL_COMPLETED,
                payload={"stop_reason": response.stop_reason},
                node=self.name,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                cost_usd=response.cost_usd,
            )
            await agent._engine.record_event(state, llm_event)

            assistant_msg = Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
            messages.append(assistant_msg)

            if not response.tool_calls:
                # Model is done — no more tools to call. Persist the full
                # conversation back into scratch and end the run.
                return NodeResult(
                    next_node=None,
                    state_updates={
                        "_messages": [_message_to_dict(m) for m in messages],
                        "final_output": response.text,
                    },
                )

            # Execute every requested tool call. ApprovalRequired propagates
            # naturally — engine.call_tool raises it, this loop doesn't
            # catch it, Engine._drive's except clause handles the pause.
            for call in response.tool_calls:
                result = await agent._engine.call_tool(state, call.name, **call.arguments)
                messages.append(
                    Message(role="tool", tool_call_id=call.id, content=str(result.output))
                )

            # Persist progress before looping again, so a crash mid-loop
            # (after some tool calls but before the model's next reply)
            # doesn't lose the tool results already gathered. The Engine
            # checkpoints after this node returns either way, but storing
            # the in-progress message list in scratch here means even a
            # raised ApprovalRequired carries the latest messages with it.
            state.scratch["_messages"] = [_message_to_dict(m) for m in messages]


class Agent:
    """
    Agent(model=..., tools=[...], store=...) — the ergonomic entry point.
    Wraps a single-node Engine running the LLM tool-calling loop above.
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Tool] | None = None,
        store: str = "sqlite:///kestrion_runs.db",
        system_prompt: str | None = None,
    ):
        self._provider = provider
        self._tools: dict[str, Tool] = {t.spec.name: t for t in (tools or [])}
        self.system_prompt = system_prompt
        self._store = _store_from_url(store) if isinstance(store, str) else store

        loop_node = _AgentLoopNode(self)
        self._engine = Engine(
            nodes={"agent_loop": loop_node},
            tools=self._tools,
            store=self._store,
            entry_node="agent_loop",
        )

    async def run(self, prompt: str, run_id: str | None = None) -> RunResult:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        initial_messages = [_message_to_dict(Message(role="user", content=prompt))]
        state = await self._engine.start(run_id=run_id, _messages=initial_messages)
        return RunResult(
            run_id=state.run_id,
            status=state.status,
            output=state.scratch.get("final_output"),
            state=state,
        )

    async def resume(self, run_id: str) -> RunResult:
        state = await self._engine.resume(run_id)
        return RunResult(
            run_id=state.run_id,
            status=state.status,
            output=state.scratch.get("final_output"),
            state=state,
        )

    def approve(self, run_id: str, tool: str) -> None:
        """
        Marks a tool as approved for a paused run. NOTE: this is the same
        stub-shaped gap flagged in Engine.approve_pending_tool — real
        approval persistence needs its own durable record, not just an
        in-memory flag, before this is production-safe. Implemented here
        as a minimal version so Agent's resume() flow is usable today.
        """
        raise NotImplementedError(
            "Agent.approve() needs a durable approval-persistence layer — "
            "see Engine.approve_pending_tool. For now, set "
            "state.scratch['_approved_tools'] = {tool: True} and save a "
            "checkpoint manually before calling resume(), as shown in "
            "examples/kubectl_agent."
        )