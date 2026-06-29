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

import asyncio
import uuid
from dataclasses import asdict, dataclass

from kestrion.core.engine import ApprovalRequired, Engine
from kestrion.core.types import AgentState, Event, EventType, NodeResult, RunStatus, Tool, ToolResult, ToolSpec
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

            # Multiple tool calls in one turn run CONCURRENTLY, not one at
            # a time — but only after every gated call in the batch has
            # been pre-checked for approval via the SAME check_approval
            # method call_tool uses internally (no duplicated gating
            # logic to drift out of sync). This two-phase design (check
            # everything, THEN dispatch everything) guarantees a batch is
            # either fully run or cleanly paused with nothing partially
            # executed — never "2 of 3 tools already ran, then we paused
            # on the 3rd." This is safe because call_tool's gating check
            # happens strictly before any side effect (before
            # TOOL_CALL_STARTED is even emitted) — raising here is
            # equivalent to raising inside call_tool itself, just earlier
            # and for the whole batch at once.
            for call in response.tool_calls:
                if call.name in agent._tools:
                    agent._engine.check_approval(state, call.name, call.arguments)
                # unknown tool names are left for call_tool to raise its
                # own KeyError on, same as before this feature existed

            # All gates clear — dispatch every call in this turn
            # concurrently. asyncio.gather (default, no
            # return_exceptions) means a real exception from one call
            # still propagates immediately, same fail-fast behavior as
            # the old sequential loop had — we're not silently swallowing
            # failures just because calls are now concurrent.
            call_kwargs_list = []
            for call in response.tool_calls:
                kwargs = dict(call.arguments)
                if isinstance(agent._tools.get(call.name), HandoffTool):
                    # HandoffTool needs the CALLER's full conversation,
                    # not whatever arguments the model happened to pass —
                    # inject it here rather than asking the model to
                    # somehow produce the entire message history itself.
                    kwargs["_handoff_messages"] = [_message_to_dict(m) for m in messages]
                call_kwargs_list.append(kwargs)

            try:
                results = await asyncio.gather(
                    *[
                        agent._engine.call_tool(state, call.name, **kwargs)
                        for call, kwargs in zip(response.tool_calls, call_kwargs_list)
                    ]
                )
            except HandoffCompleted as handoff:
                # The conversation has been fully transferred — this
                # agent's run ends HERE, not by falling through to the
                # normal "no more tool calls" completion path, and
                # critically not by feeding a handoff "result" back to
                # the model as an ordinary tool message (which could
                # cause this agent to keep talking about a conversation
                # it no longer owns).
                return NodeResult(
                    next_node=None,
                    state_updates={
                        "_messages": [_message_to_dict(m) for m in messages],
                        "final_output": (
                            f"Handed off to another agent (run_id={handoff.target_run_id})."
                        ),
                        "_handed_off_to": handoff.target_run_id,
                    },
                )

            for call, result in zip(response.tool_calls, results):
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


class SubAgentTool(Tool):
    """
    Wraps an Agent so it satisfies the Tool contract — making "one agent
    calls another agent" require zero new engine machinery. From the
    parent Engine's perspective, calling a sub-agent looks exactly like
    calling any other tool.

    Design decisions worth being explicit about:

    1. The sub-agent's run gets its OWN run_id and checkpoint history in
       the same store, not nested inside the parent's event log. This
       means the sub-agent's run is independently resumable — if the
       parent crashes after the sub-agent already completed, that work
       isn't redone. Sharing the same store (rather than a separate one)
       is what makes resume() work for it using the exact same mechanism
       as any top-level run.

    2. If the sub-agent's run pauses on WAITING_ON_HUMAN, that must
       propagate to the PARENT, not be silently swallowed as if the
       sub-agent just "returned" a paused status as a normal result. This
       is handled by re-raising ApprovalRequired in the parent's context,
       tagged with a synthetic role name carrying the sub-agent's run_id
       (sub_agent:<run_id>) so whoever resolves the parent's pending
       approval knows exactly which sub-agent run needs resuming
       separately. This reuses the existing missing_roles mechanism
       unchanged — no new pause/resume concept was needed.
    """

    def __init__(self, sub_agent: "Agent", name: str, description: str):
        self._sub_agent = sub_agent
        self.spec = ToolSpec(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
            requires_approval=False,
        )

    async def call(self, **kwargs) -> ToolResult:
        prompt = kwargs["prompt"]
        result = await self._sub_agent.run(prompt)

        if result.status == RunStatus.WAITING_ON_HUMAN:
            # Propagate the pause to the parent. The parent's Engine
            # catches this exactly like any other ApprovalRequired —
            # parent pauses too, parent's _pending_approval will show
            # missing_roles=["sub_agent:<run_id>"], which is the signal
            # for "go resume THIS sub-agent run separately, then resume
            # the parent."
            raise ApprovalRequired(
                tool_name=self.spec.name,
                kwargs=kwargs,
                missing_roles=[f"sub_agent:{result.run_id}"],
            )

        if result.status == RunStatus.FAILED:
            return ToolResult(
                tool_name=self.spec.name,
                output=None,
                error=f"Sub-agent run {result.run_id} failed",
            )

        if result.status == RunStatus.EXPIRED:
            return ToolResult(
                tool_name=self.spec.name,
                output=None,
                error=f"Sub-agent run {result.run_id} expired waiting for approval",
            )

        return ToolResult(tool_name=self.spec.name, output=result.output)


class HandoffCompleted(Exception):
    """
    Raised by HandoffTool.call() to signal that the conversation has been
    fully transferred to another agent — NOT an error. This is a
    deliberate control-flow signal, same pattern as ApprovalRequired:
    _AgentLoopNode.run() catches it specifically and ends the CALLING
    agent's run immediately, recording the handoff, rather than letting
    the loop continue and feed a handoff "result" back to the model as
    if it were an ordinary tool result. A normal ToolResult would risk
    the original agent continuing to talk about a conversation it no
    longer owns.
    """

    def __init__(self, target_run_id: str, target_status: RunStatus, target_output: str | None):
        self.target_run_id = target_run_id
        self.target_status = target_status
        self.target_output = target_output


class HandoffTool(Tool):
    """
    Wraps an Agent as a handoff target. Unlike SubAgentTool (delegation —
    the parent stays in control and gets an answer back), calling a
    HandoffTool transfers the ENTIRE conversation to the target agent,
    which takes over completely. The calling agent's run ends as soon as
    the handoff happens; it never resumes control of this conversation.

    Design decisions, made deliberately rather than by default:

    1. The target agent gets its OWN run_id, not the caller's. This was
       chosen over "same run_id continues" specifically to avoid a real
       correctness risk: if the same run_id were reused, the target
       agent's tool calls would be checked against
       scratch["_approved_tools"] entries that may have been recorded
       for the ORIGINAL agent's differently-scoped tools — a same-named
       tool on the target agent could be incorrectly treated as
       pre-approved. Separate run_ids keep approval scoping correct,
       same reasoning that already justified separate run_ids for
       sub-agents.

    2. The full message history transfers via Agent.run_with_history(),
       not just a summary or a fresh prompt — the target agent needs to
       see everything that already happened to handle the conversation
       coherently.

    3. The two runs are linked via scratch, not by sharing an identity —
       the caller's final state records target_run_id under
       "_handed_off_to", so anyone tracking the conversation across the
       handoff boundary can follow the link without the engine needing
       any special "this is actually one logical conversation" concept.
    """

    def __init__(self, target_agent: "Agent", name: str, description: str):
        self._target_agent = target_agent
        self.spec = ToolSpec(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {}},
            requires_approval=False,
        )

    async def call(self, **kwargs) -> ToolResult:
        # kwargs intentionally unused here — the transferred history
        # comes from the CALLER's conversation, injected by
        # _AgentLoopNode via the special _handoff_messages kwarg it
        # passes when dispatching this specific tool. See
        # _AgentLoopNode.run()'s tool-dispatch loop for where that's set.
        messages = kwargs["_handoff_messages"]
        result = await self._target_agent.run_with_history(messages)
        raise HandoffCompleted(
            target_run_id=result.run_id,
            target_status=result.status,
            target_output=result.output,
        )


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
        initial_messages = [_message_to_dict(Message(role="user", content=prompt))]
        return await self.run_with_history(initial_messages, run_id=run_id)

    async def run_with_history(
        self, messages: list[dict], run_id: str | None = None
    ) -> RunResult:
        """
        Like run(), but seeds the new run with an existing message
        history instead of a single fresh prompt. `messages` must be a
        list of dicts in the same shape _message_to_dict produces.

        This exists specifically for handoff (see HandoffTool below): when
        agent A hands off to agent B, B needs to start with the ENTIRE
        transferred conversation, not just a new prompt — run() alone
        can't express that since it only ever wraps a single string into
        a one-message history.
        """
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        state = await self._engine.start(run_id=run_id, _messages=messages)
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

    def as_tool(self, name: str, description: str) -> SubAgentTool:
        """
        Wraps this Agent as a Tool another Agent can call — the
        agent-calling-agent / delegation pattern. The returned tool takes
        a single `prompt` argument and runs this agent against it.

        Usage:
            specialist = Agent(provider=..., tools=[...], store=shared_store_url)
            planner = Agent(
                provider=...,
                tools=[specialist.as_tool("query_database", "Ask the database specialist a question")],
                store=shared_store_url,  # SAME store — required for the
                                          # sub-agent's run to be
                                          # independently resumable
            )

        If the sub-agent's run pauses for approval, the PARENT run also
        pauses (see SubAgentTool for why) — resuming requires resuming
        the sub-agent's run_id first, then resuming the parent. The
        parent's pending-approval record's missing_roles will contain
        "sub_agent:<run_id>" naming exactly which one.
        """
        return SubAgentTool(self, name=name, description=description)

    def as_handoff_target(self, name: str, description: str) -> HandoffTool:
        """
        Wraps this Agent as a HANDOFF target — distinct from as_tool().
        Calling the returned tool transfers the ENTIRE conversation to
        this agent, which takes over completely; the calling agent's run
        ends as soon as the handoff happens and never resumes control.

        Usage:
            billing_agent = Agent(provider=..., tools=[...], store=shared_store_url)
            router = Agent(
                provider=...,
                tools=[billing_agent.as_handoff_target("transfer_to_billing", "Hand off to the billing specialist")],
                store=shared_store_url,
            )
            result = await router.run("I have a question about my invoice")
            # result.status == COMPLETED, result.state.scratch["_handed_off_to"]
            # names the billing agent's NEW, separate run_id to follow.

        Use as_tool() instead if you want the original agent to stay in
        control and just ask the other agent a question (delegation,
        not a full handoff).
        """
        return HandoffTool(self, name=name, description=description)