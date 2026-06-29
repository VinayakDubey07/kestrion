"""
The execution engine. This is the part that makes "production-grade" true
rather than aspirational: every step is checkpointed, every run is
resumable, every tool call that mutates external state can be gated on
human approval.

Worked example used in comments throughout: a kubectl agent that (1) reads
cluster state, (2) proposes a YAML change, (3) requires human approval,
(4) applies it, (5) verifies rollout. This mirrors the bastion-host kubectl
MCP agent you already built — the approval gate is exactly the kind of
thing that's awkward to retrofit into LangGraph but trivial if it's in the
engine's core loop from day one.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable, Literal

from .types import (
    AgentState,
    Checkpoint,
    CheckpointStore,
    Event,
    EventType,
    Node,
    RunStatus,
    Tool,
    ToolResult,
    new_id,
    utcnow,
)

logger = logging.getLogger("agentframework.engine")


class RunExpiredError(Exception):
    """
    Raised by resume(on_expired="raise") when a run's pending approval
    deadline has passed without all required roles approving. Distinct
    from ApprovalRequired (which the engine catches internally and never
    lets escape to the caller) — this one IS meant to surface to whoever
    called resume(), since "this approval window closed" is a fact about
    the world the caller needs to react to, not an internal control-flow
    signal.
    """

    def __init__(self, run_id: str, tool_name: str, expired_at: str):
        self.run_id = run_id
        self.tool_name = tool_name
        self.expired_at = expired_at
        super().__init__(
            f"Run {run_id}'s pending approval for tool {tool_name!r} expired at {expired_at}"
        )


class ApprovalRequired(Exception):
    """
    Raised (and caught by the engine, not the user) when a node wants to
    call a tool whose required-approval roles aren't all satisfied yet.
    The engine persists a checkpoint and parks the run in
    WAITING_ON_HUMAN rather than blocking a thread — this is what makes
    "1000 agents waiting on approval" cheap.
    """

    def __init__(self, tool_name: str, kwargs: dict, missing_roles: list[str]):
        self.tool_name = tool_name
        self.kwargs = kwargs
        # Which required roles have NOT yet approved. For simple
        # requires_approval=True tools this is always ["__any__"]. For a
        # chain like ["engineer", "manager"], this narrows as approvals
        # come in — e.g. after the engineer approves, a re-raised
        # ApprovalRequired on retry would show only ["manager"] missing.
        self.missing_roles = missing_roles


class Engine:
    """
    Drives a graph of Nodes to completion, persisting an event log and
    periodic checkpoints. Stateless across runs by design — you can run
    many Engine instances behind a load balancer, because all durable
    state lives in the CheckpointStore, not in this object.
    """

    def __init__(
        self,
        nodes: dict[str, Node],
        tools: dict[str, Tool],
        store: CheckpointStore,
        entry_node: str,
        approval_callback: Callable[[str, dict], bool] | None = None,
    ):
        self.nodes = nodes
        self.tools = tools
        self.store = store
        self.entry_node = entry_node
        # sync hook the host app can override; default = auto-deny.
        # In production this is what a UI "approve kubectl apply" button
        # calls into, or a Slack approval workflow, etc.
        self.approval_callback = approval_callback or (lambda *_: False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, run_id: str | None = None, **initial_scratch) -> AgentState:
        run_id = run_id or new_id("run")
        state = AgentState(run_id=run_id, status=RunStatus.RUNNING, current_node=self.entry_node)
        state.scratch.update(initial_scratch)

        await self._emit(state, EventType.RUN_STARTED, {"entry_node": self.entry_node})
        return await self._drive(state)

    async def resume(
        self, run_id: str, on_expired: Literal["status", "raise"] = "status"
    ) -> AgentState:
        """
        Crash recovery / human-approval resume in one code path. Load the
        latest checkpoint, replay any events since it (in case the process
        died between checkpoint and completion), then keep driving.

        on_expired controls what happens if this run's pending approval
        deadline (set via ToolSpec.approval_timeout_seconds) has passed
        without all required roles approving:
          - "status" (default): the run transitions to RunStatus.EXPIRED
            and is returned normally — consistent with how every other
            terminal state (COMPLETED, FAILED) already works in this
            codebase: callers check .status, no exception needed for the
            common case.
          - "raise": raises RunExpiredError instead. For callers who want
            a hard failure if they accidentally try to resume something
            stale — e.g. a cron job that should alert loudly rather than
            silently return a status nobody checks.
        """
        checkpoint = await self.store.latest(run_id)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for run {run_id}")

        state = checkpoint.state
        newer_events = await self.store.events_since(run_id, checkpoint.event_seq)
        for evt in newer_events:
            self._fold(state, evt)

        if state.status == RunStatus.WAITING_ON_HUMAN:
            pending = state.scratch.get("_pending_approval") or {}
            expires_at = pending.get("expires_at")
            if expires_at is not None and datetime.fromisoformat(expires_at) < utcnow():
                state.status = RunStatus.EXPIRED
                await self._emit(
                    state,
                    EventType.RUN_EXPIRED,
                    {"tool": pending.get("tool"), "expired_at": expires_at},
                )
                await self._checkpoint(state)
                if on_expired == "raise":
                    raise RunExpiredError(run_id, pending.get("tool", "<unknown>"), expires_at)
                return state
            state.status = RunStatus.RUNNING

        return await self._drive(state)

    def approve_pending_tool(self, run_id: str) -> None:
        """
        Host app calls this after a human clicks 'approve'. Sets a flag the
        next resume() call will honor. (Left as a stub here — real impl
        would persist the approval decision via the store so it survives
        a process restart, same pattern as everything else.)
        """
        raise NotImplementedError("Wire this to your approval persistence layer")

    @staticmethod
    def record_approval(state: AgentState, tool_name: str, role: str = "__any__") -> None:
        """
        Correctly appends a role to state.scratch["_approved_tools"][tool_name]
        without clobbering any roles already recorded there. This exists
        specifically to replace the error-prone pattern of hand-writing
        scratch["_approved_tools"] = {tool: True} — that overwrite pattern
        silently destroys a partially-satisfied approval chain (e.g. if
        "engineer" already approved and you then overwrite the whole dict
        to record "manager" approving, the engineer's approval vanishes).

        Caller is still responsible for persisting a checkpoint (via
        save() against the store) after calling this and before calling
        resume() — this only mutates the in-memory AgentState, consistent
        with every other state-mutating helper in this module.
        """
        approved = state.scratch.setdefault("_approved_tools", {})
        existing = approved.get(tool_name)
        if existing is True:
            # Already fully approved under the old bool shape; nothing to add.
            return
        roles = set(existing) if isinstance(existing, list) else set()
        roles.add(role)
        approved[tool_name] = list(roles)

    # ------------------------------------------------------------------
    # Internal: the actual graph-walking loop
    # ------------------------------------------------------------------

    async def _drive(self, state: AgentState) -> AgentState:
        while state.status == RunStatus.RUNNING and state.current_node is not None:
            node = self.nodes[state.current_node]

            try:
                result = await node.run(state)
            except ApprovalRequired as approval:
                # This is the key production behavior: we don't block a
                # thread waiting for a human. We checkpoint and return
                # control. Some other process/request resumes us later.
                state.status = RunStatus.WAITING_ON_HUMAN

                # The deadline is set ONCE, the first time approval is
                # requested for this tool — not reset on every retry (e.g.
                # a partial chain still missing one role re-raises this
                # same exception on each resume() attempt). Re-deriving
                # "first requested at" from any existing pending-approval
                # record for this exact tool is what makes that correct.
                existing_pending = state.scratch.get("_pending_approval")
                if existing_pending and existing_pending.get("tool") == approval.tool_name:
                    requested_at = existing_pending.get("requested_at", utcnow().isoformat())
                else:
                    requested_at = utcnow().isoformat()

                tool_spec = self.tools[approval.tool_name].spec
                expires_at = None
                if tool_spec.approval_timeout_seconds is not None:
                    requested_dt = datetime.fromisoformat(requested_at)
                    expires_at = (
                        requested_dt + timedelta(seconds=tool_spec.approval_timeout_seconds)
                    ).isoformat()

                state.scratch["_pending_approval"] = {
                    "tool": approval.tool_name,
                    "kwargs": approval.kwargs,
                    "resume_node": state.current_node,
                    "missing_roles": approval.missing_roles,
                    "requested_at": requested_at,
                    "expires_at": expires_at,
                }
                await self._emit(
                    state,
                    EventType.HUMAN_INTERVENTION,
                    {
                        "reason": "approval_required",
                        "tool": approval.tool_name,
                        "missing_roles": approval.missing_roles,
                        "expires_at": expires_at,
                    },
                )
                await self._checkpoint(state)
                return state
            except Exception as exc:
                state.status = RunStatus.FAILED
                await self._emit(state, EventType.RUN_FAILED, {"error": str(exc)})
                await self._checkpoint(state)
                raise

            # Fold this node's events into state, then advance.
            for evt in result.events:
                self._fold(state, evt)
            state.scratch.update(result.state_updates)

            if result.next_node is None:
                state.status = RunStatus.COMPLETED
                state.current_node = None
                await self._emit(state, EventType.RUN_COMPLETED, {})
            else:
                await self._emit(
                    state,
                    EventType.STATE_TRANSITION,
                    {"from": node.name, "to": result.next_node},
                )
                state.current_node = result.next_node

            # Checkpoint every transition. For high-frequency graphs you'd
            # make this configurable (e.g. every N steps) — but the
            # *default* should be safe, not fast.
            await self._checkpoint(state)

        return state

    def check_approval(self, state: AgentState, tool_name: str, kwargs: dict) -> None:
        """
        Raises ApprovalRequired if `tool_name` has unsatisfied required
        roles, otherwise returns normally. This is the SAME check
        call_tool performs internally before executing a tool — exposed
        publicly so callers like Agent's parallel-tool-call dispatch can
        pre-check an entire batch of calls for approval before running
        ANY of them, without duplicating this logic (and risking it
        silently drifting out of sync the next time approval-gating
        changes, which has already happened twice in this codebase).
        """
        tool = self.tools[tool_name]
        required_roles = tool.spec.required_roles()
        if not required_roles:
            return
        recorded = state.scratch.get("_approved_tools", {}).get(tool_name)
        # Backward-compat shape: scratch["_approved_tools"][name] = True
        # (from before chains existed) means "approved, role-agnostic" —
        # treat it as satisfying any/all required roles. Must check this
        # BEFORE trying to treat `recorded` as an iterable of role names,
        # since True is not iterable (caught by the regression test
        # this exact bug produced — test_resume_from_independent_engine
        # _after_approval and others, which all pre-date the chain
        # feature and store the bool shape).
        if recorded is True:
            approved_roles = set(required_roles)
        elif isinstance(recorded, list):
            approved_roles = set(recorded)
        else:
            approved_roles = set()
        missing = [r for r in required_roles if r not in approved_roles]
        if missing:
            raise ApprovalRequired(tool_name, kwargs, missing_roles=missing)

    async def call_tool(self, state: AgentState, tool_name: str, **kwargs) -> ToolResult:
        """
        Nodes call tools through the engine, never directly — this is the
        single choke point where approval-gating and event emission for
        tool calls happens, so individual nodes can't accidentally bypass
        the safety gate for a mutating kubectl/SQL call.
        """
        self.check_approval(state, tool_name, kwargs)
        tool = self.tools[tool_name]

        await self._emit(state, EventType.TOOL_CALL_STARTED, {"tool": tool_name, "args": kwargs})
        try:
            result = await tool.call(**kwargs)
        except ApprovalRequired:
            # A tool's OWN call() can raise this directly — the
            # motivating case is SubAgentTool, where a sub-agent's run
            # pausing for approval needs to propagate to the parent as a
            # pause, not get logged as a tool failure. Re-raise as-is,
            # bypassing the generic except-Exception branch below, which
            # would otherwise emit a misleading TOOL_CALL_FAILED event
            # for what is actually a clean pause, not a failure.
            raise
        except Exception as exc:
            if exc.__class__.__name__ == "HandoffCompleted":
                # Same reasoning as ApprovalRequired above — a successful
                # handoff is not a tool failure. Checked by class name
                # rather than importing kestrion.agent.agent here, since
                # core/ must not depend on agent/ (see architecture.md's
                # dependency-direction rule) — HandoffCompleted is an
                # agent/-layer control-flow signal, not a core concept,
                # but call_tool still needs to avoid mislabeling it.
                raise
            await self._emit(state, EventType.TOOL_CALL_FAILED, {"tool": tool_name, "error": str(exc)})
            raise
        await self._emit(
            state,
            EventType.TOOL_CALL_COMPLETED,
            {"tool": tool_name, "output": str(result.output)[:2000]},
        )
        return result

    async def record_event(self, state: AgentState, event: Event) -> None:
        """
        Public escape hatch for nodes that need to durably record
        something that isn't a tool call — the motivating case is
        Agent's LLM-loop node recording LLM_CALL_COMPLETED events (for
        token/cost tracking) as they happen, rather than batching them
        into the eventual NodeResult. Batching would lose them if the
        node later raises ApprovalRequired and never returns a
        NodeResult at all. Appends to the store AND folds into state
        immediately — same durability guarantee as call_tool's internal
        _emit calls.
        """
        seq = await self.store.append_event(event)
        state.last_event_seq = seq
        self._fold(state, event)

    # ------------------------------------------------------------------
    # Event sourcing plumbing
    # ------------------------------------------------------------------

    async def _emit(self, state: AgentState, type: EventType, payload: dict) -> Event:
        evt = Event.create(run_id=state.run_id, type=type, payload=payload, node=state.current_node)
        seq = await self.store.append_event(evt)
        state.last_event_seq = seq
        return evt

    def _fold(self, state: AgentState, evt: Event) -> None:
        """
        The fold function: Event -> AgentState mutation. This is the ONLY
        place state changes happen as a result of an event. Keeping this
        centralized is what makes replay/debugging tractable — you can
        rebuild any historical state by replaying events through this
        one function.
        """
        if evt.type == EventType.TOOL_CALL_COMPLETED:
            state.history.append({"type": "tool_result", **evt.payload})
        elif evt.type == EventType.LLM_CALL_COMPLETED:
            state.history.append({"type": "llm_response", **evt.payload})
            state.total_tokens += evt.tokens_in + evt.tokens_out
            state.total_cost_usd += evt.cost_usd

    async def _checkpoint(self, state: AgentState) -> Checkpoint:
        checkpoint = Checkpoint(
            checkpoint_id=new_id("ckpt"),
            run_id=state.run_id,
            state=state,
            created_at=utcnow(),
            event_seq=state.last_event_seq,
        )
        await self.store.save(checkpoint)
        await self._emit(state, EventType.CHECKPOINT_SAVED, {"checkpoint_id": checkpoint.checkpoint_id})
        return checkpoint