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

import asyncio
import logging
from typing import Callable

from .types import (
    AgentState,
    Checkpoint,
    CheckpointStore,
    Event,
    EventType,
    Node,
    NodeResult,
    RunStatus,
    Tool,
    ToolResult,
    new_id,
    utcnow,
)

logger = logging.getLogger("agentframework.engine")


class ApprovalRequired(Exception):
    """
    Raised (and caught by the engine, not the user) when a node wants to
    call a tool marked requires_approval=True. The engine persists a
    checkpoint and parks the run in WAITING_ON_HUMAN rather than blocking
    a thread — this is what makes "1000 agents waiting on approval" cheap.
    """

    def __init__(self, tool_name: str, kwargs: dict):
        self.tool_name = tool_name
        self.kwargs = kwargs


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

    async def resume(self, run_id: str) -> AgentState:
        """
        Crash recovery / human-approval resume in one code path. Load the
        latest checkpoint, replay any events since it (in case the process
        died between checkpoint and completion), then keep driving.
        """
        checkpoint = await self.store.latest(run_id)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for run {run_id}")

        state = checkpoint.state
        newer_events = await self.store.events_since(run_id, checkpoint.event_seq)
        for evt in newer_events:
            self._fold(state, evt)

        if state.status == RunStatus.WAITING_ON_HUMAN:
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
                state.scratch["_pending_approval"] = {
                    "tool": approval.tool_name,
                    "kwargs": approval.kwargs,
                    "resume_node": state.current_node,
                }
                await self._emit(
                    state,
                    EventType.HUMAN_INTERVENTION,
                    {"reason": "approval_required", "tool": approval.tool_name},
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

    async def call_tool(self, state: AgentState, tool_name: str, **kwargs) -> ToolResult:
        """
        Nodes call tools through the engine, never directly — this is the
        single choke point where approval-gating and event emission for
        tool calls happens, so individual nodes can't accidentally bypass
        the safety gate for a mutating kubectl/SQL call.
        """
        tool = self.tools[tool_name]
        if tool.spec.requires_approval and not state.scratch.get("_approved_tools", {}).get(tool_name):
            raise ApprovalRequired(tool_name, kwargs)

        await self._emit(state, EventType.TOOL_CALL_STARTED, {"tool": tool_name, "args": kwargs})
        try:
            result = await tool.call(**kwargs)
        except Exception as exc:
            await self._emit(state, EventType.TOOL_CALL_FAILED, {"tool": tool_name, "error": str(exc)})
            raise
        await self._emit(
            state,
            EventType.TOOL_CALL_COMPLETED,
            {"tool": tool_name, "output": str(result.output)[:2000]},
        )
        return result

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
