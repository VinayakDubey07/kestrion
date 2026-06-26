"""
Core type definitions for the agent framework.

Design principle: everything that happens during agent execution is an
immutable Event. State is never mutated in place — it's *derived* by folding
events. This gives you replay, time-travel debugging, and crash recovery
almost for free, which is exactly the observability/production story we want.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Events — the atomic, immutable unit of "things that happened"
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    RUN_STARTED = "run_started"
    MESSAGE_RECEIVED = "message_received"   # input to the agent
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    STATE_TRANSITION = "state_transition"    # graph node A -> node B
    CHECKPOINT_SAVED = "checkpoint_saved"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_EXPIRED = "run_expired"   # approval deadline passed before all required roles approved
    HUMAN_INTERVENTION = "human_intervention"  # e.g. approve a kubectl apply


@dataclass(frozen=True)
class Event:
    """
    Immutable fact. Never mutated, never deleted. The event log for a run
    IS the source of truth — current state is just a fold over these.
    """
    event_id: str
    run_id: str
    type: EventType
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    # which node/step emitted this, for tracing
    node: str | None = None
    # token/cost accounting lives on the event, not bolted on after
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    @staticmethod
    def create(run_id: str, type: EventType, **kwargs) -> "Event":
        return Event(
            event_id=new_id("evt"),
            run_id=run_id,
            type=type,
            timestamp=utcnow(),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# State — derived, never the source of truth, always rebuildable from events
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_ON_HUMAN = "waiting_on_human"   # e.g. paused for kubectl approval
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"   # approval deadline passed before all required roles approved


@dataclass
class AgentState:
    """
    The *current* derived state of a run. This is what your agent's logic
    reads/writes during a step — but every write happens by emitting an
    Event, never by mutating AgentState directly from outside the engine.
    """
    run_id: str
    status: RunStatus = RunStatus.PENDING
    current_node: str | None = None
    # arbitrary structured memory the agent's graph nodes read/write,
    # e.g. {"kubectl_context": "...", "pending_apply": {...}}
    scratch: dict[str, Any] = field(default_factory=dict)
    # conversation / tool history, summarized form (not full event log)
    history: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    last_event_seq: int = 0  # for incremental replay

    def to_dict(self) -> dict[str, Any]:
        """
        Explicit, stable serialization for checkpoint storage. Deliberately
        NOT pickle: pickle ties the on-disk format to this exact class
        definition and Python version, which is a liability the moment
        anyone outside this process depends on reading old checkpoints
        (e.g. after an engine upgrade). JSON-compatible dicts are boring
        and durable, which is exactly what a checkpoint format should be.
        """
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "current_node": self.current_node,
            "scratch": self.scratch,
            "history": self.history,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "last_event_seq": self.last_event_seq,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AgentState":
        return AgentState(
            run_id=data["run_id"],
            status=RunStatus(data["status"]),
            current_node=data.get("current_node"),
            scratch=data.get("scratch", {}),
            history=data.get("history", []),
            total_tokens=data.get("total_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            last_event_seq=data.get("last_event_seq", 0),
        )


# ---------------------------------------------------------------------------
# Checkpointing — durable execution is a first-class concept, not an add-on
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    run_id: str
    state: AgentState
    created_at: datetime
    event_seq: int  # event log position this checkpoint corresponds to


@runtime_checkable
class CheckpointStore(Protocol):
    """
    Pluggable storage. SQLite implementation for local/single-agent use,
    Postgres/Redis for the scaled, multi-tenant production case.
    This boundary is the thing that lets the framework scale from your
    laptop to 10,000 concurrent agent sessions without an API change.
    """

    async def save(self, checkpoint: Checkpoint) -> None: ...

    async def latest(self, run_id: str) -> Checkpoint | None: ...

    async def append_event(self, event: Event) -> int:
        """Returns the new event sequence number."""
        ...

    async def events_since(self, run_id: str, seq: int) -> list[Event]: ...


# ---------------------------------------------------------------------------
# Tools — MCP-compatible from day one, not bolted on later
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema, same shape MCP expects
    # Approval requirement. Three shapes, in increasing strictness:
    #   False           — no approval needed (default)
    #   True             — any single approval needed (role-agnostic)
    #   "role_name"      — approval needed from this specific role
    #   ["role_a", "role_b"] — approval needed from ALL listed roles (a chain)
    # A bare True/False keeps existing code working unchanged; the str/list
    # forms are the multi-step approval chain extension.
    requires_approval: bool | str | list[str] = False
    # If set, and the run is still WAITING_ON_HUMAN this many seconds
    # after the approval was first requested, resume() treats the run
    # as EXPIRED rather than re-raising ApprovalRequired forever. None
    # (default) means no deadline — the run can wait indefinitely,
    # exactly like every approval-gated tool before this feature existed.
    approval_timeout_seconds: float | None = None

    def required_roles(self) -> list[str]:
        """
        Normalizes requires_approval into a list of role names that must
        all appear in an approval record before this tool can run.
        True normalizes to a single anonymous role ("__any__") so the
        existing scratch["_approved_tools"][name] = True shape still
        satisfies it — this is what keeps bool-style approval backward
        compatible with the new chain mechanism underneath.
        """
        ra = self.requires_approval
        if ra is False:
            return []
        if ra is True:
            return ["__any__"]
        if isinstance(ra, str):
            return [ra]
        return list(ra)


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    output: Any
    error: str | None = None
    duration_ms: float = 0.0


class Tool(ABC):
    """
    Every Tool is structurally an MCP tool. A Tool can be backed by:
      - local Python code
      - a remote MCP server (kubectl-mcp, oracle-sql-mcp, etc.)
    From the engine's perspective these look identical.
    """

    spec: ToolSpec

    @abstractmethod
    async def call(self, **kwargs) -> ToolResult: ...


# ---------------------------------------------------------------------------
# Nodes — the unit of the execution graph
# ---------------------------------------------------------------------------

@runtime_checkable
class Node(Protocol):
    """
    A single step in the agent's graph. Pure-ish: given state, produce
    (next_node_name, state_updates, events_to_emit). The engine is
    responsible for actually persisting events and advancing state —
    nodes never touch storage directly. This keeps nodes testable
    without a database.
    """

    name: str

    async def run(self, state: AgentState) -> "NodeResult": ...


@dataclass
class NodeResult:
    next_node: str | None          # None means run is complete
    state_updates: dict[str, Any]  # merged into AgentState.scratch
    events: list[Event] = field(default_factory=list)