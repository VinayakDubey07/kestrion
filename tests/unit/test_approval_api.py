"""
Tests for Agent.approve() and Engine.approve_pending_tool().

Uses the standard tmp_store fixture (real SQLite in a temp file) and
scripted fake LLM providers — same pattern as test_agent.py.
Every test is async, driven by pytest-asyncio (mode=auto in pyproject.toml).

NOTE: no `from __future__ import annotations` — the @tool decorator
introspects annotations at definition time using inspect, and that
import turns annotations into strings which breaks the type-to-schema
conversion.
"""

import pytest
from dataclasses import dataclass, field

from kestrion.agent.agent import Agent, RunResult
from kestrion.agent.decorators import tool
from kestrion.core.types import EventType, RunStatus
from kestrion.core.errors import (
    CheckpointNotFoundError,
    InvalidRunStatusError,
    InvalidToolApprovalError,
)
from kestrion.llm.base import LLMProvider, LLMResponse, ToolCallRequest


# ---------------------------------------------------------------------------
# Scripted fake LLM
# ---------------------------------------------------------------------------

@dataclass
class _Turn:
    text: str = None
    tool_calls: list = field(default_factory=list)


class FakeLLM(LLMProvider):
    """Plays back pre-scripted turns in order."""

    def __init__(self, turns):
        self._turns = iter(turns)

    async def complete(self, messages, tools=None, system=None):
        turn = next(self._turns)
        return LLMResponse(
            text=turn.text,
            tool_calls=turn.tool_calls,
            stop_reason="end_turn" if not turn.tool_calls else "tool_use",
            tokens_in=10, tokens_out=10, cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# Helper: build an Agent against tmp_store
# ---------------------------------------------------------------------------

def _agent(llm, tools_list, tmp_store):
    return Agent(
        provider=llm,
        tools=tools_list,
        store=f"sqlite:///{tmp_store.path}",
    )


# ---------------------------------------------------------------------------
# Test tools  (unique names to avoid collision with test_agent.py's tools)
# ---------------------------------------------------------------------------

@tool(requires_approval=True)
def appr_gated(value: str) -> dict:
    """Approval API test: gated tool."""
    return {"executed": True, "value": value}


@tool(requires_approval=["engineer", "manager"])
def appr_chain(payload: str) -> dict:
    """Approval API test: multi-role chain tool."""
    return {"done": True, "payload": payload}


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

async def test_approve_and_resume_completes_run(tmp_store):
    """Run pauses, approve() approves and resumes, run completes."""
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "hello"})]),
        _Turn(text="All done."),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)

    result = await agent.run("do the thing")
    assert result.status == RunStatus.WAITING_ON_HUMAN

    final = await agent.approve(result.run_id)
    assert isinstance(final, RunResult)
    assert final.status == RunStatus.COMPLETED
    assert final.output == "All done."


async def test_approve_explicit_tool_name(tmp_store):
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "x"})]),
        _Turn(text="Done."),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)
    result = await agent.run("go")
    assert result.status == RunStatus.WAITING_ON_HUMAN
    final = await agent.approve(result.run_id, tool="appr_gated")
    assert final.status == RunStatus.COMPLETED


async def test_approve_tool_none_defaults_to_pending(tmp_store):
    """tool=None (default) picks up the pending tool automatically."""
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "y"})]),
        _Turn(text="OK."),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)
    result = await agent.run("go")
    assert result.status == RunStatus.WAITING_ON_HUMAN
    final = await agent.approve(result.run_id)   # no tool= arg
    assert final.status == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# and_resume=False tests
# ---------------------------------------------------------------------------

async def test_and_resume_false_returns_none(tmp_store):
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "z"})]),
        _Turn(text="Done."),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)
    result = await agent.run("go")
    assert result.status == RunStatus.WAITING_ON_HUMAN
    ret = await agent.approve(result.run_id, and_resume=False)
    assert ret is None


async def test_and_resume_false_approval_is_durable(tmp_store):
    """
    After approve(and_resume=False) the checkpoint must already contain
    the approval — crash between approve() and a later resume() loses nothing.
    """
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "z"})]),
        _Turn(text="Done."),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)
    result = await agent.run("go")
    await agent.approve(result.run_id, and_resume=False)

    ckpt = await tmp_store.latest(result.run_id)
    assert ckpt is not None
    approved = ckpt.state.scratch.get("_approved_tools", {})
    assert "appr_gated" in approved, (
        "Approval must be persisted in checkpoint before resume is called"
    )


async def test_and_resume_false_then_explicit_resume(tmp_store):
    """Calling resume() separately after approve(and_resume=False) completes the run."""
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "w"})]),
        _Turn(text="Finished."),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)
    result = await agent.run("go")
    await agent.approve(result.run_id, and_resume=False)
    final = await agent.resume(result.run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.output == "Finished."


# ---------------------------------------------------------------------------
# Multi-role chain tests
# ---------------------------------------------------------------------------

async def test_chain_requires_both_roles(tmp_store):
    """
    Engineer approves → run resumes → LLM re-requests the tool → hits the
    approval wall again (manager still missing) → run pauses again.
    Manager approves → run resumes → LLM gets tool result → completes.

    The FakeLLM needs 3 turns because the LLM is called on EACH resume:
      Turn 1: initial request → tool call (blocked, run pauses)
      Turn 2: after engineer approves → LLM re-requests tool (still blocked)
      Turn 3: after manager approves → tool runs, LLM produces final answer
    """
    llm = FakeLLM([
        # Turn 1: initial LLM call — requests the chain_tool
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_chain", arguments={"payload": "deploy"})]),
        # Turn 2: after engineer approves, LLM re-requests the tool (manager still needed)
        _Turn(tool_calls=[ToolCallRequest(id="c2", name="appr_chain", arguments={"payload": "deploy"})]),
        # Turn 3: after manager approves, tool runs and LLM produces final text
        _Turn(text="Deployed."),
    ])
    agent = _agent(llm, [appr_chain], tmp_store)
    result = await agent.run("deploy")
    assert result.status == RunStatus.WAITING_ON_HUMAN

    # Engineer approves → resumes → LLM re-requests tool → manager still missing → re-pauses
    after_engineer = await agent.approve(result.run_id, role="engineer")
    assert after_engineer.status == RunStatus.WAITING_ON_HUMAN, (
        "Run must stay paused until manager also approves"
    )

    # Manager approves → resumes → tool runs → LLM gives final answer
    final = await agent.approve(result.run_id, role="manager")
    assert final.status == RunStatus.COMPLETED
    assert final.output == "Deployed."


async def test_chain_both_roles_and_resume_false(tmp_store):
    """Both roles via and_resume=False; explicit resume finishes the run.

    Same 3-turn scripting as test_chain_requires_both_roles because the
    LLM is re-invoked on each resume.
    """
    llm = FakeLLM([
        # Turn 1: initial call
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_chain", arguments={"payload": "ship"})]),
        # Turn 2: after engineer; manager still needed → re-requests
        _Turn(tool_calls=[ToolCallRequest(id="c2", name="appr_chain", arguments={"payload": "ship"})]),
        # Turn 3: after manager; tool runs
        _Turn(text="Shipped."),
    ])
    agent = _agent(llm, [appr_chain], tmp_store)
    result = await agent.run("ship it")
    assert result.status == RunStatus.WAITING_ON_HUMAN

    await agent.approve(result.run_id, role="engineer", and_resume=False)
    await agent.approve(result.run_id, role="manager", and_resume=False)

    final = await agent.resume(result.run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.output == "Shipped."


# ---------------------------------------------------------------------------
# Event log test
# ---------------------------------------------------------------------------

async def test_approval_granted_event_emitted(tmp_store):
    """A HUMAN_INTERVENTION/approval_granted event is written to the event log."""
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "v"})]),
        _Turn(text="Done."),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)
    result = await agent.run("go")
    await agent.approve(result.run_id, role="ops")

    all_events = await tmp_store.events_since(result.run_id, 0)
    granted = [
        e for e in all_events
        if e.type == EventType.HUMAN_INTERVENTION
        and e.payload.get("reason") == "approval_granted"
    ]
    assert len(granted) == 1, "Exactly one approval_granted event must be emitted"
    assert granted[0].payload["tool"] == "appr_gated"
    assert granted[0].payload["role"] == "ops"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

async def test_unknown_run_id_raises(tmp_store):
    agent = _agent(FakeLLM([]), [appr_gated], tmp_store)
    with pytest.raises(CheckpointNotFoundError):
        await agent.approve("run_does_not_exist")


async def test_wrong_tool_name_raises(tmp_store):
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "q"})]),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)
    result = await agent.run("go")
    assert result.status == RunStatus.WAITING_ON_HUMAN
    with pytest.raises(InvalidToolApprovalError):
        await agent.approve(result.run_id, tool="some_other_tool")


async def test_approving_completed_run_raises(tmp_store):
    llm = FakeLLM([_Turn(text="Already done.")])
    agent = _agent(llm, [], tmp_store)
    result = await agent.run("hello")
    assert result.status == RunStatus.COMPLETED
    with pytest.raises(InvalidRunStatusError):
        await agent.approve(result.run_id)


# ---------------------------------------------------------------------------
# Engine.approve_pending_tool() — raw engine API
# ---------------------------------------------------------------------------

async def test_engine_approve_pending_tool_then_resume(tmp_store):
    """Engine.approve_pending_tool() records approval; engine.resume() completes."""
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "e"})]),
        _Turn(text="Engine done."),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)
    result = await agent.run("run")
    assert result.status == RunStatus.WAITING_ON_HUMAN

    # approve_pending_tool does NOT resume — returned state still WAITING_ON_HUMAN
    state = await agent._engine.approve_pending_tool(result.run_id, tool="appr_gated")
    assert state.status == RunStatus.WAITING_ON_HUMAN

    final = await agent._engine.resume(result.run_id)
    assert final.status == RunStatus.COMPLETED


async def test_engine_approve_wrong_status_raises(tmp_store):
    llm = FakeLLM([_Turn(text="done")])
    agent = _agent(llm, [], tmp_store)
    result = await agent.run("hi")
    assert result.status == RunStatus.COMPLETED
    with pytest.raises(InvalidRunStatusError):
        await agent._engine.approve_pending_tool(result.run_id)


async def test_engine_approve_wrong_tool_raises(tmp_store):
    llm = FakeLLM([
        _Turn(tool_calls=[ToolCallRequest(id="c1", name="appr_gated", arguments={"value": "g"})]),
    ])
    agent = _agent(llm, [appr_gated], tmp_store)
    result = await agent.run("go")
    assert result.status == RunStatus.WAITING_ON_HUMAN
    with pytest.raises(InvalidToolApprovalError):
        await agent._engine.approve_pending_tool(result.run_id, tool="nonexistent")
