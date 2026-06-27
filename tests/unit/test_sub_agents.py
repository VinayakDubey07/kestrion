"""
Tests for sub-agents: Agent.as_tool() wraps an Agent as a Tool, so one
agent can delegate to another using nothing but the existing tool-call
mechanism. Covers the normal-completion case and, more importantly, the
approval-propagation case — a sub-agent pausing for approval must pause
the PARENT too, not be silently swallowed.
"""

from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.core.types import EventType, RunStatus
from kestrion.llm.base import LLMResponse, ToolCallRequest


def _store_url(tmp_store) -> str:
    return f"sqlite:///{tmp_store.path}"


@tool
def get_inventory_count(sku: str) -> dict:
    """Look up inventory count for a SKU."""
    return {"sku": sku, "count": 42}


@tool(requires_approval=True)
def issue_refund(order_id: str) -> dict:
    """Issue a refund. Requires approval."""
    return {"refunded": True, "order_id": order_id}


class FakeSpecialistProvider:
    """A sub-agent's provider: answers a single inventory question."""

    async def complete(self, messages, tools, system=None):
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        if last_user and not any(m.role == "tool" for m in messages):
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="c1", name="get_inventory_count", arguments={"sku": "ABC123"})],
                stop_reason="tool_use",
            )
        return LLMResponse(text="There are 42 units of ABC123 in stock.", tool_calls=[], stop_reason="end_turn")


class FakeRefundSpecialistProvider:
    """A sub-agent that tries a gated tool, then finishes once it sees the tool result."""

    async def complete(self, messages, tools, system=None):
        if messages and messages[-1].role == "tool":
            return LLMResponse(text="Refund processed.", tool_calls=[], stop_reason="end_turn")
        return LLMResponse(
            text=None,
            tool_calls=[ToolCallRequest(id="c1", name="issue_refund", arguments={"order_id": "ORD-1"})],
            stop_reason="tool_use",
        )


# ---------------------------------------------------------------------------
# Basic delegation: parent calls sub-agent, gets back its final answer
# ---------------------------------------------------------------------------

async def test_parent_agent_delegates_to_sub_agent_and_gets_its_answer(tmp_store):
    store_url = _store_url(tmp_store)
    specialist = Agent(provider=FakeSpecialistProvider(), tools=[get_inventory_count], store=store_url)
    specialist_tool = specialist.as_tool("check_inventory", "Ask the inventory specialist")

    class FakePlannerProvider:
        def __init__(self):
            self.calls = 0

        async def complete(self, messages, tools, system=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    text=None,
                    tool_calls=[ToolCallRequest(id="p1", name="check_inventory", arguments={"prompt": "How many ABC123 in stock?"})],
                    stop_reason="tool_use",
                )
            return LLMResponse(text="The specialist says 42 units are in stock.", tool_calls=[], stop_reason="end_turn")

    planner = Agent(provider=FakePlannerProvider(), tools=[specialist_tool], store=store_url)
    result = await planner.run("How much ABC123 do we have?")

    assert result.status == RunStatus.COMPLETED
    assert "42" in result.output


async def test_sub_agent_tool_spec_has_a_simple_prompt_only_schema():
    specialist = Agent(provider=FakeSpecialistProvider(), tools=[get_inventory_count], store="sqlite:///unused.db")
    sub_tool = specialist.as_tool("check_inventory", "Ask the inventory specialist")

    assert sub_tool.spec.name == "check_inventory"
    assert sub_tool.spec.parameters["required"] == ["prompt"]
    assert sub_tool.spec.requires_approval is False


# ---------------------------------------------------------------------------
# The critical case: sub-agent pausing for approval must pause the PARENT
# ---------------------------------------------------------------------------

async def test_sub_agent_pausing_for_approval_pauses_the_parent_run(tmp_store):
    store_url = _store_url(tmp_store)
    refund_specialist = Agent(provider=FakeRefundSpecialistProvider(), tools=[issue_refund], store=store_url)
    refund_tool = refund_specialist.as_tool("process_refund", "Ask the refund specialist to process a refund")

    class FakePlannerProvider:
        async def complete(self, messages, tools, system=None):
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="p1", name="process_refund", arguments={"prompt": "Refund order ORD-1"})],
                stop_reason="tool_use",
            )

    planner = Agent(provider=FakePlannerProvider(), tools=[refund_tool], store=store_url)
    result = await planner.run("Please refund order ORD-1")

    # The PARENT (planner) run must show waiting_on_human, not completed
    # with some swallowed/lost pause from the sub-agent.
    assert result.status == RunStatus.WAITING_ON_HUMAN
    pending = result.state.scratch["_pending_approval"]
    assert pending["tool"] == "process_refund"
    # missing_roles carries the sub-agent's run_id so it's clear which
    # nested run actually needs approving/resuming.
    assert len(pending["missing_roles"]) == 1
    assert pending["missing_roles"][0].startswith("sub_agent:run_")


async def test_sub_agent_pause_does_not_get_logged_as_a_tool_call_failure(tmp_store):
    """
    Regression test for the bug caught during development: ApprovalRequired
    raised from inside a tool's call() (the sub-agent case) must NOT be
    caught by call_tool's generic except-Exception branch and logged as
    TOOL_CALL_FAILED — that would misreport a clean pause as an error.
    """
    store_url = _store_url(tmp_store)
    refund_specialist = Agent(provider=FakeRefundSpecialistProvider(), tools=[issue_refund], store=store_url)
    refund_tool = refund_specialist.as_tool("process_refund", "Ask the refund specialist")

    class FakePlannerProvider:
        async def complete(self, messages, tools, system=None):
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="p1", name="process_refund", arguments={"prompt": "Refund ORD-1"})],
                stop_reason="tool_use",
            )

    planner = Agent(provider=FakePlannerProvider(), tools=[refund_tool], store=store_url)
    result = await planner.run("Refund order ORD-1")

    events = await tmp_store.events_since(result.run_id, 0)
    event_types = [e.type.value for e in events]
    assert EventType.TOOL_CALL_FAILED.value not in event_types
    assert EventType.HUMAN_INTERVENTION.value in event_types


async def test_sub_agent_run_is_independently_resumable(tmp_store):
    """
    The sub-agent's run has its own run_id in the same store and can be
    resumed on its own, independent of the parent — this is what
    "doesn't redo completed sub-agent work after a parent crash" means
    in practice.
    """
    store_url = _store_url(tmp_store)
    refund_specialist = Agent(provider=FakeRefundSpecialistProvider(), tools=[issue_refund], store=store_url)
    refund_tool = refund_specialist.as_tool("process_refund", "Ask the refund specialist")

    class FakePlannerProvider:
        async def complete(self, messages, tools, system=None):
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="p1", name="process_refund", arguments={"prompt": "Refund ORD-1"})],
                stop_reason="tool_use",
            )

    planner = Agent(provider=FakePlannerProvider(), tools=[refund_tool], store=store_url)
    result = await planner.run("Refund order ORD-1")
    sub_run_id = result.state.scratch["_pending_approval"]["missing_roles"][0].split("sub_agent:")[1]

    # Resume the SUB-AGENT's run directly, independent of the parent,
    # using a freshly constructed Agent instance for it — same
    # independent-process pattern as the engine-level crash recovery test.
    fresh_specialist = Agent(provider=FakeRefundSpecialistProvider(), tools=[issue_refund], store=store_url)

    from datetime import datetime, timezone

    from kestrion.core.engine import Engine
    from kestrion.core.types import Checkpoint, new_id

    # Fetch the sub-agent's current state via its store to approve it directly.
    sub_checkpoint = await tmp_store.latest(sub_run_id)
    Engine.record_approval(sub_checkpoint.state, "issue_refund", role="__any__")
    await tmp_store.save(Checkpoint(
        checkpoint_id=new_id("ckpt"),
        run_id=sub_run_id,
        state=sub_checkpoint.state,
        created_at=datetime.now(timezone.utc),
        event_seq=sub_checkpoint.state.last_event_seq,
    ))

    sub_result = await fresh_specialist.resume(sub_run_id)
    assert sub_result.status == RunStatus.COMPLETED