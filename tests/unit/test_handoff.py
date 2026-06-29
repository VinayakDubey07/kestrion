"""
Tests for multi-agent handoff: Agent.as_handoff_target() transfers an
entire conversation to another agent, which takes over completely — as
opposed to SubAgentTool (delegation), where the parent stays in control.

Key things tested:
- the full message history actually transfers, not just a fresh prompt
- the calling agent's run ends cleanly, doesn't keep talking afterward
- the target agent gets a NEW run_id, not the caller's (the deliberate
  design choice made to avoid approval-scoping bleed between agents)
- HandoffCompleted is not misreported as a tool failure (same class of
  bug already fixed once for ApprovalRequired during sub-agent work)
"""

from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.core.types import EventType, RunStatus
from kestrion.llm.base import LLMResponse, ToolCallRequest


def _store_url(tmp_store) -> str:
    return f"sqlite:///{tmp_store.path}"


@tool(requires_approval=True)
def issue_credit(amount: int) -> dict:
    """Issue an account credit. Requires approval."""
    return {"credited": amount}


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice."""
    return {"invoice_id": invoice_id, "amount": 100}


class FakeRouterProvider:
    """A router that immediately hands off to billing."""

    async def complete(self, messages, tools, system=None):
        return LLMResponse(
            text=None,
            tool_calls=[ToolCallRequest(id="r1", name="transfer_to_billing", arguments={})],
            stop_reason="tool_use",
        )


class FakeBillingProvider:
    """The billing agent: looks up the invoice the moment it gets control."""

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, tools, system=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="b1", name="lookup_invoice", arguments={"invoice_id": "INV-1"})],
                stop_reason="tool_use",
            )
        return LLMResponse(text="Your invoice INV-1 is for $100.", tool_calls=[], stop_reason="end_turn")


# ---------------------------------------------------------------------------
# Basic handoff: router transfers, billing agent takes over and completes
# ---------------------------------------------------------------------------

async def test_router_hands_off_and_billing_agent_completes(tmp_store):
    store_url = _store_url(tmp_store)
    billing_agent = Agent(provider=FakeBillingProvider(), tools=[lookup_invoice], store=store_url)
    handoff_tool = billing_agent.as_handoff_target("transfer_to_billing", "Hand off to billing")

    router = Agent(provider=FakeRouterProvider(), tools=[handoff_tool], store=store_url)
    result = await router.run("I have a question about invoice INV-1")

    # The ROUTER's run completes (it ends, doesn't keep going) —
    # handoff means the router's job is done, not that it's waiting.
    assert result.status == RunStatus.COMPLETED
    assert "Handed off" in result.output

    target_run_id = result.state.scratch["_handed_off_to"]
    assert target_run_id != result.run_id  # genuinely a different run

    # The BILLING agent's run, fetched independently via its own run_id,
    # actually completed the conversation.
    target_checkpoint = await tmp_store.latest(target_run_id)
    assert target_checkpoint.state.status == RunStatus.COMPLETED
    assert "$100" in target_checkpoint.state.scratch["final_output"]


async def test_full_conversation_history_transfers_to_the_target_agent(tmp_store):
    """
    The target agent must see what the user originally said, not just a
    blank slate or a generic prompt — that's the whole point of handoff
    vs. a fresh agent.run() call.
    """
    store_url = _store_url(tmp_store)
    billing_agent = Agent(provider=FakeBillingProvider(), tools=[lookup_invoice], store=store_url)
    handoff_tool = billing_agent.as_handoff_target("transfer_to_billing", "Hand off to billing")

    router = Agent(provider=FakeRouterProvider(), tools=[handoff_tool], store=store_url)
    result = await router.run("I have a question about invoice INV-1, please help")

    target_run_id = result.state.scratch["_handed_off_to"]
    target_checkpoint = await tmp_store.latest(target_run_id)
    target_messages = target_checkpoint.state.scratch["_messages"]

    # The original user message must be present in the TARGET's history.
    user_messages = [m["content"] for m in target_messages if m["role"] == "user"]
    assert any("invoice INV-1" in (c or "") for c in user_messages)


# ---------------------------------------------------------------------------
# The critical design check: separate run_ids avoid approval-scope bleed
# ---------------------------------------------------------------------------

async def test_target_agent_gets_a_new_run_id_not_the_routers(tmp_store):
    store_url = _store_url(tmp_store)
    billing_agent = Agent(provider=FakeBillingProvider(), tools=[lookup_invoice], store=store_url)
    handoff_tool = billing_agent.as_handoff_target("transfer_to_billing", "Hand off to billing")

    router = Agent(provider=FakeRouterProvider(), tools=[handoff_tool], store=store_url)
    result = await router.run("help")

    target_run_id = result.state.scratch["_handed_off_to"]
    assert target_run_id.startswith("run_")
    assert target_run_id != result.run_id


async def test_handoff_does_not_leak_router_approvals_to_a_same_named_target_tool(tmp_store):
    """
    The actual correctness risk separate run_ids were chosen to avoid:
    if some OTHER run had already recorded an approval for a tool name,
    a DIFFERENT run with a same-named gated tool must not see that
    approval — they're different run_ids, hence different AgentState
    objects entirely, never shared. This directly constructs that
    scenario rather than asserting something incidental.
    """
    store_url = _store_url(tmp_store)

    # An unrelated, separate run records an approval for "issue_credit"
    # under ITS OWN run_id — simulating some other part of the system
    # having approved a same-named tool in a completely different
    # conversation.
    class FakeProviderJustCallsGatedTool:
        async def complete(self, messages, tools, system=None):
            if messages and messages[-1].role == "tool":
                return LLMResponse(text="Credited.", tool_calls=[], stop_reason="end_turn")
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="x1", name="issue_credit", arguments={"amount": 10})],
                stop_reason="tool_use",
            )

    unrelated_agent = Agent(provider=FakeProviderJustCallsGatedTool(), tools=[issue_credit], store=store_url)
    unrelated_result = await unrelated_agent.run("unrelated conversation")
    assert unrelated_result.status == RunStatus.WAITING_ON_HUMAN
    from kestrion.core.engine import Engine
    Engine.record_approval(unrelated_result.state, "issue_credit", role="__any__")
    from datetime import datetime, timezone
    from kestrion.core.types import Checkpoint, new_id
    await tmp_store.save(Checkpoint(
        checkpoint_id=new_id("ckpt"),
        run_id=unrelated_result.run_id,
        state=unrelated_result.state,
        created_at=datetime.now(timezone.utc),
        event_seq=unrelated_result.state.last_event_seq,
    ))
    unrelated_final = await unrelated_agent.resume(unrelated_result.run_id)
    assert unrelated_final.status == RunStatus.COMPLETED  # the unrelated run's own approval worked for ITSELF

    # NOW: router hands off to a billing agent that also tries the same
    # gated tool, "issue_credit" — on a BRAND NEW run_id, sharing only
    # the store with the unrelated run above. If approvals leaked across
    # runs, this would incorrectly complete instead of pausing.
    class FakeRouterHandsOff:
        async def complete(self, messages, tools, system=None):
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="r1", name="transfer_to_billing", arguments={})],
                stop_reason="tool_use",
            )

    class FakeBillingTriesGatedCredit:
        async def complete(self, messages, tools, system=None):
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="b1", name="issue_credit", arguments={"amount": 50})],
                stop_reason="tool_use",
            )

    billing_agent = Agent(provider=FakeBillingTriesGatedCredit(), tools=[issue_credit], store=store_url)
    handoff_tool = billing_agent.as_handoff_target("transfer_to_billing", "Hand off to billing")
    router = Agent(provider=FakeRouterHandsOff(), tools=[handoff_tool], store=store_url)

    result = await router.run("I need a credit")
    target_run_id = result.state.scratch["_handed_off_to"]
    target_checkpoint = await tmp_store.latest(target_run_id)

    # The target's run must show WAITING_ON_HUMAN — the unrelated run's
    # approval for the same-named tool must NOT have leaked across.
    assert target_checkpoint.state.status == RunStatus.WAITING_ON_HUMAN


# ---------------------------------------------------------------------------
# HandoffCompleted must not be misreported as a tool failure
# ---------------------------------------------------------------------------

async def test_handoff_does_not_get_logged_as_a_tool_call_failure(tmp_store):
    store_url = _store_url(tmp_store)
    billing_agent = Agent(provider=FakeBillingProvider(), tools=[lookup_invoice], store=store_url)
    handoff_tool = billing_agent.as_handoff_target("transfer_to_billing", "Hand off to billing")

    router = Agent(provider=FakeRouterProvider(), tools=[handoff_tool], store=store_url)
    result = await router.run("help")

    events = await tmp_store.events_since(result.run_id, 0)
    event_types = [e.type.value for e in events]
    assert EventType.TOOL_CALL_FAILED.value not in event_types
    assert EventType.RUN_COMPLETED.value in event_types