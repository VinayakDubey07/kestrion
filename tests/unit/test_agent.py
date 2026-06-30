"""
Tests for the Phase 2 decorator/Agent layer. Uses a fake LLMProvider
throughout — no real API key or network call needed, and no real-money
cost to run this suite. The pause/resume test here is the Agent-level
equivalent of test_engine.py's crash-recovery test: same guarantee,
now exercised through the API real users will actually call.
"""

from datetime import datetime, timezone

from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.core.types import Checkpoint, new_id
from kestrion.llm.base import LLMResponse, ToolCallRequest


# ---------------------------------------------------------------------------
# Test tools — exercised through the real @tool decorator, not stubs
# ---------------------------------------------------------------------------

@tool
def get_cluster_state() -> dict:
    """Read current deployment replica counts."""
    return {"deployment": "checkout-api", "replicas": 2}


@tool(requires_approval=True)
def apply_manifest(yaml: str) -> dict:
    """kubectl apply a manifest against the cluster."""
    return {"applied": True, "yaml": yaml}


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool(requires_approval=["engineer", "manager"])
def deploy_to_prod() -> dict:
    """Deploys to production. Needs both an engineer and a manager."""
    return {"deployed": True}


@tool(requires_approval=True, approval_timeout_seconds=3600.0)
def restart_service() -> dict:
    """Restarts a service. Must be approved within an hour."""
    return {"restarted": True}


# ---------------------------------------------------------------------------
# Decorator / schema introspection tests
# ---------------------------------------------------------------------------

def test_tool_decorator_bare_usage_produces_correct_spec():
    spec = get_cluster_state.spec
    assert spec.name == "get_cluster_state"
    assert spec.requires_approval is False
    assert spec.parameters == {"type": "object", "properties": {}}
    assert "replica" in spec.description.lower()


def test_tool_decorator_with_args_sets_requires_approval():
    spec = apply_manifest.spec
    assert spec.requires_approval is True
    assert spec.parameters["properties"]["yaml"] == {"type": "string"}
    assert spec.parameters["required"] == ["yaml"]


def test_tool_decorator_accepts_a_role_list_for_approval_chains():
    """
    Regression test: ToolSpec.requires_approval was widened to accept
    str | list[str] when approval chains were built, but the @tool
    decorator's own signature was never updated to match — meaning
    @tool(requires_approval=["a", "b"]) raised TypeError until this was
    fixed. Found while building an integration demo that used @tool
    directly instead of constructing ToolSpec by hand, which is what
    every chain/timeout unit test had done up to that point.
    """
    spec = deploy_to_prod.spec
    assert spec.requires_approval == ["engineer", "manager"]
    assert spec.required_roles() == ["engineer", "manager"]


def test_tool_decorator_accepts_approval_timeout_seconds():
    """Same regression class as above, for approval_timeout_seconds."""
    spec = restart_service.spec
    assert spec.requires_approval is True
    assert spec.approval_timeout_seconds == 3600.0


def test_tool_decorator_infers_integer_types():
    spec = add.spec
    assert spec.parameters["properties"]["a"] == {"type": "integer"}
    assert spec.parameters["properties"]["b"] == {"type": "integer"}
    assert set(spec.parameters["required"]) == {"a", "b"}


async def test_decorated_function_is_callable_as_a_tool():
    result = await add.call(a=2, b=3)
    assert result.output == 5
    assert result.error is None


async def test_decorated_function_captures_errors_without_raising():
    @tool
    def always_fails() -> str:
        """A tool that always raises."""
        raise RuntimeError("boom")

    result = await always_fails.call()
    assert result.error == "boom"
    assert result.output is None


# ---------------------------------------------------------------------------
# Fake providers — scripted, deterministic, no network
# ---------------------------------------------------------------------------

class FakeProviderCallsToolThenFinishes:
    """First turn: call get_cluster_state. Second turn: produce a final answer."""

    def __init__(self):
        self.call_count = 0

    async def complete(self, messages, tools, system=None):
        self.call_count += 1
        if self.call_count == 1:
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="call_1", name="get_cluster_state", arguments={})],
                tokens_in=50, tokens_out=20, cost_usd=0.001,
                stop_reason="tool_use",
            )
        return LLMResponse(
            text="checkout-api has 2 replicas.",
            tool_calls=[],
            tokens_in=80, tokens_out=15, cost_usd=0.002,
            stop_reason="end_turn",
        )


class FakeProviderCallsGatedTool:
    async def complete(self, messages, tools, system=None):
        return LLMResponse(
            text=None,
            tool_calls=[ToolCallRequest(id="call_1", name="apply_manifest", arguments={"yaml": "replicas: 3"})],
            tokens_in=60, tokens_out=25, cost_usd=0.001,
            stop_reason="tool_use",
        )


class FakeProviderFinishesAfterToolResult:
    """Used on resume: sees the tool result already in history and stops."""

    async def complete(self, messages, tools, system=None):
        if messages and messages[-1].role == "tool":
            return LLMResponse(text="Scaled successfully.", tool_calls=[], stop_reason="end_turn")
        return LLMResponse(
            text=None,
            tool_calls=[ToolCallRequest(id="call_1", name="apply_manifest", arguments={"yaml": "replicas: 3"})],
            stop_reason="tool_use",
        )


# ---------------------------------------------------------------------------
# Agent integration tests
# ---------------------------------------------------------------------------

def _store_url(tmp_store) -> str:
    return f"sqlite:///{tmp_store.path}"


async def test_agent_run_completes_with_a_non_gated_tool_call(tmp_store):
    agent = Agent(
        provider=FakeProviderCallsToolThenFinishes(),
        tools=[get_cluster_state],
        store=_store_url(tmp_store),
    )

    result = await agent.run("How many replicas does checkout-api have?")

    assert result.status.value == "completed"
    assert "2 replicas" in result.output


async def test_agent_run_pauses_at_waiting_on_human_for_gated_tool(tmp_store):
    agent = Agent(
        provider=FakeProviderCallsGatedTool(),
        tools=[apply_manifest],
        store=_store_url(tmp_store),
    )

    result = await agent.run("Scale up checkout-api")

    assert result.status.value == "waiting_on_human"
    pending = result.state.scratch.get("_pending_approval")
    assert pending is not None
    assert pending["tool"] == "apply_manifest"


async def test_agent_resume_from_independent_instance_completes_gated_run(tmp_store):
    """
    The Agent-level equivalent of test_engine.py's crash-recovery test:
    a SECOND, independently constructed Agent (sharing only the store)
    resumes a paused run and completes it.
    """
    store_url = _store_url(tmp_store)

    agent_a = Agent(provider=FakeProviderCallsGatedTool(), tools=[apply_manifest], store=store_url)
    paused = await agent_a.run("Scale up checkout-api")
    assert paused.status.value == "waiting_on_human"

    # Simulate approval being granted and persisted (mirrors the manual
    # checkpoint dance in examples/kubectl_agent until Agent.approve()
    # has a real persistence layer behind it).
    paused.state.scratch["_approved_tools"] = {"apply_manifest": True}
    await tmp_store.save(
        Checkpoint(
            checkpoint_id=new_id("ckpt"),
            run_id=paused.run_id,
            state=paused.state,
            created_at=datetime.now(timezone.utc),
            event_seq=paused.state.last_event_seq,
        )
    )

    agent_b = Agent(provider=FakeProviderFinishesAfterToolResult(), tools=[apply_manifest], store=store_url)
    final = await agent_b.resume(paused.run_id)

    assert final.status.value == "completed"
    assert final.output == "Scaled successfully."


async def test_agent_tracks_token_and_cost_totals(tmp_store):
    agent = Agent(
        provider=FakeProviderCallsToolThenFinishes(),
        tools=[get_cluster_state],
        store=_store_url(tmp_store),
    )

    result = await agent.run("How many replicas does checkout-api have?")

    # Both LLM calls (tool-call turn + final-answer turn) should have
    # contributed to the running token/cost totals via the engine's fold.
    # FakeProviderCallsToolThenFinishes reports 50+20 then 80+15 tokens.
    assert result.state.total_tokens == (50 + 20) + (80 + 15)
    assert result.state.total_cost_usd > 0


async def test_agent_records_llm_tokens_even_when_run_pauses_on_approval(tmp_store):
    """
    Regression test for the gap caught during Phase 2 development: if a
    node raises ApprovalRequired partway through a turn, it never returns
    a NodeResult — anything batched into that NodeResult (like LLM call
    events) would be silently lost. record_event's immediate
    append-and-fold must survive that exception path.
    """
    agent = Agent(
        provider=FakeProviderCallsGatedTool(),
        tools=[apply_manifest],
        store=_store_url(tmp_store),
    )

    result = await agent.run("Scale up checkout-api")

    assert result.status.value == "waiting_on_human"
    # The LLM call that decided to invoke the gated tool happened BEFORE
    # the ApprovalRequired exception — its tokens must still be recorded.
    assert result.state.total_tokens > 0