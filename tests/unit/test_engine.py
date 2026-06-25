"""
These three tests are the automated version of what was previously only
verified by hand via examples/kubectl_agent.py. They exist to make sure
that as the Agent/@tool decorator layer gets built on top of this engine,
none of these three behaviors can silently regress.

Test doubles here are deliberately simpler than the kubectl example (no
kubectl/k8s framing) so a failure points at the engine itself, not at
example-specific logic.
"""

from kestrion.core.engine import ApprovalRequired, Engine
from kestrion.core.types import (
    AgentState,
    NodeResult,
    RunStatus,
    Tool,
    ToolResult,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# Minimal test doubles: a 2-node graph, one safe tool, one gated tool
# ---------------------------------------------------------------------------

class EchoTool(Tool):
    """A harmless, non-mutating tool — never requires approval."""

    spec = ToolSpec(
        name="echo",
        description="Returns whatever it's given",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        requires_approval=False,
    )

    async def call(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output=kwargs.get("value"))


class MutatingTool(Tool):
    """Simulates a real-world side effect — always requires approval."""

    spec = ToolSpec(
        name="mutate",
        description="Simulates an external side effect",
        parameters={"type": "object", "properties": {}},
        requires_approval=True,
    )

    async def call(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output={"mutated": True})


class StepOneNode:
    """First node: calls the safe tool, always advances to step_two."""

    name = "step_one"

    def __init__(self, engine_ref: dict):
        self._engine_ref = engine_ref

    async def run(self, state: AgentState) -> NodeResult:
        engine = self._engine_ref["engine"]
        result = await engine.call_tool(state, "echo", value="hello")
        return NodeResult(next_node="step_two", state_updates={"echoed": result.output})


class StepTwoNode:
    """Second node: calls the gated tool, then completes the run."""

    name = "step_two"

    def __init__(self, engine_ref: dict):
        self._engine_ref = engine_ref

    async def run(self, state: AgentState) -> NodeResult:
        engine = self._engine_ref["engine"]
        result = await engine.call_tool(state, "mutate")
        return NodeResult(next_node=None, state_updates={"mutate_result": result.output})


def build_engine(store) -> Engine:
    """
    Builds a fresh Engine + node graph against the given store. Used
    multiple times per test to simulate independent processes sharing
    only the store, never in-memory state.
    """
    engine_ref: dict = {}
    nodes = {
        "step_one": StepOneNode(engine_ref),
        "step_two": StepTwoNode(engine_ref),
    }
    tools = {"echo": EchoTool(), "mutate": MutatingTool()}
    engine = Engine(nodes=nodes, tools=tools, store=store, entry_node="step_one")
    engine_ref["engine"] = engine
    return engine


# ---------------------------------------------------------------------------
# Test 1: normal completion through a graph with no gated tools
# ---------------------------------------------------------------------------

async def test_run_completes_normally_when_no_approval_required(tmp_store):
    """A graph using only safe tools should run start-to-finish in one call."""
    engine_ref: dict = {}
    nodes = {"step_one": StepOneNode(engine_ref)}
    tools = {"echo": EchoTool()}

    # step_one alone, modified to terminate, isolates this test from
    # step_two's gated tool entirely.
    class TerminalStepOne(StepOneNode):
        async def run(self, state: AgentState) -> NodeResult:
            engine = self._engine_ref["engine"]
            result = await engine.call_tool(state, "echo", value="hello")
            return NodeResult(next_node=None, state_updates={"echoed": result.output})

    nodes = {"step_one": TerminalStepOne(engine_ref)}
    engine = Engine(nodes=nodes, tools=tools, store=tmp_store, entry_node="step_one")
    engine_ref["engine"] = engine

    final_state = await engine.start()

    assert final_state.status == RunStatus.COMPLETED
    assert final_state.current_node is None
    assert final_state.scratch["echoed"] == "hello"


# ---------------------------------------------------------------------------
# Test 2: a gated tool call pauses the run, not just the function call
# ---------------------------------------------------------------------------

async def test_run_pauses_at_waiting_on_human_for_gated_tool(tmp_store):
    engine = build_engine(tmp_store)

    state = await engine.start()

    assert state.status == RunStatus.WAITING_ON_HUMAN
    assert state.current_node == "step_two"
    pending = state.scratch.get("_pending_approval")
    assert pending is not None
    assert pending["tool"] == "mutate"
    assert pending["resume_node"] == "step_two"
    # the run must NOT have completed step_two's mutation yet
    assert "mutate_result" not in state.scratch


async def test_engine_call_tool_raises_approval_required_directly(tmp_store):
    """
    Unit-level check on the gating mechanism itself, independent of the
    full run loop: calling a gated tool without prior approval must raise,
    not silently execute.
    """
    engine = build_engine(tmp_store)
    state = AgentState(run_id="run_direct_check")

    try:
        await engine.call_tool(state, "mutate")
        assert False, "expected ApprovalRequired to be raised"
    except ApprovalRequired as exc:
        assert exc.tool_name == "mutate"


# ---------------------------------------------------------------------------
# Test 3: crash recovery — a second, independent Engine resumes correctly
# ---------------------------------------------------------------------------

async def test_resume_from_independent_engine_after_approval(tmp_store):
    """
    This is the core differentiator, made automatic: a run pauses, then a
    SECOND Engine instance — constructed fresh, sharing nothing but the
    store — picks it up and finishes it. This is what proves durability
    isn't dependent on any in-memory state surviving a process boundary.
    """
    # "Process A": starts the run, which pauses on the gated tool.
    engine_a = build_engine(tmp_store)
    paused_state = await engine_a.start()
    assert paused_state.status == RunStatus.WAITING_ON_HUMAN

    # Simulate the approval being granted, then persist that as a new
    # checkpoint — this models what a real approval-persistence layer
    # would do (see Engine.approve_pending_tool's NotImplementedError).
    paused_state.scratch["_approved_tools"] = {"mutate": True}
    from datetime import datetime, timezone
    from kestrion.core.types import Checkpoint, new_id

    await tmp_store.save(
        Checkpoint(
            checkpoint_id=new_id("ckpt"),
            run_id=paused_state.run_id,
            state=paused_state,
            created_at=datetime.now(timezone.utc),
            event_seq=paused_state.last_event_seq,
        )
    )

    # "Process B": a brand new Engine object. It shares nothing with
    # engine_a except the store. This is the actual crash-recovery proof.
    engine_b = build_engine(tmp_store)
    final_state = await engine_b.resume(paused_state.run_id)

    assert final_state.status == RunStatus.COMPLETED
    assert final_state.scratch["mutate_result"] == {"mutated": True}


async def test_resume_raises_clearly_when_no_checkpoint_exists(tmp_store):
    engine = build_engine(tmp_store)

    try:
        await engine.resume("run_that_never_existed")
        assert False, "expected ValueError for missing checkpoint"
    except ValueError as exc:
        assert "run_that_never_existed" in str(exc)


async def test_full_event_log_is_present_after_pause_and_resume(tmp_store):
    """
    Confirms the observability claim: every step of a paused-then-resumed
    run is recorded, not just the final outcome.
    """
    engine_a = build_engine(tmp_store)
    paused_state = await engine_a.start()

    paused_state.scratch["_approved_tools"] = {"mutate": True}
    from datetime import datetime, timezone
    from kestrion.core.types import Checkpoint, new_id

    await tmp_store.save(
        Checkpoint(
            checkpoint_id=new_id("ckpt"),
            run_id=paused_state.run_id,
            state=paused_state,
            created_at=datetime.now(timezone.utc),
            event_seq=paused_state.last_event_seq,
        )
    )

    engine_b = build_engine(tmp_store)
    await engine_b.resume(paused_state.run_id)

    events = await tmp_store.events_since(paused_state.run_id, 0)
    event_types = [e.type.value for e in events]

    assert "run_started" in event_types
    assert "tool_call_completed" in event_types
    assert "human_intervention" in event_types
    assert "run_completed" in event_types
