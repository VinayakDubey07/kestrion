"""
Tests for multi-step approval chains — ToolSpec.requires_approval widened
from bool to bool | str | list[str], and Engine.record_approval as the
correct way to record a role's approval without clobbering others
already recorded.
"""

from kestrion.core.engine import Engine
from kestrion.core.types import AgentState, NodeResult, RunStatus, Tool, ToolResult, ToolSpec


class ChainGatedTool(Tool):
    """A tool requiring approval from BOTH 'engineer' and 'manager'."""

    spec = ToolSpec(
        name="deploy_to_prod",
        description="Deploys to production",
        parameters={"type": "object", "properties": {}},
        requires_approval=["engineer", "manager"],
    )

    async def call(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output={"deployed": True})


class SingleRoleGatedTool(Tool):
    spec = ToolSpec(
        name="restart_service",
        description="Restarts a service",
        parameters={"type": "object", "properties": {}},
        requires_approval="oncall_engineer",
    )

    async def call(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output={"restarted": True})


class CallerNode:
    """Generic single-tool-calling node, parameterized by tool name."""

    name = "caller"

    def __init__(self, engine_ref: dict, tool_name: str):
        self._engine_ref = engine_ref
        self._tool_name = tool_name

    async def run(self, state: AgentState) -> NodeResult:
        engine = self._engine_ref["engine"]
        result = await engine.call_tool(state, self._tool_name)
        return NodeResult(next_node=None, state_updates={"result": result.output})


def _build_engine(store, tool: Tool, tool_name: str) -> Engine:
    engine_ref: dict = {}
    nodes = {"caller": CallerNode(engine_ref, tool_name)}
    tools = {tool_name: tool}
    engine = Engine(nodes=nodes, tools=tools, store=store, entry_node="caller")
    engine_ref["engine"] = engine
    return engine


# ---------------------------------------------------------------------------
# ToolSpec.required_roles() normalization
# ---------------------------------------------------------------------------

def test_required_roles_normalizes_false_to_empty_list():
    spec = ToolSpec(name="t", description="d", parameters={}, requires_approval=False)
    assert spec.required_roles() == []


def test_required_roles_normalizes_true_to_any_sentinel():
    spec = ToolSpec(name="t", description="d", parameters={}, requires_approval=True)
    assert spec.required_roles() == ["__any__"]


def test_required_roles_normalizes_string_to_single_role_list():
    spec = ToolSpec(name="t", description="d", parameters={}, requires_approval="manager")
    assert spec.required_roles() == ["manager"]


def test_required_roles_passes_through_explicit_list():
    spec = ToolSpec(name="t", description="d", parameters={}, requires_approval=["a", "b"])
    assert spec.required_roles() == ["a", "b"]


# ---------------------------------------------------------------------------
# Engine gating behavior with chains
# ---------------------------------------------------------------------------

async def test_chain_gated_tool_blocks_until_all_roles_approve(tmp_store):
    engine = _build_engine(tmp_store, ChainGatedTool(), "deploy_to_prod")
    state = await engine.start()

    assert state.status == RunStatus.WAITING_ON_HUMAN
    pending = state.scratch["_pending_approval"]
    assert set(pending["missing_roles"]) == {"engineer", "manager"}


async def test_partial_approval_still_blocks_with_narrowed_missing_roles(tmp_store):
    engine = _build_engine(tmp_store, ChainGatedTool(), "deploy_to_prod")
    state = await engine.start()

    # Engineer approves, but manager hasn't yet.
    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    await tmp_store.save(_checkpoint_for(state))

    resumed = await engine.resume(state.run_id)

    assert resumed.status == RunStatus.WAITING_ON_HUMAN
    assert resumed.scratch["_pending_approval"]["missing_roles"] == ["manager"]


async def test_full_chain_approval_completes_the_run(tmp_store):
    engine = _build_engine(tmp_store, ChainGatedTool(), "deploy_to_prod")
    state = await engine.start()

    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="manager")
    await tmp_store.save(_checkpoint_for(state))

    final = await engine.resume(state.run_id)

    assert final.status == RunStatus.COMPLETED
    assert final.scratch["result"] == {"deployed": True}


async def test_wrong_role_approval_does_not_satisfy_a_different_required_role(tmp_store):
    """A 'security_lead' approving a tool that needs 'manager' must not count."""
    engine = _build_engine(tmp_store, ChainGatedTool(), "deploy_to_prod")
    state = await engine.start()

    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="security_lead")  # not a required role
    await tmp_store.save(_checkpoint_for(state))

    resumed = await engine.resume(state.run_id)

    assert resumed.status == RunStatus.WAITING_ON_HUMAN
    assert resumed.scratch["_pending_approval"]["missing_roles"] == ["manager"]


async def test_single_string_role_gate_behaves_like_a_one_item_chain(tmp_store):
    engine = _build_engine(tmp_store, SingleRoleGatedTool(), "restart_service")
    state = await engine.start()
    assert state.status == RunStatus.WAITING_ON_HUMAN
    assert state.scratch["_pending_approval"]["missing_roles"] == ["oncall_engineer"]

    Engine.record_approval(state, "restart_service", role="oncall_engineer")
    await tmp_store.save(_checkpoint_for(state))
    final = await engine.resume(state.run_id)
    assert final.status == RunStatus.COMPLETED


async def test_old_bool_true_approval_shape_still_works_via_record_approval(tmp_store):
    """
    Backward compatibility: code written before chains existed sets
    scratch["_approved_tools"][name] = True directly. record_approval
    must not be required for this old shape to keep working — it's
    handled directly in Engine.call_tool's backward-compat branch.
    """
    engine = _build_engine(tmp_store, SingleRoleGatedTool(), "restart_service")
    state = await engine.start()
    assert state.status == RunStatus.WAITING_ON_HUMAN

    # Old-style manual approval, exactly as examples/kubectl_agent does.
    state.scratch["_approved_tools"] = {"restart_service": True}
    await tmp_store.save(_checkpoint_for(state))

    final = await engine.resume(state.run_id)
    assert final.status == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# Engine.record_approval correctness
# ---------------------------------------------------------------------------

def test_record_approval_does_not_clobber_previously_recorded_roles():
    state = AgentState(run_id="run_1")
    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="manager")

    recorded = set(state.scratch["_approved_tools"]["deploy_to_prod"])
    assert recorded == {"engineer", "manager"}


def test_record_approval_is_idempotent_for_the_same_role():
    state = AgentState(run_id="run_1")
    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="engineer")

    assert state.scratch["_approved_tools"]["deploy_to_prod"] == ["engineer"]


def test_record_approval_on_already_true_approved_tool_is_a_noop():
    state = AgentState(run_id="run_1")
    state.scratch["_approved_tools"] = {"deploy_to_prod": True}
    Engine.record_approval(state, "deploy_to_prod", role="engineer")

    # Already fully approved under the bool shape; must not be downgraded
    # to a partial role list.
    assert state.scratch["_approved_tools"]["deploy_to_prod"] is True


def test_record_approval_result_is_json_serializable(tmp_store):
    """
    Regression-style check: the whole point of chains is that the
    approval record persists through a checkpoint. A list of strings is
    JSON-safe by construction, but worth confirming the actual shape
    record_approval produces round-trips through AgentState.to_dict().
    """
    state = AgentState(run_id="run_1")
    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="manager")

    restored = AgentState.from_dict(state.to_dict())
    assert set(restored.scratch["_approved_tools"]["deploy_to_prod"]) == {"engineer", "manager"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _checkpoint_for(state: AgentState):
    from datetime import datetime, timezone

    from kestrion.core.types import Checkpoint, new_id

    return Checkpoint(
        checkpoint_id=new_id("ckpt"),
        run_id=state.run_id,
        state=state,
        created_at=datetime.now(timezone.utc),
        event_seq=state.last_event_seq,
    )