
"""
Tests for multi-step approval chains — ToolSpec.requires_approval widened
from bool to bool | str | list[str], and Engine.record_approval as the
correct way to record a role's approval without clobbering others
already recorded.
"""

import logging

from kestrion.core.engine import ApprovalRequired, Engine
from kestrion.core.types import (
    AgentState,
    Checkpoint,
    NodeResult,
    RunStatus,
    Tool,
    ToolResult,
    ToolSpec,
    new_id,
)

logger = logging.getLogger(__name__)


class ChainGatedTool(Tool):
    """A tool requiring approval from BOTH 'engineer' and 'manager'."""

    spec = ToolSpec(
        name="deploy_to_prod",
        description="Deploys to production",
        parameters={"type": "object", "properties": {}},
        requires_approval=["engineer", "manager"],
    )

    async def call(self, **kwargs) -> ToolResult:
        logger.info("Executing deploy_to_prod")
        return ToolResult(tool_name=self.spec.name, output={"deployed": True})


class SingleRoleGatedTool(Tool):
    spec = ToolSpec(
        name="restart_service",
        description="Restarts a service",
        parameters={"type": "object", "properties": {}},
        requires_approval="oncall_engineer",
    )

    async def call(self, **kwargs) -> ToolResult:
        logger.info("Executing restart_service")
        return ToolResult(tool_name=self.spec.name, output={"restarted": True})


class CallerNode:
    """Generic single-tool-calling node."""

    name = "caller"

    def __init__(self, engine_ref: dict, tool_name: str):
        self._engine_ref = engine_ref
        self._tool_name = tool_name

    async def run(self, state: AgentState) -> NodeResult:
        logger.info(
            "CallerNode.run(run_id=%s, tool=%s)",
            state.run_id,
            self._tool_name,
        )
        engine = self._engine_ref["engine"]
        result = await engine.call_tool(state, self._tool_name)
        logger.info("Tool result: %s", result.output)
        return NodeResult(next_node=None, state_updates={"result": result.output})


def _build_engine(store, tool: Tool, tool_name: str) -> Engine:
    logger.info("Building engine with tool '%s'", tool_name)
    engine_ref = {}
    nodes = {"caller": CallerNode(engine_ref, tool_name)}
    tools = {tool_name: tool}
    engine = Engine(nodes=nodes, tools=tools, store=store, entry_node="caller")
    engine_ref["engine"] = engine
    return engine


def test_required_roles_normalizes_false_to_empty_list():
    logger.info("Testing False -> [] normalization")
    spec = ToolSpec(name="t", description="d", parameters={}, requires_approval=False)
    assert spec.required_roles() == []


def test_required_roles_normalizes_true_to_any_sentinel():
    logger.info("Testing True -> ['__any__'] normalization")
    spec = ToolSpec(name="t", description="d", parameters={}, requires_approval=True)
    assert spec.required_roles() == ["__any__"]


def test_required_roles_normalizes_string_to_single_role_list():
    logger.info("Testing string normalization")
    spec = ToolSpec(name="t", description="d", parameters={}, requires_approval="manager")
    assert spec.required_roles() == ["manager"]


def test_required_roles_passes_through_explicit_list():
    logger.info("Testing explicit list normalization")
    spec = ToolSpec(name="t", description="d", parameters={}, requires_approval=["a", "b"])
    assert spec.required_roles() == ["a", "b"]


async def test_chain_gated_tool_blocks_until_all_roles_approve(tmp_store):
    logger.info("Starting chain approval test")
    engine = _build_engine(tmp_store, ChainGatedTool(), "deploy_to_prod")
    state = await engine.start()

    logger.info("Status=%s", state.status)
    assert state.status == RunStatus.WAITING_ON_HUMAN

    pending = state.scratch["_pending_approval"]
    logger.info("Missing roles=%s", pending["missing_roles"])
    assert set(pending["missing_roles"]) == {"engineer", "manager"}


async def test_partial_approval_still_blocks_with_narrowed_missing_roles(tmp_store):
    logger.info("Testing partial approval")
    engine = _build_engine(tmp_store, ChainGatedTool(), "deploy_to_prod")
    state = await engine.start()

    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    logger.info("Engineer approval recorded")

    await tmp_store.save(_checkpoint_for(state))
    resumed = await engine.resume(state.run_id)

    logger.info("Resumed status=%s", resumed.status)
    logger.info("Missing=%s", resumed.scratch["_pending_approval"]["missing_roles"])

    assert resumed.status == RunStatus.WAITING_ON_HUMAN
    assert resumed.scratch["_pending_approval"]["missing_roles"] == ["manager"]


async def test_full_chain_approval_completes_the_run(tmp_store):
    logger.info("Testing full approval chain")
    engine = _build_engine(tmp_store, ChainGatedTool(), "deploy_to_prod")
    state = await engine.start()

    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="manager")
    logger.info("All approvals recorded")

    await tmp_store.save(_checkpoint_for(state))
    final = await engine.resume(state.run_id)

    logger.info("Final status=%s", final.status)
    logger.info("Result=%s", final.scratch.get("result"))

    assert final.status == RunStatus.COMPLETED
    assert final.scratch["result"] == {"deployed": True}


async def test_wrong_role_approval_does_not_satisfy_a_different_required_role(tmp_store):
    logger.info("Testing incorrect role approval")
    engine = _build_engine(tmp_store, ChainGatedTool(), "deploy_to_prod")
    state = await engine.start()

    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="security_lead")

    await tmp_store.save(_checkpoint_for(state))
    resumed = await engine.resume(state.run_id)

    logger.info("Missing=%s", resumed.scratch["_pending_approval"]["missing_roles"])

    assert resumed.status == RunStatus.WAITING_ON_HUMAN
    assert resumed.scratch["_pending_approval"]["missing_roles"] == ["manager"]


async def test_single_string_role_gate_behaves_like_a_one_item_chain(tmp_store):
    logger.info("Testing single-role approval")
    engine = _build_engine(tmp_store, SingleRoleGatedTool(), "restart_service")
    state = await engine.start()

    assert state.status == RunStatus.WAITING_ON_HUMAN

    Engine.record_approval(state, "restart_service", role="oncall_engineer")
    logger.info("Approval recorded")

    await tmp_store.save(_checkpoint_for(state))
    final = await engine.resume(state.run_id)

    logger.info("Final status=%s", final.status)
    assert final.status == RunStatus.COMPLETED


async def test_old_bool_true_approval_shape_still_works_via_record_approval(tmp_store):
    logger.info("Testing legacy bool approval compatibility")
    engine = _build_engine(tmp_store, SingleRoleGatedTool(), "restart_service")
    state = await engine.start()

    state.scratch["_approved_tools"] = {"restart_service": True}

    await tmp_store.save(_checkpoint_for(state))
    final = await engine.resume(state.run_id)

    logger.info("Final status=%s", final.status)
    assert final.status == RunStatus.COMPLETED


def test_record_approval_does_not_clobber_previously_recorded_roles():
    logger.info("Testing approval accumulation")
    state = AgentState(run_id="run_1")

    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="manager")

    recorded = set(state.scratch["_approved_tools"]["deploy_to_prod"])
    logger.info("Recorded=%s", recorded)

    assert recorded == {"engineer", "manager"}


def test_record_approval_is_idempotent_for_the_same_role():
    logger.info("Testing idempotent approval")
    state = AgentState(run_id="run_1")

    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="engineer")

    logger.info("Stored=%s", state.scratch["_approved_tools"]["deploy_to_prod"])

    assert state.scratch["_approved_tools"]["deploy_to_prod"] == ["engineer"]


def test_record_approval_on_already_true_approved_tool_is_a_noop():
    logger.info("Testing bool approval no-op")
    state = AgentState(run_id="run_1")
    state.scratch["_approved_tools"] = {"deploy_to_prod": True}

    Engine.record_approval(state, "deploy_to_prod", role="engineer")

    assert state.scratch["_approved_tools"]["deploy_to_prod"] is True


def test_record_approval_result_is_json_serializable(tmp_store):
    logger.info("Testing JSON serialization")
    state = AgentState(run_id="run_1")

    Engine.record_approval(state, "deploy_to_prod", role="engineer")
    Engine.record_approval(state, "deploy_to_prod", role="manager")

    restored = AgentState.from_dict(state.to_dict())

    logger.info("Restored=%s", restored.scratch["_approved_tools"]["deploy_to_prod"])

    assert set(restored.scratch["_approved_tools"]["deploy_to_prod"]) == {
        "engineer",
        "manager",
    }


def _checkpoint_for(state: AgentState):
    from datetime import datetime, timezone

    logger.debug(
        "Creating checkpoint run_id=%s event_seq=%s",
        state.run_id,
        state.last_event_seq,
    )

    return Checkpoint(
        checkpoint_id=new_id("ckpt"),
        run_id=state.run_id,
        state=state,
        created_at=datetime.now(timezone.utc),
        event_seq=state.last_event_seq,
    )
