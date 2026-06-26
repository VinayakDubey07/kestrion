"""
Tests for time-boxed approvals: ToolSpec.approval_timeout_seconds, the
new RunStatus.EXPIRED terminal state, and resume()'s on_expired
parameter ("status" vs "raise").
"""

from datetime import datetime, timedelta, timezone

from kestrion.core.engine import Engine, RunExpiredError
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


class TimeBoxedTool(Tool):
    """Requires approval, with a 1-second timeout — short enough to test without sleeping long."""

    spec = ToolSpec(
        name="deploy_now",
        description="Deploys immediately",
        parameters={"type": "object", "properties": {}},
        requires_approval=True,
        approval_timeout_seconds=1.0,
    )

    async def call(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output={"deployed": True})


class NoTimeoutTool(Tool):
    """Requires approval, no timeout — must behave exactly as before this feature existed."""

    spec = ToolSpec(
        name="deploy_whenever",
        description="Deploys, no rush",
        parameters={"type": "object", "properties": {}},
        requires_approval=True,
    )

    async def call(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output={"deployed": True})


class CallerNode:
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


def _checkpoint_for(state: AgentState) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=new_id("ckpt"),
        run_id=state.run_id,
        state=state,
        created_at=datetime.now(timezone.utc),
        event_seq=state.last_event_seq,
    )


def _force_expiry(state: AgentState) -> None:
    """
    Backdates the pending approval's expires_at into the past, so tests
    don't need to actually sleep 1+ seconds to exercise the expired path.
    """
    state.scratch["_pending_approval"]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).isoformat()


# ---------------------------------------------------------------------------
# No-timeout tools: must behave exactly as before this feature
# ---------------------------------------------------------------------------

async def test_tool_without_timeout_has_no_expiry_recorded(tmp_store):
    engine = _build_engine(tmp_store, NoTimeoutTool(), "deploy_whenever")
    state = await engine.start()

    assert state.status == RunStatus.WAITING_ON_HUMAN
    assert state.scratch["_pending_approval"]["expires_at"] is None


async def test_tool_without_timeout_never_expires_no_matter_how_long_paused(tmp_store):
    engine = _build_engine(tmp_store, NoTimeoutTool(), "deploy_whenever")
    state = await engine.start()
    await tmp_store.save(_checkpoint_for(state))

    # No expires_at to force into the past — resuming with no approval
    # granted should just re-raise the same WAITING_ON_HUMAN pause,
    # never EXPIRED, regardless of how "stale" this looks.
    resumed = await engine.resume(state.run_id)
    assert resumed.status == RunStatus.WAITING_ON_HUMAN


# ---------------------------------------------------------------------------
# Timeout recorded correctly
# ---------------------------------------------------------------------------

async def test_tool_with_timeout_records_an_expiry_deadline(tmp_store):
    engine = _build_engine(tmp_store, TimeBoxedTool(), "deploy_now")
    state = await engine.start()

    pending = state.scratch["_pending_approval"]
    assert pending["expires_at"] is not None
    requested = datetime.fromisoformat(pending["requested_at"])
    expires = datetime.fromisoformat(pending["expires_at"])
    assert (expires - requested).total_seconds() == 1.0


async def test_deadline_is_not_reset_on_repeated_pending_resume(tmp_store):
    """
    If a run is resumed before approval is granted (still missing), the
    deadline must stay anchored to the FIRST request, not get pushed
    back every time someone resumes without approving.
    """
    engine = _build_engine(tmp_store, TimeBoxedTool(), "deploy_now")
    state = await engine.start()
    first_expires_at = state.scratch["_pending_approval"]["expires_at"]
    await tmp_store.save(_checkpoint_for(state))

    # Resume without granting approval — still WAITING_ON_HUMAN, not expired yet.
    resumed = await engine.resume(state.run_id)
    assert resumed.status == RunStatus.WAITING_ON_HUMAN
    assert resumed.scratch["_pending_approval"]["expires_at"] == first_expires_at


# ---------------------------------------------------------------------------
# Expiry behavior
# ---------------------------------------------------------------------------

async def test_resume_transitions_to_expired_status_by_default(tmp_store):
    engine = _build_engine(tmp_store, TimeBoxedTool(), "deploy_now")
    state = await engine.start()
    _force_expiry(state)
    await tmp_store.save(_checkpoint_for(state))

    resumed = await engine.resume(state.run_id)

    assert resumed.status == RunStatus.EXPIRED


async def test_resume_with_on_expired_raise_raises_run_expired_error(tmp_store):
    engine = _build_engine(tmp_store, TimeBoxedTool(), "deploy_now")
    state = await engine.start()
    _force_expiry(state)
    await tmp_store.save(_checkpoint_for(state))

    try:
        await engine.resume(state.run_id, on_expired="raise")
        assert False, "expected RunExpiredError"
    except RunExpiredError as exc:
        assert exc.run_id == state.run_id
        assert exc.tool_name == "deploy_now"


async def test_expired_run_records_a_run_expired_event(tmp_store):
    engine = _build_engine(tmp_store, TimeBoxedTool(), "deploy_now")
    state = await engine.start()
    _force_expiry(state)
    await tmp_store.save(_checkpoint_for(state))

    await engine.resume(state.run_id)

    events = await tmp_store.events_since(state.run_id, 0)
    event_types = [e.type.value for e in events]
    assert "run_expired" in event_types


async def test_approval_granted_before_deadline_still_completes_normally(tmp_store):
    """
    The timeout must only matter if the deadline has actually passed —
    approving in time must work exactly as it did before this feature.
    """
    engine = _build_engine(tmp_store, TimeBoxedTool(), "deploy_now")
    state = await engine.start()
    # Approve immediately, well before the 1-second deadline.
    Engine.record_approval(state, "deploy_now", role="__any__")
    await tmp_store.save(_checkpoint_for(state))

    resumed = await engine.resume(state.run_id)

    assert resumed.status == RunStatus.COMPLETED
    assert resumed.scratch["result"] == {"deployed": True}


async def test_expired_status_is_a_terminal_state_resume_again_stays_expired(tmp_store):
    """Once expired, resuming again should not somehow un-expire the run."""
    engine = _build_engine(tmp_store, TimeBoxedTool(), "deploy_now")
    state = await engine.start()
    _force_expiry(state)
    await tmp_store.save(_checkpoint_for(state))
    expired_state = await engine.resume(state.run_id)
    assert expired_state.status == RunStatus.EXPIRED
    await tmp_store.save(_checkpoint_for(expired_state))

    # Resuming an already-EXPIRED run: status isn't WAITING_ON_HUMAN
    # anymore, so resume()'s expiry-check branch doesn't even apply —
    # but _drive's while-loop condition (status == RUNNING) means it
    # also won't proceed. The run simply stays EXPIRED.
    again = await engine.resume(expired_state.run_id)
    assert again.status == RunStatus.EXPIRED