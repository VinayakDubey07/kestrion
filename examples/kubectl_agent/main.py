"""
Worked example: a 3-node kubectl agent.

  inspect_cluster -> propose_change -> apply_change (gated) -> [done]

Mirrors a bastion-host kubectl MCP agent: read state, propose a mutation,
require a human to approve before anything touches the cluster.

Run this file directly to see:
  1. The run pausing at WAITING_ON_HUMAN before the kubectl apply.
  2. Resuming after approval, as if this were a brand new process
     (simulating a pod restart / crash) — proving the engine doesn't
     need to keep anything in memory between approval and resume.

This is NOT the final API ergonomics (that's the @tool/Agent decorator
layer, planned for Phase 2). This is here so the raw engine primitives
are proven to work end-to-end on your machine before we build on top.
"""

import asyncio

from kestrion.core.engine import Engine
from kestrion.store.sqlite_store import SQLiteCheckpointStore
from kestrion.core.types import (
    AgentState,
    Event,
    EventType,
    NodeResult,
    Tool,
    ToolResult,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# Tools (these would proxy to a real kubectl MCP server over SSH to the
# bastion host in Phase 3 — stubbed here so the example runs standalone)
# ---------------------------------------------------------------------------

class GetClusterStateTool(Tool):
    spec = ToolSpec(
        name="get_cluster_state",
        description="Read current deployment replica counts",
        parameters={"type": "object", "properties": {}},
        requires_approval=False,  # read-only, safe to auto-run
    )

    async def call(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output={"deployment": "checkout-api", "replicas": 2})


class ApplyManifestTool(Tool):
    spec = ToolSpec(
        name="apply_manifest",
        description="kubectl apply a manifest against the cluster",
        parameters={"type": "object", "properties": {"yaml": {"type": "string"}}},
        requires_approval=True,  # mutating -> must go through the gate
    )

    async def call(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output={"applied": True, "yaml": kwargs.get("yaml")})


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class InspectClusterNode:
    name = "inspect_cluster"

    def __init__(self, engine_ref: dict):
        self._engine_ref = engine_ref  # late-bound, see wiring below

    async def run(self, state: AgentState) -> NodeResult:
        engine = self._engine_ref["engine"]
        result = await engine.call_tool(state, "get_cluster_state")
        return NodeResult(
            next_node="propose_change",
            state_updates={"current_replicas": result.output["replicas"]},
        )


class ProposeChangeNode:
    name = "propose_change"

    async def run(self, state: AgentState) -> NodeResult:
        current = state.scratch["current_replicas"]
        proposed_yaml = f"spec:\n  replicas: {current + 1}  # scale up by 1"
        evt = Event.create(
            run_id=state.run_id,
            type=EventType.STATE_TRANSITION,
            payload={"proposed_yaml": proposed_yaml},
        )
        return NodeResult(
            next_node="apply_change",
            state_updates={"proposed_yaml": proposed_yaml},
            events=[evt],
        )


class ApplyChangeNode:
    name = "apply_change"

    def __init__(self, engine_ref: dict):
        self._engine_ref = engine_ref

    async def run(self, state: AgentState) -> NodeResult:
        engine = self._engine_ref["engine"]
        # This call raises ApprovalRequired the first time through, which
        # the Engine catches and turns into a checkpoint + pause.
        result = await engine.call_tool(state, "apply_manifest", yaml=state.scratch["proposed_yaml"])
        return NodeResult(next_node=None, state_updates={"apply_result": result.output})


# ---------------------------------------------------------------------------
# Wiring + demo
# ---------------------------------------------------------------------------

async def main():
    store = SQLiteCheckpointStore(path="demo_runs.db")

    engine_ref: dict = {}  # nodes need a reference to the engine to call tools
    nodes = {
        "inspect_cluster": InspectClusterNode(engine_ref),
        "propose_change": ProposeChangeNode(),
        "apply_change": ApplyChangeNode(engine_ref),
    }
    tools = {
        "get_cluster_state": GetClusterStateTool(),
        "apply_manifest": ApplyManifestTool(),
    }

    engine = Engine(nodes=nodes, tools=tools, store=store, entry_node="inspect_cluster")
    engine_ref["engine"] = engine

    print("--- First run: should pause before apply_manifest ---")
    state = await engine.start(scratch_note="kubectl scale-up agent")
    print(f"status={state.status.value} current_node={state.current_node} run_id={state.run_id}")
    print(f"pending approval: {state.scratch.get('_pending_approval')}")

    print("\n--- Simulating approval + process restart, then resume ---")
    state.scratch["_approved_tools"] = {"apply_manifest": True}

    from kestrion.core.types import Checkpoint, new_id
    from datetime import datetime, timezone

    ckpt = Checkpoint(
        checkpoint_id=new_id("ckpt"),
        run_id=state.run_id,
        state=state,
        created_at=datetime.now(timezone.utc),
        event_seq=state.last_event_seq,
    )
    await store.save(ckpt)

    final_state = await engine.resume(state.run_id)
    print(f"status={final_state.status.value}")
    print(f"apply_result={final_state.scratch.get('apply_result')}")
    print(f"total events for this run: {len(await store.events_since(state.run_id, 0))}")


if __name__ == "__main__":
    asyncio.run(main())
