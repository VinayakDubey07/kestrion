import pytest

from kestrion.agent.tools import ask_human
from kestrion.core.engine import Engine
from kestrion.core.types import AgentState, RunStatus, NodeResult
from kestrion.core.errors import InvalidToolApprovalError, InvalidToolInputError

class MockNode:
    name = "mock_node"
    def __init__(self, engine_ref):
        self.engine_ref = engine_ref

    async def run(self, state: AgentState):
        if state.current_node == "start":
            return NodeResult(next_node="ask", state_updates={})
        elif state.current_node == "ask":
            # the agent "calls" ask_human
            ans = await self.engine_ref["engine"].call_tool(state, "ask_human", question="What is your favorite color?")
            state.scratch["color"] = ans.output
            return NodeResult(next_node=None, state_updates={})


@pytest.mark.asyncio
async def test_human_input_pauses_and_resumes(tmp_store):
    engine_ref = {}
    nodes = {"start": MockNode(engine_ref), "ask": MockNode(engine_ref)}
    tools = {"ask_human": ask_human}
    
    engine = Engine(nodes=nodes, tools=tools, store=tmp_store, entry_node="start")
    engine_ref["engine"] = engine
    
    # 1. Start run, it should pause at ask_human
    state = await engine.start("run_1")
    assert state.status == RunStatus.WAITING_ON_HUMAN
    assert state.scratch["_pending_input"]["tool"] == "ask_human"
    assert state.scratch["_pending_input"]["question"] == "What is your favorite color?"
    
    # 2. Approve tool should FAIL because it's waiting for input, not approval
    with pytest.raises(InvalidToolApprovalError):
        await engine.approve_pending_tool("run_1")
        
    # 2b. Empty input should FAIL
    with pytest.raises(InvalidToolInputError, match="cannot be empty"):
        await engine.provide_input("run_1", "")
        
    # 2c. Wrong tool name should FAIL
    with pytest.raises(InvalidToolInputError, match="does not match"):
        await engine.provide_input("run_1", "Blue", tool="wrong_tool")
        
    # 3. Provide input
    state = await engine.provide_input("run_1", "Blue", tool="ask_human")
    import hashlib
    h = hashlib.md5("What is your favorite color?".encode()).hexdigest()
    assert state.scratch["_human_inputs"][h] == "Blue"
    
    # 3b. Idempotent guard: providing again should not crash, but logs a warning (which we won't assert on, just ensure it doesn't crash)
    state = await engine.provide_input("run_1", "Red", tool="ask_human")
    assert state.scratch["_human_inputs"][h] == "Red"
    
    # 4. Resume run, it should finish
    state = await engine.resume("run_1")
    assert state.status == RunStatus.COMPLETED
    assert state.scratch["color"] == "Red"
    
    # Check events
    events = await tmp_store.events_since("run_1", 0)
    event_types = [e.type.value for e in events]
    assert "human_intervention" in event_types
