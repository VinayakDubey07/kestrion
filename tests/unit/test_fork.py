import pytest
from kestrion.core.engine import Engine
from kestrion.core.types import AgentState, EventType, RunStatus, Node, NodeResult
from kestrion.store.memory_store import MemoryCheckpointStore

class DummyNode(Node):
    name = "dummy"
    async def run(self, state: AgentState) -> NodeResult:
        return NodeResult(next_node=None, state_updates={})

@pytest.mark.asyncio
async def test_engine_fork():
    store = MemoryCheckpointStore()
    engine = Engine(nodes={"dummy": DummyNode()}, tools={}, store=store, entry_node="dummy")
    
    # Create an initial run and add 5 events to it
    run_id = "run_original"
    state = AgentState(run_id=run_id, status=RunStatus.RUNNING, current_node="dummy")
    
    await engine._emit(state, EventType.RUN_STARTED, {"entry_node": "dummy"})
    await engine._emit(state, EventType.LLM_CALL_STARTED, {})
    await engine._emit(state, EventType.LLM_CALL_COMPLETED, {"content": "Hello"})
    await engine._emit(state, EventType.TOOL_CALL_STARTED, {"tool": "dummy"})
    await engine._emit(state, EventType.TOOL_CALL_COMPLETED, {"output": "Done"})
    
    # Fork at seq 3 (which corresponds to evt3)
    forked_run_id = await engine.fork(run_id, at_seq=3, new_run_id="run_forked")
    assert forked_run_id == "run_forked"
    
    # Check the store for the forked run
    fork_events = await store.events_since(forked_run_id, 0)
    
    # Since at_seq=3 keeps the first 3 events, and fork() calls _checkpoint which emits CHECKPOINT_SAVED
    assert len(fork_events) == 4
    assert fork_events[0].type == EventType.RUN_STARTED
    assert fork_events[1].type == EventType.LLM_CALL_STARTED
    assert fork_events[2].type == EventType.LLM_CALL_COMPLETED
    assert fork_events[3].type == EventType.CHECKPOINT_SAVED
    
    # Check the state of the forked run
    checkpoint = await store.latest(forked_run_id)
    assert checkpoint is not None
    forked_state = checkpoint.state
    
    assert forked_state.run_id == "run_forked"
    assert forked_state.status == RunStatus.RUNNING
    assert forked_state.current_node == "dummy"
    
    # Because fold was called on the first 3 events, history should only have the LLM call
    assert len(forked_state.history) == 1
    assert forked_state.history[0]["type"] == "llm_response"
