import os
import pytest

from kestrion.agent.decorators import tool
from kestrion.core.engine import Engine
from kestrion.core.secrets import EnvVarSecretProvider, SecretProvider
from kestrion.core.types import AgentState, NodeResult

class MockSecretProvider(SecretProvider):
    async def get_secret(self, key: str) -> str | None:
        return "mock_secret" if key == "TEST_KEY" else None

class MockNode:
    name = "mock_node"
    def __init__(self, engine_ref):
        self.engine_ref = engine_ref

    async def run(self, state: AgentState):
        if state.current_node == "start":
            res = await self.engine_ref["engine"].call_tool(state, "get_sensitive_data")
            state.scratch["data"] = res.output
            
            res2 = await self.engine_ref["engine"].call_tool(state, "ignorant_tool")
            state.scratch["ignorant_data"] = res2.output
            
            return NodeResult(next_node=None, state_updates={})

@tool
async def get_sensitive_data(_secrets: SecretProvider) -> str:
    val = await _secrets.get_secret("TEST_KEY")
    return f"Got secret: {val}"

@tool
def ignorant_tool() -> str:
    # Does not request _secrets in signature
    return "I don't need secrets"

@pytest.mark.asyncio
async def test_env_var_secret_provider(monkeypatch):
    monkeypatch.setenv("MY_FAKE_KEY", "super_secret_value")
    provider = EnvVarSecretProvider()
    
    val = await provider.get_secret("MY_FAKE_KEY")
    assert val == "super_secret_value"
    
    missing = await provider.get_secret("NONEXISTENT")
    assert missing is None

@pytest.mark.asyncio
async def test_engine_injects_secrets(tmp_store):
    engine_ref = {}
    nodes = {"start": MockNode(engine_ref)}
    tools = {
        "get_sensitive_data": get_sensitive_data,
        "ignorant_tool": ignorant_tool
    }
    
    mock_secrets = MockSecretProvider()
    engine = Engine(
        nodes=nodes, 
        tools=tools, 
        store=tmp_store, 
        entry_node="start",
        secrets=mock_secrets
    )
    engine_ref["engine"] = engine
    
    state = await engine.start("run_1")
    
    assert state.scratch["data"] == "Got secret: mock_secret"
    assert state.scratch["ignorant_data"] == "I don't need secrets"
