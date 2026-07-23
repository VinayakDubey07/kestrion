import pytest
from kestrion.agent.agent import Agent
from kestrion.agent.registry import ToolRegistry
from kestrion.agent.decorators import tool
from kestrion.llm.base import LLMResponse, ToolCallRequest

class MockProvider:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages, tools, system):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                stop_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="1", name="find_and_load_tool", arguments={"query": "weather"})],
                tokens_in=10, tokens_out=10, cost_usd=0.01
            )
        elif self.calls == 2:
            return LLMResponse(
                text="",
                stop_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="2", name="get_weather", arguments={"location": "London"})],
                tokens_in=10, tokens_out=10, cost_usd=0.01
            )
        else:
            return LLMResponse(text="The weather in London is sunny.", stop_reason="stop", tool_calls=[], tokens_in=10, tokens_out=10, cost_usd=0.01)

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Sunny in {location}"

@pytest.mark.asyncio
async def test_dynamic_tools():
    registry = ToolRegistry([get_weather])
    provider = MockProvider()
    agent = Agent(provider=provider, tool_registry=registry)
    
    # Run the agent
    result = await agent.run("What's the weather in London?")
    
    # Assertions
    assert "The weather in London is sunny" in result.output
    
    # The tool should have been added to the dynamic list
    assert "get_weather" in result.state.scratch.get("_active_dynamic_tools", [])
