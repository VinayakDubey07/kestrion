
import pytest
from kestrion.core.nodes import SupervisorNode
from kestrion.core.types import AgentState
from kestrion.llm.base import LLMResponse, ToolCallRequest

class MockProvider:
    async def complete(self, messages, tools, system):
        return LLMResponse(
            text="",
            stop_reason="tool_calls",
            tool_calls=[ToolCallRequest(id="1", name="route_to", arguments='{"destination": "billing_agent"}')],
            tokens_in=10,
            tokens_out=10,
            cost_usd=0.01
        )

@pytest.mark.asyncio
async def test_supervisor_routing():
    provider = MockProvider()
    node = SupervisorNode(
        provider=provider,
        destinations={"billing_agent": "Billing", "support_agent": "Support"}
    )
    
    state = AgentState(run_id="run1", scratch={"_messages": [{"role": "user", "content": "I need a refund"}]})
    result = await node.run(state)
    
    assert result.next_node == "billing_agent"
    assert len(result.events) == 1
    assert result.events[0].payload["destination"] == "billing_agent"
