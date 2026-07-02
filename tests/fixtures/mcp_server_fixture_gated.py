"""
Second MCP server fixture for tests/unit/test_mcp_server.py — this one
serves an agent with a GATED tool, to test that a paused run is
surfaced correctly through the MCP protocol. Same pattern as
mcp_server_fixture.py; separate file because the tool/provider
combination needs to differ for this test case.

Run directly with: python3 tests/fixtures/mcp_server_fixture_gated.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.llm.base import LLMResponse, ToolCallRequest
from kestrion.mcp.server import serve_agent


@tool(requires_approval=True)
def send_alert(message: str) -> dict:
    """Send an alert. Requires approval."""
    return {"sent": True, "message": message}


class _FixtureProviderCallsGatedTool:
    async def complete(self, messages, tools, system=None):
        return LLMResponse(
            text=None,
            tool_calls=[ToolCallRequest(id="c1", name="send_alert", arguments={"message": "disk full"})],
            stop_reason="tool_use",
        )


if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp()
    store_path = str(Path(tmpdir) / "mcp_server_fixture_gated.db")

    agent = Agent(
        provider=_FixtureProviderCallsGatedTool(),
        tools=[send_alert],
        store=f"sqlite:///{store_path}",
    )
    mcp_server = serve_agent(agent, name="ops-agent")
    mcp_server.run(transport="stdio")