"""
A real MCP server process, built with kestrion.mcp.server.serve_agent,
used as a test fixture for tests/unit/test_mcp_server.py. Launched as a
subprocess over stdio by MCPClient.stdio() — the exact same proven
pattern tests/fixtures/mock_mcp_server.py already uses successfully for
testing the CLIENT side. This is the server-side mirror of that file.

Uses a fake, scripted LLMProvider (not a real model) so this fixture
runs deterministically with zero network calls and no API key — the
behavior under test is serve_agent()'s MCP wiring, not model behavior.

Run directly with: python3 tests/fixtures/mcp_server_fixture.py
(it will sit waiting on stdio for a client — that's normal)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.llm.base import LLMResponse
from kestrion.mcp.server import serve_agent


@tool
def get_weather(city: str) -> dict:
    """Look up the weather for a city."""
    return {"city": city, "condition": "sunny"}


class _FixtureProvider:
    """Scripted, deterministic: always answers directly, never calls a tool."""

    async def complete(self, messages, tools, system=None):
        return LLMResponse(text="It's sunny in Bangalore.", tool_calls=[], stop_reason="end_turn")


if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp()
    store_path = str(Path(tmpdir) / "mcp_server_fixture.db")

    agent = Agent(
        provider=_FixtureProvider(),
        tools=[get_weather],
        store=f"sqlite:///{store_path}",
    )
    mcp_server = serve_agent(agent, name="weather-agent", description="Ask about the weather, very specifically")
    mcp_server.run(transport="stdio")