"""
Tests for kestrion.mcp.server, run against REAL MCP server subprocesses
(tests/fixtures/mcp_server_fixture.py and mcp_server_fixture_gated.py)
via kestrion.mcp.client.MCPClient.stdio() — the exact same proven
stdio-subprocess pattern test_mcp_client.py already uses successfully
to test the CLIENT side. This is the server-side mirror.

NOTE: an earlier version of this file used `from mcp import Client`,
following documentation for the separate `fastmcp` standalone package
(a different library with a confusingly similar name) rather than the
official `mcp` SDK this project actually depends on — that import does
not exist in mcp's real API surface and failed at collection time. This
version uses MCPClient.stdio(), the same mechanism already proven
correct elsewhere in this codebase, rather than guessing at an
unverified API a second time.

Skips automatically if the `mcp` package isn't installed.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp package not installed — install with: pip install kestrion[mcp]")

from kestrion.mcp.client import MCPClient

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
WEATHER_SERVER = str(FIXTURE_DIR / "mcp_server_fixture.py")
GATED_SERVER = str(FIXTURE_DIR / "mcp_server_fixture_gated.py")


async def test_served_agent_advertises_a_single_ask_agent_tool():
    async with MCPClient.stdio(command=sys.executable, args=[WEATHER_SERVER]) as client:
        tools = await client.list_tools()

    assert len(tools) == 1
    assert tools[0].spec.name == "ask_agent"


async def test_ask_agent_tool_description_is_set_correctly():
    """
    Regression test: f-string as first statement is NOT a docstring —
    only a literal string constant is. Fixed by assigning __doc__
    explicitly before the decorator reads it.
    """
    async with MCPClient.stdio(command=sys.executable, args=[WEATHER_SERVER]) as client:
        tools = await client.list_tools()

    assert tools[0].spec.description == "Ask about the weather, very specifically"


async def test_default_description_used_when_none_provided():
    async with MCPClient.stdio(command=sys.executable, args=[GATED_SERVER]) as client:
        tools = await client.list_tools()

    assert "ops-agent" in tools[0].spec.description


async def test_calling_ask_agent_runs_the_full_agent_and_returns_its_answer():
    async with MCPClient.stdio(command=sys.executable, args=[WEATHER_SERVER]) as client:
        tools = await client.list_tools()
        ask_agent = tools[0]
        result = await ask_agent.call(prompt="What's the weather in Bangalore?")

    assert result.error is None
    assert "sunny" in str(result.output)


async def test_paused_run_is_surfaced_as_a_normal_result_not_an_mcp_error():
    """
    A paused run is a legitimate state to report back, not a protocol
    failure — isError must NOT be set, and the missing roles + tool
    name must be clearly present in the response text.
    """
    async with MCPClient.stdio(command=sys.executable, args=[GATED_SERVER]) as client:
        tools = await client.list_tools()
        ask_agent = tools[0]
        result = await ask_agent.call(prompt="Send an alert about the full disk")

    assert result.error is None
    text = str(result.output)
    assert "paused" in text
    assert "send_alert" in text
    assert "missing roles" in text.lower() or "__any__" in text