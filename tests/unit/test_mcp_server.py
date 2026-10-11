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


async def test_serve_agent_in_process_branches():
    from unittest.mock import AsyncMock, MagicMock
    from kestrion.agent.agent import RunResult
    from kestrion.core.types import AgentState, RunStatus
    from kestrion.mcp.server import serve_agent

    mock_agent = MagicMock()
    server = serve_agent(mock_agent, name="inproc-agent", description="In-process agent description")
    assert server.name == "inproc-agent"

    # 1. COMPLETED
    mock_agent.run = AsyncMock(return_value=RunResult(
        run_id="run_1", status=RunStatus.COMPLETED, output="All done!", state=AgentState(run_id="run_1")
    ))
    res, _ = await server.call_tool("ask_agent", {"prompt": "hello"})
    assert "All done!" in res[0].text

    # 2. WAITING_ON_HUMAN
    st = AgentState(run_id="run_2")
    st.scratch["_pending_approval"] = {"tool": "dangerous_action", "missing_roles": ["admin"]}
    mock_agent.run = AsyncMock(return_value=RunResult(
        run_id="run_2", status=RunStatus.WAITING_ON_HUMAN, output=None, state=st
    ))
    res, _ = await server.call_tool("ask_agent", {"prompt": "run dangerous"})
    assert "paused: run run_2 is waiting on approval" in res[0].text

    # 3. EXPIRED
    mock_agent.run = AsyncMock(return_value=RunResult(
        run_id="run_3", status=RunStatus.EXPIRED, output=None, state=AgentState(run_id="run_3")
    ))
    res, _ = await server.call_tool("ask_agent", {"prompt": "expired"})
    assert "expired" in res[0].text

    # 4. FAILED
    mock_agent.run = AsyncMock(return_value=RunResult(
        run_id="run_4", status=RunStatus.FAILED, output=None, state=AgentState(run_id="run_4")
    ))
    res, _ = await server.call_tool("ask_agent", {"prompt": "failed"})
    assert "failed" in res[0].text