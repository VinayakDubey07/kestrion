"""
Tests for kestrion.mcp.client, run against a REAL MCP server — the test
fixture at tests/fixtures/mock_mcp_server.py — launched as a subprocess
over stdio, exactly like a real third-party MCP server would be. This
is not testing against fakes; it's testing against the real `mcp`
package's real protocol implementation, just with a server we wrote
ourselves instead of a third party's.

Skips automatically if the `mcp` package isn't installed, so this
doesn't break `pytest tests/` for anyone who hasn't installed the
optional mcp extra.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp package not installed — install with: pip install kestrion[mcp]")

from kestrion.mcp.client import MCPClient

FIXTURE_SERVER = str(Path(__file__).parent.parent / "fixtures" / "mock_mcp_server.py")


async def test_list_tools_returns_tools_advertised_by_the_server():
    async with MCPClient.stdio(command=sys.executable, args=[FIXTURE_SERVER]) as client:
        tools = await client.list_tools()
        names = {t.spec.name for t in tools}

    assert names == {"get_cluster_state", "apply_manifest", "failing_tool"}


async def test_tool_spec_carries_description_and_schema_from_the_server():
    async with MCPClient.stdio(command=sys.executable, args=[FIXTURE_SERVER]) as client:
        tools = await client.list_tools()
        apply_tool = next(t for t in tools if t.spec.name == "apply_manifest")

    assert "manifest" in apply_tool.spec.description.lower()
    assert apply_tool.spec.parameters["properties"]["yaml"]["type"] == "string"


async def test_mcp_tool_call_returns_real_result_from_the_server():
    async with MCPClient.stdio(command=sys.executable, args=[FIXTURE_SERVER]) as client:
        tools = await client.list_tools()
        get_state = next(t for t in tools if t.spec.name == "get_cluster_state")
        result = await get_state.call()

    assert result.error is None
    # Output may arrive as structured content (a dict) or as text
    # depending on transport/SDK version — check for the data either way
    # rather than assuming one specific shape.
    output_str = str(result.output)
    assert "checkout-api" in output_str
    assert "2" in output_str


async def test_mcp_tool_call_with_arguments():
    async with MCPClient.stdio(command=sys.executable, args=[FIXTURE_SERVER]) as client:
        tools = await client.list_tools()
        apply_tool = next(t for t in tools if t.spec.name == "apply_manifest")
        result = await apply_tool.call(yaml="replicas: 3")

    assert result.error is None
    assert "replicas: 3" in str(result.output)


async def test_mcp_tool_call_failure_surfaces_as_recoverable_error_not_a_crash():
    """
    Mirrors how Engine.call_tool already treats tool failures elsewhere:
    a tool raising on the server side must become a ToolResult.error,
    never an unhandled exception that crashes the whole agent run.
    """
    async with MCPClient.stdio(command=sys.executable, args=[FIXTURE_SERVER]) as client:
        tools = await client.list_tools()
        failing = next(t for t in tools if t.spec.name == "failing_tool")
        result = await failing.call()  # must not raise

    assert result.error is not None
    assert "intentional failure" in result.error or "error" in result.error.lower()


async def test_list_tools_requires_approval_gates_named_tools():
    """
    Regression test for the design decision in client.py: MCP itself has
    no approval concept, so without an explicit opt-in every MCP tool
    would silently default to requires_approval=False. This confirms the
    override actually works and is name-scoped (only the named tool is
    gated, not every tool from the server).
    """
    async with MCPClient.stdio(command=sys.executable, args=[FIXTURE_SERVER]) as client:
        tools = await client.list_tools(requires_approval=["apply_manifest"])

    by_name = {t.spec.name: t.spec.requires_approval for t in tools}
    assert by_name["apply_manifest"] is True
    assert by_name["get_cluster_state"] is False
    assert by_name["failing_tool"] is False


async def test_gated_mcp_tool_integrates_with_agent_approval_flow(tmp_store):
    """
    The actual end-to-end proof: an MCP-sourced tool, marked gated via
    list_tools(requires_approval=...), pauses an Agent run exactly the
    same way a @tool(requires_approval=True) function does. This is the
    real point of designing MCPTool to satisfy the same Tool contract —
    the engine genuinely cannot tell the difference.
    """
    from kestrion.agent.agent import Agent
    from kestrion.llm.base import LLMResponse, ToolCallRequest

    class FakeProviderCallsGatedMCPTool:
        async def complete(self, messages, tools, system=None):
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="c1", name="apply_manifest", arguments={"yaml": "replicas: 3"})],
                tokens_in=10, tokens_out=10, cost_usd=0.0001,
                stop_reason="tool_use",
            )

    async with MCPClient.stdio(command=sys.executable, args=[FIXTURE_SERVER]) as client:
        tools = await client.list_tools(requires_approval=["apply_manifest"])
        agent = Agent(
            provider=FakeProviderCallsGatedMCPTool(),
            tools=tools,
            store=f"sqlite:///{tmp_store.path}",
        )
        result = await agent.run("Scale up checkout-api")

    assert result.status.value == "waiting_on_human"
    assert result.state.scratch["_pending_approval"]["tool"] == "apply_manifest"
