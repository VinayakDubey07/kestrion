"""
MCP client. Connects to an external MCP server (stdio or streamable-HTTP
transport) and exposes every tool it advertises as a Kestrion Tool —
meaning they drop straight into Agent(tools=[...]) alongside @tool
functions, indistinguishable from the engine's perspective. This is the
actual payoff of designing ToolSpec.parameters as a JSON schema from
day one: an MCP tool's inputSchema IS a ToolSpec.parameters value,
no translation needed.

Requires the `mcp` package — install via `pip install kestrion[mcp]`,
consistent with every other optional integration in this project.

NOTE ON VERIFICATION STATUS: this client is implemented against the
documented MCP Python SDK API surface (ClientSession, StdioServerParameters,
stdio_client, CallToolResult) but has only been exercised against a
purpose-built test server (see tests/unit/test_mcp_client.py), not yet
against a real third-party MCP server. Same honesty standard as the
Anthropic/OpenAI providers in llm/ — see README "Known gaps".
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from kestrion.core.types import Tool, ToolResult, ToolSpec

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as exc:
    raise ImportError(
        "The mcp package is required to use MCP integration. "
        "Install it with: pip install kestrion[mcp]"
    ) from exc

try:
    from mcp.client.streamable_http import streamablehttp_client
    _HTTP_TRANSPORT_AVAILABLE = True
except ImportError:
    # Older mcp SDK versions may not have this transport yet — stdio
    # still works without it, so don't hard-fail the whole import.
    _HTTP_TRANSPORT_AVAILABLE = False


def _extract_text(content_blocks: list[Any]) -> str:
    """
    MCP CallToolResult.content is a list of content blocks (TextContent,
    ImageContent, EmbeddedResource, ...). Most tool results are plain
    text. This joins every text block found; non-text blocks are
    represented by a short placeholder rather than silently dropped, so
    at least their presence is visible to whoever reads the result.
    """
    parts = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(block.text)
        else:
            parts.append(f"[{block_type or 'unknown'} content omitted]")
    return "\n".join(parts)


class MCPTool(Tool):
    """
    Wraps a single tool advertised by a connected MCP server. Constructed
    by MCPClient.list_tools() — not meant to be instantiated directly.
    """

    def __init__(self, session: ClientSession, spec: ToolSpec):
        self._session = session
        self.spec = spec

    async def call(self, **kwargs) -> ToolResult:
        result = await self._session.call_tool(self.spec.name, arguments=kwargs)

        # Defend against the documented gap (modelcontextprotocol/python-sdk
        # issue #3313): some servers return data via structuredContent
        # instead of (or in addition to) content. Checking only .content
        # would silently lose that data.
        text_output = _extract_text(result.content) if result.content else ""
        structured = getattr(result, "structuredContent", None) or getattr(
            result, "structured_content", None
        )
        output = structured if structured is not None else text_output

        if getattr(result, "isError", False):
            return ToolResult(tool_name=self.spec.name, output=None, error=text_output or "MCP tool reported an error")

        return ToolResult(tool_name=self.spec.name, output=output)


class MCPClient:
    """
    Manages one connection to one MCP server. Use as an async context
    manager so the underlying subprocess/connection is always cleaned
    up, even on error:

        async with MCPClient.stdio(command="python", args=["server.py"]) as client:
            tools = await client.list_tools()
            agent = Agent(provider=..., tools=tools)
    """

    def __init__(self):
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    @classmethod
    def stdio(cls, command: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> "MCPClient":
        """Connect to a server launched as a local subprocess over stdio."""
        client = cls()
        client._connect_kind = "stdio"
        client._stdio_params = StdioServerParameters(command=command, args=args or [], env=env)
        return client

    @classmethod
    def http(cls, url: str) -> "MCPClient":
        """Connect to a remote server over streamable HTTP."""
        if not _HTTP_TRANSPORT_AVAILABLE:
            raise ImportError(
                "Streamable HTTP transport is not available in the installed "
                "mcp package version. Upgrade with: pip install -U kestrion[mcp]"
            )
        client = cls()
        client._connect_kind = "http"
        client._http_url = url
        return client

    async def __aenter__(self) -> "MCPClient":
        if self._connect_kind == "stdio":
            read, write = await self._exit_stack.enter_async_context(stdio_client(self._stdio_params))
        else:
            read, write, _get_session_id = await self._exit_stack.enter_async_context(
                streamablehttp_client(self._http_url)
            )
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._exit_stack.aclose()

    async def list_tools(self, requires_approval: list[str] | None = None) -> list[MCPTool]:
        """
        Fetches every tool the connected server advertises and wraps each
        as an MCPTool — a real Kestrion Tool, ready to pass straight into
        Agent(tools=[...]).

        MCP itself has no first-class "requires human approval" concept —
        that's a Kestrion-specific safety mechanism, not part of the MCP
        spec. Without an explicit opt-in, every MCP-sourced tool would
        default to NOT requiring approval, which would silently undermine
        the approval-gating story for exactly the case (kubectl, infra
        tools) this project cares most about. `requires_approval` lets the
        caller name which tools, by their MCP-advertised name, should be
        gated — e.g. requires_approval=["apply_manifest", "delete_pod"].
        """
        if self._session is None:
            raise RuntimeError("MCPClient must be used as 'async with MCPClient...() as client:'")

        gated = set(requires_approval or [])
        result = await self._session.list_tools()
        tools = []
        for mcp_tool in result.tools:
            spec = ToolSpec(
                name=mcp_tool.name,
                description=mcp_tool.description or f"Calls {mcp_tool.name} via MCP",
                parameters=mcp_tool.inputSchema,
                requires_approval=mcp_tool.name in gated,
            )
            tools.append(MCPTool(self._session, spec))
        return tools