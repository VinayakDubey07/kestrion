# MCP

Kestrion integrates with the Model Context Protocol (MCP) in both directions:

1. **MCP Client**: Connect to real MCP servers and use their tools exactly like `@tool` functions, including approval gating. Live-verified against a real test-fixture server — see [`tests/unit/test_mcp_client.py`](https://github.com/VinayakDubey07/kestrion/blob/main/tests/unit/test_mcp_client.py).
2. **MCP Server**: Expose a Kestrion `Agent` as a real MCP server so that external clients (e.g. Claude Code) can invoke it. See [`tests/unit/test_mcp_server.py`](https://github.com/VinayakDubey07/kestrion/blob/main/tests/unit/test_mcp_server.py).

::: kestrion.mcp.client

::: kestrion.mcp.server