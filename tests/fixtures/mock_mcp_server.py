"""
A real, minimal MCP server, used purely as a test fixture for
tests/unit/test_mcp_client.py. This is NOT a mock in the sense of fake
objects — it's a genuine MCP server, speaking the real protocol over
stdio, built with the same FastMCP class anyone would use to build a
real one. The client test launches this as a subprocess and talks to
it over the real wire protocol.

Deliberately mirrors the shape of the kubectl example: one safe,
read-only tool and one "mutating" tool, so the same approval-gating
pattern can be exercised through MCP instead of through @tool.

Run directly with: python3 tests/fixtures/mock_mcp_server.py
(it will sit waiting on stdio for a client — that's normal, see
mcp/client.py for the test that drives it)
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("kestrion-test-fixture")


@mcp.tool()
def get_cluster_state() -> dict:
    """Read current deployment replica counts."""
    return {"deployment": "checkout-api", "replicas": 2}


@mcp.tool()
def apply_manifest(yaml: str) -> dict:
    """Apply a manifest against the cluster. (Test fixture — does not touch anything real.)"""
    return {"applied": True, "yaml": yaml}


@mcp.tool()
def failing_tool() -> str:
    """A tool that always raises, used to test MCP error-result handling."""
    raise RuntimeError("intentional failure for testing")


if __name__ == "__main__":
    mcp.run(transport="stdio")
