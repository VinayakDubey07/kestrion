"""
Worked example: Exposing a Kestrion Agent as an MCP server.

This allows external platforms (like Claude Code, Cursor, or another Kestrion Agent)
to connect to this agent and call it as a tool.

Usage (testing locally via stdio):
    python examples/mcp_server.py
"""

import sys
import os

from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.llm.ollama_provider import OllamaProvider
from kestrion.mcp.server import serve_agent


@tool(requires_approval=False)
def get_cluster_state() -> str:
    """Read current deployment replica counts."""
    return '{"deployment": "checkout-api", "replicas": 2}'


@tool(requires_approval=True)
def apply_manifest(yaml: str) -> str:
    """kubectl apply a manifest against the cluster."""
    return f"Applied manifest successfully:\n{yaml}"


def main():
    # Setup our agent (Using Ollama as it's locally available without API keys)
    provider = OllamaProvider(model=os.environ.get("OLLAMA_MODEL", "llama3.2"))
    
    agent = Agent(
        provider=provider,
        tools=[get_cluster_state, apply_manifest],
        store="sqlite:///mcp_agent_demo.db"
    )

    # Wrap the Kestrion Agent in an MCP Server
    mcp_server = serve_agent(
        agent, 
        name="kestrion-ops-agent", 
        description="Ops agent capable of checking cluster state and applying manifests."
    )
    
    # We print to stderr because stdout is used by the MCP stdio protocol
    print("Starting Kestrion MCP Server over stdio...", file=sys.stderr)
    
    # Run the server on standard IO
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
