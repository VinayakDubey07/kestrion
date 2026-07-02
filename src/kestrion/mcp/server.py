"""
MCP server. Exposes a Kestrion Agent as a real MCP server — the reverse
direction of mcp/client.py, which lets Kestrion CONSUME tools from an
external MCP server. This module lets a Kestrion agent BE the thing
something like Claude Code or Codex CLI connects to and calls.

DESIGN DECISION, made deliberately rather than the more obvious-seeming
alternative: this exposes the agent's full reasoning loop as ONE MCP
tool (ask_agent(prompt)), not the agent's individual tools advertised
separately. Exposing raw tools individually would let an MCP caller
invoke them directly, bypassing Engine.call_tool's approval gating
entirely — since that gate only runs when a call goes through Agent.
That would be a real safety regression, not just a different API shape.
This mirrors how SubAgentTool already wraps a full agent loop as one
callable thing internally; this module does the same thing, over the
wire, via MCP.

Requires the `mcp` package — install via `pip install kestrion[mcp]`,
consistent with every other optional integration in this project.

NOTE ON VERIFICATION STATUS: implemented against the documented FastMCP
API surface (same one tests/fixtures/mock_mcp_server.py already uses
successfully) but this server side has not yet been exercised against a
real third-party MCP client (e.g. actually connecting from Claude Code).
See README "Known gaps" for the project's standard honesty pattern on
this kind of claim.
"""

from __future__ import annotations

from kestrion.core.types import RunStatus

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "The mcp package is required to use MCP server integration. "
        "Install it with: pip install kestrion[mcp]"
    ) from exc


def serve_agent(agent, name: str, description: str | None = None) -> FastMCP:
    """
    Builds a FastMCP server exposing `agent` as a single tool, ask_agent.
    Returns the FastMCP instance — call .run(transport=...) on it to
    actually start serving (stdio for local/subprocess use, matching the
    transport examples/kubectl_agent and the MCP client tests already
    use; streamable-http for a network-reachable server).

    Usage:
        agent = Agent(provider=..., tools=[...], store="sqlite:///agent.db")
        mcp_server = serve_agent(agent, name="kestrion-ops-agent")
        mcp_server.run(transport="stdio")

    The exposed tool's behavior on a paused run is deliberate: a run
    pausing for approval is NOT treated as an MCP error (isError=True).
    It's a legitimate, useful state — the caller gets back the run_id
    and a clear message that approval is needed, exactly the same
    "status to check, not an exception to catch" pattern WAITING_ON_HUMAN
    uses everywhere else in this project. Resuming a paused run isn't
    exposed as a separate MCP tool yet — Agent.approve() is still a
    stub (see README Known Gaps), and that gap applies here too.
    """
    mcp = FastMCP(name)

    async def ask_agent(prompt: str) -> str:
        result = await agent.run(prompt)

        if result.status == RunStatus.COMPLETED:
            return result.output or ""

        if result.status == RunStatus.WAITING_ON_HUMAN:
            pending = result.state.scratch.get("_pending_approval", {})
            return (
                f"[paused: run {result.run_id} is waiting on approval for tool "
                f"'{pending.get('tool')}', missing roles: {pending.get('missing_roles')}. "
                f"Approve via Engine.record_approval and resume separately — "
                f"see the README for the current manual pattern, or check back "
                f"once Agent.approve() is implemented.]"
            )

        if result.status == RunStatus.EXPIRED:
            return f"[run {result.run_id} expired waiting for approval before completing.]"

        if result.status == RunStatus.FAILED:
            return f"[run {result.run_id} failed.]"

        return f"[run {result.run_id} ended with unexpected status: {result.status}]"

    # ask_agent's docstring needs to be the user-provided description,
    # which is only known at runtime (it's a function parameter, not a
    # literal). An f-string as the first statement in the function body
    # is NOT a docstring in Python -- only a literal string constant is
    # assigned to __doc__; an f-string is just a discarded expression.
    # That was a real bug caught here before shipping: it would have
    # silently left every served agent's MCP tool description empty,
    # which matters because the description is the only thing an MCP
    # client's model has to decide whether to call the tool at all.
    # Fixed by assigning __doc__ directly before the decorator reads it.
    ask_agent.__doc__ = description or f"Ask the {name} agent a question and get its answer."
    mcp.tool()(ask_agent)

    return mcp