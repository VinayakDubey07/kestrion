# Kestrion

A durable-execution-first framework for building production AI agents.

Status: pre-alpha (`0.2.0`), published on PyPI. Core engine, the `Agent`/`@tool` decorator API,
three LLM providers, a live-verified MCP client, and four agentic features (multi-step approval
chains, time-boxed approvals, parallel tool calls, sub-agents) are built and tested — 75 passing
tests. Multi-agent handoff, memory/context compaction, an MCP server, a scheduler, a CLI, and
Postgres support are designed but not yet implemented — see [Roadmap](#roadmap) below.

## Why Kestrion

Most agent frameworks are strong at *authoring* an agent loop. Kestrion is built around a narrower,
specific bet: **state is never mutated directly — it's derived by folding an immutable log of
events.** That single decision is what makes the following true by construction, not by careful
discipline on the part of whoever writes a given agent:

- **Crash recovery is the default.** Any process — the original one or a brand new one — can
  reconstruct a run's exact state from the store and continue it.
- **Human-approval gates pause the run itself**, not just a function call. A tool marked as
  requiring approval can't be invoked without it, enforced centrally by the engine.
- **Observability comes from the same log everything else does** — token counts, cost, and full
  trace history, not a separate system bolted on after.

## Install

```bash
pip install kestrion[anthropic]   # or [openai], [ollama], [mcp], or [all]
```

Each LLM provider and MCP support are optional extras. If you only use Ollama, you never need the
`anthropic` or `openai` packages installed.

## Quickstart

```python
import asyncio
from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.llm.anthropic_provider import AnthropicProvider

@tool
def get_cluster_state() -> dict:
    """Read current deployment replica counts."""
    return {"deployment": "checkout-api", "replicas": 2}

@tool(requires_approval=True)
def apply_manifest(yaml: str) -> dict:
    """kubectl apply a manifest against the cluster."""
    # real kubectl call would go here
    return {"applied": True}

async def main():
    agent = Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        tools=[get_cluster_state, apply_manifest],
        store="sqlite:///agent_runs.db",
    )
    result = await agent.run("Check checkout-api and scale it up by one if it's under 3 replicas")
    print(result.status)   # "waiting_on_human" — paused before the mutating call
    print(result.output)

asyncio.run(main())
```

The run pauses with `status=waiting_on_human` the moment the model decides to call
`apply_manifest`, since that tool is marked `requires_approval=True`. Nothing executes against the
real cluster until that's explicitly approved.

### Resuming a paused run

Resuming works from a completely independent process — this is the actual crash-recovery
guarantee, not just a convenience method:

```python
# Anywhere else, any time later, sharing only the same store file:
from kestrion.core.engine import Engine
Engine.record_approval(state, "apply_manifest", role="__any__")
# (persist that as a checkpoint — see examples/kubectl_agent for the full pattern;
#  Agent.approve() is not yet a polished one-liner, see Known Gaps below)

result = await agent.resume(run_id)
print(result.status)  # "completed"
```

### Calling REST or SOAP APIs from a tool

`@tool` wraps any Python function, so calling an external API is no different from any other
tool — there's no special Kestrion API for this:

```python
import httpx
from kestrion.agent.decorators import tool

@tool
def get_order_status(order_id: str) -> dict:
    """Fetch order status from the orders API."""
    response = httpx.get(f"https://api.example.com/orders/{order_id}", timeout=10.0)
    response.raise_for_status()
    return response.json()
```

Any exception the function raises — a timeout, a 4xx/5xx, a connection error — is automatically
caught and turned into a clean `ToolResult.error` rather than crashing the run, exactly like every
other tool. What's **not** automatic: timeouts, retries, and secrets handling are on you to write
explicitly. See [`examples/rest_api_tool`](examples/rest_api_tool) for the patterns that matter in
practice. SOAP follows the identical shape with `zeep` instead of `httpx`.

### Multi-step approval chains

A tool can require approval from more than one role, not just a single yes/no:

```python
@tool(requires_approval=["engineer", "manager"])
def deploy_to_prod() -> dict:
    """Deploys to production. Needs both an engineer and a manager to sign off."""
    ...
```

The run stays paused until every required role has approved — recorded via
`Engine.record_approval(state, "deploy_to_prod", role="engineer")`, which adds a role without
clobbering any already recorded (writing to `scratch` directly can silently destroy a
partially-satisfied chain — use `record_approval`, not a manual dict assignment).

### Time-boxed approvals

A gated tool can carry a deadline. If nobody approves in time, the run transitions to a new
terminal status, `EXPIRED`, instead of waiting forever:

```python
@tool(requires_approval=True, approval_timeout_seconds=3600.0)
def restart_service() -> dict:
    """Restarts a service. Must be approved within an hour."""
    ...

result = await agent.resume(run_id)            # default: status -> EXPIRED if the deadline passed
result = await agent.resume(run_id, on_expired="raise")  # or raise RunExpiredError instead
```

### Parallel tool calls

If a model requests multiple tool calls in one turn, Kestrion runs them concurrently rather than
one at a time — with a safety guarantee: a batch either fully executes or cleanly pauses with
nothing partially run. If any call in the batch is gated and unapproved, **none** of the calls in
that batch run, not even the safe ones sitting alongside it.

### Sub-agents

Any `Agent` can be wrapped as a tool another agent calls — delegation with zero new engine
machinery:

```python
specialist = Agent(provider=..., tools=[...], store=shared_store_url)
planner = Agent(
    provider=...,
    tools=[specialist.as_tool("check_inventory", "Ask the inventory specialist")],
    store=shared_store_url,  # SAME store — required for the sub-agent's run to be independently resumable
)
```

If the sub-agent's run pauses for approval, the **parent** run pauses too — the parent's
`scratch["_pending_approval"]["missing_roles"]` will contain `"sub_agent:<run_id>"`, naming exactly
which nested run needs resuming first.

### MCP client

Connect to a real MCP server and use its tools exactly like `@tool` functions, including approval
gating:

```python
from kestrion.mcp.client import MCPClient

async with MCPClient.stdio(command="python3", args=["my_mcp_server.py"]) as client:
    tools = await client.list_tools(requires_approval=["apply_manifest"])
    agent = Agent(provider=..., tools=tools, store="sqlite:///agent_runs.db")
```

MCP itself has no approval concept — `requires_approval` here is how you opt specific MCP tools
into Kestrion's gating, by name.

## What you can build with this today

- Tool-calling agents where some actions are safe to auto-run and others need a human in the loop
  first — infrastructure agents, ops bots, anything touching a database or cluster.
- Multi-step approval workflows requiring sign-off from more than one role, optionally with a
  deadline after which the request expires.
- Agents that delegate sub-tasks to other agents, including correct approval propagation when a
  sub-agent's action needs sign-off.
- Agents that call tools sourced from a real MCP server, not just hand-written Python functions.
- Agents that need to survive a crash or restart mid-task. `agent.resume(run_id)` works from a
  totally different process than the one that started the run.
- Multi-turn tool use, including multiple tool calls per turn running concurrently.

## Known gaps (honest, not aspirational)

- **MCP client is live-verified; MCP server is not built.** `kestrion.mcp.client.MCPClient`
  connects to real MCP servers (stdio or streamable-HTTP) and is tested against a real test-fixture
  server, including the full approval-gating flow. Exposing a Kestrion `Agent` itself as an MCP
  server (so it's callable from Claude Code or Codex CLI) is designed but not implemented.
- **Anthropic and OpenAI providers are implemented against documented API shapes but not yet
  smoke-tested against a live API call** — no API key has been used to verify them in practice.
  **Ollama is verified live** — `tests/unit/test_smoke_ollama.py` runs a real agent against a real
  local Ollama server and passes.
- **Multi-agent handoff and memory/context compaction are not yet built.** Sub-agents (delegation,
  where the parent stays in control) exist; handoff (transferring an entire conversation to a
  different agent that takes over) does not.
- **No real concurrency control across multiple agent runs.** Parallel tool calls *within* one
  agent's turn are supported; running many separate agents at once against a shared rate limit is
  not.
- **No CLI or deploy story.** `kestrion deploy --target k8s` doesn't exist yet — you'd containerize
  and deploy this yourself today.
- **`Agent.approve()` is a stub.** Approving a paused run currently means manually calling
  `Engine.record_approval` and saving a checkpoint by hand (see `examples/kubectl_agent`), not a
  polished one-line call.
- **SQLite only.** A `CheckpointStore` Protocol exists so Postgres can be added without touching
  the engine, but that implementation doesn't exist yet.

## Examples

- [`examples/kubectl_agent`](examples/kubectl_agent) — the original worked example, demonstrating
  pause-on-approval and resume-after-restart using the raw `Engine`/`Node` primitives directly
  (useful for understanding what `Agent` builds on top of).
- [`examples/rest_api_tool`](examples/rest_api_tool) — patterns for calling REST/SOAP APIs from a
  tool: explicit timeouts, gating a mutating call, reading secrets from the environment, and
  writing your own retry loop.
- [`tests/unit/test_smoke_ollama.py`](tests/unit/test_smoke_ollama.py) — a live, real smoke test
  against a local Ollama server. Skips automatically if Ollama isn't running.
- [`tests/unit/test_mcp_client.py`](tests/unit/test_mcp_client.py) — a live test against a real
  MCP server (`tests/fixtures/mock_mcp_server.py`), including the approval-gating integration.

## Documentation

- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- Concepts: [Event Sourcing](docs/concepts/event-sourcing.md) ·
  [Checkpointing](docs/concepts/checkpointing.md) ·
  [Approval Gates](docs/concepts/approval-gates.md)

## Development

```bash
git clone https://github.com/VinayakDubey07/kestrion.git
cd kestrion
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## Roadmap

Next up: multi-agent handoff, memory/context compaction, an MCP server, a scheduler for safe
concurrent execution, a CLI with Kubernetes deploy support, Postgres-backed storage, and a docs
site.

## License

Apache 2.0