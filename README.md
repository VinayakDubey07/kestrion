# Kestrion

A durable-execution-first framework for building production AI agents.

Status: **v0.3 — actively developed**. Published on PyPI. Core engine, the `Agent`/`@tool` decorator API,
three LLM providers, a live-verified MCP client and server, a CLI, **ten** agentic features
(vision/multi-modal support, multi-step approval chains, time-boxed approvals, parallel tool calls, sub-agents, multi-agent
handoff, memory/context compaction, human-in-the-loop input, enterprise secret management, advanced context window management), and a **DAG-based async scheduler** for multi-agent
orchestration are built and tested — 162 passing tests. **Horizontally scalable, multi-worker Postgres support** and rate-limited scheduler are built.

| Feature | Status | Proof |
|---------|--------|-------|
| Event Sourcing & Crash Recovery | Proven | Built-in, tested across all agent runs |
| Approval Gates (RBAC, Timeouts) | Proven | Live-verified with MCP client/server |
| Horizontal Scale (Multi-worker) | Proven | `postgres_store.py` + DAG Scheduler |
| Security (Secret injection) | Proven | `SecretProvider` protocol |
| Vision / Multi-modal Support | Proven | Native `TextBlock` / `ImageBlock` |
| OpenTelemetry / Observability | Proven | `OpenTelemetryProvider` |

See [Roadmap](#roadmap) below.

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
guarantee. Approving a paused run is a clean one-liner:

```python
# Anywhere else, any time later, sharing only the same store file:
result = await agent.approve(run_id)
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

### Human-in-the-Loop Input (Ask for Input)

While approval gates pause the engine for a binary yes/no decision, sometimes the agent needs a specific piece of information from a human (like an API key, 2FA code, or clarification). You can use the built-in `ask_human` tool (or raise `InputRequired` directly in your own tools). 

When triggered, the engine suspends execution exactly like an approval, transitioning the run to `WAITING_ON_HUMAN`. You can then provide the text response back via the API to resume execution:

```python
from kestrion.agent.tools import ask_human

# The agent invokes ask_human and the run halts.
# The user's application queries the status and prompts the human for input:
await agent.provide_input(run_id, text="My favorite color is Blue.", tool="ask_human")
# The engine records the input and resumes execution cleanly!
```

### Vision & Multi-modal Support

Kestrion treats image inputs natively via strictly typed content blocks (`TextBlock` and `ImageBlock`). You can seamlessly pass a list of these blocks directly to the agent's `run()` method instead of a standard string. Kestrion automatically maps them to the correct wire format for your chosen provider (Anthropic, OpenAI, or Ollama):

```python
import base64
from kestrion.llm.base import TextBlock, ImageBlock

with open("receipt.jpg", "rb") as f:
    base64_img = base64.b64encode(f.read()).decode("utf-8")

result = await agent.run([
    TextBlock(text="Extract the total amount from this receipt and format it as JSON."),
    ImageBlock(data=base64_img, media_type="image/jpeg")
])
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

### MCP server

Expose a Kestrion `Agent` as a real MCP server so something like Claude Code can connect to it
and call its full reasoning loop as a single tool:

```python
from kestrion.mcp.server import serve_agent

agent = Agent(provider=..., tools=[...], store="sqlite:///agent.db")
mcp_server = serve_agent(agent, name="ops-agent", description="Ask the ops agent a question")
mcp_server.run(transport="stdio")
```

This exposes one MCP tool — `ask_agent(prompt)` — rather than the agent's individual raw tools.
That's intentional: exposing raw tools would let a caller invoke them directly, bypassing
`Engine.call_tool`'s approval gating entirely. The full reasoning loop, including all approval
gates, runs on every `ask_agent` call. A paused run is surfaced as a clear message (not an MCP
error) so the caller knows a human approval is pending.

### Pipeline / multi-agent orchestration

Run a team of agents as a **DAG** — some agents execute concurrently, others wait until their
upstream dependencies finish:

```python
from kestrion.scheduler import Pipeline, AgentTask, RateLimiterConfig

pipeline = Pipeline(
    tasks=[
        AgentTask("researcher_a", agent=agent_a, prompt="Research Python's history and use cases"),
        AgentTask("researcher_b", agent=agent_b, prompt="Research Rust's history and use cases"),
        AgentTask(
            "synthesizer",
            agent=synth_agent,
            prompt="Compare the two languages",
            depends_on=["researcher_a", "researcher_b"],   # waits for both
        ),
    ],
    max_workers=3,
    rate_limiter_config=RateLimiterConfig(requests_per_minute=60),  # None for Ollama
)

results = await pipeline.run()
for name, result in results.items():
    print(f"{name}: {result.status.value} — {result.run_result.output[:80]}")
```

`researcher_a` and `researcher_b` run **concurrently** in the worker pool. `synthesizer` only
starts once both complete — dependency resolution uses `asyncio.Event` per task, with no polling.
If one task fails, independent branches keep running (`fail_fast=False` default); dependents are
marked `SKIPPED` with a clear reason. Every run is stored in the shared SQLite store and visible
in `kestrion dashboard`.

### CLI

```bash
pip install kestrion

# Scaffold a new project
kestrion init ./my-agent

# Run an agent script
kestrion run agent.py

# Generate Kubernetes manifests
kestrion deploy --target k8s --name my-agent --image registry.example.com/my-agent:latest

# Print the event timeline trace of a run
kestrion trace run_id --store kestrion_runs.db

# Output execution trace as a Mermaid flowchart diagram
kestrion trace run_id --store kestrion_runs.db --mermaid

# Time-travel debug: fork a run at a specific event sequence
kestrion fork run_id --at-seq 5

# Start an interactive terminal chat with your agent (REPL)
kestrion chat agent.py

# Launch the visual web dashboard
kestrion dashboard --port 8000

# Launch the dashboard with live-chat enabled (requires your agent script)
kestrion dashboard agent.py --port 8000
```

`kestrion deploy` generates a complete K8s manifest (Namespace, ConfigMap, Secret stub,
Deployment, PersistentVolumeClaim, Service) and a `Dockerfile`. It never puts a real API key
in the output — only a clearly-labelled placeholder. See the generated files' inline comments
for what to fill in before `kubectl apply`.

## Security & Compliance

Kestrion is designed for enterprise environments where security is a prerequisite, not an afterthought:
- **Secret Management**: API keys and tokens are never stored in plain text or passed in state. They are injected at runtime via the `SecretProvider` protocol.
- **Immutable Audit Log**: Every LLM request, tool execution, and state change is durably recorded in the Event Log.
- **RBAC Approval Gates**: Mutating actions can be gated behind role-based multi-step approval chains.
- **Vulnerability Disclosure**: See [SECURITY.md](SECURITY.md) for reporting policies.

For a deep dive into Kestrion's security posture, see the [Security Architecture](docs/security.md) documentation.

## Governance & Commercial Support

Kestrion is built to be a dependable foundation for enterprise workloads. We have structured the project to ensure longevity, transparent roadmapping, and reliable support.

- **Contribution Model:** See [CONTRIBUTING.md](CONTRIBUTING.md) for how we accept patches and govern major design changes.
- **Vulnerability Disclosure:** See [SECURITY.md](SECURITY.md) for private reporting policies.
- **Public Roadmap:** The living roadmap and active design discussions are available on [GitHub Issues](https://github.com/VinayakDubey07/kestrion/issues) and [roadmap.md](roadmap.md).

**Enterprise Support**
If your team is evaluating Kestrion for a production deployment, we offer dedicated design partnerships and commercial support to ensure your success.
- **Enterprise Support & Inquiries:** vinayak@kestrion.in
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
- **Multi-agent research pipelines and orchestration** — run a team of agents as a DAG where
  some agents run concurrently and others wait for upstream results to arrive before starting.

## Known gaps (honest, not aspirational)

- **MCP is fully two-directional, both sides live-verified.** `kestrion.mcp.client.MCPClient`
  connects to real MCP servers (stdio or streamable-HTTP) and is tested against a real test-fixture
  server, including the full approval-gating flow. `kestrion.mcp.server.serve_agent()` exposes a
  Kestrion `Agent` as a real MCP server, also live-verified end to end — a caller like Claude Code
  can connect and invoke the agent's full reasoning loop (including approval gating) as a single
  MCP tool. See [`examples/ops_demo`](examples/ops_demo) for a worked example using both.
- **Anthropic and OpenAI providers are implemented against documented API shapes but not yet
  smoke-tested against a live API call** — no API key has been used to verify them in practice.
  **Ollama is verified live** — `tests/unit/test_smoke_ollama.py` and
  [`examples/ops_demo`](examples/ops_demo) both run real agents against a real local Ollama server
  and pass. One real, observed limitation worth knowing: small local models can produce plain-text
  output *describing* a tool call and a plausible-sounding result without ever actually emitting a
  real tool-call request — Kestrion has no way to detect this, because there's genuinely no
  `ToolCallRequest` for the engine to act on; the model just wrote a paragraph claiming success.
  This is a model-capability limitation, not something Kestrion's approval gating or event logging
  can catch, since nothing was actually called. Hosted models (Claude, GPT) are far more reliable
  about this in practice, though unverified live as of this writing (see above).
- **Multi-agent handoff is built.** `Agent.as_handoff_target()` transfers an entire conversation to
  another agent, which takes over completely (distinct from sub-agents/delegation, where the
  original agent stays in control).
- **Memory/context compaction is built.** Long-running conversations are automatically summarized
  by the agent when history thresholds (`max_history_turns` or `max_history_tokens`) are exceeded.
- **No real concurrency control across multiple agent runs.** ~~Parallel tool calls *within* one
- **Pipeline orchestration is built.** The `kestrion.scheduler` package provides a `Pipeline` class with a DAG-based runner, bounded `WorkerPool`, and shared token-bucket `RateLimiter` with exponential backoff. See [`examples/pipeline_demo.py`](examples/pipeline_demo.py) and the `### Pipeline / multi-agent orchestration` section.
- **OpenAI/Anthropic APIs.** These providers are fully implemented according to the SDKs but not yet smoke-tested against a live API (only Ollama is live-verified).

## Examples

- [`examples/pipeline_demo.py`](examples/pipeline_demo.py) — a live demo of the `Pipeline`
  scheduler: two researcher agents run concurrently, a synthesizer agent waits for both to finish,
  all results saved to a shared SQLite store and viewable in the dashboard.
- [`examples/kubectl_agent`](examples/kubectl_agent) — the original worked example, demonstrating
  pause-on-approval and resume-after-restart using the raw `Engine`/`Node` primitives directly
  (useful for understanding what `Agent` builds on top of).
- [`examples/rest_api_tool`](examples/rest_api_tool) — patterns for calling REST/SOAP APIs from a
  tool: explicit timeouts, gating a mutating call, reading secrets from the environment, and
  writing your own retry loop.
- [`examples/ops_demo`](examples/ops_demo) — an integration demo exercising parallel tool calls,
  sub-agent delegation, a multi-role approval chain with a timeout, and multi-agent handoff
  together against a real local Ollama model. Run live, this also surfaced a real limitation: a
  small local model can describe a fabricated tool call and result in plain text without ever
  emitting a real tool-call request — see the Known Gaps note above.
- [`tests/unit/test_smoke_ollama.py`](tests/unit/test_smoke_ollama.py) — a live, real smoke test
  against a local Ollama server. Skips automatically if Ollama isn't running.
- [`tests/unit/test_mcp_client.py`](tests/unit/test_mcp_client.py) — a live test against a real
  MCP server (`tests/fixtures/mock_mcp_server.py`), including the approval-gating integration.
- [`examples/mcp_server.py`](examples/mcp_server.py) — a worked example exposing a Kestrion agent
  as an MCP server over stdio.
- [`tests/unit/test_mcp_server.py`](tests/unit/test_mcp_server.py) — a live test of the MCP
  server side (`serve_agent()`), verifying a paused run is correctly surfaced through the MCP
  protocol rather than misreported as an error.
- [`tests/unit/test_cli.py`](tests/unit/test_cli.py) — CLI integration tests for `kestrion init`,
  `kestrion run`, and `kestrion deploy --target k8s`, including the security check that generated
  manifests never contain real-looking API keys.

## Documentation

- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- Concepts: [Event Sourcing](docs/concepts/event-sourcing.md) ·
  [Checkpointing](docs/concepts/checkpointing.md) ·
  [Approval Gates](docs/concepts/approval-gates.md) ·
  [Sub-Agents vs. Handoff](docs/concepts/sub-agents-vs-handoff.md)

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

Next up: Further hardening, OpenTelemetry exporters, and Human-in-the-Loop "Ask for Input". The scheduler and Postgres store are now built — see `src/kestrion/store/postgres_store.py` and `src/kestrion/scheduler/`.
See [roadmap.md](roadmap.md) for the detailed, dated 3-month plan.

## License

Apache 2.0