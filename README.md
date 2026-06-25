# Kestrion

A durable-execution-first framework for building production AI agents.

Status: pre-alpha (`0.1.0`). Core engine, decorator API, and three LLM providers are built and
tested (35 passing tests). MCP integration, scheduler, CLI, and Postgres support are designed but
not yet implemented — see [Roadmap](#roadmap) below.

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
pip install kestrion[anthropic]   # or [openai], [ollama], or [all]
```

Each LLM provider is an optional extra. If you only use Ollama, you never need the `anthropic` or
`openai` packages installed.

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
state.scratch["_approved_tools"] = {"apply_manifest": True}
# (persist that as a checkpoint — see examples/kubectl_agent for the full pattern;
#  Agent.approve() is not yet a polished one-liner, see Known Gaps below)

result = await agent.resume(run_id)
print(result.status)  # "completed"
```

## What you can build with this today

- Tool-calling agents where some actions are safe to auto-run and others need a human in the loop
  first — infrastructure agents, ops bots, anything touching a database or cluster.
- Agents that need to survive a crash or restart mid-task. `agent.resume(run_id)` works from a
  totally different process than the one that started the run.
- Long-running approval workflows — start a run, let it sit paused for hours, approve it from a
  Slack bot or web UI later, resume it from anywhere with access to the same store.
- Multi-turn tool use — the agent loop keeps calling tools and feeding results back to the model
  until it produces a final answer with no more tool calls.

## Known gaps (honest, not aspirational)

- **No MCP integration yet.** Every tool today is a hand-written Python function via `@tool`.
  Connecting to external MCP servers is the next phase of work.
- **No real concurrency control.** Running many agents at once against a shared rate limit isn't
  implemented.
- **No CLI or deploy story.** `kestrion deploy --target k8s` doesn't exist yet — you'd containerize
  and deploy this yourself today.
- **`Agent.approve()` is a stub.** Approving a paused run currently means manually setting
  `state.scratch["_approved_tools"]` and saving a checkpoint by hand (see
  `examples/kubectl_agent`), not a polished one-line call.
- **SQLite only.** A `CheckpointStore` Protocol exists so Postgres can be added without touching
  the engine, but that implementation doesn't exist yet.

## Examples

- [`examples/kubectl_agent`](examples/kubectl_agent) — the original worked example, demonstrating
  pause-on-approval and resume-after-restart using the raw `Engine`/`Node` primitives directly
  (useful for understanding what `Agent` builds on top of).

## Development

```bash
git clone https://github.com/<your-username>/kestrion.git
cd kestrion
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Roadmap

See the full phased build plan in the repo for the path to `1.0.0`. Short version: MCP client/server
integration, a scheduler for safe concurrent execution, a CLI with Kubernetes deploy support,
Postgres-backed storage, and a docs site are next.

## License

Apache 2.0
