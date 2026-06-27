# Getting Started

This walks through installing Kestrion, building your first agent, and seeing the pause-on-approval
behavior that makes Kestrion different from other agent frameworks.

## Install

```bash
pip install kestrion[anthropic]
```

Swap `anthropic` for `openai`, `ollama`, or `all` depending on which model provider you want. Each
is an optional install — you never need a package you're not using.

## Your first tool

A tool is just a Python function with a `@tool` decorator. The function's type hints become the
JSON schema the model sees; the docstring becomes the description:

```python
from kestrion.agent.decorators import tool

@tool
def get_weather(city: str) -> dict:
    """Look up the current weather for a city."""
    return {"city": city, "condition": "sunny", "temp_c": 28}
```

That's it — no schema to write by hand, no registration step.

## Your first agent

`Agent` wraps a model provider and a list of tools into something you can run with a single
prompt:

```python
import asyncio
from kestrion.agent.agent import Agent
from kestrion.llm.anthropic_provider import AnthropicProvider

async def main():
    agent = Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        tools=[get_weather],
        store="sqlite:///my_agent.db",
    )
    result = await agent.run("What's the weather in Bangalore?")
    print(result.status)   # RunStatus.COMPLETED
    print(result.output)   # the model's final answer

asyncio.run(main())
```

`store` is where Kestrion persists the run's event log and checkpoints — a plain SQLite file.
You'll see why that matters in the next section.

## Gating a tool that has side effects

Most interesting tools eventually need to *do* something — write to a database, call an API that
changes state, apply a configuration. For those, add `requires_approval=True`:

```python
@tool(requires_approval=True)
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email. Requires approval before sending."""
    ...
    return {"sent": True}
```

Now run an agent that might call it:

```python
agent = Agent(provider=AnthropicProvider(model="claude-sonnet-4-6"), tools=[send_email], store="sqlite:///my_agent.db")
result = await agent.run("Email the team that the deploy is done")
print(result.status)   # RunStatus.WAITING_ON_HUMAN
```

The run stops itself the moment the model decides to call `send_email`. Nothing was sent. No
thread is blocked waiting — the process could exit right now and nothing would be lost, because
everything that happened so far is already in `my_agent.db`.

## Approving and resuming

In a real application, the approval step usually happens in a different process than the one that
started the run — a person clicks "approve" in a UI, hours later, on a different machine. Kestrion
models this directly: approving is just recording that fact in the run's state and persisting it,
then calling `resume()`.

```python
from datetime import datetime, timezone
from kestrion.core.engine import Engine
from kestrion.core.types import Checkpoint, new_id

# Record the approval (this mutates result.state in place — see the
# "Approval Gates" concept doc for why record_approval exists instead
# of writing to scratch by hand).
Engine.record_approval(result.state, "send_email", role="__any__")

# Persist it as a checkpoint so resume() (possibly in a different
# process) can see it.
await agent._store.save(Checkpoint(
    checkpoint_id=new_id("ckpt"),
    run_id=result.run_id,
    state=result.state,
    created_at=datetime.now(timezone.utc),
    event_seq=result.state.last_event_seq,
))

final = await agent.resume(result.run_id)
print(final.status)   # RunStatus.COMPLETED
```

This is more manual than it will eventually be — `Agent.approve()` is a planned ergonomic
shortcut for exactly this, not yet built (see the project README's "Known gaps"). What's shown
above is the real mechanism underneath it, and it's already fully functional.

## What just happened, mechanically

Every step above — the LLM call, the decision to call `send_email`, the pause, the approval, the
resume — was recorded as an event in `my_agent.db`. Nothing about `result.status` or
`result.output` is stored directly; it's all *derived* by replaying that log. This is the core idea
behind Kestrion, explained in full in [Event Sourcing](concepts/event-sourcing.md).

## Where to go next

- [Event Sourcing](concepts/event-sourcing.md) — why state is never mutated directly
- [Checkpointing](concepts/checkpointing.md) — how crash recovery actually works
- [Approval Gates](concepts/approval-gates.md) — multi-step chains, timeouts, and the safety
  guarantees around parallel tool calls
- [`examples/kubectl_agent`](../examples/kubectl_agent) — a complete worked example using the raw
  `Engine`/`Node` primitives directly, for when `Agent`'s single-loop model doesn't fit your
  workflow
- [`examples/rest_api_tool`](../examples/rest_api_tool) — patterns for calling REST/SOAP APIs from
  a tool: timeouts, secrets, retries