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
started the run — a person clicks "approve" in a UI, hours later, on a different machine.

Kestrion makes this extremely simple with the one-liner `approve()` method:

```python
# Anywhere else, any time later, sharing only the same store database:
final = await agent.approve(result.run_id)
print(final.status)   # RunStatus.COMPLETED
```

Under the hood, `agent.approve()` handles loading the latest checkpoint, validating status, appending a role approval (`Engine.record_approval`), logging the decision, persisting a new checkpoint, and calling `resume()`.

For multi-step approval chains (e.g. `requires_approval=["engineer", "manager"]`), you can pass the specific role and prevent immediate resumption by setting `and_resume=False`:

```python
# Engineer signs off:
await agent.approve(result.run_id, role="engineer", and_resume=False)

# Manager signs off later, letting execution continue:
final = await agent.approve(result.run_id, role="manager")
print(final.status)   # RunStatus.COMPLETED
```

## What just happened, mechanically

Every step above — the LLM call, the decision to call `send_email`, the pause, the approval, the
resume — was recorded as an event in `my_agent.db`. Nothing about `result.status` or
`result.output` is stored directly; it's all *derived* by replaying that log. This is the core idea
behind Kestrion, explained in full in [Event Sourcing](concepts/event-sourcing.md).

## Visualizing runs and approving tools

Kestrion provides a built-in Console to help you inspect and manage runs.

### 1. Terminal Trace Inspector
To see a colorized, step-by-step event timeline of a run directly in your terminal, run:
```bash
kestrion trace <run_id>
```

You can also output the execution trace as a Mermaid flowchart diagram to visualize state transitions, tool invocations, and LLM calls:
```bash
kestrion trace <run_id> --mermaid
```
Copy and paste this output into any Markdown editor or visualizer (like GitHub, VS Code, or mermaid.live) to render the graph.

### 2. Web Dashboard & Approvals
To launch an interactive web dashboard where you can view runs, inspect event payloads/costs, and approve pending gated tools via a visual interface:
```bash
kestrion dashboard --port 8000
```
Open `http://localhost:8000` in your browser. Inside the run details section, select the **Visual Trace** tab to see the Mermaid flowchart layout of your agent's run rendered live and interactively.

### 3. Interactive Chat (Terminal & Web)
If you want to have a live, multi-turn conversation with your agent, you can launch Kestrion in Interactive Mode.

To chat directly in your terminal:
```bash
kestrion chat agent.py
```
*(Your agent will pause and prompt for inline `[y/N]` tool approvals right in the terminal!)*

To enable the Live Chat UI in the web dashboard, pass your script when launching the dashboard:
```bash
kestrion dashboard agent.py --port 8000
```
*(You will now see a Chat Input bar at the bottom of the Chat History tab in the dashboard.)*

## Asking the user for input

Approval gates are binary — the run pauses for a yes/no decision. Sometimes an agent needs a
specific piece of information from a human (a 2FA code, a preferred name, a clarification). The
built-in `ask_human` tool handles this:

```python
from kestrion.agent.tools import ask_human

agent = Agent(
    provider=OllamaProvider(model="llama3.2"),
    tools=[ask_human],
    store="sqlite:///my_agent.db",
)
result = await agent.run("Write a poem about my favorite color")
# result.status == RunStatus.WAITING_ON_HUMAN
# result.state.scratch["_pending_input"]["question"] == "What is your favorite color?"
```

The run suspends cleanly — no thread blocked, everything checkpointed. When the human answers:

```python
final = await agent.provide_input(result.run_id, text="Blue")
print(final.output)  # a poem about blue
```

You can also raise `InputRequired` directly in your own tools for the same pause-and-resume
behavior without using the built-in `ask_human`.

## Scaling to Production (PostgreSQL)

SQLite is great for local development, but enterprise agents need high availability and horizontal scaling. Kestrion supports PostgreSQL out of the box for multi-worker, concurrent agent deployments.

To switch to PostgreSQL, install the `postgres` extra (which includes `asyncpg`):

```bash
pip install "kestrion[postgres]"
```

Then simply pass a Postgres URL as your store:

```python
agent = Agent(
    provider=OllamaProvider(model="llama3.2"),
    tools=[my_tool],
    # Kestrion will automatically initialize the PostgresCheckpointStore
    # and connection pool using this URL.
    store="postgres://user:password@localhost:5432/kestrion_db",
)
```

No code changes are required. Crash recovery, approvals, and immutable event logs all operate exactly the same way, just backed by a production-ready database.

## OpenTelemetry (OTel) Integration

Kestrion provides first-class support for OpenTelemetry to monitor agent runs, LLM calls, and tool executions. This enables seamless tracing of an agent's reasoning loop across your entire enterprise observability stack (Datadog, Jaeger, Splunk, etc).

To enable OpenTelemetry, install the `otel` extra:

```bash
pip install "kestrion[otel]"
```

Then configure the `OpenTelemetryProvider` and attach it to your engine/agent:

```python
from kestrion.telemetry.otel import OpenTelemetryProvider

# Initialize the OTel provider
telemetry = OpenTelemetryProvider(service_name="kestrion-ops-agent")

agent = Agent(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    tools=[my_tool],
    store="sqlite:///my_agent.db",
    telemetry=telemetry, # Attach telemetry provider
)
```

As the agent runs, Kestrion automatically emits nested spans representing the overall Run, individual Steps, Model Inferences, and Tool Calls.

## Vision and Multi-modal Support

Kestrion treats image inputs natively via strictly typed content blocks (`TextBlock` and `ImageBlock`). You can seamlessly pass a list of these blocks directly to the agent's `run()` method instead of a standard string. Kestrion automatically maps them to the correct wire format for your chosen provider (Anthropic, OpenAI, or Ollama).

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