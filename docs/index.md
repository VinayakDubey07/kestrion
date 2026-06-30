# Kestrion Documentation

Kestrion is a durable-execution-first framework for building production AI agents. The core idea:
state is never mutated directly — it's derived by folding an immutable log of events. Everything
else in this documentation follows from that one decision.

## Start here

- **[Getting Started](getting-started.md)** — install, build your first tool and agent, see the
  pause-on-approval behavior that makes Kestrion different, in about ten minutes.

## Concepts

The "why" behind the design, not just the "how":

- **[Event Sourcing](concepts/event-sourcing.md)** — why `AgentState` is never mutated directly,
  and what that buys you for free.
- **[Checkpointing](concepts/checkpointing.md)** — how a paused or crashed run resumes from a
  completely independent process, using nothing but a store.
- **[Approval Gates](concepts/approval-gates.md)** — the base approval mechanism, multi-step
  chains, time-boxed approvals, and the safety guarantee around parallel tool calls.

## Reference

- **[Architecture](architecture.md)** — the full module-by-module account of how Kestrion is
  built, including the real bugs found during development and the design rules they led to. The
  most detailed single document in this set.
- **[API Reference](api-reference/index.md)** — generated from docstrings; the exact signatures
  and behavior of every public class and function.

## Examples

Code, not just prose:

- [`examples/kubectl_agent.py`](../examples/kubectl_agent.py) — pause-on-approval and crash
  recovery using the raw `Engine`/`Node` primitives directly.
- [`examples/rest_api_tool`](../examples/rest_api_tool) — calling REST/SOAP APIs from a tool:
  timeouts, secrets, retries.
- [`examples/ops_demo`](../examples/ops_demo) — an integration demo exercising five agentic
  features together (parallel tool calls, sub-agents, approval chains, timeouts, handoff) against
  a real local model.

## What's built vs. planned

This documentation describes what exists today. For an honest, dated account of what's verified,
what's designed-but-unbuilt, and what's coming next, see the main
[README's Known Gaps section](../README.md#known-gaps-honest-not-aspirational) and
[ROADMAP.md](../ROADMAP.md).