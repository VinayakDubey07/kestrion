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
- **[Sub-Agents vs. Handoff](concepts/sub-agents-vs-handoff.md)** — delegation and transfer patterns for multi-agent workflows.

## Reference

- **[Architecture](architecture.md)** — the full module-by-module account of how Kestrion is
  built, including the real bugs found during development and the design rules they led to. The
  most detailed single document in this set.
- **[Security & Compliance](security.md)** — enterprise security posture, secrets management, immutable audit logs, and RBAC approvals.
- **[API Reference](api-reference/index.md)** — generated from docstrings; the exact signatures
  and behavior of every public class and function.

## Examples

Code, not just prose:

- [`examples/kubectl_agent/main.py`](../examples/kubectl_agent/main.py) — pause-on-approval and crash
  recovery using the raw `Engine`/`Node` primitives directly.
- [`examples/rest_api_tool`](../examples/rest_api_tool) — calling REST/SOAP APIs from a tool:
  timeouts, secrets, retries.
- [`examples/mcp_server.py`](../examples/mcp_server.py) — a worked example exposing a Kestrion
  agent as an MCP server over stdio.
- [`examples/ops_demo`](../examples/ops_demo) — an integration demo exercising nineteen agentic
  features together (parallel tool calls, sub-agents, approval chains, timeouts, handoff, memory compaction, human input, secrets, custom context windows, OTel tracing, Code Sandbox, Structured Outputs, RAG, and Data Loss Prevention)
  against a real local model.
- [`examples/rag_demo.py`](../examples/rag_demo.py) — an end-to-end example of initializing an in-memory vector store, chunking internal documentation, and using the `RAGToolkit` with an agent.
- [`examples/ask_input_demo.py`](../examples/ask_input_demo.py) — interactive demo of the
  human-in-the-loop `ask_human` tool: the agent pauses to ask the user a question, then resumes
  with the answer.
- [`examples/pipeline_demo.py`](../examples/pipeline_demo.py) — concurrent execution of multiple agents coordinated via a DAG scheduler with token-bucket rate limits.

## What's built vs. planned

This documentation describes what exists today. For an honest, dated account of what's verified,
what's designed-but-unbuilt, and what's coming next, see the main
[README's Known Gaps section](../README.md#known-gaps-honest-not-aspirational) and
[roadmap.md](../roadmap.md).