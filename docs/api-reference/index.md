# API Reference

This page is hand-curated so it's useful immediately, even before running the generator below.
Each linked page (once generated) pulls its content directly from the real source docstrings —
every class and method here has the same documentation you'd see reading the source code.

## Generating the full reference

This reference is built with [mkdocs](https://www.mkdocs.org/) and
[mkdocstrings](https://mkdocstrings.github.io/). Not yet wired into CI — run locally:

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs serve   # http://127.0.0.1:8000, live-reloads on docstring changes
# or: mkdocs build   # static site in site/
```

This requires `mkdocs.yml` at the repo root (see below) and reads docstrings directly from
`src/kestrion/` — there is no separate reference content to keep in sync by hand; if the
docstrings are accurate, the generated reference is accurate.

## Core (`kestrion.core`)

The execution engine and its data contracts — see [Architecture](../architecture.md) for the full
design rationale, this is the API surface itself.

- `kestrion.core.types.Event` — the immutable unit of the event log
- `kestrion.core.types.AgentState` — derived state; `to_dict()`/`from_dict()` for checkpoint
  serialization
- `kestrion.core.types.Checkpoint` — a point-in-time `AgentState` snapshot
- `kestrion.core.types.CheckpointStore` — the storage Protocol (`SQLiteCheckpointStore` is the only
  implementation today)
- `kestrion.core.types.ToolSpec` — `requires_approval` (`bool | str | list[str]`),
  `approval_timeout_seconds`, `required_roles()`
- `kestrion.core.types.Tool` / `ToolResult` — the tool contract every `@tool` function and
  `MCPTool` satisfies
- `kestrion.core.engine.Engine` — `start()`, `resume()`, `call_tool()`, `check_approval()`,
  `record_approval()` (static)
- `kestrion.core.engine.ApprovalRequired` / `RunExpiredError` — control-flow exceptions raised by
  the engine, not meant to be caught by ordinary application code except at the top level

## Agent (`kestrion.agent`)

The ergonomic, user-facing layer built on top of `core`.

- `kestrion.agent.decorators.tool` — turns a function into a `Tool` via signature introspection
- `kestrion.agent.agent.Agent` — `run()`, `run_with_history()`, `resume()`, `as_tool()`,
  `as_handoff_target()`
- `kestrion.agent.agent.SubAgentTool` — delegation (parent stays in control); returned by
  `Agent.as_tool()`
- `kestrion.agent.agent.HandoffTool` / `HandoffCompleted` — full conversation transfer; returned by
  `Agent.as_handoff_target()`

## LLM providers (`kestrion.llm`)

- `kestrion.llm.base.LLMProvider` — the Protocol every provider satisfies
- `kestrion.llm.base.Message`, `LLMResponse`, `ToolCallRequest` — normalized types shared across
  providers
- `kestrion.llm.anthropic_provider.AnthropicProvider`
- `kestrion.llm.openai_provider.OpenAIProvider`
- `kestrion.llm.ollama_provider.OllamaProvider` — the only provider live-verified end to end; see
  the README's Known Gaps for what that means for the other two

## MCP (`kestrion.mcp`)

- `kestrion.mcp.client.MCPClient` — `stdio()`, `http()`, `list_tools(requires_approval=[...])`
- `kestrion.mcp.client.MCPTool` — wraps one MCP-advertised tool as a Kestrion `Tool`

## Storage (`kestrion.store`)

- `kestrion.store.sqlite_store.SQLiteCheckpointStore` — the reference `CheckpointStore`
  implementation