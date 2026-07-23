# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0, so the public API may still shift between minor versions).

## [0.4.0] - 2026-07-23

### Added
- **Swarm Routing (`SupervisorNode`)**: Dynamically route user intents to specialized agents using LLM classification.
- **Dynamic Tool Discovery (`ToolRegistry`)**: Agents can search and load required tools on the fly during a run.
- **Interactive Playground Builder**: A visual Drag-and-Drop builder in the Web Dashboard (`/playground`) that exports architectures directly to Kestrion Python code.

### Fixed
- Fixed **BUG-004** and **BUG-005**: `Engine.fork()` no longer crashes on list-slicing sequence mismatch. It explicitly queries database sequence boundaries (`events_up_to`), fully supporting concurrent runs and gap sequences (e.g. from deleted dashboard runs).
- Fixed **BUG-011**: Resolved weak MD5 hash vulnerability flags in Bandit security scanners for cache key generation.

## [0.3.0] - 2026-07-18

### Added
- **Enterprise Secret Management (`SecretProvider`)**: Inject credentials safely via `EnvVarSecretProvider` or custom classes without leaking into the durable database or event log.
- **Advanced Context Window Management (`SummarizationNode`)**: Bounded database blob growth via O(1) serialization overhead. `AgentState.history` cleanly truncates older turns and saves a summary marker directly in the engine event loop.
- **Human-in-the-Loop Input (`ask_human`)**: Built-in tool that pauses the run to collect text input from the user. Includes `Agent.provide_input()` and `Engine.provide_input()` APIs.
- **PostgreSQL Checkpoint Store**: Added `PostgresCheckpointStore` for production-grade, distributed state persistence. Enabled by passing a `postgres://` connection string.
- **OpenTelemetry Integration**: Added `OpenTelemetryProvider` to emit nested spans for agent runs, LLM calls, and tool execution.
- **Vision / Multi-modal Support**: Native processing of `TextBlock` and `ImageBlock` types inside agent event loops and provider mappings (Anthropic, OpenAI, Ollama).
- **Time-Travel Debugging (Forking)**: Added the `fork` command to clone agent runs from a specific event sequence for debugging.
- **CLI Interactive & Chat Dashboard**: Built REPL terminal chat (`kestrion chat`), Live Chat UI in the visual dashboard, and interactive trace visualization (Mermaid flowchart diagrams in both CLI and dashboard).
- New `InputRequired` and `InvalidToolInputError` exceptions in `kestrion.core.errors`.
- Interactive demo script: `examples/ask_input_demo.py`.

### Fixed
- `_FunctionTool.call()` now filters out injected internal kwargs (`_state`) before invoking the wrapped function, preventing `TypeError` on tools that don't accept `**kwargs`.
- `ApprovalRequired.kwargs` no longer leaks non-serializable `AgentState` objects into checkpoint storage.
- Relaxed flaky text assertions in `test_live_ollama_agent.py` that depended on exact LLM output.

## [0.2.2] - 2026-07-06

### Added
- **MCP client & server integration**: Full two-way support for the Model Context Protocol. You can now use tools from external MCP servers and expose Kestrion agents as MCP servers.
- **Memory / Context Compaction**: Automatically summarize older conversation turns to stay within context windows.
- **Multi-Agent Handoff**: `Agent.as_handoff_target()` enables full conversation transfer to specialized agents.
- **Sub-agents**: Delegate tasks to child agents.
- **Parallel Tool Calls**: Concurrent execution of tools requested by the model in a single turn.
- **Time-boxed Approvals**: Set deadlines for approvals before the run expires.
- **Web Dashboard**: Run `kestrion dashboard` for a visual debugging experience.
- Refactored `Agent.approve()` to use the Engine API properly instead of hand-rolled workarounds.

### Fixed
- Fixed Anthropic/OpenAI alternating-role violations during memory compaction.
- Fixed premature `RUNNING` status transitions in the dashboard approval workflow.

## [0.1.0] - 2026-06-25

### Added

- Core execution engine: event-sourced state, checkpointing, approval gating
- `@tool` / `Agent` decorator API with automatic JSON-schema generation from
  function signatures
- Three LLM providers behind one protocol: Anthropic, OpenAI, Ollama
- SQLite-backed `CheckpointStore`
- `examples/kubectl_agent` worked example (pause-on-approval, resume-after-restart)
- 35 passing tests across engine, types, and store

## [0.0.1] - 2026-06-25

- Initial PyPI placeholder release