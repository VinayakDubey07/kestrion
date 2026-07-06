# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0, so the public API may still shift between minor versions).

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