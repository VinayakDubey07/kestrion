# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0, so the public API may still shift between minor versions).

## [Unreleased]

- MCP client/server integration
- Scheduler for rate-limited concurrent runs
- CLI (`kestrion deploy --target k8s`)
- Postgres-backed `CheckpointStore`
- Docs site

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