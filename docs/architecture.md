# Kestrion Architecture

This document describes how Kestrion is actually built — the core abstractions, how they compose,
and the specific design decisions behind the durability guarantees the project is built around.
It reflects the code as it exists at `v0.3.0`, not the aspirational end state. Where something is
designed but not implemented, that's called out explicitly rather than glossed over.

## 1. The central idea: state is derived, never mutated

Everything else in this document follows from one decision: **`AgentState` is never the source of
truth — it's a cache, derived by folding an immutable log of `Event` objects.**

```
Event 1 ──┐
Event 2 ──┼──► fold ──► AgentState (current)
Event 3 ──┤
Event N ──┘
```

Concretely, this means:

- Every meaningful thing that happens during a run — an LLM call completing, a tool being invoked,
  a transition between graph nodes — is recorded as an `Event` *before* anything else happens.
- `AgentState` (the thing your code reads `scratch`, `history`, `total_tokens` from) is rebuilt by
  replaying events through a single function, `Engine._fold`.
- Because state is derivable from the log, **any process holding the log can reconstruct the exact
  state of a run**, regardless of whether it's the process that started the run.

This is the mechanism underneath the project's central claim — durable execution by default, not
as an opt-in feature.

## 2. Module map

```
kestrion/
├── core/
│   ├── types.py      — data contracts: Event, AgentState, Checkpoint, ToolSpec, Node, NodeResult
│   ├── engine.py      — the execution loop, approval gating, event sourcing plumbing
│   ├── errors.py      — control-flow exceptions: ApprovalRequired, InputRequired, RunExpiredError
│   ├── secrets.py     — secret injection logic (SecretProvider)
│   └── nodes.py       — SummarizationNode for context compaction
├── store/
│   ├── sqlite_store.py — SQLite CheckpointStore implementation
│   ├── postgres_store.py — PostgreSQL CheckpointStore implementation
│   └── memory_store.py — In-memory CheckpointStore implementation
├── agent/
│   ├── agent.py        — Agent class, the ergonomic user-facing API
│   ├── decorators.py   — @tool, turning Python functions into Tool objects
│   └── tools.py        — built-in tools (ask_human)
├── llm/
│   ├── base.py                — LLMProvider Protocol, ContentBlock, Message/LLMResponse types
│   ├── anthropic_provider.py  — Anthropic Claude implementation
│   ├── openai_provider.py     — OpenAI implementation
│   └── ollama_provider.py     — local-model implementation via Ollama's HTTP API
├── scheduler/
│   ├── pipeline.py     — DAG-based async pipeline scheduler
│   ├── worker_pool.py  — task worker pool
│   └── rate_limiter.py — rate-limiter logic for API endpoints
├── telemetry/
│   └── otel.py        — OpenTelemetry integration
├── cli/
│   ├── main.py        — CLI parser and command handlers
│   ├── deploy.py      — Kubernetes manifest generator
│   └── dashboard.py   — HTTP backend for the visual dashboard
└── mcp/
    ├── client.py      — Model Context Protocol client
    └── server.py      — Model Context Protocol server UIs
```

Two architectural rules hold across every module boundary in this list:

1. **Protocols, not inheritance.** `CheckpointStore` and `LLMProvider` are `typing.Protocol`
   classes. Concrete implementations (`SQLiteCheckpointStore`, `AnthropicProvider`, etc.) satisfy
   them structurally — there's no base class to subclass, no import-time coupling between `core/`
   and any specific storage backend or LLM vendor.
2. **`core/` depends on nothing above it.** `engine.py` and `types.py` have zero imports from
   `store/`, `agent/`, or `llm/`. Dependencies only flow downward: `agent/` depends on `core/` and
   `llm/`; nothing in `core/` knows `Agent` or any LLM provider exists.

## 3. Core data types (`core/types.py`)

### Event

```python
@dataclass(frozen=True)
class Event:
    event_id: str
    run_id: str
    type: EventType
    timestamp: datetime
    payload: dict[str, Any]
    node: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float
```

Immutable by construction (`frozen=True`). An `Event` is a fact about something that already
happened — nothing in the codebase ever mutates an `Event` after creation, only creates new ones.

`EventType` is a closed enum: `RUN_STARTED`, `MESSAGE_RECEIVED`, `LLM_CALL_STARTED`,
`LLM_CALL_COMPLETED`, `TOOL_CALL_STARTED`, `TOOL_CALL_COMPLETED`, `TOOL_CALL_FAILED`,
`STATE_TRANSITION`, `CHECKPOINT_SAVED`, `RUN_COMPLETED`, `RUN_FAILED`, `RUN_EXPIRED`,
`HUMAN_INTERVENTION`, `CONTEXT_COMPACTED`.

Token/cost fields live directly on `Event`, not bolted on as separate logging — this is what makes
cost tracking "free": any code path that records an event for any reason gets cost accounting for
free if it populates those fields.

### AgentState

```python
@dataclass
class AgentState:
    run_id: str
    status: RunStatus              # PENDING | RUNNING | WAITING_ON_HUMAN | COMPLETED | FAILED | EXPIRED
    current_node: str | None
    scratch: dict[str, Any]        # arbitrary node-writable working memory
    history: list[dict[str, Any]]  # summarized tool/LLM history, NOT the full event log
    total_tokens: int
    total_cost_usd: float
    last_event_seq: int            # position in the event log this state reflects
```

`AgentState` is mutable — but the only code permitted to mutate it is `Engine._fold` (for fields
derived from events) and the engine's own driving loop (for `status`/`current_node` transitions).
Node implementations never mutate `AgentState` fields directly except by writing into `scratch`
via the `state_updates` they return.

`AgentState.to_dict()` / `from_dict()` provide explicit, version-stable serialization for
persistence — see §6 for why this replaced an earlier `pickle`-based approach.

### Checkpoint

```python
@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    run_id: str
    state: AgentState
    created_at: datetime
    event_seq: int
```

A `Checkpoint` is a point-in-time snapshot of `AgentState`, tagged with the event-log position it
corresponds to. Checkpoints are what `resume()` actually loads — they exist so that resuming a run
doesn't require replaying the *entire* event history from event zero, only the events since the
last checkpoint (see §5).

### CheckpointStore (Protocol)

```python
class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None: ...
    async def latest(self, run_id: str) -> Checkpoint | None: ...
    async def append_event(self, event: Event) -> int: ...
    async def events_since(self, run_id: str, seq: int) -> list[Event]: ...
```

This is the seam where storage backends plug in. Kestrion provides `SQLiteCheckpointStore` and `PostgresCheckpointStore` out of the box. Both fully satisfy this Protocol.

### ToolSpec, Tool, ToolResult

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON schema
    requires_approval: bool

class Tool(ABC):
    spec: ToolSpec
    async def call(self, **kwargs) -> ToolResult: ...
```

`requires_approval` is the single field that drives the entire human-in-the-loop mechanism — see
§4. `parameters` is a JSON schema, deliberately the same shape MCP tools use, so that an MCP-backed
tool (which is fully integrated) and a `@tool`-decorated Python function look structurally identical to
the engine.

### Node, NodeResult

```python
class Node(Protocol):
    name: str
    async def run(self, state: AgentState) -> "NodeResult": ...

@dataclass
class NodeResult:
    next_node: str | None          # None = run is complete
    state_updates: dict[str, Any]  # merged into AgentState.scratch
    events: list[Event]
```

A `Node` is one step of an execution graph. Nodes are deliberately storage-ignorant: they receive
`AgentState`, return a `NodeResult`, and never touch the `CheckpointStore` directly (except via
`Engine.call_tool` / `Engine.record_event`, which exist precisely so nodes don't need direct store
access). This keeps node logic unit-testable without a database.

## 4. The execution engine (`core/engine.py`)

### The drive loop

`Engine._drive` is the heart of the system:

```
while state.status == RUNNING and state.current_node is not None:
    node = nodes[state.current_node]
    try:
        result = await node.run(state)
    except ApprovalRequired:
        → pause: status = WAITING_ON_HUMAN, checkpoint, return
    except Exception:
        → status = FAILED, checkpoint, re-raise

    fold result.events into state
    state.scratch.update(result.state_updates)

    if result.next_node is None:
        status = COMPLETED
    else:
        current_node = result.next_node

    checkpoint(state)   # after EVERY transition, not periodically
```

Two things about this loop are deliberate, not incidental:

- **Checkpointing happens after every single transition**, not on a timer or a sampled interval.
  The tradeoff is more write volume in exchange for the simplest possible recovery story: there's
  never a window where a meaningful state change exists only in memory.
- **`ApprovalRequired` is the only exception with a special path.** Everything else becomes a hard
  `FAILED` state. This is intentional — an unexpected exception in your tool/node code is a bug to
  surface, not something to silently swallow and retry.

### Approval gating

```python
async def call_tool(self, state, tool_name, **kwargs) -> ToolResult:
    tool = self.tools[tool_name]
    if tool.spec.requires_approval and not state.scratch.get("_approved_tools", {}).get(tool_name):
        raise ApprovalRequired(tool_name, kwargs)
    ...
```

This is the actual mechanism behind "the engine enforces approval gates, not the node author."
Every tool call — from any node, from the `Agent` loop, from anywhere — funnels through this one
method. There is no other path to invoke a tool's `.call()` that bypasses this check. A node author
cannot accidentally skip the gate, because there's no API surface that would let them.

When `ApprovalRequired` is raised, `_drive`'s `except` clause catches it *outside* the node's own
code — the node itself never has a chance to catch and discard it. The engine:
1. Sets `status = WAITING_ON_HUMAN`
2. Records exactly which tool/arguments are pending in `scratch["_pending_approval"]`
3. Records a `HUMAN_INTERVENTION` event
4. Checkpoints
5. Returns control to the caller — no thread is blocked waiting

### Event sourcing plumbing

Three methods, each with a distinct durability contract:

- **`_emit`** — private. Used internally by `call_tool` and the drive loop for engine-owned events
  (`TOOL_CALL_STARTED`, `RUN_COMPLETED`, etc.). Appends to the store and advances
  `state.last_event_seq`, but does not fold.
- **`record_event`** — public. The escape hatch for node implementations that need to durably
  record something that isn't a tool call. Appends *and* folds immediately. This exists
  specifically because of a bug found during development (see §7) — `Agent`'s LLM loop uses this to
  record `LLM_CALL_COMPLETED` events the instant they happen, so token/cost data survives even if
  the very next line raises `ApprovalRequired`.
- **`_fold`** — private, and the *only* place `AgentState` fields are derived from events. Centralizing
  this is what makes "rebuild state by replaying the log" an actual guarantee rather than an
  aspiration — there's exactly one function to audit if you want to know how any field is computed.

## 5. How resume actually works

This is worth tracing in full, since it's the project's core differentiating claim.

```python
async def resume(self, run_id: str) -> AgentState:
    checkpoint = await self.store.latest(run_id)
    if checkpoint is None:
        raise ValueError(...)

    state = checkpoint.state
    newer_events = await self.store.events_since(run_id, checkpoint.event_seq)
    for evt in newer_events:
        self._fold(state, evt)

    if state.status == WAITING_ON_HUMAN:
        state.status = RUNNING

    return await self._drive(state)
```

Step by step:

1. **Load the latest checkpoint** for this `run_id` from the store. This is a full `AgentState`
   snapshot as of some point in the event log.
2. **Replay any events newer than that checkpoint.** This handles the edge case where an event was
   appended after the checkpoint was saved but the process died before a *new* checkpoint captured
   it — without this step, that event's effect on state would be lost.
3. **Flip `WAITING_ON_HUMAN` back to `RUNNING`.** Resuming implies the human-in-the-loop condition
   has been satisfied (the caller is expected to have set `scratch["_approved_tools"]` before
   calling `resume`).
4. **Re-enter `_drive`.** From here, execution proceeds exactly as it would have if the process had
   never stopped — same loop, same checkpointing behavior, same approval gating for any *other*
   gated tool calls later in the graph.

Crucially: **none of this requires the resuming `Engine` to be the same object, or even the same
process, that called `start()`.** `Engine` holds no run-specific state on `self` — `nodes`, `tools`,
and `store` are configured once at construction and never change per-run. This was verified
directly: `tests/unit/test_engine.py::test_resume_from_independent_engine_after_approval`
constructs a second, independent `Engine` instance sharing nothing with the first except the
`CheckpointStore`, and shows it correctly completing a run the first instance paused.

## 6. The `Agent` / `@tool` layer

### Why this layer exists

The raw `Engine` + `Node` API requires hand-writing graph nodes and explicitly calling
`engine.call_tool`. `Agent` exists to remove that — it's a thin layer that auto-generates a single
implicit node implementing a standard "LLM decides what to do, possibly call tools, repeat until
done" loop, so most users never touch `Node`/`NodeResult` directly.

### `@tool`: function signatures to JSON schema

`decorators.py` introspects a function's signature via `inspect.signature` and converts type
annotations to JSON schema fragments:

| Python annotation | JSON schema |
|---|---|
| `str`, `int`, `float`, `bool` | `{"type": "string"/"integer"/"number"/"boolean"}` |
| `list[X]` | `{"type": "array", "items": <schema for X>}` |
| `dict` | `{"type": "object"}` |
| `Optional[X]` / `X \| None` | unwrapped to X's schema (JSON schema has no first-class `None`) |
| anything else (Pydantic models, custom classes) | raises `NotImplementedError` explicitly |

The last row is a deliberate design choice: rather than silently producing a wrong or
under-specified schema for an annotation it doesn't understand, the decorator fails loudly at
decoration time. A wrong schema reaching an LLM is a worse failure mode than a clear error at
import time.

The function's docstring becomes the `ToolSpec.description` the model sees — the only information
the model has to decide whether and how to call the tool.

### The `Agent` loop

```
messages = rebuild from state.scratch["_messages"]   # NOT held in memory across resumes

loop:
    response = provider.complete(messages, tools, system)
    record_event(LLM_CALL_COMPLETED) immediately      # durability, see below
    messages.append(assistant message)

    if no tool_calls:
        persist messages + final_output to scratch
        return (run complete)

    for each tool_call:
        result = engine.call_tool(...)   # ApprovalRequired propagates naturally
        messages.append(tool result message)

    persist in-progress messages to scratch
    # loop continues
```

Two non-obvious design decisions here:

1. **The entire multi-turn tool-calling loop lives inside one `Node.run()` call**, not spread
   across multiple graph nodes. This means an entire conversation turn — including several tool
   calls — is checkpointed as a single engine "step," not after every individual tool call within
   it. The tradeoff: simpler mental model and fewer checkpoint writes, at the cost of slightly
   coarser resume granularity if the process dies *mid-turn* before any gated tool is hit (you'd
   replay that turn rather than resume from its middle). This doesn't affect the approval-gating
   guarantee, since `ApprovalRequired` is a real exception that still propagates out of the loop
   and is still caught by `Engine._drive`, not swallowed internally.

2. **`record_event` is called immediately after every LLM response, not batched into the
   eventual `NodeResult`.** This was a real bug found and fixed during development (see §7) — if it
   were batched, an `ApprovalRequired` raised partway through the loop would mean `node.run()` never
   returns a `NodeResult` at all, silently dropping the token/cost data for every LLM call made
   earlier in that same turn.

### Message serialization

`agent/agent.py` defines `_message_to_dict` / `_message_from_dict` rather than relying on
`dataclasses.asdict`'s default behavior naively, or `Message.__dict__`. This matters because
`Message.tool_calls` is a list of nested `ToolCallRequest` dataclass instances — a shallow
`__dict__` call leaves those un-converted, which breaks JSON serialization the moment the engine
tries to checkpoint `state.scratch`. `dataclasses.asdict()` recurses correctly; this was also a
real bug caught during development (see §7).

## 7. Design decisions, and the bugs that shaped them

This section exists because several real defects were found and fixed during development, each of
which led directly to a design rule documented above. Listing them here is more useful than
pretending the design was correct on the first attempt.

**Checkpoint serialization: `pickle` → explicit `to_dict`/`from_dict`.**
The original `SQLiteCheckpointStore` used `pickle.dumps(checkpoint.state)`. This was replaced with
explicit JSON serialization via `AgentState.to_dict()`/`from_dict()` because `pickle` ties the
on-disk checkpoint format to the exact Python class definition and interpreter version at the time
of writing — a liability the moment anyone depends on reading an old checkpoint after the code
changes. The JSON-based `save()` now also fails loudly (`ValueError`, not a silent fallback) if
`scratch` ever contains something non-JSON-serializable, rather than letting `pickle` silently
paper over the problem.

**`Message.__dict__` does not recursively serialize nested dataclasses.**
Found while building the `Agent` loop: a `Message` with `tool_calls` set produced a dict whose
`tool_calls` value was still a list of `ToolCallRequest` *objects*, not dicts — which crashed
checkpointing the first time a tool-calling turn tried to persist. Fixed by using
`dataclasses.asdict()` via the `_message_to_dict`/`_message_from_dict` helpers described in §6.

**Token/cost data was silently lost on approval pauses.**
The first version of the `Agent` loop batched `LLM_CALL_COMPLETED` events into the `NodeResult` it
would eventually return. But when a turn's tool call triggers `ApprovalRequired`, `node.run()` exits
via exception — there is no `NodeResult`, ever, for that turn. Any LLM call made earlier in the same
turn (the one that decided to call the gated tool) would have its token/cost data silently
dropped. This is why `Engine.record_event` exists as a public method: it lets `agent.py` durably
record each LLM call's event the instant it happens, rather than batching and risking loss. There's
a dedicated regression test for this:
`test_agent_records_llm_tokens_even_when_run_pauses_on_approval`.

**`OllamaProvider` could crash the whole run on malformed tool-call JSON.**
Smaller local models are less reliable than hosted models at producing well-formed JSON in tool
call arguments. The original implementation called `json.loads()` on the raw arguments string with
no error handling — a malformed response would raise an uncaught `JSONDecodeError` and crash the
entire agent run. Fixed to catch the parse failure and surface it as a visible note in the
response text with empty arguments, consistent with how `Engine.call_tool` already treats tool
failures elsewhere (recoverable, not fatal).

## 8. What's deliberately NOT in this document

This document describes the core execution engine and agent layer. It does not cover (because
they're either separate concerns or not yet built):

- **MCP client/server** — both now built and live-verified (`mcp/client.py`, `mcp/server.py`),
  but at a higher layer than this document covers. The client wraps MCP-sourced tools as Kestrion
  `Tool` objects; the server exposes a full `Agent` loop as one MCP tool via `serve_agent()`.
  See the README's MCP sections for usage.
- **CLI** — `kestrion init`, `kestrion run`, `kestrion deploy --target k8s`, and `kestrion fork` are now built
  (`cli/main.py`, `cli/deploy.py`). Not covered here because the CLI is an entry-point layer on
  top of the engine/agent, not a change to how either works.
- **The scheduler / concurrent-execution layer** (`scheduler/`) — built! The `Pipeline` orchestrates DAG-based multi-agent execution, rate-limited via a token-bucket `RateLimiter` and bounded `WorkerPool`.
- **Postgres-backed storage** (`store/postgres_store.py`) — built! The async `PostgresCheckpointStore` implements the `CheckpointStore` protocol for production-grade, distributed state persistence.
- **A `Trace` viewer** for the event log — built! Developers can inspect events via the terminal timeline (`kestrion trace <run_id>`) or launch the web dashboard (`kestrion dashboard`).
- **`core/errors.py`** — built! Exceptions are centralized under a semantic `KestrionError` class hierarchy, including `CheckpointNotFoundError`, `InvalidRunStatusError`, `InvalidStoreURLError`, `InvalidToolApprovalError`, `RunExpiredError`, `ApprovalRequired`, and `HandoffCompleted`. This also enables `core/engine.py` to import and check exceptions like `HandoffCompleted` cleanly without circular dependencies.
- **Enterprise Secret Management** (`core/secrets.py`) — built! Resolves credentials at runtime (via `EnvVarSecretProvider` or custom providers) and intercepts them during `Engine.call_tool` before they ever hit `AgentState.scratch` or the event log, preventing leaks into the durable SQLite/Postgres stores.
- **Advanced Context Window Management** (`core/nodes.py`, `_fold`) — built! Solves the massive DB serialization bloat of long-running agents. The `SummarizationNode` watches conversation length and triggers an `EventType.CONTEXT_COMPACTED` event. The `Engine._fold` hook cleanly replaces the massive `history` list with a bounded `{"type": "summary", "content": "..."}` block, ensuring O(1) storage scale.
- **Time-Travel Debugging (Run Forking)** (`Engine.fork`) — built! Uses the immutable event log to seamlessly branch off past checkpoints (`kestrion fork <run_id> --at-seq <seq>`), allowing developers to debug failures and replay runs without starting over.
- **`agent/graph.py`** — empty stub. Multi-step workflows are possible today via raw `Node`
  classes directly; this would add a more ergonomic builder API on top.

See the project's `README.md` "Known gaps" section and `roadmap.md` for what's planned versus
implemented at any given time.