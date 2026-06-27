# Checkpointing

[Event Sourcing](event-sourcing.md) explains why state is derived from an event log rather than
mutated directly. Checkpointing is the mechanism that makes replaying that log *fast* — and the
reason `resume()` can hand a run back to a completely different process than the one that started
it.

## The problem checkpoints solve

If recovery meant replaying every event from the very first one, a long-running agent would get
slower to resume the longer it ran. A `Checkpoint` is a snapshot: `AgentState` as of some specific
point in the event log, tagged with that position (`event_seq`).

```python
@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    run_id: str
    state: AgentState
    created_at: datetime
    event_seq: int
```

Resuming a run means: load the *latest* checkpoint, replay only the events that happened *after*
it, and continue. Not replaying from zero.

## When checkpoints get written

The engine checkpoints after every single state transition — every tool call, every node
advancing to the next one, every pause. Not on a timer, not sampled. The tradeoff is explicit: more
write volume, in exchange for the simplest possible recovery story. There is never a meaningful
window where something happened but only exists in memory.

## What `resume()` actually does, step by step

```python
async def resume(self, run_id, on_expired="status"):
    checkpoint = await self.store.latest(run_id)
    state = checkpoint.state

    newer_events = await self.store.events_since(run_id, checkpoint.event_seq)
    for evt in newer_events:
        self._fold(state, evt)
    ...
    return await self._drive(state)
```

1. **Load the latest checkpoint.** This is a full `AgentState` snapshot.
2. **Replay anything newer than it.** Handles the edge case where an event was appended after the
   last checkpoint was saved, but the process died before a *new* checkpoint captured that event's
   effect. Without this step, that event's effect on state would be lost.
3. **Re-enter the drive loop.** From here, execution proceeds exactly as if the process never
   stopped.

## Why this works from a different process

`Engine` holds no run-specific state on `self`. `nodes`, `tools`, and `store` are configured once
when you construct it; nothing about a specific run lives on the `Engine` object itself. That's
what makes this true, and it's not just a claim — it's directly tested:

```python
# Process A starts a run, it pauses on a gated tool.
engine_a = Engine(nodes=..., tools=..., store=store, entry_node=...)
state = await engine_a.start()

# Process B: a BRAND NEW Engine instance, sharing nothing with engine_a
# except the store, resumes the same run.
engine_b = Engine(nodes=..., tools=..., store=store, entry_node=...)
final_state = await engine_b.resume(state.run_id)
```

`tests/unit/test_engine.py::test_resume_from_independent_engine_after_approval` does exactly this,
and it passes. That's the actual proof behind "any process can resume any run" — not an assumption.

## The storage backend is swappable, by design

`CheckpointStore` is a `Protocol`, not a base class:

```python
class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None: ...
    async def latest(self, run_id: str) -> Checkpoint | None: ...
    async def append_event(self, event: Event) -> int: ...
    async def events_since(self, run_id: str, seq: int) -> list[Event]: ...
```

`SQLiteCheckpointStore` is the only implementation that exists today. A Postgres-backed store
(planned, not yet built) would need to satisfy exactly this interface — nothing in `Engine` would
need to change.

## A real bug this design caught early

The original checkpoint store used `pickle.dumps(checkpoint.state)` to serialize `AgentState`.
That was replaced with explicit `AgentState.to_dict()` / `from_dict()` methods, because `pickle`
ties the on-disk format to the exact class definition and Python version at write time — a
liability the moment the engine evolves and someone needs to read an old checkpoint. The current
store also fails loudly (`ValueError`) if something non-JSON-serializable ends up in `scratch`,
rather than letting `pickle` silently paper over a problem that JSON can't represent. See the
[architecture document](../architecture.md#7-design-decisions-and-the-bugs-that-shaped-them) for
the full account of this and other bugs found during development.

## Related

- [Event Sourcing](event-sourcing.md) — the model checkpoints are snapshots *of*
- [Approval Gates](approval-gates.md) — pausing a run is, mechanically, just a checkpoint plus a
  status flag