# Event Sourcing

The single idea underneath everything Kestrion does: **`AgentState` is never mutated directly —
it's derived by folding an immutable log of events.**

## What this means concretely

Every meaningful thing that happens during a run — an LLM call finishing, a tool being called, a
transition from one step to the next — is recorded as an `Event` before anything else happens.
`Event` objects are frozen dataclasses; nothing in the codebase ever changes one after it's
created.

```python
@dataclass(frozen=True)
class Event:
    event_id: str
    run_id: str
    type: EventType
    timestamp: datetime
    payload: dict
    tokens_in: int
    tokens_out: int
    cost_usd: float
```

`AgentState` — the thing your code actually reads (`result.state.scratch`,
`result.state.total_cost_usd`, and so on) — is rebuilt by replaying these events through one
function, `Engine._fold`:

```
Event 1 ──┐
Event 2 ──┼──► fold ──► AgentState (current)
Event 3 ──┤
Event N ──┘
```

There is exactly one place in the entire codebase where an event changes `AgentState`. That's
deliberate — it's what makes the rest of this document true.

## Why this is the foundation for everything else

**Crash recovery is free, not engineered per-feature.** If state were just a mutable object held
in memory, recovering from a crash would mean serializing whatever that object looked like at some
arbitrary point — and you'd need to get that serialization right for every new field you ever add.
Because state is *derived*, recovery just means: load the last checkpoint, replay whatever events
happened after it, you're back to the exact same state. No special-casing.

**Observability is the same data, not a separate system.** Token counts and cost live directly on
the `Event` that produced them. There's no separate logging pipeline to keep in sync with the
engine — anything that creates an event for any reason gets cost/token accounting for free if it
populates those fields.

**Approval gating composes cleanly with everything else** (see
[Approval Gates](approval-gates.md)) because pausing a run is just: stop folding new events, record
one `HUMAN_INTERVENTION` event, return. Resuming is: load the last checkpoint, check whether the
gate condition is now satisfied, keep folding.

## The fold function, in full

```python
def _fold(self, state: AgentState, evt: Event) -> None:
    if evt.type == EventType.TOOL_CALL_COMPLETED:
        state.history.append({"type": "tool_result", **evt.payload})
    elif evt.type == EventType.LLM_CALL_COMPLETED:
        state.history.append({"type": "llm_response", **evt.payload})
        state.total_tokens += evt.tokens_in + evt.tokens_out
        state.total_cost_usd += evt.cost_usd
```

Notice what's *not* here: `RUN_COMPLETED`, `RUN_FAILED`, `RUN_EXPIRED`, `HUMAN_INTERVENTION`,
`CHECKPOINT_SAVED` don't change any `AgentState` field via fold. Status transitions
(`RunStatus.COMPLETED`, `WAITING_ON_HUMAN`, etc.) are set directly by the engine's drive loop, not
derived from folding — they're control-flow state, not accumulated data. If you're ever unsure
whether some piece of state is "derived" or "directly set," this function is the place to check:
if an event type isn't handled here, that event is informational/audit-trail only, not something
that feeds back into what the engine does next.

## A consequence worth knowing: the event log is the real history

`AgentState.history` is described in the code as "summarized... NOT the full event log" — it's a
convenience view. If you want the complete picture of everything that happened during a run,
including events that don't show up in `history` at all (checkpoints saved, human interventions,
exact timestamps), go to the store directly:

```python
events = await store.events_since(run_id, 0)
```

This is also how a future `Trace` viewer (planned, not yet built — see the project roadmap) will
work: it's just a different presentation of data that already exists, not a new tracking system.

## Related

- [Checkpointing](checkpointing.md) — how the event log and periodic snapshots work together for
  recovery
- [Approval Gates](approval-gates.md) — how pausing/resuming is built entirely on top of this model
- The full [architecture document](../architecture.md) — module-by-module detail beyond this one
  concept